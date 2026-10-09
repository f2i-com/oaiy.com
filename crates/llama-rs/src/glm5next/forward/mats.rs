//! A matrix of the model, on the host or on a device, and the calls through it: one row of `x`, a batch of
//! rows, a pair of matrices one after the other, and the experts' feed-forward.

use super::*;

/// A matrix in this model, either resident f32 on the host or a `Weight` on a
/// device.
///
/// This is the seam that lets **one** forward implementation run on the host
/// reference and on CUDA. Only the matrices go through it; vectors (norms,
/// biases, `ssm_a`, `dt_bias`, the indexer APE) stay plain `&[f32]` because the
/// stages index them directly rather than multiplying by them.
///
/// The device arm copies the activation in and the result out per call. For the
/// matrices that matter that is the right trade: the transfer is `n_embd` floats
/// against a matmul of tens of millions of MACs. Keeping activations resident
/// between stages would be faster and is the obvious next optimisation; it is
/// not needed for correctness.
#[derive(Clone, Copy)]
pub enum Mat<'a> {
    Host(&'a [f32]),
    Device {
        w: &'a Weight,
        backend: &'a dyn Backend,
    },
}

/// A per-head stack of matrices, `[b, m, k]`, applied as `b` independent
/// matvecs: the [`Mat`] seam for a weight that is not one flat matrix.
///
/// Absorbed MLA has two of these per layer and they are the largest weights in
/// the trunk: `k_b` is `[n_head, kv_lora, qk_head]`, `v_b` is
/// `[n_head, v_head, kv_lora]`, 8.39 M values each at the released shapes.
/// Profiling found them costing 61.7 ms of a 159.8 ms token, 39% of the trunk,
/// purely because they were host f32 and every token read 739 MB of them
/// through scalar code. On the device the same read is HBM.
pub enum Bat<'a> {
    Host(&'a [f32]),
    Device {
        /// A dense f32 `[b, m, k]` tensor on `backend`.
        t: &'a Tensor,
        backend: &'a dyn Backend,
    },
}

impl Bat<'_> {
    /// `out[i] = W[i] @ x[i]` for `i` in `0..b`.
    pub fn apply(
        &self,
        x: &[f32],
        b: usize,
        m: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != b * k || out.len() != b * m {
            return Err(LlamaError::Config(format!(
                "forward: batched gemv got x {} / out {}, expected {} / {} for [{b}, {m}, {k}]",
                x.len(),
                out.len(),
                b * k,
                b * m
            )));
        }
        match self {
            Bat::Host(w) => batched_gemv_host(w, x, b, m, k, out),
            Bat::Device { t, backend } => {
                if t.numel() != b * m * k {
                    return Err(LlamaError::Config(format!(
                        "forward: batched gemv weight has {} values, expected {} for [{b}, {m}, {k}]",
                        t.numel(),
                        b * m * k
                    )));
                }
                let xd = backend.to_device(Tensor::from_vec(x.to_vec(), vec![b, k]));
                let yd = backend.batched_gemv(t, &xd, b, m, k);
                let yh = backend.to_host(yd);
                let got = yh.data();
                if got.len() != out.len() {
                    return Err(LlamaError::Config(format!(
                        "forward: batched gemv returned {} values, expected {}",
                        got.len(),
                        out.len()
                    )));
                }
                out.copy_from_slice(got);
                Ok(())
            }
        }
    }
}

/// `out[i] = W[i] @ x[i]` on the host, each row summed in index order.
///
/// This is the oracle the CUDA kernel is checked against, and the arithmetic
/// [`mla::absorb_query`](super::super::mla::absorb_query) has always done.
pub(crate) fn batched_gemv_host(
    w: &[f32],
    x: &[f32],
    b: usize,
    m: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    if w.len() != b * m * k {
        return Err(LlamaError::Config(format!(
            "forward: batched gemv weight is {} wide, expected {} for [{b}, {m}, {k}]",
            w.len(),
            b * m * k
        )));
    }
    for i in 0..b {
        let wi = &w[i * m * k..(i + 1) * m * k];
        let xi = &x[i * k..(i + 1) * k];
        let oi = &mut out[i * m..(i + 1) * m];
        for (r, o) in oi.iter_mut().enumerate() {
            let row = &wi[r * k..(r + 1) * k];
            *o = row.iter().zip(xi).map(|(a, c)| a * c).sum();
        }
    }
    Ok(())
}

/// Two projections that share an input, as one matrix where the loader could
/// stack them and as two where it could not.
///
/// glm5next has three of these per layer whose halves have the same dtype and the
/// same input, so the GGUF rows can be concatenated and the pair done in one
/// matmul: `attn_q`+`attn_k` (Q4_K, `attn_v` is Q6_K so it stays out),
/// `ssm_f_a`+`ssm_g_a` (Q8_0), and `ffn_gate_shexp`+`ffn_up_shexp` (Q4_K). That is
/// 110 of the ~710 `Mat::apply` calls a token, each carrying a ~32 us
/// synchronisation whatever its size.
///
/// The arithmetic is unchanged either way: every output row is an independent dot
/// product over the same `x`, so a stacked matvec is exactly the two separate ones
/// concatenated, bit for bit.
///
/// `Split` exists because [`bridge::HostModel`](super::super::bridge) holds its weights
/// as f32 and stacking them there would copy about 9 GB. The host oracle keeps two
/// matrices; the device path fuses.
pub enum Pair<'a> {
    Fused(Mat<'a>),
    Split(Mat<'a>, Mat<'a>),
}

impl Pair<'_> {
    /// The backend this pair lives on, or `None` for host weights.
    ///
    /// `Split` answers `None` even on a device: it exists because the host oracle
    /// holds its halves separately, and concatenating two device results to make
    /// the fused layout back would cost more than the hop it saves. Every
    /// device-side pair in this model is `Fused` -- see the type's own note.
    pub fn device(&self) -> Option<&dyn Backend> {
        match self {
            Self::Fused(m) => m.device(),
            Self::Split(..) => None,
        }
    }

    /// The two results as one device tensor, `[.., 2 * split]`, left on the card.
    pub fn linear_dev(&self, xd: &Tensor) -> Option<Tensor> {
        match self {
            Self::Fused(m) => m.linear_dev(xd),
            Self::Split(..) => None,
        }
    }

    /// `out` is the two results end to end, the first `split` values then the rest.
    pub fn apply(&self, x: &[f32], split: usize, out: &mut [f32]) -> Result<()> {
        if split > out.len() {
            return Err(LlamaError::Config(format!(
                "forward: pair split {split} past the {}-value output",
                out.len()
            )));
        }
        match self {
            Self::Fused(m) => m.apply(x, out),
            Self::Split(a, b) => {
                let (lo, hi) = out.split_at_mut(split);
                a.apply(x, lo)?;
                b.apply(x, hi)
            }
        }
    }
}

impl Mat<'_> {
    // VENDORED-LOCAL: PERF. The device-resident seam.
    //
    // `apply` below is upload, launch, download-and-synchronise. Measured, a token
    // makes 1319 of those round trips at 29.2 us each -- 38.5 ms of a 114 ms token
    // -- while the cards sit at 2-18% busy with their memory controllers at 0-6%.
    // The maths they exist for is 2.3 ms of that token.
    //
    // These two let a caller that has its input on the card already keep the result
    // there, so a chain of projections is a chain of launches rather than a
    // stop-start per link. `None` means this weight is host f32 (the
    // `bridge::HostModel` oracle), and the caller keeps the host path -- which is
    // also what makes that oracle still an oracle.

    /// The backend this weight lives on, or `None` for a host weight.
    pub fn device(&self) -> Option<&dyn Backend> {
        match self {
            Mat::Host(_) => None,
            Mat::Device { backend, .. } => Some(*backend),
        }
    }

    /// `W @ x` with `x` already on this weight's device, result left there.
    pub fn linear_dev(&self, xd: &Tensor) -> Option<Tensor> {
        match self {
            Mat::Host(_) => None,
            Mat::Device { w, backend } => Some(w.linear(*backend, xd)),
        }
    }

    /// `out = W @ x`, with `W` of `[out.len(), x.len()]`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            Mat::Host(w) => matvec(w, x, out),
            Mat::Device { w, backend } => {
                let xd = backend.to_device(Tensor::from_vec(x.to_vec(), vec![1, x.len()]));
                let yd = w.linear(*backend, &xd);
                let yh = backend.to_host(yd);
                let got = yh.data();
                if got.len() != out.len() {
                    return Err(LlamaError::Config(format!(
                        "forward: device linear returned {} values, expected {}",
                        got.len(),
                        out.len()
                    )));
                }
                out.copy_from_slice(got);
                Ok(())
            }
        }
    }
}

/// Runs one routed expert's whole FFN: `down(clamped_swiglu(gate·x, up·x))`.
///
/// **A dispatch-level seam, not a weight-level one.** Yielding an expert's
/// weights would force them to f32 on the host — 43 layers x 288 experts x 3 x
/// 2048 x 4096 x 4 bytes is about **1.2 TB**. Handing the whole expert FFN to the
/// implementation lets a device version keep quantised weights where they are and
/// a host version dequantise just the one expert it needs.
pub trait ExpertFfn {
    /// `ord` is the **MoE-layer ordinal** — block `n_dense_lead + ord`.
    /// Accumulates nothing: `out` is overwritten with this expert's output.
    fn apply(
        &self,
        ord: usize,
        e: usize,
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()>;

    /// The whole routed half of one MoE layer: every selected expert, scaled by
    /// its routing weight and summed into `out`.
    ///
    /// **The dispatch granularity matters more than it looks.** DeepSeek's decode
    /// step is per *layer*, not per expert (`dsv41-cuda/src/model.rs`), and an
    /// implementation that sees the whole route at once can do four things the
    /// per-expert call cannot: look every expert up in the VRAM cache together
    /// and pin the slots for the batch, launch one grouped kernel over the
    /// resident ones, hand the misses to the CPU while that runs, and pay one
    /// host round trip instead of `n_expert_used` of them. At 42 MoE layers x 8
    /// experts that last one alone is 336 round trips a token against 42, and a
    /// round trip measured 32 us.
    ///
    /// `experts` is `(expert id, routing weight)`. The default does exactly what
    /// the per-expert path always did, so an implementation need not override it.
    fn apply_layer(
        &self,
        ord: usize,
        experts: &[(u32, f32)],
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()> {
        out.fill(0.0);
        let mut e_out = vec![0.0f32; out.len()];
        for &(e, wt) in experts {
            self.apply(ord, e as usize, x, limit, &mut e_out)?;
            for (o, &ev) in out.iter_mut().zip(e_out.iter()) {
                *o += wt * ev;
            }
        }
        Ok(())
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. The routed half for SEVERAL tokens at once.
    /// One MoE layer's routed experts for a whole chunk of tokens.
    ///
    /// `routes` is one `(expert, weight)` list per token, `xs` and `outs` are those
    /// tokens' vectors end to end, `n_embd` apart.
    ///
    /// This exists because prefill costs the same per token as decode and should
    /// not. Per token the routed experts are 49 ms of a 99 ms token, and almost
    /// none of it is arithmetic: ~19 ms copying promoted records into pinned
    /// memory, ~20 ms of PCIe the compute stream waits on, ~12 ms of CPU tier. All
    /// three are **per distinct expert**, not per token, so a chunk of T tokens
    /// pays them once for the union of its routes rather than T times for each
    /// route. Eight experts a token over 64 tokens is 512 resolutions; the union
    /// is nearer 200, and every one of those reads its weights once and applies
    /// them to all the tokens that chose it.
    ///
    /// The default is the per-token loop, so an implementation need not override
    /// it and the host oracle does not have to.
    fn apply_batch(
        &self,
        ord: usize,
        routes: &[Vec<(u32, f32)>],
        xs: &[f32],
        n_embd: usize,
        limit: f32,
        outs: &mut [f32],
    ) -> Result<()> {
        if routes.len() * n_embd != xs.len() || xs.len() != outs.len() {
            return Err(LlamaError::Config(format!(
                "forward: expert batch has {} routes, {} inputs and {} outputs at n_embd {n_embd}",
                routes.len(),
                xs.len(),
                outs.len()
            )));
        }
        for (t, route) in routes.iter().enumerate() {
            let lo = t * n_embd;
            let (x, out) = (&xs[lo..lo + n_embd], &mut outs[lo..lo + n_embd]);
            self.apply_layer(ord, route, x, limit, out)?;
        }
        Ok(())
    }
}
