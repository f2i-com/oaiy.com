// VENDORED-LOCAL: whole module. GLM-5.3-Flash host reference forward pass.
//! The layer loop: a host-f32 reference that sequences the stages in
//! [`super::hc`], [`super::kda`], [`super::mla`], [`super::indexer`],
//! [`super::kpool`] and [`super::routing`] into a whole forward pass.
//!
//! Reference: llama.cpp `llama_model_glm5next::graph::graph` (the trunk loop),
//! `build_kda_layer`, `build_dsa_layer` and `build_layer_ffn` (PR #27754, pinned
//! at `86ebfef`), plus `llm_graph_context::build_ffn`'s clamped-SwiGLU branch.
//!
//! This is the same methodology `dsv41` used: a CPU reference first, then the
//! device path validated against it. It is **not** fast — one token at a time,
//! dense f32 matvecs — and it is not meant to be. It is the thing a greedy-token
//! comparison against llama.cpp can be run on, and the thing a future
//! `Tensor`/`Backend` implementation gets checked against.
//!
//! ## Per-token, deliberately
//!
//! Every stage module is per-token, the KDA recurrence is definitionally
//! sequential, and the reference's own autoregressive path is what its chunked
//! prefill is validated against. Looping this over a prompt gives prefill for
//! free — slowly, but with the same arithmetic. Batched prefill is an
//! optimisation for later.
//!
//! ## The trunk loop
//!
//! ```text
//! stream = hc::init(embd[token])                  // HC exact copies
//! for il in 0..n_layer:
//!     residual = stream
//!     mix  = hc::mixes(stream, hc_attn)
//!     cur  = rms_norm(hc::collapse(stream, mix.pre), attn_norm)
//!     cur  = kda_layer(cur) | mla_layer(cur)
//!     stream = hc::combine(cur, residual, mix)
//!
//!     residual = stream
//!     mix  = hc::mixes(stream, hc_ffn)
//!     cur  = rms_norm(hc::collapse(stream, mix.pre), ffn_norm)
//!     cur  = dense_ffn(cur) | moe_ffn(cur)
//!     stream = hc::combine(cur, residual, mix)
//!
//! logits = output @ rms_norm(hc::mean(stream), output_norm)
//! ```
//!
//! Note the mixes are derived from the **un-normed** stream — `build_hc_pre`
//! does its own RMSNorm over the flattened `hc * n_embd` vector internally, and
//! `attn_norm` / `ffn_norm` apply only to the collapsed result.
//!
//! The NextN/MTP block (`blk.45`) is not run: it has no `hc_*` mixer and is a
//! draft head, not part of a plain decode.
//!
//! ## Weight layout
//!
//! Every weight is `&[f32]` in the **loader's reversed layout**, so `shape()[0]`
//! is the output dimension — the same convention `Glm5NextModel`'s `want_shape`
//! checks. A 2-D weight `[out, in]` is row-major, so row `o` is
//! `w[o*in..(o+1)*in]` and a matvec is a dot product per row. Dequantisation is
//! the caller's problem; keeping it out makes this module pure and testable on
//! synthetic weights.
//!
//! **Runs on real weights** through [`super::bridge::HostModel`], which loads
//! the non-expert tensors resident as f32 and serves routed experts from the
//! `.gguf` per dispatch. `Glm5NextModel::forward` itself is still the stub: that
//! path wants the device implementation, not this one.

use std::sync::Arc;
use ggml_rs::{Backend, Tensor};

use super::{hc, indexer, kda, kpool, mla, routing};
use crate::loader::Weight;
use crate::{LlamaError, Result};

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
/// [`mla::absorb_query`](super::mla::absorb_query) has always done.
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
/// `Split` exists because [`bridge::HostModel`](super::bridge) holds its weights
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

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `out = w @ x` for `w` of `[out_dim, in_dim]` row-major.
pub(crate) fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) -> Result<()> {
    let n = x.len();
    if w.len() != out.len() * n {
        return Err(LlamaError::Config(format!(
            "forward: matvec weight is {} for [{}, {n}]",
            w.len(),
            out.len()
        )));
    }
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    Ok(())
}

/// RMSNorm with a learned scale, in place.
fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) -> Result<()> {
    if w.len() != x.len() {
        return Err(LlamaError::Config(format!(
            "forward: rms_norm weight is {} for a {}-wide vector",
            w.len(),
            x.len()
        )));
    }
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (v, &s) in x.iter_mut().zip(w) {
        *v = *v * inv * s;
    }
    Ok(())
}

/// LayerNorm with scale **and bias** — the indexer's `k_norm`, and the only
/// non-RMS norm in this architecture.
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) -> Result<()> {
    if w.len() != x.len() || b.len() != x.len() {
        return Err(LlamaError::Config(format!(
            "forward: layer_norm weight/bias are {} / {} for a {}-wide vector",
            w.len(),
            b.len(),
            x.len()
        )));
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((v, &s), &bb) in x.iter_mut().zip(w).zip(b) {
        *v = (*v - mean) * inv * s + bb;
    }
    Ok(())
}

/// L2-normalise in place: `x / sqrt(sum(x^2) + eps)`. The KDA path applies this
/// per head at the reference's own hardcoded `1e-6`, not the model's norm eps.
fn l2_norm(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The clamped SwiGLU, per `build_ffn`'s generic branch:
/// `clamp(silu(gate), -inf, L) * clamp(up, -L, L)`.
///
/// `limit <= 1e-6` disables the clamp, matching the reference's `limit > eps`
/// guard, and gives a plain SwiGLU.
///
/// Note the clamp is on the **activated** gate, after the SiLU, and is
/// one-sided; `up` is clamped symmetrically.
pub fn swiglu_clamped(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || out.len() != gate.len() {
        return Err(LlamaError::Config(format!(
            "forward: swiglu widths {} / {} / {}",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    let clamped = limit > 1e-6;
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        let mut a = silu(g);
        let mut b = u;
        if clamped {
            if a > limit {
                a = limit;
            }
            b = b.clamp(-limit, limit);
        }
        *o = a * b;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

/// One sublayer's hyper-connection mixer.
pub struct HcW<'a> {
    /// `[hc_mix, hc * n_embd]`
    pub fn_: Mat<'a>,
    /// `[hc_mix]`
    pub base: &'a [f32],
    /// `[3]`
    pub scale: &'a [f32],
}

pub struct KdaW<'a> {
    /// `[d_inner, n_embd]` each.
    /// `attn_q` and `attn_k` -- the same dtype, so the device path fuses them.
    pub qk: Pair<'a>,
    pub v: Mat<'a>,
    /// `[d_inner, 1, d_conv]` each — depthwise, so row `c` is
    /// `conv[c*d_conv..(c+1)*d_conv]`.
    pub conv_q: &'a [f32],
    pub conv_k: &'a [f32],
    pub conv_v: &'a [f32],
    /// `[head_dim, n_embd]` then `[d_inner, head_dim]`: the low-rank decay.
    /// `ssm_f_a` and `ssm_g_a`: both Q8_0, both applied to the layer input.
    pub fga: Pair<'a>,
    pub f_b: Mat<'a>,
    pub g_b: Mat<'a>,
    /// `[n_head, n_embd]`
    pub beta: Mat<'a>,
    /// `[n_head]`, holding `-exp(A_log)`.
    pub a: &'a [f32],
    /// `[d_inner]`
    pub dt_bias: &'a [f32],
    /// `[head_dim]`
    pub o_norm: &'a [f32],
    /// `[n_embd, d_inner]`
    pub out: Mat<'a>,
}

pub struct IndexerW<'a> {
    /// `[d_idx, n_embd]`
    pub attn_k: Mat<'a>,
    /// `[n_ihead * d_idx, q_lora]`
    pub attn_q_b: Mat<'a>,
    /// `[d_idx]` each.
    pub k_norm: &'a [f32],
    pub k_norm_bias: &'a [f32],
    /// `[n_ihead, n_embd]`
    pub proj: Mat<'a>,
    /// `[d_idx, n_embd]`
    pub comp_gate: Mat<'a>,
    /// `[kpool, d_idx]`
    pub comp_ape: &'a [f32],
}

pub struct MlaW<'a> {
    /// `[q_lora, n_embd]`
    pub q_a: Mat<'a>,
    /// `[q_lora]`
    pub q_a_norm: &'a [f32],
    /// `[n_head * qk_head, q_lora]`
    pub q_b: Mat<'a>,
    /// `[kv_lora, n_embd]`
    pub kv_a_mqa: Mat<'a>,
    /// `[kv_lora]`
    pub kv_a_norm: &'a [f32],
    /// `[n_head, kv_lora, qk_head]`
    pub k_b: Bat<'a>,
    /// `[n_head, v_head, kv_lora]`
    pub v_b: Bat<'a>,
    /// `[n_embd, n_head * v_head]`
    pub out: Mat<'a>,
    pub indexer: IndexerW<'a>,
}

pub enum AttnW<'a> {
    Kda(KdaW<'a>),
    Mla(MlaW<'a>),
}

pub struct MoeW<'a> {
    /// `[n_expert, n_embd]`
    pub router: Mat<'a>,
    /// `[n_expert]`
    pub probs_b: &'a [f32],
    /// Routed experts, run per dispatch rather than borrowed.
    pub experts: &'a dyn ExpertFfn,
    /// This layer's MoE ordinal, for [`ExpertSource::expert`].
    pub ord: usize,
    /// `[n_ff_shexp, n_embd]`, `[n_ff_shexp, n_embd]`, `[n_embd, n_ff_shexp]`.
    /// The shared expert is always active, so it stays resident.
    /// `ffn_gate_shexp` and `ffn_up_shexp`, fused on the device path.
    pub sh_gate_up: Pair<'a>,
    pub sh_down: Mat<'a>,
}

pub enum FfnW<'a> {
    Dense {
        gate: Mat<'a>,
        up: Mat<'a>,
        down: Mat<'a>,
    },
    Moe(MoeW<'a>),
}

pub struct LayerW<'a> {
    /// `[n_embd]` each.
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub hc_attn: HcW<'a>,
    pub hc_ffn: HcW<'a>,
    pub attn: AttnW<'a>,
    pub ffn: FfnW<'a>,
}

pub struct ModelW<'a> {
    /// `[n_vocab, n_embd]`
    pub tok_embd: &'a [f32],
    /// `[n_embd]`
    pub output_norm: &'a [f32],
    /// `[n_vocab, n_embd]`
    pub output: Mat<'a>,
    /// Trunk layers only — `n_layer` of them, MTP excluded.
    pub layers: Vec<LayerW<'a>>,
}

/// Geometry the loop needs, in host terms. A subset of [`super::Glm5NextConfig`]
/// plus the dims it derives, so this module can be exercised at synthetic sizes.
#[derive(Debug, Clone)]
pub struct Shape {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub kda_head_dim: usize,
    pub d_conv: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_head: usize,
    pub v_head: usize,
    pub d_idx: usize,
    pub n_ihead: usize,
    pub kpool: usize,
    pub indexer_top_k: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_ff_dense: usize,
    pub n_dense_lead: usize,
    /// `n_layer_all` entries; only the first `n_layer` are run.
    pub layer_kinds: Vec<super::LayerKind>,
    pub hc_count: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub rms_eps: f32,
    pub norm_eps: f32,
    pub kda_gate_lower_bound: f32,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    pub max_len: usize,
}

impl Shape {
    pub fn d_inner(&self) -> usize {
        self.n_head * self.kda_head_dim
    }
    /// Ordinal of a block among the MLA layers, for the per-layer caches.
    fn mla_ordinal(&self, il: usize) -> usize {
        self.layer_kinds[..il]
            .iter()
            .filter(|k| **k == super::LayerKind::Mla)
            .count()
    }
    /// Ordinal of a block among the KDA layers.
    fn kda_ordinal(&self, il: usize) -> usize {
        self.layer_kinds[..il]
            .iter()
            .filter(|k| **k == super::LayerKind::Kda)
            .count()
    }
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

/// Everything that persists between tokens.
pub struct State {
    pub len: usize,
    /// Per KDA layer, `[n_head * head_dim * head_dim]`. Used when the state is
    /// on the host; empty when [`Self::kda_dev`] holds it instead.
    kda: Vec<Vec<f32>>,
    /// VENDORED-LOCAL: GLM-5.3-Flash. The same state, resident on a backend.
    ///
    /// 34 KDA layers x 4.2 MB is 143 MB that the recurrence touches twice per
    /// token. Keeping it here is what makes a kernel worth having: a version that
    /// uploaded and downloaded it each layer would move 285 MB a token, which at
    /// the ~11 GB/s this machine's x4/x8 links manage is slower than the scalar
    /// loop it replaces.
    kda_dev: Option<(Vec<Tensor>, Arc<dyn Backend>)>,
    /// Per KDA layer, `[(d_conv - 1) * 3 * d_inner]` — the last `d_conv - 1`
    /// **pre-conv** `q‖k‖v` vectors, which is what the reference's conv state
    /// holds so a rollback restores one block.
    conv: Vec<Vec<f32>>,
    /// Per MLA layer, `[max_len * kv_lora]` — the cached latent, serving as both
    /// K and V.
    latents: Vec<Vec<f32>>,
    /// VENDORED-LOCAL: GLM-5.3-Flash. The same cache, mirrored on a backend.
    ///
    /// `[max_len, 1, kv_lora]` per MLA layer. The attention over it is O(len) work a
    /// token -- n_head dot products kv_lora wide against every cached row, twice --
    /// which made a prompt O(n^2) and, on the host, 124 ms a token at position 1740
    /// while both cards sat idle. Doing it on the card means the cache has to live
    /// there: uploading `len * kv_lora` a token instead would be the same quadratic
    /// cost over PCIe.
    ///
    /// A mirror, not a move: `latents` stays authoritative, so snapshots and the
    /// host oracle are unchanged, and a row costs one extra device-side copy of
    /// `kv_lora` floats when it is written.
    latents_dev: Option<(Vec<Tensor>, Arc<dyn Backend>)>,
    /// Indexer key / gate / pooled, per MLA layer.
    kpool: kpool::KpoolCache,
    max_len: usize,
}

impl State {
    pub fn new(sh: &Shape) -> Result<Self> {
        let n_kda = sh.layer_kinds[..sh.n_layer]
            .iter()
            .filter(|k| **k == super::LayerKind::Kda)
            .count();
        let n_mla = sh.n_layer - n_kda;
        let hd = sh.kda_head_dim;
        Ok(Self {
            len: 0,
            kda: vec![vec![0.0f32; sh.n_head * hd * hd]; n_kda],
            kda_dev: None,
            conv: vec![vec![0.0f32; (sh.d_conv - 1) * 3 * sh.d_inner()]; n_kda],
            latents: vec![vec![0.0f32; sh.max_len * sh.kv_lora]; n_mla],
            latents_dev: None,
            kpool: kpool::KpoolCache::new(n_mla, sh.max_len, sh.d_idx)?,
            max_len: sh.max_len,
        })
    }

    /// As [`Self::new`], with the KDA recurrent state resident on `backend`.
    ///
    /// Everything else stays on the host: the conv ring is small, and the latents
    /// and indexer caches are read by host code that has not moved yet.
    pub fn new_on(sh: &Shape, backend: Arc<dyn Backend>) -> Result<Self> {
        let mut st = Self::new(sh)?;
        let hd = sh.kda_head_dim;
        let n = sh.n_head * hd * hd;
        let dev = st
            .kda
            .iter()
            .map(|_| backend.to_device(Tensor::from_vec(vec![0.0f32; n], vec![sh.n_head, hd, hd])))
            .collect();
        // The latent cache is mirrored too, `[max_len, 1, kv_lora]` a layer: one KV
        // head, which is what absorbed MLA is, and what lets `bmm_qkt`/`bmm_av`
        // serve it without broadcasting the rows to every query head.
        let lat = st
            .latents
            .iter()
            .map(|l| {
                backend.to_device(Tensor::from_vec(
                    vec![0.0f32; l.len()],
                    vec![sh.max_len, 1, sh.kv_lora],
                ))
            })
            .collect();
        st.kda = Vec::new();
        st.kda_dev = Some((dev, Arc::clone(&backend)));
        st.latents_dev = Some((lat, backend));
        Ok(st)
    }

    /// How many KDA layers this state covers, whichever side it lives on.
    pub fn n_kda(&self) -> usize {
        match &self.kda_dev {
            Some((d, _)) => d.len(),
            None => self.kda.len(),
        }
    }

    /// Largest magnitude anywhere in the KDA recurrent state. Host state only.
    ///
    /// For watching whether the recurrence is stable as a sequence grows: the state
    /// is scaled by `exp(g_log)` every token, so a positive `g_log` on any channel
    /// shows up here as geometric growth.
    pub fn kda_max_abs(&self) -> f32 {
        // The state is usually on a card, where `kda` is empty -- reading only the
        // host copy reported 0.0 for every device model, which makes the diagnostic
        // useless exactly where it is wanted.
        if let Some((dev, backend)) = &self.kda_dev {
            return dev.iter().fold(0.0f32, |a, t| {
                backend
                    .to_host(t.clone())
                    .data()
                    .iter()
                    .fold(a, |b, v| b.max(v.abs()))
            });
        }
        self.kda
            .iter()
            .flat_map(|s| s.iter())
            .fold(0.0f32, |a, v| a.max(v.abs()))
    }

    /// Largest magnitude in the cached MLA latents.
    pub fn latent_max_abs(&self) -> f32 {
        self.latents
            .iter()
            .flat_map(|s| s.iter())
            .fold(0.0f32, |a, v| a.max(v.abs()))
    }

    /// Whether the recurrent state is resident on a backend.
    pub fn kda_on_device(&self) -> bool {
        self.kda_dev.is_some()
    }

    pub fn reset(&mut self) {
        self.len = 0;
        if let Some((dev, backend)) = &mut self.kda_dev {
            for t in dev.iter_mut() {
                let shape = t.shape().to_vec();
                let n = t.numel();
                *t = backend.to_device(Tensor::from_vec(vec![0.0f32; n], shape));
            }
        }
        for s in self.kda.iter_mut() {
            s.fill(0.0);
        }
        for c in self.conv.iter_mut() {
            c.fill(0.0);
        }
        for l in self.latents.iter_mut() {
            l.fill(0.0);
        }
        // The mirror as well: a stale row past `len` would be a previous
        // conversation's, and the mask only hides rows it knows about.
        if let Some((dev, backend)) = &mut self.latents_dev {
            for t in dev.iter_mut() {
                let shape = t.shape().to_vec();
                let n = t.numel();
                *t = backend.to_device(Tensor::from_vec(vec![0.0f32; n], shape));
            }
        }
        self.kpool.reset();
    }

    pub fn kpool_cache(&self) -> &kpool::KpoolCache {
        &self.kpool
    }
}

// ---------------------------------------------------------------------------
// stages
// ---------------------------------------------------------------------------

/// The hyper-connection coefficients for one token.
///
/// The `[hc_mix, hc * n_embd]` projection goes through [`Mat`] like any other
/// matrix; the RMS reciprocal and the Sinkhorn tail are 24 numbers and a 4x4, so
/// they stay scalar wherever the matmul ran.
fn hc_mixes(stream: &[f32], w: &HcW<'_>, sh: &Shape) -> Result<hc::Mix> {
    let mut proj = [0.0f32; hc::MIX];
    w.fn_.apply(stream, &mut proj)?;
    let sumsq: f32 = stream.iter().map(|v| v * v).sum();
    let r = 1.0 / (sumsq / stream.len() as f32 + sh.rms_eps).sqrt();
    Ok(hc::mixes_from_projection(
        &proj,
        r,
        w.base,
        w.scale,
        sh.hc_sinkhorn_iters,
        sh.hc_eps,
    ))
}

/// One KDA linear-attention layer. `x` is the post-`attn_norm` input; every
/// projection here reads it, including `f`, `g` and `beta` — the reference is
/// explicit that those read the layer input and **not** the convolved `q‖k‖v`.
/// One KDA layer's recurrent state, on whichever side it lives.
///
/// The [`Mat`] seam for the recurrence: `Host` runs [`kda::step`], the oracle,
/// and `Device` runs [`Backend::kda_delta_step`] against a state that never
/// leaves the card.
pub enum KdaSt<'a> {
    Host(&'a mut [f32]),
    Device {
        t: &'a mut Tensor,
        backend: &'a dyn Backend,
    },
}

fn kda_layer(
    sh: &Shape,
    w: &KdaW<'_>,
    state: KdaSt<'_>,
    conv_state: &mut [f32],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (nh, hd, d_conv) = (sh.n_head, sh.kda_head_dim, sh.d_conv);
    let di = sh.d_inner();
    let cd = 3 * di;

    // q‖k‖v from the layer input.
    let t = std::time::Instant::now();
    let mut qkv = vec![0.0f32; cd];
    w.qk.apply(x, di, &mut qkv[0..2 * di])?;
    w.v.apply(x, &mut qkv[2 * di..cd])?;
    prof::add(&prof::KDA_PROJ, t);

    // One depthwise conv over the whole concatenation, SiLU on its output (not
    // on the projections). The three weights concatenate in q, k, v order.
    let t = std::time::Instant::now();
    let mut conv_out = vec![0.0f32; cd];
    for c in 0..cd {
        let (third, ch) = (c / di, c % di);
        let cw = match third {
            0 => w.conv_q,
            1 => w.conv_k,
            _ => w.conv_v,
        };
        let cw = &cw[ch * d_conv..(ch + 1) * d_conv];
        let mut acc = 0.0f32;
        for k in 0..d_conv - 1 {
            acc += conv_state[k * cd + c] * cw[k];
        }
        acc += qkv[c] * cw[d_conv - 1];
        conv_out[c] = silu(acc);
    }
    // Shift the pre-conv inputs through the state.
    for c in 0..cd {
        for k in 0..d_conv.saturating_sub(2) {
            conv_state[k * cd + c] = conv_state[(k + 1) * cd + c];
        }
        if d_conv >= 2 {
            conv_state[(d_conv - 2) * cd + c] = qkv[c];
        }
    }

    prof::add(&prof::KDA_CONV, t);

    let mut q: Vec<f32> = conv_out[0..di].to_vec();
    let mut k: Vec<f32> = conv_out[di..2 * di].to_vec();
    let v: Vec<f32> = conv_out[2 * di..cd].to_vec();

    // Per-head L2 norm at the reference's own constant.
    for h in 0..nh {
        l2_norm(&mut q[h * hd..(h + 1) * hd], 1e-6);
        l2_norm(&mut k[h * hd..(h + 1) * hd], 1e-6);
    }

    // g = lower_bound * sigmoid(-(ssm_a * (f_b(f_a(x)) + dt_bias))), per channel.
    // `ssm_a` holds -exp(A_log), so the negation inside the sigmoid recovers the
    // reference's `sigmoid(exp(A_log) * ...)`.
    let t = std::time::Instant::now();
    // f_a and g_a share x, so one matmul gives both roots; g_a's half is used by
    // the output gate further down.
    let mut roots = vec![0.0f32; 2 * hd];
    w.fga.apply(x, hd, &mut roots)?;
    let mut g_log = vec![0.0f32; di];
    w.f_b.apply(&roots[..hd], &mut g_log)?;
    for h in 0..nh {
        for i in 0..hd {
            let idx = h * hd + i;
            let t = (g_log[idx] + w.dt_bias[idx]) * w.a[h];
            g_log[idx] = sh.kda_gate_lower_bound * sigmoid(-t);
        }
    }

    let mut beta = vec![0.0f32; nh];
    w.beta.apply(x, &mut beta)?;
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }

    prof::add(&prof::KDA_GATE, t);

    let t = std::time::Instant::now();
    let mut scan = vec![0.0f32; di];
    match state {
        KdaSt::Host(st) => kda::step(st, &q, &k, &v, &g_log, &beta, nh, hd, &mut scan)?,
        KdaSt::Device { t: st, backend } => {
            // q, k, v and g_log go up as one [4, di] tensor rather than four, so a
            // layer costs two uploads and one download instead of six crossings.
            let mut packed = Vec::with_capacity(4 * di);
            packed.extend_from_slice(&q);
            packed.extend_from_slice(&k);
            packed.extend_from_slice(&v);
            packed.extend_from_slice(&g_log);
            let qkvg = backend.to_device(Tensor::from_vec(packed, vec![4, di]));
            let bt = backend.to_device(Tensor::from_vec(beta.clone(), vec![nh]));
            let o = backend.kda_delta_step(st, &qkvg, &bt, nh, hd);
            let oh = backend.to_host(o);
            if oh.data().len() != scan.len() {
                return Err(LlamaError::Config(format!(
                    "forward: kda step returned {} values, expected {}",
                    oh.data().len(),
                    scan.len()
                )));
            }
            scan.copy_from_slice(oh.data());
        }
    }
    prof::add(&prof::KDA_STEP, t);

    // Per-head RMSNorm by ssm_norm, gated by a PLAIN sigmoid of g_b(g_a(x)) --
    // not the SiLU a FusedRMSNormGated would default to.
    let mut gate = vec![0.0f32; di];
    w.g_b.apply(&roots[hd..], &mut gate)?;
    for h in 0..nh {
        let head = &mut scan[h * hd..(h + 1) * hd];
        rms_norm(head, w.o_norm, sh.rms_eps)?;
        for (i, s) in head.iter_mut().enumerate() {
            *s *= sigmoid(gate[h * hd + i]);
        }
    }

    w.out.apply(&scan, out)
}

/// One full-attention layer: absorbed MLA plus the sparse indexer.
#[allow(clippy::too_many_arguments)]
fn mla_layer(
    sh: &Shape,
    w: &MlaW<'_>,
    latents: &mut [f32],
    // VENDORED-LOCAL: GLM-5.3-Flash. The latent cache on a card, when there is one,
    // so the attention over it runs there: see `mla::attend_latent_device`.
    latents_dev: Option<(&mut Tensor, &dyn Backend)>,
    kp: &mut kpool::KpoolCache,
    kp_layer: usize,
    pos: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (nh, r, d_idx) = (sh.n_head, sh.kpool, sh.d_idx);
    let len = pos + 1;

    // (a) the shared query root.
    let t = std::time::Instant::now();
    let mut qr = vec![0.0f32; sh.q_lora];
    w.q_a.apply(x, &mut qr)?;
    rms_norm(&mut qr, w.q_a_norm, sh.rms_eps)?;
    prof::add(&prof::MLA_PROJ, t);
    let t_ix = std::time::Instant::now();

    // (b) store this cell's indexer key and compressor gate. Unconditional: the
    // reference stores on the dense path too, or cells written below the
    // scoring threshold would have no indexer state once a later batch crosses
    // it.
    let mut ik = vec![0.0f32; d_idx];
    w.indexer.attn_k.apply(x, &mut ik)?;
    layer_norm(
        &mut ik,
        w.indexer.k_norm,
        w.indexer.k_norm_bias,
        sh.norm_eps,
    )?;
    let mut ig = vec![0.0f32; d_idx];
    w.indexer.comp_gate.apply(x, &mut ig)?;
    kp.store(kp_layer, pos, &ik, &ig)?;

    // (c) this token may have just completed a pool.
    if kpool::slot_of(pos, r) == r - 1 {
        let pool = kpool::pool_of(pos, r);
        let mut keys = vec![0.0f32; r * d_idx];
        let mut gates = vec![0.0f32; r * d_idx];
        for (s, cell) in kpool::pool_members(pool, r).enumerate() {
            keys[s * d_idx..(s + 1) * d_idx].copy_from_slice(kp.key(kp_layer, cell)?);
            gates[s * d_idx..(s + 1) * d_idx].copy_from_slice(kp.gate(kp_layer, cell)?);
        }
        let mut pooled = vec![0.0f32; d_idx];
        indexer::pooled_key(&keys, &gates, w.indexer.comp_ape, r, d_idx, &mut pooled)?;
        kp.set_pooled(kp_layer, pool, r, &pooled)?;
    }

    // (d) score and select, or take the dense path.
    let selected: Option<Vec<u32>> =
        if kpool::indexer_scores(sh.max_len, sh.indexer_top_k, r) {
            let mut iq = vec![0.0f32; sh.n_ihead * d_idx];
            w.indexer.attn_q_b.apply(&qr, &mut iq)?;
            let mut hw = vec![0.0f32; sh.n_ihead];
            w.indexer.proj.apply(x, &mut hw)?;

            let n_pools = kpool::n_pools(len, r);
            let mut pool_keys = vec![0.0f32; n_pools * d_idx];
            let complete = kpool::n_complete_pools(len, r);
            for p in 0..complete {
                pool_keys[p * d_idx..(p + 1) * d_idx]
                    .copy_from_slice(kp.pooled(kp_layer, p, r)?);
            }
            let mut bias = vec![0.0f32; n_pools];
            kpool::pool_bias_row(pos, len, r, &mut bias)?;

            Some(indexer::candidates(
                &iq,
                &hw,
                &pool_keys,
                &bias,
                sh.n_ihead,
                d_idx,
                len,
                r,
                sh.indexer_top_k,
            )?)
        } else {
            None
        };

    prof::add(&prof::MLA_INDEX, t_ix);

    // (e) the latent, cached before attending so the token sees itself.
    let t = std::time::Instant::now();
    let mut latent = vec![0.0f32; sh.kv_lora];
    w.kv_a_mqa.apply(x, &mut latent)?;
    rms_norm(&mut latent, w.kv_a_norm, sh.rms_eps)?;
    latents[pos * sh.kv_lora..(pos + 1) * sh.kv_lora].copy_from_slice(&latent);
    // And into the mirror, if there is one: a device-side copy of `kv_lora` floats
    // into row `pos`, which is what lets the attention read the whole cache without
    // it crossing PCIe every token.
    let latents_dev = match latents_dev {
        Some((cache, be)) => {
            let row = be.to_device(Tensor::from_vec(latent.clone(), vec![1, 1, sh.kv_lora]));
            be.copy_axis0_into(cache, pos, &row);
            Some((&*cache, be))
        }
        None => None,
    };

    // (f) absorbed attention over the candidate set.
    let mut q = vec![0.0f32; nh * sh.qk_head];
    w.q_b.apply(&qr, &mut q)?;
    prof::add(&prof::MLA_PROJ, t);
    let t = std::time::Instant::now();
    let mut q_abs = vec![0.0f32; nh * sh.kv_lora];
    w.k_b
        .apply(&q, nh, sh.kv_lora, sh.qk_head, &mut q_abs)?;
    prof::add(&prof::MLA_ABSORB, t);

    let mut mask = vec![0.0f32; len];
    mla::attn_mask(pos, len, r, selected.as_deref(), &mut mask)?;

    // On the card when the cache is there, and on the host otherwise -- which is
    // the host oracle's path, and the one `bridge::HostModel` is checked against.
    //
    // This used to say that the softmax and the weighted sum "stay on the host: they
    // only touch the latents, which are at most len * kv_lora". That reasoning was
    // wrong about which part is expensive: `len * kv_lora` grows with the prompt, so
    // it is O(len) work a token and O(n^2) over a prompt, and it measured 124.25 ms
    // a token at position 1740 against the absorb's 3.76.
    let t = std::time::Instant::now();
    let mut ctx = vec![0.0f32; nh * sh.kv_lora];
    match latents_dev {
        Some((cache, be)) => mla::attend_latent_device(
            be,
            &q_abs,
            cache,
            &mask,
            mla::kq_scale(sh.qk_head),
            nh,
            sh.kv_lora,
            len,
            &mut ctx,
        )?,
        None => mla::attend_latent(
            &q_abs,
            &latents[..len * sh.kv_lora],
            &mask,
            mla::kq_scale(sh.qk_head),
            nh,
            sh.kv_lora,
            len,
            &mut ctx,
        )?,
    }
    let mut attn = vec![0.0f32; nh * sh.v_head];
    w.v_b
        .apply(&ctx, nh, sh.v_head, sh.kv_lora, &mut attn)?;
    prof::add(&prof::MLA_ATTEND, t);

    w.out.apply(&attn, out)
}

/// The FFN half of a layer: a plain clamped SwiGLU on the leading dense blocks,
/// VENDORED-LOCAL: PERF. One MoE layer with the activations kept on the card.
///
/// The host arm of [`ffn_layer`] costs eight round trips a layer: the router, the
/// routed experts, the shared expert's two halves, and a host-side SwiGLU between
/// them, each an upload, a launch and a synchronising download. This costs three,
/// and two of those are inside the expert tier's own seam.
///
/// What is left is one genuine hop: the 288 router logits. Picking the top eight
/// with a bias, a normalisation and a scale is a host decision, and the expert
/// cache is indexed on the host, so the ids have to exist here. That is the floor
/// for this layer, not an omission.
///
/// The shared expert never touches the host: gate||up, the clamped SwiGLU
/// (`swiglu_clamped_split`, which CUDA does in one kernel) and down chain on the
/// card. Its result is added unscaled, because `expert_weights_scale` applies to
/// the routed weights only.
fn ffn_moe_device(
    sh: &Shape,
    m: &MoeW<'_>,
    il: usize,
    be: &dyn Backend,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    // One upload for the layer: the router and the shared expert read the same x.
    let xd = be.to_device(Tensor::from_vec(x.to_vec(), vec![1, sh.n_embd]));

    let t_router = std::time::Instant::now();
    let logits_d = m
        .router
        .linear_dev(&xd)
        .ok_or_else(|| LlamaError::Config("forward: device router lost its device".into()))?;
    let logits_h = be.to_host(logits_d);
    if logits_h.numel() != sh.n_expert {
        return Err(LlamaError::Config(format!(
            "forward: router produced {} logits, expected {}",
            logits_h.numel(),
            sh.n_expert
        )));
    }
    let (ids, weights) = routing::route_token(
        logits_h.data(),
        Some(m.probs_b),
        sh.n_expert_used,
        sh.expert_weights_norm,
        sh.expert_weights_scale,
    );
    prof::add(&prof::FFN_ROUTER, t_router);

    // The routed experts keep the host seam: it is where the three tiers live, and
    // a miss may be computed on the CPU from a record in RAM.
    let t_routed = std::time::Instant::now();
    let route: Vec<(u32, f32)> = ids.iter().copied().zip(weights.iter().copied()).collect();
    m.experts
        .apply_layer(m.ord, &route, x, sh.swiglu_clamp_exp[il], out)?;
    prof::add(&prof::FFN_ROUTED, t_routed);

    let t_shared = std::time::Instant::now();
    let sff = sh.n_ff_shexp;
    let gu = m
        .sh_gate_up
        .linear_dev(&xd)
        .ok_or_else(|| LlamaError::Config("forward: device shared pair lost its device".into()))?;
    if gu.numel() != 2 * sff {
        return Err(LlamaError::Config(format!(
            "forward: shared gate||up produced {} values, expected {}",
            gu.numel(),
            2 * sff
        )));
    }
    let h = be.swiglu_clamped_split(&gu, sff, sh.swiglu_clamp_shexp[il], true);
    let s_out = m
        .sh_down
        .linear_dev(&h)
        .ok_or_else(|| LlamaError::Config("forward: device shared down lost its device".into()))?;
    let s_host = be.to_host(s_out);
    if s_host.numel() != sh.n_embd {
        return Err(LlamaError::Config(format!(
            "forward: shared expert produced {} values, expected {}",
            s_host.numel(),
            sh.n_embd
        )));
    }
    for (o, &sv) in out.iter_mut().zip(s_host.data()) {
        *o += sv;
    }
    prof::add(&prof::FFN_SHARED, t_shared);
    Ok(())
}

/// or routed experts plus an **unscaled** shared expert after them.
fn ffn_layer(sh: &Shape, w: &FfnW<'_>, il: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    match w {
        FfnW::Dense { gate, up, down } => {
            let ff = sh.n_ff_dense;
            let mut g = vec![0.0f32; ff];
            let mut u = vec![0.0f32; ff];
            gate.apply(x, &mut g)?;
            up.apply(x, &mut u)?;
            let mut h = vec![0.0f32; ff];
            // build_ffn reads the *shexp* clamp array for every call it makes,
            // so the dense blocks are clamped by it too.
            swiglu_clamped(&g, &u, sh.swiglu_clamp_shexp[il], &mut h)?;
            down.apply(&h, out)
        }
        FfnW::Moe(m) => {
            // VENDORED-LOCAL: PERF. With the weights on a card, this layer's chain
            // runs there: see `ffn_moe_device`. The host arm below stays because
            // `bridge::HostModel` is the oracle both are checked against.
            if let (Some(be), Some(_)) = (m.router.device(), m.sh_gate_up.device()) {
                return ffn_moe_device(sh, m, il, be, x, out);
            }
            let ne = sh.n_expert;
            let mut logits = vec![0.0f32; ne];
            let t_router = std::time::Instant::now();
            m.router.apply(x, &mut logits)?;
            let (ids, weights) = routing::route_token(
                &logits,
                Some(m.probs_b),
                sh.n_expert_used,
                sh.expert_weights_norm,
                sh.expert_weights_scale,
            );

            // One call for the layer, not one per expert: see
            // [`ExpertFfn::apply_layer`].
            let route: Vec<(u32, f32)> =
                ids.iter().copied().zip(weights.iter().copied()).collect();
            prof::add(&prof::FFN_ROUTER, t_router);
            let t_routed = std::time::Instant::now();
            m.experts
                .apply_layer(m.ord, &route, x, sh.swiglu_clamp_exp[il], out)?;
            prof::add(&prof::FFN_ROUTED, t_routed);
            let t_shared = std::time::Instant::now();

            // The shared expert is added unscaled: expert_weights_scale applies
            // to the routed weights only.
            let sff = sh.n_ff_shexp;
            let mut sgu = vec![0.0f32; 2 * sff];
            m.sh_gate_up.apply(x, sff, &mut sgu)?;
            let (sg, su) = sgu.split_at(sff);
            let mut sh_h = vec![0.0f32; sff];
            swiglu_clamped(sg, su, sh.swiglu_clamp_shexp[il], &mut sh_h)?;
            let mut s_out = vec![0.0f32; sh.n_embd];
            m.sh_down.apply(&sh_h, &mut s_out)?;
            for (o, &sv) in out.iter_mut().zip(s_out.iter()) {
                *o += sv;
            }
            prof::add(&prof::FFN_SHARED, t_shared);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// the loop
// ---------------------------------------------------------------------------

/// Run one token through the trunk and return its logits, advancing `state`.
///
/// `state.len` must be the token's position, and grows by one on success.
/// Wall time per phase of a forward pass, so a slow token can be attributed
/// rather than guessed at.
///
/// Always compiled: it is one [`std::time::Instant::now`] per phase per layer,
/// about 250 calls a token against a token measured in milliseconds. Nothing
/// synchronises the device here, so a phase that only launches kernels will
/// look cheap and the phase that next reads a result will carry the wait --
/// which is the honest picture for a host-driven forward, where every
/// [`Mat::apply`] reads its own result back.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Nanoseconds in each phase since the last [`reset`].
    pub static HC: AtomicU64 = AtomicU64::new(0);
    pub static KDA: AtomicU64 = AtomicU64::new(0);
    pub static MLA: AtomicU64 = AtomicU64::new(0);
    pub static FFN: AtomicU64 = AtomicU64::new(0);
    pub static HEAD: AtomicU64 = AtomicU64::new(0);

    /// Inside [`KDA`]: the projections, the depthwise conv and its state shift,
    /// the decay gate, and the delta-rule recurrence.
    pub static KDA_PROJ: AtomicU64 = AtomicU64::new(0);
    pub static KDA_CONV: AtomicU64 = AtomicU64::new(0);
    pub static KDA_GATE: AtomicU64 = AtomicU64::new(0);
    pub static KDA_STEP: AtomicU64 = AtomicU64::new(0);

    /// Inside [`MLA`]: the projections, the sparse indexer, the query absorb
    /// through `k_b`, and the attention through `v_b`.
    // VENDORED-LOCAL: GLM-5.3-Flash. The FFN's three parts. Two rounds of
    // optimising the routed experts moved the FFN 88.6 -> 57.5 ms, which is a lot
    // less than the arithmetic said it should, so the question of which part of an
    // FFN layer the time is in has to be measured rather than reasoned about.
    pub static FFN_ROUTER: AtomicU64 = AtomicU64::new(0);
    pub static FFN_ROUTED: AtomicU64 = AtomicU64::new(0);
    pub static FFN_SHARED: AtomicU64 = AtomicU64::new(0);
    // Inside the routed experts: resolving a route (three cache tiers, the LFRU
    // ranking, the staging) against actually computing it. The routed experts are
    // 48.8 ms of a 98.7 ms token and the parts that are accounted for -- ~20 ms of
    // PCIe stall and ~12 ms of CPU tier -- do not add up to it.
    pub static FFN_RESOLVE: AtomicU64 = AtomicU64::new(0);
    pub static FFN_DISPATCH: AtomicU64 = AtomicU64::new(0);

    pub static MLA_PROJ: AtomicU64 = AtomicU64::new(0);
    pub static MLA_INDEX: AtomicU64 = AtomicU64::new(0);
    pub static MLA_ABSORB: AtomicU64 = AtomicU64::new(0);
    pub static MLA_ATTEND: AtomicU64 = AtomicU64::new(0);

    /// The five top-level phases, in forward order. These sum to the token.
    pub fn all() -> [(&'static str, &'static AtomicU64); 5] {
        [
            ("hyper-connections", &HC),
            ("KDA attention", &KDA),
            ("MLA attention", &MLA),
            ("FFN (router, shared, routed)", &FFN),
            ("output head", &HEAD),
        ]
    }

    /// Sub-phases of [`KDA`] and [`MLA`]. These sum to less than their parents:
    /// what is left over is the projections and glue not counted here.
    pub fn inner() -> [(&'static str, &'static AtomicU64); 13] {
        [
            ("  FFN router", &FFN_ROUTER),
            ("  FFN routed experts", &FFN_ROUTED),
            ("    of which: resolve", &FFN_RESOLVE),
            ("    of which: dispatch", &FFN_DISPATCH),
            ("  FFN shared expert", &FFN_SHARED),
            ("  KDA q/k/v projections", &KDA_PROJ),
            ("  KDA depthwise conv + shift", &KDA_CONV),
            ("  KDA decay gate", &KDA_GATE),
            ("  KDA delta-rule recurrence", &KDA_STEP),
            ("  MLA projections", &MLA_PROJ),
            ("  MLA sparse indexer", &MLA_INDEX),
            ("  MLA absorb through k_b", &MLA_ABSORB),
            ("  MLA attend through v_b", &MLA_ATTEND),
        ]
    }

    pub fn reset() {
        for (_, c) in all() {
            c.store(0, Ordering::Relaxed);
        }
        for (_, c) in inner() {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// Add the time since `t` to `c`.
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn ms(c: &AtomicU64) -> f64 {
        c.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Total over every phase.
    pub fn total_ms() -> f64 {
        all().iter().map(|(_, c)| ms(c)).sum()
    }
}

pub fn forward_token(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    token: u32,
) -> Result<Vec<f32>> {
    if hc::HC != sh.hc_count {
        return Err(LlamaError::Config(format!(
            "forward: hyper_connection.count is {} but this build fixes HC at {}",
            sh.hc_count,
            hc::HC
        )));
    }
    if w.layers.len() != sh.n_layer {
        return Err(LlamaError::Config(format!(
            "forward: {} layer weights for an {}-layer trunk",
            w.layers.len(),
            sh.n_layer
        )));
    }
    let pos = state.len;
    if pos >= state.max_len {
        return Err(LlamaError::Config(format!(
            "forward: position {pos} past the cache capacity {}",
            state.max_len
        )));
    }
    let tok = token as usize;
    if tok >= sh.n_vocab {
        return Err(LlamaError::Config(format!(
            "forward: token {tok} outside the {}-entry vocabulary",
            sh.n_vocab
        )));
    }

    let n_embd = sh.n_embd;
    // The residual stream opens as HC exact copies of the embedding: no scaling
    // and no one-hot into stream 0.
    let embd = &w.tok_embd[tok * n_embd..(tok + 1) * n_embd];
    let mut stream = hc::init(embd);

    let mut sub_out = vec![0.0f32; n_embd];
    // Reused across all 45 layers and both halves: the hyper-connection path used
    // to allocate a collapsed vector, a combined stream and a clone of the stream
    // twice a layer -- 270 allocations a token, on the critical path.
    let mut cur = vec![0.0f32; n_embd];
    let mut residual = vec![0.0f32; hc::HC * n_embd];
    let mut next = vec![0.0f32; hc::HC * n_embd];

    for il in 0..sh.n_layer {
        let lw = &w.layers[il];

        // --- attention half -------------------------------------------------
        let t_hc = std::time::Instant::now();
        residual.copy_from_slice(&stream);
        let mix = hc_mixes(&stream, &lw.hc_attn, sh)?;
        hc::collapse_into(&stream, &mix.pre, &mut cur);
        rms_norm(&mut cur, lw.attn_norm, sh.rms_eps)?;
        prof::add(&prof::HC, t_hc);

        match &lw.attn {
            AttnW::Kda(kw) => {
                let t = std::time::Instant::now();
                let ord = sh.kda_ordinal(il);
                // Split the borrow: conv is host, the recurrent state may not be.
                let State { conv, kda, kda_dev, .. } = &mut *state;
                let cst = &mut conv[ord];
                let kst = match kda_dev {
                    Some((dev, backend)) => KdaSt::Device {
                        t: &mut dev[ord],
                        backend: &**backend,
                    },
                    None => KdaSt::Host(&mut kda[ord]),
                };
                kda_layer(sh, kw, kst, cst, &cur, &mut sub_out)?;
                prof::add(&prof::KDA, t);
            }
            AttnW::Mla(mw) => {
                let t = std::time::Instant::now();
                let ord = sh.mla_ordinal(il);
                // Split the borrow: latents and its mirror are per-layer, kpool is
                // shared, and the mirror's backend comes out of the same field.
                let State { latents, latents_dev, kpool, .. } = &mut *state;
                let lat = &mut latents[ord];
                let dev = latents_dev
                    .as_mut()
                    .map(|(t, be)| (&mut t[ord], &**be as &dyn Backend));
                mla_layer(sh, mw, lat, dev, kpool, ord, pos, &cur, &mut sub_out)?;
                prof::add(&prof::MLA, t);
            }
        }
        let t_hc = std::time::Instant::now();
        hc::combine_into(&sub_out, &residual, &mix, &mut next);
        std::mem::swap(&mut stream, &mut next);

        // --- FFN half -------------------------------------------------------
        residual.copy_from_slice(&stream);
        let mix = hc_mixes(&stream, &lw.hc_ffn, sh)?;
        hc::collapse_into(&stream, &mix.pre, &mut cur);
        rms_norm(&mut cur, lw.ffn_norm, sh.rms_eps)?;
        prof::add(&prof::HC, t_hc);

        let t = std::time::Instant::now();
        ffn_layer(sh, &lw.ffn, il, &cur, &mut sub_out)?;
        prof::add(&prof::FFN, t);
        let t_hc = std::time::Instant::now();
        hc::combine_into(&sub_out, &residual, &mix, &mut next);
        std::mem::swap(&mut stream, &mut next);
        prof::add(&prof::HC, t_hc);
    }

    // The trunk collapses with an UNWEIGHTED mean, not DeepSeek-V4.1's learned
    // gated head; glm5next ships no hc_head_* tensors.
    let t = std::time::Instant::now();
    let mut x = hc::mean(&stream);
    rms_norm(&mut x, w.output_norm, sh.rms_eps)?;

    let mut logits = vec![0.0f32; sh.n_vocab];
    w.output.apply(&x, &mut logits)?;
    prof::add(&prof::HEAD, t);

    state.len = pos + 1;
    Ok(logits)
}

/// VENDORED-LOCAL: GLM-5.3-Flash. How many tokens a batched prefill pass holds.
///
/// The chunk decides what is amortised. Bigger means more tokens sharing each
/// expert read, and `chunk * hc::HC * n_embd * 4` bytes of hidden state held at
/// once -- 8 MB at 128 tokens for this model, so the ceiling is not memory. It is
/// that a chunk's routes are a union: past a few hundred tokens nearly all 288
/// experts of every layer are in it, and reading all of them costs more than
/// reading eight per token did.
///
/// GLM5_PREFILL_CHUNK overrides it; 1 turns batching off.
pub fn prefill_chunk() -> usize {
    std::env::var("GLM5_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64)
}

/// VENDORED-LOCAL: GLM-5.3-Flash. A chunk of tokens, layer by layer.
///
/// [`forward_token`] walks the layers for one token, so a prompt of T tokens walks
/// them T times and reads every layer's routed experts T times over. Prefill
/// therefore costs the same per token as decode -- measured, ~100 ms -- and a
/// coder-cli system prompt of a few thousand tokens takes minutes before the first
/// reply.
///
/// This walks the layers once for the whole chunk. Two things follow:
///
///   * The FFN sees every token's route together, so a layer's distinct experts are
///     resolved and read once for all of them: see [`ExpertFfn::apply_batch`].
///   * The attention stays per token, because it has to. KDA's delta rule is a
///     recurrence -- token t+1's state update needs token t's -- and MLA appends to
///     the latent cache in position order. So they run in order inside the chunk,
///     against the same state the one-token path uses, which is what makes this
///     produce identical logits rather than merely similar ones.
///
/// Returns the **last** token's logits, like [`forward_prompt`], and advances the
/// state by `tokens.len()`.
pub fn forward_chunk(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    if hc::HC != sh.hc_count {
        return Err(LlamaError::Config(format!(
            "forward: hyper_connection.count is {} but this build fixes HC at {}",
            sh.hc_count,
            hc::HC
        )));
    }
    if w.layers.len() != sh.n_layer {
        return Err(LlamaError::Config(format!(
            "forward: {} layer weights for an {}-layer trunk",
            w.layers.len(),
            sh.n_layer
        )));
    }
    if tokens.is_empty() {
        return Err(LlamaError::Config("forward: empty chunk".into()));
    }
    let n = tokens.len();
    let base = state.len;
    if base + n > state.max_len {
        return Err(LlamaError::Config(format!(
            "forward: positions {base}..{} past the cache capacity {}",
            base + n,
            state.max_len
        )));
    }
    let n_embd = sh.n_embd;

    // One hyper-connection stream per token, seeded from its embedding exactly as
    // the one-token path seeds its own.
    let mut streams: Vec<Vec<f32>> = Vec::with_capacity(n);
    for &token in tokens {
        let tok = token as usize;
        if tok >= sh.n_vocab {
            return Err(LlamaError::Config(format!(
                "forward: token {tok} is outside the {}-entry vocabulary",
                sh.n_vocab
            )));
        }
        streams.push(hc::init(&w.tok_embd[tok * n_embd..(tok + 1) * n_embd]));
    }

    let mut cur = vec![0.0f32; n * n_embd];
    let mut sub_out = vec![0.0f32; n * n_embd];
    let mut residual = vec![0.0f32; n * hc::HC * n_embd];
    let mut mixes: Vec<hc::Mix> = Vec::with_capacity(n);
    let mut next = vec![0.0f32; hc::HC * n_embd];

    for (il, lw) in w.layers.iter().enumerate() {
        // --- attention half ----------------------------------------------------
        let t_hc = std::time::Instant::now();
        mixes.clear();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            residual[rlo..rlo + hc::HC * n_embd].copy_from_slice(&streams[t]);
            let mix = hc_mixes(&streams[t], &lw.hc_attn, sh)?;
            hc::collapse_into(&streams[t], &mix.pre, &mut cur[clo..clo + n_embd]);
            rms_norm(&mut cur[clo..clo + n_embd], lw.attn_norm, sh.rms_eps)?;
            mixes.push(mix);
        }
        prof::add(&prof::HC, t_hc);

        // In position order: both mixers carry state forward from one token to the
        // next, so this is the one part of a chunk that cannot be reordered.
        for t in 0..n {
            let clo = t * n_embd;
            let (x, out) = split_at_chunk(&cur, &mut sub_out, clo, n_embd);
            match &lw.attn {
                AttnW::Kda(kw) => {
                    let tm = std::time::Instant::now();
                    let ord = sh.kda_ordinal(il);
                    let State { conv, kda, kda_dev, .. } = &mut *state;
                    let cst = &mut conv[ord];
                    let kst = match kda_dev {
                        Some((dev, backend)) => KdaSt::Device {
                            t: &mut dev[ord],
                            backend: &**backend,
                        },
                        None => KdaSt::Host(&mut kda[ord]),
                    };
                    kda_layer(sh, kw, kst, cst, x, out)?;
                    prof::add(&prof::KDA, tm);
                }
                AttnW::Mla(mw) => {
                    let tm = std::time::Instant::now();
                    let ord = sh.mla_ordinal(il);
                    let State { latents, latents_dev, kpool, .. } = &mut *state;
                    let lat = &mut latents[ord];
                    let dev = latents_dev
                        .as_mut()
                        .map(|(dt, be)| (&mut dt[ord], &**be as &dyn Backend));
                    mla_layer(sh, mw, lat, dev, kpool, ord, base + t, x, out)?;
                    prof::add(&prof::MLA, tm);
                }
            }
        }

        let t_hc = std::time::Instant::now();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            hc::combine_into(
                &sub_out[clo..clo + n_embd],
                &residual[rlo..rlo + hc::HC * n_embd],
                &mixes[t],
                &mut next,
            );
            streams[t].copy_from_slice(&next);
        }

        // --- FFN half ----------------------------------------------------------
        mixes.clear();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            residual[rlo..rlo + hc::HC * n_embd].copy_from_slice(&streams[t]);
            let mix = hc_mixes(&streams[t], &lw.hc_ffn, sh)?;
            hc::collapse_into(&streams[t], &mix.pre, &mut cur[clo..clo + n_embd]);
            rms_norm(&mut cur[clo..clo + n_embd], lw.ffn_norm, sh.rms_eps)?;
            mixes.push(mix);
        }
        prof::add(&prof::HC, t_hc);

        let t = std::time::Instant::now();
        ffn_chunk(sh, &lw.ffn, il, n, &cur, &mut sub_out)?;
        prof::add(&prof::FFN, t);

        let t_hc = std::time::Instant::now();
        for t in 0..n {
            let (rlo, clo) = (t * hc::HC * n_embd, t * n_embd);
            hc::combine_into(
                &sub_out[clo..clo + n_embd],
                &residual[rlo..rlo + hc::HC * n_embd],
                &mixes[t],
                &mut next,
            );
            streams[t].copy_from_slice(&next);
        }
        prof::add(&prof::HC, t_hc);
    }

    // Only the last token's logits are wanted: the head is the single largest
    // matmul there is (vocab x n_embd) and a prompt needs none of the others.
    let t = std::time::Instant::now();
    let mut x = hc::mean(&streams[n - 1]);
    rms_norm(&mut x, w.output_norm, sh.rms_eps)?;
    let mut logits = vec![0.0f32; sh.n_vocab];
    w.output.apply(&x, &mut logits)?;
    prof::add(&prof::HEAD, t);

    state.len = base + n;
    Ok(logits)
}

/// One token's slice of the chunk's input and output buffers.
///
/// A free function because the borrow checker will not take `&cur[..]` and
/// `&mut sub_out[..]` from inside one expression that also borrows `state`.
fn split_at_chunk<'a>(
    cur: &'a [f32],
    sub_out: &'a mut [f32],
    lo: usize,
    n_embd: usize,
) -> (&'a [f32], &'a mut [f32]) {
    (&cur[lo..lo + n_embd], &mut sub_out[lo..lo + n_embd])
}

/// One FFN layer for a chunk of tokens.
///
/// Dense layers are per token: there is nothing to share, the weights are resident
/// and every token reads the same ones. A MoE layer batches, which is the whole
/// point of a chunk -- see [`ExpertFfn::apply_batch`].
fn ffn_chunk(
    sh: &Shape,
    w: &FfnW<'_>,
    il: usize,
    n: usize,
    xs: &[f32],
    outs: &mut [f32],
) -> Result<()> {
    let n_embd = sh.n_embd;
    match w {
        FfnW::Dense { .. } => {
            for t in 0..n {
                let lo = t * n_embd;
                let (x, out) = split_at_chunk(xs, outs, lo, n_embd);
                // The borrow split hands back a shared x; the dense arm needs the
                // same thing the one-token path passes it.
                ffn_layer(sh, w, il, x, out)?;
            }
            Ok(())
        }
        FfnW::Moe(m) => {
            // Route every token first, so the layer's experts are known as a set.
            let ne = sh.n_expert;
            let mut routes: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
            let t_router = std::time::Instant::now();
            for t in 0..n {
                let lo = t * n_embd;
                let mut logits = vec![0.0f32; ne];
                m.router.apply(&xs[lo..lo + n_embd], &mut logits)?;
                let (ids, weights) = routing::route_token(
                    &logits,
                    Some(m.probs_b),
                    sh.n_expert_used,
                    sh.expert_weights_norm,
                    sh.expert_weights_scale,
                );
                routes.push(ids.into_iter().zip(weights).collect());
            }
            prof::add(&prof::FFN_ROUTER, t_router);

            let t_routed = std::time::Instant::now();
            m.experts
                .apply_batch(m.ord, &routes, xs, n_embd, sh.swiglu_clamp_exp[il], outs)?;
            prof::add(&prof::FFN_ROUTED, t_routed);

            // The shared expert runs for every token and is unscaled.
            let t_shared = std::time::Instant::now();
            let sff = sh.n_ff_shexp;
            for t in 0..n {
                let lo = t * n_embd;
                let mut sgu = vec![0.0f32; 2 * sff];
                m.sh_gate_up.apply(&xs[lo..lo + n_embd], sff, &mut sgu)?;
                let (sg, su) = sgu.split_at(sff);
                let mut sh_h = vec![0.0f32; sff];
                swiglu_clamped(sg, su, sh.swiglu_clamp_shexp[il], &mut sh_h)?;
                let mut s_out = vec![0.0f32; n_embd];
                m.sh_down.apply(&sh_h, &mut s_out)?;
                for (o, &sv) in outs[lo..lo + n_embd].iter_mut().zip(s_out.iter()) {
                    *o += sv;
                }
            }
            prof::add(&prof::FFN_SHARED, t_shared);
            Ok(())
        }
    }
}

/// VENDORED-LOCAL: GLM-5.3-Flash. Everything a prompt leaves behind, as plain f32.
///
/// A prompt is the expensive part of a request and a harness sends the same one
/// every time: coder-cli's system prompt is thousands of tokens, and reading it
/// takes minutes. This is what lets a later process start where an earlier one
/// finished instead of reading it again -- the same trick `dsv41_cuda::Snapshot`
/// does for DeepSeek, whose states `oaiy-llm-server` already keeps on disk.
///
/// The recurrent state is fixed-size (34 layers x 4.2 MB) and has to be kept whole.
/// The caches are not: `latents` and the indexer buffers are allocated for
/// `max_len` and written from the front, so only the first `len` tokens' rows mean
/// anything and the rest would be zeros on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSnapshot {
    /// Tokens this state covers.
    pub len: usize,
    /// Per KDA layer, the delta-rule state.
    pub kda: Vec<Vec<f32>>,
    /// Per KDA layer, the depthwise conv's pre-conv window.
    pub conv: Vec<Vec<f32>>,
    /// Per MLA layer, the cached latent rows for `len` tokens.
    pub latents: Vec<Vec<f32>>,
    /// Per MLA layer, the indexer rows for `len` tokens.
    pub kpool: Vec<Vec<f32>>,
}

impl StateSnapshot {
    /// Size of [`Self::encode`]'s output.
    pub fn encoded_len(&self) -> usize {
        let group = |g: &Vec<Vec<f32>>| 4 + g.iter().map(|v| 8 + 4 * v.len()).sum::<usize>();
        8 + group(&self.kda) + group(&self.conv) + group(&self.latents) + group(&self.kpool)
    }

    /// Append the snapshot's bytes, little-endian.
    ///
    /// Four groups of per-layer f32, each length-prefixed, after the token count.
    /// Plain and self-describing on purpose: a state that cannot be read back is
    /// worse than one that was never kept, and the reader checks every length
    /// against the state it is filling anyway.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.reserve(self.encoded_len());
        out.extend((self.len as u64).to_le_bytes());
        for group in [&self.kda, &self.conv, &self.latents, &self.kpool] {
            out.extend((group.len() as u32).to_le_bytes());
            for v in group {
                out.extend((v.len() as u64).to_le_bytes());
                for &f in v {
                    out.extend(f.to_le_bytes());
                }
            }
        }
    }

    /// Read back what [`Self::encode`] wrote.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = || LlamaError::Config("forward: damaged prompt state".into());
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = at.checked_add(n).ok_or_else(bad)?;
            let s = bytes.get(at..end).ok_or_else(bad)?;
            at = end;
            Ok(s)
        };
        let len = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?) as usize;
        let mut groups: Vec<Vec<Vec<f32>>> = Vec::with_capacity(4);
        for _ in 0..4 {
            let n = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
            let mut group = Vec::with_capacity(n);
            for _ in 0..n {
                let count = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?) as usize;
                let raw = take(count.checked_mul(4).ok_or_else(bad)?)?;
                group.push(
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
            groups.push(group);
        }
        if at != bytes.len() {
            return Err(LlamaError::Config(format!(
                "forward: prompt state has {} trailing bytes",
                bytes.len() - at
            )));
        }
        let mut it = groups.into_iter();
        Ok(Self {
            len,
            kda: it.next().ok_or_else(bad)?,
            conv: it.next().ok_or_else(bad)?,
            latents: it.next().ok_or_else(bad)?,
            kpool: it.next().ok_or_else(bad)?,
        })
    }
}

impl State {
    /// VENDORED-LOCAL: GLM-5.3-Flash. Copy this state out, for disk or another
    /// process.
    ///
    /// Downloads the recurrent state when it lives on a card: 143 MB, once per
    /// snapshot rather than once per layer per token, which is why the state is
    /// kept there in the first place.
    pub fn snapshot(&self, kv_lora: usize) -> Result<StateSnapshot> {
        let kda = match &self.kda_dev {
            Some((tensors, backend)) => tensors
                .iter()
                .map(|t| backend.to_host(t.clone()).data().to_vec())
                .collect(),
            None => self.kda.clone(),
        };
        let keep = self.len * kv_lora;
        Ok(StateSnapshot {
            len: self.len,
            kda,
            conv: self.conv.clone(),
            latents: self
                .latents
                .iter()
                .map(|l| l.get(..keep.min(l.len())).unwrap_or(l).to_vec())
                .collect(),
            kpool: self.kpool.rows_upto(self.len),
        })
    }

    /// Put a snapshot back, so the next token continues from it.
    ///
    /// Everything past `len` is zeroed rather than left alone: a state being
    /// restored may be shorter than whatever this one held, and the difference has
    /// to read as "not written yet" and not as another prompt's rows.
    pub fn restore(&mut self, snap: &StateSnapshot, kv_lora: usize) -> Result<()> {
        if snap.len > self.max_len {
            return Err(LlamaError::Config(format!(
                "forward: restoring {} tokens into a {}-token state",
                snap.len, self.max_len
            )));
        }
        if snap.kda.len() != self.conv.len()
            || snap.conv.len() != self.conv.len()
            || snap.latents.len() != self.latents.len()
        {
            return Err(LlamaError::Config(format!(
                "forward: snapshot has {} KDA / {} conv / {} MLA layers, state has {} / {}",
                snap.kda.len(),
                snap.conv.len(),
                snap.latents.len(),
                self.conv.len(),
                self.latents.len()
            )));
        }
        match &mut self.kda_dev {
            Some((tensors, backend)) => {
                for (t, src) in tensors.iter_mut().zip(&snap.kda) {
                    let want = t.numel();
                    if src.len() != want {
                        return Err(LlamaError::Config(format!(
                            "forward: restoring {} KDA values into {want}",
                            src.len()
                        )));
                    }
                    *t = backend.to_device(Tensor::from_vec(src.clone(), t.shape().to_vec()));
                }
            }
            None => {
                for (dst, src) in self.kda.iter_mut().zip(&snap.kda) {
                    if src.len() != dst.len() {
                        return Err(LlamaError::Config(format!(
                            "forward: restoring {} KDA values into {}",
                            src.len(),
                            dst.len()
                        )));
                    }
                    dst.copy_from_slice(src);
                }
            }
        }
        for (dst, src) in self.conv.iter_mut().zip(&snap.conv) {
            if src.len() != dst.len() {
                return Err(LlamaError::Config(format!(
                    "forward: restoring {} conv values into {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        let _ = kv_lora;
        for (dst, src) in self.latents.iter_mut().zip(&snap.latents) {
            if src.len() > dst.len() {
                return Err(LlamaError::Config(format!(
                    "forward: restoring {} latent values into {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst[..src.len()].copy_from_slice(src);
            dst[src.len()..].fill(0.0);
        }
        // And the device mirror, from what was just put back.
        if let Some((dev, backend)) = &mut self.latents_dev {
            for (t, src) in dev.iter_mut().zip(&self.latents) {
                let shape = t.shape().to_vec();
                *t = backend.to_device(Tensor::from_vec(src.clone(), shape));
            }
        }
        self.kpool.restore_rows(&snap.kpool)?;
        self.len = snap.len;
        Ok(())
    }
}

/// Run a whole prompt, returning the last token's logits. Prefill is this loop —
/// the reference's own autoregressive path, one token at a time.
pub fn forward_prompt(
    sh: &Shape,
    w: &ModelW<'_>,
    state: &mut State,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    if tokens.is_empty() {
        return Err(LlamaError::Config("forward: empty prompt".into()));
    }
    let mut last = Vec::new();
    for &t in tokens {
        last = forward_token(sh, w, state, t)?;
    }
    Ok(last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next::LayerKind;

    /// Deterministic pseudo-random weights, small enough not to saturate.
    fn fill(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i * 2654435761 + seed * 40503) % 1009;
                (h as f32 - 504.0) / 5040.0
            })
            .collect()
    }

    /// A tiny model with the real *structure*: a dense KDA block, a MoE KDA
    /// block and a MoE MLA block, driven by a per-layer kind array.
    fn shape() -> Shape {
        let kinds = vec![LayerKind::Kda, LayerKind::Kda, LayerKind::Mla];
        Shape {
            n_embd: 8,
            n_vocab: 11,
            n_layer: 3,
            n_head: 2,
            kda_head_dim: 4,
            d_conv: 4,
            q_lora: 6,
            kv_lora: 4,
            qk_head: 4,
            v_head: 4,
            d_idx: 4,
            n_ihead: 2,
            kpool: 2,
            // n_select = 3, and select_k = 1 pool, so with 4+ tokens the
            // indexer genuinely leaves cells masked rather than selecting
            // everything visible.
            indexer_top_k: 2,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 6,
            n_ff_shexp: 6,
            n_ff_dense: 10,
            n_dense_lead: 1,
            layer_kinds: kinds,
            hc_count: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            rms_eps: 1e-5,
            norm_eps: 1e-6,
            kda_gate_lower_bound: -5.0,
            expert_weights_norm: true,
            expert_weights_scale: 2.5,
            swiglu_clamp_exp: vec![10.0; 3],
            swiglu_clamp_shexp: vec![10.0; 3],
            // Above n_select (3), so the MLA layer scores.
            max_len: 32,
        }
    }

    /// An owning [`ExpertFfn`] for the fixture: one per MoE layer, so each
    /// carries a single layer and is addressed with `ord = 0`.
    struct TestExperts {
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
        per: usize,
        ff: usize,
        n_embd: usize,
    }

    impl ExpertFfn for TestExperts {
        fn apply(
            &self,
            _ord: usize,
            e: usize,
            x: &[f32],
            limit: f32,
            out: &mut [f32],
        ) -> Result<()> {
            let off = e * self.per;
            let (mut g, mut u, mut h) = (
                vec![0.0f32; self.ff],
                vec![0.0f32; self.ff],
                vec![0.0f32; self.ff],
            );
            matvec(&self.gate[off..off + self.per], x, &mut g)?;
            matvec(&self.up[off..off + self.per], x, &mut u)?;
            swiglu_clamped(&g, &u, limit, &mut h)?;
            matvec(&self.down[off..off + self.per], &h, out)?;
            let _ = self.n_embd;
            Ok(())
        }
    }

    /// Owns the buffers a [`ModelW`] borrows.
    struct Owned {
        tok_embd: Vec<f32>,
        output_norm: Vec<f32>,
        output: Vec<f32>,
        per_layer: Vec<std::collections::BTreeMap<&'static str, Vec<f32>>>,
        /// `None` for the dense leading blocks.
        experts: Vec<Option<TestExperts>>,
    }

    fn weights(sh: &Shape) -> Owned {
        let mut experts: Vec<Option<TestExperts>> = Vec::new();
        let di = sh.d_inner();
        let hd = sh.kda_head_dim;
        let e = sh.n_embd;
        let mut per_layer = Vec::new();
        for il in 0..sh.n_layer {
            let mut m: std::collections::BTreeMap<&'static str, Vec<f32>> = Default::default();
            let s = il * 17 + 1;
            m.insert("attn_norm", vec![1.0; e]);
            m.insert("ffn_norm", vec![1.0; e]);
            for (k, n) in [
                ("hc_attn_fn", hc::MIX * sh.hc_count * e),
                ("hc_ffn_fn", hc::MIX * sh.hc_count * e),
            ] {
                m.insert(k, fill(n, s + k.len()));
            }
            m.insert("hc_attn_base", vec![0.0; hc::MIX]);
            m.insert("hc_ffn_base", vec![0.0; hc::MIX]);
            m.insert("hc_attn_scale", vec![1.0, 1.0, 1.0]);
            m.insert("hc_ffn_scale", vec![1.0, 1.0, 1.0]);

            match sh.layer_kinds[il] {
                LayerKind::Kda => {
                    for (k, n) in [
                        ("q", di * e),
                        ("k", di * e),
                        ("v", di * e),
                        ("f_a", hd * e),
                        ("f_b", di * hd),
                        ("g_a", hd * e),
                        ("g_b", di * hd),
                        ("beta", sh.n_head * e),
                        ("attn_out", e * di),
                    ] {
                        m.insert(k, fill(n, s + k.len() * 3));
                    }
                    for k in ["conv_q", "conv_k", "conv_v"] {
                        m.insert(k, fill(di * sh.d_conv, s + k.len() * 5));
                    }
                    m.insert("a", vec![-1.0; sh.n_head]);
                    m.insert("dt_bias", vec![0.0; di]);
                    m.insert("o_norm", vec![1.0; hd]);
                }
                LayerKind::Mla => {
                    for (k, n) in [
                        ("q_a", sh.q_lora * e),
                        ("q_b", sh.n_head * sh.qk_head * sh.q_lora),
                        ("kv_a_mqa", sh.kv_lora * e),
                        ("k_b", sh.n_head * sh.kv_lora * sh.qk_head),
                        ("v_b", sh.n_head * sh.v_head * sh.kv_lora),
                        ("attn_out", e * sh.n_head * sh.v_head),
                        ("idx_attn_k", sh.d_idx * e),
                        ("idx_attn_q_b", sh.n_ihead * sh.d_idx * sh.q_lora),
                        ("idx_proj", sh.n_ihead * e),
                        ("idx_gate", sh.d_idx * e),
                        ("idx_ape", sh.kpool * sh.d_idx),
                    ] {
                        m.insert(k, fill(n, s + k.len() * 7));
                    }
                    m.insert("q_a_norm", vec![1.0; sh.q_lora]);
                    m.insert("kv_a_norm", vec![1.0; sh.kv_lora]);
                    m.insert("idx_k_norm", vec![1.0; sh.d_idx]);
                    m.insert("idx_k_norm_b", vec![0.0; sh.d_idx]);
                }
            }

            if il < sh.n_dense_lead {
                m.insert("ffn_gate", fill(sh.n_ff_dense * e, s + 101));
                m.insert("ffn_up", fill(sh.n_ff_dense * e, s + 103));
                m.insert("ffn_down", fill(e * sh.n_ff_dense, s + 107));
            } else {
                m.insert("router", fill(sh.n_expert * e, s + 109));
                m.insert("probs_b", vec![0.0; sh.n_expert]);
                experts.push(Some(TestExperts {
                    gate: fill(sh.n_expert * sh.n_ff_exp * e, s + 113),
                    up: fill(sh.n_expert * sh.n_ff_exp * e, s + 127),
                    down: fill(sh.n_expert * e * sh.n_ff_exp, s + 131),
                    per: sh.n_ff_exp * e,
                    ff: sh.n_ff_exp,
                    n_embd: e,
                }));
                m.insert("sh_gate", fill(sh.n_ff_shexp * e, s + 137));
                m.insert("sh_up", fill(sh.n_ff_shexp * e, s + 139));
                m.insert("sh_down", fill(e * sh.n_ff_shexp, s + 149));
            }
            if il < sh.n_dense_lead {
                experts.push(None);
            }
            per_layer.push(m);
        }
        Owned {
            tok_embd: fill(sh.n_vocab * sh.n_embd, 3),
            output_norm: vec![1.0; sh.n_embd],
            output: fill(sh.n_vocab * sh.n_embd, 5),
            per_layer,
            experts,
        }
    }

    fn model<'a>(sh: &Shape, o: &'a Owned) -> ModelW<'a> {
        let g = |il: usize, k: &str| -> &'a [f32] { o.per_layer[il][k].as_slice() };
        let mh = |il: usize, k: &str| -> Mat<'a> { Mat::Host(o.per_layer[il][k].as_slice()) };
        let mut layers = Vec::new();
        for il in 0..sh.n_layer {
            let attn = match sh.layer_kinds[il] {
                LayerKind::Kda => AttnW::Kda(KdaW {
                    qk: Pair::Split(mh(il, "q"), mh(il, "k")),
                    v: mh(il, "v"),
                    conv_q: g(il, "conv_q"),
                    conv_k: g(il, "conv_k"),
                    conv_v: g(il, "conv_v"),
                    fga: Pair::Split(mh(il, "f_a"), mh(il, "g_a")),
                    f_b: mh(il, "f_b"),
                    g_b: mh(il, "g_b"),
                    beta: mh(il, "beta"),
                    a: g(il, "a"),
                    dt_bias: g(il, "dt_bias"),
                    o_norm: g(il, "o_norm"),
                    out: mh(il, "attn_out"),
                }),
                LayerKind::Mla => AttnW::Mla(MlaW {
                    q_a: mh(il, "q_a"),
                    q_a_norm: g(il, "q_a_norm"),
                    q_b: mh(il, "q_b"),
                    kv_a_mqa: mh(il, "kv_a_mqa"),
                    kv_a_norm: g(il, "kv_a_norm"),
                    k_b: Bat::Host(g(il, "k_b")),
                    v_b: Bat::Host(g(il, "v_b")),
                    out: mh(il, "attn_out"),
                    indexer: IndexerW {
                        attn_k: mh(il, "idx_attn_k"),
                        attn_q_b: mh(il, "idx_attn_q_b"),
                        k_norm: g(il, "idx_k_norm"),
                        k_norm_bias: g(il, "idx_k_norm_b"),
                        proj: mh(il, "idx_proj"),
                        comp_gate: mh(il, "idx_gate"),
                        comp_ape: g(il, "idx_ape"),
                    },
                }),
            };
            let ffn = if il < sh.n_dense_lead {
                FfnW::Dense {
                    gate: mh(il, "ffn_gate"),
                    up: mh(il, "ffn_up"),
                    down: mh(il, "ffn_down"),
                }
            } else {
                FfnW::Moe(MoeW {
                    router: mh(il, "router"),
                    probs_b: g(il, "probs_b"),
                    experts: o.experts[il]
                        .as_ref()
                        .expect("a MoE layer needs an expert source"),
                    ord: 0,
                    sh_gate_up: Pair::Split(mh(il, "sh_gate"), mh(il, "sh_up")),
                    sh_down: mh(il, "sh_down"),
                })
            };
            layers.push(LayerW {
                attn_norm: g(il, "attn_norm"),
                ffn_norm: g(il, "ffn_norm"),
                hc_attn: HcW {
                    fn_: mh(il, "hc_attn_fn"),
                    base: g(il, "hc_attn_base"),
                    scale: g(il, "hc_attn_scale"),
                },
                hc_ffn: HcW {
                    fn_: mh(il, "hc_ffn_fn"),
                    base: g(il, "hc_ffn_base"),
                    scale: g(il, "hc_ffn_scale"),
                },
                attn,
                ffn,
            });
        }
        ModelW {
            tok_embd: &o.tok_embd,
            output_norm: &o.output_norm,
            output: Mat::Host(&o.output),
            layers,
        }
    }

    #[test]
    fn a_prompt_produces_finite_varied_logits() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();

        let tokens = [1u32, 4, 2, 7, 0, 3];
        let logits = forward_prompt(&sh, &m, &mut st, &tokens).unwrap();

        assert_eq!(logits.len(), sh.n_vocab);
        assert!(logits.iter().all(|x| x.is_finite()), "logits must be finite");
        let first = logits[0];
        assert!(
            logits.iter().any(|&x| (x - first).abs() > 1e-6),
            "logits must not be uniform: {logits:?}"
        );
        assert_eq!(st.len, tokens.len());
    }

    /// A chunk must give exactly what the one-token loop gives.
    ///
    /// Bit-identical, not close: the batched pass reorders nothing that the
    /// arithmetic depends on. It walks the layers once instead of once a token, but
    /// within a layer the attention still runs token by token in position order
    /// (KDA is a recurrence, MLA appends to the latent cache), and the FFN's default
    /// `apply_batch` is the per-token loop. A real implementation may reassociate
    /// and is held to a tolerance on the released weights instead; this is the gate
    /// on the scaffolding -- the state threading, the per-token hyper-connection
    /// mixes, the position arithmetic.
    #[test]
    fn a_chunk_matches_the_one_token_loop() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let tokens = [1u32, 4, 2, 7, 0, 3];

        let mut seq = State::new(&sh).unwrap();
        let want = forward_prompt(&sh, &m, &mut seq, &tokens).unwrap();

        let mut bat = State::new(&sh).unwrap();
        let got = forward_chunk(&sh, &m, &mut bat, &tokens).unwrap();

        assert_eq!(got.len(), want.len());
        assert_eq!(got, want, "a chunk diverged from the one-token loop");
        assert_eq!(bat.len, seq.len, "the chunk left the state at a different length");
    }

    /// Chunk boundaries must not matter: the state carries across them.
    #[test]
    fn chunks_of_different_sizes_agree() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let tokens = [1u32, 4, 2, 7, 0, 3, 5, 2];

        let mut whole = State::new(&sh).unwrap();
        let want = forward_chunk(&sh, &m, &mut whole, &tokens).unwrap();

        // The same tokens in two chunks, then in three.
        for splits in [vec![5usize, 3], vec![2, 2, 4], vec![1, 6, 1]] {
            let mut st = State::new(&sh).unwrap();
            let mut at = 0usize;
            let mut last = Vec::new();
            for len in &splits {
                last = forward_chunk(&sh, &m, &mut st, &tokens[at..at + len]).unwrap();
                at += len;
            }
            assert_eq!(at, tokens.len());
            assert_eq!(last, want, "chunking as {splits:?} changed the answer");
            assert_eq!(st.len, whole.len);
        }
    }

    /// And a chunk of one is the one-token path.
    #[test]
    fn a_chunk_of_one_is_a_token() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);

        let mut a = State::new(&sh).unwrap();
        let want = forward_token(&sh, &m, &mut a, 3).unwrap();
        let mut b = State::new(&sh).unwrap();
        let got = forward_chunk(&sh, &m, &mut b, &[3]).unwrap();
        assert_eq!(got, want);
    }

    /// A restored snapshot continues a sequence exactly as the original would.
    ///
    /// This is the gate on the prompt cache: the point of keeping a prompt's state
    /// is that the tokens after it come out the same, so what is checked is not the
    /// snapshot's bytes but the next token's logits.
    #[test]
    fn a_restored_snapshot_continues_the_sequence() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let prompt = [1u32, 4, 2, 7];
        let after = [3u32, 5];

        // Straight through: prompt, then two more tokens.
        let mut direct = State::new(&sh).unwrap();
        forward_prompt(&sh, &m, &mut direct, &prompt).unwrap();
        let mut want = Vec::new();
        for &t in &after {
            want = forward_token(&sh, &m, &mut direct, t).unwrap();
        }

        // The same, but the prompt's state came off a snapshot.
        let mut taken = State::new(&sh).unwrap();
        forward_prompt(&sh, &m, &mut taken, &prompt).unwrap();
        let snap = taken.snapshot(sh.kv_lora).unwrap();
        assert_eq!(snap.len, prompt.len());

        let mut fresh = State::new(&sh).unwrap();
        fresh.restore(&snap, sh.kv_lora).unwrap();
        assert_eq!(fresh.len, prompt.len());
        let mut got = Vec::new();
        for &t in &after {
            got = forward_token(&sh, &m, &mut fresh, t).unwrap();
        }

        assert_eq!(got, want, "a restored state diverged from one built in place");
    }

    /// Restoring over a longer state must not leave its rows behind.
    #[test]
    fn restoring_a_shorter_state_clears_what_was_there() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);

        // A short snapshot.
        let mut short = State::new(&sh).unwrap();
        forward_prompt(&sh, &m, &mut short, &[1u32, 4]).unwrap();
        let snap = short.snapshot(sh.kv_lora).unwrap();

        // A state that has seen more, then restored back to the short one.
        let mut long = State::new(&sh).unwrap();
        forward_prompt(&sh, &m, &mut long, &[7u32, 0, 3, 5, 2, 6]).unwrap();
        long.restore(&snap, sh.kv_lora).unwrap();

        // It must now behave exactly like the short one.
        let mut a = long;
        let mut b = State::new(&sh).unwrap();
        b.restore(&snap, sh.kv_lora).unwrap();
        assert_eq!(
            forward_token(&sh, &m, &mut a, 3).unwrap(),
            forward_token(&sh, &m, &mut b, 3).unwrap(),
            "rows from the longer sequence survived the restore"
        );
    }

    /// The wire format round-trips, and refuses damage.
    #[test]
    fn a_snapshot_survives_encoding() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();
        forward_prompt(&sh, &m, &mut st, &[1u32, 4, 2]).unwrap();
        let snap = st.snapshot(sh.kv_lora).unwrap();

        let mut bytes = Vec::new();
        snap.encode(&mut bytes);
        assert_eq!(bytes.len(), snap.encoded_len(), "encoded_len disagrees with encode");
        let back = StateSnapshot::decode(&bytes).expect("decode");
        assert_eq!(back, snap);

        // A truncated state is rejected rather than half-read.
        assert!(StateSnapshot::decode(&bytes[..bytes.len() - 4]).is_err());
        // So are trailing bytes, which would mean a format mismatch.
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(StateSnapshot::decode(&extra).is_err());
    }

    #[test]
    fn the_same_prompt_is_deterministic() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let tokens = [2u32, 2, 5, 1];

        let mut a = State::new(&sh).unwrap();
        let la = forward_prompt(&sh, &m, &mut a, &tokens).unwrap();
        let mut b = State::new(&sh).unwrap();
        let lb = forward_prompt(&sh, &m, &mut b, &tokens).unwrap();
        assert_eq!(la, lb);

        // And a reset returns the state to its opening condition.
        a.reset();
        let lc = forward_prompt(&sh, &m, &mut a, &tokens).unwrap();
        assert_eq!(la, lc);
    }

    /// State must actually accumulate: the second occurrence of a token cannot
    /// produce the same logits as the first, or attention and the recurrence are
    /// not seeing history.
    #[test]
    fn history_changes_the_output_for_a_repeated_token() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();

        let first = forward_token(&sh, &m, &mut st, 3).unwrap();
        let second = forward_token(&sh, &m, &mut st, 3).unwrap();
        let diff: f32 = first.iter().zip(&second).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "history must change the output, diff {diff}");
    }

    /// The caches advance as the geometry says: one latent row and one indexer
    /// cell per token, and a pooled key every `kpool` tokens.
    #[test]
    fn caches_advance_with_the_geometry() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();

        for t in 0..6u32 {
            forward_token(&sh, &m, &mut st, t % sh.n_vocab as u32).unwrap();
        }
        assert_eq!(st.len, 6);
        assert_eq!(kpool::n_complete_pools(st.len, sh.kpool), 3);

        // Every completed pool has a non-zero pooled key; the next one does not.
        let kc = st.kpool_cache();
        for p in 0..3 {
            let pooled = kc.pooled(0, p, sh.kpool).unwrap();
            assert!(
                pooled.iter().any(|&x| x != 0.0),
                "pool {p} should have been pooled"
            );
        }
    }

    /// A shape whose `n_select` exceeds the cache capacity, so the indexer never
    /// scores and the MLA layer takes the plain causal path. Only `indexer_top_k`
    /// changes, so it shares the sparse fixture's weights exactly.
    fn dense_shape() -> Shape {
        let mut sh = shape();
        sh.indexer_top_k = 64;
        assert!(!kpool::indexer_scores(sh.max_len, sh.indexer_top_k, sh.kpool));
        sh
    }

    /// Below the indexer's threshold the MLA layer must still run, and still see
    /// history, via dense causal attention.
    #[test]
    fn a_small_cache_takes_the_dense_attention_path() {
        let sh = dense_shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();
        let first = forward_token(&sh, &m, &mut st, 1).unwrap();
        forward_prompt(&sh, &m, &mut st, &[2, 3, 4, 5]).unwrap();
        assert_eq!(st.len, 5);
        // The dense path is not a no-op: a later token differs from the first.
        let later = forward_token(&sh, &m, &mut st, 1).unwrap();
        let diff: f32 = first.iter().zip(&later).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "dense attention must accumulate history");
    }

    /// The sparse path must actually mask something. With `select_k = 1` of 3
    /// visible pools it attends to 2 of 6 cells, so its logits must differ from
    /// the dense run over the same weights and the same prompt.
    #[test]
    fn the_sparse_path_differs_from_the_dense_path() {
        let sparse = shape();
        let dense = dense_shape();
        assert_eq!(kpool::n_select(sparse.indexer_top_k, sparse.kpool), 3);
        assert!(kpool::indexer_scores(sparse.max_len, sparse.indexer_top_k, sparse.kpool));
        // 3 complete pools at len 6, but only one may be selected.
        assert_eq!(kpool::select_k(3, sparse.indexer_top_k, sparse.kpool).unwrap(), 1);

        let o = weights(&sparse);
        let tokens = [1u32, 2, 3, 4, 0, 6];

        let ms = model(&sparse, &o);
        let mut ss = State::new(&sparse).unwrap();
        let sparse_logits = forward_prompt(&sparse, &ms, &mut ss, &tokens).unwrap();

        let md = model(&dense, &o);
        let mut sd = State::new(&dense).unwrap();
        let dense_logits = forward_prompt(&dense, &md, &mut sd, &tokens).unwrap();

        assert!(sparse_logits.iter().all(|x| x.is_finite()));
        assert!(dense_logits.iter().all(|x| x.is_finite()));
        let diff: f32 = sparse_logits
            .iter()
            .zip(&dense_logits)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-7,
            "masking 4 of 6 cells must change the logits, diff {diff}"
        );
    }

    /// `hc::init` then `hc::mean` is the identity, so a layer whose sublayers
    /// output zero and whose mixer is the identity leaves the stream alone. This
    /// pins that the loop's HC wrapping is a residual path, not a replacement.
    #[test]
    fn mean_of_the_initial_stream_is_the_embedding() {
        let embd = vec![0.25f32, -1.5, 3.0, 0.0];
        let s = hc::init(&embd);
        assert_eq!(hc::mean(&s), embd);
    }

    #[test]
    fn clamped_swiglu_matches_the_reference_branch() {
        // limit applies to silu(gate) as an UPPER bound and to up symmetrically.
        let gate = vec![40.0f32, -40.0, 1.0];
        let up = vec![100.0f32, -100.0, 2.0];
        let mut out = vec![0.0f32; 3];
        swiglu_clamped(&gate, &up, 10.0, &mut out).unwrap();

        // silu(40) ~ 40 -> clamped to 10; up 100 -> clamped to 10.
        assert!((out[0] - 100.0).abs() < 1e-3, "got {}", out[0]);
        // silu(-40) ~ 0, below the limit, untouched; up -100 -> -10.
        assert!(out[1].abs() < 1e-3, "got {}", out[1]);
        // Well inside the limit: a plain SwiGLU.
        assert!((out[2] - silu(1.0) * 2.0).abs() < 1e-6);

        // A zero limit disables the clamp.
        let mut raw = vec![0.0f32; 3];
        swiglu_clamped(&gate, &up, 0.0, &mut raw).unwrap();
        assert!(raw[0] > 1000.0, "unclamped, got {}", raw[0]);
    }

    #[test]
    fn rejects_a_bad_token_or_an_overfull_cache() {
        let sh = shape();
        let o = weights(&sh);
        let m = model(&sh, &o);
        let mut st = State::new(&sh).unwrap();
        assert!(forward_token(&sh, &m, &mut st, 999).is_err(), "bad token");
        assert!(forward_prompt(&sh, &m, &mut st, &[]).is_err(), "empty prompt");

        let mut tiny = shape();
        tiny.max_len = 2;
        let mut st2 = State::new(&tiny).unwrap();
        let m2 = model(&tiny, &o);
        forward_token(&tiny, &m2, &mut st2, 1).unwrap();
        forward_token(&tiny, &m2, &mut st2, 1).unwrap();
        assert!(
            forward_token(&tiny, &m2, &mut st2, 1).is_err(),
            "past capacity"
        );
    }
}
