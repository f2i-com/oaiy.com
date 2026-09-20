// VENDORED-LOCAL: whole module. GLM-5.3-Flash hyper-connections.
//! Hyper-connections (mHC): the residual stream is `HC` = 4 parallel copies of
//! the model width, and each sublayer derives three coefficient sets from the
//! stream itself — `pre` (collapse the copies into the sublayer's input),
//! `post` (spread its output back over the copies) and `comb` (mix the residual
//! copies, made doubly stochastic by Sinkhorn iterations).
//!
//! Reference: llama.cpp `llama_model_deepseek4::graph::build_hc_pre` /
//! `build_hc_sinkhorn` / `build_hc_post` / `dsv4_hc_mean`
//! (`src/models/deepseek4.cpp`). `glm5next::graph` inherits from
//! `deepseek4::graph` and calls these unchanged, so the formulation here is
//! DeepSeek-V4.1's — which is why `dsv41::hc` in this workspace is the same
//! math. This is a **port, not a shared module**: `dsv41` is std-only, takes its
//! weights through `StIndex` (safetensors) and rounds activations to bf16.
//!
//! ## Three deliberate divergences from `dsv41::hc`
//!
//!   1. **f32 activations, no bf16 rounding.** `dsv41::hc::pre` / `post` wrap
//!      every result in `to_bf16` because DeepSeek-V4.1's reference keeps its
//!      residual in bf16. The GGUF path is f32 end to end, so rounding here
//!      would inject error the reference does not have. There is a test that
//!      pins this.
//!   2. **The trunk collapses with an unweighted [`mean`].** DeepSeek-V4.1 ends
//!      with `build_hc_head`, a learned gated collapse; glm5next uses
//!      `build_hc_mean` (the PR promotes that helper to a member for exactly
//!      this reason). glm5next has no `hc_head_*` tensors.
//!   3. **Streams start as exact copies** ([`init`]) — no scaling and no
//!      one-hot into stream 0.
//!
//! ## The eps that is easy to get wrong
//!
//! There are two, and they are not interchangeable:
//!   * The RMSNorm over the flattened `[HC * n_embd]` stream uses the model's
//!     **`attention.layer_norm_rms_epsilon`** (`norm_rms_eps` in the reference,
//!     1e-5 for glm5next) — see `build_hc_pre`'s
//!     `ggml_rms_norm(flat, norm_rms_eps)`.
//!   * `hyper_connection.epsilon` (1e-6) is the `+eps` added to `pre` and to
//!     `comb` inside the Sinkhorn loop. It never touches the RMSNorm.
//!
//! `dsv41::hc`'s own unit test passes `1e-20` for the first of these, which is
//! a test convenience, not the model's value. Do not copy it.
//!
//! **Wired into `forward`: no.** `Glm5NextModel::forward` is still the pending
//! stub; this is piece (2), tested standalone.

use crate::{LlamaError, Result};

/// Hyper-connection multiplicity. `hyper_connection.count` is 4 and the
/// reference asserts it (`GGML_ASSERT(hc == 4)`), so the coefficient sets are
/// fixed-size arrays rather than allocations.
pub const HC: usize = 4;

/// Mix coefficients per token and sublayer: `HC` pre, `HC` post, `HC*HC` comb.
/// Matches `hc_mix_dim = (2 + hc) * hc` = 24, the second dim of `hc_*_fn`.
pub const MIX: usize = (2 + HC) * HC;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One sublayer's per-token coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mix {
    /// Collapse weights, one per stream. Strictly positive (`sigmoid + eps`).
    pub pre: [f32; HC],
    /// Spread weights, one per stream. In `(0, 2)` (`2 * sigmoid`).
    pub post: [f32; HC],
    /// `comb[src][dst]`: how much residual copy `src` contributes to output
    /// copy `dst`. Doubly stochastic after Sinkhorn.
    pub comb: [[f32; HC]; HC],
}

/// Derive the coefficients for one token from its `[HC * dim]` stream.
///
/// `fn_` is `hc_*_fn` dequantised to f32, `[MIX, HC * dim]` row-major (the
/// loader reverses GGUF dims, so `MIX` is the outer dim). `base` is `[MIX]`,
/// `scale` is `[3]` — pre, post, comb in that order.
///
/// `rms_eps` is the model's RMSNorm epsilon; `hc_eps` is
/// `hyper_connection.epsilon`. See the module docs: they are different values.
pub fn mixes(
    stream: &[f32],
    fn_: &[f32],
    base: &[f32],
    scale: &[f32],
    rms_eps: f32,
    sinkhorn_iters: usize,
    hc_eps: f32,
) -> Result<Mix> {
    let n = stream.len();
    if n == 0 || n % HC != 0 {
        return Err(LlamaError::Config(format!(
            "hc: stream is {n} wide, expected a non-zero multiple of {HC}"
        )));
    }
    if fn_.len() != MIX * n {
        return Err(LlamaError::Config(format!(
            "hc: hc_fn has {} entries, expected MIX({MIX}) * {n}",
            fn_.len()
        )));
    }
    if base.len() != MIX || scale.len() != 3 {
        return Err(LlamaError::Config(format!(
            "hc: base is {} (want {MIX}) and scale is {} (want 3)",
            base.len(),
            scale.len()
        )));
    }

    // The reference normalises the stream and then matmuls. A matmul is linear,
    // so projecting the raw stream and scaling the 24 results by the same `r` is
    // identical arithmetic with 24 multiplies instead of `HC * dim`.
    let mut proj = [0.0f32; MIX];
    for (j, pj) in proj.iter_mut().enumerate() {
        *pj = fn_[j * n..(j + 1) * n]
            .iter()
            .zip(stream)
            .map(|(w, v)| w * v)
            .sum::<f32>();
    }
    let sumsq: f32 = stream.iter().map(|v| v * v).sum();
    let r = 1.0 / (sumsq / n as f32 + rms_eps).sqrt();

    Ok(mixes_from_projection(&proj, r, base, scale, sinkhorn_iters, hc_eps))
}

/// The tail of [`mixes`] given the 24 raw projections and the RMS reciprocal
/// `r`. Split out because a device computes the projection and the norm, then
/// only these 24 numbers per token need the scalar path.
pub fn mixes_from_projection(
    proj: &[f32; MIX],
    r: f32,
    base: &[f32],
    scale: &[f32],
    sinkhorn_iters: usize,
    hc_eps: f32,
) -> Mix {
    let m: [f32; MIX] = std::array::from_fn(|j| proj[j] * r);

    let mut mix = Mix {
        pre: [0.0; HC],
        post: [0.0; HC],
        comb: [[0.0; HC]; HC],
    };
    for j in 0..HC {
        // pre: sigmoid then + eps, so a collapse weight is never exactly zero.
        mix.pre[j] = sigmoid(m[j] * scale[0] + base[j]) + hc_eps;
        // post: 2 * sigmoid, so a sublayer can amplify up to 2x.
        mix.post[j] = 2.0 * sigmoid(m[j + HC] * scale[1] + base[j + HC]);
    }

    // comb is laid out src-major: flat index 2*HC + src*HC + dst.
    for src in 0..HC {
        for dst in 0..HC {
            let j = 2 * HC + src * HC + dst;
            mix.comb[src][dst] = m[j] * scale[2] + base[j];
        }
    }
    sinkhorn(&mut mix.comb, sinkhorn_iters, hc_eps);
    mix
}

/// Make `comb` doubly stochastic, following `build_hc_sinkhorn`: a softmax over
/// `dst`, `+eps`, one column normalisation, then `iters - 1` rounds of
/// row-then-column normalisation.
///
/// The asymmetry is deliberate — the first pass has no row normalisation
/// because the softmax already made each row sum to 1.
fn sinkhorn(comb: &mut [[f32; HC]; HC], iters: usize, eps: f32) {
    for row in comb.iter_mut() {
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
        }
        let s: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v = *v / s + eps;
        }
    }
    normalize_cols(comb, eps);
    for _ in 1..iters {
        for row in comb.iter_mut() {
            let s: f32 = row.iter().sum::<f32>() + eps;
            for v in row.iter_mut() {
                *v /= s;
            }
        }
        normalize_cols(comb, eps);
    }
}

fn normalize_cols(comb: &mut [[f32; HC]; HC], eps: f32) {
    for dst in 0..HC {
        let s: f32 = (0..HC).map(|src| comb[src][dst]).sum::<f32>() + eps;
        for row in comb.iter_mut() {
            row[dst] /= s;
        }
    }
}

/// `build_hc_pre`'s collapse: `out[k] = sum_src pre[src] * stream[src][k]`.
/// `stream` is `[HC, dim]` row-major. f32 out — no bf16 rounding.
pub fn collapse(stream: &[f32], pre: &[f32; HC]) -> Vec<f32> {
    let mut out = vec![0.0f32; stream.len() / HC];
    collapse_into(stream, pre, &mut out);
    out
}

/// [`collapse`] into a caller-owned buffer, one sequential pass per stream.
///
/// The obvious form -- for each output `k`, sum over the `HC` streams -- reads
/// `stream` at stride `d`, so it touches `HC` cache lines per element and does it
/// `d` times. Accumulating one whole stream at a time instead makes both the read
/// and the write sequential, which is also the form a compiler will vectorise.
/// With `HC = 4` and `d = 4096` that is four 16 KB passes rather than 4096
/// four-way gathers, and it happens 90 times a token.
pub fn collapse_into(stream: &[f32], pre: &[f32; HC], out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(stream.len(), HC * d);
    let (first, rest) = stream.split_at(d);
    let p0 = pre[0];
    for (o, &s) in out.iter_mut().zip(first) {
        *o = p0 * s;
    }
    for src in 1..HC {
        let p = pre[src];
        let chunk = &rest[(src - 1) * d..src * d];
        for (o, &s) in out.iter_mut().zip(chunk) {
            *o += p * s;
        }
    }
}

/// `build_hc_post`: for each output copy `dst`,
/// `post[dst] * out[k] + sum_src comb[src][dst] * residual[src][k]`.
///
/// `out` is the sublayer's `[dim]` result, `residual` the `[HC, dim]` stream as
/// it entered the sublayer. f32 out — no bf16 rounding.
pub fn combine(out: &[f32], residual: &[f32], mix: &Mix) -> Vec<f32> {
    let mut y = vec![0.0f32; HC * out.len()];
    combine_into(out, residual, mix, &mut y);
    y
}

/// [`combine`] into a caller-owned buffer, sequentially.
///
/// Same change of order as [`collapse_into`], for the same reason and a bigger
/// win: the per-element form re-reads all of `residual` once per destination
/// stream, `HC * HC * d` strided loads in all. Written this way each destination
/// is seeded from `out` and then accumulated one source stream at a time, so
/// every read and every write walks forward.
///
/// Summation order per output element is unchanged -- still `post*out` first, then
/// sources 0..HC in order -- so this is bit-identical to the version it replaces.
pub fn combine_into(out: &[f32], residual: &[f32], mix: &Mix, y: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(y.len(), HC * d);
    debug_assert_eq!(residual.len(), HC * d);
    for dst in 0..HC {
        let row = &mut y[dst * d..(dst + 1) * d];
        let p = mix.post[dst];
        for (t, &o) in row.iter_mut().zip(out) {
            *t = p * o;
        }
        for src in 0..HC {
            let c = mix.comb[src][dst];
            let chunk = &residual[src * d..(src + 1) * d];
            for (t, &r) in row.iter_mut().zip(chunk) {
                *t += c * r;
            }
        }
    }
}

/// `dsv4_hc_mean`: the unweighted mean over streams, which is how glm5next
/// collapses the trunk before `output_norm`. DeepSeek-V4.1 uses a learned gated
/// head here instead; glm5next ships no `hc_head_*` tensors.
pub fn mean(stream: &[f32]) -> Vec<f32> {
    let d = stream.len() / HC;
    let inv = 1.0 / HC as f32;
    (0..d)
        .map(|k| (0..HC).map(|src| stream[src * d + k]).sum::<f32>() * inv)
        .collect()
}

/// Open the residual stream from token embeddings: `HC` **exact copies**. The
/// reference is a plain `ggml_repeat_4d` — no scaling, and no one-hot into
/// stream 0.
pub fn init(embd: &[f32]) -> Vec<f32> {
    let mut y = Vec::with_capacity(HC * embd.len());
    for _ in 0..HC {
        y.extend_from_slice(embd);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequential rewrites must be bit-identical, not merely close: they are
    /// on the residual path of every layer, so a rounding difference would move
    /// the logits.
    #[test]
    fn the_sequential_forms_are_bit_identical() {
        let d = 37usize;
        let stream: Vec<f32> = (0..HC * d)
            .map(|i| (((i * 41) % 83) as f32 - 41.0) / 41.0)
            .collect();
        let out: Vec<f32> = (0..d).map(|i| (((i * 29) % 61) as f32 - 30.0) / 30.0).collect();
        let pre: [f32; HC] = std::array::from_fn(|j| 0.3 + j as f32 * 0.21);
        let mut mix = Mix {
            pre,
            post: std::array::from_fn(|j| 0.7 + j as f32 * 0.13),
            comb: std::array::from_fn(|s| std::array::from_fn(|t| 0.11 * (s + 1) as f32 - 0.07 * t as f32)),
        };
        sinkhorn(&mut mix.comb, 20, 1e-6);

        let want_c = collapse(&stream, &pre);
        let mut got_c = vec![0.0f32; d];
        collapse_into(&stream, &pre, &mut got_c);
        assert_eq!(want_c, got_c);
        assert!(want_c.iter().any(|v| v.abs() > 1e-6));

        let want_m = combine(&out, &stream, &mix);
        let mut got_m = vec![0.0f32; HC * d];
        combine_into(&out, &stream, &mix, &mut got_m);
        assert_eq!(want_m, got_m);
        assert!(want_m.iter().any(|v| v.abs() > 1e-6));
    }

    /// glm5next's real values.
    const RMS_EPS: f32 = 1e-5;
    const HC_EPS: f32 = 1e-6;
    const ITERS: usize = 20;

    fn params(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let fn_: Vec<f32> = (0..MIX * n)
            .map(|i| ((i * 7919 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let base: Vec<f32> = (0..MIX).map(|i| i as f32 * 0.1 - 1.0).collect();
        let scale = vec![0.5f32, 0.7, 1.3];
        (fn_, base, scale)
    }

    #[test]
    fn sinkhorn_makes_comb_doubly_stochastic() {
        let n = HC * 8;
        let (fn_, base, scale) = params(n);
        let stream: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) / 10.0).collect();
        let m = mixes(&stream, &fn_, &base, &scale, RMS_EPS, ITERS, HC_EPS).expect("mixes");

        for j in 0..HC {
            let row: f32 = m.comb[j].iter().sum();
            let col: f32 = (0..HC).map(|i| m.comb[i][j]).sum();
            assert!(
                (row - 1.0).abs() < 1e-3,
                "row {j} sums to {row}, want 1"
            );
            assert!(
                (col - 1.0).abs() < 1e-3,
                "col {j} sums to {col}, want 1"
            );
            // pre is sigmoid + eps, post is 2 * sigmoid.
            assert!(m.pre[j] > 0.0 && m.pre[j] < 1.0 + HC_EPS);
            assert!(m.post[j] > 0.0 && m.post[j] < 2.0);
        }
    }

    /// One Sinkhorn iteration must still column-normalise (the reference runs
    /// `norm_cols()` before the loop, not inside it).
    #[test]
    fn a_single_sinkhorn_iteration_normalises_columns() {
        let n = HC * 4;
        let (fn_, base, scale) = params(n);
        let stream: Vec<f32> = (0..n).map(|i| (i as f32) / 7.0 - 1.0).collect();
        let m = mixes(&stream, &fn_, &base, &scale, RMS_EPS, 1, HC_EPS).expect("mixes");
        for dst in 0..HC {
            let col: f32 = (0..HC).map(|src| m.comb[src][dst]).sum();
            assert!((col - 1.0).abs() < 1e-3, "col {dst} sums to {col}");
        }
    }

    #[test]
    fn collapse_is_the_pre_weighted_sum() {
        // 4 streams of width 2, distinct values per stream.
        let stream = vec![
            1.0f32, 2.0, // src 0
            10.0, 20.0, // src 1
            100.0, 200.0, // src 2
            1000.0, 2000.0, // src 3
        ];
        let pre = [1.0f32, 0.5, 0.25, 0.125];
        let got = collapse(&stream, &pre);
        assert_eq!(got.len(), 2);
        assert!((got[0] - (1.0 + 5.0 + 25.0 + 125.0)).abs() < 1e-4);
        assert!((got[1] - (2.0 + 10.0 + 50.0 + 250.0)).abs() < 1e-4);
    }

    #[test]
    fn combine_adds_the_spread_output_to_the_mixed_residual() {
        let out = vec![1.0f32, -2.0];
        let residual = vec![
            1.0f32, 1.0, //
            2.0, 2.0, //
            4.0, 4.0, //
            8.0, 8.0,
        ];
        // An identity comb routes src i to dst i unchanged.
        let mut comb = [[0.0f32; HC]; HC];
        for i in 0..HC {
            comb[i][i] = 1.0;
        }
        let mix = Mix {
            pre: [1.0; HC],
            post: [1.0, 1.0, 1.0, 1.0],
            comb,
        };
        let y = combine(&out, &residual, &mix);
        assert_eq!(y.len(), HC * 2);
        // dst j gets out + residual[j].
        for (j, r) in [1.0f32, 2.0, 4.0, 8.0].iter().enumerate() {
            assert!((y[j * 2] - (1.0 + r)).abs() < 1e-5, "dst {j} lane 0");
            assert!((y[j * 2 + 1] - (-2.0 + r)).abs() < 1e-5, "dst {j} lane 1");
        }
    }

    #[test]
    fn mean_divides_by_the_stream_count() {
        let stream = vec![
            1.0f32, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0,
        ];
        let got = mean(&stream);
        assert_eq!(got, vec![(1.0 + 3.0 + 5.0 + 7.0) / 4.0, (2.0 + 4.0 + 6.0 + 8.0) / 4.0]);
    }

    #[test]
    fn init_makes_exact_copies() {
        let embd = vec![0.5f32, -1.5, 3.25];
        let s = init(&embd);
        assert_eq!(s.len(), HC * 3);
        for src in 0..HC {
            assert_eq!(&s[src * 3..(src + 1) * 3], &embd[..], "stream {src}");
        }
        // A round trip through mean() must return the embedding unchanged: the
        // streams are copies, so their mean is the original.
        assert_eq!(mean(&s), embd);
    }

    /// Pins divergence (1) from the module docs. `dsv41::hc` rounds every
    /// `pre`/`post` result to bf16; this module must not, or it injects error
    /// the reference does not have. bf16 keeps 8 mantissa bits, so a value
    /// perturbed in bit 16 survives f32 and dies in bf16.
    #[test]
    fn activations_stay_f32_and_are_not_bf16_rounded() {
        let delta = f32::from_bits(1.0f32.to_bits() + 1);
        let bf16_of = |x: f32| f32::from_bits(x.to_bits() & 0xFFFF_0000);
        assert_ne!(
            bf16_of(delta), delta,
            "the probe value must be one bf16 would round"
        );

        let mix = Mix {
            pre: [1.0; HC],
            post: [1.0; HC],
            comb: [[0.0; HC]; HC],
        };
        let y = combine(&[delta], &vec![0.0f32; HC], &mix);
        for (dst, v) in y.iter().enumerate() {
            assert_eq!(
                *v, delta,
                "dst {dst}: combine must preserve f32 precision, got {v}"
            );
        }

        let collapsed = collapse(&[delta, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(collapsed[0], delta, "collapse must preserve f32 precision");
    }

    #[test]
    fn mixes_rejects_mismatched_parameter_widths() {
        let n = HC * 4;
        let (fn_, base, scale) = params(n);
        let stream = vec![0.0f32; n];

        // stream not a multiple of HC
        assert!(mixes(&vec![0.0; 7], &fn_, &base, &scale, RMS_EPS, ITERS, HC_EPS).is_err());
        // fn_ wrong size
        assert!(mixes(&stream, &fn_[..n], &base, &scale, RMS_EPS, ITERS, HC_EPS).is_err());
        // base wrong size
        assert!(mixes(&stream, &fn_, &base[..MIX - 1], &scale, RMS_EPS, ITERS, HC_EPS).is_err());
        // scale wrong size
        assert!(mixes(&stream, &fn_, &base, &scale[..2], RMS_EPS, ITERS, HC_EPS).is_err());
    }

    /// The two epsilons are different values and must not be conflated. Using
    /// the tiny `hc_eps` where the RMSNorm eps belongs changes the mixes.
    #[test]
    fn the_rms_eps_is_not_the_hc_eps() {
        let n = HC * 4;
        let (fn_, base, scale) = params(n);
        // A near-zero stream makes the RMSNorm epsilon dominate.
        let stream: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-4).collect();

        let a = mixes(&stream, &fn_, &base, &scale, RMS_EPS, ITERS, HC_EPS).expect("rms eps");
        let b = mixes(&stream, &fn_, &base, &scale, 1e-20, ITERS, HC_EPS).expect("tiny eps");
        assert_ne!(
            a.pre, b.pre,
            "the RMSNorm epsilon must affect the mixes; 1e-20 is dsv41's test value, \
             not the model's 1e-5"
        );
    }
}
