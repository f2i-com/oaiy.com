//! Mixture-of-Experts FFN block. Per-token routing: a small `router` linear
//! produces `[seq, num_experts]` logits; for each token we pick the top-K
//! highest-scoring experts, softmax their logits to weights, run each
//! selected expert's SwiGLU FFN, and weighted-sum the outputs.
//!
//! Targets: Mixtral 8x7B (8 experts, top-2), Qwen3-30B-A3B (typically 128
//! experts, top-8), Qwen3.6-30B-A3B, Qwen3.6-VL-30B-A3B, Gemma 4 26B-A4B.
//!
//! Performance note: this first implementation is a host-driven loop over
//! (token, expert) pairs — correct but un-fused. Each expert dispatch goes
//! through the existing `linear` op (a tiled GEMM / coop GEMV). For decode
//! (seq=1, top-K typically 2 or 8), that's K expert FFNs per layer per token —
//! manageable. Prefill is the pain point: O(seq × top_k) dispatches per layer.
//! A future fused-MoE CUDA kernel (one launch handles routing + dispatch +
//! reduction) is the speedup path.

#[cfg(feature = "cuda")]
use std::sync::Arc;

use ggml_rs::{Backend, Tensor};

use crate::loader::Weight;

// VENDORED-LOCAL: PERF-01 — route-trace hook. `OAIY run --trace-out` records
// the (layer, expert) routing sequence per token for deterministic offline
// replay (docs/ROADMAP.md Phase 0). The streaming path's routing decision is
// computed inside `expert_stream.rs`, which does not expose the routed ids;
// the one place both the resident and streaming paths pass through with the
// router logits in hand is `moe_forward_with_logits`, so the trace re-derives
// the ids here with the same `top_k_softmax` — identical by construction.
// Recording costs one extra host pull of the router logits per MoE block and
// is compiled in but inert unless a sink is installed.
mod route_trace {
    use std::sync::Mutex;

    type Sink = Box<dyn FnMut(&[Vec<u32>]) + Send>;

    static SINK: Mutex<Option<Sink>> = Mutex::new(None);

    /// Install (or clear, with `None`) the global route sink. The sink
    /// receives, per `moe_forward` call, one entry per sequence position:
    /// the top-k expert ids routed to, in routing order.
    pub fn set(sink: Option<Sink>) {
        let mut g = SINK.lock().unwrap_or_else(|e| e.into_inner());
        *g = sink;
    }

    /// Record one MoE block's routing: `rows` are the `[seq, n_experts]`
    /// router logits on the host, `top_k` the block's active expert count.
    pub fn record(rows: &[f32], seq: usize, n_experts: usize, top_k: usize) {
        let mut g = SINK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(sink) = g.as_mut() else { return };
        let mut per_token = Vec::with_capacity(seq);
        for t in 0..seq {
            let row = &rows[t * n_experts..(t + 1) * n_experts];
            let (indices, _weights) = super::top_k_softmax(row, top_k);
            per_token.push(indices.into_iter().map(|i| i as u32).collect::<Vec<u32>>());
        }
        sink(&per_token);
    }

    /// True when a sink is installed — lets the caller skip the host pull of
    /// the router logits when nobody is listening.
    pub fn active() -> bool {
        let g = SINK.lock().unwrap_or_else(|e| e.into_inner());
        g.is_some()
    }
}

/// VENDORED-LOCAL: PERF-01 — see the `route_trace` module above.
/// Install (or clear) the global route-trace sink.
pub fn set_route_trace_sink(sink: Option<Box<dyn FnMut(&[Vec<u32>]) + Send>>) {
    route_trace::set(sink);
}

/// Per-block MoE weights.
#[derive(Debug)]
pub struct MoeFfn {
    /// Router projection: `[num_experts, hidden_dim]`. Small (e.g. 8×4096
    /// = 32K floats for Mixtral); typically Dense / not quantized.
    pub router: Weight,
    /// Per-expert SwiGLU gate+up pair: when `Fused`, one matmul produces
    /// `[seq, 2*ff_dim]` and `silu_mul_split` halves it. When `Split`, runs
    /// two matmuls (the original path). Auto-fused at load time when both
    /// halves share dtype + shape.
    pub gate_up_experts: Vec<crate::loader::FfnPair>,
    /// Per-expert SwiGLU down weight: `[hidden_dim, ff_dim]` × num_experts.
    pub down_experts: Vec<Weight>,
    /// How many experts to activate per token. Mixtral=2, Qwen3-30B-A3B=8.
    pub top_k: usize,
    /// VENDORED-LOCAL: streaming expert backend (see `expert_stream.rs`).
    /// When `Some`, `gate_up_experts` / `down_experts` are empty and routed
    /// experts are read per dispatch from the .gguf file through a bounded
    /// cache. Resident models leave this `None`.
    pub stream: Option<crate::expert_stream::LayerStream>,
    /// VENDORED-LOCAL: MOE-02 — lazily-built CUDA grouped-decode plan
    /// (static per-layer pointer tables; see `moe_cuda.rs`). `OnceLock`
    /// because eligibility is decided on the first decode forward, when the
    /// backend and the weights' final placement are known; `None` inside
    /// means "checked, not eligible" (fall back to the reference loop).
    #[cfg(feature = "cuda")]
    pub(crate) gpu_plan: std::sync::OnceLock<Option<Arc<crate::moe_cuda::GpuLayerPlan>>>,
}

impl MoeFfn {
    pub fn num_experts(&self) -> usize {
        // VENDORED-LOCAL: streaming variant carries no resident expert Vecs.
        match &self.stream {
            Some(s) => s.n_experts(),
            None => self.gate_up_experts.len(),
        }
    }

    pub fn move_to_device(self, backend: &dyn Backend, safety_margin_bytes: usize) -> Self {
        Self {
            router:           self.router.try_to_device(backend, safety_margin_bytes),
            gate_up_experts:  self.gate_up_experts.into_iter()
                                  .map(|p| p.try_to_device(backend, safety_margin_bytes)).collect(),
            down_experts:     self.down_experts.into_iter()
                                  .map(|w| w.try_to_device(backend, safety_margin_bytes)).collect(),
            top_k:            self.top_k,
            stream:           self.stream,
            // Weights may have changed placement — rebuild the grouped plan
            // against their new storage on first use.
            #[cfg(feature = "cuda")]
            gpu_plan:         std::sync::OnceLock::new(),
        }
    }
}

/// Per-expert scaling configuration for activation function and output. Gemma 4 MoE:
///   * `act = LLM_FFN_GELU` (vs SwiGLU's silu)
///   * `down_exps_scale: Some(per-expert F32 scalar)` applied to each expert's down output
///   * Mixtral / Qwen3-MoE: `act = SwiGLU` (silu+mul), no per-expert output scale.
#[derive(Debug, Clone, Default)]
pub struct MoeOptions<'a> {
    /// If `true`, use GELU(gate) * up instead of silu(gate) * up. Gemma 4 MoE
    /// uses GELU. Mixtral / Qwen3-MoE use silu (default).
    pub use_gelu: bool,
    /// Per-expert scalar applied to each expert's down output before the
    /// softmax-weight scaling. Gemma 4 MoE has this; Mixtral / Qwen3-MoE don't.
    /// Borrowed slice — the per-block scales live on the model and don't change
    /// across forward calls, so callers pass `Some(&block.down_exps_s_host)`
    /// instead of cloning the Vec each token.
    pub down_exps_scale_host: Option<&'a [f32]>,
}

/// Run an MoE FFN block on `x` (shape `[seq, hidden_dim]`). Returns the FFN
/// output of the same shape, ready to be added to the residual stream.
///
/// On CUDA at decode (seq=1) the whole block runs on-device: top-k routing
/// happens in a kernel and the layer's experts run as 3 grouped launches
/// (MOE-01/MOE-02, see `moe_cuda.rs`). Otherwise only the compact
/// `[seq, top_k]` routing mailbox (ids + weights) crosses the host boundary
/// — the backend computes it on-device where supported — and per-(token,
/// expert) dispatch runs on the backend (slice → expert FFN →
/// scaled-accumulate into the output buffer).
pub fn moe_forward(backend: &dyn Backend, x: &Tensor, moe: &MoeFfn) -> Tensor {
    let router_logits = moe.router.linear(backend, x);
    moe_forward_with_logits(backend, x, moe, &router_logits, &MoeOptions::default())
}

/// Variant of [`moe_forward`] that also takes a custom `MoeOptions` (for GELU-
/// vs-silu activation choice and per-expert output scales). For SwiGLU defaults
/// just use [`moe_forward`].
pub fn moe_forward_with_opts(
    backend: &dyn Backend,
    x:       &Tensor,
    moe:     &MoeFfn,
    opts:    &MoeOptions<'_>,
) -> Tensor {
    let router_logits = moe.router.linear(backend, x);
    moe_forward_with_logits(backend, x, moe, &router_logits, opts)
}

/// Same as [`moe_forward`] but the router logits are pre-computed (so the
/// caller can apply a custom pipeline like Gemma 4 MoE's
/// `rmsnorm_no_scale → scale 1/sqrt(n_embd) → ffn_gate_inp_s → matmul`). Also
/// supports the per-expert scaling/activation knobs in [`MoeOptions`].
pub fn moe_forward_with_logits(
    backend:       &dyn Backend,
    x:             &Tensor,
    moe:           &MoeFfn,
    router_logits: &Tensor,
    opts:          &MoeOptions<'_>,
) -> Tensor {
    // VENDORED-LOCAL: PERF-01 route trace — record the routing decision for
    // both the streaming and resident paths before dispatch (see route_trace
    // above). Only pays the extra host pull when a sink is installed.
    if route_trace::active() {
        let h = backend.to_host(router_logits.clone());
        route_trace::record(h.data(), x.dim(0), moe.num_experts(), moe.top_k);
    }
    // VENDORED-LOCAL: streaming dispatch — experts come from the cache/store
    // per (token, expert) instead of the resident Vecs. Same routing math,
    // same backend ops, only the weight storage differs.
    if let Some(stream) = &moe.stream {
        return stream.forward_with_logits(backend, x, router_logits, moe.top_k, opts);
    }
    // VENDORED-LOCAL: MOE-02 — grouped CUDA decode: routing stays on device
    // and the layer's k expert FFNs run as 3 kernel launches (see
    // moe_cuda.rs). `None` → reference loop below (non-CUDA backend,
    // prefill, or an ineligible layer).
    #[cfg(feature = "cuda")]
    if let Some(out) = crate::moe_cuda::try_grouped_forward(backend, x, moe, router_logits, opts)
    {
        return out;
    }
    let seq = x.dim(0);
    let hidden = x.dim(1);
    let top_k = moe.top_k;

    // VENDORED-LOCAL: MOE-01 — routing is a backend op now: on CUDA the
    // top-k+softmax runs on device and only the compact (ids, weights)
    // mailbox (`seq*top_k*8` bytes) crosses D2H, instead of the full
    // `[seq, n_experts]` logits. The CPU backend runs the same reference
    // math host-side (identical selection to the old inline top_k_softmax).
    let (ids_flat, w_flat) = backend.moe_route_topk(router_logits, top_k);

    // Device-side zero alloc — avoids the host `vec![0.0; seq*hidden]` +
    // `to_device` round trip per layer. CUDA backend uses cudarc's
    // `alloc_zeros` (a device memset, no h2d transfer).
    let mut output = backend.alloc_zeros(vec![seq, hidden]);

    for t in 0..seq {
        let indices = &ids_flat[t * top_k..(t + 1) * top_k];
        let weights = &w_flat[t * top_k..(t + 1) * top_k];

        let xt = backend.slice_axis0_range(x, t, 1);

        for (idx, w) in indices.iter().zip(weights.iter()) {
            let idx = *idx as usize;
            // Per-expert gated MLP: one matmul → silu_mul_split (fused path) or
            // two matmuls + silu_mul (split path). Activation choice (silu vs
            // gelu) comes from MoeOptions.
            let activated = if opts.use_gelu {
                moe.gate_up_experts[idx].geglu(backend, &xt)
            } else {
                moe.gate_up_experts[idx].swiglu(backend, &xt)
            };
            let expert_out = moe.down_experts[idx].linear(backend, &activated);
            // Fold per-expert pre-weighting scale (Gemma 4 MoE) into the routing
            // weight, then do `output[t] += expert_out * combined` in a single
            // kernel launch (saves the `mul_scalar_inplace` round trip and a
            // global write of `expert_out`).
            let combined = match opts.down_exps_scale_host {
                Some(scales) => *w * scales[idx],
                None         => *w,
            };
            backend.add_to_axis0_range_scaled(&mut output, t, 1, &expert_out, combined);
        }
    }

    output
}

/// Pick the top-K largest values from `logits`, softmax-normalize them.
/// Returns `(indices, normalized_weights)` both of length K.
// VENDORED-LOCAL: pub(crate) so expert_stream's streaming forward reuses the
// exact same routing math (equivalence with the resident path by construction).
pub(crate) fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(logits.len());
    // Index + value pairs.
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    // Partial sort to get the top-K (largest values first).
    pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.partial_cmp(&a.1).unwrap());
    pairs.truncate(k);
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Softmax over the top-K logits.
    let max_l = pairs[0].1;
    let exps: Vec<f32> = pairs.iter().map(|(_, l)| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    let indices: Vec<usize> = pairs.iter().map(|(i, _)| *i).collect();
    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::Weight;
    use ggml_rs::default_backend;

    #[test]
    fn top_k_softmax_picks_correct_indices() {
        // logits [0.1, 5.0, 0.2, 3.0, 0.05] — top-2 should be (1, 3) with
        // weight ~0.881 / 0.119.
        let (idx, w) = top_k_softmax(&[0.1, 5.0, 0.2, 3.0, 0.05], 2);
        assert_eq!(idx, vec![1, 3]);
        assert!((w[0] - 0.881).abs() < 0.005, "expert 1 weight: got {}, want ~0.881", w[0]);
        assert!((w[1] - 0.119).abs() < 0.005, "expert 3 weight: got {}, want ~0.119", w[1]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to ~1.0; got {sum}");
    }

    #[test]
    fn moe_forward_with_synthetic_weights_runs_end_to_end() {
        // Construct a minimal MoE: 1 token, hidden=8, ff=16, 4 experts, top-2.
        // We hand-pick router weights so expert 2 dominates (weight ~1.0) for
        // a known input, and verify the output matches expert 2's FFN on that
        // input scaled by ~1.0.
        let backend = default_backend();
        let hidden = 8usize;
        let ff = 16usize;
        let n_experts = 4usize;
        let top_k = 2usize;

        // Input: a single row of 1.0s.
        let x = Tensor::from_vec(vec![1.0f32; hidden], vec![1, hidden]);

        // Router that maximally favours expert 2: row 2 = all 10.0, others = 0.0.
        // logits[2] = 10*8 = 80; others = 0; softmax is ~(0, 0, ~1, 0, 0).
        let mut router_data = vec![0.0f32; n_experts * hidden];
        for j in 0..hidden { router_data[2 * hidden + j] = 10.0; }
        let router = Weight::Dense(Tensor::from_vec(router_data, vec![n_experts, hidden]));

        // Per-expert FFN weights — different per expert so we can check which one ran.
        // Use simple deterministic patterns. Build FfnPair via from_halves so the
        // test exercises the auto-fusion path (Dense halves of matching shape fuse).
        use crate::loader::FfnPair;
        let mut gate_up_experts = Vec::with_capacity(n_experts);
        let mut down_experts = Vec::with_capacity(n_experts);
        let mut ref_gates = Vec::with_capacity(n_experts);
        let mut ref_ups   = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let scale = 0.1f32 * (e as f32 + 1.0);
            let gate_data: Vec<f32> = (0..ff*hidden).map(|i| scale * ((i % 7) as f32 - 3.0)).collect();
            let up_data:   Vec<f32> = (0..ff*hidden).map(|i| scale * ((i % 5) as f32 - 2.0)).collect();
            let down_data: Vec<f32> = (0..hidden*ff).map(|i| scale * ((i % 3) as f32 - 1.0)).collect();
            let g_w = Weight::Dense(Tensor::from_vec(gate_data.clone(), vec![ff, hidden]));
            let u_w = Weight::Dense(Tensor::from_vec(up_data.clone(),   vec![ff, hidden]));
            ref_gates.push(Weight::Dense(Tensor::from_vec(gate_data, vec![ff, hidden])));
            ref_ups  .push(Weight::Dense(Tensor::from_vec(up_data,   vec![ff, hidden])));
            gate_up_experts.push(FfnPair::from_halves(g_w, u_w));
            down_experts.push(Weight::Dense(Tensor::from_vec(down_data, vec![hidden, ff])));
        }

        let moe = MoeFfn { router, gate_up_experts, down_experts, top_k, stream: None,
            #[cfg(feature = "cuda")]
            gpu_plan: std::sync::OnceLock::new(),
        };

        // Reference: run expert 2's FFN by hand using the pre-fusion gate/up
        // weights and confirm the MoE output ≈ that (router favours expert 2).
        let xt = Tensor::from_vec(vec![1.0f32; hidden], vec![1, hidden]);
        let g_ref = ref_gates[2].linear(&*backend, &xt);
        let u_ref = ref_ups[2].linear(&*backend, &xt);
        let act_ref = backend.silu_mul(&g_ref, &u_ref);
        let ref_out = moe.down_experts[2].linear(&*backend, &act_ref);
        let ref_data = ref_out.to_host();

        // MoE forward.
        let out = moe_forward(&*backend, &x, &moe);
        let out_h = out.to_host();

        // Check: MoE output ≈ expert 2 output (router gives expert 2 a weight ~1.0).
        for j in 0..hidden {
            let diff = (out_h.data()[j] - ref_data.data()[j]).abs();
            assert!(diff < 1e-2,
                "MoE output[{}] = {} differs too much from expert-2-only output {} (diff {})",
                j, out_h.data()[j], ref_data.data()[j], diff);
        }
    }

    // VENDORED-LOCAL: MOE-01/MOE-02 — end-to-end check that the grouped CUDA
    // path through `moe_forward` (plan build, on-device routing, grouped
    // kernels) matches the CPU reference loop on the same weights.
    #[cfg(feature = "cuda")]
    #[test]
    fn moe_forward_cuda_grouped_matches_cpu() {
        use ggml_quants::GgmlType;
        use ggml_rs::{Backend, CpuBackend, QuantizedTensor};

        let cuda = match ggml_rs_cuda::CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[skipping] CUDA init failed: {e}");
                return;
            }
        };
        let cpu = CpuBackend::new();

        let hidden = 256usize;
        let ff = 768usize;
        let n_experts = 6usize;
        let top_k = 3usize;

        // Deterministic pseudo-random bytes / floats (no external crates).
        struct Sm(u64);
        impl Sm {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^ (z >> 31)
            }
            fn f32(&mut self, scale: f32) -> f32 {
                (((self.next() >> 40) & 0xFF) as f32 / 255.0 - 0.5) * scale
            }
        }
        let mut rng = Sm(0x1234_5678_9abc_def0);

        // Q8_0 experts: [2ff, hidden] fused gate‖up + [hidden, ff] down per
        // expert, random packed bytes with small scales (dequant is
        // deterministic from the bytes — that's all the kernels read).
        let small_d = half::f16::from_f32(0.01).to_bits().to_le_bytes();
        let quant_bytes = |rows: usize, k: usize, rng: &mut Sm| {
            let bpr = k / 32;
            let mut bytes: Vec<u8> =
                (0..rows * bpr * 34).map(|_| rng.next() as u8).collect();
            for row in 0..rows {
                for blk in 0..bpr {
                    let off = (row * bpr + blk) * 34;
                    bytes[off..off + 2].copy_from_slice(&small_d);
                }
            }
            bytes
        };

        // Generate the weights ONCE so both backends see identical tensors.
        let router_data: Vec<f32> = (0..n_experts * hidden).map(|_| rng.f32(0.5)).collect();
        let gu_bytes: Vec<Vec<u8>> = (0..n_experts)
            .map(|_| quant_bytes(2 * ff, hidden, &mut rng))
            .collect();
        let dn_bytes: Vec<Vec<u8>> = (0..n_experts)
            .map(|_| quant_bytes(hidden, ff, &mut rng))
            .collect();
        let x_data: Vec<f32> = (0..hidden).map(|_| rng.f32(1.0)).collect();

        let build_moe = |backend: &dyn Backend| {
            let router = Weight::Dense(backend.to_device(Tensor::from_vec(
                router_data.clone(),
                vec![n_experts, hidden],
            )));
            let mut gate_up_experts = Vec::with_capacity(n_experts);
            let mut down_experts = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let gu = QuantizedTensor::from_bytes_cpu(
                    gu_bytes[e].clone(),
                    vec![2 * ff, hidden],
                    GgmlType::Q8_0,
                );
                let dn = QuantizedTensor::from_bytes_cpu(
                    dn_bytes[e].clone(),
                    vec![hidden, ff],
                    GgmlType::Q8_0,
                );
                gate_up_experts.push(crate::loader::FfnPair::Fused(Weight::Quant(
                    backend.to_device_quant(gu),
                )));
                down_experts.push(Weight::Quant(backend.to_device_quant(dn)));
            }
            MoeFfn {
                router,
                gate_up_experts,
                down_experts,
                top_k,
                stream: None,
                gpu_plan: std::sync::OnceLock::new(),
            }
        };

        let moe_gpu = build_moe(&cuda);
        let moe_cpu = build_moe(&cpu);
        let x = Tensor::from_vec(x_data, vec![1, hidden]);

        let out_gpu = moe_forward(&cuda, &x, &moe_gpu).to_host();
        let out_cpu = moe_forward(&cpu, &x, &moe_cpu).to_host();

        // The grouped plan must actually have engaged (else this test is
        // vacuous — it would just be the reference loop on both sides).
        assert!(
            moe_gpu.gpu_plan.get().and_then(|p| p.as_ref()).is_some(),
            "grouped CUDA plan was not built — grouped path did not engage"
        );
        // Second call reuses the cached plan.
        let out_gpu2 = moe_forward(&cuda, &x, &moe_gpu).to_host();
        assert_eq!(out_gpu.data(), out_gpu2.data(), "grouped path not deterministic");

        let max_diff = out_gpu
            .data()
            .iter()
            .zip(out_cpu.data().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let any_bad = out_gpu
            .data()
            .iter()
            .zip(out_cpu.data().iter())
            .any(|(a, b)| {
                let d = (a - b).abs();
                d >= 1e-2 && d >= 1e-2 * (a.abs() + b.abs())
            });
        assert!(!any_bad, "grouped CUDA vs CPU MoE output differs; max_diff={max_diff}");
    }
}
