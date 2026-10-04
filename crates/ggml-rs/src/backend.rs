//! Backend trait.
//!
//! All inference primitives go through this trait so a CPU implementation and
//! a future GPU implementation can be swapped at runtime. For today only
//! `CpuBackend` is wired up.

use std::fmt::Debug;

use crate::quantized::QuantizedTensor;
use crate::tensor::Tensor;

/// RoPE flavor. Llama / Mistral / Qwen / Gemma all use NeoX-style.
/// "Normal" (interleaved pairs) is included for completeness — original GPT-J
/// uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Pairs are `(x[2k], x[2k+1])`.
    Normal,
    /// Pairs are `(x[k], x[k + head_dim/2])`. The HF / Llama / Qwen / Gemma default.
    NeoX,
}
/// The delta-net step on the host (`Backend::delta_net_step`'s and `delta_net_step_sigmoid`'s default), a token at a
/// time as the CUDA kernels go: depthwise conv1d, the per-head autoregressive update, the norm-gated output (silu(z),
/// or sigmoid(z) for Qwen3.8-Flash-Next). The output `[seq, num_v_heads * head_v_dim]`, then the conv and recurrent
/// state after the last token, all on the host.
#[allow(clippy::too_many_arguments)]
fn host_delta_net(
    mixed_qkv:   &Tensor,
    z_in:        &Tensor,
    beta_alpha:  &Tensor,
    conv_weight: &Tensor,
    ssm_a:       &Tensor,
    dt_bias:     &Tensor,
    ssm_norm:    &Tensor,
    conv_state:  &Tensor,
    state:       &Tensor,
    seq:         usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim:  usize,
    head_k_dim:  usize,
    scale_q:     f32,
    eps:         f32,
    sigmoid_gate: bool,
) -> (Tensor, Tensor, Tensor) {
    assert!(head_k_dim <= 128 && head_v_dim <= 128, "delta-net heads of at most 128 on the host");
    // Host fallback: per-token loop matching the CUDA kernels.
    let conv_dim    = mixed_qkv.numel() / seq;
    let conv_kernel = conv_weight.dim(1);
    let mqkv_h = mixed_qkv.to_host();   let mqkv = mqkv_h.data();
    let z_full_h = z_in.to_host();      let z_full = z_full_h.data();
    let ba_full_h = beta_alpha.to_host(); let ba_full = ba_full_h.data();
    let cw_h   = conv_weight.to_host(); let cw  = cw_h.data();
    let sa_h   = ssm_a.to_host();       let sa  = sa_h.data();
    let dt_h_  = dt_bias.to_host();     let dt  = dt_h_.data();
    let nm_h   = ssm_norm.to_host();    let nm  = nm_h.data();

    let mut conv_h  = conv_state.to_host();
    let cs          = conv_h.data_mut();
    let mut state_h = state.to_host();
    let st          = state_h.data_mut();
    let mut output  = vec![0.0f32; seq * num_v_heads * head_v_dim];
    let mut conv_out = vec![0.0f32; conv_dim];

    let q_base = 0;
    let k_base = num_k_heads * head_k_dim;
    let v_base = 2 * num_k_heads * head_k_dim;

    for t in 0..seq {
        // ---- conv1d for this token ------------------------------------
        let mqkv_t = &mqkv[t * conv_dim..(t + 1) * conv_dim];
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for k in 0..(conv_kernel - 1) {
                acc += cs[k * conv_dim + c] * cw[c * conv_kernel + k];
            }
            acc += mqkv_t[c] * cw[c * conv_kernel + (conv_kernel - 1)];
            conv_out[c] = acc / (1.0 + (-acc).exp());
            for k in 0..(conv_kernel - 2) {
                cs[k * conv_dim + c] = cs[(k + 1) * conv_dim + c];
            }
            cs[(conv_kernel - 2) * conv_dim + c] = mqkv_t[c];
        }
        // ---- per-head delta-net step (TILED V order) ------------------
        let z_t  = &z_full[t * num_v_heads * head_v_dim..(t + 1) * num_v_heads * head_v_dim];
        let ba_t = &ba_full[t * 2 * num_v_heads..(t + 1) * 2 * num_v_heads];
        let out_t = &mut output[t * num_v_heads * head_v_dim..(t + 1) * num_v_heads * head_v_dim];
        let conv_out = &conv_out;
        // The heads are independent within a token: each takes its own state and output (rayon), where one thread
        // did all 48 of a Flash-Next layer.
        use rayon::prelude::*;
        st.par_chunks_mut(head_v_dim * head_v_dim).zip(out_t.par_chunks_mut(head_v_dim)).enumerate().for_each(|(h_v, (st_h, out_h))| {
            let h_k = h_v % num_k_heads;
            let q_h = &conv_out[q_base + h_k * head_k_dim..q_base + (h_k + 1) * head_k_dim];
            let k_h = &conv_out[k_base + h_k * head_k_dim..k_base + (h_k + 1) * head_k_dim];
            let v_h = &conv_out[v_base + h_v * head_v_dim..v_base + (h_v + 1) * head_v_dim];
            let z_h = &z_t[h_v * head_v_dim..(h_v + 1) * head_v_dim];

            let inv_q = 1.0 / (q_h.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            let inv_k = 1.0 / (k_h.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            let mut q_n = [0.0f32; 128]; let mut k_n = [0.0f32; 128];
            for i in 0..head_k_dim { q_n[i] = q_h[i] * inv_q * scale_q; k_n[i] = k_h[i] * inv_k; }

            let bh = 1.0 / (1.0 + (-ba_t[h_v]).exp());
            let alph_b = ba_t[num_v_heads + h_v] + dt[h_v];
            let alph_sp = if alph_b > 20.0 { alph_b } else { alph_b.exp().ln_1p() };
            let g_t = (alph_sp * sa[h_v]).exp();

            for s in st_h.iter_mut() { *s *= g_t; }

            let mut kv_mem = [0.0f32; 128];
            for i in 0..head_v_dim {
                let row = &st_h[i * head_v_dim..(i + 1) * head_v_dim];
                let mut acc = 0.0f32;
                for j in 0..head_k_dim { acc += row[j] * k_n[j]; }
                kv_mem[i] = acc;
            }
            let mut delta = [0.0f32; 128];
            for i in 0..head_v_dim { delta[i] = (v_h[i] - kv_mem[i]) * bh; }
            for i in 0..head_v_dim {
                let row = &mut st_h[i * head_v_dim..(i + 1) * head_v_dim];
                let di = delta[i];
                for j in 0..head_k_dim { row[j] += di * k_n[j]; }
            }
            let mut core = [0.0f32; 128];
            for i in 0..head_v_dim {
                let row = &st_h[i * head_v_dim..(i + 1) * head_v_dim];
                let mut acc = 0.0f32;
                for j in 0..head_k_dim { acc += row[j] * q_n[j]; }
                core[i] = acc;
            }
            let mean_sq = core[..head_v_dim].iter().map(|v| v * v).sum::<f32>() / head_v_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..head_v_dim {
                let normed = core[i] * inv_rms * nm[i];
                let gate = if sigmoid_gate { 1.0 / (1.0 + (-z_h[i]).exp()) } else { z_h[i] / (1.0 + (-z_h[i]).exp()) };
                out_h[i] = normed * gate;
            }
        });
    }
    (Tensor::from_vec(output, vec![seq, num_v_heads * head_v_dim]), conv_h, state_h)
}


/// Operations needed by a transformer forward pass.
pub trait Backend: Send + Sync + Debug + 'static {
    fn name(&self) -> &str;
    /// Keep unfused mapped weights available for host/SSD streaming.
    fn streams_weights(&self) -> bool { false }

    // VENDORED-LOCAL: MOE-01 — downcast support so model code can reach
    // backend-specific fast paths (the CUDA grouped-MoE kernels) without
    // widening this trait with device concepts. Required (not defaulted):
    // the `&Self -> &dyn Any` coercion needs a concrete Sized implementor.
    fn as_any(&self) -> &dyn std::any::Any;

    // VENDORED-LOCAL: PERF-01 — explicit sync point so benchmarks can stop
    // the clock at a phase boundary knowing all queued work has completed.
    /// Block until all work queued on this backend has completed. The CPU
    /// backend is fully synchronous already, so the default is a no-op;
    /// async (GPU) backends override it.
    fn synchronize(&self) {}

    /// Returns `(free_bytes, total_bytes)` of device memory for backends that
    /// have a separate device pool (CUDA, etc.). CPU backend returns `None`.
    /// Loaders use this to make placement decisions: weights that won't fit
    /// in remaining VRAM stay host-resident and dispatch to a CPU op.
    fn vram_status(&self) -> Option<(usize, usize)> { None }

    /// Move a tensor onto this backend's preferred storage. CPU backend is
    /// no-op; GPU backends upload host data to device. Idempotent: calling on
    /// a tensor already on this backend is cheap.
    fn to_device(&self, t: Tensor) -> Tensor { t }

    /// Try to place a tensor on the device, but keep it on host if it would
    /// blow the remaining VRAM budget plus a safety margin. Returns the tensor
    /// either way. Idempotent for tensors already on the right side. Default
    /// impl just calls `to_device` (CPU backend ignores the budget).
    fn try_to_device(&self, t: Tensor, safety_margin_bytes: usize) -> Tensor {
        match self.vram_status() {
            Some((free, _)) => {
                let bytes = t.numel() * std::mem::size_of::<f32>();
                if bytes + safety_margin_bytes <= free { self.to_device(t) }
                else { t }
            }
            None => self.to_device(t),
        }
    }

    /// Same as `try_to_device` for packed-quant weights — they're typically
    /// the largest tensors per layer, so the placement decision matters most
    /// here. Returns the tensor either way.
    fn try_to_device_quant(&self, w: QuantizedTensor, safety_margin_bytes: usize) -> QuantizedTensor {
        match self.vram_status() {
            Some((free, _)) => {
                let bytes = w.bytes().len();
                if bytes + safety_margin_bytes <= free { self.to_device_quant(w) }
                else { w }
            }
            None => self.to_device_quant(w),
        }
    }

    /// Materialize a host copy. Default: rely on `Tensor::to_host()`.
    fn to_host(&self, t: Tensor) -> Tensor {
        if t.is_device() { t.to_host() } else { t }
    }

    /// Move a packed-quant tensor onto this backend's preferred storage.
    /// Default: identity (the GGUF bytes stay on the host). GPU backends
    /// override this to upload bytes to device memory.
    fn to_device_quant(&self, w: QuantizedTensor) -> QuantizedTensor { w }

    /// Look up `tokens` rows from a `[vocab, embedding_dim]` table.
    /// Returns a `[len(tokens), embedding_dim]` tensor on this backend's storage.
    /// Default impl works on CPU; GPU backends override for an on-device gather.
    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        let _ = (table, tokens, embedding_dim);
        unimplemented!("embed_lookup not provided by this backend")
    }

    /// Materialize multiple device tensors into one contiguous host `Vec<f32>`,
    /// concatenated in the order given. The output length is the sum of each
    /// tensor's `numel()`. Used by Qwen3.5's SSM forward to fuse the three
    /// projection-output d2h transfers into one driver-side roundtrip — the
    /// per-call cudaMemcpy overhead dominates over actual transfer time at
    /// decode (seq=1) sizes.
    ///
    /// Default impl is a per-tensor `to_host` + `extend_from_slice`. GPU
    /// backends should override with N async memcpy_dtoh calls into one
    /// pre-allocated Vec to amortize driver overhead.
    fn concat_to_host_flat(&self, tensors: &[&Tensor]) -> Vec<f32> {
        let total: usize = tensors.iter().map(|t| t.numel()).sum();
        let mut out = Vec::with_capacity(total);
        for t in tensors {
            let h = t.to_host();
            out.extend_from_slice(h.data());
        }
        out
    }

    /// Slice the last row of a `[seq, last]` tensor into a `[last]` tensor on
    /// host (always CPU). Useful for sampling, which lives on CPU.
    fn last_row_to_host(&self, t: &Tensor) -> Tensor {
        let last = t.dim(t.rank() - 1);
        let host = t.to_host();
        let n = host.numel();
        let off = n - last;
        Tensor::from_vec(host.data()[off..].to_vec(), vec![last])
    }

    // VENDORED-LOCAL: MOE-01 — MoE routing as a backend op. Given
    /// `router_logits` (`[seq, n_experts]`), pick the top-`top_k` experts per
    /// token and softmax-normalize their logits into routing weights.
    /// Returns flat `(ids, weights)`, each `seq * top_k` long, in
    /// descending-logit (routing) order per token.
    ///
    /// The default pulls the logits to host and runs the reference math
    /// ([`moe_route_topk_host`]). The CUDA override keeps the selection on
    /// device and copies back only the compact (ids, weights) mailbox —
    /// `seq * top_k * 8` bytes instead of the full `[seq, n_experts]` logits.
    fn moe_route_topk(&self, router_logits: &Tensor, top_k: usize) -> (Vec<u32>, Vec<f32>) {
        let n_experts = router_logits.dim(router_logits.rank() - 1);
        let h = self.to_host(router_logits.clone());
        moe_route_topk_host(h.data(), top_k, n_experts)
    }

    /// `y[..., o] = sum_i w[o, i] * x[..., i]`
    /// `x` shape: `[..., in]`. `w` shape: `[out, in]`. Result: `[..., out]`.
    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor;

    /// VENDORED-LOCAL: add an unquantized low-rank adapter to a projection.
    /// Backends may fuse the update without materializing a full-size delta.
    fn add_lora(&self, y: &mut Tensor, x: &Tensor, a: &Tensor, b: &Tensor) {
        let low = self.linear(x, a);
        let delta = self.linear(&low, b);
        self.add_inplace(y, &delta);
    }

    /// Same as `linear` but `w` is in packed quantized form. The default
    /// implementation dequantizes `w` to F32 and calls `linear`. GPU backends
    /// should override with a kernel that reads the packed bytes directly,
    /// dequantizes per block in registers/shared memory, and accumulates.
    /// This is the load-bearing op for fitting big models in memory: with
    /// the override, weights stay packed (4 bits / element for Q4_K) instead
    /// of being inflated 8× to F32.
    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        // Default: materialize as F32 then call linear. Slow, but correct.
        let bytes_host = if w.is_cpu() {
            w.bytes().to_vec()
        } else {
            w.to_host().bytes().to_vec()
        };
        let mut dequant = vec![0.0f32; w.numel()];
        ggml_quants::dequantize(w.dtype(), &bytes_host, &mut dequant)
            .expect("default linear_q: dequantize failed");
        let w_dense = Tensor::from_vec(dequant, w.shape().to_vec());
        let w_dense = self.to_device(w_dense);
        self.linear(x, &w_dense)
    }

    /// `y = x / sqrt(mean(x²) + eps) * weight`. Norm is over the last axis.
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor;

    /// Fused residual add + rmsnorm: in-place sets `x += y` and returns
    /// `rmsnorm(x_new, weight, eps)`. Replaces the back-to-back
    /// `add_inplace(x, y)` + `rmsnorm(x, w)` pair every transformer block does
    /// between sub-layers — saves one launch and one full read of `y` (the
    /// residual is consumed exactly once instead of being added to x in pass
    /// A and then x re-read by pass B). Default impl is the host fallback;
    /// CUDA backend overrides with a single kernel.
    fn add_inplace_then_rmsnorm(&self, x: &mut Tensor, y: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        self.add_inplace(x, y);
        self.rmsnorm(x, weight, eps)
    }

    /// Full LayerNorm over the last axis: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
    /// Distinct from `rmsnorm` (which doesn't subtract mean and has no bias).
    /// Used by SigLIP/CLIP/Qwen3-VL vision towers. Default impl is a host
    /// fallback — GPU backends should override to avoid the round trip when
    /// the vision tower is run repeatedly (e.g. multi-image batches).
    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let mut host = x.to_host();
        let last = host.shape()[host.rank() - 1];
        let n_rows = host.numel() / last;
        let w_host = weight.to_host();
        let b_host = bias.to_host();
        let w_data = w_host.data();
        let b_data = b_host.data();
        let data = host.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            let row = &mut data[off..off + last];
            let mean = row.iter().sum::<f32>() / last as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / last as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..last {
                row[j] = (row[j] - mean) * inv * w_data[j] + b_data[j];
            }
        }
        self.to_device(host)
    }

    /// `x[..., j] += bias[j]` — broadcast-add along the last axis. Counterpart
    /// to `mul_inplace_broadcast_last`. Used after every linear in the vision
    /// tower (SigLIP / Qwen3-VL projections all carry per-channel biases).
    /// Default impl is host; GPU backends override.
    fn add_inplace_broadcast_last(&self, x: &mut Tensor, bias: &Tensor) {
        debug_assert_eq!(bias.rank(), 1);
        let last = bias.numel();
        debug_assert_eq!(x.dim(x.rank() - 1), last);
        let n_rows = x.numel() / last;
        let bias_host = bias.to_host();
        let bias_data = bias_host.data();
        let mut host = x.to_host();
        let data = host.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            for j in 0..last { data[off + j] += bias_data[j]; }
        }
        *x = self.to_device(host);
    }

    /// `y = x / sqrt(mean(x²) + eps)` — RMSNorm without a learnable scale
    /// (upstream's `with_scale=False`). Used by Gemma 3n V-norm. Default impl
    /// runs on host; GPU backends override to keep `x` on device.
    fn rmsnorm_no_scale(&self, x: &Tensor, eps: f32) -> Tensor {
        let mut host = x.to_host();
        let last = host.shape()[host.rank() - 1];
        let n_rows = host.numel() / last;
        let data = host.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            let row = &mut data[off..off + last];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / last as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for v in row.iter_mut() { *v *= inv; }
        }
        self.to_device(host)
    }

    /// In-place softmax over the last axis.
    fn softmax_last(&self, x: &mut Tensor);

    /// `y = x * sigmoid(x)`.
    fn silu(&self, x: &Tensor) -> Tensor;

    /// `y = 1 / (1 + exp(-x))`. Used by Qwen3.5's Q-gate (full-attention layers)
    /// and the gated-delta-net β = sigmoid(β) step. Default impl runs on host.
    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let host = x.to_host();
        let mut data = host.data().to_vec();
        for v in &mut data { *v = 1.0 / (1.0 + (-*v).exp()); }
        self.to_device(Tensor::from_vec(data, x.shape().to_vec()))
    }

    /// In-place `x *= sigmoid(gate)`. Replaces a `sigmoid` + `mul_inplace`
    /// pair (saves one launch + one full read+write of the sigmoid intermediate).
    /// Used by Qwen3.5/3.6's gated attention output (`attn_out *= sigmoid(q_gate)`).
    fn mul_sigmoid_inplace(&self, x: &mut Tensor, gate: &Tensor) {
        debug_assert_eq!(x.shape(), gate.shape());
        let g = gate.to_host();
        let g_data = g.data();
        let x_data = x.data_mut();
        for (xv, gv) in x_data.iter_mut().zip(g_data.iter()) {
            let s = 1.0 / (1.0 + (-gv).exp());
            *xv *= s;
        }
    }

    /// `y = ln(1 + exp(x))` — numerically stable. Used by Qwen3.5's
    /// gated-delta-net to compute α_softplus(α + α_bias). Default impl on host.
    fn softplus(&self, x: &Tensor) -> Tensor {
        let host = x.to_host();
        let mut data = host.data().to_vec();
        for v in &mut data {
            // ln(1 + e^x); for x > ~20, that's just x. For very negative x, ln(1+ε) ≈ ε.
            *v = if *v > 20.0 { *v } else { (*v).exp().ln_1p() };
        }
        self.to_device(Tensor::from_vec(data, x.shape().to_vec()))
    }

    /// Element-wise `y = exp(x)`. Default impl on host.
    fn exp(&self, x: &Tensor) -> Tensor {
        let host = x.to_host();
        let mut data = host.data().to_vec();
        for v in &mut data { *v = (*v).exp(); }
        self.to_device(Tensor::from_vec(data, x.shape().to_vec()))
    }

    /// L2-normalize over the last axis: `y = x / sqrt(sum(x²) + eps)`. Note
    /// this is *sum*, not *mean* — distinct from `rmsnorm_no_scale`. Used by
    /// Qwen3.5's gated-delta-net Q/K normalization. Default impl on host.
    fn l2_norm(&self, x: &Tensor, eps: f32) -> Tensor {
        let mut host = x.to_host();
        let last = host.shape()[host.rank() - 1];
        let n_rows = host.numel() / last;
        let data = host.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            let row = &mut data[off..off + last];
            let sum_sq = row.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (sum_sq + eps).sqrt();
            for v in row.iter_mut() { *v *= inv; }
        }
        self.to_device(host)
    }

    /// Approximate GeLU (tanh form) — Gemma's FFN activation.
    /// `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
    fn gelu_approx(&self, x: &Tensor) -> Tensor;

    /// Fused `GeLU(a) * b`. Mirrors `silu_mul`. Used by Gemma 4 MoE's gated MLP
    /// (replaces `gelu_approx(gate)` + `mul_inplace(g, up)` — saves one launch
    /// and one global write/read of the gate buffer per call).
    fn gelu_approx_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        debug_assert_eq!(a.shape(), b.shape());
        let mut out = self.gelu_approx(a);
        self.mul_inplace(&mut out, b);
        out
    }

    /// Fused gate/up SwiGLU split: takes a `[seq, 2*ff]` tensor (typically the
    /// output of one matmul against a stacked `[gate; up]` weight) and returns
    /// `[seq, ff]` with `silu(fused[s, 0..ff]) * fused[s, ff..2*ff]`. Replaces
    /// the slice + `silu_mul` pair when gate and up share the same input — cuts
    /// the FFN matmul count from 2→1 plus eliminates one slice. Default impl
    /// is the host fallback; CUDA backend overrides with a single kernel.
    // VENDORED-LOCAL: GLM-5.3-Flash. Clamped SwiGLU, both orders.
    /// `out = silu(gate) * up` with a clamp at `limit`.
    ///
    /// `after_silu` picks which side of the activation the gate clamp lands on,
    /// because glm5next uses **both**:
    ///
    /// ```text
    /// after_silu = true   clamp(silu(gate), -inf, limit) * clamp(up, -limit, limit)
    /// after_silu = false  silu(clamp(gate, -inf, limit)) * clamp(up, -limit, limit)
    /// ```
    ///
    /// The text FFN clamps the activation (llama.cpp `build_ffn`, callback
    /// `ffn_silu_clamped`); the vision tower clamps the pre-activation
    /// (`FFN_SILU_CLAMP`, callback `ffn_gate_clamped`). `limit <= 0` disables the
    /// clamp and gives a plain SwiGLU.
    ///
    /// Default impl runs on the host; CUDA overrides it. Fusing this matters less
    /// for the arithmetic than for the round trip: without it every routed
    /// expert has to bring its gate/up pair back to the host mid-FFN.
    fn swiglu_clamped(&self, gate: &Tensor, up: &Tensor, limit: f32, after_silu: bool) -> Tensor {
        let g = self.to_host(gate.clone());
        let u = self.to_host(up.clone());
        let (gd, ud) = (g.data(), u.data());
        debug_assert_eq!(gd.len(), ud.len());
        let clamped = limit > 0.0;
        let mut out = Vec::with_capacity(gd.len());
        for (&gv, &uv) in gd.iter().zip(ud.iter()) {
            let mut a = gv;
            if clamped && !after_silu && a > limit {
                a = limit;
            }
            a /= 1.0 + (-a).exp();
            if clamped && after_silu && a > limit {
                a = limit;
            }
            let b = if clamped { uv.clamp(-limit, limit) } else { uv };
            out.push(a * b);
        }
        self.to_device(Tensor::from_vec(out, gate.shape().to_vec()))
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. Clamped SwiGLU over a FUSED gate/up row.
    /// As [`Self::swiglu_clamped`], but the input is one `[seq, 2 * ff]` tensor
    /// laid out `gate` then `up` per row — the layout the expert loader prefers,
    /// because it halves the matmul launch count.
    fn swiglu_clamped_split(
        &self,
        fused: &Tensor,
        ff: usize,
        limit: f32,
        after_silu: bool,
    ) -> Tensor {
        let seq = fused.numel() / (2 * ff);
        let h = self.to_host(fused.clone());
        let d = h.data();
        let clamped = limit > 0.0;
        let mut out = Vec::with_capacity(seq * ff);
        for s in 0..seq {
            let row = &d[s * 2 * ff..(s + 1) * 2 * ff];
            for j in 0..ff {
                let mut a = row[j];
                if clamped && !after_silu && a > limit {
                    a = limit;
                }
                a /= 1.0 + (-a).exp();
                if clamped && after_silu && a > limit {
                    a = limit;
                }
                let mut b = row[ff + j];
                if clamped {
                    b = b.clamp(-limit, limit);
                }
                out.push(a * b);
            }
        }
        self.to_device(Tensor::from_vec(out, vec![seq, ff]))
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. A per-head stack of matvecs.
    /// `y[i] = W[i] @ x[i]` for `i` in `0..b`: `w` is `[b, m, k]` row-major,
    /// `x` is `[b, k]`, the result is `[b, m]`.
    ///
    /// Absorbed MLA needs exactly this twice per layer. `k_b` is
    /// `[n_head, kv_lora, qk_head]` and turns each head's query into the latent
    /// space; `v_b` is `[n_head, v_head, kv_lora]` and turns each head's context
    /// back out of it. Both are the largest weights in the glm5next trunk --
    /// 8.39 M values each -- and neither is a flat matrix, because `x` differs
    /// per head. Doing them as `b` separate `linear` calls would pay `b` launch
    /// latencies; doing them as one padded matmul would read `w` `b` times.
    ///
    /// The default runs on the host, summing each row in index order. CUDA
    /// overrides it with a warp per output row, so its sums are reassociated and
    /// agree only to f32 rounding -- the same trade every other reduction here
    /// makes.
    fn batched_gemv(&self, w: &Tensor, x: &Tensor, b: usize, m: usize, k: usize) -> Tensor {
        let wh = self.to_host(w.clone());
        let xh = self.to_host(x.clone());
        let (wd, xd) = (wh.data(), xh.data());
        debug_assert_eq!(wd.len(), b * m * k);
        debug_assert_eq!(xd.len(), b * k);
        let mut out = vec![0.0f32; b * m];
        for i in 0..b {
            let wi = &wd[i * m * k..(i + 1) * m * k];
            let xi = &xd[i * k..(i + 1) * k];
            let oi = &mut out[i * m..(i + 1) * m];
            for (r, o) in oi.iter_mut().enumerate() {
                let row = &wi[r * k..(r + 1) * k];
                *o = row.iter().zip(xi).map(|(a, c)| a * c).sum();
            }
        }
        self.to_device(Tensor::from_vec(out, vec![b, m]))
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. One step of the KDA delta rule.
    /// Advance a Kimi Delta Attention state by one token, in place, and return
    /// this token's output.
    ///
    /// `state` is `[n_head, head_dim, head_dim]`, `qkvg` is `[4, n_head*head_dim]`
    /// holding q, k, v and `g_log` in that order, `beta` is `[n_head]`. The result
    /// is `[n_head * head_dim]`.
    ///
    /// Per state row `(h, i)`, with the decay on the **kq** axis -- so it varies
    /// along the row, and every row of a head is scaled by the same vector. See
    /// `llama_rs::glm5next::kda` for the reference kernel this is read off; having
    /// it transposed here is a per-token error that compounds, and reads as a
    /// long-prompt bug rather than a wrong one.
    ///
    /// ```text
    ///   row  *= exp(g_log[h,:])
    ///   d     = (v[h,i] - dot(row, k[h])) * beta[h]
    ///   row  += d * k[h]
    ///   out   = dot(row, q[h]) / sqrt(head_dim)
    /// ```
    ///
    /// Every row is independent, which is what makes this a kernel rather than a
    /// scan: 64 heads x 128 rows is 8 192 independent rows of 128 columns.
    ///
    /// The state is `&mut` and stays wherever it already is. That is the whole
    /// point — it is 4.2 MB per layer and 34 layers deep, so a version that
    /// shipped it to the host and back each token would cost more in transfers
    /// than the scalar loop it replaces.
    ///
    /// The default runs on the host, summing in index order, and is the oracle.
    /// CUDA reduces in a warp, so its sums are reassociated; because this is a
    /// *recurrence*, that difference compounds across tokens, which
    /// `kda_delta_step_matches_cpu` measures over a run of steps rather than one.
    fn kda_delta_step(
        &self,
        state: &mut Tensor,
        qkvg: &Tensor,
        beta: &Tensor,
        n_head: usize,
        head_dim: usize,
    ) -> Tensor {
        let hd = head_dim;
        let n = n_head * hd;
        let qh = self.to_host(qkvg.clone());
        let bh = self.to_host(beta.clone());
        let (qd, bd) = (qh.data(), bh.data());
        let (q, k, v, g) = (&qd[0..n], &qd[n..2 * n], &qd[2 * n..3 * n], &qd[3 * n..4 * n]);

        let mut sh = self.to_host(state.clone());
        let st = sh.data_mut();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; n];
        let mut d = vec![0.0f32; hd];

        let mut dec = vec![0.0f32; hd];
        for h in 0..n_head {
            let base = h * hd * hd;
            for (dj, &gj) in dec.iter_mut().zip(&g[h * hd..(h + 1) * hd]) {
                *dj = gj.exp();
            }
            for i in 0..hd {
                let row = &mut st[base + i * hd..base + (i + 1) * hd];
                let mut acc = 0.0f32;
                for ((rj, &kj), &dj) in row
                    .iter_mut()
                    .zip(&k[h * hd..(h + 1) * hd])
                    .zip(dec.iter())
                {
                    *rj *= dj;
                    acc += *rj * kj;
                }
                d[i] = (v[h * hd + i] - acc) * bd[h];
            }
            for i in 0..hd {
                let row = &mut st[base + i * hd..base + (i + 1) * hd];
                let di = d[i];
                let mut acc = 0.0f32;
                for (rj, (&kj, &qj)) in row
                    .iter_mut()
                    .zip(k[h * hd..(h + 1) * hd].iter().zip(&q[h * hd..(h + 1) * hd]))
                {
                    *rj += di * kj;
                    acc += *rj * qj;
                }
                out[h * hd + i] = acc * scale;
            }
        }
        *state = self.to_device(sh);
        self.to_device(Tensor::from_vec(out, vec![n]))
    }

    fn silu_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        let seq = fused.dim(0);
        debug_assert_eq!(fused.dim(fused.rank() - 1), 2 * ff);
        let host = fused.to_host();
        let src = host.data();
        let mut out = vec![0.0f32; seq * ff];
        for s in 0..seq {
            let row = s * 2 * ff;
            let dst = s * ff;
            for j in 0..ff {
                let g = src[row + j];
                let u = src[row + ff + j];
                out[dst + j] = (g / (1.0 + (-g).exp())) * u;
            }
        }
        self.to_device(Tensor::from_vec(out, vec![seq, ff]))
    }

    /// GeLU variant of [`silu_mul_split`]: `gelu_approx(fused[..ff]) * fused[ff..]`.
    /// Used by Gemma family GeGLU once gate+up are fused at load time.
    fn gelu_approx_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        let seq = fused.dim(0);
        debug_assert_eq!(fused.dim(fused.rank() - 1), 2 * ff);
        let host = fused.to_host();
        let src = host.data();
        let mut out = vec![0.0f32; seq * ff];
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEFF: f32 = 0.044715;
        for s in 0..seq {
            let row = s * 2 * ff;
            let dst = s * ff;
            for j in 0..ff {
                let g = src[row + j];
                let u = src[row + ff + j];
                let inner = SQRT_2_OVER_PI * (g + COEFF * g * g * g);
                let gelu_v = 0.5 * g * (1.0 + inner.tanh());
                out[dst + j] = gelu_v * u;
            }
        }
        self.to_device(Tensor::from_vec(out, vec![seq, ff]))
    }

    /// In-place per-row Gaussian-top-k mask:
    ///   `cutoff = mean(x[r]) + std_multiplier * std(x[r])`  (population std)
    ///   `x[r, j] = relu(x[r, j] - cutoff)`
    /// Default impl on host; CUDA backend overrides with a single per-row block
    /// kernel. Used by Gemma 3n's activation sparsity path (first 10 layers
    /// have multiplier `Φ⁻¹(0.95) ≈ 1.6448` ⇒ keep top 5%).
    fn gaussian_topk_inplace(&self, x: &mut Tensor, std_multiplier: f32) {
        let mut host = x.to_host();
        let last = host.shape()[host.rank() - 1];
        let n_rows = host.numel() / last;
        let data = host.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            let row = &mut data[off..off + last];
            let mean = row.iter().sum::<f32>() / last as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / last as f32;
            let cutoff = mean + var.sqrt() * std_multiplier;
            for v in row.iter_mut() { *v = (*v - cutoff).max(0.0); }
        }
        *x = self.to_device(host);
    }

    /// Element-wise `tanh` in place. Used for AltUp router modalities and
    /// final-logit softcap. Default impl falls back through host.
    fn tanh_inplace(&self, x: &mut Tensor) {
        let host = x.to_host();
        let mut data = host.data().to_vec();
        for v in &mut data { *v = v.tanh(); }
        let new = Tensor::from_vec(data, x.shape().to_vec());
        *x = self.to_device(new);
    }

    /// Fused `silu(a) * b`. Saves a kernel launch in SwiGLU FFN. Default impl
    /// composes the two ops; GPU backends override with a single kernel.
    fn silu_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        debug_assert_eq!(a.shape(), b.shape());
        let mut out = self.silu(a);
        self.mul_inplace(&mut out, b);
        out
    }

    /// In-place: `x += y`. Shapes must match.
    fn add_inplace(&self, x: &mut Tensor, y: &Tensor);

    /// In-place: `x *= y`. Shapes must match.
    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor);

    /// In-place scalar multiply: `x *= s`. Default impl runs through host;
    /// CUDA override does it in one kernel launch with no allocation, replacing
    /// the "tile s into a same-shape tensor on host then mul" pattern that
    /// callers have to write otherwise.
    fn mul_scalar_inplace(&self, x: &mut Tensor, s: f32) {
        let mut host = x.to_host();
        for v in host.data_mut() { *v *= s; }
        *x = self.to_device(host);
    }

    /// In-place per-row multiply along the last axis: `x[..., j] *= w[j]`.
    /// `w` must be 1D with `w.numel() == x.dim(rank-1)`. Default impl runs on
    /// host (cheap correct fallback for CPU); GPU backends override with a
    /// single broadcast kernel — saves the per-call d2h-sync-of-w + h2d-of-tiled
    /// round trip that the naive `tile-then-mul` pattern does.
    fn mul_inplace_broadcast_last(&self, x: &mut Tensor, w: &Tensor) {
        debug_assert_eq!(w.rank(), 1);
        let last = w.numel();
        debug_assert_eq!(x.dim(x.rank() - 1), last);
        let n_rows = x.numel() / last;
        let w_host = w.to_host();
        let w_data = w_host.data();
        let x_data = x.data_mut();
        for r in 0..n_rows {
            let off = r * last;
            for j in 0..last { x_data[off + j] *= w_data[j]; }
        }
    }

    /// In-place per-row multiply along axis 0: `x[s, ...] *= g[s]`.
    /// `g` is `[seq]` or `[seq, 1]`; `x` is `[seq, ...]` with matching `seq`.
    /// Used to apply a per-row scalar gate (e.g. Qwen3.6-35B-A3B's sigmoid-gated
    /// shared expert) without round-tripping the gate to host. Default impl is
    /// the host fallback; CUDA backend overrides with a single kernel.
    fn mul_inplace_broadcast_axis0(&self, x: &mut Tensor, g: &Tensor) {
        let seq = x.dim(0);
        debug_assert_eq!(g.numel(), seq, "gate must have one entry per row");
        let inner: usize = x.shape().iter().skip(1).product::<usize>().max(1);
        let g_host = g.to_host();
        let g_data = g_host.data();
        let x_data = x.data_mut();
        for s in 0..seq {
            let g_s = g_data[s];
            let off = s * inner;
            for j in 0..inner { x_data[off + j] *= g_s; }
        }
    }

    /// Apply NeoX-style RoPE in place to ONLY the first `rotated_dim` dims of
    /// each head's `head_dim`-wide slice. Dims `[rotated_dim..head_dim)` are
    /// passed through unchanged. Used by Qwen3.5
    /// (`partial_rotary_factor = 0.25` ⇒ rotates 64 of 256). Default impl is a
    /// host loop; GPU backends should override with a kernel that covers
    /// `(seq * n_heads * rotated_dim/2)` threads.
    fn rope_partial_neox(
        &self,
        x:           &mut Tensor,
        positions:   &[u32],
        head_dim:    usize,
        rotated_dim: usize,
        theta:       f32,
    ) {
        debug_assert_eq!(x.rank(), 3, "expected [seq, n_heads, head_dim]");
        debug_assert_eq!(x.dim(2), head_dim);
        debug_assert!(rotated_dim <= head_dim && rotated_dim % 2 == 0);
        let seq = x.dim(0);
        let n_heads = x.dim(1);
        debug_assert_eq!(positions.len(), seq);
        let half = rotated_dim / 2;

        let mut host = x.to_host();
        let xd = host.data_mut();
        for s in 0..seq {
            let pos = positions[s] as f32;
            for h in 0..n_heads {
                let off = (s * n_heads + h) * head_dim;
                for k in 0..half {
                    let freq = theta.powf(-2.0 * k as f32 / rotated_dim as f32);
                    let angle = pos * freq;
                    let (sin_v, cos_v) = angle.sin_cos();
                    let i_a = off + k;
                    let i_b = off + k + half;
                    let a = xd[i_a];
                    let b = xd[i_b];
                    xd[i_a] = a * cos_v - b * sin_v;
                    xd[i_b] = a * sin_v + b * cos_v;
                }
            }
        }
        *x = self.to_device(host);
    }

    /// VENDORED-LOCAL: axial/interleaved NeoX rotations. Keep the host math as
    /// the reference; GPU backends can avoid materializing attention tensors.
    fn rope_axes(&self, x: &mut Tensor, positions: &[[u32;3]], rotated: usize,
        theta: f32, axes: &[usize], frequencies: &[usize], frequency_dim: usize) {
        assert_eq!(x.rank(),3);
        assert_eq!(positions.len(),x.dim(0));
        assert_eq!(axes.len(),rotated/2);
        assert_eq!(axes.len(),frequencies.len());
        assert!(rotated<=x.dim(2) && rotated%2==0 && frequency_dim>0);
        assert!(axes.iter().all(|&a|a<3));
        let (heads,width)=(x.dim(1),x.dim(2));
        let mut host=x.to_host();
        let data=host.data_mut();
        for (row,pos) in positions.iter().enumerate() {
            for k in 0..rotated/2 {
                let angle=pos[axes[k]] as f32*theta.powf(-2.0*frequencies[k] as f32/frequency_dim as f32);
                let (sin,cos)=angle.sin_cos();
                for head in 0..heads {
                    let a=(row*heads+head)*width+k;
                    let c=a+rotated/2;
                    (data[a],data[c])=(data[a]*cos-data[c]*sin,data[a]*sin+data[c]*cos);
                }
            }
        }
        *x=self.to_device(host);
    }

    /// Run one or more autoregressive steps of Qwen3.5 / qwen3next gated
    /// delta-net: depthwise conv1d + per-head autoregressive update + norm-gated
    /// output. Updates `conv_state` and `state` in place; returns the per-token
    /// output `[seq, num_v_heads * head_v_dim]`. For `seq=1` this is the decode
    /// fast path; for `seq>1` it loops the kernels per-token (state is sequential
    /// across tokens — no inter-token parallelism without the chunked algorithm).
    ///
    /// Inputs (all device tensors except where noted):
    ///   * `mixed_qkv`: `[seq, conv_dim]` post-projection q‖k‖v concat
    ///   * `z_in`:      `[seq, num_v_heads * head_v_dim]` z gate
    ///   * `beta_alpha`:`[seq, 2 * num_v_heads]` β first then α
    ///   * `conv_weight`:`[conv_dim, conv_kernel]` depthwise filter
    ///   * `ssm_a`:     `[num_v_heads]`
    ///   * `dt_bias`:   `[num_v_heads]`
    ///   * `ssm_norm`:  `[head_v_dim]`
    ///   * `conv_state`:`[conv_kernel - 1, conv_dim]` (persistent, in/out)
    ///   * `state`:     `[num_v_heads, head_v_dim, head_v_dim]` (persistent, in/out)
    ///
    /// Default impl falls back to a CPU version; CUDA overrides for the real win.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_step(
        &self,
        mixed_qkv:   &Tensor,
        z_in:        &Tensor,
        beta_alpha:  &Tensor,
        conv_weight: &Tensor,
        ssm_a:       &Tensor,
        dt_bias:     &Tensor,
        ssm_norm:    &Tensor,
        conv_state:  &mut Tensor,
        state:       &mut Tensor,
        seq:         usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_v_dim:  usize,
        head_k_dim:  usize,
        v_per_k:     usize,
        scale_q:     f32,
        eps:         f32,
    ) -> Tensor {
        // Host fallback: per-token loop matching the CUDA kernels (TILED V order: head h_v reads key head h_v % num_k_heads).
        let _ = v_per_k;
        let (out, conv_h, state_h) = host_delta_net(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm,
            &std::mem::replace(conv_state, Tensor::zeros(vec![1])), &std::mem::replace(state, Tensor::zeros(vec![1])),
            seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, scale_q, eps, false);
        *conv_state = self.to_device(conv_h);
        *state = self.to_device(state_h);
        self.to_device(out)
    }

    /// Split a fused `[seq, 3*d]` QKV projection (per-row layout `[Q_d | K_d | V_d]`)
    /// into three `[seq, d]` tensors. Used by the Qwen3-VL ViT block, whose
    /// `attn_qkv.weight` is stored as a single `[3*D, D]` matmul rather than
    /// three separate Q/K/V projections. Default impl is host-side; CUDA
    /// overrides with a single triple-write kernel.
    fn split_qkv_3way(&self, qkv: &Tensor, d: usize) -> (Tensor, Tensor, Tensor) {
        let seq = qkv.dim(0);
        debug_assert_eq!(qkv.dim(qkv.rank() - 1), 3 * d,
            "expected last dim 3*d={}, got {}", 3*d, qkv.dim(qkv.rank() - 1));
        let host = qkv.to_host();
        let src = host.data();
        let mut q = vec![0.0f32; seq * d];
        let mut k = vec![0.0f32; seq * d];
        let mut v = vec![0.0f32; seq * d];
        for s in 0..seq {
            let row = s * 3 * d;
            q[s*d..s*d+d].copy_from_slice(&src[row..row+d]);
            k[s*d..s*d+d].copy_from_slice(&src[row+d..row+2*d]);
            v[s*d..s*d+d].copy_from_slice(&src[row+2*d..row+3*d]);
        }
        (
            self.to_device(Tensor::from_vec(q, vec![seq, d])),
            self.to_device(Tensor::from_vec(k, vec![seq, d])),
            self.to_device(Tensor::from_vec(v, vec![seq, d])),
        )
    }

    /// Split a Qwen3.5 attention `q_proj` output into the query and gate halves.
    /// Input `q_full` has shape `[seq, n_heads * 2 * head_dim]` with each head's
    /// 2*head_dim block laid out as `[query (head_dim) | gate (head_dim)]`.
    /// Returns `(q_only, q_gate)`, each `[seq, n_heads * head_dim]`.
    /// Default impl is host-side (slow d2h + h2d); CUDA overrides with a single
    /// kernel that writes both halves directly on device.
    /// VENDORED-LOCAL: Qwen3.8-Flash-Next — the delta-net step with a `sigmoid(z)` output
    /// gate instead of `silu(z)` (otherwise identical to [`Backend::delta_net_step`]).
    #[allow(clippy::too_many_arguments)]
    fn delta_net_step_sigmoid(
        &self,
        mixed_qkv:   &Tensor,
        z_in:        &Tensor,
        beta_alpha:  &Tensor,
        conv_weight: &Tensor,
        ssm_a:       &Tensor,
        dt_bias:     &Tensor,
        ssm_norm:    &Tensor,
        conv_state:  &mut Tensor,
        state:       &mut Tensor,
        seq:         usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_v_dim:  usize,
        head_k_dim:  usize,
        v_per_k:     usize,
        scale_q:     f32,
        eps:         f32,
    ) -> Tensor {
        // Host fallback: the same per-token loop as `delta_net_step`'s, gated by sigmoid(z) (what the CUDA kernel's
        // gate flag selects), so Flash-Next runs on a backend without CUDA.
        let _ = v_per_k;
        let (out, conv_h, state_h) = host_delta_net(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm,
            &std::mem::replace(conv_state, Tensor::zeros(vec![1])), &std::mem::replace(state, Tensor::zeros(vec![1])),
            seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, scale_q, eps, true);
        *conv_state = self.to_device(conv_h);
        *state = self.to_device(state_h);
        self.to_device(out)
    }

    /// VENDORED-LOCAL: split each row of `x` (`[rows, a + b]`) into its first `a` and last `b`
    /// columns.
    fn split_cols(&self, x: &Tensor, a: usize) -> (Tensor, Tensor) {
        let rows = x.dim(0);
        let width = x.numel() / rows;
        let h = x.to_host();
        let d = h.data();
        let (mut l, mut r) = (Vec::with_capacity(rows * a), Vec::with_capacity(rows * (width - a)));
        for i in 0..rows { l.extend_from_slice(&d[i * width..i * width + a]); r.extend_from_slice(&d[i * width + a..(i + 1) * width]); }
        (Tensor::from_vec(l, vec![rows, a]), Tensor::from_vec(r, vec![rows, width - a]))
    }

    /// VENDORED-LOCAL: the n-gram layer's gate. Per row `r` and stream `s` (width `d`):
    /// `g = signed_sqrt(scale * rms(key[r,s]) * nk[s] . rms(x[r,s]) * nq[s])`,
    /// `gated[r,s] = sigmoid(g) * value[r]`, and `conv_in[r,s] = f16(rms(gated[r,s]) * nc[s])`.
    /// `key`, `x` are `[rows, streams * d]`, `value` `[rows, d]`; the norms `[streams * d]`.
    #[allow(clippy::too_many_arguments)]
    fn ple_gate(&self, key: &Tensor, x: &Tensor, value: &Tensor, norm_key: &Tensor, norm_query: &Tensor, norm_conv: &Tensor, streams: usize, eps: f32) -> (Tensor, Tensor) {
        let rows = value.dim(0);
        let d = value.numel() / rows;
        let width = streams * d;
        let (kh, xh, vh) = (key.to_host(), x.to_host(), value.to_host());
        let (nk, nq, nc) = (norm_key.to_host(), norm_query.to_host(), norm_conv.to_host());
        let (kd, xd, vd, nk, nq, nc) = (kh.data(), xh.data(), vh.data(), nk.data(), nq.data(), nc.data());
        let rms = |v: &[f32]| 1.0 / (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + eps).sqrt();
        let scale = 1.0 / (d as f32).sqrt();
        let mut gated = vec![0f32; rows * width];
        let mut conv_in = vec![0f32; rows * width];
        for r in 0..rows {
            for s in 0..streams {
                let o = r * width + s * d;
                let (kr, qr) = (&kd[o..o + d], &xd[o..o + d]);
                let (ik, iq) = (rms(kr), rms(qr));
                let mut dot = 0f32;
                for j in 0..d { dot += (kr[j] * ik * nk[s * d + j]) * (qr[j] * iq * nq[s * d + j]); }
                let g = dot * scale;
                let ss = if g == 0.0 { 0.0 } else { g.signum() * g.abs().max(1e-6).sqrt() };
                let gate = 1.0 / (1.0 + (-ss).exp());
                for j in 0..d { gated[o + j] = gate * vd[r * d + j]; }
                let inv = rms(&gated[o..o + d]);
                for j in 0..d { conv_in[o + j] = crate::tensor::round_f16(gated[o + j] * inv * nc[s * d + j]); }
            }
        }
        (Tensor::from_vec(gated, vec![rows, width]), Tensor::from_vec(conv_in, vec![rows, width]))
    }

    /// VENDORED-LOCAL: the n-gram layer's dilated causal conv. With `stream` = `window`
    /// (`[kernel - 1) * dilation, width]`) then `conv_in` (`[rows, width]`):
    /// `x[r, c] += gated[r, c] + silu(sum_j w[c, j] * stream[r + j * dilation, c])`.
    /// `window` becomes `stream`'s last `(kernel - 1) * dilation` rows.
    #[allow(clippy::too_many_arguments)]
    fn ple_conv(&self, x: &mut Tensor, gated: &Tensor, conv_in: &Tensor, window: &mut Tensor, weight: &Tensor, kernel: usize, dilation: usize) {
        let rows = gated.dim(0);
        let width = gated.numel() / rows;
        let state = (kernel - 1) * dilation;
        let mut stream = window.to_host().data().to_vec();
        stream.extend_from_slice(conv_in.to_host().data());
        let (gh, wh) = (gated.to_host(), weight.to_host());
        let (gd, wd) = (gh.data(), wh.data());
        let mut delta = vec![0f32; rows * width];
        for r in 0..rows {
            for c in 0..width {
                let mut acc = 0f32;
                for j in 0..kernel { acc += wd[c * kernel + j] * stream[(r + j * dilation) * width + c]; }
                delta[r * width + c] = gd[r * width + c] + acc / (1.0 + (-acc).exp());
            }
        }
        self.add_inplace(x, &Tensor::from_vec(delta, vec![rows, width]));
        *window = Tensor::from_vec(stream[rows * width..].to_vec(), vec![state, width]);
    }

    /// VENDORED-LOCAL: gated residual — `x += post * y` per stream (as `stream_apply`), then
    /// each stream RMS-normed and scaled by its part of `weight` (as `hc_norm`).
    fn hc_apply_norm(&self, x: &mut Tensor, y: &Tensor, post: &Tensor, weight: &Tensor, streams: usize, eps: f32) -> Tensor {
        self.stream_apply(x, y, post, streams);
        self.hc_norm(x, weight, streams, eps)
    }

    /// VENDORED-LOCAL: the `[rows, 2 * words]` values of `packed` (`[rows, words]`, see
    /// `tensor::pack_f16`).
    fn unpack_f16(&self, packed: &Tensor) -> Tensor {
        let rows = packed.dim(0);
        let values = crate::tensor::unpack_f16(packed.to_host().data());
        let cols = values.len() / rows.max(1);
        Tensor::from_vec(values, vec![rows, cols])
    }

    /// VENDORED-LOCAL: gated residual — `normed @ down^T` (`down`: `[rank + writes, width / 2]`,
    /// f16 packed by `tensor::pack_f16`), then as `hc_gates`: returns (`t`
    /// `[rows, rank + writes]`, the write weights).
    fn hc_down_gates(&self, normed: &Tensor, down: &Tensor, rank: usize, writes: usize, streams: usize) -> (Tensor, Tensor) {
        let mut t = self.linear(normed, &self.unpack_f16(down));
        let post = self.hc_gates(&mut t, rank, writes, streams);
        (t, post)
    }

    /// VENDORED-LOCAL: gated residual — `t @ up^T` as logits (`up`: `[width, cols / 2]`, f16
    /// packed), then as `hc_mix`.
    fn hc_up_mix(&self, t: &Tensor, up: &Tensor, normed: &Tensor, streams: usize) -> Tensor {
        let logits = self.linear(t, &self.unpack_f16(up));
        self.hc_mix(&logits, normed, streams)
    }

    /// VENDORED-LOCAL: gated residual — each stream of `x` (`[rows, streams * d]`) RMS-normed
    /// and scaled by its part of `weight` (`[streams * d]`).
    fn hc_norm(&self, x: &Tensor, weight: &Tensor, streams: usize, eps: f32) -> Tensor {
        let rows = x.dim(0);
        let d = x.numel() / rows / streams;
        let (h, w) = (x.to_host(), weight.to_host());
        let (h, w) = (h.data(), w.data());
        let mut out = vec![0f32; h.len()];
        for r in 0..rows {
            for s in 0..streams {
                let o = (r * streams + s) * d;
                let inv = 1.0 / (h[o..o + d].iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
                for j in 0..d { out[o + j] = h[o + j] * inv * w[s * d + j]; }
            }
        }
        Tensor::from_vec(out, x.shape().to_vec())
    }

    /// VENDORED-LOCAL: gated residual — `t` (`[rows, rank + writes]`: the low-rank gate's
    /// input, then each stream's write logit, if any): the first `rank` become
    /// `silu(t / streams)`, the rest 0; returns the write weights `2 * sigmoid(logit / streams)`
    /// (`[rows, writes]`).
    fn hc_gates(&self, t: &mut Tensor, rank: usize, writes: usize, streams: usize) -> Tensor {
        let rows = t.dim(0);
        let width = rank + writes;
        let mut post = vec![0f32; rows * writes];
        let data = t.data_mut();
        for r in 0..rows {
            for c in 0..width {
                let v = data[r * width + c] / streams as f32;
                if c < rank { data[r * width + c] = v / (1.0 + (-v).exp()); } else {
                    post[r * writes + c - rank] = 2.0 / (1.0 + (-v).exp());
                    data[r * width + c] = 0.0;
                }
            }
        }
        Tensor::from_vec(post, vec![rows, writes])
    }

    /// VENDORED-LOCAL: gated residual — the branch input: the mean over streams of
    /// `sigmoid(logits) * normed` (both `[rows, streams * d]`) -> `[rows, d]`.
    fn hc_mix(&self, logits: &Tensor, normed: &Tensor, streams: usize) -> Tensor {
        let rows = normed.dim(0);
        let d = normed.numel() / rows / streams;
        let (l, n) = (logits.to_host(), normed.to_host());
        let (l, n) = (l.data(), n.data());
        let mut out = vec![0f32; rows * d];
        for r in 0..rows { for s in 0..streams { for j in 0..d {
            let i = (r * streams + s) * d + j;
            out[r * d + j] += n[i] / (1.0 + (-l[i]).exp()) / streams as f32;
        } } }
        Tensor::from_vec(out, vec![rows, d])
    }

    /// VENDORED-LOCAL: `dst[r] += src[r] * scale[r]` (`scale`: one value per row).
    fn add_rows_scaled(&self, dst: &mut Tensor, src: &Tensor, scale: &Tensor) {
        let rows = src.dim(0);
        let d = src.numel() / rows.max(1);
        let (s, g) = (src.to_host(), scale.to_host());
        let (s, g) = (s.data(), g.data());
        let out = dst.data_mut();
        for r in 0..rows { for j in 0..d { out[r * d + j] += s[r * d + j] * g[r]; } }
    }

    /// VENDORED-LOCAL: rows `rows[i]` of `src` `[n, d]`, in order: `[rows.len(), d]`.
    fn gather_rows(&self, src: &Tensor, rows: &[u32]) -> Tensor {
        let d = src.numel() / src.dim(0).max(1);
        let host = src.to_host();
        let data = host.data();
        let mut out = Vec::with_capacity(rows.len() * d);
        for &r in rows { out.extend_from_slice(&data[r as usize * d..(r as usize + 1) * d]); }
        Tensor::from_vec(out, vec![rows.len(), d])
    }

    /// VENDORED-LOCAL: `dst[rows[i]] += src[i] * weights[i]` for each row `i` of `src`.
    fn scatter_add_rows(&self, dst: &mut Tensor, src: &Tensor, rows: &[u32], weights: &[f32]) {
        let d = src.numel() / src.dim(0).max(1);
        let s = src.to_host();
        let s = s.data();
        let out = dst.data_mut();
        for (i, (&r, &w)) in rows.iter().zip(weights).enumerate() {
            for j in 0..d { out[r as usize * d + j] += s[i * d + j] * w; }
        }
    }

    /// VENDORED-LOCAL: the mean of `streams` interleaved streams: `x` `[rows, streams * d]`
    /// (row-major `[rows, streams, d]`) -> `[rows, d]`.
    fn stream_mean(&self, x: &Tensor, streams: usize) -> Tensor {
        let rows = x.dim(0);
        let d = x.numel() / rows / streams;
        let h = x.to_host();
        let h = h.data();
        let mut out = vec![0.0f32; rows * d];
        for r in 0..rows { for s in 0..streams { for j in 0..d { out[r * d + j] += h[(r * streams + s) * d + j]; } } }
        for v in &mut out { *v /= streams as f32; }
        Tensor::from_vec(out, vec![rows, d])
    }

    /// VENDORED-LOCAL: the residual write-back of a hyper-connection site:
    /// `x[r, s, :] += post[r, s] * y[r, :]` (`x` `[rows, streams * d]`, `y` `[rows, d]`,
    /// `post` `[rows, streams]`).
    fn stream_apply(&self, x: &mut Tensor, y: &Tensor, post: &Tensor, streams: usize) {
        let rows = y.dim(0);
        let d = y.numel() / rows;
        let yh = y.to_host();
        let ph = post.to_host();
        let (yh, ph) = (yh.data(), ph.data());
        let out = x.data_mut();
        for r in 0..rows { for s in 0..streams { for j in 0..d { out[(r * streams + s) * d + j] += ph[r * streams + s] * yh[r * d + j]; } } }
    }

    /// VENDORED-LOCAL: QSA — the mean of each complete `ratio`-row block of `raw`
    /// (`[rows, .., d]`, the first `blocks * ratio` rows): `[blocks, d]`.
    fn qsa_pool(&self, raw: &Tensor, blocks: usize, ratio: usize) -> Tensor {
        let d = raw.numel() / raw.dim(0).max(1);
        let h = raw.to_host();
        let h = h.data();
        let mut out = vec![0f32; blocks * d];
        for b in 0..blocks { for c in 0..ratio { for j in 0..d { out[b * d + j] += h[(b * ratio + c) * d + j] / ratio as f32; } } }
        Tensor::from_vec(out, vec![blocks, d])
    }

    /// VENDORED-LOCAL: QSA block scores: `out[r, j] = scale * sum_h relu(q[r, h] . pooled[j])`
    /// for the blocks query `r` (at position `first + r`) sees whole, `-inf` for the rest.
    fn qsa_block_scores(&self, q: &Tensor, pooled: &Tensor, first: usize, ratio: usize, scale: f32) -> Tensor {
        let rows = q.dim(0);
        let nb = pooled.dim(0);
        let d = pooled.numel() / nb.max(1);
        let heads = q.numel() / rows / d;
        let (qh, ph) = (q.to_host(), pooled.to_host());
        let (qd, pd) = (qh.data(), ph.data());
        let mut out = vec![f32::NEG_INFINITY; rows * nb];
        for r in 0..rows {
            let visible = (first + r + 1) / ratio;
            for j in 0..nb.min(visible) {
                let mut total = 0f32;
                for hh in 0..heads {
                    let dot: f32 = (0..d).map(|i| qd[(r * heads + hh) * d + i] * pd[j * d + i]).sum();
                    total += dot.max(0.0);
                }
                out[r * nb + j] = total * scale;
            }
        }
        Tensor::from_vec(out, vec![rows, nb])
    }

    /// VENDORED-LOCAL: QSA selection: for each query row of `scores` (`[n, nb]`, the query at
    /// position `first + r`), the tokens it attends to: its `keep` best whole blocks (all of
    /// them when it sees no more) and its incomplete tail block. `[n, keep * ratio + ratio]`
    /// token indices as f32 (exact: under 2^24), `-1` for none.
    fn qsa_select(&self, scores: &Tensor, first: usize, ratio: usize, keep: usize) -> Tensor {
        let n = scores.dim(0);
        let nb = scores.numel() / n.max(1);
        let width = keep * ratio + ratio;
        let h = scores.to_host();
        let h = h.data();
        let mut out = vec![-1f32; n * width];
        for r in 0..n {
            let p = first + r;
            let visible = ((p + 1) / ratio).min(nb);
            let row = &h[r * nb..r * nb + visible];
            let mut chosen: Vec<usize> = (0..visible).collect();
            if visible > keep {
                chosen.select_nth_unstable_by(keep - 1, |&x, &y| row[y].total_cmp(&row[x]));
                chosen.truncate(keep);
            }
            let o = &mut out[r * width..(r + 1) * width];
            let mut i = 0;
            for &b in &chosen { for c in 0..ratio { o[i] = (b * ratio + c) as f32; i += 1; } }
            for t in visible * ratio..=p { o[i] = t as f32; i += 1; }
        }
        Tensor::from_vec(out, vec![n, width])
    }

    /// VENDORED-LOCAL: QSA attention: each query row attends only to the cache rows its row of
    /// `sel` names (from `qsa_select`; `-1`: none). `q` `[n, heads, d]`, `k`/`v` cache buffers
    /// `[capacity, kv_heads, d]`; returns `[n, heads, d]`.
    fn sparse_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, sel: &Tensor, scale: f32) -> Tensor {
        let width = sel.numel() / sel.dim(0).max(1);
        let rows: Vec<i32> = sel.to_host().data().iter().map(|&v| v as i32).collect();
        let n = q.dim(0);
        let (heads, d) = (q.dim(1), q.dim(2));
        let kv_heads = k.dim(1);
        let (qh, kh, vh) = (q.to_host(), k.to_host(), v.to_host());
        let (qd, kd, vd) = (qh.data(), kh.data(), vh.data());
        let mut out = vec![0f32; n * heads * d];
        for r in 0..n {
            for h in 0..heads {
                let g = h / (heads / kv_heads);
                let sel: Vec<usize> = rows[r * width..(r + 1) * width].iter().filter(|&&t| t >= 0).map(|&t| t as usize).collect();
                let scores: Vec<f32> = sel.iter().map(|&t| (0..d).map(|i| qd[(r * heads + h) * d + i] * kd[(t * kv_heads + g) * d + i]).sum::<f32>() * scale).collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = e.iter().sum();
                for (p, &t) in e.iter().zip(&sel) {
                    for i in 0..d { out[(r * heads + h) * d + i] += p / sum * vd[(t * kv_heads + g) * d + i]; }
                }
            }
        }
        Tensor::from_vec(out, vec![n, heads, d])
    }

    fn split_q_and_gate(
        &self,
        q_full:  &Tensor,
        n_heads: usize,
        head_dim: usize,
    ) -> (Tensor, Tensor) {
        let seq = q_full.dim(0);
        let stride = 2 * head_dim;
        let host = q_full.to_host();
        let src = host.data();
        let mut q = vec![0.0f32; seq * n_heads * head_dim];
        let mut g = vec![0.0f32; seq * n_heads * head_dim];
        for s in 0..seq {
            let src_row = s * n_heads * stride;
            let dst_row = s * n_heads * head_dim;
            for h in 0..n_heads {
                let src_off = src_row + h * stride;
                let dst_off = dst_row + h * head_dim;
                q[dst_off..dst_off + head_dim]
                    .copy_from_slice(&src[src_off..src_off + head_dim]);
                g[dst_off..dst_off + head_dim]
                    .copy_from_slice(&src[src_off + head_dim..src_off + stride]);
            }
        }
        (
            self.to_device(Tensor::from_vec(q, vec![seq, n_heads * head_dim])),
            self.to_device(Tensor::from_vec(g, vec![seq, n_heads * head_dim])),
        )
    }

    /// Apply RoPE in place. `x` shape: `[seq, n_heads, head_dim]`.
    /// `positions` is `seq`-long.
    /// `freq_factors`, if present, is a `head_dim/2`-long per-frequency divisor
    /// applied to each angle (long-rope / YaRN-style scaling). `None` = uniform.
    fn rope(
        &self,
        x: &mut Tensor,
        positions: &[u32],
        head_dim: usize,
        rope_type: RopeType,
        theta: f32,
        freq_factors: Option<&[f32]>,
    );

    /// Repeat each kv-head `n_rep` times so it matches the q-head count.
    /// Input: `[seq, n_kv_heads, head_dim]`. Output: `[seq, n_kv_heads*n_rep, head_dim]`.
    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor;

    /// Batched Q·K^T per head: scores\[s, h, t\] = Q\[s, h, :\] · K\[t, h, :\] * scale.
    /// q: \[seq, n_heads, hd\], k: \[kv_len, n_heads, hd\] (already repeated to match Q heads).
    /// Applies a causal mask: positions `t > past + s` are set to -inf.
    /// Returns: \[seq, n_heads, kv_len\].
    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor;

    /// Batched scores·V per head: out\[s, h, d\] = sum_t scores\[s, h, t\] * V\[t, h, d\].
    /// scores: \[seq, n_heads, kv_len\]. v: \[kv_len, n_heads, hd\].
    /// Returns: \[seq, n_heads, hd\] (or an equivalent flattened \[seq, n_heads*hd\]).
    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor;

    /// Full attention with GQA support and a KV-cache prefix.
    ///
    ///   q:           `[seq, n_h_q, hd]`
    ///   k_buffer:    `[max_kv_len, n_h_kv, hd]` — the full cache buffer
    ///   v_buffer:    `[max_kv_len, n_h_kv, hd]`
    ///   kv_len:      live prefix length within the buffers (\<= max_kv_len)
    ///   past:        position offset for `q`'s first row (causal mask)
    ///   scale:       1 / sqrt(head_dim)
    ///
    /// Returns `[seq, n_h_q, hd]`. Internally:
    ///   * Each Q head `h_q` consumes KV head `h_q / (n_h_q / n_h_kv)` (GQA).
    ///   * Only KV positions `[0..kv_len)` are read.
    ///   * Causal mask applied: position `t > past + s` is masked.
    ///
    /// Default impl falls back to expanded `repeat_kv + slice + bmm + softmax + bmm`
    /// — slow because of the kernel launches; GPU backends should override
    /// with a fused implementation.
    ///
    /// `sliding_window`: if `Some(w)`, mask out KV positions older than `w` from each query
    /// (Gemma 3 local-attention layers). `None` = standard full causal attention.
    fn attention(
        &self,
        q: &Tensor,
        k_buffer: &Tensor,
        v_buffer: &Tensor,
        kv_len: usize,
        scale: f32,
        past: usize,
        sliding_window: Option<usize>,
    ) -> Tensor {
        let n_h_q = q.dim(1);
        let n_h_kv = k_buffer.dim(1);
        debug_assert_eq!(n_h_q % n_h_kv, 0);
        let n_rep = n_h_q / n_h_kv;
        let seq = q.dim(0);

        // Slice the cache prefix.
        let k_pref = self.slice_axis0(k_buffer, kv_len);
        let v_pref = self.slice_axis0(v_buffer, kv_len);

        // Repeat to match Q heads if needed.
        let k_full = self.repeat_kv(&k_pref, n_rep);
        let v_full = self.repeat_kv(&v_pref, n_rep);

        let mut scores = self.bmm_qkt(q, &k_full, scale, past);

        // Apply sliding-window mask. We do this on the host because the default impl
        // already shuttles through host buffers; GPU backends override this method
        // with a fused implementation that handles the window inside the kernel.
        if let Some(w) = sliding_window {
            // scores shape: [seq, n_h_q, kv_len]
            let data = scores.data_mut();
            for s in 0..seq {
                let q_pos = past + s;
                let lo = q_pos.saturating_sub(w - 1);
                for h in 0..n_h_q {
                    let row_off = (s * n_h_q + h) * kv_len;
                    for t in 0..lo.min(kv_len) {
                        data[row_off + t] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        self.softmax_last(&mut scores);
        self.bmm_av(&scores, &v_full)
    }

    /// Argmax along the last axis. Input shape `[..., n]` -> output shape `[...]` with usize.
    fn argmax_last(&self, x: &Tensor) -> Vec<u32>;

    /// Allocate a zero-initialized tensor of the given shape on this backend's
    /// storage. For CPU = `Tensor::zeros`; for CUDA = device alloc.
    fn alloc_zeros(&self, shape: Vec<usize>) -> Tensor {
        Tensor::zeros(shape)
    }

    /// Copy `src` into `dst` along axis 0 starting at row `start`. Both tensors
    /// must agree on `shape[1..]` (CPU shapes); flat element count of `src`
    /// must fit within `dst[start..]`.
    fn copy_axis0_into(&self, dst: &mut Tensor, start: usize, src: &Tensor) {
        debug_assert!(start <= dst.dim(0));
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        let src_inner: usize = src.shape().iter().skip(1).product::<usize>().max(1);
        debug_assert_eq!(inner, src_inner, "axis-0 inner shape mismatch");
        let off = start * inner;
        let n = src.numel();
        let src_data = src.data();
        let dst_data = dst.data_mut();
        dst_data[off..off + n].copy_from_slice(src_data);
    }

    /// Gather one axis-1 slice from a 3D tensor:
    ///   `src.shape() == [d0, d1, d2]`  ⇒  `out.shape() == [d0, d2]`
    ///   `out[s, j] = src[s, idx, j]`
    /// Used by Gemma 3n's per-layer PLE injection: `per_layer_inputs` lives on
    /// device as `[seq, n_layers, pld]` and we pull layer `i`'s `[seq, pld]`
    /// slice in one device kernel launch instead of host-slicing + h2d-uploading.
    fn slice_axis1_2d(&self, src: &Tensor, idx: usize) -> Tensor {
        debug_assert_eq!(src.rank(), 3, "slice_axis1_2d expects 3D src");
        let d0 = src.dim(0);
        let d1 = src.dim(1);
        let d2 = src.dim(2);
        debug_assert!(idx < d1);
        let src_data = src.data();
        let mut out = vec![0.0f32; d0 * d2];
        for s in 0..d0 {
            let in_off  = (s * d1 + idx) * d2;
            let out_off = s * d2;
            out[out_off..out_off + d2].copy_from_slice(&src_data[in_off..in_off + d2]);
        }
        Tensor::from_vec(out, vec![d0, d2])
    }

    /// Slice the first `end` rows along axis 0 into a new tensor of shape
    /// `[end, ...src.shape[1..]]`. The returned tensor is independent of the
    /// source; backends may implement this as a deep copy.
    fn slice_axis0(&self, src: &Tensor, end: usize) -> Tensor {
        debug_assert!(end <= src.dim(0));
        let mut shape = src.shape().to_vec();
        shape[0] = end;
        let inner: usize = src.shape().iter().skip(1).product::<usize>().max(1);
        let n = end * inner;
        Tensor::from_vec(src.data()[..n].to_vec(), shape)
    }

    /// Copy `count` axis-0 rows starting at `start` into a fresh tensor of
    /// shape `[count, ...src.shape[1..]]`. CUDA backends override with a
    /// device-to-device memcpy. Used for AltUp stream unstacking.
    fn slice_axis0_range(&self, src: &Tensor, start: usize, count: usize) -> Tensor {
        debug_assert!(start + count <= src.dim(0));
        let mut shape = src.shape().to_vec();
        shape[0] = count;
        let inner: usize = src.shape().iter().skip(1).product::<usize>().max(1);
        let off = start * inner;
        let n = count * inner;
        Tensor::from_vec(src.data()[off..off + n].to_vec(), shape)
    }

    /// Add `src` (broadcast) to a contiguous range of axis-0 slices of `dst`:
    ///   `dst[start..start+count, ...] += src`.
    /// `src.numel()` must equal the product of `dst.shape[1..]`. Used for the
    /// Gemma 3n PLE injection step (`streams[1..n_alt] += normed`) — a single
    /// kernel launch instead of `count` separate `add_inplace` calls.
    fn add_to_axis0_range(&self, dst: &mut Tensor, start: usize, count: usize, src: &Tensor) {
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        debug_assert_eq!(src.numel(), inner, "broadcast src.numel() != dst inner");
        debug_assert!(start + count <= dst.dim(0));
        let dst_data = dst.data_mut();
        let src_data = src.data();
        for s in 0..count {
            let off = (start + s) * inner;
            for j in 0..inner { dst_data[off + j] += src_data[j]; }
        }
    }

    /// Fused scale + axis0-range add: `dst[start..start+count, ...] += src * scale`.
    /// Saves a `mul_scalar_inplace(src, scale)` launch in tight loops (per-expert
    /// MoE accumulation: `K * seq` calls per layer).
    fn add_to_axis0_range_scaled(
        &self,
        dst: &mut Tensor,
        start: usize,
        count: usize,
        src: &Tensor,
        scale: f32,
    ) {
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        debug_assert_eq!(src.numel(), inner, "broadcast src.numel() != dst inner");
        debug_assert!(start + count <= dst.dim(0));
        let dst_data = dst.data_mut();
        let src_data = src.data();
        for s in 0..count {
            let off = (start + s) * inner;
            for j in 0..inner { dst_data[off + j] += src_data[j] * scale; }
        }
    }

    /// Gemma 3n AltUp predict: for each output stream `i`,
    ///   `predicted[i] = streams[i] + sum_j coefs[s, i*n_alt+j] * streams[j]`.
    /// Default impl is the host fallback used by the CPU backend; CUDA backends
    /// override with a single fused kernel launch.
    fn altup_predict(&self, streams: &Tensor, coefs: &Tensor, n_alt: usize) -> Tensor {
        let s_data = streams.data();
        let c_data = coefs.data();
        let seq    = streams.dim(1);
        let hidden = streams.dim(2);
        let mut out = vec![0.0f32; n_alt * seq * hidden];
        for s in 0..seq {
            let coef_off = s * n_alt * n_alt;
            for i in 0..n_alt {
                let out_off_i = (i * seq + s) * hidden;
                let in_off_i  = (i * seq + s) * hidden;
                // residual: + stream[i]
                for h in 0..hidden { out[out_off_i + h] = s_data[in_off_i + h]; }
                for j in 0..n_alt {
                    let c = c_data[coef_off + i * n_alt + j];
                    let in_off_j = (j * seq + s) * hidden;
                    for h in 0..hidden {
                        out[out_off_i + h] += c * s_data[in_off_j + h];
                    }
                }
            }
        }
        Tensor::from_vec(out, vec![n_alt, seq, hidden])
    }

    /// Gemma 3n AltUp correct: for each stream `i`,
    ///   `corrected[i] = predictions[i] + (coefs[s, i] + 1) * (activated - predictions[active_idx])`.
    /// Default impl is the host fallback; CUDA backends override.
    fn altup_correct(
        &self,
        predictions: &Tensor,
        activated: &Tensor,
        coefs: &Tensor,
        n_alt: usize,
        active_idx: usize,
    ) -> Tensor {
        let p_data = predictions.data();
        let a_data = activated.data();
        let c_data = coefs.data();
        let seq    = predictions.dim(1);
        let hidden = predictions.dim(2);
        let mut out = vec![0.0f32; n_alt * seq * hidden];
        for s in 0..seq {
            let act_off = s * hidden;
            let pa_off  = (active_idx * seq + s) * hidden;
            for i in 0..n_alt {
                let pi_off = (i * seq + s) * hidden;
                let coef = c_data[s * n_alt + i] + 1.0;
                for h in 0..hidden {
                    let innov = a_data[act_off + h] - p_data[pa_off + h];
                    out[pi_off + h] = p_data[pi_off + h] + coef * innov;
                }
            }
        }
        Tensor::from_vec(out, vec![n_alt, seq, hidden])
    }
}

// VENDORED-LOCAL: MOE-01 — host reference for [`Backend::moe_route_topk`],
/// shared by the default trait method and the CUDA fallback path (when the
/// router logits are not device-resident). This is exactly the selection +
/// softmax math the llama-rs MoE block ran inline before routing became a
/// backend op: partial-sort top-k, descending order, softmax over the picked
/// logits. `logits` is the flat `[seq, n_experts]` row-major data; the flat
/// `(ids, weights)` results are `seq * top_k` long.
pub fn moe_route_topk_host(logits: &[f32], top_k: usize, n_experts: usize) -> (Vec<u32>, Vec<f32>) {
    assert!(n_experts > 0, "moe_route_topk_host: n_experts must be > 0");
    assert_eq!(
        logits.len() % n_experts,
        0,
        "moe_route_topk_host: logits len {} not a multiple of n_experts {n_experts}",
        logits.len()
    );
    let seq = logits.len() / n_experts;
    let k = top_k.min(n_experts);
    let mut ids = Vec::with_capacity(seq * k);
    let mut weights = Vec::with_capacity(seq * k);
    for row in logits.chunks_exact(n_experts) {
        let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_l = pairs[0].1;
        let exps: Vec<f32> = pairs.iter().map(|(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        ids.extend(pairs.iter().map(|(i, _)| *i as u32));
        weights.extend(exps.iter().map(|e| e / sum));
    }
    (ids, weights)
}
