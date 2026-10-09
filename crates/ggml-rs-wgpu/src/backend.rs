//! `ggml_rs::Backend` for the WebGPU backend: quantized projections on the device, the rest on the CPU backend.

use super::*;

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn delta_net_step(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, seq: usize, num_v_heads: usize, num_k_heads: usize, head_v_dim: usize, head_k_dim: usize,
        v_per_k: usize, scale_q: f32, eps: f32,
    ) -> Tensor {
        let d = ggml_rs::DeltaNet { rows: seq, v_heads: num_v_heads, k_heads: num_k_heads, k_dim: head_k_dim, v_dim: head_v_dim, scale_q, eps, sigmoid_gate: false };
        match self.delta_net_gpu(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, d) {
            Some(out) => out,
            None => self.cpu.delta_net_step(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps),
        }
    }
    fn delta_net_step_sigmoid(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, seq: usize, num_v_heads: usize, num_k_heads: usize, head_v_dim: usize, head_k_dim: usize,
        v_per_k: usize, scale_q: f32, eps: f32,
    ) -> Tensor {
        let d = ggml_rs::DeltaNet { rows: seq, v_heads: num_v_heads, k_heads: num_k_heads, k_dim: head_k_dim, v_dim: head_v_dim, scale_q, eps, sigmoid_gate: true };
        match self.delta_net_gpu(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, d) {
            Some(out) => out,
            None => self.cpu.delta_net_step_sigmoid(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps),
        }
    }
    fn vram_status(&self) -> Option<(usize, usize)> {
        let (used, budget) = self.usage();
        Some((budget.saturating_sub(used) as usize, budget as usize))
    }
    fn to_device_quant(&self, w: QuantizedTensor) -> QuantizedTensor {
        self.upload(w)
    }
    fn try_to_device_quant(&self, w: QuantizedTensor, _safety_margin_bytes: usize) -> QuantizedTensor {
        // The budget is the whole weight allowance; activations live on the host.
        self.upload(w)
    }
    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        if let Some(q) = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()) {
            return self.linear_gpu(x, q, w.shape());
        }
        self.cpu.linear_q(x, w)
    }
    /// Weights of one input that are all on this adapter, in one submit: a layer's q, k and v were three round trips
    /// a decode step.
    fn linear_q_many(&self, x: &Tensor, ws: &[&QuantizedTensor]) -> Vec<Tensor> {
        let on_gpu: Option<Vec<(&WgpuQuant, &[usize])>> =
            ws.iter().map(|w| Some((w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?, w.shape()))).collect();
        let host;
        let x = if x.is_device() {
            host = x.to_host();
            &host
        } else {
            x
        };
        match on_gpu {
            Some(on_gpu) if on_gpu.len() > 1 && !x.data().is_empty() => {
                let k = on_gpu[0].1[1];
                let m = x.numel() / k;
                let widest = on_gpu.iter().map(|(_, s)| s[0]).max().unwrap_or(0);
                let fits = m <= (chunk_limit(&self.gpu.limits) as usize / (4 * k.max(widest))).max(1);
                if on_gpu.iter().all(|(_, s)| s[1] == k && s[0] > 0) && fits {
                    let start = std::time::Instant::now();
                    let ys = self.linear_gpu_batch(x, &on_gpu);
                    profile::add(&profile::LINEAR, start);
                    return ys;
                }
                ws.iter().map(|w| self.linear_q(x, w)).collect()
            }
            _ => ws.iter().map(|w| self.linear_q(x, w)).collect(),
        }
    }

    // Everything else is CpuBackend's, forwarded explicitly so its optimized
    // overrides are kept rather than the trait's defaults.
    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        self.cpu.embed_lookup(table, tokens, embedding_dim)
    }
    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor {
        self.cpu.linear(x, w)
    }
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        self.cpu.rmsnorm(x, weight, eps)
    }
    fn silu_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.silu_mul_split(fused, ff)
    }
    fn chain(&self) -> Option<&dyn ggml_rs::chain::DeviceChain> {
        Some(self)
    }
    fn gelu_approx_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.gelu_approx_mul_split(fused, ff)
    }
    fn softmax_last(&self, x: &mut Tensor) {
        self.cpu.softmax_last(x)
    }
    fn silu(&self, x: &Tensor) -> Tensor {
        self.cpu.silu(x)
    }
    fn gelu_approx(&self, x: &Tensor) -> Tensor {
        self.cpu.gelu_approx(x)
    }
    fn add_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.add_inplace(x, y)
    }
    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.mul_inplace(x, y)
    }
    fn rope(&self, x: &mut Tensor, positions: &[u32], head_dim: usize, rope_type: RopeType, theta: f32, freq_factors: Option<&[f32]>) {
        self.cpu.rope(x, positions, head_dim, rope_type, theta, freq_factors)
    }
    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor {
        self.cpu.repeat_kv(x, n_rep)
    }
    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor {
        self.cpu.bmm_qkt(q, k, scale, past)
    }
    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        self.cpu.bmm_av(scores, v)
    }
    /// The CPU's fused attention (the cache is on the host): the default would copy the cache and take the softmax on
    /// one thread.
    fn attention(&self, q: &Tensor, k_buffer: &Tensor, v_buffer: &Tensor, kv_len: usize, scale: f32, past: usize, sliding_window: Option<usize>) -> Tensor {
        let start = std::time::Instant::now();
        let host = |t: &Tensor| if t.is_device() { t.to_host() } else { t.clone() };
        let out = if q.is_device() || k_buffer.is_device() || v_buffer.is_device() {
            self.cpu.attention(&host(q), &host(k_buffer), &host(v_buffer), kv_len, scale, past, sliding_window)
        } else {
            self.cpu.attention(q, k_buffer, v_buffer, kv_len, scale, past, sliding_window)
        };
        profile::add(&profile::ATTENTION, start);
        out
    }
    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        self.cpu.argmax_last(x)
    }
}
