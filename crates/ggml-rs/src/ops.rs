//! Free-function ops that delegate to a [`Backend`]. These are mostly for
//! ergonomic call sites in model code — `linear(b, x, w)` reads better than
//! `b.linear(x, w)` when a backend is being threaded through.

use crate::backend::{Backend, RopeType};
use crate::quantized::QuantizedTensor;
use crate::tensor::Tensor;

pub fn linear(b: &dyn Backend, x: &Tensor, w: &Tensor) -> Tensor { b.linear(x, w) }

pub fn linear_q(b: &dyn Backend, x: &Tensor, w: &QuantizedTensor) -> Tensor {
    b.linear_q(x, w)
}

pub fn rmsnorm(b: &dyn Backend, x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    b.rmsnorm(x, w, eps)
}

pub fn add_inplace_then_rmsnorm(
    b: &dyn Backend, x: &mut Tensor, y: &Tensor, w: &Tensor, eps: f32,
) -> Tensor {
    b.add_inplace_then_rmsnorm(x, y, w, eps)
}

/// RMSNorm without learnable scale (`y = x / sqrt(mean(x²) + eps)`). Used by
/// Gemma 3n V-norm (upstream's `with_scale=False`).
pub fn rmsnorm_no_scale(b: &dyn Backend, x: &Tensor, eps: f32) -> Tensor {
    b.rmsnorm_no_scale(x, eps)
}

pub fn softmax_last(b: &dyn Backend, x: &mut Tensor) { b.softmax_last(x); }

pub fn silu(b: &dyn Backend, x: &Tensor) -> Tensor { b.silu(x) }

pub fn sigmoid(b: &dyn Backend, x: &Tensor) -> Tensor { b.sigmoid(x) }

pub fn softplus(b: &dyn Backend, x: &Tensor) -> Tensor { b.softplus(x) }

pub fn exp(b: &dyn Backend, x: &Tensor) -> Tensor { b.exp(x) }

/// L2-norm over last axis (`x / sqrt(sum_j x_j² + eps)`). Distinct from
/// `rmsnorm_no_scale` which divides by `sqrt(mean(x²) + eps)`.
pub fn l2_norm(b: &dyn Backend, x: &Tensor, eps: f32) -> Tensor { b.l2_norm(x, eps) }

pub fn gelu_approx(b: &dyn Backend, x: &Tensor) -> Tensor { b.gelu_approx(x) }

pub fn add_inplace(b: &dyn Backend, x: &mut Tensor, y: &Tensor) { b.add_inplace(x, y); }

pub fn mul_inplace(b: &dyn Backend, x: &mut Tensor, y: &Tensor) { b.mul_inplace(x, y); }

/// In-place scalar multiply: `x *= s`. Single backend call.
pub fn mul_scalar_inplace(b: &dyn Backend, x: &mut Tensor, s: f32) {
    b.mul_scalar_inplace(x, s);
}

/// In-place broadcast multiply: `x[..., j] *= w[j]`. `w` must be 1D matching
/// the last dim of `x`. One backend call (one CUDA launch) instead of the
/// "tile w to x's shape on host then mul" pattern.
pub fn mul_inplace_broadcast_last(b: &dyn Backend, x: &mut Tensor, w: &Tensor) {
    b.mul_inplace_broadcast_last(x, w);
}

pub fn tanh_inplace(b: &dyn Backend, x: &mut Tensor) { b.tanh_inplace(x); }

/// In-place Gaussian-top-k mask along the last axis. See [`Backend::gaussian_topk_inplace`].
pub fn gaussian_topk_inplace(b: &dyn Backend, x: &mut Tensor, std_multiplier: f32) {
    b.gaussian_topk_inplace(x, std_multiplier);
}

/// Broadcast `src` into a contiguous range of axis-0 slices of `dst`:
/// `dst[start..start+count, ...] += src`. `src.numel()` must equal the product
/// of `dst.shape[1..]`. One backend call (one CUDA launch) instead of `count`
/// scalar `add_inplace` calls.
pub fn add_to_axis0_range(b: &dyn Backend, dst: &mut Tensor, start: usize, count: usize, src: &Tensor) {
    b.add_to_axis0_range(dst, start, count, src);
}

/// Gather one axis-1 slice from a 3D tensor: `out[s, j] = src[s, idx, j]`.
/// `src` must be 3D `[d0, d1, d2]`; output is 2D `[d0, d2]`. See
/// [`Backend::slice_axis1_2d`].
pub fn slice_axis1_2d(b: &dyn Backend, src: &Tensor, idx: usize) -> Tensor {
    b.slice_axis1_2d(src, idx)
}

pub fn rope(
    b: &dyn Backend,
    x: &mut Tensor,
    positions: &[u32],
    head_dim: usize,
    rope_type: RopeType,
    theta: f32,
) {
    b.rope(x, positions, head_dim, rope_type, theta, None);
}

/// RoPE with per-frequency divisors (long-rope / YaRN). `freq_factors` length
/// must be at least `head_dim/2`; each entry divides the angle for that
/// frequency dim. None ⇒ uniform (= [`rope`]).
pub fn rope_with_factors(
    b: &dyn Backend,
    x: &mut Tensor,
    positions: &[u32],
    head_dim: usize,
    rope_type: RopeType,
    theta: f32,
    freq_factors: Option<&[f32]>,
) {
    b.rope(x, positions, head_dim, rope_type, theta, freq_factors);
}

pub fn repeat_kv(b: &dyn Backend, x: &Tensor, n_rep: usize) -> Tensor {
    b.repeat_kv(x, n_rep)
}

pub fn argmax_last(b: &dyn Backend, x: &Tensor) -> Vec<u32> { b.argmax_last(x) }

pub fn attention(
    b: &dyn Backend,
    q: &Tensor,
    k_buffer: &Tensor,
    v_buffer: &Tensor,
    kv_len: usize,
    scale: f32,
    past: usize,
) -> Tensor {
    b.attention(q, k_buffer, v_buffer, kv_len, scale, past, None)
}

/// Attention with optional sliding-window mask. `sliding_window = Some(w)` masks
/// out KV positions older than `w` tokens before each query (Gemma 3 local layers).
pub fn attention_swa(
    b: &dyn Backend,
    q: &Tensor,
    k_buffer: &Tensor,
    v_buffer: &Tensor,
    kv_len: usize,
    scale: f32,
    past: usize,
    sliding_window: Option<usize>,
) -> Tensor {
    b.attention(q, k_buffer, v_buffer, kv_len, scale, past, sliding_window)
}
