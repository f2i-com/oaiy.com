//! VENDORED-LOCAL: a decode step's ops chained on a device between host round trips.
//!
//! A backend whose every call is a submit and a read back (WebGPU) spends a dense model's decode step on round trips:
//! a 3B Llama's 113 of them were most of its 37 ms. A [`DeviceChain`] keeps the activations (and a copy of the KV
//! cache) on the device and records a run of ops (quantized matmuls, RMSNorm, RoPE, adds, the SwiGLU, attention) into
//! one submit, reading back only what the host needs (the logits, the step's K and V rows for the host's cache). A
//! model asks its backend for one through [`crate::Backend::chain`] and keeps its own path where there is none.

use std::any::Any;
use std::sync::Arc;

use crate::quantized::QuantizedTensor;
use crate::tensor::Tensor;

/// An f32 vector held by a [`DeviceChain`]'s device: the backend's own buffer behind it.
#[derive(Clone)]
pub struct DeviceVec {
    pub len: usize,
    pub inner: Arc<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for DeviceVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceVec({})", self.len)
    }
}

/// The shape of a gated delta net's recurrence over a run of tokens ([`ChainRecorder::delta_net`]).
#[derive(Clone, Copy, Debug)]
pub struct DeltaNet {
    /// Tokens in the run.
    pub rows: usize,
    pub v_heads: usize,
    pub k_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub scale_q: f32,
    pub eps: f32,
    /// The output gated by `sigmoid(z)` (Qwen3.8-Flash-Next) where Qwen3.5's is `silu(z)`.
    pub sigmoid_gate: bool,
}

/// A device that runs a chain of ops on vectors it holds.
pub trait DeviceChain: Send + Sync {
    /// A zeroed vector of `len`.
    fn vec(&self, len: usize) -> DeviceVec;
    /// Zero `v` (before whatever is recorded next runs).
    fn zero(&self, v: &DeviceVec);
    /// A tensor of `shape` whose storage is `v` itself, not a copy (a model's recurrent state kept on the device in its
    /// cache between steps): its host copy reads `v` back as it is then, and a clone of it is a copy of `v`.
    fn alias(&self, v: &DeviceVec, shape: Vec<usize>) -> Tensor;
    /// The vector `t` is, if [`DeviceChain::alias`] made it on this device.
    fn aliased(&self, t: &Tensor) -> Option<DeviceVec>;
    /// Write `data` into `v` from element `offset` (before whatever is recorded next runs).
    fn upload_at(&self, v: &DeviceVec, offset: usize, data: &[f32]);
    /// Write `data` into `v` from its start.
    fn upload(&self, v: &DeviceVec, data: &[f32]) {
        self.upload_at(v, 0, data)
    }
    /// A vector of `len` holding `v`'s first `min(len, v.len)` elements (the rest zero).
    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec;
    /// Whether this device holds `w` where its matmuls read it.
    fn holds(&self, w: &QuantizedTensor) -> bool;
    /// Whether this device holds the EXL3 projection `w` where [`ChainRecorder::exl3_rows`] reads it.
    fn holds_exl3(&self, w: &dyn crate::exl3::PackedLinear) -> bool;
    /// Whether this device holds a MoE layer's experts as groups [`ChainRecorder::moe_rows`] runs.
    fn holds_experts(&self, e: &dyn crate::exl3::Experts) -> bool;
    /// The length [`ChainRecorder::attention`]'s `out` needs for `n_h` heads of `head_dim` over a cache of `cap` rows:
    /// the result and the device's scratch.
    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize;
    /// The length [`ChainRecorder::attention_rows`]'s `out` needs for `rows` queries over `kv_len` positions.
    fn attention_rows_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize;
    /// Start recording.
    fn begin(&self) -> Box<dyn ChainRecorder + '_>;
}

/// Ops recorded in order, run together by [`ChainRecorder::finish`].
pub trait ChainRecorder {
    /// `y = W x` for one row `x` (`[k]`, its first `k` elements) of a weight the device holds (`[n, k]`).
    fn matmul(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec) {
        self.matmul_rows(w, x, y, 1)
    }
    /// `y[r] = W x[r]` for `rows` rows of `x` (`[rows, k]`) into `y` (`[rows, n]`): a prompt's.
    fn matmul_rows(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// `out = x / rms(x) * w`.
    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32);
    /// [`Self::rmsnorm`] of each of `rows` rows of `x` (a head's q or k), all with the same `w` (`[x.len / rows]`).
    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32);
    /// `acc += y`.
    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec);
    /// `out = silu(fused[..ff]) * fused[ff..]`, `ff = out.len`.
    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        self.silu_mul_split_rows(fused, out, 1)
    }
    /// `out = gelu_approx(fused[..ff]) * fused[ff..]` (the tanh approximation), `ff = out.len`.
    fn gelu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        self.gelu_mul_split_rows(fused, out, 1)
    }
    /// [`Self::silu_mul_split`] of each of `rows` rows (`fused` `[rows, 2 ff]`, `out` `[rows, ff]`).
    fn silu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize);
    /// [`Self::gelu_mul_split`] of each of `rows` rows.
    fn gelu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize);
    /// Whether this recording's bind groups are kept for the steps after (true at first). A prompt's chunk, whose
    /// vectors are its own, keeps none, so they do not hold its buffers' memory after it.
    fn keep_groups(&mut self, keep: bool) {
        let _ = keep;
    }
    /// Rotate `x` (`[heads, head_dim]`) in place: pair `k` of each head by `table[2k]` (sine) and `table[2k + 1]`
    /// (cosine), the pairs `(2k, 2k + 1)` or with `neox` `(k, k + head_dim / 2)`.
    fn rope(&mut self, x: &DeviceVec, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool) {
        self.rope_rows(x, 1, heads, head_dim, table, neox)
    }
    /// [`Self::rope`] of each of `rows` rows (`x` `[rows, heads, head_dim]`), row `r` by `table[r * head_dim..]`.
    fn rope_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool);
    /// `rows` rows of `len` from `src` into `dst`'s rows `start..start + rows`, a row `stride` long, at `at` in it.
    #[allow(clippy::too_many_arguments)]
    fn store_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, len: usize, start: usize, stride: usize, at: usize);
    /// A prompt's attention: query `s` of `rows` (`q` `[rows, n_h, head_dim]`, at position `past + s`) over the cache
    /// `kv` (as [`Self::attention`]'s), positions up to its own and, with `window`, its last `window`; `out` the result
    /// `[rows, n_h, head_dim]` then scratch ([`DeviceChain::attention_rows_out_len`] long).
    #[allow(clippy::too_many_arguments)]
    fn attention_rows(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32);
    /// `dst[offset..offset + src.len] = src`.
    fn store(&mut self, src: &DeviceVec, dst: &DeviceVec, offset: usize) {
        self.copy(src, 0, dst, offset, src.len)
    }
    /// `dst[dst_at..dst_at + len] = src[src_at..src_at + len]`.
    fn copy(&mut self, src: &DeviceVec, src_at: usize, dst: &DeviceVec, dst_at: usize, len: usize);
    /// One query's attention over a layer's cache `kv` (row `t`: its K `[n_kv, head_dim]` then its V), positions
    /// `lo..kv_len`, each query head on its KV head (`h / (n_h / n_kv)`): `out[..n_h * head_dim]` the result, the rest
    /// of `out` scratch ([`DeviceChain::attention_out_len`] long, `cap` the cache's rows).
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32);
    /// `dst[r * width + i] = src[r * stride + at + i]` for each of `rows` rows: a run of columns of each row of `src`
    /// (the query or the gate half of each head of Qwen3.5's q).
    #[allow(clippy::too_many_arguments)]
    fn copy_cols(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, width: usize, stride: usize, at: usize);
    /// RoPE on the first `rot` of each head's `head_dim` in place, NeoX pairs `(k, k + rot / 2)`, row `r` of `rows` by
    /// `table[r * rot..]` (each pair's sine then cosine); the rest of each head as it is (Qwen3.5's partial rotation).
    fn rope_partial_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, rot: usize, table: &DeviceVec);
    /// `out[i] = x[i] * sigmoid(gate[i])` for `i < len` (Qwen3.5's gated attention output).
    fn mul_sigmoid(&mut self, x: &DeviceVec, gate: &DeviceVec, out: &DeviceVec, len: usize);
    /// `out[i] = silu(gate[i]) * up[i]` for `i < len` (a SwiGLU whose gate and up are two weights).
    fn silu_mul(&mut self, gate: &DeviceVec, up: &DeviceVec, out: &DeviceVec, len: usize);
    /// `y[r] = W x[r]` for `rows` rows of `x` (`[rows, k]`), `W` f32 weights `[n, k]` (row-major) held in `w`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// A gated delta net's causal depthwise conv over `rows` tokens of `qkv` (`[rows, channels]`) after the
    /// `kernel - 1` inputs `state` holds (`[kernel - 1, channels]`, oldest first), with `weight` (`[channels, kernel]`),
    /// through SiLU into `out` (`[rows, channels]`); `state` then holds the last `kernel - 1` inputs. As the host's
    /// (`Backend::delta_net_step`).
    #[allow(clippy::too_many_arguments)]
    fn ssm_conv(&mut self, qkv: &DeviceVec, weight: &DeviceVec, state: &DeviceVec, out: &DeviceVec, rows: usize, channels: usize, kernel: usize);
    /// A gated delta net's recurrence over `d.rows` tokens, as the host's (`Backend::delta_net_step`): each token's
    /// conv output `conv` (`[rows, 2 k_heads k_dim + v_heads v_dim]`: q, k, v), gate `z` (`[rows, v_heads v_dim]`)
    /// and `beta_alpha` (`[rows, 2 v_heads]`) with the layer's `ssm_a`, `dt_bias` (`[v_heads]`) and `norm`
    /// (`[v_dim]`) update `state` (`[v_heads, v_dim, k_dim]`, the host's layout) and give the norm-gated output `out`
    /// (`[rows, v_heads v_dim]`). Head `h` reads key head `h % k_heads`.
    #[allow(clippy::too_many_arguments)]
    fn delta_net(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: DeltaNet);
    /// `y[r] = W x[r]` for `rows` rows of `x` through an EXL3 projection this device holds
    /// ([`DeviceChain::holds_exl3`]): its input and output transforms (the channel maps, the scales, the Hadamard
    /// transforms and their f16 roundings) on the device as the host makes them.
    fn exl3_rows(&mut self, w: &dyn crate::exl3::PackedLinear, x: &DeviceVec, y: &DeviceVec, rows: usize);
    /// [`Self::rmsnorm_rows`] of each of `rows * streams` rows of `x`, row `r`'s stream `s` by `w[s * d..]` (`d` its
    /// width): a hyper-connection's per-stream norm (`Backend::hc_norm`).
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_streams(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, eps: f32);
    /// A hyper-connection's gates (`Backend::hc_gates`): of each of `rows` rows of `t` (`[rows, rank + writes]`) the
    /// first `rank` become `silu(t / streams)`, the rest 0, and `post` (`[rows, writes]`) gets `2 sigmoid(t /
    /// streams)` of them.
    #[allow(clippy::too_many_arguments)]
    fn hc_gates(&mut self, t: &DeviceVec, post: &DeviceVec, rows: usize, rank: usize, writes: usize, streams: usize);
    /// A hyper-connection's branch input (`Backend::hc_mix`): `out[r] = sum over s of normed[r, s] *
    /// sigmoid(logits[r, s]) / streams`, rows of `d`.
    #[allow(clippy::too_many_arguments)]
    fn hc_mix(&mut self, logits: &DeviceVec, normed: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, d: usize);
    /// A hyper-connection site's write-back (`Backend::stream_apply`): `x[r, s] += post[r, s] * y[r]`, rows of `d`.
    #[allow(clippy::too_many_arguments)]
    fn stream_apply(&mut self, x: &DeviceVec, y: &DeviceVec, post: &DeviceVec, rows: usize, streams: usize, d: usize);
    /// A MoE layer's experts the device holds as groups, for each of `assign.len()` rows of `x` (`[rows, hidden]`):
    /// row `r`'s experts `assign[r]` (expert, weight; the routed ones, then the shared one last, as
    /// `Experts::forward` routes them), their weighted outputs summed in that order into `out` (`[rows, hidden]`).
    fn moe_rows(&mut self, experts: &dyn crate::exl3::Experts, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]);
    /// `acc[i] += weights[at] * y[i]` for `i < len`: a weighted sum's term, its weight read from the device (an
    /// expert's, written by the host before the chain runs).
    fn axpy_at(&mut self, acc: &DeviceVec, y: &DeviceVec, weights: &DeviceVec, at: usize, len: usize);
    /// Read `v` back once the chain has run.
    fn read(&mut self, v: &DeviceVec) {
        self.read_range(v, 0, v.len)
    }
    /// Read `v[offset..offset + len]` back once the chain has run.
    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize);
    /// Run what was recorded (one submit) and return what was read, in order.
    fn finish(self: Box<Self>) -> Vec<Vec<f32>>;
}
