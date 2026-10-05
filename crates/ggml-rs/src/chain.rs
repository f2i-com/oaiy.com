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

/// A device that runs a chain of ops on vectors it holds.
pub trait DeviceChain: Send + Sync {
    /// A zeroed vector of `len`.
    fn vec(&self, len: usize) -> DeviceVec;
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
    /// The length [`ChainRecorder::attention`]'s `out` needs for `n_h` heads of `head_dim` over a cache of `cap` rows:
    /// the result and the device's scratch.
    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize;
    /// Start recording.
    fn begin(&self) -> Box<dyn ChainRecorder + '_>;
}

/// Ops recorded in order, run together by [`ChainRecorder::finish`].
pub trait ChainRecorder {
    /// `y = W x` for one row `x` (`[k]`, its first `k` elements) of a weight the device holds (`[n, k]`).
    fn matmul(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec);
    /// `out = x / rms(x) * w`.
    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32);
    /// [`Self::rmsnorm`] of each of `rows` rows of `x` (a head's q or k), all with the same `w` (`[x.len / rows]`).
    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32);
    /// `acc += y`.
    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec);
    /// `out = silu(fused[..ff]) * fused[ff..]`, `ff = out.len`.
    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec);
    /// Rotate `x` (`[heads, head_dim]`) in place: pair `k` of each head by `table[2k]` (sine) and `table[2k + 1]`
    /// (cosine), the pairs `(2k, 2k + 1)` or with `neox` `(k, k + head_dim / 2)`.
    fn rope(&mut self, x: &DeviceVec, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool);
    /// `dst[offset..offset + src.len] = src`.
    fn store(&mut self, src: &DeviceVec, dst: &DeviceVec, offset: usize);
    /// One query's attention over a layer's cache `kv` (row `t`: its K `[n_kv, head_dim]` then its V), positions
    /// `lo..kv_len`, each query head on its KV head (`h / (n_h / n_kv)`): `out[..n_h * head_dim]` the result, the rest
    /// of `out` scratch ([`DeviceChain::attention_out_len`] long, `cap` the cache's rows).
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32);
    /// Read `v` back once the chain has run.
    fn read(&mut self, v: &DeviceVec) {
        self.read_range(v, 0, v.len)
    }
    /// Read `v[offset..offset + len]` back once the chain has run.
    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize);
    /// Run what was recorded (one submit) and return what was read, in order.
    fn finish(self: Box<Self>) -> Vec<Vec<f32>>;
}
