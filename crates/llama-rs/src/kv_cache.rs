//! Per-layer KV cache.
//!
//! K and V buffers are pre-allocated as `[max_len, n_kv_heads, head_dim]`
//! tensors on the model's backend (CPU `Vec<f32>` or CUDA device memory).
//! `append` writes new rows in place; `attention` then reads the
//! `[0..len + new_seq)` prefix directly via `Backend::attention`. We don't
//! materialize a sliced "K_full" tensor — that's the load-bearing change for
//! good GPU perf, since it eliminates two ops (slice + repeat_kv) per layer.

use ggml_rs::{Backend, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// VENDORED-LOCAL: each cache's [`KvCache::id`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct KvCache {
    /// VENDORED-LOCAL: optional per-layer placement. These caches grow as tokens arrive.
    pub layer_backends: Vec<Arc<dyn Backend>>,
    pub k:           Vec<Tensor>,
    pub v:           Vec<Tensor>,
    pub len:         usize,
    pub max_len:     usize,
    /// For uniform-arch models the K/V head count is constant across layers;
    /// for some Gemma 4 MoE layouts (e.g. 26B-A4B) global layers have fewer
    /// kv heads than SWA layers. This stays as the default/typical count;
    /// `n_kv_heads_per_layer` overrides per layer where applicable.
    pub n_kv_heads:  usize,
    /// Per-layer kv-head count. Constructed via `new` / `new_per_layer` to
    /// `vec![n_kv_heads; n_layers]`; varies under `new_per_layer_kv`.
    pub n_kv_heads_per_layer: Vec<usize>,
    /// Per-layer head_dim. For uniform-arch models all entries are equal; Gemma 4
    /// has a different head_dim for SWA vs global layers.
    pub head_dims:   Vec<usize>,

    // ----- SSM state (Qwen3.5 / qwen3next gated-delta-net). One slot per
    // model layer; only the recurrent layers populate them. None for attention
    // layers (and all-attention archs). Lazily filled on first forward call.

    /// Per-layer recurrent state matrix `[num_v_heads, head_v_dim, head_v_dim]`
    /// (flattened to `[num_v_heads * head_v_dim, head_v_dim]` on the device).
    pub ssm_state:   Vec<Option<Tensor>>,
    /// Per-layer conv1d sliding-window state `[conv_kernel - 1, conv_dim]`,
    /// holding the last `K-1` time steps of the projected qkv concatenation.
    pub ssm_conv:    Vec<Option<Tensor>>,

    /// VENDORED-LOCAL: for a copy of this cache kept elsewhere (a chained decode step's, on the device): which cache
    /// it is, and the first row [`KvCache::append`] has written since the copy was last brought up to date (0: all of
    /// them; `usize::MAX`: none).
    pub id:          u64,
    pub dirty_from:  usize,
}

impl KvCache {
    /// Uniform head_dim across all layers (most archs).
    pub fn new(
        backend: &dyn Backend,
        n_layers: usize,
        max_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new_per_layer(backend, max_len, n_kv_heads, &vec![head_dim; n_layers])
    }

    /// Per-layer head_dim (Gemma 4 — SWA layers have head_dim=256, global=512).
    pub fn new_per_layer(
        backend: &dyn Backend,
        max_len: usize,
        n_kv_heads: usize,
        head_dims: &[usize],
    ) -> Self {
        Self::new_per_layer_kv(
            backend, max_len,
            &vec![n_kv_heads; head_dims.len()],
            head_dims,
        )
    }

    /// Per-layer head_dim AND per-layer kv-head count (Gemma 4 MoE 26B-A4B —
    /// SWA layers: n_kv=8, hd=256; global layers: n_kv=2, hd=512).
    pub fn new_per_layer_kv(
        backend: &dyn Backend,
        max_len: usize,
        n_kv_heads_per_layer: &[usize],
        head_dims: &[usize],
    ) -> Self {
        debug_assert_eq!(n_kv_heads_per_layer.len(), head_dims.len(),
            "kv heads / head_dims length mismatch");
        let k: Vec<Tensor> = n_kv_heads_per_layer.iter().zip(head_dims.iter())
            .map(|(&nkv, &hd)| backend.alloc_zeros(vec![max_len, nkv, hd]))
            .collect();
        let v: Vec<Tensor> = n_kv_heads_per_layer.iter().zip(head_dims.iter())
            .map(|(&nkv, &hd)| backend.alloc_zeros(vec![max_len, nkv, hd]))
            .collect();
        let n_layers = head_dims.len();
        // `n_kv_heads` field: report the maximum so any consumer that reads it
        // sees a safe upper bound. The per-layer source-of-truth is the vec.
        let n_kv_max = n_kv_heads_per_layer.iter().copied().max().unwrap_or(0);
        Self {
            layer_backends: Vec::new(),
            k, v, len: 0, max_len,
            n_kv_heads: n_kv_max,
            n_kv_heads_per_layer: n_kv_heads_per_layer.to_vec(),
            head_dims: head_dims.to_vec(),
            ssm_state: (0..n_layers).map(|_| None).collect(),
            ssm_conv:  (0..n_layers).map(|_| None).collect(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            dirty_from: 0,
        }
    }

    pub fn reset(&mut self) {
        self.dirty_from = 0;
        self.len = 0;
        for s in &mut self.ssm_state { *s = None; }
        for c in &mut self.ssm_conv  { *c = None; }
    }

    pub fn new_lazy_per_layer_kv(backends: Vec<Arc<dyn Backend>>, max_len: usize,
        heads: &[usize], dims: &[usize]) -> Self {
        assert_eq!(backends.len(), heads.len());
        assert_eq!(heads.len(), dims.len());
        let allocate = || backends.iter().zip(heads).zip(dims)
            .map(|((b, &h), &d)| b.alloc_zeros(vec![max_len.min(256).max(1), h, d])).collect();
        Self { k: allocate(), v: allocate(), len: 0, max_len,
            n_kv_heads: heads.iter().copied().max().unwrap_or(0),
            n_kv_heads_per_layer: heads.to_vec(), head_dims: dims.to_vec(),
            ssm_state: (0..heads.len()).map(|_| None).collect(),
            ssm_conv: (0..heads.len()).map(|_| None).collect(), layer_backends: backends,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed), dirty_from: 0 }
    }

    pub fn reserve_layer(&mut self, backend: &dyn Backend, layer: usize, needed: usize) {
        assert!(needed <= self.max_len, "kv cache overflow");
        if needed <= self.k[layer].dim(0) { return; }
        let capacity = needed.next_power_of_two().min(self.max_len);
        let shape = vec![capacity, self.n_kv_heads_per_layer[layer], self.head_dims[layer]];
        // Replace one buffer at a time to bound peak device memory during growth.
        for buffer in [&mut self.k[layer], &mut self.v[layer]] {
            let mut grown = backend.alloc_zeros(shape.clone());
            if self.len != 0 {
                let prefix = backend.slice_axis0(buffer, self.len);
                backend.copy_axis0_into(&mut grown, 0, &prefix);
            }
            *buffer = grown;
        }
    }

    pub fn size_bytes(&self) -> usize {
        2 * self.k.iter().map(|t| t.numel() * 4).sum::<usize>()
    }

    /// Write `seq` new rows into K and V at the current write head. Caller
    /// must follow with `commit(seq)` once attention is done.
    pub fn append(
        &mut self,
        backend: &dyn Backend,
        layer: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) {
        let seq = new_k.dim(0);
        let hd = self.head_dims[layer];
        let nkv = self.n_kv_heads_per_layer[layer];
        debug_assert_eq!(new_k.shape(), &[seq, nkv, hd]);
        debug_assert_eq!(new_v.shape(), &[seq, nkv, hd]);
        self.reserve_layer(backend, layer, self.len + seq);
        backend.copy_axis0_into(&mut self.k[layer], self.len, new_k);
        backend.copy_axis0_into(&mut self.v[layer], self.len, new_v);
        self.dirty_from = self.dirty_from.min(self.len);
    }

    /// VENDORED-LOCAL: [`Self::append`] of `seq` rows given as a device's cache holds them, each row's K then its V
    /// (`[seq, 2, n_kv, head_dim]`, a chained run's read back): copied straight into a host cache's K and V (a 512-row
    /// chunk's 64 MB 18 ms through two vectors of their own, freshly allocated), through [`Self::append`] into a
    /// device's.
    pub fn append_rows(&mut self, backend: &dyn Backend, layer: usize, rows: &[f32], seq: usize) {
        let inner = self.n_kv_heads_per_layer[layer] * self.head_dims[layer];
        assert_eq!(rows.len(), seq * 2 * inner, "a cache's {seq} rows, K then V");
        self.reserve_layer(backend, layer, self.len + seq);
        if self.k[layer].is_device() || self.v[layer].is_device() {
            let (mut kh, mut vh) = (Vec::with_capacity(seq * inner), Vec::with_capacity(seq * inner));
            for r in rows.chunks_exact(2 * inner) {
                kh.extend_from_slice(&r[..inner]);
                vh.extend_from_slice(&r[inner..]);
            }
            let shape = vec![seq, self.n_kv_heads_per_layer[layer], self.head_dims[layer]];
            self.append(backend, layer, &Tensor::from_vec(kh, shape.clone()), &Tensor::from_vec(vh, shape));
            return;
        }
        let at = self.len * inner;
        for (half, dst) in [&mut self.k[layer], &mut self.v[layer]].into_iter().enumerate() {
            let dst = &mut dst.data_mut()[at..at + seq * inner];
            for (d, r) in dst.chunks_exact_mut(inner).zip(rows.chunks_exact(2 * inner)) {
                d.copy_from_slice(&r[half * inner..(half + 1) * inner]);
            }
        }
        self.dirty_from = self.dirty_from.min(self.len);
    }

    pub fn n_kv_heads_for(&self, layer: usize) -> usize {
        self.n_kv_heads_per_layer[layer]
    }

    pub fn commit(&mut self, seq: usize) {
        self.len += seq;
    }

    pub fn k_buffer(&self, layer: usize) -> &Tensor { &self.k[layer] }
    pub fn v_buffer(&self, layer: usize) -> &Tensor { &self.v[layer] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device's rows (each K then V) appended as they are read back give the cache [`KvCache::append`] gives: two
    /// batches, one after the other.
    #[test]
    fn rows_read_back_append_as_k_and_v_do() {
        let cpu = ggml_rs::CpuBackend::new();
        let (nkv, hd) = (2usize, 4usize);
        let inner = nkv * hd;
        let (mut a, mut b) = (KvCache::new(&cpu, 1, 64, nkv, hd), KvCache::new(&cpu, 1, 64, nkv, hd));
        for (seq, base) in [(3usize, 0.0f32), (9, 100.0)] {
            let rows: Vec<f32> = (0..seq * 2 * inner).map(|i| base + i as f32).collect();
            let (mut k, mut v) = (Vec::new(), Vec::new());
            for r in rows.chunks_exact(2 * inner) {
                k.extend_from_slice(&r[..inner]);
                v.extend_from_slice(&r[inner..]);
            }
            a.append(&cpu, 0, &Tensor::from_vec(k, vec![seq, nkv, hd]), &Tensor::from_vec(v, vec![seq, nkv, hd]));
            b.append_rows(&cpu, 0, &rows, seq);
            a.commit(seq);
            b.commit(seq);
        }
        let n = a.len * inner;
        assert_eq!(&a.k_buffer(0).data()[..n], &b.k_buffer(0).data()[..n]);
        assert_eq!(&a.v_buffer(0).data()[..n], &b.v_buffer(0).data()[..n]);
        assert_eq!(a.dirty_from, b.dirty_from);
    }
}
