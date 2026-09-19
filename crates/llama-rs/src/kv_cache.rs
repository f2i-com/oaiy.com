//! Per-layer KV cache.
//!
//! K and V buffers are pre-allocated as `[max_len, n_kv_heads, head_dim]`
//! tensors on the model's backend (CPU `Vec<f32>` or CUDA device memory).
//! `append` writes new rows in place; `attention` then reads the
//! `[0..len + new_seq)` prefix directly via `Backend::attention`. We don't
//! materialize a sliced "K_full" tensor — that's the load-bearing change for
//! good GPU perf, since it eliminates two ops (slice + repeat_kv) per layer.

use ggml_rs::{Backend, Tensor};

#[derive(Debug)]
pub struct KvCache {
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
            k, v, len: 0, max_len,
            n_kv_heads: n_kv_max,
            n_kv_heads_per_layer: n_kv_heads_per_layer.to_vec(),
            head_dims: head_dims.to_vec(),
            ssm_state: (0..n_layers).map(|_| None).collect(),
            ssm_conv:  (0..n_layers).map(|_| None).collect(),
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for s in &mut self.ssm_state { *s = None; }
        for c in &mut self.ssm_conv  { *c = None; }
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
        debug_assert!(self.len + seq <= self.max_len, "kv cache overflow");
        backend.copy_axis0_into(&mut self.k[layer], self.len, new_k);
        backend.copy_axis0_into(&mut self.v[layer], self.len, new_v);
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
