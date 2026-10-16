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

/// each cache's [`KvCache::id`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// where a cache's rows a device holds alone are read from ([`KvCache::device_rows`]: a chained run's
/// copy of the attention cache, the one copy of the rows it wrote). The host's K and V hold the cache's first
/// [`KvCache::host_rows`] rows, the device the rest: a copy of every row on the host as well was the device's cache a
/// second time in the computer's memory (the 27B's 128 KB a token: 4 GB at 32,000 tokens, beside the card's copy).
pub trait DeviceRows: Send + Sync + std::fmt::Debug {
    /// Whether `layer` is one whose rows it holds (an attention layer's).
    fn holds(&self, layer: usize) -> bool;
    /// Rows `from..to` of layer `layer` of cache `id`: its K and its V, each `[to - from, heads, dim]`, on the host.
    /// None where it does not hold them.
    fn rows(&self, id: u64, layer: usize, from: usize, to: usize) -> Option<(Tensor, Tensor)>;
    /// Cache `id` is gone: whatever is kept for it goes.
    fn forget(&self, id: u64);
}

#[derive(Debug)]
pub struct KvCache {
    /// optional per-layer placement. These caches grow as tokens arrive.
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

    /// for a copy of this cache kept elsewhere (a chained decode step's, on the device): which cache
    /// it is, and the first row [`KvCache::append`] has written since the copy was last brought up to date (0: all of
    /// them; `usize::MAX`: none).
    pub id:          u64,
    pub dirty_from:  usize,
    /// Rows the cache is about to hold, where its user knows (a prompt's length, before its first chunk runs): a
    /// device's copy of the attention cache takes room for them at once. Grown as the rows came, a 15.6K-token
    /// prompt's copy was made three times (for 4,096 rows, 8,192 and 16,384: 3.8 GiB of new memory on the card where
    /// 2.2, and new memory there costs a third of a second a GiB).
    pub expect:      usize,
    /// with [`Self::device_rows`], the rows the host's K and V hold, from the first (no more than
    /// [`Self::len`] count); the rest are the device's alone. Without it the host holds them all.
    pub host_rows:   usize,
    pub device_rows: Option<Arc<dyn DeviceRows>>,
}

impl Drop for KvCache {
    fn drop(&mut self) {
        if let Some(d) = self.device_rows.take() {
            d.forget(self.id);
        }
    }
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
            expect: 0,
            host_rows: 0,
            device_rows: None,
        }
    }

    pub fn reset(&mut self) {
        self.dirty_from = 0;
        self.expect = 0;
        self.len = 0;
        for s in &mut self.ssm_state { *s = None; }
        for c in &mut self.ssm_conv  { *c = None; }
        // (a conversation begun again: what the host held of the last is let go)
        if self.device_rows.is_some() {
            self.release_host();
        }
    }

    /// the rows the host's K and V hold, from the first: all of them where no device holds rows alone.
    pub fn host_valid(&self) -> usize {
        if self.device_rows.is_some() { self.host_rows.min(self.len) } else { self.len }
    }

    /// layer `layer`'s first `upto` rows (K, then V, each `[upto, heads, dim]`) on the host: the
    /// host's own, and where the device holds the rest alone, those read from it (not kept: the host holds no more
    /// rows after than before).
    pub fn layer_prefix(&self, backend: &dyn Backend, layer: usize, upto: usize) -> (Tensor, Tensor) {
        let have = self.host_valid().min(upto);
        let host = |t: &Tensor, n: usize| backend.slice_axis0(t, n).to_host();
        if have == upto || self.device_rows.as_ref().is_none_or(|d| !d.holds(layer)) {
            return (host(&self.k[layer], upto), host(&self.v[layer], upto));
        }
        let (dk, dv) = self.device_rows.as_ref().and_then(|d| d.rows(self.id, layer, have, upto)).unwrap_or_else(|| panic!("rows {have}..{upto} of layer {layer}: the device does not hold them"));
        if have == 0 {
            return (dk, dv);
        }
        let join = |t: &Tensor, rest: Tensor| {
            let mut data = if have > 0 { host(t, have).data().to_vec() } else { Vec::new() };
            data.extend_from_slice(rest.data());
            Tensor::from_vec(data, vec![upto, self.n_kv_heads_per_layer[layer], self.head_dims[layer]])
        };
        (join(&self.k[layer], dk), join(&self.v[layer], dv))
    }

    /// the host's K and V given the rows up to `upto` that the device holds alone (for what reads the
    /// host's cache itself: a run op by op).
    pub fn materialize(&mut self, backend: &dyn Backend, upto: usize) {
        let have = self.host_valid();
        let Some(d) = self.device_rows.clone().filter(|_| have < upto) else { return };
        for layer in (0..self.k.len()).filter(|&l| d.holds(l)) {
            let (k, v) = d.rows(self.id, layer, have, upto).unwrap_or_else(|| panic!("rows {have}..{upto} of layer {layer}: the device does not hold them"));
            let b = self.layer_backends.get(layer).cloned();
            let b: &dyn Backend = b.as_deref().unwrap_or(backend);
            self.reserve_layer(b, layer, upto);
            b.copy_axis0_into(&mut self.k[layer], have, &b.to_device(k));
            b.copy_axis0_into(&mut self.v[layer], have, &b.to_device(v));
        }
        self.host_rows = upto;
    }

    /// the host's K and V let go, where the device holds every row: rows of none, grown again where
    /// the host writes some ([`Self::reserve_layer`]). (Made again whole, of zeros, they took memory the allocator had
    /// freed and zeroed every page of it: the whole of a 16,384-position cache written where none of it was wanted.)
    pub fn release_host(&mut self) {
        let Some(d) = self.device_rows.clone() else { return };
        for layer in (0..self.k.len()).filter(|&l| d.holds(l)) {
            for t in [&mut self.k[layer], &mut self.v[layer]].into_iter().filter(|t| !t.is_device() && t.numel() > 0) {
                let mut shape = t.shape().to_vec();
                shape[0] = 0;
                *t = Tensor::zeros(shape);
            }
        }
        self.host_rows = 0;
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
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed), dirty_from: 0, expect: 0, host_rows: 0, device_rows: None }
    }

    pub fn reserve_layer(&mut self, backend: &dyn Backend, layer: usize, needed: usize) {
        assert!(needed <= self.max_len, "kv cache overflow");
        if needed <= self.k[layer].dim(0) { return; }
        let capacity = needed.next_power_of_two().min(self.max_len);
        let shape = vec![capacity, self.n_kv_heads_per_layer[layer], self.head_dims[layer]];
        // (the rows the host holds: a device's alone are not here to keep)
        let keep = self.host_valid().min(self.k[layer].dim(0));
        // Replace one buffer at a time to bound peak device memory during growth.
        for buffer in [&mut self.k[layer], &mut self.v[layer]] {
            let mut grown = backend.alloc_zeros(shape.clone());
            if keep != 0 {
                let prefix = backend.slice_axis0(buffer, keep);
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
        // (rows before these the device holds alone: to the host first, which these follow)
        self.materialize(backend, self.len);
        self.reserve_layer(backend, layer, self.len + seq);
        backend.copy_axis0_into(&mut self.k[layer], self.len, new_k);
        backend.copy_axis0_into(&mut self.v[layer], self.len, new_v);
        self.dirty_from = self.dirty_from.min(self.len);
        self.host_rows = self.len + seq;
    }

    /// [`Self::append`] of `seq` rows given as a device's cache holds them, each row's K then its V
    /// (`[seq, 2, n_kv, head_dim]`, a chained run's read back): copied straight into a host cache's K and V (a 512-row
    /// chunk's 64 MB 18 ms through two vectors of their own, freshly allocated), through [`Self::append`] into a
    /// device's.
    pub fn append_rows(&mut self, backend: &dyn Backend, layer: usize, rows: &[f32], seq: usize) {
        let inner = self.n_kv_heads_per_layer[layer] * self.head_dims[layer];
        assert_eq!(rows.len(), seq * 2 * inner, "a cache's {seq} rows, K then V");
        self.materialize(backend, self.len);
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
        self.host_rows = self.len + seq;
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

    /// A device's rows for one cache: row `t` of a layer holds `t` in K and `-t` in V, layer 1 its only attention layer.
    #[derive(Debug, Default)]
    struct Device(std::sync::Mutex<Vec<u64>>);

    impl DeviceRows for Device {
        fn holds(&self, layer: usize) -> bool {
            layer == 1
        }
        fn rows(&self, _: u64, layer: usize, from: usize, to: usize) -> Option<(Tensor, Tensor)> {
            assert_eq!(layer, 1);
            let n = to - from;
            let k: Vec<f32> = (from..to).flat_map(|t| [t as f32; 4]).collect();
            Some((Tensor::from_vec(k.clone(), vec![n, 2, 2]), Tensor::from_vec(k.iter().map(|x| -x).collect(), vec![n, 2, 2])))
        }
        fn forget(&self, id: u64) {
            self.0.lock().unwrap().push(id);
        }
    }

    /// A cache whose rows a device holds alone past its host's: read whole (the host's then the device's, the host
    /// given none), brought to the host where a run op by op writes after them, let go, and its id forgotten as it goes.
    #[test]
    fn rows_a_device_holds_alone_are_read_from_it_and_kept_only_where_wanted() {
        let cpu = ggml_rs::CpuBackend::new();
        let device = Arc::new(Device::default());
        let mut kv = KvCache::new_per_layer_kv(&cpu, 64, &[1, 2], &[1, 2]);
        let row = |t: usize| Tensor::from_vec(vec![t as f32; 4], vec![1, 2, 2]);
        // three rows written here, then seven more on the device
        for t in 0..3 {
            let (k, v) = (row(t), Tensor::from_vec(vec![-(t as f32); 4], vec![1, 2, 2]));
            kv.append(&cpu, 1, &k, &v);
            kv.commit(1);
        }
        kv.device_rows = Some(device.clone());
        kv.len = 10;
        assert_eq!(kv.host_valid(), 3);
        let (k, v) = kv.layer_prefix(&cpu, 1, 10);
        let want: Vec<f32> = (0..10).flat_map(|t| [t as f32; 4]).collect();
        assert_eq!((k.shape(), k.data()), (&[10usize, 2, 2][..], &want[..]));
        assert_eq!(v.data(), want.iter().map(|x| -x).collect::<Vec<_>>().as_slice());
        assert_eq!(kv.host_valid(), 3, "the device's rows read, not kept");
        // a run op by op after them: the device's rows to the host first, then its own
        kv.append(&cpu, 1, &row(10), &Tensor::from_vec(vec![-10.0; 4], vec![1, 2, 2]));
        kv.commit(1);
        assert_eq!(kv.host_valid(), 11);
        assert_eq!(&kv.k_buffer(1).data()[..44], (0..11).flat_map(|t| [t as f32; 4]).collect::<Vec<_>>().as_slice());
        // a shorter cache (a rollback): the host's rows past it are not its
        kv.len = 6;
        assert_eq!(kv.host_valid(), 6);
        // let go: the host holds none (rows of none), and reads come from the device again
        kv.release_host();
        assert_eq!((kv.host_valid(), kv.k_buffer(1).dim(0)), (0, 0));
        assert_eq!(kv.layer_prefix(&cpu, 1, 6).0.data(), (0..6).flat_map(|t| [t as f32; 4]).collect::<Vec<_>>().as_slice());
        // a recurrent layer's are the host's, whatever the device holds
        assert_eq!(kv.layer_prefix(&cpu, 0, 2).0.shape(), &[2, 1, 1]);
        let id = kv.id;
        drop(kv);
        assert_eq!(*device.0.lock().unwrap(), vec![id]);
    }

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
