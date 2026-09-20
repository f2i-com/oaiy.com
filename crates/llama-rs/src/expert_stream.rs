//! VENDORED-LOCAL: streaming MoE expert backend (nrob bridge).
//!
//! The resident loaders ([`crate::qwen3moe`], [`crate::mixtral`]) materialize
//! every expert's weights host-resident at load time. This module is the
//! alternative for MoE models opened with [`crate::Model::open_streaming`]:
//! expert tensors are never materialized; instead each expert's packed bytes
//! are fetched on demand through [`nrob::ecache::Ecache`], a bounded RAM
//! cache, from the [`gguf::TensorBytes`] source attached to the
//! [`GgufFile`] (a [`gguf::FileSource`]: positioned reads from the `.gguf`).
//!
//! # Record layout
//!
//! One expert = one fixed-size record, addressed by `(layer, expert)`:
//!
//! ```text
//!   record = gate_bytes || up_bytes || down_bytes
//! ```
//!
//! Each part is the expert's raw ggml-quantized byte slice:
//!   * stacked layout (`blk.N.ffn_gate_exps.weight`, `[n_experts, ff, hidden]`):
//!     the slice `[e * (total / n_experts), (e+1) * (total / n_experts))` —
//!     exactly the split [`crate::qwen3moe::split_stacked_experts`] performs;
//!   * per-expert layout (`blk.N.ffn_gate.E.weight`): the whole tensor.
//!
//! The per-expert *shape* must be uniform within and across layers (it is,
//! for every GGUF MoE dump seen); this is *verified* at store construction.
//! Mixed-*quant* models (Q4_K_M: some layers' down projection Q6_K, others
//! Q4_K) make per-expert byte counts layer-dependent, so each part reserves
//! a fixed region of the per-model max size inside the record and the
//! per-layer dtype/length is recorded alongside — fixed-size records for
//! the cache, exact bytes for the kernels.
//!
//! # Numeric equivalence
//!
//! A fetched record is reconstructed into the same `FfnPair::Fused` /
//! `Weight` values the resident path builds (gate||up byte-concat along
//! axis 0, down as-is), and the forward loop runs the identical backend ops
//! (`linear_q` on the packed bytes, `silu_mul_split`, scaled accumulate).
//! Same bytes + same kernels => token-identical output to the resident path.
//!
//! # Errors in an infallible forward pass
//!
//! `Model::forward` returns a `Tensor`, not a `Result`, so a fetch failure
//! mid-forward cannot propagate. Instead the first error is recorded in
//! [`StreamShared::error`] and the layer contributes zeros; callers
//! ([`crate::Model::expert_stream_error`]) check after each decode step and
//! abort. This keeps the "never panic on malformed data" rule without
//! re-plumbing every forward signature in the crate.

use std::sync::{Arc, Mutex};

use ggml_quants::GgmlType;
use ggml_rs::quantized::{QuantizedHostBytes, QuantizedTensor};
use ggml_rs::{Backend, Tensor};
use gguf::{GgufFile, TensorInfo};

use nrob::ecache::Ecache;
use nrob::store::WeightStore;
use nrob::Error;

use crate::loader::{dtype_supports_packed_matmul, FfnPair, Weight};
use crate::moe::MoeOptions;

/// Minimum records the cache must hold for a streaming open to be accepted.
/// Below a handful of records the cache keeps nothing alive between layers
/// and every dispatch is a disk read; refuse rather than pretend.
pub const MIN_CACHE_RECORDS: usize = 8;

/// Most threads [`GgufExpertStore::fetch_many`] reads a batch with. An NVMe
/// drive saturates at a modest queue depth; more threads only add spawns.
const FETCH_WORKERS: usize = 8;

/// Zero-copy view over a byte range of a cache-leased record, handed to
/// [`QuantizedTensor::from_mmap`] so a cached expert's packed bytes are read
/// directly by the quant kernels with no `.to_vec()` doubling.
///
// VENDORED-LOCAL (CACHE-01): the view owns an `nrob::ecache::HostLease`
// instead of an `Arc<Vec<u8>>` copy. The lease pins the cache slot's record
// buffer for as long as any `QuantizedTensor` built from it lives — through
// the packed matvec — with zero copies on a cache hit.
struct RecordView {
    lease: nrob::ecache::HostLease,
    start: usize,
    len:   usize,
}

impl QuantizedHostBytes for RecordView {
    fn as_bytes(&self) -> &[u8] {
        &self.lease[self.start..self.start + self.len]
    }
}

/// One of the three expert projections (gate / up / down), resolved for
/// every layer.
///
/// Mixed-quant models (e.g. Q4_K_M, where some layers' down projection is
/// Q6_K and others' Q4_K) make the per-expert byte count layer-dependent,
/// while the cache wants fixed-size records. The layout therefore reserves
/// each part a region of `max_per_expert_bytes` inside the record and
/// stores the *actual* per-layer byte count + dtype; a fetch fills only the
/// layer's actual bytes and reconstruction reads exactly those.
#[derive(Debug, Clone)]
struct PartLayout {
    /// Per-layer base offset (GGUF data-section frame) of this part's
    /// expert-0 bytes.
    base_offsets: Vec<u64>,
    // VENDORED-LOCAL: which shard of a split GGUF each layer's bytes live in.
    // `base_offsets` are relative to that shard's own data section, so the read
    // has to go through that shard's source.
    shards: Vec<u32>,
    /// Per-layer bytes per expert slice (the stride between experts; within
    /// a layer the stacked split is uniform — checked at resolve).
    per_expert_bytes: Vec<usize>,
    /// Per-layer dtype.
    dtypes: Vec<GgmlType>,
    /// Region size reserved in the record: max over layers.
    max_per_expert_bytes: usize,
    /// Per-expert logical shape, loader convention (`[rows, cols]`) —
    /// uniform across layers (verified; a model that varies this cannot
    /// stream as fixed records at all).
    shape: [usize; 2],
}

impl PartLayout {
    /// Byte range of expert `e` within layer `l`, in the data-section frame.
    fn range(&self, layer: usize, expert: usize) -> (u64, usize) {
        (
            self.base_offsets[layer] + (expert * self.per_expert_bytes[layer]) as u64,
            self.per_expert_bytes[layer],
        )
    }
}

/// Resolved expert storage layout for one model. Built once at open from the
/// GGUF tensor table; validates uniformity.
#[derive(Debug)]
pub struct ExpertLayout {
    gate: PartLayout,
    up:   PartLayout,
    down: PartLayout,
    n_layers:  u32,
    n_experts: u32,
    // VENDORED-LOCAL: index of the first block that HAS experts. 0 for
    // qwen3moe/mixtral, where every block is MoE; 3 for glm5next, whose
    // `leading_dense_block_count` blocks have a dense FFN and no expert
    // tensors at all. Records stay addressed by MoE-relative layer, so a
    // `(layer, expert)` key means "the layer'th MoE block", not "block layer".
    first_layer: u32,
}

/// Loader-convention shape of a stacked `[n_experts, a, b]` or per-expert
/// `[a, b]` tensor's expert slice: GGUF stores dims fastest-varying-first,
/// so reversing gives `[n_experts, a, b]` and the expert shape is `[a, b]`.
fn expert_shape(info: &TensorInfo, stacked: bool, n_experts: usize) -> crate::Result<[usize; 2]> {
    let rev: Vec<usize> = info.shape.iter().map(|&d| d as usize).rev().collect();
    if stacked {
        if rev.len() != 3 || rev[0] != n_experts {
            return Err(crate::LlamaError::Config(format!(
                "tensor '{}' expected stacked shape [{n_experts}, a, b], got {rev:?}",
                info.name
            )));
        }
        Ok([rev[1], rev[2]])
    } else {
        if rev.len() != 2 {
            return Err(crate::LlamaError::Config(format!(
                "tensor '{}' expected per-expert shape [a, b], got {rev:?}",
                info.name
            )));
        }
        Ok([rev[0], rev[1]])
    }
}

impl ExpertLayout {
    /// Resolve the expert layout for a MoE GGUF. Auto-detects the stacked
    /// (`ffn_*_exps.weight`) vs per-expert (`ffn_*.E.weight`) naming, the
    /// same way [`crate::qwen3moe::load_per_layer_experts_compat`] does.
    ///
    /// Every layer must agree on dtype, per-expert bytes, and per-expert
    /// shape for all three parts; anything else is rejected.
    pub fn resolve(g: &GgufFile, n_layers: usize, n_experts: usize) -> crate::Result<Self> {
        Self::resolve_range(g, 0, n_layers, n_experts)
    }

    /// VENDORED-LOCAL: as [`Self::resolve`], but the MoE blocks start at
    /// `first_layer` instead of block 0, and `n_layers` counts MoE blocks rather
    /// than the model's depth. glm5next needs this: its blocks 0..=2 are dense,
    /// so probing `blk.0.ffn_gate_exps.weight` finds nothing and the unoffset
    /// path failed with a MissingTensor on a perfectly good file.
    pub fn resolve_range(
        g: &GgufFile,
        first_layer: usize,
        n_layers: usize,
        n_experts: usize,
    ) -> crate::Result<Self> {
        if n_layers == 0 || n_experts == 0 {
            return Err(crate::LlamaError::Config(
                "expert layout: n_layers and n_experts must be > 0".into(),
            ));
        }
        let stacked = g
            .tensor_by_name(&format!("blk.{first_layer}.ffn_gate_exps.weight"))
            .is_some();

        let parts = |part: &str| -> crate::Result<PartLayout> {
            let mut base_offsets = Vec::with_capacity(n_layers);
            let mut shards = Vec::with_capacity(n_layers);
            let mut per_expert_bytes = Vec::with_capacity(n_layers);
            let mut dtypes = Vec::with_capacity(n_layers);
            let mut max_per = 0usize;
            let mut shape: Option<[usize; 2]> = None;
            for l in first_layer..first_layer + n_layers {
                let name = if stacked {
                    format!("blk.{l}.ffn_{part}_exps.weight")
                } else {
                    format!("blk.{l}.ffn_{part}.0.weight")
                };
                let info = g.tensor_by_name(&name).ok_or_else(|| {
                    crate::LlamaError::MissingTensor(name.clone())
                })?;
                let nbytes = info.nbytes() as usize;
                let per = if stacked {
                    if nbytes % n_experts != 0 {
                        return Err(crate::LlamaError::Config(format!(
                            "tensor '{name}' bytes ({nbytes}) not divisible by n_experts ({n_experts})"
                        )));
                    }
                    nbytes / n_experts
                } else {
                    nbytes
                };
                let sh = expert_shape(info, stacked, n_experts)?;
                match shape {
                    None => shape = Some(sh),
                    Some(prev) if prev != sh => {
                        return Err(crate::LlamaError::Config(format!(
                            "expert part '{part}' of layer {l} has shape {sh:?}, \
                             layer {first_layer} has {prev:?}; non-uniform expert shapes \
                             cannot stream as fixed-size records"
                        )));
                    }
                    _ => {}
                }
                max_per = max_per.max(per);
                per_expert_bytes.push(per);
                dtypes.push(info.dtype);
                base_offsets.push(info.offset);
                shards.push(g.shard_of(info) as u32);
            }
            Ok(PartLayout {
                base_offsets,
                shards,
                per_expert_bytes,
                dtypes,
                max_per_expert_bytes: max_per,
                shape: shape.unwrap_or([0, 0]),
            })
        };

        let gate = parts("gate")?;
        let up = parts("up")?;
        let down = parts("down")?;
        Ok(Self {
            gate,
            up,
            down,
            n_layers: n_layers as u32,
            n_experts: n_experts as u32,
            first_layer: first_layer as u32,
        })
    }

    /// VENDORED-LOCAL: the model block index of MoE layer 0.
    pub fn first_layer(&self) -> u32 { self.first_layer }

    /// Fixed record size: the per-part region maxima summed. Mixed-quant
    /// layers pad their smaller parts inside their regions; reconstruction
    /// reads only the layer's actual bytes.
    pub fn record_bytes(&self) -> usize {
        self.gate.max_per_expert_bytes
            + self.up.max_per_expert_bytes
            + self.down.max_per_expert_bytes
    }

    /// Total bytes all expert records would occupy resident (actual bytes,
    /// no padding): `n_experts * sum_layers(gate + up + down)`.
    pub fn total_expert_bytes(&self) -> u64 {
        let per_layer: u64 = (0..self.n_layers as usize)
            .map(|l| {
                (self.gate.per_expert_bytes[l]
                    + self.up.per_expert_bytes[l]
                    + self.down.per_expert_bytes[l]) as u64
            })
            .sum();
        self.n_experts as u64 * per_layer
    }
}

/// [`WeightStore`] over a GGUF's expert tensors, read through the file's
/// [`gguf::TensorBytes`] source ([`gguf::FileSource`] for a streaming open;
/// a test can attach an in-memory source). `fetch` for one record is 3
/// ranged reads (gate, up, down); for the stacked layout each is a
/// contiguous slice of one tensor.
pub struct GgufExpertStore {
    file:   GgufFile,
    layout: ExpertLayout,
}

impl std::fmt::Debug for GgufExpertStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufExpertStore")
            .field("shape", &self.layout)
            .finish()
    }
}

impl PartLayout {
    /// VENDORED-LOCAL: the shard a layer's bytes live in.
    fn shard(&self, layer: usize) -> usize {
        self.shards[layer] as usize
    }
}

impl GgufExpertStore {
    pub fn new(file: GgufFile, layout: ExpertLayout) -> crate::Result<Self> {
        // VENDORED-LOCAL: every shard must be source-backed, not just the
        // primary: on a split GGUF the experts are spread across all of them.
        for shard in 0..file.n_shards() {
            if !file.shard_is_source_backed(shard) {
                return Err(crate::LlamaError::Config(format!(
                    "GgufExpertStore needs a TensorBytes source for every shard \
                     (GgufFile::open_streaming, or an explicit source); shard {shard} \
                     of {} has none",
                    file.n_shards()
                )));
            }
        }
        Ok(Self { file, layout })
    }

    pub fn layout(&self) -> &ExpertLayout {
        &self.layout
    }

    /// Batch fetch: fill `bufs[i]` with the record for `keys[i]`. Every key
    /// and buffer is validated before any I/O, so a bad entry fails the
    /// batch without side effects and names its position.
    ///
    // VENDORED-LOCAL (Wave C).
    pub fn fetch_many(&self, keys: &[(u32, u32)], bufs: &mut [Vec<u8>]) -> nrob::Result<()> {
        if keys.len() != bufs.len() {
            return Err(Error::Arg(format!(
                "fetch_many: {} keys but {} buffers",
                keys.len(),
                bufs.len()
            )));
        }
        let rec = self.record_bytes();
        for (i, (&(layer, expert), buf)) in keys.iter().zip(bufs.iter()).enumerate() {
            if layer >= self.layout.n_layers || expert >= self.layout.n_experts {
                return Err(Error::Arg(format!(
                    "fetch_many entry {i}: expert record ({layer}, {expert}) \
                     outside store shape {:?}",
                    self.shape()
                )));
            }
            if buf.len() != rec {
                return Err(Error::Arg(format!(
                    "fetch_many entry {i}: dst is {} bytes, record is {rec}",
                    buf.len()
                )));
            }
        }
        // One positioned read per part, from a small scoped pool so a
        // layer's cold batch reaches real SSD queue depth; a single record
        // is read inline. Chunks are joined in order, so the error returned
        // is the lowest-index failure.
        let workers = keys.len().min(nrob::backend::hardware_concurrency()).min(FETCH_WORKERS);
        if workers < 2 {
            for (&(layer, expert), buf) in keys.iter().zip(bufs.iter_mut()) {
                self.fetch(layer, expert, buf)?;
            }
            return Ok(());
        }
        let chunk = keys.len().div_ceil(workers);
        std::thread::scope(|s| {
            let handles: Vec<_> = keys
                .chunks(chunk)
                .zip(bufs.chunks_mut(chunk))
                .map(|(ks, bs)| {
                    s.spawn(move || -> nrob::Result<()> {
                        for (&(layer, expert), buf) in ks.iter().zip(bs.iter_mut()) {
                            self.fetch(layer, expert, buf)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            let mut first = Ok(());
            for h in handles {
                let r = h.join().unwrap_or_else(|_| {
                    Err(Error::Arg("fetch_many: a read worker panicked".into()))
                });
                if first.is_ok() {
                    first = r;
                }
            }
            first
        })
    }
}

impl WeightStore for GgufExpertStore {
    fn record_bytes(&self) -> usize {
        self.layout.record_bytes()
    }

    fn shape(&self) -> (u32, u32) {
        (self.layout.n_layers, self.layout.n_experts)
    }

    fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> nrob::Result<()> {
        let io_err = |msg: String| {
            Error::Io(std::io::Error::new(std::io::ErrorKind::Other, msg))
        };
        if layer >= self.layout.n_layers || expert >= self.layout.n_experts {
            return Err(Error::Arg(format!(
                "expert record ({layer}, {expert}) outside store shape {:?}",
                self.shape()
            )));
        }
        if dst.len() != self.record_bytes() {
            return Err(Error::Arg(format!(
                "fetch dst is {} bytes, record is {}",
                dst.len(),
                self.record_bytes()
            )));
        }
        let (l, e) = (layer as usize, expert as usize);
        let g = self.layout.gate.max_per_expert_bytes;
        let u = self.layout.up.max_per_expert_bytes;
        let parts = [
            (&self.layout.gate, 0usize),
            (&self.layout.up, g),
            (&self.layout.down, g + u),
        ];
        for (part, region_off) in parts {
            let (off, len) = part.range(l, e);
            // VENDORED-LOCAL: the source for THIS part's shard.
            let src = self
                .file
                .shard_source_at(part.shard(l))
                .ok_or_else(|| io_err("expert store lost its TensorBytes source".into()))?;
            src.read_range(off, &mut dst[region_off..region_off + len])
                .map_err(|err| {
                    io_err(format!(
                        "expert {e} of layer {l}: ranged read failed: {err}"
                    ))
                })?;
        }
        Ok(())
    }
}

/// Model-level shared state for the streaming backend: one store, one
/// bounded cache, one error slot. All layers share it; the cache keys on
/// `(layer, expert)` so a single budget covers the whole model.
pub struct StreamShared {
    store: GgufExpertStore,
    cache: Ecache,
    /// Resident (non-expert) weight bytes the budget was charged against —
    /// reported so callers can show the resident/cache split.
    resident_est_bytes: u64,
    /// First fetch/reconstruction failure, for the infallible-forward
    /// error channel (see module docs).
    error: Mutex<Option<String>>,
    /// VENDORED-LOCAL (CACHE-02): the VRAM expert cache, when the model
    /// streams on a CUDA backend and `--vram-cache` enabled it. Set once
    /// after open, before generation; read per layer-forward.
    #[cfg(feature = "cuda")]
    device: Mutex<Option<Arc<device_cache::DeviceCache>>>,
}

impl StreamShared {
    /// Build the shared state. `cache_budget_bytes` is the RAM the expert
    /// cache may use (already net of resident weights); it must buy at
    /// least [`MIN_CACHE_RECORDS`] records or the open is refused.
    /// `resident_est_bytes` is carried for reporting only.
    pub fn new(
        store: GgufExpertStore,
        cache_budget_bytes: usize,
        resident_est_bytes: u64,
    ) -> crate::Result<Arc<Self>> {
        let rec = store.record_bytes();
        if rec == 0 {
            return Err(crate::LlamaError::Config(
                "expert record size is 0 bytes".into(),
            ));
        }
        if cache_budget_bytes < rec * MIN_CACHE_RECORDS {
            return Err(crate::LlamaError::Config(format!(
                "expert cache budget {cache_budget_bytes} buys fewer than \
                 {MIN_CACHE_RECORDS} records of {rec} bytes; raise --budget \
                 (the model's experts alone need {} bytes resident, streaming \
                 exists to avoid that)",
                store.layout().total_expert_bytes()
            )));
        }
        Ok(Arc::new(Self {
            store,
            cache: Ecache::new(cache_budget_bytes, rec, nrob::types::CachePolicy::Lfru),
            resident_est_bytes,
            error: Mutex::new(None),
            #[cfg(feature = "cuda")]
            device: Mutex::new(None),
        }))
    }

    /// VENDORED-LOCAL (CACHE-02): attach a VRAM expert cache over `backend`.
    /// The model's forward backend must be this same `CudaBackend` — the
    /// entries' device tensors are only zero-copy for the backend that
    /// created them. Call once after open, before generation; without this
    /// the streaming path behaves exactly as before (per-dispatch uploads).
    #[cfg(feature = "cuda")]
    pub fn enable_device_cache(
        &self,
        backend: Arc<ggml_rs_cuda::CudaBackend>,
        budget_bytes: usize,
    ) -> Result<(), String> {
        let dc = device_cache::DeviceCache::new(backend, budget_bytes, self.store.record_bytes())?;
        *self.device.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(dc));
        Ok(())
    }

    /// Test hook: attach an already-built cache (lets tests force the
    /// staging order via `DeviceCache::set_eager`).
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn enable_device_cache_with(&self, dc: Arc<device_cache::DeviceCache>) {
        *self.device.lock().unwrap_or_else(|e| e.into_inner()) = Some(dc);
    }

    /// The VRAM expert cache, if enabled (CACHE-02).
    #[cfg(feature = "cuda")]
    pub fn device_cache(&self) -> Option<Arc<device_cache::DeviceCache>> {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Device-cache counters; `None` when no VRAM cache is attached.
    #[cfg(feature = "cuda")]
    pub fn device_cache_stats(&self) -> Option<device_cache::DeviceCacheStats> {
        self.device_cache().map(|dc| dc.stats())
    }

    /// Per-layer handle for a block's `MoeFfn`.
    pub fn layer(self: &Arc<Self>, layer: u32) -> LayerStream {
        LayerStream {
            shared: Arc::clone(self),
            layer,
        }
    }

    pub fn cache_stats(&self) -> nrob::ecache::CacheStats {
        self.cache.stats()
    }

    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    pub fn record_bytes(&self) -> usize {
        self.store.record_bytes()
    }

    /// Bytes the RAM budget was charged for resident (non-expert) weights.
    pub fn resident_est_bytes(&self) -> u64 {
        self.resident_est_bytes
    }

    /// Bytes the expert cache was sized with (budget − resident estimate).
    pub fn cache_budget_bytes(&self) -> usize {
        self.cache.budget_bytes()
    }

    pub fn n_experts(&self) -> usize {
        self.store.layout().n_experts as usize
    }

    /// The first streaming error, if any (sticky).
    pub fn error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn record_error(&self, msg: String) {
        let mut g = self.error.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some(msg);
        }
    }
}

/// Per-layer streaming handle carried by a `MoeFfn`. Dispatches routed
/// experts through the shared cache and runs the same packed-matvec ops as
/// the resident path.
pub struct LayerStream {
    shared: Arc<StreamShared>,
    layer:  u32,
}

impl std::fmt::Debug for LayerStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerStream")
            .field("layer", &self.layer)
            .field("record_bytes", &self.shared.record_bytes())
            .finish()
    }
}

impl LayerStream {
    pub fn n_experts(&self) -> usize {
        self.shared.n_experts()
    }

    /// Fetch expert `e`'s record through the cache and rebuild the
    /// `(FfnPair, down Weight)` pair the resident loader would hold for it.
    ///
    // VENDORED-LOCAL (CACHE-01): `Ecache::acquire` leases the resident
    // record instead of `get` copying 3 MB into a fresh Vec on every
    // dispatch. The lease moves into the `RecordView`s below, so the pinned
    // buffer outlives the packed matvec; reconstruction stays zero-copy.
    // VENDORED-LOCAL: pub(crate) so glm5next's device path can reuse the
    // cached reconstruction instead of re-reading and re-uploading per token.
    pub(crate) fn expert_weights(&self, e: u32) -> Result<(FfnPair, Weight), String> {
        let layout = self.shared.store.layout();
        let l = self.layer as usize;
        let lease = self
            .shared
            .cache
            .acquire(self.layer, e, &self.shared.store)
            .map_err(|err| format!("expert {e} of layer {}: {err}", self.layer))?;

        let g = layout.gate.max_per_expert_bytes;
        let u = layout.up.max_per_expert_bytes;
        let glen = layout.gate.per_expert_bytes[l];
        let ulen = layout.up.per_expert_bytes[l];

        // VENDORED-LOCAL (LAYOUT-01): fused gate/up as a VIEW, not a copy.
        // The record is gate_region || up_region || down_region; when this
        // layer's gate bytes fill the gate region exactly (glen == g — true
        // for uniform-quant models like Qwen3-MoE, where only `down` varies
        // between Q4_K/Q6_K) the bytes at record[0 .. glen+ulen] ARE the
        // fused gate||up tensor: same dtype and row geometry means the
        // axis-0 stack is a plain concatenation with doubled rows, which is
        // exactly what Weight::stack_axis0 would allocate and copy. Build
        // the packed view directly over the lease instead.
        let fused_view = glen == g
            && layout.gate.dtypes[l] == layout.up.dtypes[l]
            && layout.gate.shape == layout.up.shape
            && dtype_supports_packed_matmul(layout.gate.dtypes[l]);
        let pair = if fused_view {
            let view = RecordView {
                lease: lease.clone(),
                start: 0,
                len:   glen + ulen,
            };
            let shape = vec![layout.gate.shape[0] + layout.up.shape[0], layout.gate.shape[1]];
            FfnPair::Fused(Weight::Quant(QuantizedTensor::from_mmap(
                Arc::new(view),
                shape,
                layout.gate.dtypes[l],
            )))
        } else {
            // Genuine restack cases only: a mixed-quant layer whose gate
            // region is padded to a larger per-model max (the halves are not
            // contiguous in the record), a dtype/shape mismatch between the
            // halves (from_halves falls back to Split), or a dtype without a
            // packed matvec (dequantize path). Keep the copy here.
            let gate = make_weight(&lease, 0, glen, layout.gate.dtypes[l], layout.gate.shape)?;
            let up = make_weight(&lease, g, ulen, layout.up.dtypes[l], layout.up.shape)?;
            FfnPair::from_halves(gate, up)
        };
        let down = make_weight(
            &lease,
            g + u,
            layout.down.per_expert_bytes[l],
            layout.down.dtypes[l],
            layout.down.shape,
        )?;
        Ok((pair, down))
    }

    /// Host-cache batch prewarm for a layer's routed experts. Probe the
    /// distinct set, fetch ALL host-cache misses in one
    /// [`GgufExpertStore::fetch_many`] batch (3 ranged reads per expert),
    /// then admit each fetched record into the cache.
    ///
    // VENDORED-LOCAL (Wave C): batch admission.
    //
    // Accounting: a cold expert costs one miss (counted by `admit`,
    // previously by the prewarm's own `acquire`) plus one hit per dispatch
    // acquire; the double-acquire hit quirk (prewarm hit + dispatch hit for
    // an already-warm expert) is gone, because `probe` counts nothing —
    // warm experts now cost exactly their dispatch hits.
    //
    // Errors are deliberately dropped here: a failed read admits nothing
    // and re-surfaces when the dispatch loop acquires the same expert and
    // is recorded there, exactly as without the prewarm (the failed slot is
    // not cached, so the loop's acquire retries the read once).
    fn prewarm_host(&self, experts: &[u32]) {
        let misses: Vec<u32> = experts
            .iter()
            .copied()
            .filter(|&e| !self.shared.cache.probe(self.layer, e))
            .collect();
        if misses.is_empty() {
            return;
        }
        let rec = self.shared.store.record_bytes();
        let keys: Vec<(u32, u32)> = misses.iter().map(|&e| (self.layer, e)).collect();
        let mut bufs: Vec<Vec<u8>> = (0..misses.len()).map(|_| vec![0u8; rec]).collect();
        if self.shared.store.fetch_many(&keys, &mut bufs).is_ok() {
            for ((l, e), buf) in keys.iter().copied().zip(bufs) {
                // Cannot fail (the buffer is the store's own record
                // size); a racing dispatch may have admitted the
                // same key meanwhile — admit keeps one copy
                // (identical bytes, documented on Ecache::admit_owned).
                // admit_owned moves the fetch buffer into the slot:
                // no rec_bytes copy on the miss path.
                let _ = self.shared.cache.admit_owned(l, e, buf);
            }
        }
    }

    /// Streaming analogue of [`crate::moe::moe_forward_with_logits`]: same
    /// routing math, same backend ops; only the expert *storage* differs.
    /// On a fetch error the first failure is recorded in the shared slot and
    /// this layer's output is zeros — the caller is expected to notice via
    /// [`StreamShared::error`] and abort the generation.
    ///
    // VENDORED-LOCAL: MOE-01/MOE-02 — routing runs on-device where the
    // backend supports it; only the compact ids mailbox (top_k u32s per
    // token) comes back D2H, as the miss list for staging. At decode
    // (seq=1), when every routed expert resolves to a device-cache entry
    // whose dtype the grouped kernels cover, the whole layer runs as
    // route → gate_up+act → down+scale → reduce (4 launches + the mailbox),
    // instead of ~4 ops per (token, expert). Everything else falls through
    // to the reference per-expert loop.
    pub fn forward_with_logits(
        &self,
        backend:       &dyn Backend,
        x:             &Tensor,
        router_logits: &Tensor,
        top_k:         usize,
        opts:          &MoeOptions<'_>,
    ) -> Tensor {
        let seq = x.dim(0);
        let hidden = x.dim(1);
        let n_experts = self.shared.n_experts();
        let top_k = top_k.min(n_experts);

        let mut output = backend.alloc_zeros(vec![seq, hidden]);

        // Route. CUDA: ids+weights computed on device (`dev_routing`); the
        // ids mailbox is the staging miss list. Host weights are pulled
        // lazily — only the reference dispatch loop needs them.
        #[cfg(feature = "cuda")]
        let cuda = backend
            .as_any()
            .downcast_ref::<ggml_rs_cuda::CudaBackend>();
        #[cfg(feature = "cuda")]
        let dev_routing =
            cuda.and_then(|c| c.moe_route_device(router_logits, top_k));
        #[cfg(feature = "cuda")]
        let (ids_flat, mut w_flat): (Vec<u32>, Option<Vec<f32>>) = match &dev_routing {
            Some(r) => (r.ids_to_host(), None),
            None => {
                let (i, w) = backend.moe_route_topk(router_logits, top_k);
                (i, Some(w))
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (ids_flat, w_flat): (Vec<u32>, Option<Vec<f32>>) = {
            let (i, w) = backend.moe_route_topk(router_logits, top_k);
            (i, Some(w))
        };

        // Distinct routed experts of the whole layer (every token): the
        // unit both prewarm and device staging work on.
        let mut distinct = ids_flat.clone();
        distinct.sort_unstable();
        distinct.dedup();

        // VENDORED-LOCAL (CACHE-02 / STREAM-01): VRAM expert cache. With a
        // device cache attached, routed experts resolve to device-resident
        // tensors: a HIT launches the matvec against VRAM with zero H2D
        // bytes; a MISS is staged through the pinned ring onto the transfer
        // stream and admitted, then its ticket is waited once before the
        // first matvec that reads it. Without one (CPU backend, or no
        // --vram-cache) the flow is exactly the pre-CACHE-02 one.
        #[cfg(feature = "cuda")]
        let mut dev: Option<DeviceDispatch> = match self.shared.device_cache() {
            Some(dc) => {
                let mut d = DeviceDispatch::new(dc);
                if d.eager() {
                    // Whole-layer prefetch: look every routed expert up
                    // (counting the hits/misses), host-prewarm ONLY the
                    // device misses in one batched read, then stage every
                    // miss's upload on the transfer stream BEFORE the
                    // layer's first matvec. Compute below walks the routed
                    // order waiting one upload ticket per expert — never a
                    // device-wide sync.
                    let misses = d.lookup_all(self.layer, &distinct);
                    self.prewarm_host(&misses);
                    if let Err(msg) = d.stage_misses(self, &misses) {
                        self.shared.record_error(msg);
                        return output;
                    }
                } else {
                    // Lazy order: each miss uploads right before its first
                    // compute (the classic i / i+1 pipeline). Host records
                    // are still batch-prewarmed; only the H2D timing differs.
                    self.prewarm_host(&distinct);
                }
                Some(d)
            }
            None => {
                self.prewarm_host(&distinct);
                None
            }
        };
        #[cfg(not(feature = "cuda"))]
        self.prewarm_host(&distinct);

        // MOE-02 grouped fast path: single-token decode, SwiGLU without
        // per-expert output scales, every routed expert device-resident and
        // kernel-eligible → one grouped call for the whole layer.
        #[cfg(feature = "cuda")]
        if seq == 1
            && !opts.use_gelu
            && opts.down_exps_scale_host.is_none()
            && crate::moe_cuda::grouped_enabled()
        {
            if let (Some(c), Some(r), Some(d)) = (cuda, &dev_routing, dev.as_mut()) {
                match self.try_grouped_cuda(c, r, d, x, &ids_flat, hidden) {
                    Ok(Some(out)) => return out,
                    Ok(None) => {} // not eligible — reference loop below
                    Err(msg) => {
                        self.shared.record_error(msg);
                        return output;
                    }
                }
            }
        }

        // Reference per-(token, expert) dispatch. Host routing weights are
        // needed here; pull the weights mailbox when routing ran on device.
        #[cfg(feature = "cuda")]
        if w_flat.is_none() {
            w_flat = Some(
                dev_routing
                    .as_ref()
                    .expect("device routing present when weights not yet pulled")
                    .weights_to_host(),
            );
        }
        let w_flat = w_flat.expect("routing weights available");

        for t in 0..seq {
            let indices = &ids_flat[t * top_k..(t + 1) * top_k];
            let weights = &w_flat[t * top_k..(t + 1) * top_k];
            let xt = backend.slice_axis0_range(x, t, 1);

            for (idx, w) in indices.iter().zip(weights.iter()) {
                let idx = *idx as usize;
                let e = idx as u32;
                let host_w;
                #[cfg(feature = "cuda")]
                let dev_w = match dev.as_mut() {
                    Some(d) => match d.resolve(self, e) {
                        Ok(v) => v,
                        Err(msg) => {
                            self.shared.record_error(msg);
                            return output; // zeros from this token on; caller checks error()
                        }
                    },
                    None => None,
                };
                #[cfg(feature = "cuda")]
                let dev_pd = dev_w.as_ref().map(|en| (&en.pair, &en.down));
                #[cfg(not(feature = "cuda"))]
                let dev_pd: Option<(&FfnPair, &Weight)> = None;
                let (pair, down) = match dev_pd {
                    Some((p, d)) => (p, d),
                    None => {
                        host_w = match self.expert_weights(e) {
                            Ok(v) => v,
                            Err(msg) => {
                                self.shared.record_error(msg);
                                return output; // zeros from this token on; caller checks error()
                            }
                        };
                        (&host_w.0, &host_w.1)
                    }
                };
                let activated = if opts.use_gelu {
                    pair.geglu(backend, &xt)
                } else {
                    pair.swiglu(backend, &xt)
                };
                let expert_out = down.linear(backend, &activated);
                let combined = match opts.down_exps_scale_host {
                    Some(scales) => *w * scales[idx],
                    None => *w,
                };
                backend.add_to_axis0_range_scaled(&mut output, t, 1, &expert_out, combined);
            }
        }
        output
    }

    /// MOE-02 grouped streaming dispatch. Resolves every routed expert to a
    /// device-cache entry (staging lazy misses on the way), builds the
    /// layer's pointer table over them — one small H2D covering all k
    /// experts — and runs the grouped kernels against the device-resident
    /// routing. `Ok(None)` = some expert can't participate (host fallback
    /// or kernel-ineligible dtype/geometry); the caller runs the reference
    /// loop, which reuses the memoized leases.
    ///
    /// The pointer table is sized `2 * n_experts` (gate_up | down, indexed
    /// by expert id) with only the k routed entries filled — the kernels
    /// only ever read routed entries, and one flat upload beats k small
    /// ones on WDDM.
    #[cfg(feature = "cuda")]
    fn try_grouped_cuda(
        &self,
        cuda: &ggml_rs_cuda::CudaBackend,
        routing: &ggml_rs_cuda::MoeRoutingDevice,
        d: &mut DeviceDispatch,
        x: &Tensor,
        ids: &[u32],
        hidden: usize,
    ) -> Result<Option<Tensor>, String> {
        let n_experts = self.shared.n_experts();
        let mut gu_tab = vec![0u64; n_experts];
        let mut dn_tab = vec![0u64; n_experts];
        let mut gu_dt: Option<GgmlType> = None;
        let mut dn_dt: Option<GgmlType> = None;
        let mut ff = 0usize;
        // Keep the entry Arcs alive until after the launches are enqueued:
        // DeviceEntry::drop orders its frees behind the compute stream's
        // queued work (the PERF-02 discipline), so dropping the leases only
        // after `moe_grouped_ffn` returns keeps the read→free edge correct.
        let mut entries = Vec::with_capacity(ids.len());
        for &e in ids {
            let en = match d.resolve(self, e)? {
                Some(en) => en,
                None => return Ok(None), // host-path expert — not groupable
            };
            let crate::loader::FfnPair::Fused(Weight::Quant(gu)) = &en.pair else {
                return Ok(None);
            };
            let Weight::Quant(dn) = &en.down else { return Ok(None) };
            if gu.shape().len() != 2 || dn.shape().len() != 2 || gu.dim(0) % 2 != 0 {
                return Ok(None);
            }
            let f = gu.dim(0) / 2;
            let h = gu.dim(1);
            if h != hidden || dn.dim(0) != h || dn.dim(1) != f {
                return Ok(None);
            }
            if ff == 0 {
                ff = f;
            } else if ff != f {
                return Ok(None);
            }
            match gu_dt {
                Some(dt) if dt != gu.dtype() => return Ok(None),
                None => gu_dt = Some(gu.dtype()),
                _ => {}
            }
            match dn_dt {
                Some(dt) if dt != dn.dtype() => return Ok(None),
                None => dn_dt = Some(dn.dtype()),
                _ => {}
            }
            let (Some(gup), Some(dnp)) = (
                ggml_rs_cuda::quant_device_ptr(gu),
                ggml_rs_cuda::quant_device_ptr(dn),
            ) else {
                return Ok(None);
            };
            gu_tab[e as usize] = gup;
            dn_tab[e as usize] = dnp;
            entries.push(en);
        }
        let (Some(gu_dt), Some(dn_dt)) = (gu_dt, dn_dt) else {
            return Ok(None);
        };
        if !ggml_rs_cuda::grouped_kernel_covers(gu_dt, hidden)
            || !ggml_rs_cuda::grouped_kernel_covers(dn_dt, ff)
        {
            return Ok(None);
        }
        let plan = ggml_rs_cuda::MoeDevicePlan::new(
            cuda, &gu_tab, &dn_tab, None, gu_dt, dn_dt, ff, hidden, false,
        );
        let out = cuda.moe_grouped_ffn(x, &plan, routing);
        drop(entries);
        Ok(Some(out))
    }
}

/// VENDORED-LOCAL (CACHE-02 / STREAM-01): per-layer-forward device dispatch
/// state. Holds the VRAM cache plus this forward's leases: an entry looked
/// up or staged once is reused for every token routed to the same expert in
/// this forward, so the cache sees each distinct expert at most once per
/// layer-forward.
#[cfg(feature = "cuda")]
struct DeviceDispatch {
    dc: Arc<device_cache::DeviceCache>,
    leases: std::collections::HashMap<u32, Arc<device_cache::DeviceEntry>>,
}

#[cfg(feature = "cuda")]
impl DeviceDispatch {
    fn new(dc: Arc<device_cache::DeviceCache>) -> Self {
        Self {
            dc,
            leases: std::collections::HashMap::new(),
        }
    }

    fn eager(&self) -> bool {
        self.dc.is_eager()
    }

    /// Look every routed expert up, memoizing hits; return the misses.
    fn lookup_all(&mut self, layer: u32, experts: &[u32]) -> Vec<u32> {
        let mut misses = Vec::new();
        for &e in experts {
            match self.dc.acquire(layer, e) {
                Some(en) => {
                    self.leases.insert(e, en);
                }
                None => misses.push(e),
            }
        }
        misses
    }

    /// Host-acquire expert `e`'s record (a host-cache hit after prewarm,
    /// else a fetch) and stage its upload into the device cache. Staging
    /// failures (pinned-alloc errors) are not fatal: the expert is left out
    /// of the lease map and the dispatch loop falls back to the host path
    /// for it. Experts whose dtype needs the dequantize fallback stay on
    /// the host path by design.
    fn stage_one(&mut self, ls: &LayerStream, e: u32) -> Result<(), String> {
        let lease = ls
            .shared
            .cache
            .acquire(ls.layer, e, &ls.shared.store)
            .map_err(|err| format!("expert {e} of layer {}: {err}", ls.layer))?;
        let plan = record_plan(ls.shared.store.layout(), ls.layer as usize);
        if !plan.packed {
            return Ok(());
        }
        match self.dc.stage_and_admit(ls.layer, e, &lease, &plan) {
            Ok(en) => {
                self.leases.insert(e, en);
            }
            Err(_) => self.dc.note_stage_failure(),
        }
        Ok(())
    }

    /// Eager half of STREAM-01: stage every miss's upload before the
    /// layer's first matvec.
    fn stage_misses(&mut self, ls: &LayerStream, misses: &[u32]) -> Result<(), String> {
        for &e in misses {
            self.stage_one(ls, e)?;
        }
        Ok(())
    }

    /// Resolve expert `e` for compute: memoized lease, cache hit, or (lazy
    /// mode / an expert eager staging skipped) a stage on first use. The
    /// entry's upload tickets are waited once here, before its first
    /// matvec. `Ok(None)` = use the host path for this expert.
    fn resolve(
        &mut self,
        ls: &LayerStream,
        e: u32,
    ) -> Result<Option<Arc<device_cache::DeviceEntry>>, String> {
        if let Some(en) = self.leases.get(&e) {
            self.dc.ensure_ready(en);
            return Ok(Some(Arc::clone(en)));
        }
        if let Some(en) = self.dc.acquire(ls.layer, e) {
            self.dc.ensure_ready(&en);
            self.leases.insert(e, Arc::clone(&en));
            return Ok(Some(en));
        }
        self.stage_one(ls, e)?;
        match self.leases.get(&e) {
            Some(en) => {
                self.dc.ensure_ready(en);
                Ok(Some(Arc::clone(en)))
            }
            None => Ok(None),
        }
    }
}

/// Rebuild one `Weight` from a record byte range, mirroring
/// [`Weight::load`]'s packed-vs-dense decision: packed-matmul dtypes stay
/// packed (zero-copy view over the leased record), everything else
/// dequantizes to F32 — for a record of one expert, a bounded transient.
///
// VENDORED-LOCAL (CACHE-01): takes the cache lease, not an `Arc<Vec<u8>>`.
fn make_weight(
    rec: &nrob::ecache::HostLease,
    start: usize,
    len: usize,
    dtype: GgmlType,
    shape: [usize; 2],
) -> Result<Weight, String> {
    if dtype_supports_packed_matmul(dtype) {
        let view = RecordView { lease: rec.clone(), start, len };
        return Ok(Weight::Quant(QuantizedTensor::from_mmap(
            Arc::new(view),
            shape.to_vec(),
            dtype,
        )));
    }
    let numel = shape[0] * shape[1];
    let mut out = vec![0.0f32; numel];
    ggml_quants::dequantize(dtype, &rec[start..start + len], &mut out)
        .map_err(|e| format!("dequantize {dtype:?} expert slice: {e}"))?;
    Ok(Weight::Dense(Tensor::from_vec(out, shape.to_vec())))
}

/// Per-layer plan for rebuilding an expert's `(FfnPair, down Weight)` from
/// its record — the lengths/dtypes/shapes both the host reconstruction
/// ([`LayerStream::expert_weights`]) and the device staging path
/// ([`device_cache::DeviceCache::stage_and_admit`]) must agree on.
///
/// `fused` mirrors the host path's fused-view decision exactly: the bytes at
/// `record[0 .. glen+ulen]` ARE the fused gate‖up tensor when the gate
/// region is unpadded, the halves share dtype and shape, and the dtype has a
/// packed matvec. `packed` is false when any part needs the dequantize
/// fallback — such experts stay on the host path (never staged to VRAM).
///
// VENDORED-LOCAL (CACHE-02).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub(crate) struct RecordPlan {
    pub fused: bool,
    pub packed: bool,
    /// (record offset, byte len, dtype, shape) per part.
    pub gate: (usize, usize, GgmlType, [usize; 2]),
    pub up: (usize, usize, GgmlType, [usize; 2]),
    pub down: (usize, usize, GgmlType, [usize; 2]),
}

#[cfg(feature = "cuda")]
pub(crate) fn record_plan(layout: &ExpertLayout, l: usize) -> RecordPlan {
    let g = layout.gate.max_per_expert_bytes;
    let u = layout.up.max_per_expert_bytes;
    let (glen, ulen, dlen) = (
        layout.gate.per_expert_bytes[l],
        layout.up.per_expert_bytes[l],
        layout.down.per_expert_bytes[l],
    );
    let (gd, ud, dd) = (layout.gate.dtypes[l], layout.up.dtypes[l], layout.down.dtypes[l]);
    let fused = glen == g
        && gd == ud
        && layout.gate.shape == layout.up.shape
        && dtype_supports_packed_matmul(gd);
    let packed = dtype_supports_packed_matmul(gd)
        && dtype_supports_packed_matmul(ud)
        && dtype_supports_packed_matmul(dd);
    RecordPlan {
        fused,
        packed,
        gate: (0, glen, gd, layout.gate.shape),
        up: (g, ulen, ud, layout.up.shape),
        down: (g + u, dlen, dd, layout.down.shape),
    }
}

#[cfg(feature = "cuda")]
pub mod device_cache;

#[cfg(test)]
mod tests;

impl crate::Model {
    /// VENDORED-LOCAL: open a GGUF MoE model with **streaming experts**.
    ///
    /// Non-expert weights (attention, norms, embeddings, routers) load
    /// resident as in [`crate::Model::load`]; expert tensors are never
    /// materialized: routed experts are read on demand from the `.gguf`
    /// file ([`GgufFile::open_streaming`], positioned reads) through a
    /// bounded [`nrob::ecache::Ecache`].
    ///
    /// `ram_budget_bytes` covers resident weights + expert cache: the cache
    /// is sized as `budget − resident_estimate`, where the estimate is
    /// (total tensor bytes − expert tensor bytes). Refused when the
    /// remainder buys fewer than `MIN_CACHE_RECORDS` records. The resident
    /// estimate excludes KV cache and activations, which are small at
    /// decode-scale contexts.
    pub fn open_streaming(
        gguf_path: impl AsRef<std::path::Path>,
        backend: Arc<dyn Backend>,
        ram_budget_bytes: u64,
    ) -> crate::Result<Self> {
        use crate::config::{Architecture, ModelConfig};
        use crate::{Glm5NextModel, LlamaError, MixtralModel, Qwen3MoeModel};
        let g = GgufFile::open_streaming(gguf_path)?;
        let arch_str = g.architecture()?.to_string();
        let arch = Architecture::from_str(&arch_str);

        let is_qwen3moe = matches!(arch, Architecture::Qwen3Moe | Architecture::Qwen3VlMoe);
        // VENDORED-LOCAL: glm5next streams too, but its first 3 blocks are dense.
        let is_glm5next = matches!(arch, Architecture::Glm5Next);
        let is_mixtral = matches!(
            arch,
            Architecture::Llama | Architecture::Mistral | Architecture::Qwen2
        ) && g.get_u64(&format!("{arch_str}.expert_count")).unwrap_or(0) > 0;
        if !is_qwen3moe && !is_mixtral && !is_glm5next {
            return Err(LlamaError::UnsupportedArch(format!(
                "{arch_str}: streaming experts are implemented for qwen3moe and \
                 mixtral-family and glm5next MoE models only"
            )));
        }

        let config = ModelConfig::from_gguf(&g)?;
        let n_experts = g
            .get_u64(&format!("{arch_str}.expert_count"))
            .map(|v| v as usize)
            .map_err(|_| LlamaError::Config(format!("missing {arch_str}.expert_count")))?;

        // VENDORED-LOCAL: glm5next's experts start at `leading_dense_block_count`,
        // so the layout covers blocks first_moe..n_layers, not 0..n_layers. Every
        // other arch here is MoE from block 0 and keeps the unoffset call.
        let first_moe = if is_glm5next {
            g.get_u64(&format!("{arch_str}.leading_dense_block_count")).unwrap_or(0) as usize
        } else {
            0
        };
        if first_moe >= config.n_layers {
            return Err(LlamaError::Config(format!(
                "leading_dense_block_count ({first_moe}) leaves no MoE blocks in {}",
                config.n_layers
            )));
        }
        let n_moe_layers = config.n_layers - first_moe;
        let layout = ExpertLayout::resolve_range(&g, first_moe, n_moe_layers, n_experts)?;
        let total_tensor_bytes: u64 = g.tensors().iter().map(|t| t.nbytes()).sum();
        let resident_est = total_tensor_bytes.saturating_sub(layout.total_expert_bytes());
        if ram_budget_bytes <= resident_est {
            return Err(LlamaError::Config(format!(
                "--budget {ram_budget_bytes} does not cover the resident \
                 (non-expert) weights, ~{resident_est} bytes; the expert cache \
                 needs room on top of that"
            )));
        }
        let cache_budget = (ram_budget_bytes - resident_est) as usize;
        let store = GgufExpertStore::new(g.clone(), layout)?;
        let shared = StreamShared::new(store, cache_budget, resident_est)?;

        if is_glm5next {
            Ok(Self::Glm5Next(Glm5NextModel::from_gguf_streaming(&g, backend, shared)?))
        } else if is_qwen3moe {
            Ok(Self::Qwen3Moe(Qwen3MoeModel::from_gguf_streaming(&g, backend, shared)?))
        } else {
            Ok(Self::Mixtral(MixtralModel::from_gguf_streaming(&g, backend, shared)?))
        }
    }
}
