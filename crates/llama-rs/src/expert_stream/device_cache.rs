//! VENDORED-LOCAL (CACHE-02 / STREAM-01): per-device, byte-budgeted VRAM
//! expert cache — the roadmap's "single most important streaming feature"
//! (docs/ROADMAP.md §8, Phase 2).
//!
//! Without this cache every expert dispatch uploads the packed record to the
//! GPU again (`cuda_quant_input` in ggml-rs-cuda): a host-cache hit removes
//! the disk read but still pays the PCIe transfer. A hit here removes the
//! transfer too — the entry's [`QuantizedTensor`] is already device-resident
//! (`QuantizedStorage::Device` wrapping `CudaQuantStorage`), so
//! `cuda_quant_input` takes its `Borrowed` arm and the matvec launches with
//! **zero H2D bytes**.
//!
//! # Keying and layout
//!
//! Entries are keyed `(layer, expert)`. The roadmap's wider key
//! (`model_id, device, dtype, layout_generation`) is constant for the life of
//! one cache: a cache is built per model open, per `CudaBackend` (one device),
//! and stores kernel-ready packed bytes in exactly the layout the matvec
//! consumes — never a transformed intermediate. Multi-GPU is one cache per
//! device (the cache holds an `Arc<CudaBackend>`; nothing here is global).
//!
//! One admitted expert holds its fused gate‖up view (or split gate/up pair)
//! and its down view as separate `QuantizedTensor`s, each backed by ONE
//! device allocation made by `CudaBackend::upload_quantized_async`. (The
//! roadmap's "one device allocation per record with subviews" would need a
//! typed-subview constructor over a shared `CudaSlice`, which the FINAL
//! transfer API does not expose; two/three allocations per expert carry the
//! same bytes and the same kernel-ready layout, so the observable contract —
//! one upload, zero copies on hit — is unchanged.)
//!
//! # Eviction safety
//!
//! Eviction is LFRU mirroring the host [`Ecache`](nrob::ecache) policy:
//! frequency first, recency (`last` clock) as tiebreak. The host cache
//! samples 16 slots per eviction; here entries number in the hundreds
//! (budget ÷ ~3 MiB record), so a full scan picks the exact LFRU victim at
//! negligible cost. A victim is only chosen among entries whose
//! `Arc::strong_count == 1` — i.e. the cache holds the only reference. An
//! entry a dispatch is still computing with (the per-forward lease map holds
//! a clone) is never evicted, exactly the host cache's live-lease rule.
//!
//! The harder question is eviction while a *launched but unfinished* kernel
//! still reads the entry. cudarc's per-slice event tracking — which would
//! defer the free automatically — is disabled context-wide for launch-enqueue
//! speed (PERF-02 in ggml-rs-cuda `backend.rs`), so the read→free edge is
//! explicit instead: [`DeviceEntry::drop`] calls
//! `CudaBackend::order_transfer_after_compute`, recording the compute
//! stream's enqueued work and making the H2D transfer stream (on which the
//! entry's `CudaSlice`s free via `cuMemFreeAsync`) wait for it. Every kernel
//! that reads these bytes is enqueued before the last lease is dropped, so
//! the recorded point covers them all — "drop is safe mid-compute". The
//! write side is unchanged: upload tickets order the compute stream after
//! the H2D copy before first use. The same argument covers eviction with an
//! upload still in flight (the entry is dropped with its ticket unwatched —
//! the free is stream-ordered after the copy on the same H2D stream).
//!
//! # Overlap (STREAM-01)
//!
//! On a miss the record is staged through the pinned ring and uploaded with
//! `upload_quantized_async` on the backend's H2D transfer stream; the
//! completion event lands in the entry's ticket list. `ensure_ready` calls
//! `wait_upload` once per ticket before the entry's first matvec — a
//! `cuStreamWaitEvent` on the compute stream, never a device-wide sync and
//! never a host block. Two staging orders exist (see
//! [`DeviceCache::is_eager`]):
//!
//! * **eager** (default): after routing, ALL of the layer's misses are
//!   staged before the first matvec of the layer, then compute walks the
//!   routed order waiting one ticket per expert. This is the roadmap's
//!   "enqueue all misses' uploads, then compute in order" and also its
//!   "prefetch the whole layer's misses, then compute" — for a whole layer
//!   batch the two coincide.
//! * **lazy** (`NROB_VRAM_UPLOAD_MODE=lazy`): each miss is uploaded right
//!   before its first compute, the classic "compute expert i while i+1
//!   transfers" pipeline.
//!
//! Which wins is hardware-dependent: on Windows/WDDM the copy engine is
//! throttled to ~12% of its rate while compute is busy, so eager (uploads
//! issued while the SMs idle between layers) can beat the theoretically
//! better-overlapped lazy order. Both are kept so the choice is measured,
//! not assumed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ggml_quants::GgmlType;
use ggml_rs::quantized::QuantizedTensor;
use ggml_rs_cuda::{CudaBackend, PinnedPool, UploadTicket};

use crate::loader::{FfnPair, Weight};

use super::RecordPlan;

/// Per-run counters for the device expert cache. `bytes_hit` /
/// `h2d_bytes` are the byte-weighted demand stream: a hit serves
/// `entry.bytes` from VRAM, a miss uploaded `h2d_bytes` over PCIe.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Device-resident bytes served on hits.
    pub bytes_hit: u64,
    /// Bytes uploaded H2D on misses (the miss half of the byte stream).
    pub h2d_bytes: u64,
    pub evictions: u64,
    /// Host time inside `ensure_ready`. `wait_upload` is a stream-level
    /// enqueue (`cuStreamWaitEvent`), not a host block, so this stays ~0 by
    /// design; real stalls show up in the inter-token timings, not here.
    pub wait_ns: u64,
    /// `ensure_ready` calls that found the upload already complete
    /// (overlap worked: the transfer finished before its expert was needed).
    pub waits_free: u64,
    /// `ensure_ready` calls that had to order the compute stream behind a
    /// still-in-flight upload (the GPU will stall on that expert).
    pub waits_pending: u64,
    /// Staging failures fallen back to the host path (pinned-alloc errors).
    pub stage_failures: u64,
    pub entries: u64,
    pub bytes_used: u64,
    pub budget_bytes: u64,
}

// VENDORED-LOCAL: GLM-5.3-Flash, from `dsv41-cuda/src/expert_cache.rs`.
/// Every frequency halves after this many decode tokens.
///
/// LFU without aging keeps an old topic's experts in VRAM forever and never lets
/// a new topic's collect enough uses to displace them. Counted in tokens rather
/// than accesses on purpose: a prefill touches nearly every expert and would age
/// the counts away.
const AGE_TOKENS: u64 = 128;

/// One cached expert: kernel-ready device tensors plus the upload tickets
/// that must be waited before first compute use.
///
/// `pair` / `down` mirror what [`super::LayerStream::expert_weights`] builds
/// host-side from the same record bytes, so the rest of the dispatch loop is
/// literally identical between the host and device paths.
///
/// `backend` is kept so [`DeviceEntry::drop`] can order the free: cudarc's
/// per-slice event tracking is disabled context-wide (PERF-02 in ggml-rs-cuda
/// `backend.rs`), so the read→free edge for these H2D-written, compute-read
/// bytes is explicit — drop records the compute stream's work so far and
/// makes the transfer stream (on which the tensors' `CudaSlice`s free) wait
/// for it before the fields' `cuMemFreeAsync` runs.
pub struct DeviceEntry {
    pub(crate) pair: FfnPair,
    pub(crate) down: Weight,
    /// Device bytes charged against the budget (sum of the tensor storages).
    bytes: usize,
    /// Completion events of this entry's uploads; drained by the first
    /// [`DeviceCache::ensure_ready`]. A Mutex, not a Cell: the entry is
    /// shared through `Arc` and `Send+Sync` must hold by construction.
    tickets: Mutex<Vec<UploadTicket>>,
    /// LFRU frequency term (mutated only under the cache's inner lock).
    // VENDORED-LOCAL: GLM-5.3-Flash. Atomics rather than Cells, so an entry is
    // `Sync` and a `ResolvedExpert` holding one can cross into a scope that also
    // runs CPU experts. Both are only ever touched under the cache's own mutex, so
    // Relaxed is all the ordering they need -- the atomicity is for the type, not
    // for the synchronisation.
    freq: AtomicU32,
    /// LFRU recency term (the cache's logical clock).
    last: AtomicU64,
    /// The backend that owns the streams the tensors were uploaded with;
    /// used only by `drop` for free ordering (see the struct docs).
    backend: Arc<CudaBackend>,
}

impl Drop for DeviceEntry {
    fn drop(&mut self) {
        // VENDORED-LOCAL (PERF-02): explicit read→free ordering, replacing
        // the cudarc event-tracking deferred free this used to rely on.
        // Enqueued after every kernel that could read these bytes (they were
        // all launched before the last lease was dropped), this makes the
        // fields' H2D-stream frees execute after those kernels finish.
        self.backend.order_transfer_after_compute();
    }
}

impl std::fmt::Debug for DeviceEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEntry")
            .field("bytes", &self.bytes)
            .field("pending_uploads", &self.tickets.lock().map(|t| t.len()).unwrap_or(0))
            .finish()
    }
}

/// Per-device VRAM expert cache. See the module docs for the design.
pub struct DeviceCache {
    backend: Arc<CudaBackend>,
    /// Pinned staging ring: 3 buffers of one record each (the roadmap's
    /// 2–3 reusable pinned slots). Checkout beyond that grows the ring once,
    /// never per token.
    pool: PinnedPool,
    budget: usize,
    /// Staging order — see the module docs. Read once from
    /// `NROB_VRAM_UPLOAD_MODE` (`eager` default, `lazy` opt-in).
    eager: bool,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    bytes_hit: AtomicU64,
    h2d_bytes: AtomicU64,
    evictions: AtomicU64,
    wait_ns: AtomicU64,
    waits_free: AtomicU64,
    waits_pending: AtomicU64,
    stage_failures: AtomicU64,
}

struct Inner {
    map: HashMap<(u32, u32), Arc<DeviceEntry>>,
    bytes_used: usize,
    /// Logical clock for the LFRU recency term.
    clock: u64,
    // VENDORED-LOCAL: GLM-5.3-Flash. Aging, and frequencies that outlive
    // eviction -- both from `dsv41-cuda/src/expert_cache.rs`.
    /// Access counts that survive eviction, so a hot expert that was squeezed
    /// out is not judged a newcomer when it comes back.
    ///
    /// Without this, an expert evicted under churn returns with `freq = 1`, is
    /// immediately the cheapest victim again, and thrashes: measured on a
    /// 192-token document, 16 067 and 10 968 evictions against 2 947 slots, which
    /// is every slot turned over five times. Bounded by the record count
    /// (12 096 here), so a few hundred kB.
    seen: HashMap<(u32, u32), u32>,
    /// Decode tokens observed, for [`AGE_TOKENS`].
    tokens: u64,
}

impl std::fmt::Debug for DeviceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("DeviceCache")
            .field("entries", &g.map.len())
            .field("bytes_used", &g.bytes_used)
            .field("budget", &self.budget)
            .field("eager", &self.eager)
            .finish()
    }
}

impl DeviceCache {
    /// Build the cache. `rec_bytes` is the host record size and sizes the
    /// pinned staging slots. `budget_bytes` is the device byte budget; a
    /// budget smaller than one expert's uploaded bytes caches nothing (each
    /// upload is still used for the dispatch that staged it).
    pub fn new(
        backend: Arc<CudaBackend>,
        budget_bytes: usize,
        rec_bytes: usize,
    ) -> Result<Self, String> {
        let pool = backend
            .pinned_pool(3, rec_bytes)
            .map_err(|e| format!("pinned staging pool: {e}"))?;
        let eager = match std::env::var("NROB_VRAM_UPLOAD_MODE").as_deref() {
            Ok("lazy") => false,
            _ => true,
        };
        Ok(Self {
            backend,
            pool,
            budget: budget_bytes,
            eager,
            inner: Mutex::new(Inner {
                seen: HashMap::new(),
                tokens: 0,
                map: HashMap::new(),
                bytes_used: 0,
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bytes_hit: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
            waits_free: AtomicU64::new(0),
            waits_pending: AtomicU64::new(0),
            stage_failures: AtomicU64::new(0),
        })
    }

    /// Staging order: `true` = whole-layer prefetch before the layer's first
    /// matvec; `false` = upload each miss right before its first compute.
    pub fn is_eager(&self) -> bool {
        self.eager
    }

    /// Test hook: force the staging order (the env var is process-global,
    /// so tests cannot rely on setting it safely in parallel).
    #[cfg(test)]
    pub(crate) fn set_eager(&mut self, eager: bool) {
        self.eager = eager;
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    /// Lookup only. Hit: LFRU update + hit/byte counters, returns the lease.
    /// Miss: counts the miss and returns `None` — the caller stages the
    /// upload via [`DeviceCache::stage_and_admit`].
    pub fn acquire(&self, layer: u32, expert: u32) -> Option<Arc<DeviceEntry>> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.clock += 1;
        let f = g.seen.entry((layer, expert)).or_insert(0);
        *f = f.saturating_add(1);
        let freq = *f;
        let clock = g.clock;
        match g.map.get(&(layer, expert)) {
            Some(en) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                self.bytes_hit.fetch_add(en.bytes as u64, Ordering::Relaxed);
                en.freq.store(freq, Ordering::Relaxed);
                en.last.store(clock, Ordering::Relaxed);
                Some(Arc::clone(en))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    // VENDORED-LOCAL: GLM-5.3-Flash.
    /// One decode token has run. Every [`AGE_TOKENS`] tokens, halve every
    /// frequency so a cooling expert can actually be displaced.
    pub fn tick_token(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.tokens += 1;
        if g.tokens % AGE_TOKENS != 0 {
            return;
        }
        g.seen.retain(|_, f| {
            *f /= 2;
            *f > 0
        });
        for en in g.map.values() {
            en.freq.store(en.freq.load(Ordering::Relaxed) / 2, Ordering::Relaxed);
        }
    }

    /// How many `(layer, expert)` pairs this cache has ever been asked for, and
    /// how many it holds. The gap is what aging and admission act on.
    // VENDORED-LOCAL: GLM-5.3-Flash.
    /// How often `(layer, expert)` has been asked for, surviving eviction.
    ///
    /// The hybrid dispatch promotes at most one miss a layer into VRAM and sends
    /// the rest to the CPU; this is what it ranks them by, so the slot goes to the
    /// expert most likely to be wanted again rather than to whichever happened to
    /// be routed first.
    pub fn freq(&self, layer: u32, expert: u32) -> u32 {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.seen.get(&(layer, expert)).copied().unwrap_or(0)
    }

    pub fn tracked(&self) -> (usize, usize) {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (g.seen.len(), g.map.len())
    }

    /// Upload one host record into a new entry and admit it under the byte
    /// budget, evicting per LFRU. The record must be packed-matmul dtypes
    /// (`plan.packed`) — dequant-fallback dtypes never reach VRAM here.
    ///
    /// The returned entry's uploads are in flight on the transfer stream;
    /// [`DeviceCache::ensure_ready`] must run before its first matvec. If
    /// the entry does not fit the budget it is still returned (this dispatch
    /// uses it once) but not cached.
    pub(crate) fn stage_and_admit(
        &self,
        layer: u32,
        expert: u32,
        rec: &[u8],
        plan: &RecordPlan,
    ) -> Result<Arc<DeviceEntry>, String> {
        debug_assert!(plan.packed, "device cache only stages packed dtypes");
        let mut tickets: Vec<UploadTicket> = Vec::with_capacity(3);
        let mut bytes = 0usize;

        // One pinned fill + one async upload per device tensor. The pinned
        // slot returns to the ring on drop; its refill is event-guarded by
        // cudarc, so a later fill can never overwrite bytes a copy engine is
        // still reading.
        let mut upload = |off: usize,
                          len: usize,
                          dtype: GgmlType,
                          shape: Vec<usize>|
         -> Result<QuantizedTensor, String> {
            let mut slot = self
                .pool
                .checkout(len)
                .map_err(|e| format!("pinned checkout: {e}"))?;
            slot.fill(&rec[off..off + len]);
            let dev = self.backend.device_slot(len);
            let (qt, ticket) = self
                .backend
                .upload_quantized_async(&slot, dev, dtype, shape);
            tickets.push(ticket);
            bytes += len;
            self.h2d_bytes.fetch_add(len as u64, Ordering::Relaxed);
            Ok(qt)
        };

        let pair = if plan.fused {
            let (glen, ulen) = (plan.gate.1, plan.up.1);
            let shape = vec![plan.gate.3[0] + plan.up.3[0], plan.gate.3[1]];
            FfnPair::Fused(Weight::Quant(upload(0, glen + ulen, plan.gate.2, shape)?))
        } else {
            let gate = Weight::Quant(upload(
                plan.gate.0,
                plan.gate.1,
                plan.gate.2,
                plan.gate.3.to_vec(),
            )?);
            let up = Weight::Quant(upload(
                plan.up.0,
                plan.up.1,
                plan.up.2,
                plan.up.3.to_vec(),
            )?);
            FfnPair::Split { gate, up }
        };
        let down = Weight::Quant(upload(
            plan.down.0,
            plan.down.1,
            plan.down.2,
            plan.down.3.to_vec(),
        )?);

        // Frequencies outlive eviction (see `Inner::seen`), so a returning hot
        // expert is admitted at the weight it earned rather than as a newcomer.
        let seed_freq = {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.seen.get(&(layer, expert)).copied().unwrap_or(1).max(1)
        };
        let entry = Arc::new(DeviceEntry {
            pair,
            down,
            bytes,
            tickets: Mutex::new(tickets),
            // Whatever this key was worth before it was last evicted.
            freq: AtomicU32::new(seed_freq),
            last: AtomicU64::new(0),
            backend: Arc::clone(&self.backend),
        });

        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.clock += 1;
        entry.last.store(g.clock, Ordering::Relaxed);
        self.evict_until(&mut g, entry.bytes);
        if g.bytes_used + entry.bytes <= self.budget {
            g.bytes_used += entry.bytes;
            // A racing admit of the same key keeps one copy (identical bytes,
            // both from the same store — the Ecache::admit_owned rule).
            g.map.insert((layer, expert), Arc::clone(&entry));
        }
        drop(g);
        Ok(entry)
    }

    /// Order the compute stream after the entry's uploads (once — the
    /// tickets are drained). Stream-level wait, no host block; the free /
    /// pending split records whether the transfer had already finished.
    pub fn ensure_ready(&self, entry: &DeviceEntry) {
        let tickets: Vec<UploadTicket> = {
            let mut t = entry.tickets.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *t)
        };
        if tickets.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        for ticket in tickets {
            if self.backend.upload_done(&ticket) {
                self.waits_free.fetch_add(1, Ordering::Relaxed);
            } else {
                self.waits_pending.fetch_add(1, Ordering::Relaxed);
            }
            self.backend.wait_upload(ticket);
        }
        self.wait_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Count a staging failure the dispatch fell back to the host path for.
    pub fn note_stage_failure(&self) {
        self.stage_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Evict LFRU victims until `need` more bytes fit, or only entries with
    /// live compute references remain. Mirrors the host Ecache policy —
    /// frequency first, recency tiebreak — with an exact full scan instead
    /// of the 16-slot sample (hundreds of entries; the scan is µs). An
    /// entry with `strong_count > 1` is pinned by an in-flight dispatch:
    /// evicting it would free nothing (the lease keeps the bytes alive) and
    /// only lose the key, so it is skipped — the same rule the host cache
    /// applies to slots with live leases.
    fn evict_until(&self, g: &mut Inner, need: usize) {
        while g.bytes_used + need > self.budget {
            let victim = g
                .map
                .iter()
                .filter(|(_, en)| Arc::strong_count(en) == 1)
                .min_by_key(|(_, en)| (en.freq.load(Ordering::Relaxed), en.last.load(Ordering::Relaxed)))
                .map(|(k, _)| *k);
            let Some(k) = victim else { break };
            let en = g.map.remove(&k).expect("victim key came from the map");
            g.bytes_used -= en.bytes;
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn stats(&self) -> DeviceCacheStats {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        DeviceCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_hit: self.bytes_hit.load(Ordering::Relaxed),
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            wait_ns: self.wait_ns.load(Ordering::Relaxed),
            waits_free: self.waits_free.load(Ordering::Relaxed),
            waits_pending: self.waits_pending.load(Ordering::Relaxed),
            stage_failures: self.stage_failures.load(Ordering::Relaxed),
            entries: g.map.len() as u64,
            bytes_used: g.bytes_used as u64,
            budget_bytes: self.budget as u64,
        }
    }
}

#[cfg(test)]
mod tests;
