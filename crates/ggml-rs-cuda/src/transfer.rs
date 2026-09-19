//! VENDORED-LOCAL: GPU-02 — pinned staging ring, H2D transfer stream, and
//! event-based upload completion for the streaming-MoE pipeline.
//!
//! This module is the transfer infrastructure the VRAM expert cache builds
//! on. The steady-state decode flow it enables is:
//!
//! ```text
//! fetcher thread                     compute thread
//! --------------                     --------------
//! slot = pool.checkout(bytes)        (previous experts' matvecs running)
//! slot.fill(record_bytes)
//! ticket = backend.upload_async(&slot, &mut dev_slot, len)
//!                                    backend.wait_upload(ticket)   // compute stream waits event
//!                                    linear_q(x, &tensor)          // reads dev_slot bytes
//! ```
//!
//! Design notes:
//!
//! * [`PinnedPool`] holds a small ring (typically 3) of reusable pinned
//!   (`cuMemAllocHost`, write-combined) host buffers sized to the largest
//!   expert record. Checkout takes any free buffer big enough; a bigger
//!   request allocates (grow-once), it never reallocates per token. Checkin
//!   is `Drop`: the buffer goes back to the pool's free list.
//! * Refill safety comes from cudarc's [`PinnedHostSlice`] event: every H2D
//!   copy issued from a slot records into that slot's internal event, and
//!   `fill`/`as_mut_slice` synchronizes it first — a producer can never
//!   overwrite bytes a copy engine is still reading. With N slots the
//!   producer only blocks when all N are in flight, which is exactly the
//!   ring throttle we want.
//! * [`CudaBackend::upload_async`] issues `cuMemcpyHtoDAsync` on the
//!   dedicated transfer stream (`CudaBackend::h2d()`, non-blocking, created
//!   lazily on first transfer use so resident-only workloads never pay
//!   cudarc's multi-stream event tracking) and records a recycled
//!   [`CudaEvent`] into an [`UploadTicket`].
//! * [`CudaBackend::wait_upload`] calls `cuStreamWaitEvent` on the COMPUTE
//!   stream — stream-level ordering, never a device-wide synchronize.
//! * [`DeviceSlot`] is one fixed device allocation (stream-ordered
//!   `cuMemAllocAsync` via cudarc). Allocate once per cached expert, reuse
//!   forever; the free side is ordered explicitly — cudarc's per-slice event
//!   tracking is DISABLED context-wide (see PERF-02 in `backend.rs`), so
//!   before bytes a compute kernel may still be reading are dropped, the
//!   owner must call [`CudaBackend::order_transfer_after_compute`] (the VRAM
//!   expert cache does this from `DeviceEntry::drop`).

use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, HostSlice,
    PinnedHostSlice, SyncOnDrop,
};
use ggml_quants::GgmlType;
use ggml_rs::quantized::QuantizedTensor;

use crate::backend::{CudaBackend, CudaError, CudaQuantStorage};

/// Recycling pool for upload-completion events. `cuEventCreate` per upload
/// would be a per-token driver allocation; tickets return their event here
/// on drop so steady state reuses a handful of events.
pub(crate) type EventPool = Arc<Mutex<Vec<CudaEvent>>>;

// ----- Pinned staging ring ---------------------------------------------------

/// A pool of reusable pinned host buffers for staging expert records before
/// H2D upload. Safe under one producer thread (the expert fetcher); the free
/// list itself is mutex-guarded so checkout/checkin from multiple threads is
/// also sound, just not the intended hot path.
pub struct PinnedPool {
    ctx:  Arc<cudarc::driver::CudaContext>,
    free: Arc<Mutex<Vec<PinnedHostSlice<u8>>>>,
}

impl std::fmt::Debug for PinnedPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedPool")
            .field("idle_slots", &self.free.lock().unwrap().len())
            .finish()
    }
}

impl PinnedPool {
    /// Pre-allocate `slots` pinned buffers of `bytes` each. `bytes` should be
    /// the largest expert record size; larger checkouts later grow the pool
    /// once instead of failing.
    pub fn new(
        ctx: &Arc<cudarc::driver::CudaContext>,
        slots: usize,
        bytes: usize,
    ) -> Result<Self, CudaError> {
        let mut free = Vec::with_capacity(slots);
        for _ in 0..slots {
            free.push(alloc_pinned(ctx, bytes)?);
        }
        Ok(Self {
            ctx:  ctx.clone(),
            free: Arc::new(Mutex::new(free)),
        })
    }

    /// Take a buffer with room for at least `bytes`. Reuses a free slot when
    /// one is big enough; otherwise allocates a new (larger) buffer — this is
    /// the allowed realloc-on-grow, not a per-token allocation.
    pub fn checkout(&self, bytes: usize) -> Result<PinnedSlot, CudaError> {
        let buf = {
            let mut free = self.free.lock().unwrap();
            free.iter()
                .position(|b| b.len() >= bytes)
                .map(|pos| free.swap_remove(pos))
        };
        let buf = match buf {
            Some(b) => b,
            None => alloc_pinned(&self.ctx, bytes)?,
        };
        Ok(PinnedSlot {
            buf:    Some(buf),
            filled: 0,
            pool:   self.free.clone(),
        })
    }

    /// Number of currently-idle buffers (test/diagnostic aid).
    pub fn idle_slots(&self) -> usize { self.free.lock().unwrap().len() }
}

/// Allocate one page-locked (write-combined) host buffer.
fn alloc_pinned(
    ctx: &Arc<cudarc::driver::CudaContext>,
    bytes: usize,
) -> Result<PinnedHostSlice<u8>, CudaError> {
    // Safety: `alloc_pinned` is unsafe only because the memory starts
    // uninitialized. Every read path in this module is bounded by `filled`,
    // which is only ever set after `fill` has written those bytes.
    Ok(unsafe { ctx.alloc_pinned::<u8>(bytes) }?)
}

/// One checked-out pinned staging buffer. Checkin is `Drop` — the buffer
/// returns to the pool automatically; keep it alive until the upload that
/// reads it has at least been *issued* (completion is enforced by the event
/// sync inside `fill`, not by this type).
pub struct PinnedSlot {
    buf:    Option<PinnedHostSlice<u8>>,
    filled: usize,
    pool:   Arc<Mutex<Vec<PinnedHostSlice<u8>>>>,
}

impl PinnedSlot {
    /// Total capacity of the pinned buffer.
    pub fn capacity(&self) -> usize { self.buf.as_ref().expect("pinned slot").len() }

    /// Bytes currently filled (== what `upload_async` will copy).
    pub fn len(&self) -> usize { self.filled }

    pub fn is_empty(&self) -> bool { self.filled == 0 }

    /// Copy `data` into the buffer. Blocks until any previously-issued H2D
    /// copy out of this buffer has completed (cudarc event sync inside
    /// `PinnedHostSlice::as_mut_slice`) — that wait is the ring backpressure.
    pub fn fill(&mut self, data: &[u8]) {
        assert!(
            data.len() <= self.capacity(),
            "pinned slot overflow: {} > {}",
            data.len(),
            self.capacity()
        );
        let dst = self
            .buf
            .as_mut()
            .expect("pinned slot")
            .as_mut_slice()
            .expect("pinned buffer access");
        dst[..data.len()].copy_from_slice(data);
        self.filled = data.len();
    }

    /// The filled prefix, for host-side verification.
    pub fn as_slice(&self) -> &[u8] {
        &self
            .buf
            .as_ref()
            .expect("pinned slot")
            .as_slice()
            .expect("pinned buffer access")[..self.filled]
    }
}

impl Drop for PinnedSlot {
    fn drop(&mut self) {
        // Checkin: hand the buffer back to the pool. If an H2D copy is still
        // reading it, the next `fill` on this buffer will wait on the copy's
        // recorded event before touching the bytes, so reuse is safe.
        if let Some(buf) = self.buf.take() {
            self.pool.lock().unwrap().push(buf);
        }
    }
}

/// Lets a `PinnedSlot` be the source of `CudaStream::memcpy_htod`. cudarc
/// copies `src.len()` bytes, so `len` reports the *filled* prefix (not the
/// whole buffer) — this is what makes partial-record uploads possible
/// without a sub-allocation API on `PinnedHostSlice`.
impl HostSlice<u8> for PinnedSlot {
    fn len(&self) -> usize { self.filled }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        // Safety: we only narrow the slice handed out by the inner
        // PinnedHostSlice to the filled prefix; stream ordering and
        // completion recording are still handled by the returned guard.
        let (s, guard) = self
            .buf
            .as_ref()
            .expect("pinned slot")
            .stream_synced_slice(stream);
        (&s[..self.filled], guard)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        // Safety: same narrowing as above, mutable variant; the checkout
        // model guarantees exclusive access to the buffer.
        let filled = self.filled;
        let (s, guard) = self
            .buf
            .as_mut()
            .expect("pinned slot")
            .stream_synced_mut_slice(stream);
        (&mut s[..filled], guard)
    }
}

// ----- Device slots and upload tickets ---------------------------------------

/// One fixed device allocation, sized once and reused for every upload of
/// the same (or smaller) record. For the VRAM expert cache: keep one of
/// these alive per cached expert — see [`CudaBackend::upload_quantized_async`]
/// for turning it into a `QuantizedTensor` directly.
pub struct DeviceSlot {
    buf: CudaSlice<u8>,
}

impl DeviceSlot {
    /// Byte capacity.
    pub fn len(&self) -> usize { self.buf.len() }

    pub fn is_empty(&self) -> bool { self.buf.is_empty() }

    /// Raw device pointer — stable for the life of the slot; tests use it to
    /// prove upload reuse performs no reallocation.
    pub fn raw_ptr(&self) -> u64 {
        let (p, _guard) = self.buf.device_ptr(self.buf.stream());
        p as u64
    }

    /// Byte-range view (e.g. gate/up/down subviews of a packed record).
    pub fn view(&self, offset: usize, len: usize) -> CudaView<'_, u8> {
        self.buf.slice(offset..offset + len)
    }

    /// Mutable byte-range view.
    pub fn view_mut(&mut self, offset: usize, len: usize) -> CudaViewMut<'_, u8> {
        self.buf.slice_mut(offset..offset + len)
    }

    /// The whole allocation as a `&CudaSlice<u8>` (what the quantized
    /// matvec kernels consume).
    pub fn as_cuda_slice(&self) -> &CudaSlice<u8> { &self.buf }

    /// Unwrap into the underlying allocation (used to build
    /// `CudaQuantStorage` without a copy).
    pub fn into_cuda_slice(self) -> CudaSlice<u8> { self.buf }
}

/// Handle to an in-flight (or completed) H2D upload. Returned by
/// [`CudaBackend::upload_async`]; consumed by [`CudaBackend::wait_upload`]
/// or dropped (the event is recycled either way).
pub struct UploadTicket {
    event: Option<CudaEvent>,
    pool:  EventPool,
}

impl UploadTicket {
    fn event(&self) -> &CudaEvent { self.event.as_ref().expect("upload ticket") }
}

impl Drop for UploadTicket {
    fn drop(&mut self) {
        // Recycling is safe even if the compute stream has already been told
        // to wait on this event: cuStreamWaitEvent snapshots the event's
        // captured work at call time, so a later re-record cannot disturb
        // waits that are already enqueued.
        if let Some(e) = self.event.take() {
            self.pool.lock().unwrap().push(e);
        }
    }
}

// ----- CudaBackend transfer API ------------------------------------------------

impl CudaBackend {
    /// Create a pinned staging ring: `slots` buffers of `bytes` each (the
    /// roadmap's 2–3 reusable pinned slots; use 3 for one producer).
    pub fn pinned_pool(&self, slots: usize, bytes: usize) -> Result<PinnedPool, CudaError> {
        PinnedPool::new(&self.ctx, slots, bytes)
    }

    /// Allocate one reusable device staging slot of `bytes`. Allocate once
    /// (per expert, or per in-flight record) — never per token.
    pub fn device_slot(&self, bytes: usize) -> DeviceSlot {
        // Safety: `alloc` is unsafe because the memory starts uninitialized.
        // The slot is only ever read after `upload_async` has written the
        // first `len` bytes, and `len` is always `<= bytes`.
        let buf = unsafe { self.h2d().alloc::<u8>(bytes) }.expect("device slot alloc");
        DeviceSlot { buf }
    }

    /// Upload the filled contents of `pinned` into `slot` on the transfer
    /// stream and return a ticket for the copy's completion event.
    ///
    /// Truly asynchronous (pinned source + non-blocking stream): returns
    /// while the copy engine is still working. `len` must equal
    /// `pinned.len()` — upload exactly what was filled.
    pub fn upload_async(
        &self,
        pinned: &PinnedSlot,
        slot: &mut DeviceSlot,
        len: usize,
    ) -> UploadTicket {
        assert_eq!(
            len,
            pinned.len(),
            "upload_async: len must equal the pinned slot's filled length"
        );
        assert!(
            len <= slot.len(),
            "upload_async: device slot too small ({len} > {})",
            slot.len()
        );
        let mut dst = slot.buf.slice_mut(0..len);
        self.h2d()
            .memcpy_htod(pinned, &mut dst)
            .expect("async h2d upload");
        let event = self.take_event();
        event.record(self.h2d()).expect("record upload event");
        UploadTicket {
            event: Some(event),
            pool:  self.event_pool.clone(),
        }
    }

    /// Make the COMPUTE stream wait for the upload (cuStreamWaitEvent
    /// semantics). This is a stream-level dependency — no device-wide
    /// synchronize, and the calling CPU thread does not block.
    pub fn wait_upload(&self, ticket: UploadTicket) {
        self.stream
            .wait(ticket.event())
            .expect("compute stream wait on upload event");
    }

    /// Non-blocking completion query (cuEventQuery).
    pub fn upload_done(&self, ticket: &UploadTicket) -> bool { ticket.event().is_complete() }

    /// Async variant of the `cuda_quant_input` upload path: copies packed
    /// quant bytes from `pinned` into `slot` on the transfer stream and
    /// wraps the slot as a `QuantizedTensor` — no fresh allocation, no
    /// default-stream involvement.
    ///
    /// The returned tensor MUST NOT be used in a compute op until
    /// `wait_upload(ticket)` (or a positive `upload_done` poll) has ordered
    /// the compute stream after the copy.
    pub fn upload_quantized_async(
        &self,
        pinned: &PinnedSlot,
        mut slot: DeviceSlot,
        dtype: GgmlType,
        shape: Vec<usize>,
    ) -> (QuantizedTensor, UploadTicket) {
        let len = pinned.len();
        let ticket = self.upload_async(pinned, &mut slot, len);
        let storage = CudaQuantStorage {
            bytes:  slot.into_cuda_slice(),
            dtype,
            stream: self.stream.clone(),
            name:   self.name.clone(),
        };
        (QuantizedTensor::from_device(Box::new(storage), shape), ticket)
    }

    /// The dedicated H2D transfer stream. Created on first use — calling
    /// this on a backend that never streams experts flips the context into
    /// cudarc's multi-stream mode (see the `h2d` field comment), so query it
    /// only when transfers will actually run.
    pub fn transfer_stream(&self) -> &Arc<CudaStream> { self.h2d() }

    /// The compute stream (the backend's default stream) — exposed so
    /// consumers can reason about `wait_upload` ordering.
    pub fn compute_stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// Order the transfer stream after everything enqueued on the compute
    /// stream so far (record an event on the compute stream, then
    /// `cuStreamWaitEvent` it on the H2D stream). Stream-level, never a host
    /// block.
    ///
    /// With cudarc's per-slice event tracking disabled context-wide (PERF-02
    /// in `backend.rs`), this is the read→free edge for device bytes that
    /// cross streams: an H2D-written, compute-read allocation whose
    /// `CudaSlice` frees on the H2D stream at drop must not be freed while a
    /// queued compute kernel can still read it. Dropping the slices right
    /// after this call enqueues their `cuMemFreeAsync` behind the wait.
    pub fn order_transfer_after_compute(&self) {
        let event = self.take_event();
        event.record(&self.stream).expect("record compute guard event");
        self.h2d().wait(&event).expect("h2d wait on compute guard");
        // Recycle immediately: cuStreamWaitEvent snapshots the recorded work
        // at call time, so a later re-record cannot disturb the wait above.
        self.event_pool.lock().unwrap().push(event);
    }

    /// Pop a recycled event or create a new one (timing disabled — these
    /// events are pure synchronization primitives).
    fn take_event(&self) -> CudaEvent {
        if let Some(e) = self.event_pool.lock().unwrap().pop() {
            return e;
        }
        self.ctx.new_event(None).expect("create cuda event")
    }
}
