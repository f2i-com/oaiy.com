//! Pinned, device-mapped host memory for the two small exchanges of a
//! hybrid decode step, both without a driver round trip:
//!
//! - [`Handoff`], CPU to GPU ("launch-ahead"): when a layer has experts on
//!   the CPU, the host does not wait for them. It starts the CPU job and
//!   keeps queueing GPU work (the layer's reduction, the next layer's
//!   attention, its router); the reduction kernel waits on the device for a
//!   flag the CPU job raises once its output rows are written.
//! - [`Inbox`], GPU to CPU: a kernel copies the router's logits and the
//!   layer's activation straight into host memory and raises a flag the host
//!   spins on, instead of two driver downloads (each ~35-50 us of latency on
//!   Windows before its copy even starts).
//!
//! Protocol, per buffer: sequence numbers only grow. The writer fills the
//! data, then publishes the number (with fences, so the data is visible
//! first); the reader waits for the number, then reads. A buffer is rewritten
//! only for a later number, after the reader of the previous one is done
//! (the host orders both sides: see each type).

use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cudarc::driver::{sys, CudaContext};
use nrob::{Error, Result};

use crate::gpu::{cu, Gpu};

/// Bytes before the data: the flag, padded so the data starts aligned.
const HEADER: usize = 256;

/// A pinned host allocation mapped into one device's address space:
/// `[flag u32 | pad][data]`.
struct Mapped {
    ctx: Arc<CudaContext>,
    host: *mut u8,
    dev: u64,
    bytes: usize,
}

impl Mapped {
    fn new(g: &Gpu, data_bytes: usize) -> Result<Mapped> {
        let bytes = HEADER + data_bytes;
        cu(g.context().bind_to_thread())?;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc writes the allocation's address into `p`;
        // DEVICEMAP makes it addressable from the device, PORTABLE from every
        // context. The result is checked before `p` is used.
        cu(unsafe { sys::cuMemHostAlloc(&mut p, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE) }.result())?;
        // SAFETY: `p` is a fresh, exclusively owned allocation of `bytes` bytes.
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `p` is a DEVICEMAP allocation and this context is current;
        // the device address is written into `dev`.
        let mapped = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, p, 0) }.result();
        if mapped.is_err() {
            // SAFETY: `p` came from cuMemHostAlloc and nothing else refers to it.
            unsafe { sys::cuMemFreeHost(p) };
        }
        cu(mapped)?;
        Ok(Mapped { ctx: Arc::clone(g.context()), host: p as *mut u8, dev, bytes })
    }

    fn flag(&self) -> &AtomicU32 {
        // SAFETY: the flag is the first, 4-byte-aligned word of the pinned
        // allocation; the host only ever accesses it atomically.
        unsafe { &*(self.host as *const AtomicU32) }
    }

    fn data_bytes(&self) -> usize {
        self.bytes - HEADER
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: `host` came from cuMemHostAlloc in `new`, and this is the
        // last reference to it (the owners are only shared through Arcs).
        unsafe { sys::cuMemFreeHost(self.host as *mut std::ffi::c_void) };
    }
}

/// CPU to GPU: output rows of a layer's CPU experts.
///
/// Rows are rewritten only by a later job, which the host starts only after
/// a synchronization that the reading kernel precedes in stream order.
pub struct Handoff {
    m: Mapped,
    rows: usize,
    dim: usize,
    error: Mutex<Option<String>>,
}

// SAFETY: the pinned allocation is owned by `m` until Drop. Concurrent access
// follows the module protocol: only the one CPU job in flight writes rows
// (`write_row`), the flag is written atomically (`release`), and the device
// reads rows only after the flag has reached the job's number.
unsafe impl Send for Handoff {}
unsafe impl Sync for Handoff {}

impl Handoff {
    /// Room for `rows` output rows of `dim` floats, mapped into `g`'s context.
    pub fn new(g: &Gpu, rows: usize, dim: usize) -> Result<Handoff> {
        Ok(Handoff { m: Mapped::new(g, rows * dim * 4)?, rows, dim, error: Mutex::new(None) })
    }

    /// Device address of the flag (`u32`).
    pub fn flag_dev(&self) -> u64 {
        self.m.dev
    }

    /// Device address of row 0 (`rows x dim` floats).
    pub fn rows_dev(&self) -> u64 {
        self.m.dev + HEADER as u64
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Write output row `r` (a CPU job, before it releases its number).
    pub fn write_row(&self, r: usize, v: &[f32]) {
        assert!(r < self.rows && v.len() == self.dim, "hand-off row {r} of {}, {} floats for {}", self.rows, v.len(), self.dim);
        debug_assert!((r + 1) * self.dim * 4 <= self.m.data_bytes());
        // SAFETY: row r lies inside the allocation (checked above); only the
        // job in flight writes rows, and the device reads them only after
        // this job's release (module protocol).
        unsafe {
            let dst = (self.m.host.add(HEADER) as *mut f32).add(r * self.dim);
            std::ptr::copy_nonoverlapping(v.as_ptr(), dst, self.dim);
        }
    }

    /// Publish sequence number `seq`: every row written before this call is
    /// visible to a kernel that has seen the flag reach `seq`.
    pub fn release(&self, seq: u32) {
        fence(Ordering::SeqCst);
        self.m.flag().store(seq, Ordering::SeqCst);
        fence(Ordering::SeqCst);
    }

    /// Record a failed job (its rows are garbage; the next check reports it).
    pub fn fail(&self, why: String) {
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
    }

    /// The failure of a job since the last call, if any.
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// GPU to CPU: a few KB a kernel publishes ([`Gpu::publish`]) for the host.
///
/// The host reads a number's data before it queues the kernel that
/// publishes the next one, so the data is never rewritten under it.
pub struct Inbox {
    m: Mapped,
    floats: usize,
}

// SAFETY: the pinned allocation is owned by `m` until Drop; the device
// writes the data and then the flag, and the host reads the data only after
// seeing the flag (module protocol), from one thread at a time (`&mut self`
// is not needed because reads copy out and never overlap device writes).
unsafe impl Send for Inbox {}
unsafe impl Sync for Inbox {}

impl Inbox {
    /// Room for `floats` values, mapped into `g`'s context.
    pub fn new(g: &Gpu, floats: usize) -> Result<Inbox> {
        Ok(Inbox { m: Mapped::new(g, floats * 4)?, floats })
    }

    pub fn flag_dev(&self) -> u64 {
        self.m.dev
    }

    pub fn data_dev(&self) -> u64 {
        self.m.dev + HEADER as u64
    }

    pub fn floats(&self) -> usize {
        self.floats
    }

    /// Wait until the device has published `seq`. Spins (the answer is
    /// usually microseconds away); after 5 ms it synchronizes the stream
    /// instead, which also surfaces a device error rather than spinning on.
    pub fn wait(&self, g: &Gpu, seq: u32) -> Result<()> {
        let t0 = Instant::now();
        while self.m.flag().load(Ordering::Acquire) < seq {
            if t0.elapsed() > Duration::from_millis(5) {
                g.sync()?;
                if self.m.flag().load(Ordering::Acquire) < seq {
                    return Err(Error::Format(format!("device did not publish {seq}")));
                }
                break;
            }
            std::hint::spin_loop();
        }
        fence(Ordering::Acquire);
        Ok(())
    }

    /// Copy `out.len()` floats from offset `off` (after [`wait`](Self::wait)).
    pub fn read(&self, off: usize, out: &mut [f32]) {
        assert!(off + out.len() <= self.floats);
        // SAFETY: the range lies inside the allocation (checked above), and
        // the device finished writing it before publishing the number the
        // caller waited for (module protocol).
        unsafe {
            let src = (self.m.host.add(HEADER) as *const f32).add(off);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
    }
}
