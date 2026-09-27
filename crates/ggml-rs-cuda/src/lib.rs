//! CUDA backend for `ggml-rs`.
//!
//! This crate plugs an NVIDIA-GPU implementation of the [`Backend`] trait into
//! the rest of the stack. Today it is **correctness-first**: every op
//! transfers its inputs to the device, runs a (naive) kernel, and copies the
//! output back to host. That makes per-op overhead substantial on small
//! workloads (small models — like stories15M — may actually be slower than
//! CPU). The plumbing is designed so the next milestone — keeping tensors on
//! the device between ops — can be added without churning the model code.
//!
//! Build requirements:
//!   * NVIDIA GPU + driver
//!   * CUDA Toolkit 12 or newer (we tested with 12.8) on PATH so `nvrtc` is reachable
//!
//! Usage:
//!
//! ```no_run
//! use std::sync::Arc;
//! use ggml_rs::Backend;
//! use ggml_rs_cuda::CudaBackend;
//!
//! let backend: Arc<dyn Backend> = Arc::new(CudaBackend::new(0).unwrap());
//! ```

#![deny(rust_2018_idioms)]

pub mod backend;
pub mod kernels;
// VENDORED-LOCAL: MOE-01/MOE-02 — GPU MoE routing + grouped expert kernels.
pub mod moe;
// VENDORED-LOCAL: GPU-02 — pinned staging ring + transfer-stream/event
// overlap APIs for the streaming-MoE pipeline.
pub mod transfer;
#[cfg(test)]
mod graph_bench;

/// VENDORED-LOCAL: PERF. Host<->device round trips, counted.
///
/// A `to_host` of a tensor the GPU just wrote is a synchronisation: the host waits,
/// and the card has nothing queued behind it. Sampled during a GLM decode the cards
/// sat at 2-18% with their memory controllers at 0-6%, which is what being starved
/// by these looks like rather than being slow at the maths. Counting them turns
/// "probably the round trips" into a number, and a number into a target.
pub mod xfer {
    use std::sync::atomic::{AtomicU64, Ordering};

    static TO_HOST: AtomicU64 = AtomicU64::new(0);
    static TO_HOST_BYTES: AtomicU64 = AtomicU64::new(0);
    static TO_DEVICE: AtomicU64 = AtomicU64::new(0);
    static TO_DEVICE_BYTES: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn host(bytes: usize) {
        TO_HOST.fetch_add(1, Ordering::Relaxed);
        TO_HOST_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn device(bytes: usize) {
        TO_DEVICE.fetch_add(1, Ordering::Relaxed);
        TO_DEVICE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// `(d2h calls, d2h bytes, h2d calls, h2d bytes)`, cumulative.
    pub fn get() -> (u64, u64, u64, u64) {
        (
            TO_HOST.load(Ordering::Relaxed),
            TO_HOST_BYTES.load(Ordering::Relaxed),
            TO_DEVICE.load(Ordering::Relaxed),
            TO_DEVICE_BYTES.load(Ordering::Relaxed),
        )
    }

    pub fn reset() {
        for c in [&TO_HOST, &TO_HOST_BYTES, &TO_DEVICE, &TO_DEVICE_BYTES] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

pub use backend::{CudaBackend, CudaError};

// VENDORED-LOCAL: GLM-5.3-Flash. Host RAM, alongside `Backend::vram_status`.
/// Free and total physical host memory, in bytes.
///
/// The companion to [`ggml_rs::Backend::vram_status`]: a tiered expert cache has
/// to size its RAM tier from what the machine actually has, the same way it sizes
/// each VRAM tier from `cuMemGetInfo`.
///
/// **Why it lives here.** `ggml-rs` and `llama-rs` contain no `unsafe` at all and
/// that is worth keeping; this needs one call on Windows, and this crate is
/// already the one that talks to the driver. It is also only ever wanted by the
/// multi-GPU tiering, which is behind the `cuda` feature anyway.
///
/// Returns `None` on a platform this has no reader for, so callers must have a
/// fallback rather than treat it as authoritative.
pub fn host_memory() -> Option<(usize, usize)> {
    host_memory_impl()
}

#[cfg(windows)]
fn host_memory_impl() -> Option<(usize, usize)> {
    // MEMORYSTATUSEX, exactly as documented: `length` must be set by the caller
    // and every other field is written by the call.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `s` is a fully initialised MEMORYSTATUSEX of the size its own
    // `length` field declares, and the call only writes into it. The pointer is
    // valid for the duration of the call and nothing retains it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    (ok != 0).then_some((s.avail_phys as usize, s.total_phys as usize))
}

#[cfg(target_os = "linux")]
fn host_memory_impl() -> Option<(usize, usize)> {
    // No unsafe needed: /proc/meminfo reports both, in kB.
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<usize> {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<usize>().ok())
            .map(|kb| kb * 1024)
    };
    // MemAvailable is the kernel's own estimate of what a new allocation can
    // have; MemFree undercounts badly because of the page cache.
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn host_memory_impl() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    /// Whatever the platform, the answer must be self-consistent or absent.
    #[test]
    fn host_memory_is_plausible_or_absent() {
        match super::host_memory() {
            None => {}
            Some((free, total)) => {
                assert!(total > 0, "total physical memory reported as 0");
                assert!(free <= total, "free {free} exceeds total {total}");
                // Any machine that can build this has more than 1 GB.
                assert!(total > (1 << 30), "total {total} is implausibly small");
            }
        }
    }
}
pub mod exl3;
pub use moe::{grouped_kernel_covers, quant_device_ptr, MoeDevicePlan, MoeRoutingDevice};
pub use transfer::{DeviceSlot, PinnedPool, PinnedSlot, UploadTicket};

/// VENDORED-LOCAL: give GPU memory freed by a dropped model back to the driver. Allocations are
/// stream-ordered (the device's default memory pool), and the pool keeps what they free for
/// reuse; after a model is unloaded that reserve can hold most of its size, so the next model
/// finds the GPU full. Synchronizes each device first, so every pending free has landed.
pub fn release_unused_memory() {
    use cudarc::driver::{sys, CudaContext};
    let count = CudaContext::device_count().unwrap_or(0);
    for ordinal in 0..count.max(0) as usize {
        let Ok(ctx) = CudaContext::new(ordinal) else { continue };
        let _ = ctx.synchronize();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // SAFETY: a valid device handle and an out pointer to a local; trimming only releases
        // pool memory no allocation holds.
        unsafe {
            if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS {
                sys::cuMemPoolTrimTo(pool, 0);
            }
        }
    }
}
