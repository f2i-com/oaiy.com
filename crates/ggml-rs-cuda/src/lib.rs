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
//!   * CUDA Toolkit (we tested with 12.8) on PATH so `nvrtc` is reachable
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
pub use moe::{grouped_kernel_covers, quant_device_ptr, MoeDevicePlan, MoeRoutingDevice};
pub use transfer::{DeviceSlot, PinnedPool, PinnedSlot, UploadTicket};
