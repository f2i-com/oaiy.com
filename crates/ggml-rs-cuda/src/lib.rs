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
pub use moe::{grouped_kernel_covers, quant_device_ptr, MoeDevicePlan, MoeRoutingDevice};
pub use transfer::{DeviceSlot, PinnedPool, PinnedSlot, UploadTicket};
