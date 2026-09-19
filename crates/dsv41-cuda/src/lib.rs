//! CUDA path for DeepSeek-V4.1 on nrob (docs/DEEPSEEK_V41.md, Phase C).
//!
//! Every kernel mirrors a CPU function in `dsv41` — which is validated
//! against the reference model — and is tested against it
//! (`tests/kernels.rs`). The CPU crate stays the numerical oracle; this one
//! only has to agree with it.
//!
//! - [`gpu`] — context, NVRTC-compiled kernels, the typed launch wrappers
//! - [`model`] — the backbone on one or more GPUs, hybrid CPU/GPU decode
//! - [`expert_cache`] — the VRAM expert tier
//! - [`cpu`] — the CPU row kernel for hybrid decode (AVX-512 when present;
//!   its one `unsafe` call is the feature-checked dispatch)
//! - [`handoff`] — pinned, device-mapped memory for the CPU/GPU exchanges
//!   of a decode step (launch-ahead hand-off, router mailbox)
//! - [`vision`] — the ViT and aligner (images to LLM input rows)
//!
//! `unsafe` lives in three places, each block with its SAFETY argument: the
//! kernel launches and context setup (`gpu`), the pinned mappings (`handoff`)
//! and the feature-checked AVX-512 call (`cpu`).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod cpu;
pub mod expert_cache;
pub mod gpu;
pub mod handoff;
pub mod model;
pub mod vision;

pub use gpu::Gpu;
pub use model::{Checkpoint, GpuModel, GpuOptions, ImageSpan};
pub use vision::GpuVision;
