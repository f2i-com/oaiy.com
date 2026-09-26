//! nrob core: the pieces every engine in the workspace shares.
//!
//! nrob runs models whose weights do not fit in VRAM (or RAM) by moving
//! them through three tiers: NVMe -> RAM -> GPU. The engines live in their
//! own crates (`llama-rs` for GGUF models, `dsv41` / `dsv41-cuda` for
//! DeepSeek-V4.1 safetensors); this crate holds what they have in common:
//!
//! - [`store`]: the [`store::WeightStore`] seam, one record per
//!   (layer, expert), read from wherever the weights live
//! - [`ecache`]: the bounded RAM expert cache (LFRU) over a store
//! - [`backend`]: the row-parallel pool and platform probes
//! - [`json`]: a small JSON reader (safetensors headers, configs)
//! - [`http`]: the minimal HTTP/1.1 server and client the API hosts share
//! - [`error`], [`types`]: shared error and policy types
//!
//! std-only and no `unsafe`; see CONVENTIONS.md at the workspace root.

#![forbid(unsafe_code)]

pub mod backend;
pub mod ecache;
pub mod error;
pub mod http;
pub mod json;
pub mod store;
pub mod types;

pub use error::{Error, Result};
pub use types::{CachePolicy, VERSION};
