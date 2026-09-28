//! DeepSeek-V4.1-Flash for OAIY. See `docs/DEEPSEEK_V41.md` for the plan
//! and the measurements behind it.
//!
//! The checkpoint is used in place: [`safetensors::StIndex`] reads only the
//! shard headers, and [`expert::SafetensorsExpertStore`] serves the 15,360
//! routed experts to OAIY's cache layer straight from the shards, two
//! positioned reads per expert. Nothing is converted.
//!
//! Module map:
//! - [`safetensors`], [`io`] — shard index, positioned / page-cache-bypassing reads
//! - [`formats`] — bf16 / f16 / fp8 e4m3 / e8m0 / fp4 e2m1 codecs, fp8 activation quantization
//! - [`config`] — hyperparameters from `config.json`
//! - [`linear`], [`ops`] — dense weights and the reference `linear()`, norms, RoPE
//! - [`hc`] — hyper-connections (4 residual copies, Sinkhorn mixing)
//! - [`attention`] — sliding window + compressed sparse attention, compressor, indexer
//! - [`expert`], [`moe`] — the in-place expert store, routed/shared experts, router
//! - [`cpu_experts`] — decode-time routed experts on a CPU thread pool (hybrid tier)
//! - [`engram`] — n-gram hash memory
//! - [`model`] — the backbone forward pass
//! - [`vision`] — image preprocessing (bit-exact to the reference's Pillow
//!   pipeline) and the ViT + aligner on the CPU

#![forbid(unsafe_code)]

pub mod attention;
pub mod chat;
pub mod config;
pub mod cpu_experts;
pub mod detok;
pub mod engram;
pub mod expert;
pub mod ternary;
pub mod formats;
pub mod golden;
pub mod hc;
pub mod io;
pub mod linear;
pub mod model;
pub mod moe;
pub mod ops;
pub mod safetensors;
pub mod tokenizer;
pub mod unicode;
pub mod vision;
