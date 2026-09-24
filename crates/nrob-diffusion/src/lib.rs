//! Qwen Image 2.1 text-to-image, implemented in Rust on Candle tensor primitives.
//! No Python process, C++ diffusion runtime, or model-format conversion.
#![forbid(unsafe_code)]

pub mod math;
pub mod pipeline;
pub mod schedule;
pub mod text;
pub mod transformer;
pub mod vae;
pub mod weights;
