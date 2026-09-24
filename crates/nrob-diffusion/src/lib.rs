//! Qwen Image generation/editing and LTX video in Rust on Candle tensor primitives.
//! No Python process, C++ diffusion runtime, or model-format conversion.
#![forbid(unsafe_code)]

pub mod math;
pub mod pipeline;
pub mod reference;
pub mod schedule;
pub mod text;
pub mod transformer;
pub mod vae;
pub mod vision;
pub mod weights;
mod comfy_quant;
pub mod ltx;
