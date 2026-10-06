//! Qwen Image generation/editing and LTX video in Rust on Candle tensor primitives.
//! No Python process, C++ diffusion runtime, or model-format conversion.
#![forbid(unsafe_code)]

pub mod math;
pub mod pipeline;
pub mod reference;
pub mod residency;
pub mod schedule;
pub mod text;
pub mod transformer;
#[cfg(feature = "webgpu")]
pub mod wgpu_weights;
#[cfg(feature = "webgpu")]
pub mod qwen_wgpu;
#[cfg(feature = "webgpu")]
pub mod vae_wgpu;
#[cfg(feature = "webgpu")]
pub mod text_wgpu;
#[cfg(feature = "webgpu")]
pub mod ltx_wgpu;
#[cfg(feature = "webgpu")]
pub mod ltx_vae_wgpu;
#[cfg(feature = "webgpu")]
pub mod ltx_text_wgpu;
pub mod vae;
pub mod vision;
pub mod weights;
pub mod lora;
mod comfy_quant;
pub mod ltx;
pub mod sdxl;
pub mod music;
pub mod tts;
pub mod sound;
pub mod model3d;
pub mod birefnet;
pub mod esrgan;
pub mod picture;
pub mod klein;
