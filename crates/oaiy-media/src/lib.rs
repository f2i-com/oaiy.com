//! Qwen Image generation/editing and LTX video in Rust on Candle tensor primitives.
//! No Python process, C++ diffusion runtime, or model-format conversion.
//!
//! One module a model family, each with its WebGPU parts in its own `webgpu` module (built with the `webgpu`
//! feature), and what they share in [`common`].
#![forbid(unsafe_code)]

pub mod common;

// Pictures.
pub mod flux3;
pub mod klein;
pub mod picture;
pub mod qwen_image;
pub mod sdxl;

// Video.
pub mod ltx;

// Sound: songs, speech, sound effects.
pub mod music;
pub mod sound;
pub mod tts;

// 3D models.
pub mod model3d;
