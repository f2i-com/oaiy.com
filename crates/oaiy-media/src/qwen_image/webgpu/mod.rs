//! Qwen Image 2.1 on WebGPU: the transformer, the VAE, the text encoder and the vision tower, each as its Candle
//! counterpart computes it.

pub mod text;
pub mod transformer;
pub mod vae;
pub mod vision;
