//! What the models share: tensor math, reading a checkpoint's weights (on the CPU and onto a WebGPU device), keeping
//! them in RAM or on the GPU within a budget, LoRA adapters, Comfy's quantized checkpoints, and a Qwen3 decoder's
//! prompt on WebGPU (MOSS-SoundEffect's and Qwen3-TTS's text encoders).

pub mod comfy_quant;
pub mod job;
pub mod lora;
pub mod math;
pub mod residency;
pub mod weights;
#[cfg(feature = "webgpu")]
pub mod qwen3_webgpu;
#[cfg(feature = "webgpu")]
pub mod webgpu_weights;
#[cfg(test)]
pub(crate) mod timing;
