//! The architectures: each model family's forward pass on a [`ggml_rs::Backend`], as its GGUF lays it out.

pub mod gemma3;
pub mod gemma3n;
pub mod gemma4;
pub mod gemma4moe;
// GLM-5.3-Flash.
pub mod glm5next;
pub mod llama;
pub mod mixtral;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3moe;
