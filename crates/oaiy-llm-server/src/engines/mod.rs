//! The engines: each takes [`job::Job`]s off a channel and sends [`job::Event`]s back, for the models it serves.

// The request/reply types every engine shares.
pub(crate) mod job;
// Which backend runs GGUF models: WebGPU or the CPU.
pub(crate) mod backend;
// Prompt states kept on disk, so a new process starts where an earlier one left off.
pub(crate) mod disk;

// GLM-5.3-Flash served through the same job/event contract.
pub mod glm;
// Qwen's dense hybrids, Qwen3.8-Flash-Next and OrcaSAQ (its cache, the conversations it parks in host RAM, the vision
// tower an EXL3 checkpoint comes with).
pub(crate) mod qwen;
// LiquidAI's LFM2 chat format (LFM2.5-24B-A2B, run as a Qwen3.5 hybrid).
pub(crate) mod lfm2;
// OrcaSAQ (EXL3: packed on any GPU through WebGPU, else decoded on the CPU), its PEFT adapters and its Qwen vision
// tower.
pub(crate) mod orcasaq;
// Qwen3.8-Flash-Next (qwen4_exp, EXL3, on any GPU through WebGPU).
#[cfg(feature = "webgpu")]
pub(crate) mod flashnext;
// PEFT and GGUF LoRA adapters beside the packed weights.
pub(crate) mod lora;
// DeepSeek-V4.1 without CUDA: the CPU model with its dense trunk on the WebGPU adapter.
#[cfg(feature = "webgpu")]
pub(crate) mod dsv41_portable;
