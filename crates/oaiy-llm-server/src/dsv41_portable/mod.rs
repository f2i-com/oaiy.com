//! DeepSeek-V4.1 without CUDA: the CPU model (`dsv41::model`, the reference the CUDA path is tested against) with its
//! dense trunk on the WebGPU adapter. Measured on the CPU alone, a prompt is compute-bound and its attention (the fp8
//! trunk's projections among it) costs more than its experts, and a warm decode step spends a flat second there
//! (docs/DEEPSEEK_V41.md, "The CPU model, measured"): so the trunk goes to the GPU, every fp8 and bf16 matrix the
//! budget holds (`ggml_rs_wgpu::dense`), the activation still quantized and the result still rounded by the CPU model
//! as the reference does. The routed experts go there too ([`WgpuExperts`]): a prompt's busy ones through slots its
//! records are uploaded into, and the ones used most kept there between passes, a decode step's computed there while
//! the CPU reads and computes the rest.

use std::sync::{Arc, Mutex};

use dsv41::linear::{DenseKernel, Weight};
use ggml_rs_wgpu::dense::{DenseData, DenseGpu, RecordSlots};
use ggml_rs_wgpu::WgpuBackend;

mod engine;
mod experts;
mod resident;
#[cfg(test)]
mod tests;
mod trunk;
mod usage;

pub(crate) use engine::Engine;
pub(crate) use experts::WgpuExperts;
pub(crate) use trunk::offload;
pub(crate) use usage::{read_usage, write_usage};
