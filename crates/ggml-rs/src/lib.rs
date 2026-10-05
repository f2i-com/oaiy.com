//! Tensor + ops core.
//!
//! Today the only operating dtype is `f32`. Quantized weights are dequantized
//! once at load time (see `ggml-quants`). Working with quantized matmul kernels
//! directly is on the roadmap (it's the right answer for big models, but is a
//! substantial body of work that deserves its own milestone).
//!
//! The [`Backend`] trait lets a future GPU backend (`wgpu`-based) plug in
//! without changing model code.

#![deny(rust_2018_idioms)]
#![allow(non_camel_case_types)]

pub mod backend;
pub mod chain;
pub mod exl3;
pub mod cpu;
pub mod ops;
pub mod quantized;
pub mod tensor;

pub use backend::{Backend, RopeType};
pub use chain::{ChainRecorder, DeltaNet, DeviceChain, DeviceVec};
pub use cpu::CpuBackend;
pub use quantized::{QuantizedDeviceStorage, QuantizedStorage, QuantizedTensor};
pub use tensor::{DeviceStorage, Shape, Tensor, TensorError, TensorStorage};

/// Pick the default CPU backend. Future: detect SIMD capability, choose accordingly.
pub fn default_backend() -> std::sync::Arc<dyn Backend> {
    std::sync::Arc::new(CpuBackend::new())
}
