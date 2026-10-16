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
pub use chain::{ChainRecorder, CleanRows, DeltaNet, DeviceChain, DeviceVec, RowNorm};
pub use cpu::CpuBackend;
pub use quantized::{QuantizedDeviceStorage, QuantizedStorage, QuantizedTensor};
pub use tensor::{DeviceStorage, Shape, Tensor, TensorError, TensorStorage};

/// VENDORED-LOCAL: memory the process freed given back to the system where its allocator keeps it: macOS's kept some
/// 0.6 GB a model's load had used and freed (its weights' copies as each went to the GPU), the process's still while
/// the model's weights were on the card. Elsewhere nothing.
pub fn give_back_freed_memory() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: the system allocator's own call, for every zone (a null zone) and as much as it can (goal 0): it
        // hands free pages back and touches nothing in use.
        unsafe {
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
    }
}

/// Pick the default CPU backend. Future: detect SIMD capability, choose accordingly.
pub fn default_backend() -> std::sync::Arc<dyn Backend> {
    std::sync::Arc::new(CpuBackend::new())
}
