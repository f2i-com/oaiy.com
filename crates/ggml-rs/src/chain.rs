//! VENDORED-LOCAL: a decode step's ops chained on a device between host round trips.
//!
//! A backend whose every call is a submit and a read back (WebGPU) spends a dense model's decode step on round trips:
//! a 3B Llama's 113 of them were most of its 37 ms. A [`DeviceChain`] keeps the activations on the device and records
//! a run of ops (quantized matmuls, RMSNorm, adds, the SwiGLU) into one submit, reading back only what the host needs
//! next (a layer's q, k and v for its attention, the logits at the end). A model asks its backend for one through
//! [`crate::Backend::chain`] and keeps its own path where there is none.

use std::any::Any;
use std::sync::Arc;

use crate::quantized::QuantizedTensor;

/// An f32 vector held by a [`DeviceChain`]'s device: the backend's own buffer behind it.
#[derive(Clone)]
pub struct DeviceVec {
    pub len: usize,
    pub inner: Arc<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for DeviceVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceVec({})", self.len)
    }
}

/// A device that runs a chain of ops on vectors it holds.
pub trait DeviceChain: Send + Sync {
    /// A zeroed vector of `len`.
    fn vec(&self, len: usize) -> DeviceVec;
    /// Write `data` into `v` (before whatever is recorded next runs).
    fn upload(&self, v: &DeviceVec, data: &[f32]);
    /// Whether this device holds `w` where its matmuls read it.
    fn holds(&self, w: &QuantizedTensor) -> bool;
    /// Start recording.
    fn begin(&self) -> Box<dyn ChainRecorder + '_>;
}

/// Ops recorded in order, run together by [`ChainRecorder::finish`].
pub trait ChainRecorder {
    /// `y = W x` for one row `x` (`[k]`) of a weight the device holds (`[n, k]`).
    fn matmul(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec);
    /// `out = x / rms(x) * w`.
    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32);
    /// `acc += y`.
    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec);
    /// `out = silu(fused[..ff]) * fused[ff..]`, `ff = out.len`.
    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec);
    /// Read `v` back once the chain has run.
    fn read(&mut self, v: &DeviceVec);
    /// Run what was recorded (one submit) and return what was read, in order.
    fn finish(self: Box<Self>) -> Vec<Vec<f32>>;
}
