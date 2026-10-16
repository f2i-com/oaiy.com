//! The models chained on a GPU's device ([`ggml_rs::chain`]): a decode step or a prompt's chunk in one submission, as
//! each architecture's op-by-op forward pass computes it.

// A dense model's decode step.
pub(crate) mod decode;
// Gemma 4's dense models, and its vision tower.
pub(crate) mod gemma4;
pub mod gemma4v;
// MiniCPM-V's and Qwen3-VL's vision towers.
pub(crate) mod minicpmv;
pub(crate) mod mmproj;
// Qwen3.5's hybrid (and its drafts, and a model split over two devices).
pub(crate) mod qwen35;
