//! Qwen Image 2.1: pictures from a prompt and edits of reference pictures. The pipeline ([`pipeline`]) runs the text
//! encoder (Qwen3-VL-8B's language model, [`text`], with its vision tower for references, [`vision`] and
//! [`reference`]), the single-stream transformer ([`transformer`]) over a flow-matching schedule ([`schedule`]) and the
//! RGBA VAE ([`vae`]), on the CPU through Candle or on WebGPU ([`webgpu`]).

pub mod pipeline;
pub mod reference;
pub mod schedule;
pub mod text;
pub mod transformer;
pub mod vae;
pub mod vision;
#[cfg(feature = "webgpu")]
pub mod webgpu;
