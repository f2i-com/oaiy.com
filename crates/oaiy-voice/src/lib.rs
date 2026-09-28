//! OAIY's resident speech server: speech-to-text with NVIDIA Parakeet
//! (FastConformer with a TDT or RNN-T head) on Candle, on the GPU when there
//! is one, loaded once and kept loaded.
//!
//! It answers the same command line and HTTP routes as Aokie's
//! `aokie-voice-server`, so OAIY Desktop's `aokie-stt` service can run it in
//! its place: `GET /v1/models`, `POST /v1/audio/transcriptions` (JSON with a
//! base64 WAV, or multipart), and `POST /v1/audio/speech` reserved for
//! text-to-speech.
#![forbid(unsafe_code)]

pub mod audio;
pub mod cli;
pub mod server;
pub mod stt;
