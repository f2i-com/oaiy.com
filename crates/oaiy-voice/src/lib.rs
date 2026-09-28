//! OAIY's resident speech server: speech-to-text with NVIDIA Parakeet
//! (FastConformer with a TDT or RNN-T head) and text-to-speech with
//! Qwen3-TTS (`oaiy-tts`, in a voice cloned from a clip), on Candle, on the
//! GPU with the most free memory, loaded once and kept loaded.
//!
//! It answers the same command line and HTTP routes as Aokie's
//! `aokie-voice-server`, so OAIY Desktop runs it for calls in their place:
//! `GET /v1/models`, `POST /v1/audio/transcriptions` (JSON with a base64 WAV,
//! or multipart), `POST /v1/audio/speech` (streamed PCM or a WAV) and
//! `GET /v1/audio/voices`.
#![forbid(unsafe_code)]

pub mod audio;
pub mod cli;
pub mod server;
pub mod stt;
pub mod tts;
