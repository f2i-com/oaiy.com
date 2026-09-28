//! Speech for calls: speech-to-text and text-to-speech from OAIY's own voice
//! server (`oaiy-voice`: Parakeet and Qwen3-TTS in Rust on the GPU, in the
//! voice chosen from the clips in `<data>/voices`), run as the `oaiy-voice`
//! service and started when a call needs it. Aokie's servers answer the same
//! routes, and remain as the `aokie-stt` / `aokie-tts` services.

use std::time::{Duration, Instant};

use base64::Engine as _;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::audio::{self, Resampler};
use crate::services::registry::RegistryHandle;

/// One process serves both: speech-to-text and text-to-speech.
pub const STT_SERVICE: &str = "oaiy-voice";
pub const TTS_SERVICE: &str = "oaiy-voice";
/// The rate calls run at on the wire (Aokie's realtime stream).
pub const WIRE_RATE: u32 = 24_000;
/// What speech-to-text takes.
pub const STT_RATE: u32 = 16_000;
/// How long a starting service may take to answer (it loads its model first).
const START_WAIT: Duration = Duration::from_secs(180);

#[derive(Clone)]
pub struct Engines {
    registry: Option<RegistryHandle>,
    http: reqwest::Client,
    /// Fixed addresses (tests): no services are started.
    fixed: Option<(String, String)>,
}

impl Engines {
    pub fn new(registry: RegistryHandle) -> Self {
        Self { registry: Some(registry), http: reqwest::Client::new(), fixed: None }
    }

    /// Speech at fixed addresses (tests, or a voice server run by hand).
    pub fn at(stt: &str, tts: &str) -> Self {
        Self { registry: None, http: reqwest::Client::new(), fixed: Some((stt.to_string(), tts.to_string())) }
    }

    /// A speech service's address, started if it is not running and waited on until it answers.
    async fn base(&self, id: &'static str) -> Result<String, String> {
        if let Some((stt, tts)) = &self.fixed {
            return Ok(if id == STT_SERVICE { stt.clone() } else { tts.clone() });
        }
        let registry = self.registry.clone().ok_or("no services")?;
        let port = tokio::task::spawn_blocking(move || -> Result<u16, String> {
            let mut r = registry.lock().map_err(|_| "the services are unavailable".to_string())?;
            r.start(id).map_err(|e| format!("{id} did not start: {e}"))?;
            r.service_port(id).ok_or_else(|| format!("{id} has no port"))
        })
        .await
        .map_err(|e| e.to_string())??;
        let base = format!("http://127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            match self.http.get(format!("{base}/v1/models")).timeout(Duration::from_secs(3)).send().await {
                Ok(r) if r.status().is_success() => return Ok(base),
                _ if started.elapsed() > START_WAIT => return Err(format!("{id} did not answer within {} s", START_WAIT.as_secs())),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }

    /// Start both (a call is coming: the first words should not wait for a model to load).
    pub async fn warm(&self) -> Result<(), String> {
        if STT_SERVICE == TTS_SERVICE && self.fixed.is_none() {
            return self.base(STT_SERVICE).await.map(|_| ());
        }
        let (a, b) = tokio::join!(self.base(STT_SERVICE), self.base(TTS_SERVICE));
        a.and(b).map(|_| ())
    }

    /// What was said in an utterance (24 kHz phone audio).
    pub async fn transcribe(&self, utterance: &[i16]) -> Result<String, String> {
        let pcm = Resampler::new(WIRE_RATE, STT_RATE).process(utterance);
        self.transcribe_wav(audio::wav(&pcm, STT_RATE)).await
    }

    /// What was said in a recording: a 16 kHz mono 16-bit WAV.
    pub async fn transcribe_wav(&self, wav: Vec<u8>) -> Result<String, String> {
        let base = self.base(STT_SERVICE).await?;
        let body = serde_json::json!({
            "audio": base64::engine::general_purpose::STANDARD.encode(wav),
            "response_format": "json",
        });
        let resp = self.http.post(format!("{base}/v1/audio/transcriptions")).json(&body).timeout(Duration::from_secs(60)).send().await.map_err(|e| format!("speech-to-text: {e}"))?;
        let status = resp.status();
        let value: serde_json::Value = resp.json().await.map_err(|e| format!("speech-to-text answer: {e}"))?;
        if !status.is_success() {
            return Err(format!("speech-to-text: HTTP {status}: {value}"));
        }
        Ok(value.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string())
    }

    /// Speak `text`: its audio at the wire rate, chunk by chunk as it is made,
    /// until done or `out` is closed (the item was cancelled).
    pub async fn speak(&self, text: &str, voice: Option<&str>, out: &mpsc::Sender<Vec<i16>>) -> Result<(), String> {
        let base = self.base(TTS_SERVICE).await?;
        let mut body = serde_json::json!({ "input": text, "response_format": "pcm" });
        if let Some(v) = voice {
            body["voice"] = serde_json::Value::String(v.to_string());
        }
        let resp = self.http.post(format!("{base}/v1/audio/speech")).json(&body).timeout(Duration::from_secs(120)).send().await.map_err(|e| format!("text-to-speech: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(format!("text-to-speech: HTTP {status}: {}", resp.text().await.unwrap_or_default()));
        }
        let rate = resp.headers().get("x-sample-rate").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u32>().ok()).unwrap_or(WIRE_RATE);
        let mut resample = Resampler::new(rate, WIRE_RATE);
        let mut stream = resp.bytes_stream();
        let mut odd: Option<u8> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("text-to-speech stream: {e}"))?;
            let mut bytes = Vec::with_capacity(chunk.len() + 1);
            if let Some(b) = odd.take() {
                bytes.push(b);
            }
            bytes.extend_from_slice(&chunk);
            if bytes.len() % 2 == 1 {
                odd = bytes.pop();
            }
            let samples = resample.process(&audio::samples(&bytes));
            if !samples.is_empty() && out.send(samples).await.is_err() {
                // Cancelled: stop reading, which ends the request.
                return Ok(());
            }
        }
        Ok(())
    }
}
