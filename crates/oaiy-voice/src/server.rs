//! The HTTP server: Aokie's voice-server routes on OAIY's std HTTP.
//!
//! - `GET /v1/models`: the loaded models (the desktop's health probe).
//! - `GET /health`: mode, device and readiness.
//! - `POST /v1/audio/transcriptions`: JSON `{"audio" | "file": base64 WAV,
//!   "response_format": "json" | "text" | "verbose_json"}`, or a multipart
//!   form with a `file` part; answers `{"text": ...}` (or plain text).
//! - `POST /v1/audio/speech`: text-to-speech, not in this server yet (501 in
//!   tts mode; 404 in stt mode, as Aokie answers).
//!
//! The model loads before the port opens, so a probe that answers means the
//! model is ready. Transcriptions run one at a time (the GPU is shared by
//! the whole request anyway); each connection has its own thread, so
//! `/v1/models` answers while a transcription runs.

use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use oaiy_engine::http::{self, Request};
use oaiy_engine::json::Json;

use crate::audio;
use crate::cli::Mode;

/// Connections served at once; past this a client gets an immediate 503.
const MAX_CONNECTIONS: usize = 32;

/// A speech-to-text engine as the server uses it.
pub trait SpeechToText: Send {
    /// The rate the engine takes audio at.
    fn sample_rate(&self) -> usize;
    /// Text of mono audio at [`sample_rate`](Self::sample_rate).
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String>;
}

pub struct Server {
    pub mode: Mode,
    /// The model id `/v1/models` lists.
    pub stt_id: String,
    /// Where it runs ("cuda:1 f16", "cpu f32"), for `/health`.
    pub device: String,
    stt: Option<Mutex<Box<dyn SpeechToText>>>,
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: Json) -> Self {
        Self { status, content_type: "application/json", body: value.to_json().into_bytes() }
    }

    fn error(status: u16, kind: &str, message: impl Into<String>) -> Self {
        Self::json(status, Json::obj([("error", Json::obj([("message", Json::str(message)), ("type", Json::str(kind))]))]))
    }

    fn bad(message: impl Into<String>) -> Self {
        Self::error(400, "invalid_request_error", message)
    }
}

impl Server {
    pub fn new(mode: Mode, stt: Option<Box<dyn SpeechToText>>, stt_id: impl Into<String>, device: impl Into<String>) -> Self {
        Self { mode, stt_id: stt_id.into(), device: device.into(), stt: stt.map(Mutex::new) }
    }

    pub fn handle(&self, req: &Request) -> Response {
        match (req.method.as_str(), req.route()) {
            ("GET", "/v1/models") => self.models(),
            ("GET", "/health") => self.health(),
            ("POST", "/v1/audio/transcriptions") => {
                if !self.mode.stt() {
                    return Response::error(404, "invalid_request_error", "this voice server runs in tts mode - transcriptions are not served here (start with --mode stt or --mode both)");
                }
                self.transcriptions(req)
            }
            ("POST", "/v1/audio/speech") => {
                if !self.mode.tts() {
                    return Response::error(404, "invalid_request_error", "this voice server runs in stt mode - speech synthesis is not served here (start with --mode tts or --mode both)");
                }
                // The seam for text-to-speech: the route, request checks and
                // pcm streaming (X-Sample-Rate) go here with the engine.
                Response::error(501, "not_implemented", "text-to-speech is not in oaiy-voice yet")
            }
            _ => Response::error(404, "invalid_request_error", "unknown route"),
        }
    }

    fn models(&self) -> Response {
        let mut data = Vec::new();
        if self.mode.stt() {
            data.push(Json::obj([("id", Json::str(self.stt_id.clone())), ("object", Json::str("model")), ("owned_by", Json::str("oaiy"))]));
        }
        Response::json(200, Json::obj([("object", Json::str("list")), ("data", Json::Arr(data))]))
    }

    fn health(&self) -> Response {
        let stt_ready = self.stt.is_some();
        let ok = (!self.mode.stt() || stt_ready) && !self.mode.tts();
        Response::json(
            200,
            Json::obj([
                ("status", Json::str(if ok { "ok" } else { "degraded" })),
                ("mode", Json::str(self.mode.as_str())),
                ("stt", Json::Bool(stt_ready)),
                ("tts", Json::Bool(false)),
                ("model", Json::str(self.stt_id.clone())),
                ("device", Json::str(self.device.clone())),
            ]),
        )
    }

    fn transcriptions(&self, req: &Request) -> Response {
        let content_type = req.header("content-type").unwrap_or("").to_string();
        let lower = content_type.to_ascii_lowercase();
        let (wav, format) = if lower.starts_with("multipart/form-data") {
            match audio::parse_multipart(&req.body, &content_type) {
                Ok(form) => match form.file.clone() {
                    Some(f) => (f, form.field("response_format").map(str::to_string)),
                    None => return Response::bad("multipart request has no file part"),
                },
                Err(e) => return Response::bad(e),
            }
        } else if lower.is_empty() || lower.starts_with("application/json") {
            let json = match Json::parse(&req.body) {
                Ok(j) => j,
                Err(e) => return Response::bad(format!("malformed JSON: {e}")),
            };
            let Some(encoded) = json.get("audio").or_else(|| json.get("file")).and_then(Json::as_str) else {
                return Response::bad("missing audio or file field");
            };
            match audio::decode_base64_field(encoded) {
                Ok(b) => (b, json.get("response_format").and_then(Json::as_str).map(str::to_string)),
                Err(e) => return Response::bad(format!("invalid audio field: {e}")),
            }
        } else {
            return Response::bad("transcriptions require application/json or multipart/form-data");
        };
        // Checked before inference: a bad format never costs a transcription.
        let format = match format.as_deref().map(str::trim) {
            None | Some("") => "json",
            Some(f) if f.eq_ignore_ascii_case("json") => "json",
            Some(f) if f.eq_ignore_ascii_case("text") => "text",
            Some(f) if f.eq_ignore_ascii_case("verbose_json") => "verbose_json",
            Some(other) => return Response::bad(format!("only response_format \"json\", \"text\" or \"verbose_json\" is supported, got {other:?}")),
        };
        let wav = match audio::parse_wav(&wav) {
            Ok(w) => w,
            Err(e) => return Response::bad(format!("invalid WAV: {e}")),
        };
        let Some(stt) = &self.stt else { return Response::error(503, "server_error", "no speech-to-text model is loaded") };
        let mut stt = match stt.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let rate = stt.sample_rate();
        let samples = audio::resample(&wav.samples, wav.sample_rate as usize, rate);
        let seconds = samples.len() as f64 / rate as f64;
        let started = Instant::now();
        let text = match stt.transcribe(&samples) {
            Ok(t) => t,
            Err(e) => return Response::error(500, "server_error", format!("transcription failed: {e}")),
        };
        drop(stt);
        eprintln!("[oaiy-voice] transcribed {seconds:.2} s in {:.0} ms: {} chars", started.elapsed().as_secs_f64() * 1e3, text.len());
        match format {
            "text" => Response { status: 200, content_type: "text/plain; charset=utf-8", body: text.into_bytes() },
            "verbose_json" => Response::json(200, Json::obj([("task", Json::str("transcribe")), ("language", Json::str("english")), ("duration", Json::Num(seconds)), ("text", Json::str(text))])),
            _ => Response::json(200, Json::obj([("text", Json::str(text))])),
        }
    }
}

fn write(w: &mut TcpStream, r: &Response, keep_alive: bool) -> io::Result<()> {
    http::respond(w, r.status, r.content_type, &r.body, keep_alive)
}

struct Permit(Arc<AtomicUsize>);

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Serve until the process ends.
pub fn run(server: Server, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    eprintln!("[oaiy-voice] listening on http://{host}:{port} ({} mode)", server.mode.as_str());
    let server = Arc::new(server);
    let connections = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        if connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            connections.fetch_sub(1, Ordering::SeqCst);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            let _ = write(&mut stream, &Response::error(503, "server_error", "voice server connection limit reached"), false);
            continue;
        }
        let permit = Permit(Arc::clone(&connections));
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            let _permit = permit;
            http::serve(stream, |req, w| {
                let r = server.handle(req);
                write(w, &r, true)?;
                w.flush()?;
                Ok(true)
            });
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Echoes the audio's length in samples.
    struct Fake;

    impl SpeechToText for Fake {
        fn sample_rate(&self) -> usize {
            16_000
        }
        fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
            Ok(format!("{} samples", samples.len()))
        }
    }

    fn server(mode: Mode) -> Server {
        Server::new(mode, Some(Box::new(Fake)), "parakeet-test", "cpu f32")
    }

    fn b64(bytes: &[u8]) -> String {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in bytes.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            for k in 0..4 {
                out.push(if k <= c.len() { A[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
            }
        }
        out
    }

    fn post(path: &str, content_type: &str, body: Vec<u8>) -> Request {
        Request::new("POST", path, vec![("Content-Type".into(), content_type.into())], body)
    }

    fn json(r: &Response) -> Json {
        Json::parse(&r.body).unwrap()
    }

    #[test]
    fn models_lists_the_stt_model() {
        let r = server(Mode::Stt).handle(&Request::new("GET", "/v1/models", vec![], vec![]));
        assert_eq!(r.status, 200);
        let j = json(&r);
        assert_eq!(j.get("data").and_then(|d| d.at(0)).and_then(|m| m.get("id")).and_then(Json::as_str), Some("parakeet-test"));
    }

    #[test]
    fn json_transcription_answers_text() {
        let wav = audio::write_wav(&[0.0; 1600], 16_000);
        let body = format!("{{\"audio\":\"{}\",\"response_format\":\"json\"}}", b64(&wav));
        let r = server(Mode::Stt).handle(&post("/v1/audio/transcriptions", "application/json", body.into_bytes()));
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        assert_eq!(json(&r).get("text").and_then(Json::as_str), Some("1600 samples"));
    }

    #[test]
    fn other_rates_are_resampled_and_text_format_is_plain() {
        let wav = audio::write_wav(&[0.0; 2400], 24_000);
        let body = format!("{{\"file\":\"data:audio/wav;base64,{}\",\"response_format\":\"text\"}}", b64(&wav));
        let r = server(Mode::Both).handle(&post("/v1/audio/transcriptions", "application/json; charset=utf-8", body.into_bytes()));
        assert_eq!((r.status, r.content_type), (200, "text/plain; charset=utf-8"));
        assert_eq!(r.body, b"1600 samples");
    }

    #[test]
    fn multipart_transcription() {
        let wav = audio::write_wav(&[0.0; 800], 16_000);
        let mut body = b"--b0\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
        body.extend_from_slice(&wav);
        body.extend_from_slice(b"\r\n--b0\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--b0--\r\n");
        let r = server(Mode::Stt).handle(&post("/v1/audio/transcriptions", "multipart/form-data; boundary=b0", body));
        assert_eq!(r.status, 200);
        assert_eq!(json(&r).get("text").and_then(Json::as_str), Some("800 samples"));
    }

    #[test]
    fn bad_requests_are_400_before_inference() {
        let s = server(Mode::Stt);
        for (ct, body) in [
            ("application/json", b"{".to_vec()),
            ("application/json", b"{}".to_vec()),
            ("application/json", b"{\"audio\":\"!!!\"}".to_vec()),
            ("application/json", format!("{{\"audio\":\"{}\"}}", b64(b"not a wav")).into_bytes()),
            ("application/json", format!("{{\"audio\":\"{}\",\"response_format\":\"srt\"}}", b64(&audio::write_wav(&[0.0; 4], 16_000))).into_bytes()),
            ("text/plain", b"hello".to_vec()),
        ] {
            let r = s.handle(&post("/v1/audio/transcriptions", ct, body));
            assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
            assert!(json(&r).get("error").and_then(|e| e.get("message")).is_some());
        }
    }

    #[test]
    fn modes_gate_routes_and_speech_is_reserved() {
        let wav = audio::write_wav(&[0.0; 16], 16_000);
        let body = format!("{{\"audio\":\"{}\"}}", b64(&wav)).into_bytes();
        assert_eq!(server(Mode::Tts).handle(&post("/v1/audio/transcriptions", "application/json", body)).status, 404);
        let speech = || post("/v1/audio/speech", "application/json", b"{\"input\":\"hi\"}".to_vec());
        assert_eq!(server(Mode::Stt).handle(&speech()).status, 404);
        assert_eq!(server(Mode::Tts).handle(&speech()).status, 501);
        assert_eq!(server(Mode::Stt).handle(&Request::new("GET", "/nope", vec![], vec![])).status, 404);
        let h = json(&server(Mode::Stt).handle(&Request::new("GET", "/health", vec![], vec![])));
        assert_eq!(h.get("status").and_then(Json::as_str), Some("ok"));
    }
}
