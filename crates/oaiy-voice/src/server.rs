//! The HTTP server: Aokie's voice-server routes on OAIY's std HTTP.
//!
//! - `GET /v1/models`: the loaded models (the desktop's health probe).
//! - `GET /health`: mode, device and readiness.
//! - `POST /v1/audio/transcriptions`: JSON `{"audio" | "file": base64 WAV,
//!   "response_format": "json" | "text" | "verbose_json"}`, or a multipart
//!   form with a `file` part; answers `{"text": ...}` (or plain text).
//! - `POST /v1/audio/speech`: JSON `{"input": text, "voice"?: name,
//!   "response_format": "pcm" | "wav"}`. `pcm` streams raw 16-bit mono PCM
//!   (`X-Sample-Rate: 24000`) as it is made, and stops when the client goes
//!   (a call's barge-in); `wav` answers the whole line. 404 in stt mode, as
//!   Aokie answers.
//! - `GET /v1/audio/voices`: the voices a request may name, and the default.
//!
//! The models load before the port opens, so a probe that answers means they
//! are ready. Transcriptions run one at a time, as do spoken lines (the GPU
//! is shared by the whole request anyway); each connection has its own
//! thread, so `/v1/models` answers while either runs.

use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

/// A speech-to-text engine shared by the server and the voices it hears.
pub type SharedStt = Arc<Mutex<Box<dyn SpeechToText>>>;

/// The rate spoken audio is handed out at.
pub const SPEECH_RATE: u32 = 24_000;
/// The longest line a request may ask for, in characters.
const MAX_INPUT: usize = 4_096;

/// A text-to-speech engine as the server uses it.
pub trait TextToSpeech: Send + Sync {
    /// The model id `/v1/models` lists.
    fn model(&self) -> String;
    /// The voices a request may name, and the one used when it names none.
    fn voices(&self) -> (Vec<String>, Option<String>);
    /// Whether `voice` can be spoken in (before anything is made).
    fn check_voice(&self, voice: Option<&str>) -> Result<(), String>;
    /// Speak `text`, handing out 16-bit mono PCM at [`SPEECH_RATE`] as it is
    /// made; `on_chunk` returning false (or `cancel` set) stops it.
    fn speak(&self, text: &str, voice: Option<&str>, cancel: Arc<AtomicBool>, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String>;
}

pub struct Server {
    pub mode: Mode,
    /// The model id `/v1/models` lists.
    pub stt_id: String,
    /// Where it runs ("cuda:1 f16", "cpu f32"), for `/health`.
    pub device: String,
    stt: Option<SharedStt>,
    tts: Option<Arc<dyn TextToSpeech>>,
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
    pub fn new(mode: Mode, stt: Option<SharedStt>, stt_id: impl Into<String>, device: impl Into<String>, tts: Option<Arc<dyn TextToSpeech>>) -> Self {
        Self { mode, stt_id: stt_id.into(), device: device.into(), stt, tts }
    }

    /// Answer one request on `w`: a spoken line streams, the rest answer whole.
    /// Returns whether the connection may be kept open.
    pub fn serve(&self, req: &Request, w: &mut impl Write) -> io::Result<bool> {
        if req.method == "POST" && req.route() == "/v1/audio/speech" && self.mode.tts() {
            return self.speech(req, w);
        }
        let r = self.handle(req);
        http::respond(w, r.status, r.content_type, &r.body, true)?;
        Ok(true)
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
                // Spoken lines are answered by `serve`, which can stream.
                Response::error(400, "invalid_request_error", "speech is answered as a stream")
            }
            ("GET", "/v1/audio/voices") => match &self.tts {
                Some(tts) => {
                    let (voices, default) = tts.voices();
                    Response::json(200, Json::obj([("voices", Json::Arr(voices.into_iter().map(Json::str).collect())), ("default", default.map(Json::str).unwrap_or(Json::Null))]))
                }
                None => Response::error(404, "invalid_request_error", "this voice server does not speak (start with --mode tts or --mode both)"),
            },
            _ => Response::error(404, "invalid_request_error", "unknown route"),
        }
    }

    fn models(&self) -> Response {
        let mut data = Vec::new();
        if self.mode.stt() {
            data.push(Json::obj([("id", Json::str(self.stt_id.clone())), ("object", Json::str("model")), ("owned_by", Json::str("oaiy"))]));
        }
        if let (true, Some(tts)) = (self.mode.tts(), &self.tts) {
            data.push(Json::obj([("id", Json::str(tts.model())), ("object", Json::str("model")), ("owned_by", Json::str("oaiy"))]));
        }
        Response::json(200, Json::obj([("object", Json::str("list")), ("data", Json::Arr(data))]))
    }

    fn health(&self) -> Response {
        let stt_ready = self.stt.is_some();
        let tts_ready = self.tts.is_some();
        let ok = (!self.mode.stt() || stt_ready) && (!self.mode.tts() || tts_ready);
        Response::json(
            200,
            Json::obj([
                ("status", Json::str(if ok { "ok" } else { "degraded" })),
                ("mode", Json::str(self.mode.as_str())),
                ("stt", Json::Bool(stt_ready)),
                ("tts", Json::Bool(tts_ready)),
                ("model", Json::str(self.stt_id.clone())),
                ("device", Json::str(self.device.clone())),
            ]),
        )
    }

    /// A spoken line: checked first (a bad request costs nothing), then made.
    /// `pcm` streams from the first piece; an error before it is an error
    /// answer, one after it ends the stream early.
    fn speech(&self, req: &Request, w: &mut impl Write) -> io::Result<bool> {
        let json = match Json::parse(&req.body) {
            Ok(j) => j,
            Err(e) => return answer(w, Response::bad(format!("malformed JSON: {e}"))),
        };
        let text = json.get("input").and_then(Json::as_str).map(str::trim).unwrap_or("");
        if text.is_empty() {
            return answer(w, Response::bad("missing input: the text to speak"));
        }
        if text.chars().count() > MAX_INPUT {
            return answer(w, Response::bad(format!("input is longer than {MAX_INPUT} characters: send it a sentence or two at a time")));
        }
        let voice = json.get("voice").and_then(Json::as_str).map(str::trim).filter(|v| !v.is_empty());
        let format = match json.get("response_format").and_then(Json::as_str).map(str::trim) {
            None | Some("") => "wav",
            Some(f) if f.eq_ignore_ascii_case("wav") => "wav",
            Some(f) if f.eq_ignore_ascii_case("pcm") => "pcm",
            Some(other) => return answer(w, Response::bad(format!("only response_format \"pcm\" (streamed, 24 kHz 16-bit mono) or \"wav\" is supported, got {other:?}"))),
        };
        let Some(tts) = &self.tts else { return answer(w, Response::error(503, "server_error", "no text-to-speech model is loaded")) };
        if let Err(e) = tts.check_voice(voice) {
            return answer(w, Response::bad(e));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        if format == "wav" {
            let mut all: Vec<i16> = Vec::new();
            return match tts.speak(text, voice, cancel, &mut |pcm| {
                all.extend_from_slice(pcm);
                true
            }) {
                Ok(()) => {
                    let samples: Vec<f32> = all.iter().map(|&s| s as f32 / 32_768.0).collect();
                    answer(w, Response { status: 200, content_type: "audio/wav", body: audio::write_wav(&samples, SPEECH_RATE) })
                }
                Err(e) => answer(w, Response::error(500, "server_error", format!("speech failed: {e}"))),
            };
        }
        let (mut streaming, mut gone, mut first) = (false, false, None);
        let result = tts.speak(text, voice, cancel, &mut |pcm| {
            if gone {
                return false;
            }
            let sent = (|| -> io::Result<()> {
                if !streaming {
                    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: {SPEECH_RATE}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n")?;
                    streaming = true;
                    first = Some(started.elapsed());
                }
                let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                write!(w, "{:x}\r\n", bytes.len())?;
                w.write_all(&bytes)?;
                w.write_all(b"\r\n")?;
                w.flush()
            })();
            gone = sent.is_err();
            !gone
        });
        if gone {
            eprintln!("[oaiy-voice] the listener went: the line stopped");
            return Ok(false);
        }
        match (streaming, result) {
            (false, Err(e)) => answer(w, Response::error(500, "server_error", format!("speech failed: {e}"))),
            (true, Err(e)) => {
                // Part of the line was sent: end it there (the client has what came).
                eprintln!("[oaiy-voice] speech failed part way: {e}");
                w.write_all(b"0\r\n\r\n")?;
                w.flush()?;
                Ok(true)
            }
            (streaming, Ok(())) => {
                if !streaming {
                    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: {SPEECH_RATE}\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n")?;
                }
                w.write_all(b"0\r\n\r\n")?;
                w.flush()?;
                if let Some(first) = first {
                    eprintln!("[oaiy-voice] spoke {} chars, first audio after {:.0} ms, all in {:.0} ms", text.len(), first.as_secs_f64() * 1e3, started.elapsed().as_secs_f64() * 1e3);
                }
                Ok(true)
            }
        }
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

/// A whole answer; the connection stays open.
fn answer<W: Write>(w: &mut W, r: Response) -> io::Result<bool> {
    http::respond(w, r.status, r.content_type, &r.body, true).map(|_| true)
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
                let keep = server.serve(req, w)?;
                w.flush()?;
                Ok(keep)
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

    /// Speaks each character of its input as one sample of its code; "fail" fails.
    struct FakeTts;

    impl TextToSpeech for FakeTts {
        fn model(&self) -> String {
            "qwen3-tts-test".into()
        }
        fn voices(&self) -> (Vec<String>, Option<String>) {
            (vec!["phone".into()], Some("phone".into()))
        }
        fn check_voice(&self, voice: Option<&str>) -> Result<(), String> {
            match voice {
                None | Some("phone") => Ok(()),
                Some(v) => Err(format!("no voice called {v:?}")),
            }
        }
        fn speak(&self, text: &str, _voice: Option<&str>, _cancel: Arc<AtomicBool>, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String> {
            if text == "fail" {
                return Err("the engine broke".into());
            }
            for c in text.chars() {
                if !on_chunk(&[c as i16]) {
                    break;
                }
            }
            Ok(())
        }
    }

    fn server(mode: Mode) -> Server {
        let stt: SharedStt = Arc::new(Mutex::new(Box::new(Fake)));
        Server::new(mode, Some(stt), "parakeet-test", "cpu f32", mode.tts().then(|| Arc::new(FakeTts) as Arc<dyn TextToSpeech>))
    }

    /// A request answered through `serve`, as the client reads it.
    fn served(s: &Server, req: &Request) -> String {
        let mut out = Vec::new();
        s.serve(req, &mut out).unwrap();
        String::from_utf8_lossy(&out).into_owned()
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
    fn modes_gate_routes() {
        let wav = audio::write_wav(&[0.0; 16], 16_000);
        let body = format!("{{\"audio\":\"{}\"}}", b64(&wav)).into_bytes();
        assert_eq!(server(Mode::Tts).handle(&post("/v1/audio/transcriptions", "application/json", body)).status, 404);
        let speech = post("/v1/audio/speech", "application/json", b"{\"input\":\"hi\"}".to_vec());
        assert!(served(&server(Mode::Stt), &speech).starts_with("HTTP/1.1 404"));
        assert_eq!(server(Mode::Stt).handle(&Request::new("GET", "/nope", vec![], vec![])).status, 404);
        for mode in [Mode::Stt, Mode::Tts, Mode::Both] {
            let h = json(&server(mode).handle(&Request::new("GET", "/health", vec![], vec![])));
            assert_eq!(h.get("status").and_then(Json::as_str), Some("ok"), "{mode:?}");
        }
        let models = json(&server(Mode::Both).handle(&Request::new("GET", "/v1/models", vec![], vec![])));
        assert_eq!(models.get("data").and_then(|d| d.at(1)).and_then(|m| m.get("id")).and_then(Json::as_str), Some("qwen3-tts-test"));
        let voices = json(&server(Mode::Both).handle(&Request::new("GET", "/v1/audio/voices", vec![], vec![])));
        assert_eq!(voices.get("default").and_then(Json::as_str), Some("phone"));
    }

    #[test]
    fn speech_streams_pcm_at_24khz_or_answers_a_whole_wav() {
        let s = server(Mode::Both);
        let pcm = served(&s, &post("/v1/audio/speech", "application/json", br#"{"input":"hey","response_format":"pcm"}"#.to_vec()));
        assert!(pcm.starts_with("HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: 24000\r\n"), "{pcm}");
        // One chunk of two bytes per character, then the end.
        assert!(pcm.ends_with("\r\n\r\n2\r\nh\0\r\n2\r\ne\0\r\n2\r\ny\0\r\n0\r\n\r\n"), "{pcm:?}");
        let mut out = Vec::new();
        s.serve(&post("/v1/audio/speech", "application/json", br#"{"input":"hi","voice":"phone"}"#.to_vec()), &mut out).unwrap();
        let body = &out[out.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4..];
        let wav = audio::parse_wav(body).unwrap();
        assert_eq!((wav.sample_rate, wav.samples.len()), (24_000, 2));
    }

    #[test]
    fn bad_speech_requests_are_answered_before_anything_is_made() {
        let s = server(Mode::Tts);
        for (body, status) in [
            (r#"{"#, "400"),
            (r#"{"input":"  "}"#, "400"),
            (r#"{"input":"hi","response_format":"mp3"}"#, "400"),
            (r#"{"input":"hi","voice":"alloy"}"#, "400"),
            (r#"{"input":"fail","response_format":"pcm"}"#, "500"),
        ] {
            let r = served(&s, &post("/v1/audio/speech", "application/json", body.as_bytes().to_vec()));
            assert!(r.starts_with(&format!("HTTP/1.1 {status}")), "{body}: {r}");
        }
        let long = format!(r#"{{"input":"{}"}}"#, "a".repeat(MAX_INPUT + 1));
        assert!(served(&s, &post("/v1/audio/speech", "application/json", long.into_bytes())).starts_with("HTTP/1.1 400"));
        let none = Server::new(Mode::Tts, None, "", "none", None);
        assert!(served(&none, &post("/v1/audio/speech", "application/json", br#"{"input":"hi"}"#.to_vec())).starts_with("HTTP/1.1 503"));
    }

    #[test]
    fn a_listener_that_goes_stops_the_line() {
        /// Takes the head and one chunk, then fails like a closed connection.
        struct Hangs(usize);
        impl Write for Hangs {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                self.0 += 1;
                if self.0 > 4 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "gone"));
                }
                Ok(b.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let keep = server(Mode::Tts).serve(&post("/v1/audio/speech", "application/json", br#"{"input":"a long line","response_format":"pcm"}"#.to_vec()), &mut Hangs(0)).unwrap();
        assert!(!keep, "a stream cut short closes the connection");
    }
}
