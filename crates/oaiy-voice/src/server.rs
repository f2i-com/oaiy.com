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
//!
//! A web page in this computer's browser can reach the server too, so it is
//! gated as `oaiy-llm-server` is ([`http::loopback_gate`]): on a loopback
//! address it answers only a `Host` naming loopback (DNS rebinding), and no
//! page but this computer's own; with a key (`OAIY_VOICE_API_KEY`) every
//! request but the probes' must carry it. Its answers allow no other origin
//! (no `Access-Control-Allow-Origin`).

use std::cell::{Cell, RefCell};
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use oaiy_engine::http::{self, Refusal, Request};
use oaiy_engine::json::Json;

use crate::audio;
use crate::cli::Mode;

/// Connections served at once; past this a client gets an immediate 503.
const MAX_CONNECTIONS: usize = 32;
/// Connections past [`MAX_CONNECTIONS`] kept for a probe (`GET /v1/models`, `/health`), one request each: connections
/// left open between requests (300 s each) could fill the rest, and a supervisor whose probe went unanswered would
/// start the server again mid-call.
const PROBE_SLOTS: usize = 4;
/// The routes a probe may ask for, which a key does not guard either (they tell only which models are loaded: OAIY
/// Desktop's probe and its health check ask them without one).
const OPEN_ROUTES: [&str; 3] = ["/", "/health", "/v1/models"];

/// The largest body each route takes, refused on its head (before a byte of it is read). A transcription's: ten
/// minutes of 16 kHz 16-bit audio (the longest heard, [`audio::MAX_SECONDS`]) in base64, 25.6 MB, with room; a spoken
/// line's: [`MAX_INPUT`] characters of JSON. (A chunked body says nothing of its size first: it is held to
/// `oaiy-engine`'s 64 MiB as it comes.)
const MAX_TRANSCRIPTION_BODY: usize = 32 << 20;
const MAX_SPEECH_BODY: usize = 1 << 20;
const MAX_OTHER_BODY: usize = 64 << 10;

/// Transcriptions whose bodies are read and decoded at once (each is some 100 MB at the most, as base64, a WAV and
/// samples); more wait their turn before their bodies are read, and give up after [`TURN_WAIT`].
const TRANSCRIBING: usize = 3;
const TURN_WAIT: Duration = Duration::from_secs(60);
/// How long a long transcription, between its passes, lets the short ones waiting for the engine go first.
const MAKE_WAY: Duration = Duration::from_secs(30);

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
    /// made; `on_chunk` returning false, `gone` answering true (asked while the
    /// line waits its turn and is made: its listener went) or `cancel` set stops it.
    fn speak(&self, text: &str, voice: Option<&str>, cancel: Arc<AtomicBool>, gone: &dyn Fn() -> bool, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String>;
    /// Make `text` whole and keep it to be said from memory when it is asked for (a caller's greeting, made as the
    /// call settles): how like its voice it came out and its seed, where it was made. Nothing, where the engine
    /// keeps nothing. `gone` as for [`speak`](Self::speak).
    fn prepare(&self, text: &str, voice: Option<&str>, gone: &dyn Fn() -> bool) -> Result<Option<(f64, u64)>, String> {
        let _ = (text, voice, gone);
        Ok(None)
    }
    /// Whether the engine is still there to speak (false once its thread has ended).
    fn alive(&self) -> bool {
        true
    }
}

pub struct Server {
    pub mode: Mode,
    /// The model id `/v1/models` lists.
    pub stt_id: String,
    /// Where it runs ("cuda:1 f16", "cpu f32"), for `/health`.
    pub device: String,
    stt: Option<SharedStt>,
    /// The rate the speech-to-text engine takes (read once: audio is resampled to it before the engine is waited on).
    stt_rate: usize,
    tts: Option<Arc<dyn TextToSpeech>>,
    /// The key every request but a probe's must carry ([`OPEN_ROUTES`]), if any.
    key: Option<String>,
    /// Whether it listens on loopback alone: a request must then name loopback as its `Host`.
    loopback: bool,
    /// Turns at reading and decoding a transcription's body ([`TRANSCRIBING`]).
    turns: Arc<Turns>,
    /// Short transcriptions (one pass) waiting for the engine: a long one lets them go first between its passes.
    waiting: AtomicUsize,
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

/// Turns taken from a count, waited for when none is free.
struct Turns {
    free: Mutex<usize>,
    freed: Condvar,
}

/// A turn taken from [`Turns`], given back when dropped.
pub struct Turn(Arc<Turns>);

impl Turns {
    fn new(n: usize) -> Arc<Turns> {
        Arc::new(Turns { free: Mutex::new(n), freed: Condvar::new() })
    }

    /// A turn, waited for up to `within`; none if none came free by then.
    fn take(self: &Arc<Self>, within: Duration) -> Option<Turn> {
        let until = Instant::now() + within;
        let mut free = self.free.lock().unwrap_or_else(|p| p.into_inner());
        while *free == 0 {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            free = self.freed.wait_timeout(free, left).unwrap_or_else(|p| p.into_inner()).0;
        }
        *free -= 1;
        Some(Turn(Arc::clone(self)))
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(|p| p.into_inner()) += 1;
        self.0.freed.notify_one();
    }
}

/// A short transcription counted as waiting for the engine until it has it.
struct Waiting<'a>(&'a AtomicUsize);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Server {
    pub fn new(mode: Mode, stt: Option<SharedStt>, stt_id: impl Into<String>, device: impl Into<String>, tts: Option<Arc<dyn TextToSpeech>>) -> Self {
        let stt_rate = stt.as_ref().map_or(16_000, |s| s.lock().unwrap_or_else(|p| p.into_inner()).sample_rate());
        Self {
            mode,
            stt_id: stt_id.into(),
            device: device.into(),
            stt,
            stt_rate,
            tts,
            key: None,
            loopback: true,
            turns: Turns::new(TRANSCRIBING),
            waiting: AtomicUsize::new(0),
        }
    }

    /// The key requests must carry (`Authorization: Bearer KEY`), none for none.
    pub fn with_key(mut self, key: Option<String>) -> Self {
        self.key = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
        self
    }

    /// A request's head, before its body is read: refused (by [`http::loopback_gate`], for its size, or on a
    /// connection kept for probes, `probe_only`, for being no probe), or let in, with its turn at a transcription
    /// where it is one (waited for here, before its body is read: a body waiting in the socket costs nothing).
    pub fn admit(&self, req: &Request, probe_only: bool) -> Result<Option<Turn>, Refusal> {
        if let Some(refused) = http::loopback_gate(self.key.as_deref(), self.loopback, req, &OPEN_ROUTES) {
            return Err(refused);
        }
        let route = req.route();
        if probe_only && !(req.method == "GET" && OPEN_ROUTES.contains(&route)) {
            return Err(Refusal::new(503, "voice server connection limit reached", "busy"));
        }
        let transcription = req.method == "POST" && route == "/v1/audio/transcriptions";
        let cap = match (req.method.as_str(), route) {
            _ if transcription => MAX_TRANSCRIPTION_BODY,
            ("POST", "/v1/audio/speech") => MAX_SPEECH_BODY,
            _ => MAX_OTHER_BODY,
        };
        if req.header("content-length").and_then(|l| l.trim().parse::<u64>().ok()).is_some_and(|len| len > cap as u64) {
            return Err(Refusal::new(413, &format!("the body is larger than this route takes ({cap} bytes)"), "too_large"));
        }
        if !(transcription && self.mode.stt()) {
            return Ok(None);
        }
        match self.turns.take(TURN_WAIT) {
            Some(turn) => Ok(Some(turn)),
            None => Err(Refusal::new(503, "the voice server is busy with other transcriptions: try again shortly", "busy")),
        }
    }

    /// Answer one request on `w`: a spoken line streams, the rest answer whole.
    /// Returns whether the connection may be kept open.
    pub fn serve(&self, req: &Request, w: &mut impl Write) -> io::Result<bool> {
        self.serve_watched(req, w, &|| false)
    }

    /// [`serve`](Self::serve), `gone` telling whether the client has gone (asked while a spoken line waits for the
    /// engine and is made, which then stops: the one engine is not held for a line nobody hears).
    pub fn serve_watched(&self, req: &Request, w: &mut impl Write, gone: &dyn Fn() -> bool) -> io::Result<bool> {
        if req.method == "POST" && req.route() == "/v1/audio/speech" && self.mode.tts() {
            return self.speech(req, w, gone);
        }
        let r = self.handle(req);
        respond(w, &r, true)?;
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

    /// Whether the speech engine this server should have is gone (its thread ended): it answers its probes 503 then.
    fn speech_gone(&self) -> bool {
        self.mode.tts() && self.tts.as_ref().is_some_and(|t| !t.alive())
    }

    fn models(&self) -> Response {
        if self.speech_gone() {
            return Response::error(503, "server_error", "the speech engine has stopped");
        }
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
        let tts_ready = self.tts.as_ref().is_some_and(|t| t.alive());
        let ok = (!self.mode.stt() || stt_ready) && (!self.mode.tts() || tts_ready);
        Response::json(
            if self.speech_gone() { 503 } else { 200 },
            Json::obj([
                ("status", Json::str(if ok { "ok" } else if self.speech_gone() { "failed" } else { "degraded" })),
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
    /// answer, one after it cuts the stream off (no end: the client knows the
    /// line did not end). A client that went closes the connection.
    fn speech(&self, req: &Request, w: &mut impl Write, gone: &dyn Fn() -> bool) -> io::Result<bool> {
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
        if !tts.alive() {
            return answer(w, Response::error(503, "server_error", "the speech engine has stopped"));
        }
        if let Err(e) = tts.check_voice(voice) {
            return answer(w, Response::bad(e));
        }
        // (the client's going noted, so nothing more is written to it and its connection is closed)
        let left = Cell::new(false);
        let gone = || {
            let g = gone();
            left.set(left.get() || g);
            g
        };
        // ("prepare": made whole and kept, to be said from memory when asked for; nothing is said now)
        if json.get("prepare").and_then(Json::as_bool) == Some(true) {
            let made = tts.prepare(text, voice, &gone);
            if left.get() {
                return Ok(false);
            }
            return match made {
                Ok(made) => {
                    let body = match made {
                        Some((likeness, seed)) => Json::obj([("prepared", Json::Bool(true)), ("likeness", Json::Num(likeness)), ("seed", Json::Int(seed as i64))]),
                        None => Json::obj([("prepared", Json::Bool(false))]),
                    };
                    answer(w, Response::json(200, body))
                }
                Err(e) => answer(w, Response::error(500, "server_error", e)),
            };
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        if format == "wav" {
            let mut all: Vec<i16> = Vec::new();
            let spoken = tts.speak(text, voice, cancel, &gone, &mut |pcm| {
                all.extend_from_slice(pcm);
                true
            });
            if left.get() {
                return Ok(false);
            }
            return match spoken {
                Ok(()) => {
                    let samples: Vec<f32> = all.iter().map(|&s| s as f32 / 32_768.0).collect();
                    answer(w, Response { status: 200, content_type: "audio/wav", body: audio::write_wav(&samples, SPEECH_RATE) })
                }
                Err(e) => answer(w, Response::error(500, "server_error", format!("speech failed: {e}"))),
            };
        }
        let (mut streaming, mut gone_writing, mut first) = (false, false, None);
        let result = tts.speak(text, voice, cancel, &gone, &mut |pcm| {
            if gone_writing {
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
            gone_writing = sent.is_err();
            !gone_writing
        });
        if gone_writing || left.get() {
            eprintln!("[oaiy-voice] the listener went: the line stopped");
            return Ok(false);
        }
        match (streaming, result) {
            (false, Err(e)) => answer(w, Response::error(500, "server_error", format!("speech failed: {e}"))),
            (true, Err(e)) => {
                // Part of the line was sent: the stream is cut off there, without its end, so the client sees the
                // line did not end as it should (it has what came).
                eprintln!("[oaiy-voice] speech failed part way: {e}");
                Ok(false)
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

    /// The speech-to-text engine, waited for; a short request (`short`) counted as waiting meanwhile, so a long one
    /// lets it in between its passes ([`Server::make_way`]).
    fn engine<'a>(&'a self, stt: &'a SharedStt, short: bool) -> MutexGuard<'a, Box<dyn SpeechToText>> {
        let _waiting = short.then(|| {
            self.waiting.fetch_add(1, Ordering::SeqCst);
            Waiting(&self.waiting)
        });
        stt.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Between a long transcription's passes: the short ones waiting for the engine go first (for [`MAKE_WAY`] at most).
    fn make_way(&self) {
        let until = Instant::now() + MAKE_WAY;
        while self.waiting.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn transcriptions(&self, req: &Request) -> Response {
        let content_type = req.header("content-type").unwrap_or("").to_string();
        let lower = content_type.to_ascii_lowercase();
        let (wav, format) = if lower.starts_with("multipart/form-data") {
            match audio::parse_multipart(&req.body, &content_type) {
                Ok(mut form) => {
                    let format = form.field("response_format").map(str::to_string);
                    match form.file.take() {
                        Some(f) => (f, format),
                        None => return Response::bad("multipart request has no file part"),
                    }
                }
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
        let parsed = audio::parse_wav(&wav);
        drop(wav);
        let wav = match parsed {
            Ok(w) => w,
            Err(e) => return Response::bad(format!("invalid WAV: {e}")),
        };
        let Some(stt) = &self.stt else { return Response::error(503, "server_error", "no speech-to-text model is loaded") };
        // (resampled before the engine is waited on: another request's transcription is not held up by it)
        let rate = self.stt_rate;
        let samples = audio::resample(&wav.samples, wav.sample_rate as usize, rate);
        drop(wav);
        let seconds = samples.len() as f64 / rate as f64;
        let started = Instant::now();
        // One pass of the model at a time, the engine let go between them, so a long recording does not hold a
        // call's words for minutes: each pass is what the engine would have made of the whole (its own pieces).
        let passes = crate::stt::passes(&samples, rate);
        let mut said = Vec::with_capacity(passes.len());
        for (i, &(a, b)) in passes.iter().enumerate() {
            if i > 0 {
                self.make_way();
            }
            let mut engine = self.engine(stt, passes.len() == 1);
            match engine.transcribe(&samples[a..b]) {
                Ok(t) => said.push(t),
                Err(e) => return Response::error(500, "server_error", format!("transcription failed: {e}")),
            }
        }
        let text = said.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
        eprintln!("[oaiy-voice] transcribed {seconds:.2} s in {:.0} ms: {} chars", started.elapsed().as_secs_f64() * 1e3, text.len());
        match format {
            "text" => Response { status: 200, content_type: "text/plain; charset=utf-8", body: text.into_bytes() },
            "verbose_json" => Response::json(200, Json::obj([("task", Json::str("transcribe")), ("language", Json::str("english")), ("duration", Json::Num(seconds)), ("text", Json::str(text))])),
            _ => Response::json(200, Json::obj([("text", Json::str(text))])),
        }
    }
}

/// A whole answer, with no `Access-Control-*` headers (`oaiy-engine`'s answers allow every origin: this server
/// allows none but its own).
fn respond<W: Write>(w: &mut W, r: &Response, keep_alive: bool) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
        r.status,
        http::reason(r.status),
        r.content_type,
        r.body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    w.write_all(head.as_bytes())?;
    w.write_all(&r.body)?;
    w.flush()
}

/// A whole answer; the connection stays open.
fn answer<W: Write>(w: &mut W, r: Response) -> io::Result<bool> {
    respond(w, &r, true).map(|_| true)
}

/// Whether the client on `stream` has gone: its end closed (a read would find the end) or the connection broken. Asked
/// between the thread's own writes only: the socket is made non-blocking for the look, which its writes share.
fn caller_gone(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut byte = [0u8; 1];
    let gone = match stream.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => !matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted),
    };
    let _ = stream.set_nonblocking(false);
    gone
}

struct Permit(Arc<AtomicUsize>);

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Serve until the process ends.
pub fn run(mut server: Server, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    server.loopback = http::listens_on_loopback(host);
    let keyed = if server.key.is_some() { ", requests need its key" } else { "" };
    eprintln!("[oaiy-voice] listening on http://{host}:{port} ({} mode{keyed})", server.mode.as_str());
    let server = Arc::new(server);
    let connections = Arc::new(AtomicUsize::new(0));
    let mut pause = Duration::from_millis(5);
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            // Out of file descriptors (EMFILE) fails every accept at once until one closes: a pause, longer each
            // time, rather than a core spun at 100%.
            Err(e) if !matches!(e.kind(), io::ErrorKind::Interrupted | io::ErrorKind::ConnectionAborted | io::ErrorKind::ConnectionReset) => {
                std::thread::sleep(pause);
                pause = (pause * 2).min(Duration::from_secs(1));
                continue;
            }
            Err(_) => continue,
        };
        pause = Duration::from_millis(5);
        let open = connections.fetch_add(1, Ordering::SeqCst);
        let permit = Permit(Arc::clone(&connections));
        if open >= MAX_CONNECTIONS + PROBE_SLOTS {
            drop(permit);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            let _ = respond(&mut stream, &Response::error(503, "server_error", "voice server connection limit reached"), false);
            continue;
        }
        // (past the limit: a probe's connection, one request)
        let probe_only = open >= MAX_CONNECTIONS;
        let server = Arc::clone(&server);
        // (a thread that cannot start drops its connection with it, rather than taking the accept loop down)
        let started = std::thread::Builder::new().name("voice request".into()).spawn(move || {
            let _permit = permit;
            // (the connection looked at while a spoken line waits and is made: its own handle, used between writes)
            let watch = stream.try_clone().ok();
            let gone = || watch.as_ref().is_some_and(caller_gone);
            let turn: RefCell<Option<Turn>> = RefCell::new(None);
            http::serve_gated(
                stream,
                |req| match server.admit(req, probe_only) {
                    Ok(t) => {
                        *turn.borrow_mut() = t;
                        None
                    }
                    Err(refused) => Some(refused),
                },
                |req, w| {
                    let _turn = turn.borrow_mut().take();
                    let keep = server.serve_watched(req, w, &gone)?;
                    w.flush()?;
                    Ok(keep && !probe_only)
                },
            );
        });
        if let Err(e) = started {
            eprintln!("[oaiy-voice] a connection dropped, its thread would not start: {e}");
        }
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
        fn speak(&self, text: &str, _voice: Option<&str>, _cancel: Arc<AtomicBool>, gone: &dyn Fn() -> bool, on_chunk: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String> {
            if text == "fail" {
                return Err("the engine broke".into());
            }
            for c in text.chars() {
                if gone() || !on_chunk(&[c as i16]) {
                    break;
                }
                if c == '!' {
                    return Err("the engine broke part way".into());
                }
            }
            Ok(())
        }
    }

    /// A speech engine whose thread has ended.
    struct DeadTts;

    impl TextToSpeech for DeadTts {
        fn model(&self) -> String {
            "qwen3-tts-test".into()
        }
        fn voices(&self) -> (Vec<String>, Option<String>) {
            (Vec::new(), None)
        }
        fn check_voice(&self, _voice: Option<&str>) -> Result<(), String> {
            Ok(())
        }
        fn speak(&self, _: &str, _: Option<&str>, _: Arc<AtomicBool>, _: &dyn Fn() -> bool, _: &mut dyn FnMut(&[i16]) -> bool) -> Result<(), String> {
            Err("the speech engine has stopped".into())
        }
        fn alive(&self) -> bool {
            false
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
        // a line prepared where the engine keeps nothing: nothing to do, said so
        let prepared = served(&server(Mode::Both), &post("/v1/audio/speech", "application/json", br#"{"input":"hi","prepare":true}"#.to_vec()));
        assert!(prepared.starts_with("HTTP/1.1 200") && prepared.contains(r#""prepared":false"#), "{prepared}");
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

    fn head(method: &str, path: &str, headers: &[(&str, &str)]) -> Request {
        Request::new(method, path, headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), Vec::new())
    }

    fn refused(s: &Server, req: &Request, probe_only: bool) -> Option<u16> {
        s.admit(req, probe_only).err().map(|r| r.status)
    }

    #[test]
    fn no_other_site_and_no_rebound_name_is_answered_and_a_key_guards_all_but_the_probes() {
        let s = server(Mode::Both);
        // DNS rebinding (a site's name for 127.0.0.1), another site's page
        assert_eq!(refused(&s, &head("GET", "/v1/audio/voices", &[("Host", "evil.example:17920")]), false), Some(403));
        assert_eq!(refused(&s, &head("POST", "/v1/audio/speech", &[("Host", "127.0.0.1:17920"), ("Origin", "https://evil.example"), ("Content-Type", "text/plain")]), false), Some(403));
        assert_eq!(refused(&s, &head("POST", "/v1/audio/speech", &[("Host", "127.0.0.1:17920")]), false), None);
        assert_eq!(refused(&s, &head("GET", "/v1/models", &[("Host", "localhost:17920"), ("Origin", "http://localhost:5173")]), false), None);
        // with a key: every request but a probe's carries it
        let keyed = server(Mode::Both).with_key(Some(" k ".into()));
        assert_eq!(refused(&keyed, &head("GET", "/v1/audio/voices", &[("Host", "127.0.0.1")]), false), Some(401));
        assert_eq!(refused(&keyed, &head("GET", "/v1/audio/voices", &[("Host", "127.0.0.1"), ("Authorization", "Bearer k")]), false), None);
        for probe in ["/v1/models", "/health"] {
            assert_eq!(refused(&keyed, &head("GET", probe, &[("Host", "127.0.0.1")]), false), None, "{probe}");
        }
        assert!(server(Mode::Both).with_key(Some("  ".into())).key.is_none());
    }

    #[test]
    fn bodies_too_large_are_refused_on_their_head_and_a_probe_slot_answers_probes_alone() {
        let s = server(Mode::Both);
        let sized = |path: &str, len: usize| head("POST", path, &[("Host", "127.0.0.1"), ("Content-Length", &len.to_string())]);
        assert_eq!(refused(&s, &sized("/v1/audio/speech", MAX_SPEECH_BODY + 1), false), Some(413));
        assert_eq!(refused(&s, &sized("/v1/audio/speech", 2_000), false), None);
        assert_eq!(refused(&s, &sized("/v1/audio/transcriptions", MAX_TRANSCRIPTION_BODY + 1), false), Some(413));
        assert!(matches!(s.admit(&sized("/v1/audio/transcriptions", 1 << 20), false), Ok(Some(_))), "a transcription takes a turn");
        assert_eq!(refused(&s, &sized("/v1/other", MAX_OTHER_BODY + 1), false), Some(413));
        assert_eq!(refused(&s, &head("GET", "/v1/models", &[("Host", "127.0.0.1")]), true), None);
        assert_eq!(refused(&s, &head("GET", "/health", &[("Host", "127.0.0.1")]), true), None);
        assert_eq!(refused(&s, &sized("/v1/audio/speech", 20), true), Some(503));
    }

    #[test]
    fn turns_are_waited_for_and_given_back() {
        let turns = Turns::new(2);
        let (a, b) = (turns.take(Duration::ZERO), turns.take(Duration::ZERO));
        assert!(a.is_some() && b.is_some());
        assert!(turns.take(Duration::from_millis(20)).is_none(), "none free");
        let later = {
            let turns = Arc::clone(&turns);
            std::thread::spawn(move || turns.take(Duration::from_secs(5)).is_some())
        };
        std::thread::sleep(Duration::from_millis(50));
        drop(a);
        assert!(later.join().unwrap(), "a turn given back is taken by one waiting");
        drop(b);
    }

    #[test]
    fn answers_allow_no_other_origin() {
        let r = served(&server(Mode::Both), &Request::new("GET", "/v1/audio/voices", vec![], vec![]));
        assert!(r.starts_with("HTTP/1.1 200 OK\r\n") && !r.to_ascii_lowercase().contains("access-control"), "{r}");
    }

    #[test]
    fn a_speech_engine_gone_fails_the_probes() {
        let stt: SharedStt = Arc::new(Mutex::new(Box::new(Fake)));
        let s = Server::new(Mode::Both, Some(stt), "parakeet-test", "cpu f32", Some(Arc::new(DeadTts) as Arc<dyn TextToSpeech>));
        assert_eq!(s.handle(&Request::new("GET", "/v1/models", vec![], vec![])).status, 503);
        let h = s.handle(&Request::new("GET", "/health", vec![], vec![]));
        assert_eq!((h.status, json(&h).get("status").and_then(Json::as_str)), (503, Some("failed")));
        assert!(served(&s, &post("/v1/audio/speech", "application/json", br#"{"input":"hi"}"#.to_vec())).starts_with("HTTP/1.1 503"));
        // speech to text alone: no speech engine to miss
        let stt: SharedStt = Arc::new(Mutex::new(Box::new(Fake)));
        let only = Server::new(Mode::Stt, Some(stt), "parakeet-test", "cpu f32", None);
        assert_eq!(only.handle(&Request::new("GET", "/health", vec![], vec![])).status, 200);
    }

    #[test]
    fn a_stream_that_fails_part_way_is_cut_off_without_its_end() {
        let mut out = Vec::new();
        let keep = server(Mode::Tts).serve(&post("/v1/audio/speech", "application/json", br#"{"input":"ab!cd","response_format":"pcm"}"#.to_vec()), &mut out).unwrap();
        let out = String::from_utf8_lossy(&out);
        assert!(!keep, "the connection is closed");
        assert!(out.starts_with("HTTP/1.1 200 OK") && out.ends_with("2\r\n!\0\r\n") && !out.contains("0\r\n\r\n"), "{out:?}");
    }

    #[test]
    fn a_client_gone_before_its_line_is_made_gets_nothing_and_its_connection_closes() {
        let s = server(Mode::Tts);
        for format in ["wav", "pcm"] {
            let mut out = Vec::new();
            let body = format!(r#"{{"input":"hello","response_format":"{format}"}}"#);
            let keep = s.serve_watched(&post("/v1/audio/speech", "application/json", body.into_bytes()), &mut out, &|| true).unwrap();
            assert!(!keep && out.is_empty(), "{format}: {}", String::from_utf8_lossy(&out));
        }
    }

    #[test]
    fn a_client_that_closed_its_end_is_seen_gone_and_one_waiting_is_not() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server_end, _) = listener.accept().unwrap();
        assert!(!caller_gone(&server_end), "waiting for its answer");
        let mut writer = server_end.try_clone().unwrap();
        writer.write_all(b"still blocking").unwrap();
        drop(client);
        std::thread::sleep(Duration::from_millis(50));
        assert!(caller_gone(&server_end), "its end closed");
    }

    #[test]
    fn a_long_recording_is_heard_a_pass_at_a_time_and_odd_rates_cheaply() {
        // 200 s at 16 kHz: three passes of at most 90 s, all of it heard
        let samples = vec![0.0f32; 200 * 16_000];
        let passes = crate::stt::passes(&samples, 16_000);
        assert_eq!(passes.len(), 3);
        let wav = audio::write_wav(&samples, 16_000);
        let r = server(Mode::Stt).handle(&post("/v1/audio/transcriptions", "application/json", format!("{{\"audio\":\"{}\"}}", b64(&wav)).into_bytes()));
        let text = json(&r).get("text").and_then(Json::as_str).unwrap().to_string();
        let heard: usize = text.split(" samples").filter_map(|n| n.trim().parse::<usize>().ok()).sum();
        assert_eq!((r.status, heard, text.matches("samples").count()), (200, samples.len(), 3), "{text}");
        // 44,101 Hz shares no factor with 16 kHz: read as 44,100, at once (a second of it, 16,001 samples at 16 kHz)
        let odd = audio::write_wav(&vec![0.0f32; 44_101], 44_101);
        let started = Instant::now();
        let r = server(Mode::Stt).handle(&post("/v1/audio/transcriptions", "application/json", format!("{{\"audio\":\"{}\"}}", b64(&odd)).into_bytes()));
        assert_eq!(json(&r).get("text").and_then(Json::as_str), Some("16001 samples"));
        assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
    }
}
