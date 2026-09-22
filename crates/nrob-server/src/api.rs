//! The OpenAI-compatible endpoints: `POST /v1/chat/completions` (streamed as
//! server-sent events or whole), `POST /v1/completions` (raw text),
//! `GET /v1/models` and `GET /health`.
//!
//! Chat requests go through the DeepSeek-V4.1 chat format ([`dsv41::chat`]):
//! `tools` and `response_format` attach to the system message, reasoning
//! comes back as `reasoning_content` (as DeepSeek's own API does), and the
//! model's DSML tool calls come back as OpenAI `tool_calls`.
//!
//! A streamed chat reply starts at once: while the prompt goes in (minutes,
//! for a long one on cold caches) chunks with no choices carry
//! `nrob_progress: {prompt_done, prompt_total}` (tokens the prefix cache did
//! not hold), which OpenAI clients skip. A tool call comes whole at the end
//! of the reply; while one is being written, such chunks carry
//! `nrob_tool: {calls, name, parameter, chars, tail}` (the last lines of
//! the parameter being written), a few times a second.
//!
//! Images (`image_url` parts: base64 `data:` URLs, or local paths when the
//! server allows them) are decoded and sized here, on the request's thread,
//! with the reference's preprocessing; the worker runs them through the
//! vision tower.

use std::io;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dsv41::chat::{self, Delta, Mode};
use dsv41::config::VisionConfig;
use dsv41::vision;
use nrob::json::Json;

use crate::engine::{Event, Finish, Job, JobImage, Sampling};
use crate::http::{respond, Request, Stream};

pub struct Config {
    pub model_name: String,
    pub api_key: Option<String>,
    pub max_seq: usize,
    /// Mode when a request does not ask for one.
    pub thinking: bool,
    pub effort: u32,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// The vision tower's config when it is loaded (else images are refused).
    pub vision: Option<VisionConfig>,
    pub image_token_id: u32,
    /// Whether an image may name a file on this machine (a path or file://
    /// URL); off unless the server only listens on loopback.
    pub local_images: bool,
}

pub struct Server {
    // VENDORED-LOCAL: more than one configured model, switched on demand.
    /// The configured models and whichever is loaded. A request naming a different
    /// one unloads the current model first — see [`crate::models::Models::activate`].
    pub models: Arc<crate::models::Models>,
    /// Server-wide, not per model.
    pub api_key: Option<String>,
    pub local_images: bool,
    /// The configured context cap, for `/v1/models`. What a model actually allows
    /// is in its own `Config` once it is loaded.
    pub ctx: usize,
}

struct ApiError {
    status: u16,
    message: String,
    code: &'static str,
}

fn bad(message: impl Into<String>) -> ApiError {
    ApiError { status: 400, message: message.into(), code: "invalid_request" }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn random_id(prefix: &str) -> String {
    let mut s = SystemTime::now().duration_since(UNIX_EPOCH).map_or(7, |d| d.as_nanos() as u64) ^ (std::process::id() as u64) << 32;
    let alphabet = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut out = String::from(prefix);
    for _ in 0..24 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(alphabet[(s >> 33) as usize % alphabet.len()] as char);
    }
    out
}

/// How long a request waits on the model before showing it is still alive:
/// a comment line on an event stream, or a check that the client is still
/// connected. A long prompt takes minutes, and harnesses time out on silence.
const KEEPALIVE: Duration = Duration::from_secs(10);

/// How often a tool call being written is previewed, and how much of it.
const PREVIEW_EVERY: Duration = Duration::from_millis(300);
const PREVIEW_LINES: usize = 8;
const PREVIEW_CHARS: usize = 800;

/// The last `lines` lines of `text`, at most `max_chars` characters.
fn tail_of(text: &str, lines: usize, max_chars: usize) -> &str {
    let start = text.rmatch_indices('\n').nth(lines.saturating_sub(1)).map_or(0, |(i, _)| i + 1);
    let tail = &text[start..];
    let skip = tail.chars().count().saturating_sub(max_chars);
    tail.char_indices().nth(skip).map_or(tail, |(i, _)| &tail[i..])
}

/// Whether the client is still connected (has not closed its end).
fn client_alive(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return true;
    }
    let r = s.peek(&mut [0u8; 1]);
    let _ = s.set_nonblocking(false);
    match r {
        Ok(0) => false,
        Ok(_) => true,
        Err(e) => e.kind() == io::ErrorKind::WouldBlock,
    }
}

/// The next event from the model, keeping the connection alive meanwhile:
/// `None` when the model side is gone. A client that has left cancels the
/// job (and `gone` is set).
fn next_event(rx: &mpsc::Receiver<Event>, sse: &mut Option<Stream>, peer: Option<&TcpStream>, cancel: &AtomicBool, gone: &mut bool) -> Option<Event> {
    loop {
        match rx.recv_timeout(KEEPALIVE) {
            Ok(ev) => return Some(ev),
            Err(RecvTimeoutError::Disconnected) => return None,
            Err(RecvTimeoutError::Timeout) => {
                let alive = match sse.as_mut() {
                    Some(s) => s.send(b": keepalive\n\n").is_ok(),
                    None => peer.is_none_or(client_alive),
                };
                if !alive {
                    cancel.store(true, Ordering::Relaxed);
                    *gone = true;
                }
            }
        }
    }
}

fn json_response(w: &mut TcpStream, status: u16, v: &Json) -> io::Result<bool> {
    respond(w, status, "application/json", v.to_json().as_bytes(), true)?;
    Ok(true)
}

fn error_response(w: &mut TcpStream, e: &ApiError) -> io::Result<bool> {
    let v = Json::obj([(
        "error",
        Json::obj([("message", Json::str(&e.message)), ("type", Json::str("invalid_request_error")), ("code", Json::str(e.code))]),
    )]);
    json_response(w, e.status, &v)
}

impl Server {
    /// One request; returns whether the connection may stay open.
    pub fn handle(&self, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
        if req.method == "OPTIONS" {
            respond(w, 204, "text/plain", b"", true)?;
            return Ok(true);
        }
        let path = req.path.split('?').next().unwrap_or("");
        if path == "/health" || path == "/" {
            return json_response(w, 200, &Json::obj([("status", Json::str("ok"))]));
        }
        if let Some(key) = &self.api_key {
            let ok = req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")).is_some_and(|v| v.trim() == key);
            if !ok {
                return error_response(w, &ApiError { status: 401, message: "missing or wrong API key".into(), code: "invalid_api_key" });
            }
        }
        let result = match (req.method.as_str(), path) {
            ("GET", "/v1/models") => return json_response(w, 200, &self.models()),
            ("POST", "/v1/chat/completions") => self.chat(req, w),
            ("POST", "/v1/completions") => self.completions(req, w),
            (_, "/v1/models" | "/v1/chat/completions" | "/v1/completions") => {
                Err(ApiError { status: 405, message: format!("{} not allowed here", req.method), code: "method_not_allowed" })
            }
            _ => Err(ApiError { status: 404, message: format!("no route {path}"), code: "not_found" }),
        };
        match result {
            Ok(keep) => Ok(keep),
            Err(e) => error_response(w, &e),
        }
    }

    /// Every configured model, with the loaded one marked.
    fn models(&self) -> Json {
        let loaded = self.models.loaded();
        let data: Vec<Json> = self
            .models
            .names()
            .into_iter()
            .map(|name| {
                let is_loaded = loaded.as_deref() == Some(name.as_str());
                Json::obj([
                    ("id", Json::str(&name)),
                    ("object", Json::str("model")),
                    ("created", Json::Int(now() as i64)),
                    ("owned_by", Json::str("nrob")),
                    ("context_length", Json::Int(self.ctx as i64)),
                    // Not OpenAI's, but a client switching models wants to know
                    // which one is resident: the others cost a load.
                    ("loaded", Json::Bool(is_loaded)),
                ])
            })
            .collect();
        Json::obj([("object", Json::str("list")), ("data", Json::Arr(data))])
    }

    /// The model a request asks for, loading it if it is not the live one.
    ///
    /// A switch unloads the current model first, so this can take as long as a load
    /// — which is why `/v1/models` says which one is already resident.
    fn active(&self, body: &Json) -> Result<crate::models::Active, ApiError> {
        let want = body.get("model").and_then(Json::as_str).filter(|s| !s.is_empty());
        self.models.activate(want).map_err(|message| ApiError {
            status: 400,
            message,
            code: "model_not_found",
        })
    }

    fn parse_body(req: &Request) -> Result<Json, ApiError> {
        let body = Json::parse(&req.body).map_err(|e| bad(format!("request body: {e}")))?;
        if body.as_object().is_none() {
            return Err(bad("request body must be a JSON object"));
        }
        Ok(body)
    }

    fn sampling(&self, a: &crate::models::Active, body: &Json) -> Result<(Sampling, usize), ApiError> {
        let num = |k: &str| body.get(k).filter(|v| !matches!(v, Json::Null)).map(|v| v.as_f64().ok_or_else(|| bad(format!("{k} must be a number")))).transpose();
        let temperature = num("temperature")?.map_or(a.cfg.temperature, |v| v as f32);
        let top_p = num("top_p")?.map_or(a.cfg.top_p, |v| v as f32);
        if !((0.0..=2.0).contains(&temperature) && top_p > 0.0 && top_p <= 1.0) {
            return Err(bad("temperature must be in [0, 2] and top_p in (0, 1]"));
        }
        let top_k = num("top_k")?.map_or(0, |v| v.max(0.0) as usize);
        let seed = num("seed")?.map_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64), |v| v as u64);
        let max_tokens = num("max_completion_tokens")?.or(num("max_tokens")?).map_or(a.cfg.max_tokens, |v| v.max(1.0) as usize);
        if body.get("n").and_then(Json::as_i64).is_some_and(|n| n != 1) {
            return Err(bad("n > 1 is not supported"));
        }
        Ok((Sampling { temperature, top_p, top_k, seed }, max_tokens))
    }

    fn stops(body: &Json) -> Result<Vec<String>, ApiError> {
        match body.get("stop") {
            None | Some(Json::Null) => Ok(Vec::new()),
            Some(Json::Str(s)) => Ok(vec![s.clone()]),
            Some(Json::Arr(a)) => a.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad("stop must be strings"))).collect(),
            Some(_) => Err(bad("stop must be a string or a list of strings")),
        }
        .map(|v: Vec<String>| v.into_iter().filter(|s| !s.is_empty()).collect())
    }

    /// Thinking mode and effort: `reasoning_effort` ("none"/"minimal" turn
    /// it off; "low".."max" or 1-100 turn it on), or DeepSeek's
    /// `thinking: {"type": "enabled"|"disabled"}`, else the server default.
    fn mode(&self, a: &crate::models::Active, body: &Json) -> Result<(Mode, u32), ApiError> {
        let default = (if a.cfg.thinking { Mode::Thinking } else { Mode::Chat }, a.cfg.effort);
        if let Some(e) = body.get("reasoning_effort").filter(|v| !matches!(v, Json::Null)) {
            if matches!(e.as_str(), Some("none" | "minimal")) {
                return Ok((Mode::Chat, a.cfg.effort));
            }
            let effort = chat::parse_effort(e).ok_or_else(|| bad("reasoning_effort must be none, low, medium, high, max or 1-100"))?;
            return Ok((Mode::Thinking, effort));
        }
        match body.get("thinking").and_then(|t| t.get("type")).and_then(Json::as_str) {
            Some("enabled") => Ok((Mode::Thinking, default.1)),
            Some("disabled") => Ok((Mode::Chat, default.1)),
            _ => Ok(default),
        }
    }

    /// OpenAI messages to the chat format's: `developer` is a system
    /// message; tools and a response format go on the (first) system
    /// message, one being added when the conversation has none.
    fn messages(body: &Json) -> Result<Vec<Json>, ApiError> {
        let list = body.get("messages").and_then(Json::as_array).ok_or_else(|| bad("messages must be a list"))?;
        if list.is_empty() {
            return Err(bad("messages is empty"));
        }
        let mut msgs: Vec<Json> = Vec::with_capacity(list.len() + 1);
        for m in list {
            let Json::Obj(fields) = m else { return Err(bad("each message must be an object")) };
            let mut fields = fields.clone();
            for (k, v) in fields.iter_mut() {
                if k == "role" {
                    if let Json::Str(r) = v {
                        match r.as_str() {
                            "developer" => *r = "system".into(),
                            "function" => *r = "tool".into(),
                            "system" | "user" | "assistant" | "tool" => {}
                            other => return Err(bad(format!("unknown role {other:?}"))),
                        }
                    }
                }
            }
            msgs.push(Json::Obj(fields));
        }
        let tool_choice_none = body.get("tool_choice").and_then(Json::as_str) == Some("none");
        let tools = body.get("tools").and_then(Json::as_array).filter(|t| !t.is_empty() && !tool_choice_none);
        let format = body.get("response_format").filter(|f| f.get("type").and_then(Json::as_str).is_some_and(|t| t != "text"));
        if tools.is_some() || format.is_some() {
            if msgs[0].get("role").and_then(Json::as_str) != Some("system") {
                msgs.insert(0, Json::obj([("role", Json::str("system")), ("content", Json::str(""))]));
            }
            if let Json::Obj(first) = &mut msgs[0] {
                if let Some(t) = tools {
                    first.push(("tools".into(), Json::Arr(t.to_vec())));
                }
                if let Some(f) = format {
                    first.push(("response_format".into(), f.clone()));
                }
            }
        }
        Ok(msgs)
    }

    /// Decode and size a request's images and expand their placeholders in
    /// the prompt into image spans.
    fn prepare_images(&self, a: &crate::models::Active, records: &[Json], prompt: Vec<u32>) -> Result<(Vec<u32>, Vec<JobImage>), ApiError> {
        let Some(vc) = &a.cfg.vision else {
            return Err(bad("this server runs without the vision tower (started with --no-vision); images are not supported"));
        };
        let mut preps = Vec::with_capacity(records.len());
        let mut hashes = Vec::with_capacity(records.len());
        for (i, rec) in records.iter().enumerate() {
            let bytes = vision::image_bytes(rec, self.local_images).map_err(|e| bad(format!("image {}: {e}", i + 1)))?;
            let prep = vision::load_image(&bytes, vc).map_err(|e| bad(format!("image {}: {e}", i + 1)))?;
            let mut h = std::hash::DefaultHasher::new();
            std::hash::Hasher::write(&mut h, &bytes);
            hashes.push(std::hash::Hasher::finish(&h));
            preps.push(prep);
        }
        let (ids, starts) = vision::expand_placeholders(&prompt, a.cfg.image_token_id, &preps).map_err(|e| bad(e.to_string()))?;
        let images = preps.into_iter().zip(starts).zip(hashes).map(|((prep, start), hash)| JobImage { start, prep, hash }).collect();
        Ok((ids, images))
    }

    /// Reasoning tokens a reply may spend before it has to answer:
    /// `thinking.budget_tokens` when the request gives it (0: no limit),
    /// else by effort: low 2,048, medium and high 8,192, 76-99 16,384, max
    /// none. A model that goes round in circles then stops and acts.
    fn think_budget(body: &Json, effort: u32) -> Option<usize> {
        if let Some(n) = body.get("thinking").and_then(|t| t.get("budget_tokens")).and_then(Json::as_i64) {
            return (n > 0).then_some(n as usize);
        }
        match effort {
            100.. => None,
            76..=99 => Some(16_384),
            51..=75 => Some(8_192),
            _ => Some(2_048),
        }
    }

    fn submit(&self, a: &crate::models::Active, prompt: Vec<u32>, images: Vec<JobImage>, sampling: Sampling, max_tokens: usize, think_budget: Option<usize>, tool_precision: bool) -> Result<(mpsc::Receiver<Event>, Arc<AtomicBool>), ApiError> {
        if prompt.len() + 1 > a.cfg.max_seq {
            return Err(ApiError {
                status: 400,
                message: format!(
                    "the prompt is {} tokens; {}'s context here is {}",
                    prompt.len(),
                    a.cfg.model_name,
                    a.cfg.max_seq
                ),
                code: "context_length_exceeded",
            });
        }
        let max_tokens = max_tokens.min(a.cfg.max_seq - prompt.len());
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job { tool_precision, prompt, images, max_tokens, think_budget, sampling, cancel: Arc::clone(&cancel), events: tx };
        a.jobs
            .send(job)
            .map_err(|_| ApiError { status: 503, message: "the model worker has stopped".into(), code: "unavailable" })?;
        Ok((rx, cancel))
    }

    fn chat(&self, req: &Request, w: &mut TcpStream) -> Result<bool, ApiError> {
        let body = Self::parse_body(req)?;
        // Before anything else: which model, loading it if it is not the live one.
        let a = self.active(&body)?;
        let (sampling, max_tokens) = self.sampling(&a, &body)?;
        let stops = Self::stops(&body)?;
        let (mode, effort) = self.mode(&a, &body)?;
        let msgs = Self::messages(&body)?;
        // Each model's own template.
        let encoded = a
            .flavour
            .chat_prompt(&msgs, &chat::Options { mode, effort, drop_thinking: true })
            .map_err(bad)?;
        let prompt = a.flavour.encode(&encoded.prompt);
        let (prompt, images) = if encoded.images.is_empty() { (prompt, Vec::new()) } else { self.prepare_images(&a, &encoded.images, prompt)? };
        if images.is_empty() && prompt.contains(&a.cfg.image_token_id) {
            return Err(bad("the prompt holds an image placeholder but no image"));
        }
        let n_prompt = prompt.len();
        let stream = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
        let include_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Json::as_bool).unwrap_or(false);
        let think_budget = if mode == Mode::Thinking { Self::think_budget(&body, effort) } else { None };
        let (rx, cancel) = self.submit(&a, prompt, images, sampling, max_tokens, think_budget, crate::models::needs_tool_precision(&body))?;
        let peer = w.try_clone().ok();
        let id = random_id("chatcmpl-");
        let created = now();
        let model = a.cfg.model_name.clone();

        let chunk = |delta: Json, finish: Option<&str>| -> Json {
            Json::obj([
                ("id", Json::str(&id)),
                ("object", Json::str("chat.completion.chunk")),
                ("created", Json::Int(created as i64)),
                ("model", Json::str(&model)),
                (
                    "choices",
                    Json::Arr(vec![Json::obj([
                        ("index", Json::Int(0)),
                        ("delta", delta),
                        ("finish_reason", finish.map_or(Json::Null, Json::str)),
                    ])]),
                ),
            ])
        };

        let mut sse = if stream {
            let mut s = Stream::start(w, "text/event-stream").map_err(io_err)?;
            let first = chunk(Json::obj([("role", Json::str("assistant")), ("content", Json::str(""))]), None);
            s.send(format!("data: {}\n\n", first.to_json()).as_bytes()).map_err(io_err)?;
            Some(s)
        } else {
            None
        };
        let mut parser = chat::StreamParser::new(mode);
        let mut stop = StopFilter::new(stops);
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut send_err = false;
        let mut emit = |d: Delta, sse: &mut Option<Stream>, stop: &mut StopFilter, cancel: &AtomicBool| {
            let (key, text) = match d {
                Delta::Reasoning(s) => ("reasoning_content", s),
                Delta::Content(s) => {
                    let (out, hit) = stop.push(&s);
                    if hit {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    ("content", out)
                }
            };
            if text.is_empty() {
                return;
            }
            if key == "content" {
                content.push_str(&text);
            } else {
                reasoning.push_str(&text);
            }
            if let Some(s) = sse.as_mut() {
                let c = chunk(Json::obj([(key, Json::str(text))]), None);
                if s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err() {
                    cancel.store(true, Ordering::Relaxed);
                    send_err = true;
                }
            }
        };

        let (mut finish, mut completion_tokens, mut cached) = (Finish::Stop, 0usize, 0usize);
        let mut error = None;
        let mut gone = false;
        let mut last_preview = Instant::now() - PREVIEW_EVERY;
        while let Some(ev) = next_event(&rx, &mut sse, peer.as_ref(), &cancel, &mut gone) {
            match ev {
                Event::Progress { done, total } => {
                    if let Some(s) = sse.as_mut() {
                        let mut c = chunk(Json::obj::<&str>([]), None);
                        if let Json::Obj(fields) = &mut c {
                            for (k, v) in fields.iter_mut() {
                                if k == "choices" {
                                    *v = Json::Arr(Vec::new());
                                }
                            }
                            fields.push((
                                "nrob_progress".into(),
                                Json::obj([("prompt_done", Json::Int(done as i64)), ("prompt_total", Json::Int(total as i64))]),
                            ));
                        }
                        // a client that has gone shows up at the next text
                        let _ = s.send(format!("data: {}\n\n", c.to_json()).as_bytes());
                    }
                }
                Event::Prefilled { cached: c } => cached = c,
                Event::Thinking { used, budget, done } => {
                    if let Some(s) = sse.as_mut() {
                        let mut c = chunk(Json::obj::<&str>([]), None);
                        if let Json::Obj(fields) = &mut c {
                            for (k, v) in fields.iter_mut() {
                                if k == "choices" {
                                    *v = Json::Arr(Vec::new());
                                }
                            }
                            fields.push((
                                "nrob_thinking".into(),
                                Json::obj([
                                    ("used", Json::Int(used as i64)),
                                    ("budget", budget.map_or(Json::Null, |b| Json::Int(b as i64))),
                                    ("done", Json::Bool(done)),
                                ]),
                            ));
                        }
                        let _ = s.send(format!("data: {}\n\n", c.to_json()).as_bytes());
                    }
                }
                Event::Text(t) => {
                    if stop.hit {
                        continue;
                    }
                    for d in parser.push(&t) {
                        emit(d, &mut sse, &mut stop, &cancel);
                    }
                    if parser.tool_calls_ready() {
                        // Stop generation at the completed call envelope so the
                        // client can execute tools before the model continues.
                        cancel.store(true, Ordering::Relaxed);
                    }
                    // a tool call being written shows as it takes shape
                    if let (Some(s), Some(p)) = (sse.as_mut(), parser.call_preview()) {
                        if last_preview.elapsed() >= PREVIEW_EVERY {
                            last_preview = Instant::now();
                            let mut c = chunk(Json::obj::<&str>([]), None);
                            if let Json::Obj(fields) = &mut c {
                                for (k, v) in fields.iter_mut() {
                                    if k == "choices" {
                                        *v = Json::Arr(Vec::new());
                                    }
                                }
                                fields.push((
                                    "nrob_tool".into(),
                                    Json::obj([
                                        ("calls", Json::Int(p.calls as i64)),
                                        ("name", Json::Str(p.name)),
                                        ("parameter", p.parameter.map_or(Json::Null, Json::Str)),
                                        ("chars", Json::Int(p.value.chars().count() as i64)),
                                        ("tail", Json::str(tail_of(&p.value, PREVIEW_LINES, PREVIEW_CHARS))),
                                    ]),
                                ));
                            }
                            let _ = s.send(format!("data: {}\n\n", c.to_json()).as_bytes());
                        }
                    }
                }
                Event::Done { finish: f, completion_tokens: n } => {
                    finish = f;
                    completion_tokens = n;
                    break;
                }
                Event::Error(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        let malformed_tools = parser.tool_call_error();
        if error.is_none() && !stop.hit {
            if let Some(detail) = &malformed_tools {
                error = Some(format!("Model generated an invalid DSML tool call ({detail}); no tool was executed. Please start a fresh turn."));
            }
        }
        let (rest, calls) = parser.finish();
        if !stop.hit && malformed_tools.is_none() {
            for d in rest {
                emit(d, &mut sse, &mut stop, &cancel);
            }
        }
        // text held back for a stop string that never came
        let tail = stop.finish();
        if !tail.is_empty() {
            content.push_str(&tail);
            if let Some(s) = sse.as_mut() {
                let c = chunk(Json::obj([("content", Json::str(&tail))]), None);
                let _ = s.send(format!("data: {}\n\n", c.to_json()).as_bytes());
            }
        }
        let calls = if stop.hit { Vec::new() } else { calls };
        let tool_calls: Vec<Json> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let name = match &c.namespace {
                    Some(ns) => format!("{ns}::{}", c.name),
                    None => c.name.clone(),
                };
                Json::obj([
                    ("index", Json::Int(i as i64)),
                    ("id", Json::str(random_id("call_"))),
                    ("type", Json::str("function")),
                    ("function", Json::obj([("name", Json::Str(name)), ("arguments", Json::str(&c.arguments))])),
                ])
            })
            .collect();
        let reason = if !tool_calls.is_empty() {
            "tool_calls"
        } else if finish == Finish::Length && !stop.hit {
            "length"
        } else {
            "stop"
        };
        let usage = Json::obj([
            ("prompt_tokens", Json::Int(n_prompt as i64)),
            ("completion_tokens", Json::Int(completion_tokens as i64)),
            ("total_tokens", Json::Int((n_prompt + completion_tokens) as i64)),
            ("prompt_tokens_details", Json::obj([("cached_tokens", Json::Int(cached as i64))])),
        ]);

        match sse {
            Some(mut s) => {
                if send_err || gone {
                    return Ok(false);
                }
                let mut send = |v: Json| s.send(format!("data: {}\n\n", v.to_json()).as_bytes());
                if let Some(e) = &error {
                    let _ = send(Json::obj([("error", Json::obj([("message", Json::str(e)), ("type", Json::str("server_error")), ("code", Json::str(if e.contains(crate::repetition::CODE) { crate::repetition::CODE } else { "server_error" }))]))]));
                } else {
                    if !tool_calls.is_empty() {
                        send(chunk(Json::obj([("tool_calls", Json::Arr(tool_calls))]), None)).map_err(io_err)?;
                    }
                    send(chunk(Json::obj::<&str>([]), Some(reason))).map_err(io_err)?;
                    if include_usage {
                        let mut u = chunk(Json::obj::<&str>([]), None);
                        if let Json::Obj(fields) = &mut u {
                            for (k, v) in fields.iter_mut() {
                                if k == "choices" {
                                    *v = Json::Arr(Vec::new());
                                }
                            }
                            fields.push(("usage".into(), usage));
                        }
                        send(u).map_err(io_err)?;
                    }
                }
                s.send(b"data: [DONE]\n\n").map_err(io_err)?;
                s.finish().map_err(io_err)?;
                Ok(true)
            }
            None => {
                if gone {
                    return Ok(false);
                }
                if let Some(e) = error {
                    let repetition = e.contains(crate::repetition::CODE);
                    return Err(ApiError { status: if repetition { 422 } else { 500 }, message: e,
                        code: if repetition { crate::repetition::CODE } else { "server_error" } });
                }
                let mut message = vec![
                    ("role".to_string(), Json::str("assistant")),
                    ("content".to_string(), if content.is_empty() && !tool_calls.is_empty() { Json::Null } else { Json::Str(content) }),
                ];
                if !reasoning.is_empty() {
                    message.push(("reasoning_content".into(), Json::Str(reasoning)));
                }
                if !tool_calls.is_empty() {
                    let calls = tool_calls
                        .into_iter()
                        .map(|c| match c {
                            Json::Obj(f) => Json::Obj(f.into_iter().filter(|(k, _)| k != "index").collect()),
                            other => other,
                        })
                        .collect();
                    message.push(("tool_calls".into(), Json::Arr(calls)));
                }
                let v = Json::obj([
                    ("id", Json::str(&id)),
                    ("object", Json::str("chat.completion")),
                    ("created", Json::Int(created as i64)),
                    ("model", Json::str(&model)),
                    (
                        "choices",
                        Json::Arr(vec![Json::obj([
                            ("index", Json::Int(0)),
                            ("message", Json::Obj(message)),
                            ("finish_reason", Json::str(reason)),
                        ])]),
                    ),
                    ("usage", usage),
                ]);
                json_response(w, 200, &v).map_err(io_err)
            }
        }
    }

    /// Raw completion of a text prompt (no chat template).
    fn completions(&self, req: &Request, w: &mut TcpStream) -> Result<bool, ApiError> {
        let body = Self::parse_body(req)?;
        let a = self.active(&body)?;
        let (sampling, max_tokens) = self.sampling(&a, &body)?;
        let stops = Self::stops(&body)?;
        let text = body.get("prompt").and_then(Json::as_str).ok_or_else(|| bad("prompt must be a string"))?;
        let prompt = a.flavour.encode(text);
        if prompt.is_empty() {
            return Err(bad("prompt is empty"));
        }
        if prompt.contains(&a.cfg.image_token_id) {
            return Err(bad("raw completions take no images; send them to /v1/chat/completions"));
        }
        let n_prompt = prompt.len();
        let stream = body.get("stream").and_then(Json::as_bool).unwrap_or(false);
        let (rx, cancel) = self.submit(&a, prompt, Vec::new(), sampling, max_tokens, None, false)?;
        let peer = w.try_clone().ok();
        let id = random_id("cmpl-");
        let created = now();
        let chunk = |text: &str, finish: Option<&str>| -> Json {
            Json::obj([
                ("id", Json::str(&id)),
                ("object", Json::str("text_completion")),
                ("created", Json::Int(created as i64)),
                ("model", Json::str(&a.cfg.model_name)),
                (
                    "choices",
                    Json::Arr(vec![Json::obj([
                        ("index", Json::Int(0)),
                        ("text", Json::str(text)),
                        ("finish_reason", finish.map_or(Json::Null, Json::str)),
                    ])]),
                ),
            ])
        };
        let mut sse = if stream { Some(Stream::start(w, "text/event-stream").map_err(io_err)?) } else { None };
        let mut stop = StopFilter::new(stops);
        let (mut all, mut finish, mut n) = (String::new(), Finish::Stop, 0usize);
        let mut gone = false;
        while let Some(ev) = next_event(&rx, &mut sse, peer.as_ref(), &cancel, &mut gone) {
            match ev {
                Event::Text(t) if !stop.hit => {
                    let (out, hit) = stop.push(&t);
                    if hit {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    all.push_str(&out);
                    if let (Some(s), false) = (sse.as_mut(), out.is_empty()) {
                        if s.send(format!("data: {}\n\n", chunk(&out, None).to_json()).as_bytes()).is_err() {
                            cancel.store(true, Ordering::Relaxed);
                            return Ok(false);
                        }
                    }
                }
                Event::Done { finish: f, completion_tokens } => {
                    finish = f;
                    n = completion_tokens;
                    break;
                }
                Event::Error(e) => return Err(ApiError { status: 500, message: e, code: "server_error" }),
                _ => {}
            }
        }
        if gone {
            return Ok(false);
        }
        let tail = stop.finish();
        all.push_str(&tail);
        let reason = if finish == Finish::Length && !stop.hit { "length" } else { "stop" };
        match sse.as_mut() {
            Some(s) => {
                if !tail.is_empty() {
                    s.send(format!("data: {}\n\n", chunk(&tail, None).to_json()).as_bytes()).map_err(io_err)?;
                }
                s.send(format!("data: {}\n\n", chunk("", Some(reason)).to_json()).as_bytes()).map_err(io_err)?;
                s.send(b"data: [DONE]\n\n").map_err(io_err)?;
                sse.take().expect("some").finish().map_err(io_err)?;
                Ok(true)
            }
            None => {
                let mut v = chunk(&all, Some(reason));
                if let Json::Obj(fields) = &mut v {
                    for (k, val) in fields.iter_mut() {
                        if k == "object" {
                            *val = Json::str("text_completion");
                        }
                    }
                    fields.push((
                        "usage".into(),
                        Json::obj([
                            ("prompt_tokens", Json::Int(n_prompt as i64)),
                            ("completion_tokens", Json::Int(n as i64)),
                            ("total_tokens", Json::Int((n_prompt + n) as i64)),
                        ]),
                    ));
                }
                json_response(w, 200, &v).map_err(io_err)
            }
        }
    }
}

fn io_err(e: io::Error) -> ApiError {
    ApiError { status: 500, message: format!("i/o: {e}"), code: "io" }
}

/// Cuts the reply at the first stop string, holding back any tail that
/// could be the start of one.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    pub hit: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> StopFilter {
        StopFilter { stops, held: String::new(), hit: false }
    }

    /// Text that is safe to show now, and whether a stop string was found.
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.hit {
            return (String::new(), true);
        }
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.held.push_str(s);
        if let Some(at) = self.stops.iter().filter_map(|st| self.held.find(st.as_str())).min() {
            let out = self.held[..at].to_string();
            self.held.clear();
            self.hit = true;
            return (out, true);
        }
        let keep = self
            .stops
            .iter()
            .map(|st| {
                (1..st.len().min(self.held.len() + 1))
                    .rev()
                    .find(|&k| self.held.is_char_boundary(self.held.len() - k) && st.starts_with(&self.held[self.held.len() - k..]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let out: String = self.held.drain(..self.held.len() - keep).collect();
        (out, false)
    }

    /// End of the reply: whatever was held back (no stop string came).
    pub fn finish(&mut self) -> String {
        if self.hit {
            String::new()
        } else {
            std::mem::take(&mut self.held)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_filter_cuts_at_the_first_stop_across_pieces() {
        let mut f = StopFilter::new(vec!["STOP".into(), "\n\n#".into()]);
        let mut out = String::new();
        for piece in ["Hello ", "wor", "ld ST", "O", "P and more"] {
            let (s, _) = f.push(piece);
            out.push_str(&s);
        }
        assert!(f.hit);
        assert_eq!(out, "Hello world ");
        let mut f = StopFilter::new(vec!["STOP".into()]);
        assert_eq!(f.push("abc ST").0, "abc ");
        assert_eq!(f.finish(), "ST");
    }

    #[test]
    fn reasoning_budgets_follow_the_effort_unless_the_request_gives_one() {
        let none = Json::parse(br#"{}"#).unwrap();
        assert_eq!(Server::think_budget(&none, 50), Some(2_048));
        assert_eq!(Server::think_budget(&none, 75), Some(8_192));
        assert_eq!(Server::think_budget(&none, 100), None);
        let given = Json::parse(br#"{"thinking":{"type":"enabled","budget_tokens":500}}"#).unwrap();
        assert_eq!(Server::think_budget(&given, 50), Some(500));
        let unlimited = Json::parse(br#"{"thinking":{"budget_tokens":0}}"#).unwrap();
        assert_eq!(Server::think_budget(&unlimited, 50), None);
    }

    #[test]
    fn messages_get_tools_on_a_system_message() {
        let body = Json::parse(br#"{"messages":[{"role":"developer","content":"be brief"},{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#)
        .unwrap();
        let m = Server::messages(&body).ok().unwrap();
        assert_eq!(m[0].get("role").and_then(Json::as_str), Some("system"));
        assert_eq!(m[0].get("tools").map(Json::len), Some(1));
        let body = Json::parse(br#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#).unwrap();
        let m = Server::messages(&body).ok().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].get("content").and_then(Json::as_str), Some(""));
        let body = Json::parse(br#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"none","tools":[{"type":"function","function":{"name":"f"}}]}"#).unwrap();
        assert_eq!(Server::messages(&body).ok().unwrap().len(), 1);
    }
}
