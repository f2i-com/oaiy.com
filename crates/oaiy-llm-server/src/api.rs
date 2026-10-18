//! The OpenAI-compatible endpoints: `POST /v1/chat/completions` (streamed as
//! server-sent events or whole), `POST /v1/completions` (raw text),
//! `GET /v1/models` and `GET /health`.
//!
//! Chat requests go through the DeepSeek-V4.1 chat format ([`crate::dsv41::chat`]):
//! `tools` and `response_format` attach to the system message, reasoning
//! comes back as `reasoning_content` (as DeepSeek's own API does), and the
//! model's DSML tool calls come back as OpenAI `tool_calls`.
//!
//! A streamed chat reply starts at once: while the prompt goes in (minutes,
//! for a long one on cold caches) chunks with no choices carry
//! `oaiy_progress: {prompt_done, prompt_total}` (tokens the prefix cache did
//! not hold), which OpenAI clients skip. A tool call comes whole at the end
//! of the reply; while one is being written, such chunks carry
//! `oaiy_tool: {calls, name, parameter, chars, tail}` (the last lines of
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

use crate::dsv41::chat::{self, Delta, Mode};
use crate::dsv41::config::VisionConfig;
use crate::dsv41::vision;
use oaiy_engine::json::Json;

use crate::engines::job::{Event, Finish, Job, JobImage, Sampling};
use oaiy_engine::http::{respond, Refusal, Request, Stream};

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
    /// Repeat penalty on reply prose when a request gives none (1 = off).
    pub repeat_penalty: f32,
    /// The vision tower's config when it is loaded (else images are refused).
    pub vision: Option<VisionConfig>,
    pub qwen_vision: Option<crate::llm::MmProjConfig>,
    /// MiniCPM-V 4.7's pictures (a Qwen3.5 model with its projector): cut into tiles and laid out on canvases.
    pub minicpmv: Option<crate::engines::qwen::MiniCpmV>,
    /// Gemma 4's pictures, where its vision projector is loaded.
    pub gemma_vision: Option<GemmaVision>,
    /// GLM-5.3 Flash's pictures, where its vision projector is loaded.
    pub glm_vision: Option<GlmVision>,
    pub image_token_id: u32,
    /// Whether an image may name a file on this machine (a path or file://
    /// URL); off unless the server only listens on loopback.
    pub local_images: bool,
}

/// How Gemma 4 takes a picture: sized to at most `max_soft_tokens` (of `patch` pixels' patches, 3 x 3 a token), its
/// soft tokens between `open` (`<|image>`) and `close` (`<image|>`) in place of its marker (`<|image|>`, which the
/// soft tokens are too).
#[derive(Clone, Copy, Debug)]
pub struct GemmaVision {
    pub patch: usize,
    pub max_soft_tokens: usize,
    pub open: u32,
    pub close: u32,
}

/// GLM-5.3 Flash's pictures: its tower's geometry (patch and merge, for sizing a picture) and the tokens a
/// picture is at most.
#[derive(Clone, Debug)]
pub struct GlmVision {
    pub shape: crate::llm::arch::glm5next::vision::VisionShape,
    pub max_tokens: usize,
}

pub struct Server {
    pub images: Arc<crate::media::Images>,
    // more than one configured model, switched on demand.
    /// The configured models and whichever is loaded. A request naming a different
    /// one unloads the current model first — see [`crate::models::Models::activate`].
    pub models: Arc<crate::models::Models>,
    /// Server-wide, not per model.
    pub api_key: Option<String>,
    pub local_images: bool,
    /// Every request incognito (`--incognito`); otherwise per request.
    pub incognito: bool,
    /// The configured context cap (0: each model's own), for `/v1/models`. What a
    /// model actually allows is in its own `Config` once it is loaded.
    pub ctx: usize,
    /// Whether it listens on loopback only: a request then has to name loopback as its `Host` (see [`Server::gate`]).
    pub loopback: bool,
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
/// job (and `gone` is set). An event stream's client is found gone at a send;
/// a whole reply's is looked at every [`KEEPALIVE`] (`checked`: when it last
/// was) whether events come or not, as a reply whose tokens keep coming never
/// waits long enough for the timeout below.
fn next_event(rx: &mpsc::Receiver<Event>, sse: &mut Option<Stream>, peer: Option<&TcpStream>, cancel: &AtomicBool, gone: &mut bool, checked: &mut Instant) -> Option<Event> {
    loop {
        match rx.recv_timeout(KEEPALIVE) {
            Ok(ev) => {
                if sse.is_none() && !*gone && checked.elapsed() >= KEEPALIVE {
                    *checked = Instant::now();
                    if !peer.is_none_or(client_alive) {
                        cancel.store(true, Ordering::Relaxed);
                        *gone = true;
                    }
                }
                return Some(ev);
            }
            Err(RecvTimeoutError::Disconnected) => return None,
            Err(RecvTimeoutError::Timeout) => {
                *checked = Instant::now();
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

/// Whether an `Authorization` header carries `key`: its scheme in any case (as RFC 9110 has it), the key compared in a
/// time that does not tell how much of it a guess had right.
fn key_matches(header: Option<&str>, key: &str) -> bool {
    let Some((scheme, given)) = header.map(str::trim).and_then(|h| h.split_once(' ')) else { return false };
    scheme.eq_ignore_ascii_case("bearer") && same_secret(given.trim().as_bytes(), key.as_bytes())
}

fn same_secret(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    std::hint::black_box(diff) == 0
}

/// Whether a `Host` (or an origin's host), its port aside, is this computer's loopback: `localhost` (and names under
/// it, which browsers keep on loopback whatever DNS says) or a loopback address.
fn loopback_name(host: &str) -> bool {
    let host = host.trim().to_ascii_lowercase();
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None if host.parse::<std::net::IpAddr>().is_ok() => host.as_str(),
        None => host.rsplit_once(':').filter(|(_, port)| port.bytes().all(|b| b.is_ascii_digit())).map_or(host.as_str(), |(name, _)| name),
    };
    name == "localhost" || name.ends_with(".localhost") || name.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Whether an `Origin` is a page served from this computer's loopback.
fn loopback_origin(origin: &str) -> bool {
    let origin = origin.trim();
    let rest = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://"));
    rest.is_some_and(|host| !host.contains('/') && loopback_name(host))
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

/// A request refused on its head, before its body is read: a wrong or missing key; on a server with no key, a page of
/// another site (an `Origin` not of this computer's loopback: such a server answers whoever reaches it, and a page in a
/// browser can); and on one that listens on loopback only, a `Host` naming anything else (a site's name pointed at
/// 127.0.0.1, which a browser takes for that site's own server).
fn gate(key: Option<&str>, loopback: bool, req: &Request) -> Option<Refusal> {
    if loopback && req.header("host").is_some_and(|h| !loopback_name(h)) {
        return Some(Refusal::new(403, "this server answers to its loopback address only (localhost or 127.0.0.1)", "forbidden_host"));
    }
    let path = req.route();
    match key {
        None if req.header("origin").is_some_and(|o| !loopback_origin(o)) => Some(Refusal::new(
            403,
            "this server has no API key, so it answers no web page but this computer's own; start it with --api-key to serve others",
            "forbidden_origin",
        )),
        Some(key) if req.method != "OPTIONS" && path != "/health" && path != "/" && !key_matches(req.header("authorization"), key) => {
            Some(Refusal::new(401, "missing or wrong API key", "invalid_api_key"))
        }
        _ => None,
    }
}

impl Server {
    /// [`gate`]'s answer for this server.
    pub fn gate(&self, req: &Request) -> Option<Refusal> {
        gate(self.api_key.as_deref(), self.loopback, req)
    }

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
            if !key_matches(req.header("authorization"), key) {
                return error_response(w, &ApiError { status: 401, message: "missing or wrong API key".into(), code: "invalid_api_key" });
            }
        }
        let result = match (req.method.as_str(), path) {
            ("POST", "/v1/images/generations") => {
                match Self::parse_body(req).and_then(|body| {
                    crate::media::reference_paths(&body,self.local_images).map_err(bad)?;
                    self.images.submit(&body).map_err(bad)
                }) {
                    Ok(job)=>return json_response(w,202,&job),Err(e)=>Err(e),
                }
            }
            ("POST", "/v1/videos/generations") => {
                match Self::parse_body(req).and_then(|body| {
                    for key in ["image", "end_image"] {
                        crate::media::video_reference_path(&body, key, self.local_images).map_err(bad)?;
                    }
                    self.images.submit_video(&body).map_err(bad)
                }) {
                    Ok(job) => return json_response(w,202,&job), Err(e) => Err(e),
                }
            }
            ("GET", "/v1/images/status" | "/v1/videos/status") => return json_response(w,200,&self.images.status()),
            ("POST", "/v1/images/cancel" | "/v1/videos/cancel") => {self.images.cancel();return json_response(w,200,&self.images.status());}
            ("POST", "/v1/images/release" | "/v1/videos/release") => {
                match self.images.release(){Ok(())=>return json_response(w,200,&Json::obj([("released",Json::Bool(true))])),Err(e)=>Err(bad(e))}
            }
            ("GET", "/v1/models") => return json_response(w, 200, &self.models()),
            ("GET", "/v1/model-status") => return json_response(w, 200, &self.models.status()),
            ("POST", "/v1/chat/completions") => self.chat(req, w),
            ("POST", "/v1/completions") => self.completions(req, w),
            (_, "/v1/models" | "/v1/chat/completions" | "/v1/completions") => {
                Err(ApiError { status: 405, message: format!("{} not allowed here", req.method), code: "method_not_allowed" })
            }
            _ => Err(ApiError { status: 404, message: format!("no route {path}"), code: "not_found" }),
        };
        match result {
            Ok(keep) => Ok(keep),
            // the reply had begun, or the client had gone: nothing more can be said on this connection
            Err(e) if e.status == 0 => Ok(false),
            Err(e) => error_response(w, &e),
        }
    }

    /// Every configured model, with the loaded one marked.
    fn models(&self) -> Json {
        let loaded = self.models.loaded_context();
        let data: Vec<Json> = self
            .models
            .names()
            .into_iter()
            .map(|name| {
                let context = loaded.as_ref().filter(|(n, _)| *n == name).map(|&(_, c)| c);
                let is_loaded = context.is_some();
                Json::obj([
                    ("id", Json::str(&name)),
                    ("object", Json::str("model")),
                    ("created", Json::Int(now() as i64)),
                    ("owned_by", Json::str("oaiy")),
                    // The loaded model's own; the others' is known once they load.
                    ("context_length", context.or((self.ctx > 0).then_some(self.ctx)).map_or(Json::Null, |c| Json::Int(c as i64))),
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
        let reasoning_repeat_penalty = num("reasoning_repeat_penalty")?.unwrap_or(1.0);
        let reasoning_repeat_last_n = num("reasoning_repeat_last_n")?.unwrap_or(256.0);
        if !(1.0..=2.0).contains(&reasoning_repeat_penalty) || !(0.0..=4096.0).contains(&reasoning_repeat_last_n) || reasoning_repeat_last_n.fract()!=0.0 {
            return Err(bad("reasoning_repeat_penalty must be in [1,2]; reasoning_repeat_last_n must be an integer in [0,4096]"));
        }
        // The reply's prose: a repeat penalty (llama.cpp's repeat_penalty, or
        // repetition_penalty), and OpenAI's presence and frequency penalties.
        let repeat_penalty = num("repeat_penalty")?.or(num("repetition_penalty")?).unwrap_or(a.cfg.repeat_penalty as f64);
        let repeat_last_n = num("repeat_last_n")?.unwrap_or(256.0);
        let presence_penalty = num("presence_penalty")?.unwrap_or(0.0);
        let frequency_penalty = num("frequency_penalty")?.unwrap_or(0.0);
        if !(1.0..=2.0).contains(&repeat_penalty) || !(0.0..=4096.0).contains(&repeat_last_n) || repeat_last_n.fract() != 0.0 {
            return Err(bad("repeat_penalty must be in [1,2]; repeat_last_n must be an integer in [0,4096]"));
        }
        if !(-2.0..=2.0).contains(&presence_penalty) || !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(bad("presence_penalty and frequency_penalty must be in [-2,2]"));
        }
        Ok((Sampling {
            temperature, top_p, top_k, seed,
            reasoning_repeat_penalty: reasoning_repeat_penalty as f32, reasoning_repeat_last_n: reasoning_repeat_last_n as usize,
            repeat_penalty: repeat_penalty as f32, repeat_last_n: repeat_last_n as usize,
            presence_penalty: presence_penalty as f32, frequency_penalty: frequency_penalty as f32,
        }, max_tokens))
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
        if let Some(m) = &a.cfg.minicpmv {
            return crate::engines::qwen::prepare_minicpmv(records, prompt, a.cfg.image_token_id, m, |s| a.flavour.encode(s), self.local_images, a.cfg.max_seq).map_err(bad);
        }
        if let Some(vc) = &a.cfg.qwen_vision {
            return crate::engines::qwen::prepare_images(records, prompt, a.cfg.image_token_id, vc, self.local_images, a.cfg.max_seq).map_err(bad);
        }
        if let Some(gv) = &a.cfg.gemma_vision {
            return gemma_images(records, prompt, a.cfg.image_token_id, gv, self.local_images, a.cfg.max_seq).map_err(bad);
        }
        if let Some(gv) = &a.cfg.glm_vision {
            return glm_images(records, prompt, a.cfg.image_token_id, gv, self.local_images, a.cfg.max_seq).map_err(bad);
        }
        let Some(vc) = &a.cfg.vision else {
            // (worded for a client that looks for "image" or "vision" to go on in text, as OAIY's Agent does)
            return Err(bad(format!("{} has no vision here (no vision projector is set for it, or the server runs with --no-vision); images are not supported", a.cfg.model_name)));
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
        let images = preps.into_iter().zip(starts).zip(hashes).map(|((prep, start), hash)| JobImage { start, prep: crate::engines::job::ImagePrep::Deepseek(prep), hash }).collect();
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
}


/// Gemma 4's pictures: each sized as its image processor sizes it, its marker in `prompt` (`image_id`) made `open`,
/// a soft token a 3 x 3 of patches (`image_id` each), `close`.
pub(crate) fn gemma_images(records: &[Json], prompt: Vec<u32>, image_id: u32, gv: &GemmaVision, local: bool, max_seq: usize) -> Result<(Vec<u32>, Vec<JobImage>), String> {
    let count = prompt.iter().filter(|&&t| t == image_id).count();
    if count != records.len() {
        return Err("the pictures' markers in the prompt are not one a picture".into());
    }
    let (mut ids, mut images, mut records) = (Vec::with_capacity(prompt.len()), Vec::new(), records.iter());
    for id in prompt {
        if id != image_id {
            ids.push(id);
            continue;
        }
        let i = images.len() + 1;
        let bytes = vision::image_bytes(records.next().expect("a record a marker"), local).map_err(|e| format!("image {i}: {e}"))?;
        let pixels = crate::llm::vision::gemma4_image(&bytes, gv.patch, gv.max_soft_tokens).map_err(|e| format!("image {i}: {e}"))?;
        let side = 3 * gv.patch;
        let tokens = (pixels.dim(1) / side) * (pixels.dim(2) / side);
        ids.push(gv.open);
        images.push(JobImage { start: ids.len(), prep: crate::engines::job::ImagePrep::Gemma { pixels, tokens }, hash: crate::engines::disk::fnv(&bytes, 0) });
        ids.extend(std::iter::repeat_n(image_id, tokens));
        ids.push(gv.close);
    }
    if ids.len() >= max_seq {
        return Err(format!("the prompt with its pictures ({} tokens) is past the context ({max_seq})", ids.len()));
    }
    Ok((ids, images))
}

/// GLM-5.3 Flash's pictures: each sized as its image processor sizes it (within `gv.max_tokens`), its `<|image|>` in
/// `prompt` (`image_id`, between the `<|begin_of_image|>` and `<|end_of_image|>` its template writes) made a token
/// an image token.
pub(crate) fn glm_images(records: &[Json], prompt: Vec<u32>, image_id: u32, gv: &GlmVision, local: bool, max_seq: usize) -> Result<(Vec<u32>, Vec<JobImage>), String> {
    let count = prompt.iter().filter(|&&t| t == image_id).count();
    if count != records.len() {
        return Err("the pictures' markers in the prompt are not one a picture".into());
    }
    let (mut ids, mut images, mut records) = (Vec::with_capacity(prompt.len()), Vec::new(), records.iter());
    for id in prompt {
        if id != image_id {
            ids.push(id);
            continue;
        }
        let i = images.len() + 1;
        let bytes = vision::image_bytes(records.next().expect("a record a marker"), local).map_err(|e| format!("image {i}: {e}"))?;
        let (pixels, width, height) = crate::llm::arch::glm5next::vision::preprocess_bytes_within(&bytes, &gv.shape, gv.max_tokens).map_err(|e| format!("image {i}: {e}"))?;
        let tokens = gv.shape.n_out_tokens(width / gv.shape.patch, height / gv.shape.patch);
        images.push(JobImage { start: ids.len(), prep: crate::engines::job::ImagePrep::Glm { pixels, width, height, tokens }, hash: crate::engines::disk::fnv(&bytes, 0) });
        ids.extend(std::iter::repeat_n(image_id, tokens));
    }
    if ids.len() >= max_seq {
        return Err(format!("the prompt with its pictures ({} tokens) is past the context ({max_seq})", ids.len()));
    }
    Ok((ids, images))
}

/// Whether a request asked to leave nothing behind: `"incognito": true` in its
/// body, or an `X-OAIY-Incognito: 1` header.
pub fn incognito_request(req: &Request, body: Option<&Json>) -> bool {
    req.header("x-oaiy-incognito").is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        || body.and_then(|b| b.get("incognito")).and_then(Json::as_bool) == Some(true)
}

impl Server {
    fn forget(&self, req: &Request, body: &Json) -> bool {
        self.incognito || incognito_request(req, Some(body))
    }

    #[allow(clippy::too_many_arguments)]
    fn submit(&self, a: &crate::models::Active, prompt: Vec<u32>, images: Vec<JobImage>, sampling: Sampling, max_tokens: usize, think_budget: Option<usize>, tool_precision: bool, observer_context: String, tools: Vec<Json>, forget: bool, session: Option<String>) -> Result<(mpsc::Receiver<Event>, Arc<AtomicBool>), ApiError> {
        if prompt.len() + 1 > a.cfg.max_seq {
            return Err(ApiError {
                status: 400,
                // (worded as OpenAI's is, which clients look for: OAIY's Agent compacts the conversation and asks again)
                message: format!(
                    "the prompt is {} tokens, and {}'s maximum context length is {} tokens here",
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
        let job = Job { tools, tool_precision, observer_context, prompt, images, max_tokens, think_budget, sampling, cancel: Arc::clone(&cancel), events: tx, forget, session: session.filter(|_| forget), wipe: false };
        a.jobs
            .send(job)
            .map_err(|_| ApiError { status: 503, message: "the model worker has stopped".into(), code: "unavailable" })?;
        Ok((rx, cancel))
    }

    fn chat(&self, req: &Request, w: &mut TcpStream) -> Result<bool, ApiError> {
        let body = Self::parse_body(req)?;
        // The end of an incognito session: what the model holds of it is wiped
        // (a model that is not loaded holds nothing, and is not loaded for this).
        if let Some(session) = body.get("oaiy_forget_session").and_then(Json::as_str) {
            // (the model loaded now, never one loaded for this: a model unloaded since took the session's state with it)
            let mut wiped = true;
            if let Some(a) = self.models.live() {
                let (tx, rx) = mpsc::channel();
                let job = Job { tools: Vec::new(), tool_precision: false, observer_context: String::new(), prompt: Vec::new(), images: Vec::new(), max_tokens: 0, think_budget: None,
                    sampling: Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0, reasoning_repeat_penalty: 1.0, reasoning_repeat_last_n: 0, repeat_penalty: 1.0, repeat_last_n: 0, presence_penalty: 0.0, frequency_penalty: 0.0 },
                    cancel: Arc::new(AtomicBool::new(false)), events: tx, forget: true, session: Some(session.to_owned()), wipe: true };
                a.jobs.send(job).map_err(|_| ApiError { status: 503, message: "the model worker has stopped".into(), code: "unavailable" })?;
                // Still busy after two minutes: the wipe waits its turn, and the reply says it has not happened yet.
                wiped = !matches!(rx.recv_timeout(std::time::Duration::from_secs(120)), Err(RecvTimeoutError::Timeout));
            }
            let reply = if wiped {
                Json::obj([("object", Json::str("oaiy.session.forgotten")), ("session", Json::str(session))])
            } else {
                Json::obj([
                    ("object", Json::str("oaiy.session.forgetting")),
                    ("session", Json::str(session)),
                    ("message", Json::str("the model is busy: it wipes the session when it has finished what it is doing")),
                ])
            };
            return json_response(w, if wiped { 200 } else { 202 }, &reply).map_err(closed);
        }
        observer_review_policy(&body)?;
        if body.get("oaiy_file_write_yield").is_some_and(|v| !matches!(v, Json::Bool(_))) {
            return Err(bad("oaiy_file_write_yield must be a boolean"));
        }
        // Observer approval is for the full proposed batch; never cut it short.
        let yield_files = matches!(body.get("oaiy_file_write_yield"), Some(Json::Bool(true)))
            && body.get("oaiy_observer_review").and_then(Json::as_str).unwrap_or("off") == "off";
        // Before loading a model: validate the observer request policy.
        // Which model, loading it if it is not the live one.
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
        let request_clock = Instant::now();
        let mut prefilled_at = None;
        let forget = self.forget(req, &body);
        // An incognito session (X-OAIY-Session, or oaiy_session): its state is reused by its next request, in memory only.
        let session = req.header("x-oaiy-session").map(str::to_owned).or_else(|| body.get("oaiy_session").and_then(Json::as_str).map(str::to_owned))
            .map(|s| s.trim().to_owned()).filter(|s| !s.is_empty() && s.len() <= 128);
        let (rx, cancel) = self.submit(&a, prompt, images, sampling, max_tokens, think_budget, crate::models::needs_tool_precision(&body), observer_context(&body), body.get("tools").and_then(Json::as_array).unwrap_or(&[]).to_vec(), forget, session)?;
        let _stop = StopOnReturn(&cancel);
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
            let mut s = Stream::start(w, "text/event-stream").map_err(closed)?;
            let first = chunk(Json::obj([("role", Json::str("assistant")), ("content", Json::str(""))]), None);
            s.send(format!("data: {}\n\n", first.to_json()).as_bytes()).map_err(closed)?;
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
        let mut reuse = None;
        let mut error = None;
        let mut gone = false;
        let mut last_preview = Instant::now() - PREVIEW_EVERY;
        let mut live_tool_preview = false;
        let mut tool_shown = 0usize;
        let mut checked = Instant::now();
        while let Some(ev) = next_event(&rx, &mut sse, peer.as_ref(), &cancel, &mut gone, &mut checked) {
            match ev {
                Event::ReasoningPreview(text) => {
                    if let Some(s) = sse.as_mut() {
                        let c = Json::obj([("oaiy_reasoning_preview",Json::str(text))]);
                        if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() { cancel.store(true,Ordering::Relaxed); gone = true; }
                    }
                }
                Event::ObserverDelta {text,thinking,start} => {
                    if let Some(s)=sse.as_mut() {
                        let c=Json::obj([("oaiy_observer_delta",Json::obj([("text",Json::str(text)),("thinking",Json::Bool(thinking)),("start",Json::Bool(start))]))]);
                        if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() {cancel.store(true,Ordering::Relaxed); gone = true;}
                    }
                }
                Event::ContentPreview(text) => {
                    if let Some(s) = sse.as_mut() {
                        let c=Json::obj([("oaiy_content_preview",Json::str(text))]);
                        if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() {cancel.store(true,Ordering::Relaxed); gone = true;}
                    }
                }
                Event::ToolPreview {text,start} => {
                    live_tool_preview=true;
                    if let Some(s) = sse.as_mut() {
                        let c=Json::obj([("oaiy_tool_preview",Json::obj([("text",Json::str(text)),("start",Json::Bool(start))]))]);
                        if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() {cancel.store(true,Ordering::Relaxed); gone = true;}
                    }
                }
                Event::ObserverProgress {action,comment} => {
                    if let Some(s) = sse.as_mut() {
                        let c = Json::obj([("oaiy_observer_progress",Json::obj([("action",Json::str(action)),("comment",Json::str(comment))]))]);
                        if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() {cancel.store(true,Ordering::Relaxed); gone = true;}
                    }
                }
                Event::Observer(comment) => {
                    if let Some(s) = sse.as_mut() {
                        let c = Json::obj([("oaiy_observer", Json::str(comment))]);
                        if s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err() { cancel.store(true, Ordering::Relaxed); gone = true; }
                    }
                }
                Event::CacheReuse { cached: c, source, common } => {
                    cached = c;
                    reuse = Some((c, source, common));
                }
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
                                "oaiy_progress".into(),
                                Json::obj([
                                    ("prompt_done", Json::Int(done as i64)), ("prompt_total", Json::Int(total as i64)),
                                    ("cached_tokens", reuse.map_or(Json::Null, |(n,_,_)| Json::Int(n as i64))),
                                    ("cache_source", reuse.map_or(Json::Null, |(_,s,_)| Json::str(s))),
                                    ("previous_prefix_tokens", reuse.map_or(Json::Null, |(_,_,n)| Json::Int(n as i64))),
                                ]),
                            ));
                        }
                        // a client gone while the prompt goes in (minutes of it) stops it
                        if s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err() {
                            cancel.store(true, Ordering::Relaxed);
                            gone = true;
                        }
                    }
                }
                Event::Prefilled { cached: c } => { cached = c; prefilled_at = Some(Instant::now()); },
                Event::Thinking { used, budget, done } => {
                    if done { for d in parser.end_reasoning() { emit(d, &mut sse, &mut stop, &cancel); } }
                    if let Some(s) = sse.as_mut() {
                        let mut c = chunk(Json::obj::<&str>([]), None);
                        if let Json::Obj(fields) = &mut c {
                            for (k, v) in fields.iter_mut() {
                                if k == "choices" {
                                    *v = Json::Arr(Vec::new());
                                }
                            }
                            fields.push((
                                "oaiy_thinking".into(),
                                Json::obj([
                                    ("used", Json::Int(used as i64)),
                                    ("budget", budget.map_or(Json::Null, |b| Json::Int(b as i64))),
                                    ("done", Json::Bool(done)),
                                ]),
                            ));
                        }
                        if s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err() {
                            cancel.store(true, Ordering::Relaxed);
                            gone = true;
                        }
                    }
                }
                Event::Text(t) => {
                    if stop.hit {
                        continue;
                    }
                    for d in parser.push(&t) {
                        emit(d, &mut sse, &mut stop, &cancel);
                    }
                    if yield_files { parser.yield_completed_file(); }
                    if parser.tool_calls_ready() {
                        // Stop generation at the completed call envelope so the
                        // client can execute tools before the model continues.
                        cancel.store(true, Ordering::Relaxed);
                    }
                    // Also stream every raw tool character without an observer.
                    // Accepted observer replay must not duplicate the live draft.
                    if !live_tool_preview && parser.tool_draft().len()>tool_shown {
                        let draft=parser.tool_draft();
                        if let Some(s)=sse.as_mut() {
                            let c=Json::obj([("oaiy_tool_preview",Json::obj([("text",Json::str(&draft[tool_shown..])),("start",Json::Bool(tool_shown==0))]))]);
                            if s.send(format!("data: {}\n\n",c.to_json()).as_bytes()).is_err() {cancel.store(true,Ordering::Relaxed); gone = true;}
                        }
                        tool_shown=draft.len();
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
                                    "oaiy_tool".into(),
                                    Json::obj([
                                        ("calls", Json::Int(p.calls as i64)),
                                        ("name", Json::Str(p.name)),
                                        ("parameter", p.parameter.map_or(Json::Null, Json::Str)),
                                        ("chars", Json::Int(p.value.chars().count() as i64)),
                                        ("tail", Json::str(tail_of(&p.value, PREVIEW_LINES, PREVIEW_CHARS))),
                                    ]),
                                ));
                            }
                            if s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err() {
                                cancel.store(true, Ordering::Relaxed);
                                gone = true;
                            }
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
        if error.is_none() && !gone {
            if let Some(at) = prefilled_at {
                self.models.record_request(&model,n_prompt,cached,completion_tokens,at.duration_since(request_clock).as_secs_f64(),at.elapsed().as_secs_f64());
            }
        }
        let malformed_tools = parser.tool_call_error();
        if error.is_none() && !stop.hit {
            if let Some(detail) = &malformed_tools {
                error = Some(format!("{TOOL_CONTRACT_CODE}: invalid DSML tool call ({detail}); no tool from this batch was executed"));
            }
        }
        if error.is_none() && !stop.hit && parser.tool_calls_ready() {
            if let Err(detail) = validate_completed_tool_contract(&body, parser.tool_draft()) {
                error = Some(detail);
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
                gone |= s.send(format!("data: {}\n\n", c.to_json()).as_bytes()).is_err();
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
        // Why a request failed, for the log: never what the model wrote.
        if let (Some(e), false) = (&error, forget) {
            let why = e.split(". The model wrote:").next().unwrap_or(e);
            eprintln!("  chat failed: {}", why.chars().take(300).collect::<String>());
        }

        match sse {
            Some(mut s) => {
                if send_err || gone {
                    return Ok(false);
                }
                let mut send = |v: Json| s.send(format!("data: {}\n\n", v.to_json()).as_bytes());
                if let Some(e) = &error {
                    let _ = send(Json::obj([("error", Json::obj([("message", Json::str(e)), ("type", Json::str("server_error")), ("code", Json::str(if e.contains(TOOL_CONTRACT_CODE) { TOOL_CONTRACT_CODE } else if e.contains(crate::guards::repetition::CODE) { crate::guards::repetition::CODE } else if e.contains(crate::guards::observer::CODE) { crate::guards::observer::CODE } else { "server_error" }))]))]));
                } else {
                    if !tool_calls.is_empty() {
                        send(chunk(Json::obj([("tool_calls", Json::Arr(tool_calls))]), None)).map_err(closed)?;
                    }
                    send(chunk(Json::obj::<&str>([]), Some(reason))).map_err(closed)?;
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
                        send(u).map_err(closed)?;
                    }
                }
                s.send(b"data: [DONE]\n\n").map_err(closed)?;
                s.finish().map_err(closed)?;
                Ok(true)
            }
            None => {
                if gone {
                    return Ok(false);
                }
                if let Some(e) = error {
                    let repetition = e.contains(crate::guards::repetition::CODE);
                    let observer = e.contains(crate::guards::observer::CODE);
                    let tool_contract = e.contains(TOOL_CONTRACT_CODE);
                    return Err(ApiError { status: if repetition || observer || tool_contract { 422 } else { 500 }, message: e,
                        code: if tool_contract { TOOL_CONTRACT_CODE } else if repetition { crate::guards::repetition::CODE } else if observer { crate::guards::observer::CODE } else { "server_error" } });
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
                json_response(w, 200, &v).map_err(closed)
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
        let forget = self.forget(req, &body);
        let (rx, cancel) = self.submit(&a, prompt, Vec::new(), sampling, max_tokens, None, false, String::new(), Vec::new(), forget, None)?;
        let _stop = StopOnReturn(&cancel);
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
        let mut sse = if stream { Some(Stream::start(w, "text/event-stream").map_err(closed)?) } else { None };
        let mut stop = StopFilter::new(stops);
        let (mut all, mut finish, mut n) = (String::new(), Finish::Stop, 0usize);
        let mut gone = false;
        let mut checked = Instant::now();
        while let Some(ev) = next_event(&rx, &mut sse, peer.as_ref(), &cancel, &mut gone, &mut checked) {
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
                Event::Error(e) => {
                    let Some(mut s) = sse.take() else { return Err(ApiError { status: 500, message: e, code: "server_error" }) };
                    // The stream has begun: the error goes in it, as a chat's does, and the stream ends.
                    let error = Json::obj([("error", Json::obj([("message", Json::str(&e)), ("type", Json::str("server_error")), ("code", Json::str("server_error"))]))]);
                    let _ = s.send(format!("data: {}\n\n", error.to_json()).as_bytes());
                    let _ = s.send(b"data: [DONE]\n\n");
                    let _ = s.finish();
                    return Ok(false);
                }
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
                    s.send(format!("data: {}\n\n", chunk(&tail, None).to_json()).as_bytes()).map_err(closed)?;
                }
                s.send(format!("data: {}\n\n", chunk("", Some(reason)).to_json()).as_bytes()).map_err(closed)?;
                s.send(b"data: [DONE]\n\n").map_err(closed)?;
                sse.take().expect("some").finish().map_err(closed)?;
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
                json_response(w, 200, &v).map_err(closed)
            }
        }
    }
}

/// A write to the client failed, or a stream had begun: there is no answering on this connection any more (status 0,
/// which [`Server::handle`] closes it on rather than writing an error into it).
fn closed(_: io::Error) -> ApiError {
    ApiError { status: 0, message: String::new(), code: "closed" }
}

/// Stops the request's job when its handler returns, however it does: an engine goes on with a reply nobody reads
/// until it is told to (its events' receiver dropped does not), and a reply already ended is not changed by it.
struct StopOnReturn<'a>(&'a AtomicBool);

impl Drop for StopOnReturn<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
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
mod gate_tests {
    use super::*;

    fn req(method: &str, path: &str, headers: &[(&str, &str)]) -> Request {
        Request::new(method, path, headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), Vec::new())
    }

    #[test]
    fn a_key_is_read_in_any_case_of_its_scheme_and_only_whole() {
        assert!(key_matches(Some("Bearer sk-1"), "sk-1"));
        assert!(key_matches(Some("bearer  sk-1 "), "sk-1"));
        assert!(key_matches(Some("BEARER sk-1"), "sk-1"));
        for wrong in [None, Some("Bearer sk-12"), Some("Bearer sk-"), Some("Basic sk-1"), Some("sk-1"), Some("Bearer ")] {
            assert!(!key_matches(wrong, "sk-1"), "{wrong:?}");
        }
    }

    #[test]
    fn loopback_is_localhost_and_the_loopback_addresses_with_or_without_a_port() {
        for host in ["localhost", "LOCALHOST:8000", "127.0.0.1", "127.0.0.1:8000", "[::1]:8000", "::1", "oaiy.localhost:80", "127.9.9.9"] {
            assert!(loopback_name(host), "{host}");
        }
        for host in ["evil.example", "evil.example:8000", "192.168.1.4:8000", "localhost.evil.example", "[::2]:80", ""] {
            assert!(!loopback_name(host), "{host}");
        }
        assert!(loopback_origin("http://localhost:5173") && loopback_origin("https://127.0.0.1"));
        for origin in ["null", "https://evil.example", "http://localhost.evil.example", "file://", "http://localhost/x", "chrome-extension://abc"] {
            assert!(!loopback_origin(origin), "{origin}");
        }
    }

    #[test]
    fn a_server_without_a_key_answers_no_other_sites_page_and_one_on_loopback_no_other_host() {
        let page = req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8000"), ("Origin", "https://evil.example")]);
        assert_eq!(gate(None, true, &page).map(|r| r.status), Some(403));
        let preflight = req("OPTIONS", "/v1/chat/completions", &[("Host", "127.0.0.1:8000"), ("Origin", "https://evil.example")]);
        assert_eq!(gate(None, true, &preflight).map(|r| r.status), Some(403));
        let rebound = req("GET", "/v1/models", &[("Host", "evil.example:8000")]);
        assert_eq!(gate(None, true, &rebound).map(|r| r.status), Some(403));
        assert!(gate(None, false, &rebound).is_none(), "a server on the network answers any name");
        let harness = req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8000")]);
        assert!(gate(None, true, &harness).is_none());
        let own = req("POST", "/v1/chat/completions", &[("Host", "localhost:8000"), ("Origin", "http://localhost:3000")]);
        assert!(gate(None, true, &own).is_none());
        // with a key, the key decides, and a page of any site with it is served
        assert_eq!(gate(Some("k"), true, &page).map(|r| r.status), Some(401));
        let keyed = req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8000"), ("Origin", "https://evil.example"), ("Authorization", "Bearer k")]);
        assert!(gate(Some("k"), true, &keyed).is_none());
        assert!(gate(Some("k"), true, &preflight).is_none());
        assert!(gate(Some("k"), true, &req("GET", "/health", &[])).is_none());
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

const TOOL_CONTRACT_CODE: &str = "tool_contract_error";

fn observer_review_policy(body: &Json) -> Result<(), ApiError> {
    match body.get("oaiy_observer_review") {
        None => Ok(()),
        Some(Json::Str(policy)) if matches!(policy.as_str(), "auto" | "blocking" | "off") => Ok(()),
        _ => Err(bad("oaiy_observer_review must be auto, blocking, or off")),
    }
}

fn validate_completed_tool_contract(body: &Json, draft: &str) -> Result<(), String> {
    crate::guards::observer::validate_tools(&observer_context(body), draft)
        .map_err(|e| format!("{TOOL_CONTRACT_CODE}: {e}; no tool from this batch was executed"))
}

#[test]
fn direct_tool_calls_still_require_declared_names_and_valid_arguments() {
    let body = Json::parse(br#"{"oaiy_observer_review":"off","tools":[{"function":{"name":"list_files","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]}"#).unwrap();
    let draft = |name: &str, params: &str| format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"{name}\">{params}</｜DSML｜ invoke></｜DSML｜ calls>");
    let parameter = "<｜DSML｜ parameter name=\"path\" string=\"true\">.</｜DSML｜ parameter>";
    assert!(validate_completed_tool_contract(&body, &draft("list_files", parameter)).is_ok());
    for invalid in [draft("list_files", ""), draft("unlisted", parameter), draft("list_files", "<｜DSML｜ parameter name=\"path\" string=\"false\">123</｜DSML｜ parameter>")] {
        assert!(validate_completed_tool_contract(&body, &invalid).unwrap_err().starts_with(TOOL_CONTRACT_CODE));
    }
    for policy in ["auto", "blocking", "off"] {
        assert!(observer_review_policy(&Json::obj([("oaiy_observer_review", Json::str(policy))])).is_ok());
    }
    assert!(observer_review_policy(&Json::obj([("oaiy_observer_review", Json::str("skip-everything"))])).is_err());
    assert!(observer_review_policy(&Json::obj([("oaiy_observer_review", Json::Bool(false))])).is_err());
}

fn observer_context(body: &Json) -> String {
    let task = body.get("messages").and_then(Json::as_array).and_then(|m| m.iter().rev()
        .find(|m| m.get("role").and_then(Json::as_str) == Some("user")))
        .and_then(|m|m.get("content")).cloned().unwrap_or(Json::Null);
    let recent: Vec<Json> = body.get("messages").and_then(Json::as_array).into_iter().flatten()
        .filter(|m|m.get("role").and_then(Json::as_str)!=Some("system")).rev().take(6)
        .map(|m|Json::obj([("role",m.get("role").cloned().unwrap_or(Json::Null)),
            ("content_excerpt",Json::str(m.get("content").map(Json::to_json).unwrap_or_default().chars().take(800).collect::<String>()))])).collect();
    Json::obj([("latest_user_request", task), ("tools", body.get("tools").cloned().unwrap_or(Json::Null)),
        ("review_progress",body.get("oaiy_review_progress").cloned().unwrap_or(Json::Bool(false))),
        ("tool_context",body.get("oaiy_tool_context").cloned().unwrap_or(Json::Bool(false))),
        ("review_policy",body.get("oaiy_observer_review").cloned().unwrap_or(Json::str("off"))),
        ("recent_context",Json::Arr(recent))]).to_json()
}
