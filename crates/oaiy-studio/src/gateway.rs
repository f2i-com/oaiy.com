//! The public API: a route table from the configuration, each route a path, a
//! method, a target (what serves it) and a spec (the dialect it speaks).
//!
//! * `chat`, `completions`, `models`: proxied to oaiy-llm-server (OpenAI chat,
//!   streamed as it arrives -- event streams are never buffered).
//! * `images` + `openai`: OpenAI Images, synchronous: `{created, data:[{b64_json}|{url}]}`.
//! * `edits` + `openai`: OpenAI image edits, as `multipart/form-data` (what the
//!   SDKs send) or JSON with `images: [{image_url}]`; same reply as `images`.
//! * `videos` + `openai`: OpenAI Videos, asynchronous: `POST P` creates,
//!   `GET P` lists, `GET P/{id}` polls, `GET P/{id}/content` downloads,
//!   `DELETE P/{id}` forgets.
//! * `images`/`videos` + `OAIY`: oaiy-llm-server's job API (`202 {id, status_url}`,
//!   `GET P/status`, `POST P/cancel`), for clients such as coder-cli.
//! * `files`: generated media under the output root.

use crate::media::{self, Job, Kind};
use crate::util::{base64_encode, bool_or, error_json, int_or, now, str_or};
use crate::Studio;
use oaiy_engine::http::{fetch, respond, respond_with, Request, Stream};
use oaiy_engine::json::Json;
use std::io;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// How long a chat request waits for a media job holding the LLM's GPU.
const CHAT_WAIT: Duration = Duration::from_secs(15 * 60);
/// How long a request waits for the LLM to load.
const LOAD_WAIT: Duration = Duration::from_secs(10 * 60);
/// Longest silence from the LLM (oaiy-llm-server sends keepalives every 10 s while streaming).
const UPSTREAM_READ: Duration = Duration::from_secs(30 * 60);
/// A synchronous image request waits this long for its job.
const IMAGE_WAIT: Duration = Duration::from_secs(60 * 60);
/// A speech request (or designing a voice) waits this long.
const SPEECH_WAIT: Duration = Duration::from_secs(30 * 60);
/// A song asked for through the speech endpoint waits this long.
const MUSIC_WAIT: Duration = Duration::from_secs(90 * 60);

pub struct Reply {
    status: u16,
    body: Json,
}

fn ok(body: Json) -> Result<Reply, Reply> {
    Ok(Reply { status: 200, body })
}

fn fail(status: u16, message: impl AsRef<str>) -> Reply {
    let (kind, code) = match status {
        401 => ("authentication_error", "invalid_api_key"),
        404 => ("invalid_request_error", "not_found"),
        405 => ("invalid_request_error", "method_not_allowed"),
        503 => ("server_error", "unavailable"),
        s if s >= 500 => ("server_error", "internal_error"),
        _ => ("invalid_request_error", "invalid_request"),
    };
    Reply { status, body: error_json(message.as_ref(), kind, code) }
}

pub fn json_reply(w: &mut TcpStream, status: u16, v: &Json) -> io::Result<bool> {
    respond(w, status, "application/json", v.to_json().as_bytes(), true)?;
    Ok(true)
}

fn send(w: &mut TcpStream, r: Result<Reply, Reply>) -> io::Result<bool> {
    let r = r.unwrap_or_else(|e| e);
    json_reply(w, r.status, &r.body)
}

fn parse_body(req: &Request) -> Result<Json, Reply> {
    if req.header("content-type").is_some_and(|c| c.to_ascii_lowercase().starts_with("multipart/")) {
        return Err(fail(400, "multipart bodies are not supported; send JSON (images and references as data: URLs)"));
    }
    let v = Json::parse(&req.body).map_err(|e| fail(400, format!("request body: {e}")))?;
    if v.as_object().is_none() {
        return Err(fail(400, "request body must be a JSON object"));
    }
    Ok(v)
}

/// A matched route: its target, spec, and the rest of the path after it.
pub struct Matched {
    pub target: String,
    pub spec: String,
    pub rest: String,
}

/// Find the route for `method path`. Images, videos and files own the paths
/// below theirs (`/v1/videos/{id}/content`, `/files/...`, `.../status`).
pub fn route(routes: &[Json], method: &str, path: &str) -> Option<Matched> {
    let path = path.trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };
    for r in routes.iter().filter(|r| bool_or(r, "enabled", true)) {
        let base = str_or(r, "path", "").trim_end_matches('/');
        let target = str_or(r, "target", "");
        let prefix_owner = matches!(target, "images" | "videos" | "files" | "voices" | "music" | "sound" | "model3d");
        let rest = if path == base {
            ""
        } else if prefix_owner && path.starts_with(base) && path[base.len()..].starts_with('/') {
            &path[base.len()..]
        } else {
            continue;
        };
        // The configured method is the primary route's; sub-routes set their own.
        if rest.is_empty() && !matches!(target, "videos" | "files" | "voices" | "music" | "sound" | "model3d") && !str_or(r, "method", "POST").eq_ignore_ascii_case(method) {
            continue;
        }
        return Some(Matched { target: target.into(), spec: str_or(r, "spec", "openai").into(), rest: rest.into() });
    }
    None
}

/// Serve one gateway request. `trusted`: from the local UI port (no key needed,
/// local file references allowed).
pub fn handle(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, m: Matched, trusted: bool) -> io::Result<bool> {
    let method = req.method.as_str();
    match (m.target.as_str(), m.spec.as_str()) {
        ("chat", _) => proxy(studio, req, w, "/v1/chat/completions"),
        ("completions", _) => proxy(studio, req, w, "/v1/completions"),
        ("models", _) => models(studio, req, w),
        ("health", _) => json_reply(w, 200, &health(studio)),
        ("discovery", _) => json_reply(w, 200, &crate::discovery::document(studio, &public_base(studio, req), true)),
        ("files", _) => files(studio, req, w, m.rest.trim_start_matches('/')),
        ("images", "openai") if m.rest.is_empty() => send(w, parse_body(req).and_then(|b| {
            let scratch = scratch_for(studio, req, Some(&b));
            images_openai(studio, req, b, trusted, scratch)
        })),
        ("edits", _) if m.rest.is_empty() => send(w, edit_body(req).and_then(|(mut b, uploads)| {
            // Everything is parsed and checked before anything is written, so
            // uploads land in the right folder (an `incognito` form field
            // counts) and a rejected request leaves no files behind.
            check_image_body(&b)?;
            let scratch = scratch_for(studio, req, Some(&b));
            if !uploads.is_empty() {
                let saved = save_uploads(studio, scratch.as_deref(), uploads).inspect_err(|_| {
                    if let Some(s) = &scratch {
                        let _ = std::fs::remove_dir_all(s);
                    }
                })?;
                crate::util::set(&mut b, "images", Json::Arr(saved));
            }
            images_openai(studio, req, b, trusted, scratch)
        })),
        ("images", "oaiy") | ("videos", "oaiy") => {
            let kind = if m.target == "images" { Kind::Image } else { Kind::Video };
            send(w, oaiy_jobs(studio, req, kind, &m.rest, trusted))
        }
        ("videos", "openai") => videos_openai(studio, req, w, method, &m.rest, trusted),
        ("speech", _) if m.rest.is_empty() => speech_openai(studio, req, w),
        ("voices", _) => voices(studio, req, w, &m.rest),
        ("music", _) => music_openai(studio, req, w, method, &m.rest),
        ("sound", _) => sound_openai(studio, req, w, method, &m.rest),
        ("model3d", _) => model3d_openai(studio, req, w, method, &m.rest, trusted),
        ("background", _) if m.rest.is_empty() => send(w, parse_body(req).and_then(|b| picture_openai(studio, req, b, trusted, crate::picture::Op::RemoveBackground))),
        ("upscale", _) if m.rest.is_empty() => send(w, parse_body(req).and_then(|b| picture_openai(studio, req, b, trusted, crate::picture::Op::Upscale))),
        _ => send(w, Err(fail(404, format!("no route {} {}", method, req.route())))),
    }
}

pub fn health(studio: &Studio) -> Json {
    Json::obj([
        ("status", Json::str("ok")),
        ("llm", studio.llm.status().get("state").cloned().unwrap_or(Json::Null)),
        ("media_busy", Json::Bool(studio.media.busy())),
    ])
}

/// Pass a reply on as it arrives: an event stream (or a body of no stated length) chunk by chunk, anything else whole.
fn relay(w: &mut TcpStream, response: oaiy_engine::http::Response) -> io::Result<bool> {
    relay_to_its_end(w, response, None)
}

/// [`relay`], for a server whose streams say no length and end by closing the connection (tinygrad's): `mark` is what
/// a stream that ran to its end closes with, and `gone` says whether the server that sent it is there no more. A
/// stream with no mark from a server that has gone (stopped, crashed, its card unplugged) was cut short, and is not
/// finished for the client: its connection is closed on a broken stream, not on a reply that looks whole. One with
/// no mark from a server still serving is finished as it came (a tinygrad whose streams carry none).
fn relay_to_its_end(w: &mut TcpStream, response: oaiy_engine::http::Response, ended: Option<(&[u8], &dyn Fn() -> bool)>) -> io::Result<bool> {
    let mark = ended.map(|(mark, _)| mark);
    let status = response.status;
    let rtype = response.header("content-type").unwrap_or("application/json").to_string();
    let streamed = rtype.contains("event-stream") || response.header("transfer-encoding").is_some_and(|t| t.contains("chunked"));
    if streamed {
        let mut out = Stream::start_status(w, status, &rtype)?;
        // A client that leaves makes `send` fail, which drops the upstream
        // connection, which oaiy-llm-server takes as a cancel.
        // (the last bytes passed on, enough to hold the mark and the blank lines after it)
        let mut tail: Vec<u8> = Vec::new();
        let r = response.for_each_chunk(|chunk| {
            if let Some(mark) = mark {
                tail.extend_from_slice(chunk);
                let keep = mark.len() + 16;
                if tail.len() > keep {
                    tail.drain(..tail.len() - keep);
                }
            }
            out.send(chunk)
        });
        if r.is_ok() {
            if let Some((mark, gone)) = ended.filter(|(m, _)| !tail.windows(m.len()).any(|part| part == *m)) {
                // (a process that is being ended closes its connections a moment before it is seen to have gone)
                let _ = mark;
                if (0..6).any(|turn| {
                    if turn > 0 {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    gone()
                }) {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the reply's stream ended before its end, and its server has gone"));
                }
            }
            out.finish()?;
        }
        // The stream said keep-alive and ended cleanly: the client may send its
        // next request on this connection, so keep reading it rather than close
        // it under a request already on its way.
        r.map(|_| true)
    } else {
        let body = response.body(64 << 20)?;
        respond(w, status, &rtype, &body, true).map(|_| true)
    }
}

/// A chat request for a model set to run on the eGPU (macOS, `crate::egpu`): answered by tinygrad's server there
/// (`Some`), or left to this computer's engine (`None`) because the request has a picture, which tinygrad's server
/// does not read, or because that server cannot be started or has gone: the card unplugged, tinygrad not installed.
/// A server that is only still loading is waited for, and past the wait the client is told to try again.
fn egpu_chat(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, cfg: &Json, name: &str) -> Option<io::Result<bool>> {
    use crate::egpu::{self, Unready};
    let llm = cfg.get("llm")?;
    let asked = Json::parse(&req.body).ok()?;
    if egpu::has_picture(&asked) {
        return None;
    }
    let body = egpu::request_body(llm, name, &req.body)?;
    let here = |why: &str| {
        if studio.egpu.news(why) {
            let answers = egpu::fallback(llm, name);
            studio.log.push(format!("the eGPU does not answer ({}): {answers} answers on this computer", why.lines().next().unwrap_or("")));
        }
    };
    let (endpoint, lease) = match studio.egpu.ensure(cfg, &studio.root, name, LOAD_WAIT) {
        Ok(ready) => ready,
        Err(Unready::Gone(why)) => {
            here(&why);
            return None;
        }
        Err(Unready::Loading(why)) => return Some(send(w, Err(fail(503, why)))),
    };
    let auth = format!("Bearer {}", endpoint.key);
    let thinking = if egpu::thinks(llm, &asked) { "1" } else { "0" };
    let headers = [("Authorization", auth.as_str()), ("Content-Type", "application/json"), ("X-OAIY-Thinking", thinking)];
    let response = match fetch(&endpoint.addr, "POST", "/v1/chat/completions", &headers, &body, UPSTREAM_READ) {
        Ok(r) => r,
        Err(e) => {
            // Nothing has been sent to the client yet. A server whose process has gone is the case this computer's
            // engine is there for. One that is still there keeps its model (it gives no reply to a request it
            // cannot render, and one such request must not cost a model that took minutes to load), and what
            // becomes of this request depends on what would answer it here: a model named to stand in does; the
            // same model does not, for that would load it a second time beside the card's copy, for one request.
            drop(lease);
            let late = matches!(e.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock);
            return match studio.egpu.unanswered(&e.to_string()) {
                Some(gone) => {
                    here(&gone);
                    None
                }
                None if late => Some(send(w, Err(fail(504, format!("tinygrad's server did not answer within {} minutes; it is still running", UPSTREAM_READ.as_secs() / 60))))),
                None if egpu::fallback(llm, name) != name => {
                    here(&format!("tinygrad's server gave this request no reply ({e}) and is still running"));
                    None
                }
                None => Some(send(w, Err(fail(502, format!("tinygrad's server gave this request no reply ({e}): it closes the connection on a request it cannot put into the model's chat format. It is still running, and no other model is named to answer in its place"))))),
            };
        }
    };
    let gone = || !studio.egpu.serves(&endpoint);
    let result = relay_to_its_end(w, response, Some((b"[DONE]", &gone)));
    drop(lease);
    Some(result)
}

/// A request's body with another `model` (the one that answers on this computer for a model the eGPU does not).
fn with_model(body: &[u8], model: &str) -> Option<Vec<u8>> {
    let mut v = Json::parse(body).ok().filter(|v| v.as_object().is_some())?;
    crate::util::set(&mut v, "model", Json::str(model));
    Some(v.to_json().into_bytes())
}

/// Forward to oaiy-llm-server, starting it if needed, and stream the reply back.
fn proxy(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, upstream: &str) -> io::Result<bool> {
    let cfg = studio.config();
    if !cfg.get("llm").is_some_and(|l| bool_or(l, "enabled", true)) {
        return send(w, Err(fail(503, "the language model is disabled")));
    }
    // The end of an incognito session: a model that is not running holds
    // nothing of it, so it is not started just to be told.
    if studio.llm.endpoint().is_none() {
        if let Some(session) = Json::parse(&req.body).ok().and_then(|b| b.get("oaiy_forget_session").and_then(Json::as_str).map(str::to_owned)) {
            return json_reply(w, 200, &Json::obj([("object", Json::str("oaiy.session.forgotten")), ("session", Json::str(session))]));
        }
    }
    // A model on the media GPU waits for a media job there (and a server that is
    // not running loads its default model first); one on other GPUs goes ahead.
    let requested = Json::parse(&req.body).ok().and_then(|b| b.get("model").and_then(Json::as_str).map(str::to_owned));
    let mut model = media::resolve_model(&cfg, requested.as_deref());
    // A model set to run on the eGPU (a Mac's, through tinygrad's server): its chats are answered there while that
    // server answers. Otherwise, and for what that server does not do, this computer's engine answers, with the
    // model named to stand in for it where one is.
    let mut body = std::borrow::Cow::Borrowed(&req.body[..]);
    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    if let Some(name) = model.clone().filter(|m| crate::egpu::assigned(&llm, m)) {
        if upstream == "/v1/chat/completions" {
            if let Some(answered) = egpu_chat(studio, req, w, &cfg, &name) {
                return answered;
            }
        }
        let stands_in = crate::egpu::fallback(&llm, &name);
        if stands_in != name {
            if let Some(rewritten) = with_model(&req.body, &stands_in) {
                body = std::borrow::Cow::Owned(rewritten);
                model = Some(stands_in);
            }
        }
    }
    let exclusive = media::needs_media_gpu(&cfg, model.as_deref()) || (!studio.llm.is_running() && media::needs_media_gpu(&cfg, None));
    let lease = match studio.media.chat_lease(exclusive, CHAT_WAIT) {
        Ok(l) => l,
        Err(e) => return send(w, Err(fail(503, e))),
    };
    let endpoint = match studio.llm.ensure_ready(&cfg, &studio.root, model.as_deref(), LOAD_WAIT) {
        Ok(e) => e,
        Err(e) => return send(w, Err(fail(503, e))),
    };
    if let Some(m) = model {
        studio.llm.set_resident(m);
    }
    studio.llm.touch();
    let auth = format!("Bearer {}", endpoint.key);
    let ctype = req.header("content-type").unwrap_or("application/json").to_string();
    let mut headers = vec![("Authorization", auth.as_str()), ("Content-Type", ctype.as_str())];
    if let Some(a) = req.header("accept") {
        headers.push(("Accept", a));
    }
    if incognito_mode(studio) || incognito_header(req) {
        headers.push(("X-OAIY-Incognito", "1"));
    }
    // An incognito session's requests reuse its state (in memory only).
    if let Some(session) = req.header("x-oaiy-session") {
        headers.push(("X-OAIY-Session", session));
    }
    let response = match fetch(&endpoint.addr, &req.method, upstream, &headers, &body, UPSTREAM_READ) {
        Ok(r) => r,
        Err(e) => return send(w, Err(fail(502, format!("the language model did not answer: {e}")))),
    };
    let result = relay(w, response);
    studio.llm.touch();
    drop(lease);
    result
}

/// OpenAI model list: oaiy-llm-server's own when it runs, the configured names otherwise,
/// plus the image and video models (typed, so clients can tell them apart).
fn models(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let cfg = studio.config();
    let mut data: Vec<Json> = Vec::new();
    if let Some(ep) = studio.llm.endpoint() {
        let auth = format!("Bearer {}", ep.key);
        if let Ok(r) = fetch(&ep.addr, "GET", "/v1/models", &[("Authorization", &auth)], b"", Duration::from_secs(10)) {
            if let Ok(v) = r.body(8 << 20).map(|b| Json::parse(&b)) {
                if let Some(list) = v.ok().and_then(|v| v.get("data").and_then(Json::as_array).map(<[Json]>::to_vec)) {
                    data.extend(list);
                }
            }
        }
    }
    if data.is_empty() {
        for m in cfg.get("llm").and_then(|l| l.get("models")).and_then(Json::as_array).unwrap_or(&[]) {
            if bool_or(m, "enabled", true) {
                data.push(model_entry(str_or(m, "name", ""), "llm", false));
            }
        }
    }
    for kind in ["image", "video", "speech", "music", "sound", "model3d"] {
        let section = cfg.get("media").and_then(|m| m.get(kind));
        if section.is_some_and(|s| bool_or(s, "enabled", true)) {
            for (name, m) in section.and_then(|s| s.get("models")).map(|m| m.members().collect::<Vec<_>>()).unwrap_or_default() {
                if bool_or(m, "enabled", true) {
                    data.push(model_entry(name, kind, false));
                }
            }
        }
    }
    let _ = req;
    json_reply(w, 200, &Json::obj([("object", Json::str("list")), ("data", Json::Arr(data))]))
}

fn model_entry(name: &str, kind: &str, loaded: bool) -> Json {
    Json::obj([
        ("id", Json::str(name)),
        ("object", Json::str("model")),
        ("created", Json::Int(now() as i64)),
        ("owned_by", Json::str("oaiy-studio")),
        ("type", Json::str(kind)),
        ("loaded", Json::Bool(loaded)),
    ])
}

fn content_type(path: &std::path::Path) -> &'static str {
    match path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).as_deref() {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("mp4") => "video/mp4",
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("ogg" | "opus") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("json" | "jsonl") => "application/json",
        Some("glb") => "model/gltf-binary",
        _ => "application/octet-stream",
    }
}

#[derive(Debug, PartialEq)]
enum Range {
    /// No usable range (absent, several, malformed, another unit): send it all.
    Whole,
    /// `(start, end_inclusive)`.
    Part(u64, u64),
    /// A well-formed range that misses the file: 416.
    Unsatisfiable,
}

/// `bytes=a-b`, `bytes=a-` or `bytes=-n` against a file of `len` bytes. What
/// cannot be served as one range is ignored, as RFC 9110 allows.
fn byte_range(header: &str, len: u64) -> Range {
    let parsed = (|| {
        let spec = header.trim().strip_prefix("bytes=")?;
        if spec.contains(',') {
            return None;
        }
        let (a, b) = spec.split_once('-')?;
        let num = |s: &str| s.parse::<u64>().ok();
        Some(match (a.trim(), b.trim()) {
            ("", "") => return None,
            ("", n) => (None, num(n)?),
            (a, "") => (Some(num(a)?), u64::MAX),
            (a, b) => (Some(num(a)?), num(b)?),
        })
    })();
    let Some((start, end)) = parsed else { return Range::Whole };
    match start {
        None if end == 0 || len == 0 => Range::Unsatisfiable,
        None => Range::Part(len.saturating_sub(end), len - 1),
        Some(s) if end < s => Range::Whole,
        Some(s) if s >= len => Range::Unsatisfiable,
        Some(s) => Range::Part(s, end.min(len - 1)),
    }
}

/// A file, copied in pieces rather than read whole, with `Range` support so
/// players can seek in videos.
pub fn serve_file(w: &mut TcpStream, path: &std::path::Path, download: bool, range: Option<&str>) -> io::Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let name = path.file_name().map(|n| n.to_string_lossy().replace(['"', '\r', '\n'], "")).unwrap_or_default();
    let disposition = format!("{}; filename=\"{name}\"", if download { "attachment" } else { "inline" });
    // Incognito output must not outlive its job in the browser's cache either.
    let private = path.components().any(|c| c.as_os_str() == ".incognito");
    let cache = if private { "no-store" } else { "max-age=3600" };
    let mut extra = vec![("Content-Disposition", disposition), ("Cache-Control", cache.into()), ("Accept-Ranges", "bytes".into())];
    let (status, start, count) = match range.map_or(Range::Whole, |r| byte_range(r, len)) {
        Range::Whole => (200, 0, len),
        Range::Part(a, b) => {
            extra.push(("Content-Range", format!("bytes {a}-{b}/{len}")));
            (206, a, b - a + 1)
        }
        Range::Unsatisfiable => {
            let head = [("Content-Range", format!("bytes */{len}"))];
            let head: Vec<(&str, &str)> = head.iter().map(|(k, v)| (*k, v.as_str())).collect();
            respond_with(w, 416, "text/plain", &head, b"", true)?;
            return Ok(true);
        }
    };
    let extra: Vec<(&str, &str)> = extra.iter().map(|(k, v)| (*k, v.as_str())).collect();
    oaiy_engine::http::respond_head(w, status, content_type(path), &extra, count, true)?;
    file.seek(SeekFrom::Start(start))?;
    io::copy(&mut file.take(count), w)?;
    io::Write::flush(w)?;
    Ok(true)
}

pub fn files(studio: &Studio, req: &Request, w: &mut TcpStream, rel: &str) -> io::Result<bool> {
    match media::resolve_file(&studio.output_root(), rel) {
        Some(p) => serve_file(w, &p, false, req.header("range")),
        None => send(w, Err(fail(404, "no such file"))),
    }
}

/// `X-OAIY-Incognito: 1` (or `true`, `yes`).
fn incognito_header(req: &Request) -> bool {
    req.header("x-oaiy-incognito").is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Whether the studio runs in incognito mode (`privacy.incognito`).
pub fn incognito_mode(studio: &Studio) -> bool {
    studio.config().get("privacy").is_some_and(|p| bool_or(p, "incognito", false))
}

/// For an incognito request (global mode, `X-OAIY-Incognito: 1`, or
/// `"incognito": true`), a private folder its job writes everything into and
/// that is deleted with it; `None` otherwise.
fn scratch_for(studio: &Studio, req: &Request, body: Option<&Json>) -> Option<PathBuf> {
    let asked = incognito_header(req)
        || body.and_then(|b| b.get("incognito")).and_then(Json::as_bool) == Some(true);
    (incognito_mode(studio) || asked).then(|| studio.output_root().join(".incognito").join(crate::util::random_id("")))
}

/// Uploaded reference images: extension and bytes, checked but not yet saved.
type Uploads = Vec<(&'static str, Vec<u8>)>;

/// An edit request as the JSON an image job takes, plus any multipart files
/// (saved later by `save_uploads`, which puts their paths in `images`). Text
/// fields become strings or numbers. Files can arrive only as files: a text
/// field cannot name a path.
fn edit_body(req: &Request) -> Result<(Json, Uploads), Reply> {
    if req.method != "POST" {
        return Err(fail(405, "use POST"));
    }
    const NO_MASK: &str = "mask is not supported: Qwen Image edits the whole picture from the instruction in the prompt";
    let ctype = req.header("content-type").unwrap_or("");
    let mut uploads = Vec::new();
    let body = if multipart(req) {
        let parts = crate::multipart::parse(ctype, &req.body).map_err(|e| fail(400, e))?;
        let mut fields: Vec<(String, Json)> = Vec::new();
        for p in parts {
            let name = p.name.trim_end_matches("[]").to_string();
            match name.as_str() {
                "mask" => return Err(fail(400, NO_MASK)),
                "image" | "images" => {
                    let label = p.filename.clone().unwrap_or_else(|| "image".into());
                    let (ext, ..) = crate::multipart::image_info(&p.data).ok_or_else(|| fail(400, format!("{label}: not a PNG, JPEG or WebP image")))?;
                    if p.data.len() > 32 << 20 {
                        return Err(fail(400, "reference images are limited to 32 MiB"));
                    }
                    uploads.push((ext, p.data));
                }
                _ if p.filename.is_some() => return Err(fail(400, format!("unexpected file field {name}"))),
                _ => {
                    let text = String::from_utf8(p.data).map_err(|_| fail(400, format!("{name} is not UTF-8 text")))?;
                    let t = text.trim();
                    let value = if name == "prompt" || name == "negative_prompt" {
                        Json::Str(text.clone())
                    } else if let Ok(n) = t.parse::<i64>() {
                        Json::Int(n)
                    } else if let Ok(x) = t.parse::<f64>() {
                        Json::Num(x)
                    } else if t == "true" || t == "false" {
                        Json::Bool(t == "true")
                    } else {
                        Json::Str(text.clone())
                    };
                    fields.push((name, value));
                }
            }
        }
        Json::Obj(fields)
    } else {
        let body = parse_body(req)?;
        if body.get("mask").is_some_and(|m| !matches!(m, Json::Null)) {
            return Err(fail(400, NO_MASK));
        }
        body
    };
    let count: usize = ["images", "image"].iter().filter_map(|k| body.get(k)).map(|v| v.as_array().map_or(1, <[Json]>::len)).sum();
    if count + uploads.len() == 0 {
        return Err(fail(400, "an edit needs at least one image"));
    }
    Ok((body, uploads))
}

/// Write checked uploads under `scratch/inputs` (incognito) or the output
/// root's `inputs/`, returning their paths.
fn save_uploads(studio: &Studio, scratch: Option<&std::path::Path>, uploads: Uploads) -> Result<Vec<Json>, Reply> {
    let dir = scratch.map_or_else(|| studio.output_root().join("inputs"), |s| s.join("inputs"));
    std::fs::create_dir_all(&dir).map_err(|e| fail(500, e.to_string()))?;
    let mut saved: Vec<PathBuf> = Vec::new();
    for (ext, data) in uploads {
        let path = dir.join(format!("{}.{ext}", crate::util::random_id("ref_")));
        if let Err(e) = std::fs::write(&path, &data) {
            for p in &saved {
                let _ = std::fs::remove_file(p);
            }
            return Err(fail(500, e.to_string()));
        }
        saved.push(path);
    }
    Ok(saved.iter().map(|p| Json::str(p.to_string_lossy())).collect())
}

/// The request fields `images_openai` rejects, checked up front.
fn check_image_body(body: &Json) -> Result<(), Reply> {
    if !matches!(body.get("response_format").and_then(Json::as_str), None | Some("b64_json" | "url")) {
        return Err(fail(400, "response_format must be b64_json or url"));
    }
    if body.get("output_format").and_then(Json::as_str).is_some_and(|f| f != "png") {
        return Err(fail(400, "output_format: this server writes png"));
    }
    Ok(())
}

/// Multipart edits carry files the gateway saved itself, so their paths are safe.
fn multipart(req: &Request) -> bool {
    req.header("content-type").is_some_and(|c| c.to_ascii_lowercase().starts_with("multipart/form-data"))
}

/// OpenAI audio.speech: speak `input` in a saved or described voice, wait,
/// and answer with the audio itself (converted by FFmpeg as asked).
fn speech_openai(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    match speech_audio(studio, req) {
        Ok((bytes, ctype)) => {
            respond(w, 200, ctype, &bytes, true)?;
            Ok(true)
        }
        Err(e) => send(w, Err(e)),
    }
}

fn speech_audio(studio: &Arc<Studio>, req: &Request) -> Result<(Vec<u8>, &'static str), Reply> {
    if req.method != "POST" {
        return Err(fail(405, "use POST"));
    }
    let body = parse_body(req)?;
    let format = body.get("response_format").and_then(Json::as_str).unwrap_or("mp3");
    let ctype = crate::speech::FORMATS.iter().find(|(f, _)| *f == format).map(|(_, c)| *c)
        .ok_or_else(|| fail(400, "response_format must be mp3, opus, aac, flac, wav or pcm"))?;
    let speed = match body.get("speed") {
        None | Some(Json::Null) => 1.0,
        Some(v) => v.as_f64().filter(|s| (0.25..=4.0).contains(s)).ok_or_else(|| fail(400, "speed must be between 0.25 and 4"))?,
    };
    if body.get("stream_format").and_then(Json::as_str).is_some_and(|s| s != "audio") {
        return Err(fail(400, "stream_format: the whole clip is returned at once (\"audio\")"));
    }
    let cfg = studio.config();
    let scratch = scratch_for(studio, req, Some(&body));
    let out = scratch.clone().unwrap_or_else(|| studio.output_root());
    let drop_scratch = |e: String| {
        if let Some(s) = &scratch {
            let _ = std::fs::remove_dir_all(s);
        }
        fail(400, e)
    };
    // A music model sings: `input` is the lyrics and `instructions` the style,
    // as the official Music 3 server takes them.
    let music = crate::music::names_music_model(&cfg, &body);
    let (job, wait) = if music {
        let (request, model, label, frames, seconds) = crate::music::music_request(&cfg, &studio.root, &media::day_dir(&out, "music"), &body).map_err(drop_scratch)?;
        (studio.media.submit(Kind::Music, request, model, label, frames, seconds, keep_jobs(&cfg), scratch), MUSIC_WAIT)
    } else {
        let (request, model, voice, frames) = crate::speech::speech_request(&cfg, &studio.root, &media::day_dir(&out, "speech"), &body).map_err(drop_scratch)?;
        (studio.media.submit(Kind::Speech, request, model, voice, frames, 0.0, keep_jobs(&cfg), scratch), SPEECH_WAIT)
    };
    let done = studio.media.wait(&job.id, wait).ok_or_else(|| fail(500, "the job disappeared"))?;
    let finish = |studio: &Arc<Studio>| {
        if done.incognito {
            if done.finished() { studio.media.purge(&done.id) } else { studio.media.remove(&done.id); }
        }
    };
    if done.status != "completed" {
        finish(studio);
        return Err(fail(if done.status == "in_progress" { 504 } else { 500 }, done.error.clone().unwrap_or_else(|| format!("{} job {}", done.kind.name(), done.status))));
    }
    let wav = done.files.first().cloned().ok_or_else(|| fail(500, "the job wrote no audio"))?;
    let bytes = convert_audio(studio, &cfg, &wav, format, speed, music);
    finish(studio);
    Ok((bytes?, ctype))
}

/// The worker's WAV as `format` at `speed` (FFmpeg unless it is WAV at 1x).
fn convert_audio(studio: &Arc<Studio>, cfg: &Json, wav: &std::path::Path, format: &str, speed: f64, music: bool) -> Result<Vec<u8>, Reply> {
    if format == "wav" && speed == 1.0 {
        return std::fs::read(wav).map_err(|e| fail(500, e.to_string()));
    }
    let ffmpeg = crate::config::program(&studio.root, cfg.get("media").and_then(|m| m.get("video")).map_or("ffmpeg", |v| str_or(v, "ffmpeg", "ffmpeg")));
    let args = if music { crate::music::convert_args(format, speed) } else { crate::speech::convert_args(format, speed) };
    let mut command = std::process::Command::new(&ffmpeg);
    command.args(["-hide_banner", "-loglevel", "error", "-i"]).arg(wav).args(args).arg("pipe:1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    match command.output() {
        Ok(o) if o.status.success() => Ok(o.stdout),
        Ok(o) => Err(fail(500, format!("FFmpeg could not write {format}: {}", String::from_utf8_lossy(&o.stderr).trim()))),
        Err(e) => Err(fail(500, format!("FFmpeg ({}) is needed for {format} or speed: {e}", ffmpeg.display()))),
    }
}

/// A song job as an object shaped like OpenAI's video jobs.
pub fn music_object(job: &Job) -> Json {
    let (status, error) = match job.status.as_str() {
        "cancelled" => ("failed", Json::obj([("code", Json::str("cancelled")), ("message", Json::str("the job was cancelled"))])),
        "failed" => ("failed", Json::obj([("code", Json::str("generation_failed")), ("message", Json::str(job.error.as_deref().unwrap_or("failed")))])),
        s => (s, Json::Null),
    };
    let r = &job.result;
    Json::obj([
        ("id", Json::str(&job.id)),
        ("object", Json::str("music")),
        ("model", Json::str(&job.model)),
        ("status", Json::str(status)),
        ("progress", Json::Int(job.progress.floor() as i64)),
        ("created_at", Json::Int(job.created_at as i64)),
        ("completed_at", job.completed_at.map_or(Json::Null, |t| Json::Int(t as i64))),
        ("prompt", Json::str(&job.prompt)),
        ("lyrics", job.request.get("lyrics").cloned().unwrap_or(Json::Null)),
        // The cap until it is done; then the song's length.
        ("seconds", r.get("duration").cloned().unwrap_or(Json::Num(job.seconds))),
        ("finish_reason", r.get("finish_reason").cloned().unwrap_or(Json::Null)),
        ("seed", job.request.get("seed").cloned().unwrap_or(Json::Null)),
        ("sample_rate", Json::Int(44_100)),
        ("channels", Json::Int(2)),
        ("error", error),
    ])
}

/// Songs as asynchronous jobs: create, list, get, delete, download
/// (`/content?format=mp3|opus|aac|flac|wav|pcm`).
fn music_openai(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, method: &str, rest: &str) -> io::Result<bool> {
    let parts: Vec<&str> = rest.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    let song = |id: &str| studio.media.get(id).filter(|j| j.kind == Kind::Music && str_or(&j.request, "kind", "") == "music");
    match (method, parts.as_slice()) {
        ("POST", []) => send(w, parse_body(req).and_then(|b| {
            let cfg = studio.config();
            let scratch = scratch_for(studio, req, Some(&b));
            let out = scratch.clone().unwrap_or_else(|| studio.output_root());
            let (request, model, label, frames, seconds) = crate::music::music_request(&cfg, &studio.root, &media::day_dir(&out, "music"), &b).map_err(|e| {
                if let Some(s) = &scratch {
                    let _ = std::fs::remove_dir_all(s);
                }
                fail(400, e)
            })?;
            let job = studio.media.submit(Kind::Music, request, model, label, frames, seconds, keep_jobs(&cfg), scratch);
            Ok(Reply { status: 200, body: music_object(&job) })
        })),
        ("GET", []) => {
            let limit = req.query("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
            let songs: Vec<Job> = studio.media.list().into_iter().filter(|j| j.kind == Kind::Music && str_or(&j.request, "kind", "") == "music").collect();
            let has_more = songs.len() > limit;
            let data: Vec<Json> = songs.iter().take(limit).map(music_object).collect();
            let id = |i: Option<&Json>| i.and_then(|v| v.get("id")).cloned().unwrap_or(Json::Null);
            json_reply(w, 200, &Json::obj([
                ("object", Json::str("list")),
                ("first_id", id(data.first())),
                ("last_id", id(data.last())),
                ("has_more", Json::Bool(has_more)),
                ("data", Json::Arr(data)),
            ]))
        }
        ("GET", [id]) => send(w, song(id).map(|j| Reply { status: 200, body: music_object(&j) }).ok_or_else(|| fail(404, "no such song"))),
        ("DELETE", [id]) => {
            let deleted = song(id).is_some() && studio.media.remove(id);
            send(w, Ok(Reply { status: if deleted { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("object", Json::str("music.deleted")), ("deleted", Json::Bool(deleted))]) }))
        }
        ("POST", [id, "cancel"]) => {
            let cancelled = song(id).is_some() && studio.media.cancel(id);
            send(w, Ok(Reply { status: if cancelled { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("cancelled", Json::Bool(cancelled))]) }))
        }
        ("GET", [id, "content"]) => {
            let Some(job) = song(id) else { return send(w, Err(fail(404, "no such song"))) };
            if job.status != "completed" {
                return send(w, Err(fail(409, format!("the song is {}", job.status))));
            }
            let Some(wav) = job.files.first().filter(|f| f.is_file()).cloned() else { return send(w, Err(fail(404, "the file is gone"))) };
            let format = req.query("format").unwrap_or_else(|| "wav".into());
            if format == "wav" {
                return serve_file(w, &wav, true, req.header("range"));
            }
            let Some(ctype) = crate::speech::FORMATS.iter().find(|(f, _)| *f == format).map(|(_, c)| *c) else {
                return send(w, Err(fail(400, "format must be mp3, opus, aac, flac, wav or pcm")));
            };
            let cfg = studio.config();
            match convert_audio(studio, &cfg, &wav, &format, 1.0, true) {
                Ok(bytes) => {
                    respond(w, 200, ctype, &bytes, true)?;
                    Ok(true)
                }
                Err(e) => send(w, Err(e)),
            }
        }
        _ => send(w, Err(fail(405, format!("{method} is not supported here")))),
    }
}

pub fn sound_object(job: &Job) -> Json {
    let (status, error) = match job.status.as_str() {
        "cancelled" => ("failed", Json::obj([("code", Json::str("cancelled")), ("message", Json::str("the job was cancelled"))])),
        "failed" => ("failed", Json::obj([("code", Json::str("generation_failed")), ("message", Json::str(job.error.as_deref().unwrap_or("failed")))])),
        s => (s, Json::Null),
    };
    Json::obj([
        ("id", Json::str(&job.id)),
        ("object", Json::str("sound_effect")),
        ("model", Json::str(&job.model)),
        ("status", Json::str(status)),
        ("progress", Json::Int(job.progress.floor() as i64)),
        ("created_at", Json::Int(job.created_at as i64)),
        ("completed_at", job.completed_at.map_or(Json::Null, |t| Json::Int(t as i64))),
        ("prompt", Json::str(&job.prompt)),
        ("seconds", job.result.get("duration").cloned().unwrap_or(Json::Num(job.seconds))),
        ("seed", job.request.get("seed").cloned().unwrap_or(Json::Null)),
        ("sample_rate", Json::Int(crate::sound::SAMPLE_RATE)),
        ("channels", Json::Int(1)),
        ("error", error),
    ])
}

/// Sound effects as asynchronous jobs: create, list, get, cancel, delete, download
/// (`/content?format=mp3|opus|aac|flac|wav|pcm`).
fn sound_openai(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, method: &str, rest: &str) -> io::Result<bool> {
    let parts: Vec<&str> = rest.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    let effect = |id: &str| studio.media.get(id).filter(|j| j.kind == Kind::Sound);
    match (method, parts.as_slice()) {
        ("POST", []) => send(w, parse_body(req).and_then(|b| {
            let cfg = studio.config();
            let scratch = scratch_for(studio, req, Some(&b));
            let out = scratch.clone().unwrap_or_else(|| studio.output_root());
            let (request, model, label, seconds) = crate::sound::sound_request(&cfg, &studio.root, &media::day_dir(&out, "sound"), &b).map_err(|e| {
                if let Some(s) = &scratch {
                    let _ = std::fs::remove_dir_all(s);
                }
                fail(400, e)
            })?;
            let job = studio.media.submit(Kind::Sound, request, model, label, 1, seconds, keep_jobs(&cfg), scratch);
            Ok(Reply { status: 200, body: sound_object(&job) })
        })),
        ("GET", []) => {
            let limit = req.query("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
            let effects: Vec<Job> = studio.media.list().into_iter().filter(|j| j.kind == Kind::Sound).collect();
            let has_more = effects.len() > limit;
            let data: Vec<Json> = effects.iter().take(limit).map(sound_object).collect();
            let id = |i: Option<&Json>| i.and_then(|v| v.get("id")).cloned().unwrap_or(Json::Null);
            json_reply(w, 200, &Json::obj([
                ("object", Json::str("list")),
                ("first_id", id(data.first())),
                ("last_id", id(data.last())),
                ("has_more", Json::Bool(has_more)),
                ("data", Json::Arr(data)),
            ]))
        }
        ("GET", [id]) => send(w, effect(id).map(|j| Reply { status: 200, body: sound_object(&j) }).ok_or_else(|| fail(404, "no such sound effect"))),
        ("DELETE", [id]) => {
            let deleted = effect(id).is_some() && studio.media.remove(id);
            send(w, Ok(Reply { status: if deleted { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("object", Json::str("sound_effect.deleted")), ("deleted", Json::Bool(deleted))]) }))
        }
        ("POST", [id, "cancel"]) => {
            let cancelled = effect(id).is_some() && studio.media.cancel(id);
            send(w, Ok(Reply { status: if cancelled { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("cancelled", Json::Bool(cancelled))]) }))
        }
        ("GET", [id, "content"]) => {
            let Some(job) = effect(id) else { return send(w, Err(fail(404, "no such sound effect"))) };
            if job.status != "completed" {
                return send(w, Err(fail(409, format!("the sound effect is {}", job.status))));
            }
            let Some(wav) = job.files.first().filter(|f| f.is_file()).cloned() else { return send(w, Err(fail(404, "the file is gone"))) };
            let format = req.query("format").unwrap_or_else(|| "wav".into());
            if format == "wav" {
                return serve_file(w, &wav, true, req.header("range"));
            }
            let Some(ctype) = crate::speech::FORMATS.iter().find(|(f, _)| *f == format).map(|(_, c)| *c) else {
                return send(w, Err(fail(400, "format must be mp3, opus, aac, flac, wav or pcm")));
            };
            let cfg = studio.config();
            match convert_audio(studio, &cfg, &wav, &format, 1.0, true) {
                Ok(bytes) => {
                    respond(w, 200, ctype, &bytes, true)?;
                    Ok(true)
                }
                Err(e) => send(w, Err(e)),
            }
        }
        _ => send(w, Err(fail(405, format!("{method} is not supported here")))),
    }
}

pub fn model3d_object(job: &Job) -> Json {
    let (status, error) = match job.status.as_str() {
        "cancelled" => ("failed", Json::obj([("code", Json::str("cancelled")), ("message", Json::str("the job was cancelled"))])),
        "failed" => ("failed", Json::obj([("code", Json::str("generation_failed")), ("message", Json::str(job.error.as_deref().unwrap_or("failed")))])),
        s => (s, Json::Null),
    };
    let r = |k: &str| job.result.get(k).cloned().unwrap_or(Json::Null);
    Json::obj([
        ("id", Json::str(&job.id)),
        ("object", Json::str("model3d")),
        ("model", Json::str(&job.model)),
        ("status", Json::str(status)),
        ("progress", Json::Int(job.progress.floor() as i64)),
        ("stage", Json::str(&job.stage)),
        ("created_at", Json::Int(job.created_at as i64)),
        ("completed_at", job.completed_at.map_or(Json::Null, |t| Json::Int(t as i64))),
        ("seed", job.request.get("seed").cloned().unwrap_or(Json::Null)),
        ("resolution", job.request.get("resolution").cloned().unwrap_or(Json::Null)),
        ("format", Json::str("glb")),
        ("faces", r("faces")),
        ("vertices", r("vertices")),
        ("bytes", r("bytes")),
        ("matte", r("matte")),
        ("upscaled", r("upscaled")),
        ("seconds_taken", r("seconds")),
        ("error", error),
    ])
}

/// 3D models from a picture as asynchronous jobs: create, list, get, cancel, delete,
/// download the GLB (`/content`) and see the picture as it was cut out (`/input`).
fn model3d_openai(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, method: &str, rest: &str, trusted: bool) -> io::Result<bool> {
    let parts: Vec<&str> = rest.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    let model = |id: &str| studio.media.get(id).filter(|j| j.kind == Kind::Model3d);
    match (method, parts.as_slice()) {
        ("POST", []) => send(w, parse_body(req).and_then(|b| {
            let cfg = studio.config();
            let scratch = scratch_for(studio, req, Some(&b));
            let out = scratch.clone().unwrap_or_else(|| studio.output_root());
            let (request, name, label) = crate::model3d::model3d_request(&cfg, &studio.root, &media::day_dir(&out, "3d"), &b, trusted).map_err(|e| {
                if let Some(s) = &scratch {
                    let _ = std::fs::remove_dir_all(s);
                }
                fail(400, e)
            })?;
            let job = studio.media.submit(Kind::Model3d, request, name, label, 1, 0.0, keep_jobs(&cfg), scratch);
            Ok(Reply { status: 200, body: model3d_object(&job) })
        })),
        ("GET", []) => {
            let limit = req.query("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
            let models: Vec<Job> = studio.media.list().into_iter().filter(|j| j.kind == Kind::Model3d).collect();
            let has_more = models.len() > limit;
            let data: Vec<Json> = models.iter().take(limit).map(model3d_object).collect();
            let id = |i: Option<&Json>| i.and_then(|v| v.get("id")).cloned().unwrap_or(Json::Null);
            json_reply(w, 200, &Json::obj([
                ("object", Json::str("list")),
                ("first_id", id(data.first())),
                ("last_id", id(data.last())),
                ("has_more", Json::Bool(has_more)),
                ("data", Json::Arr(data)),
            ]))
        }
        ("GET", [id]) => send(w, model(id).map(|j| Reply { status: 200, body: model3d_object(&j) }).ok_or_else(|| fail(404, "no such 3D model"))),
        ("DELETE", [id]) => {
            let deleted = model(id).is_some() && studio.media.remove(id);
            send(w, Ok(Reply { status: if deleted { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("object", Json::str("model3d.deleted")), ("deleted", Json::Bool(deleted))]) }))
        }
        ("POST", [id, "cancel"]) => {
            let cancelled = model(id).is_some() && studio.media.cancel(id);
            send(w, Ok(Reply { status: if cancelled { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("cancelled", Json::Bool(cancelled))]) }))
        }
        ("GET", [id, what]) if *what == "content" || *what == "input" => {
            let Some(job) = model(id) else { return send(w, Err(fail(404, "no such 3D model"))) };
            if job.status != "completed" {
                return send(w, Err(fail(409, format!("the 3D model is {}", job.status))));
            }
            let file = if *what == "content" { job.files.first().cloned() } else { job.preview.clone() };
            match file.filter(|f| f.is_file()) {
                Some(f) => serve_file(w, &f, true, req.header("range")),
                None => send(w, Err(fail(404, "the file is gone"))),
            }
        }
        _ => send(w, Err(fail(405, format!("{method} is not supported here")))),
    }
}

/// Saved voices: list, design and save, show, sample clip, delete.
fn voices(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, rest: &str) -> io::Result<bool> {
    let cfg = studio.config();
    let dir = crate::speech::voices_dir(&cfg, &studio.root);
    let parts: Vec<String> = rest.split('/').filter(|s| !s.is_empty()).map(oaiy_engine::http::percent_decode).collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    match (req.method.as_str(), parts.as_slice()) {
        ("GET", []) => json_reply(w, 200, &Json::obj([("object", Json::str("list")), ("data", Json::Arr(crate::speech::list(&dir)))])),
        ("POST", []) => send(w, create_voice(studio, req, &dir)),
        ("GET", [name]) => match crate::speech::get(&dir, name) {
            Some(v) => json_reply(w, 200, &v),
            None => send(w, Err(fail(404, format!("no voice named {name}")))),
        },
        ("GET", [name, "sample"]) => {
            let clip = crate::speech::sample_file(&dir, name);
            if crate::speech::valid_name(name) && clip.is_file() {
                serve_file(w, &clip, false, req.header("range"))
            } else {
                send(w, Err(fail(404, format!("no sample for {name}"))))
            }
        }
        ("DELETE", [name]) => {
            if crate::speech::remove(&dir, name) {
                json_reply(w, 200, &Json::obj([("id", Json::str(*name)), ("object", Json::str("voice.deleted")), ("deleted", Json::Bool(true))]))
            } else {
                send(w, Err(fail(404, format!("no voice named {name}"))))
            }
        }
        _ => send(w, Err(fail(405, "voices: GET (list), POST (save one), GET/DELETE {name}, GET {name}/sample"))),
    }
}

/// Design a voice from a description and save it under its name, or, with
/// `keep: false` (and always in incognito), hand it back instead: the voice
/// and its sample clip, for the caller to keep and send with each request
/// (`voice` as that object), with nothing left here.
fn create_voice(studio: &Arc<Studio>, req: &Request, dir: &std::path::Path) -> Result<Reply, Reply> {
    let body = parse_body(req)?;
    let incognito = incognito_mode(studio) || incognito_header(req) || body.get("incognito").and_then(Json::as_bool) == Some(true);
    let keep = match body.get("keep").and_then(Json::as_bool) {
        Some(true) if incognito => return Err(fail(400, "incognito keeps nothing, so voices are not saved here; send keep: false to have the voice handed back instead")),
        Some(k) => k,
        None => !incognito,
    };
    let name = body.get("name").and_then(Json::as_str).unwrap_or("").trim().to_string();
    if keep && crate::speech::get(dir, &name).is_some() && body.get("replace").and_then(Json::as_bool) != Some(true) {
        return Err(fail(409, format!("a voice named {name} already exists; delete it first, or send replace: true")));
    }
    let cfg = studio.config();
    let (request, model, label) = crate::speech::voice_request(&cfg, &studio.root, &media::day_dir(&studio.output_root(), "voices"), &body).map_err(|e| fail(400, e))?;
    let job = studio.media.submit(Kind::Speech, request, model, format!("voice: {label}"), 1, 0.0, keep_jobs(&cfg), None);
    let done = studio.media.wait(&job.id, SPEECH_WAIT).ok_or_else(|| fail(500, "the job disappeared"))?;
    if done.status != "completed" {
        return Err(fail(if done.status == "in_progress" { 504 } else { 500 }, done.error.unwrap_or_else(|| format!("voice job {}", done.status))));
    }
    let path = |k: &str| done.result.get(k).and_then(Json::as_str).map(std::path::PathBuf::from).ok_or_else(|| fail(500, format!("the voice job returned no {k}")));
    if !keep {
        let handed = crate::speech::hand_back(&label, &path("voice_file")?, &path("path")?).map_err(|e| fail(500, e));
        studio.media.remove(&done.id);
        return handed.and_then(ok);
    }
    let voice = crate::speech::keep(dir, &label, &path("voice_file")?, &path("path")?).map_err(|e| fail(500, e))?;
    // Its files have moved into the voices folder; the job has nothing left to show.
    studio.media.remove(&done.id);
    ok(voice)
}

/// OpenAI Images: generate, wait, and answer with base64 PNGs or URLs.
fn images_openai(studio: &Arc<Studio>, req: &Request, body: Json, trusted: bool, scratch: Option<PathBuf>) -> Result<Reply, Reply> {
    if req.method != "POST" {
        return Err(fail(405, "use POST"));
    }
    check_image_body(&body)?;
    let format = body.get("response_format").and_then(Json::as_str).unwrap_or("b64_json");
    let job = submit_image(studio, &body, trusted || multipart(req), scratch)?;
    let done = studio.media.wait(&job.id, IMAGE_WAIT).ok_or_else(|| fail(500, "the job disappeared"))?;
    if done.status != "completed" {
        if done.incognito {
            // Still running after the wait: stop it too, or its scratch folder
            // would vanish under a worker nobody can cancel any more.
            if done.finished() {
                studio.media.purge(&done.id);
            } else {
                studio.media.remove(&done.id);
            }
        }
        return Err(fail(if done.status == "in_progress" { 504 } else { 500 }, done.error.unwrap_or_else(|| format!("image job {}", done.status))));
    }
    let root = studio.output_root();
    let base = public_base(studio, req);
    let mut data = Vec::new();
    for file in &done.files {
        let item = if format == "url" {
            let url = media::file_url(&root, file).ok_or_else(|| fail(500, "output outside the output root"))?;
            ("url", Json::str(format!("{base}{url}")))
        } else {
            let bytes = std::fs::read(file).map_err(|e| fail(500, format!("reading {}: {e}", file.display())))?;
            ("b64_json", Json::str(base64_encode(&bytes)))
        };
        data.push(Json::obj([item, ("revised_prompt", Json::str(&done.prompt))]));
    }
    // Incognito images returned inline exist nowhere else now; URLs stay
    // fetchable until the job expires (ten minutes).
    if done.incognito && format != "url" {
        studio.media.purge(&done.id);
    }
    ok(Json::obj([
        ("created", Json::Int(now() as i64)),
        ("data", Json::Arr(data)),
        ("output_format", Json::str("png")),
        ("size", Json::str(&done.size)),
        ("model", Json::str(&done.model)),
    ]))
}

/// A picture's background removed, or the picture made larger: wait, and answer
/// as OpenAI Images does, with a base64 PNG or a URL.
fn picture_openai(studio: &Arc<Studio>, req: &Request, body: Json, trusted: bool, op: crate::picture::Op) -> Result<Reply, Reply> {
    if req.method != "POST" {
        return Err(fail(405, "use POST"));
    }
    let format = body.get("response_format").and_then(Json::as_str).unwrap_or("b64_json");
    if !matches!(format, "b64_json" | "url") {
        return Err(fail(400, "response_format must be b64_json or url"));
    }
    let cfg = studio.config();
    let scratch = scratch_for(studio, req, Some(&body));
    let root = scratch.clone().unwrap_or_else(|| studio.output_root());
    let (request, model, label) = crate::picture::picture_request(&cfg, &studio.root, &root, &body, trusted, op).map_err(|e| {
        if let Some(s) = &scratch {
            let _ = std::fs::remove_dir_all(s);
        }
        fail(400, e)
    })?;
    let job = studio.media.submit(Kind::Picture, request, model, label, 1, 0.0, keep_jobs(&cfg), scratch);
    let done = studio.media.wait(&job.id, IMAGE_WAIT).ok_or_else(|| fail(500, "the job disappeared"))?;
    if done.status != "completed" {
        if done.incognito {
            if done.finished() {
                studio.media.purge(&done.id);
            } else {
                studio.media.remove(&done.id);
            }
        }
        return Err(fail(if done.status == "in_progress" { 504 } else { 500 }, done.error.unwrap_or_else(|| format!("picture job {}", done.status))));
    }
    let file = done.files.first().ok_or_else(|| fail(500, "the job made no picture"))?;
    let item = if format == "url" {
        let url = media::file_url(&studio.output_root(), file).ok_or_else(|| fail(500, "output outside the output root"))?;
        ("url", Json::str(format!("{}{url}", public_base(studio, req))))
    } else {
        let bytes = std::fs::read(file).map_err(|e| fail(500, format!("reading {}: {e}", file.display())))?;
        ("b64_json", Json::str(base64_encode(&bytes)))
    };
    if done.incognito && format != "url" {
        studio.media.purge(&done.id);
    }
    let int = |k: &str| done.result.get(k).cloned().unwrap_or(Json::Null);
    ok(Json::obj([
        ("created", Json::Int(now() as i64)),
        ("data", Json::Arr(vec![Json::obj([item])])),
        ("output_format", Json::str("png")),
        ("width", int("width")),
        ("height", int("height")),
        ("size", Json::str(format!("{}x{}", crate::util::int_or(&done.result, "width", 0), crate::util::int_or(&done.result, "height", 0)))),
        ("model", Json::str(&done.model)),
        ("seconds_taken", int("seconds")),
    ]))
}

fn submit_image(studio: &Arc<Studio>, body: &Json, allow_local: bool, scratch: Option<PathBuf>) -> Result<Job, Reply> {
    let cfg = studio.config();
    let root = scratch.clone().unwrap_or_else(|| studio.output_root());
    let prepared = media::image_request(&cfg, &studio.root, &root, body, allow_local);
    let (request, model, size, n) = prepared.map_err(|e| {
        if let Some(s) = &scratch {
            let _ = std::fs::remove_dir_all(s);
        }
        fail(400, e)
    })?;
    Ok(studio.media.submit(Kind::Image, request, model, size, n, 0.0, keep_jobs(&cfg), scratch))
}

fn submit_video(studio: &Arc<Studio>, body: &Json, trusted: bool, scratch: Option<PathBuf>) -> Result<Job, Reply> {
    let cfg = studio.config();
    let root = scratch.clone().unwrap_or_else(|| studio.output_root());
    let prepared = media::video_request(&cfg, &studio.root, &root, body, trusted);
    let (request, model, size, seconds) = prepared.map_err(|e| {
        if let Some(s) = &scratch {
            let _ = std::fs::remove_dir_all(s);
        }
        fail(400, e)
    })?;
    Ok(studio.media.submit(Kind::Video, request, model, size, 1, seconds, keep_jobs(&cfg), scratch))
}

fn keep_jobs(cfg: &Json) -> usize {
    cfg.get("media").map_or(200, |m| int_or(m, "keep_jobs", 200)).clamp(1, 100_000) as usize
}

/// Where clients reach this gateway: `gateway.public_url`, else the Host they used.
pub fn public_base(studio: &Studio, req: &Request) -> String {
    let cfg = studio.config();
    let configured = cfg.get("gateway").map_or("", |g| str_or(g, "public_url", "")).trim_end_matches('/').to_string();
    if !configured.is_empty() {
        return configured;
    }
    format!("http://{}", req.header("host").unwrap_or("127.0.0.1"))
}

/// A job as an OpenAI video object.
pub fn video_object(job: &Job) -> Json {
    let (status, error) = match job.status.as_str() {
        "cancelled" => ("failed", Json::obj([("code", Json::str("cancelled")), ("message", Json::str("the job was cancelled"))])),
        "failed" => ("failed", Json::obj([("code", Json::str("generation_failed")), ("message", Json::str(job.error.as_deref().unwrap_or("failed")))])),
        s => (s, Json::Null),
    };
    let clip = job.clip_seconds();
    let seconds = if clip.fract().abs() < 0.05 { format!("{}", clip.round() as i64) } else { format!("{:.1}", clip) };
    Json::obj([
        ("id", Json::str(&job.id)),
        ("object", Json::str("video")),
        ("model", Json::str(&job.model)),
        ("status", Json::str(status)),
        ("progress", Json::Int(job.progress.floor() as i64)),
        ("created_at", Json::Int(job.created_at as i64)),
        ("completed_at", job.completed_at.map_or(Json::Null, |t| Json::Int(t as i64))),
        ("expires_at", Json::Null),
        ("prompt", Json::str(&job.prompt)),
        ("seconds", Json::str(seconds)),
        ("size", Json::str(&job.size)),
        ("remixed_from_video_id", Json::Null),
        ("error", error),
    ])
}

fn videos_openai(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, method: &str, rest: &str, trusted: bool) -> io::Result<bool> {
    let parts: Vec<&str> = rest.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    let video = |id: &str| studio.media.get(id).filter(|j| j.kind == Kind::Video);
    match (method, parts.as_slice()) {
        ("POST", []) => send(w, parse_body(req).and_then(|b| {
            let scratch = scratch_for(studio, req, Some(&b));
            submit_video(studio, &b, trusted, scratch)
        }).map(|j| Reply { status: 200, body: video_object(&j) })),
        ("GET", []) => {
            let limit = req.query("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
            let videos: Vec<Job> = studio.media.list().into_iter().filter(|j| j.kind == Kind::Video).collect();
            let has_more = videos.len() > limit;
            let data: Vec<Json> = videos.iter().take(limit).map(video_object).collect();
            let id = |i: Option<&Json>| i.and_then(|v| v.get("id")).cloned().unwrap_or(Json::Null);
            json_reply(w, 200, &Json::obj([
                ("object", Json::str("list")),
                ("first_id", id(data.first())),
                ("last_id", id(data.last())),
                ("has_more", Json::Bool(has_more)),
                ("data", Json::Arr(data)),
            ]))
        }
        ("GET", [id]) => send(w, video(id).map(|j| Reply { status: 200, body: video_object(&j) }).ok_or_else(|| fail(404, "no such video"))),
        ("DELETE", [id]) => {
            let deleted = studio.media.remove(id);
            send(w, Ok(Reply { status: if deleted { 200 } else { 404 }, body: Json::obj([("id", Json::str(*id)), ("object", Json::str("video.deleted")), ("deleted", Json::Bool(deleted))]) }))
        }
        ("GET", [id, "content"]) => {
            let Some(job) = video(id) else { return send(w, Err(fail(404, "no such video"))) };
            if job.status != "completed" {
                return send(w, Err(fail(409, format!("the video is {}", job.status))));
            }
            let file = match req.query("variant").as_deref().unwrap_or("video") {
                "video" => job.files.first().cloned(),
                "thumbnail" => job.preview.clone(),
                other => return send(w, Err(fail(400, format!("variant {other} is not available (video or thumbnail)")))),
            };
            match file.filter(|f| f.is_file()) {
                Some(f) => serve_file(w, &f, true, req.header("range")),
                None => send(w, Err(fail(404, "the file is gone"))),
            }
        }
        _ => send(w, Err(fail(405, format!("{method} is not supported here")))),
    }
}

/// oaiy-llm-server's asynchronous job API.
fn oaiy_jobs(studio: &Arc<Studio>, req: &Request, kind: Kind, rest: &str, trusted: bool) -> Result<Reply, Reply> {
    let latest = || match req.query("id") {
        Some(id) => studio.media.get(&id).filter(|j| j.kind == kind),
        None => studio.media.list().into_iter().find(|j| j.kind == kind),
    };
    match (req.method.as_str(), rest) {
        ("POST", "") => {
            let body = parse_body(req)?;
            let scratch = scratch_for(studio, req, Some(&body));
            let job = if kind == Kind::Image { submit_image(studio, &body, trusted, scratch)? } else { submit_video(studio, &body, trusted, scratch)? };
            Ok(Reply {
                status: 202,
                body: Json::obj([
                    ("id", Json::str(&job.id)),
                    ("state", Json::str("queued")),
                    ("status_url", Json::str(format!("{}/status?id={}", req.route().trim_end_matches('/'), job.id))),
                ]),
            })
        }
        ("GET", "/status") => {
            let job = latest();
            ok(Json::obj([
                ("configured", Json::Bool(true)),
                ("job", job.map_or(Json::Null, |j| {
                    let state = match j.status.as_str() { "in_progress" => "running", s => s };
                    Json::obj([
                        ("id", Json::str(&j.id)),
                        ("kind", Json::str(j.kind.name())),
                        ("state", Json::str(state)),
                        ("progress", Json::obj([("percent", Json::Num(j.progress)), ("stage", Json::str(&j.stage))])),
                        ("result", j.result.clone()),
                        ("error", j.error.as_ref().map_or(Json::Null, Json::str)),
                    ])
                })),
            ]))
        }
        ("POST", "/cancel") => {
            let job = match req.query("id") {
                Some(id) => studio.media.get(&id).filter(|j| j.kind == kind),
                None => studio.media.list().into_iter().find(|j| j.kind == kind && !j.finished()),
            };
            let cancelled = job.is_some_and(|j| studio.media.cancel(&j.id));
            ok(Json::obj([("cancelled", Json::Bool(cancelled))]))
        }
        _ => Err(fail(404, format!("no route {} {}", req.method, req.route()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes() -> Vec<Json> {
        crate::config::default_json().get("gateway").unwrap().get("routes").unwrap().as_array().unwrap().to_vec()
    }

    #[test]
    fn routes_match_by_method_and_own_their_subpaths() {
        let r = routes();
        let m = route(&r, "POST", "/v1/chat/completions").unwrap();
        assert_eq!(m.target, "chat");
        assert!(route(&r, "GET", "/v1/chat/completions").is_none());
        let v = route(&r, "GET", "/v1/videos/video_abc/content").unwrap();
        assert_eq!((v.target.as_str(), v.rest.as_str()), ("videos", "/video_abc/content"));
        assert_eq!(route(&r, "DELETE", "/v1/videos/x").unwrap().rest, "/x");
        assert_eq!(route(&r, "GET", "/files/images/a.png").unwrap().target, "files");
        assert!(route(&r, "POST", "/v1/videosx").is_none());
        assert!(route(&r, "POST", "/v1/chat/completions/extra").is_none());
        // Custom paths and dialects come from the table.
        let custom = vec![Json::parse(br#"{"path":"/api/gen","method":"POST","target":"images","spec":"oaiy"}"#).unwrap()];
        let m = route(&custom, "GET", "/api/gen/status").unwrap();
        assert_eq!((m.spec.as_str(), m.rest.as_str()), ("oaiy", "/status"));
        let disabled = vec![Json::parse(br#"{"path":"/x","method":"POST","target":"chat","enabled":false}"#).unwrap()];
        assert!(route(&disabled, "POST", "/x").is_none());
        // 3D model jobs own their sub-paths, and listing them is a GET on the route itself.
        let m = route(&r, "GET", "/v1/3d/models/m3d_abc/content").unwrap();
        assert_eq!((m.target.as_str(), m.rest.as_str()), ("model3d", "/m3d_abc/content"));
        assert_eq!(route(&r, "GET", "/v1/3d/models").unwrap().target, "model3d");
        assert_eq!(route(&r, "POST", "/v1/3d/models/m3d_abc/cancel").unwrap().rest, "/m3d_abc/cancel");
    }

    /// A stand-in for a language-model server on a loopback port: every request's head and body are kept, and
    /// answered with `answer`'s bytes, written whole before the connection closes (as tinygrad's server ends a
    /// reply: no length, the connection closed). Dropped: nothing listens there any more.
    struct StandIn {
        addr: String,
        seen: Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>,
        stop: Arc<std::sync::atomic::AtomicBool>,
    }

    impl StandIn {
        fn new(answer: &'static str) -> StandIn {
            use std::io::{BufRead, BufReader, Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (kept, stopped) = (Arc::clone(&seen), Arc::clone(&stop));
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stopped.load(std::sync::atomic::Ordering::SeqCst) {
                        return;
                    }
                    let Ok(mut stream) = stream else { continue };
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let (mut head, mut length) = (String::new(), 0usize);
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            length = n.trim().parse().unwrap_or(0);
                        }
                        head.push_str(&line);
                    }
                    let mut body = vec![0; length];
                    let _ = reader.read_exact(&mut body);
                    kept.lock().unwrap().push((head, body));
                    let _ = stream.write_all(answer.as_bytes());
                }
            });
            StandIn { addr, seen, stop }
        }

        fn requests(&self) -> Vec<(String, Json)> {
            self.seen.lock().unwrap().iter().map(|(head, body)| (head.clone(), Json::parse(body).unwrap_or(Json::Null))).collect()
        }
    }

    impl Drop for StandIn {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(&self.addr);
        }
    }

    /// A chat request as a client of the gateway sends it: the reply's status, its type and its body.
    fn chat(studio: &Arc<Studio>, body: &str) -> (u16, String, String) {
        ask(studio, "chat", body)
    }

    /// A request for the gateway's `target` (`chat` or `completions`), likewise.
    fn ask(studio: &Arc<Studio>, target: &'static str, body: &str) -> (u16, String, String) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let served = Arc::clone(studio);
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            oaiy_engine::http::serve(stream, |req, w| handle(&served, req, w, Matched { target: target.into(), spec: "openai".into(), rest: String::new() }, true));
        });
        let reply = fetch(&addr, "POST", "/v1/whatever-the-route-is", &[("Content-Type", "application/json")], body.as_bytes(), Duration::from_secs(900)).unwrap();
        let (status, kind) = (reply.status, reply.header("content-type").unwrap_or("").to_string());
        (status, kind, String::from_utf8_lossy(&reply.body(8 << 20).unwrap()).into_owned())
    }

    /// As tinygrad's server streams a reply, and as OAIY's own engine answers one whole.
    const TINYGRAD_SAYS: &str = "HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\ndata: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"from the eGPU\"}}]}\n\ndata: [DONE]\n\n";
    const ENGINE_SAYS: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\nConnection: close\r\n\r\n{\"answered\": \"on this computer\"}";

    fn egpu_studio(root: &std::path::Path, stands_in: &str) -> Arc<Studio> {
        let cfg = format!(
            r#"{{"llm": {{"enabled": true, "temperature": 0.7, "max_tokens": 256, "server": "no-such/oaiy-llm-server", "server_webgpu": "no-such/oaiy-llm-server-webgpu",
                "egpu": {{"enabled": true, "fallback_model": "{stands_in}"}},
                "models": [{{"name": "big", "path": "big.gguf", "egpu": true}}, {{"name": "small", "path": "small.gguf"}}]}},
                "media": {{"llm_policy": "coexist"}}}}"#
        );
        Arc::new(Studio::for_test(root, Json::parse(cfg.as_bytes()).unwrap()))
    }

    #[test]
    fn a_chat_with_a_model_on_the_egpu_is_answered_by_tinygrads_server_and_the_others_by_this_computers() {
        let root = std::env::temp_dir();
        let studio = egpu_studio(&root, "small");
        let (tinygrad, engine) = (StandIn::new(TINYGRAD_SAYS), StandIn::new(ENGINE_SAYS));
        studio.egpu.adopt(&tinygrad.addr, "sk-egpu-test", "big", None);
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        // The eGPU's model: tinygrad's server, with its key, OAIY's name for the model, and the settings tinygrad
        // would otherwise decide differently. Its stream, which says no length and ends by closing, arrives whole.
        let (status, kind, body) = chat(&studio, r#"{"model": "big", "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!((status, kind.as_str()), (200, "text/event-stream"));
        assert!(body.contains("from the eGPU") && body.trim_end().ends_with("data: [DONE]"), "{body}");
        let sent = tinygrad.requests();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].0.starts_with("POST /v1/chat/completions ") && sent[0].0.contains("Authorization: Bearer sk-egpu-test"), "{}", sent[0].0);
        assert_eq!(sent[0].1.get("model").and_then(Json::as_str), Some("big"));
        assert_eq!(sent[0].1.get("temperature").and_then(Json::as_f64), Some(0.7));
        assert_eq!(sent[0].1.get("max_tokens").and_then(Json::as_i64), Some(256));
        assert!(engine.requests().is_empty(), "this computer's engine was not asked");
        // Whether to think first, which tinygrad's server reads from no request: said beside each, as OAIY's own
        // engine would decide it (not unless asked).
        assert!(sent[0].0.contains("X-OAIY-Thinking: 0"), "{}", sent[0].0);
        assert_eq!(chat(&studio, r#"{"model": "big", "reasoning_effort": "low", "messages": [{"role": "user", "content": "hi"}]}"#).0, 200);
        assert!(tinygrad.requests()[1].0.contains("X-OAIY-Thinking: 1"));
        // Another model: this computer's engine, as it was sent.
        let (status, _, body) = chat(&studio, r#"{"model": "small", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("on this computer"));
        assert_eq!(engine.requests().last().unwrap().1.get("model").and_then(Json::as_str), Some("small"));
        assert_eq!(engine.requests().last().unwrap().1.get("temperature"), None);
        // A picture, which tinygrad's server does not read: this computer's engine, with the model that stands in.
        let picture = r#"{"model": "big", "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}"#;
        assert_eq!(chat(&studio, picture).0, 200);
        assert_eq!(engine.requests().last().unwrap().1.get("model").and_then(Json::as_str), Some("small"));
        // And a plain completion, which tinygrad's server does not answer: likewise.
        assert_eq!(ask(&studio, "completions", r#"{"model": "big", "prompt": "Once"}"#).0, 200);
        let (head, sent) = engine.requests().last().unwrap().clone();
        assert!(head.starts_with("POST /v1/completions "), "{head}");
        assert_eq!(sent.get("model").and_then(Json::as_str), Some("small"));
        assert_eq!(tinygrad.requests().len(), 2, "tinygrad's server was not asked again");
    }

    /// Something that stands for a server's process and stays for a minute.
    fn idler() -> std::process::Child {
        use std::process::{Command, Stdio};
        let mut c = if cfg!(windows) { Command::new("ping") } else { Command::new("sleep") };
        if cfg!(windows) { c.args(["-n", "60", "127.0.0.1"]) } else { c.arg("60") };
        c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap()
    }

    #[test]
    fn a_request_tinygrads_server_gives_no_reply_to_does_not_cost_it_its_model() {
        let root = std::env::temp_dir();
        let studio = egpu_studio(&root, "small");
        // As tinygrad's server treats a request it cannot render: the connection closed, not a word sent. Its
        // process is still there, holding a model that took minutes to load.
        let (tinygrad, engine) = (StandIn::new(""), StandIn::new(ENGINE_SAYS));
        studio.egpu.adopt(&tinygrad.addr, "sk-egpu-test", "big", Some(idler()));
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        let (status, _, body) = chat(&studio, r#"{"model": "big", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("on this computer"), "this one request is answered here: {body}");
        assert_eq!(studio.egpu.state(), crate::llm::State::Ready, "and the server keeps its model");
        assert!(studio.log.tail(10).contains("gave this request no reply"), "{}", studio.log.tail(10));
        // The next chat is tinygrad's again.
        chat(&studio, r#"{"model": "big", "messages": []}"#);
        assert_eq!(tinygrad.requests().len(), 2);
        studio.egpu.stop();
    }

    #[test]
    fn when_tinygrads_server_is_gone_this_computers_engine_answers_in_its_place() {
        let root = std::env::temp_dir();
        for (stands_in, answers) in [("small", "small"), ("", "big")] {
            let studio = egpu_studio(&root, stands_in);
            let engine = StandIn::new(ENGINE_SAYS);
            studio.llm.adopt(&engine.addr, "sk-studio-test");
            // tinygrad's server was there and is not any more (the card unplugged): nothing listens at its address.
            let gone = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); l.local_addr().unwrap().to_string() };
            studio.egpu.adopt(&gone, "sk-egpu-test", "big", None);
            let (status, _, body) = chat(&studio, r#"{"model": "big", "messages": [{"role": "user", "content": "hi"}]}"#);
            assert_eq!(status, 200, "{body}");
            assert!(body.contains("on this computer"), "{body}");
            assert_eq!(engine.requests().last().unwrap().1.get("model").and_then(Json::as_str), Some(answers), "the model that stands in, else the same one");
            assert_eq!(studio.egpu.state(), crate::llm::State::Failed);
            let said = studio.log.tail(10);
            assert!(said.contains("the eGPU does not answer") && said.contains(&format!("{answers} answers on this computer")), "{said}");
            // The next requests go straight to this computer's engine until it is tried again, and the log says it once.
            assert_eq!(chat(&studio, r#"{"model": "big", "messages": []}"#).0, 200);
            assert_eq!(studio.log.tail(10).matches("the eGPU does not answer").count(), 1);
        }
    }

    #[test]
    fn with_the_egpu_switched_off_its_models_are_this_computers() {
        let root = std::env::temp_dir();
        let studio = egpu_studio(&root, "small");
        let mut cfg = studio.config();
        let mut llm = cfg.get("llm").cloned().unwrap();
        crate::util::set(&mut llm, "egpu", Json::parse(br#"{"enabled": false, "fallback_model": "small"}"#).unwrap());
        crate::util::set(&mut cfg, "llm", llm);
        let studio = Arc::new(Studio::for_test(&root, cfg));
        let (tinygrad, engine) = (StandIn::new(TINYGRAD_SAYS), StandIn::new(ENGINE_SAYS));
        studio.egpu.adopt(&tinygrad.addr, "sk-egpu-test", "big", None);
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        assert_eq!(chat(&studio, r#"{"model": "big", "messages": []}"#).0, 200);
        assert!(tinygrad.requests().is_empty());
        assert_eq!(engine.requests().last().unwrap().1.get("model").and_then(Json::as_str), Some("big"), "the model asked for, not a stand-in");
    }

    /// The whole way with the real tinygrad: a Python that has it (OAIY_EGPU_TEST_PYTHON), a small GGUF
    /// (OAIY_EGPU_TEST_MODEL), on the device OAIY_EGPU_TEST_DEV names (CPU unless set). The launcher starts
    /// tinygrad's server; it lists its model only with the key and listens on this computer only; a chat through the
    /// gateway is answered by it, streamed and whole; and let go of, it stops by itself.
    #[test]
    #[ignore = "needs a Python with tinygrad (OAIY_EGPU_TEST_PYTHON) and a small GGUF (OAIY_EGPU_TEST_MODEL)"]
    fn tinygrads_real_server_answers_through_the_gateway_held_to_this_computer_and_a_key() {
        let (Ok(python), Ok(file)) = (std::env::var("OAIY_EGPU_TEST_PYTHON"), std::env::var("OAIY_EGPU_TEST_MODEL")) else { return };
        let device = std::env::var("OAIY_EGPU_TEST_DEV").unwrap_or_else(|_| "CPU".into());
        let dir = std::env::temp_dir().join(format!("oaiy-egpu-real-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Json::obj([
            ("llm", Json::obj([
                ("enabled", Json::Bool(true)),
                ("temperature", Json::Num(0.0)),
                ("max_tokens", Json::Int(16)),
                ("server", Json::str("no-such/oaiy-llm-server")),
                ("server_webgpu", Json::str("no-such/oaiy-llm-server-webgpu")),
                ("egpu", Json::obj([("enabled", Json::Bool(true)), ("python", Json::str(python)), ("device", Json::str(device)), ("ctx", Json::Int(1024))])),
                ("models", Json::Arr(vec![Json::obj([("name", Json::str("m")), ("path", Json::str(file)), ("egpu", Json::Bool(true))])])),
            ])),
            ("media", Json::obj([("llm_policy", Json::str("coexist"))])),
        ]);
        let studio = Arc::new(Studio::for_test(&dir, cfg.clone()));
        let said = studio.egpu.check(&cfg, &dir);
        eprintln!("check: {}", said.to_json());
        assert_eq!(said.get("ok").and_then(Json::as_bool), Some(true), "{}", said.to_json());
        // The first chat starts it and waits for it; streamed, as tinygrad sends it.
        let t = std::time::Instant::now();
        let (status, kind, body) = chat(&studio, r#"{"model": "m", "stream": true, "messages": [{"role": "user", "content": "Say hello."}]}"#);
        eprintln!("first chat after {:.1} s: {status} {kind}\n{}", t.elapsed().as_secs_f64(), body.chars().take(700).collect::<String>());
        assert_eq!(status, 200, "{body}\n{}", studio.egpu.log.tail(40));
        assert_eq!(kind, "text/event-stream");
        assert!(body.contains("chat.completion.chunk") && body.trim_end().ends_with("data: [DONE]"), "{body}");
        let status_now = studio.egpu.status(cfg.get("llm").unwrap());
        eprintln!("status: {}", status_now.to_json());
        assert_eq!(status_now.get("state").and_then(Json::as_str), Some("ready"));
        // Whole, with its counts; and answered straight away, not thought about first, since nothing asked for that
        // (a Qwen model's chat format thinks unless told not to, and tinygrad's server does not tell it).
        let (status, _, body) = chat(&studio, r#"{"model": "m", "messages": [{"role": "user", "content": "Say hello."}]}"#);
        eprintln!("second chat: {status} {body}");
        let reply = Json::parse(body.as_bytes()).unwrap();
        assert_eq!((status, reply.get("model").and_then(Json::as_str)), (200, Some("m")));
        assert!(reply.get("usage").and_then(|u| u.get("completion_tokens")).and_then(Json::as_i64).is_some_and(|n| (1..=16).contains(&n)), "the reply limit OAIY filled in: {body}");
        let message = reply.get("choices").and_then(|c| c.at(0)).and_then(|c| c.get("message")).cloned().unwrap_or(Json::Null);
        assert!(message.get("content").and_then(Json::as_str).is_some_and(|c| !c.trim().is_empty()), "an answer: {body}");
        assert!(message.get("reasoning_content").is_none(), "no thinking unless asked: {body}");
        // Asked to think, it does.
        let (status, _, body) = chat(&studio, r#"{"model": "m", "reasoning_effort": "low", "messages": [{"role": "user", "content": "Say hello."}]}"#);
        eprintln!("third chat, asked to think: {status} {body}");
        let reply = Json::parse(body.as_bytes()).unwrap();
        let message = reply.get("choices").and_then(|c| c.at(0)).and_then(|c| c.get("message")).cloned().unwrap_or(Json::Null);
        assert!(message.get("reasoning_content").and_then(Json::as_str).is_some_and(|c| !c.trim().is_empty()), "thinking when asked: {body}");
        // Itself: nothing without the key, its model with it, and nothing at all on the computer's other address.
        let (endpoint, lease) = studio.egpu.ensure(&cfg, &dir, "m", Duration::from_secs(5)).map_err(|e| format!("{e:?}")).unwrap();
        let models = |auth: &[(&str, &str)]| fetch(&endpoint.addr, "GET", "/v1/models", auth, b"", Duration::from_secs(10)).unwrap().status;
        let auth = format!("Bearer {}", endpoint.key);
        assert_eq!((models(&[]), models(&[("Authorization", "Bearer wrong")]), models(&[("Authorization", &auth)])), (401, 401, 200));
        let port: u16 = endpoint.addr.rsplit(':').next().unwrap().parse().unwrap();
        let elsewhere = std::net::UdpSocket::bind("0.0.0.0:0").and_then(|s| s.connect("192.0.2.1:9").and_then(|()| s.local_addr())).ok().map(|a| a.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified());
        match elsewhere {
            Some(ip) => {
                let refused = std::net::TcpStream::connect_timeout(&std::net::SocketAddr::new(ip, port), Duration::from_secs(3)).is_err();
                eprintln!("on {ip}:{port}: {}", if refused { "refused" } else { "ANSWERED" });
                assert!(refused, "tinygrad's server answers on {ip}");
            }
            None => eprintln!("this computer has no other address to try"),
        }
        if let Ok(pause) = std::env::var("OAIY_EGPU_TEST_PAUSE") {
            // (time for whoever runs this to read what listens on the port from the OS)
            eprintln!("port {port}");
            std::thread::sleep(Duration::from_secs(pause.parse().unwrap_or(5)));
        }
        drop(lease);
        // Let go of without being stopped (as when the studio dies): its standard input closes and it goes.
        let gone = studio.egpu.let_go(Duration::from_secs(15));
        eprintln!("after its lifeline closed: {}", if gone { "exited by itself" } else { "STILL RUNNING" });
        assert!(gone, "tinygrad's server outlived the studio");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_request_that_gets_no_reply_is_an_error_where_no_other_model_stands_in() {
        let root = std::env::temp_dir();
        // (no stand-in named: what would answer here is the same model, loaded a second time for one request)
        let studio = egpu_studio(&root, "");
        let (tinygrad, engine) = (StandIn::new(""), StandIn::new(ENGINE_SAYS));
        studio.egpu.adopt(&tinygrad.addr, "sk-egpu-test", "big", Some(idler()));
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        let (status, _, body) = chat(&studio, r#"{"model": "big", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(status, 502, "{body}");
        assert!(body.contains("gave this request no reply"), "{body}");
        assert!(engine.requests().is_empty(), "this computer's engine was not asked to load it too");
        assert_eq!(studio.egpu.state(), crate::llm::State::Ready, "and the server keeps its model");
        studio.egpu.stop();
    }

    #[test]
    fn a_stream_with_no_end_mark_from_a_server_still_serving_is_finished_as_it_came() {
        // (a tinygrad whose streams carry no end mark: its replies are not to be broken for the lack of one. Only a
        // server that has gone leaves a stream cut short: `a_stream_cut_short_is_not_passed_on_as_a_finished_reply`)
        const UNMARKED: &str = "HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"all of it\"}}]}\n\n";
        let root = std::env::temp_dir();
        let studio = egpu_studio(&root, "small");
        let tinygrad = StandIn::new(UNMARKED);
        studio.egpu.adopt(&tinygrad.addr, "sk-egpu-test", "big", Some(idler()));
        let (status, kind, body) = chat(&studio, r#"{"model": "big", "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!((status, kind.as_str()), (200, "text/event-stream"), "{body}");
        assert!(body.contains("all of it"), "{body}");
        studio.egpu.stop();
    }

    /// tinygrad's LLM server as far as OAIY's side can tell, for [`fake_tinygrad`]: Python's socket server asked to
    /// listen on every interface, HTTP/1.0, its model listed once it has "loaded" (FAKE_LOAD seconds), a chat
    /// streamed with no length and ended by closing, or whole, and no reply at all to a chat with no messages. A
    /// reply says which model file it was started with and what it was told of thinking; a chat that says `cut`
    /// has it die mid-stream, as a card unplugged would.
    const FAKE_TINYGRAD: &str = r#"
import argparse, http.server, json, os, socketserver, time
p = argparse.ArgumentParser()
p.add_argument("--model"); p.add_argument("--serve", type=int); p.add_argument("--max_context", type=int)
a = p.parse_args()
name = os.path.basename(a.model)
time.sleep(float(os.environ.get("FAKE_LOAD", "0.3")))

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def whole(self, code, kind, body):
        self.send_response(code); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/v1/models": self.whole(200, "application/json", json.dumps({"object": "list", "data": [{"id": name}]}).encode())
        else: self.whole(200, "text/html", b"<html>the chat page</html>")
    def do_PUT(self): self.whole(200, "text/plain", b"put")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        messages = body.get("messages") or []
        if not messages: return
        said = "%s thinking=%s" % (name, self.headers.get("X-OAIY-Thinking"))
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": said}}]}) + "\n\n").encode()); self.wfile.flush()
            if messages[-1].get("content") == "cut": os._exit(3)
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            time.sleep(float(os.environ.get("FAKE_REPLY", "0")))
            self.whole(200, "application/json", json.dumps({"model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": said}}]}).encode())

class S(socketserver.TCPServer):
    allow_reuse_address = True

print("loaded model", name, "on FAKE", flush=True)
S(("", a.serve), H).serve_forever()
"#;

    /// A stand-in for tinygrad itself in `dir` ([`FAKE_TINYGRAD`]) and two model files, `x.gguf` and `y.gguf`, and a
    /// studio whose eGPU is set to them: the launcher and the supervisor then run for real. None where this computer
    /// has no Python to run it with.
    fn fake_tinygrad(dir: &std::path::Path, load_seconds: &str, stands_in: &str) -> Option<Arc<Studio>> {
        use std::process::{Command, Stdio};
        let runs = |p: &&str| Command::new(p).arg("--version").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success());
        let python = ["python3", "python"].into_iter().find(runs)?;
        let _ = std::fs::remove_dir_all(dir);
        let package = dir.join("tinygrad").join("llm");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(dir.join("tinygrad").join("__init__.py"), "").unwrap();
        std::fs::write(package.join("__init__.py"), "").unwrap();
        std::fs::write(package.join("__main__.py"), FAKE_TINYGRAD).unwrap();
        let model = |name: &str, egpu: bool| {
            let file = dir.join(format!("{name}.gguf"));
            std::fs::write(&file, b"GGUF").unwrap();
            Json::obj([("name", Json::str(name)), ("path", Json::str(file.to_string_lossy())), ("egpu", Json::Bool(egpu))])
        };
        let cfg = Json::obj([
            ("llm", Json::obj([
                ("enabled", Json::Bool(true)),
                ("temperature", Json::Num(0.0)),
                ("max_tokens", Json::Int(16)),
                ("server", Json::str("no-such/oaiy-llm-server")),
                ("server_webgpu", Json::str("no-such/oaiy-llm-server-webgpu")),
                ("egpu", Json::obj([
                    ("enabled", Json::Bool(true)),
                    ("python", Json::str(python)),
                    ("tinygrad", Json::str(dir.to_string_lossy())),
                    ("device", Json::str("CPU")),
                    ("fallback_model", Json::str(stands_in)),
                    ("env", Json::obj([("FAKE_LOAD", Json::str(load_seconds))])),
                ])),
                ("models", Json::Arr(vec![model("x", true), model("y", true), model("small", false)])),
            ])),
            ("media", Json::obj([("llm_policy", Json::str("coexist"))])),
        ]);
        Some(Arc::new(Studio::for_test(dir, cfg)))
    }

    fn fake_dir(what: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oaiy-egpu-fake-{what}-{}", std::process::id()))
    }

    /// How often the supervisor started a server, by its log.
    fn starts(studio: &Studio) -> usize {
        studio.egpu.log.tail(400).matches("studio: starting").count()
    }

    fn until(what: &str, done: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while !done() {
            assert!(std::time::Instant::now() < deadline, "waited a minute for {what}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// A stand-in for OAIY's engine on the card: it takes a buffer on the card through its socket (as its weights
    /// would), says it has its model, and answers every chat "from the card".
    const FAKE_ENGINE: &str = r#"#!/usr/bin/env python3
import http.server, json, os, socket, struct, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
card = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
card.connect(os.environ["OAIY_WEBGPU_ADAPTER"].split(":", 1)[1])
def ask(cmd, payload=b""):
    card.sendall(struct.pack("<IQ", cmd, len(payload)) + payload)
    head = b""
    while len(head) < 12: head += card.recv(12 - len(head))
    status, n = struct.unpack("<IQ", head)
    body = b""
    while len(body) < n: body += card.recv(n - len(body))
    assert status == 0, body
ask(1)
ask(2, struct.pack("<Q", 1 << 20))
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def whole(self, body):
        body = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self): self.whole({"data": [{"id": "x"}]})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.whole({"choices": [{"index": 0, "message": {"role": "assistant", "content": "from the card"}}]})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

    /// [`fake_tinygrad`]'s studio with OAIY's engine on the card ([`FAKE_ENGINE`]), the card's work on the CPU
    /// (`OAIY_EGPU_EMULATE`): the launcher, the card's server and the supervisor run for real. Only `x` is set to the
    /// eGPU.
    #[cfg(unix)]
    fn fake_card(dir: &std::path::Path) -> Option<Arc<Studio>> {
        use std::os::unix::fs::PermissionsExt;
        let studio = fake_tinygrad(dir, "0", "")?;
        let engine = dir.join("fake-engine");
        std::fs::write(&engine, FAKE_ENGINE).unwrap();
        std::fs::set_permissions(&engine, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut cfg = studio.config();
        let llm = crate::registry::obj_mut(&mut cfg, &["llm"]).unwrap();
        crate::util::set(llm, "server", Json::str(engine.to_string_lossy()));
        let egpu = crate::registry::obj_mut(&mut cfg, &["llm", "egpu"]).unwrap();
        crate::util::set(egpu, "engine", Json::str("webgpu"));
        crate::util::set(egpu, "env", Json::obj([("OAIY_EGPU_EMULATE", Json::str("1"))]));
        if let Some(Json::Arr(models)) = crate::registry::obj_mut(&mut cfg, &["llm", "models"]) {
            for m in models.iter_mut().filter(|m| str_or(m, "name", "") == "y") {
                crate::util::set(m, "egpu", Json::Bool(false));
            }
        }
        Some(Arc::new(Studio::for_test(dir, cfg)))
    }

    /// The card's server, asked who it is at `socket` (as the image worker's adapter first does).
    #[cfg(unix)]
    fn card_hello(socket: &str) -> String {
        use std::io::{Read, Write};
        let mut s = std::os::unix::net::UnixStream::connect(socket).unwrap();
        s.write_all(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let mut head = [0u8; 12];
        s.read_exact(&mut head).unwrap();
        let n = u64::from_le_bytes(head[4..].try_into().unwrap()) as usize;
        let mut body = vec![0u8; n];
        s.read_exact(&mut body).unwrap();
        String::from_utf8_lossy(&body).into_owned()
    }

    #[cfg(unix)]
    #[test]
    fn a_picture_borrows_the_card_its_engine_pauses_and_comes_back_after() {
        let dir = fake_dir("lend");
        let Some(studio) = fake_card(&dir) else { return };
        let ask_x = r#"{"model": "x", "messages": [{"role": "user", "content": "hi"}]}"#;
        let (status, _, body) = chat(&studio, ask_x);
        assert!(status == 200 && body.contains("from the card"), "{status} {body}\n{}", studio.egpu.log.tail(40));

        // Lent: the engine stopped and its buffer freed, the card still held, its socket answering the worker.
        let cfg = studio.config();
        let socket = studio.egpu.lend(&cfg, &studio.root, Duration::from_secs(30)).unwrap_or_else(|e| panic!("{e}\n{}", studio.egpu.log.tail(40)));
        assert!(card_hello(&socket).contains("\"protocol\""));
        let status_now = studio.egpu.status(cfg.get("llm").unwrap());
        assert_eq!((status_now.get("lent"), status_now.get("paused")), (Some(&Json::Bool(true)), Some(&Json::Bool(true))));
        assert!(studio.egpu.log.tail(400).contains("oaiy-egpu: paused"));

        // A chat meanwhile waits for the card, and is answered once it is given back and the engine is there again.
        let asked = {
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || chat(&studio, ask_x))
        };
        std::thread::sleep(Duration::from_millis(1500));
        assert!(!asked.is_finished(), "a chat is not answered while the card is lent");
        studio.egpu.give_back();
        let (status, _, body) = asked.join().unwrap();
        assert!(status == 200 && body.contains("from the card"), "{status} {body}");
        assert!(studio.egpu.log.tail(400).contains("oaiy-egpu: resumed"));
        assert_eq!(starts(&studio), 1, "the card was opened once");

        // Not running: lent all the same, the card opened with its engine paused, which a chat for its model brings.
        studio.egpu.stop();
        until("the launcher to go", || studio.egpu.log.tail(400).contains("studio: tinygrad's server stopped"));
        let socket = studio.egpu.lend(&cfg, &studio.root, Duration::from_secs(30)).unwrap_or_else(|e| panic!("{e}\n{}", studio.egpu.log.tail(40)));
        assert!(card_hello(&socket).contains("\"protocol\""));
        studio.egpu.give_back();
        let status_now = studio.egpu.status(cfg.get("llm").unwrap());
        assert_eq!(status_now.get("paused"), Some(&Json::Bool(true)), "not brought back for nothing");
        let (status, _, body) = chat(&studio, ask_x);
        assert!(status == 200 && body.contains("from the card"), "{status} {body}");
        assert_eq!(starts(&studio), 2);
        studio.egpu.stop();
        if std::env::var("SHOW_EGPU_LOG").is_ok() {
            eprintln!("{}", studio.egpu.log.tail(400));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_models_asked_for_at_once_are_each_answered_and_neither_load_is_stopped_for_the_other() {
        let dir = fake_dir("two");
        let Some(studio) = fake_tinygrad(&dir, "1.5", "") else { return };
        let engine = StandIn::new(ENGINE_SAYS);
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        // Both at once: one loads and answers, then the other, each started once. (Counted as waiting for the
        // model that is loading, the first's request keeps it; before, each stopped the other's load in turn.)
        let asked = ["x", "y"].map(|m| {
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || chat(&studio, &format!(r#"{{"model": "{m}", "messages": [{{"role": "user", "content": "hi"}}]}}"#)))
        });
        for (m, reply) in ["x", "y"].into_iter().zip(asked) {
            let (status, _, body) = reply.join().unwrap();
            assert_eq!(status, 200, "{m}: {body}\n{}", studio.egpu.log.tail(40));
            assert!(body.contains(&format!("{m}.gguf thinking=0")), "{m} answered by its own model, not asked to think: {body}");
        }
        assert_eq!(starts(&studio), 2, "each model started once:\n{}", studio.egpu.log.tail(40));
        assert!(engine.requests().is_empty(), "this computer's engine was not asked");
        // Itself: nothing without the key, whatever the method, and nothing at all on the computer's other address.
        let cfg = studio.config();
        let held = studio.egpu.held().unwrap();
        let (endpoint, lease) = studio.egpu.ensure(&cfg, &dir, &held, Duration::from_secs(30)).map_err(|e| format!("{e:?}")).unwrap();
        let auth = format!("Bearer {}", endpoint.key);
        let status = |method: &str, path: &str, headers: &[(&str, &str)]| fetch(&endpoint.addr, method, path, headers, b"{}", Duration::from_secs(10)).unwrap().status;
        assert_eq!((status("GET", "/v1/models", &[]), status("GET", "/", &[]), status("PUT", "/x", &[]), status("POST", "/v1/chat/completions", &[("Authorization", "Bearer wrong")])), (401, 401, 401, 401));
        assert_eq!((status("GET", "/v1/models", &[("Authorization", &auth)]), status("PUT", "/x", &[("Authorization", &auth)])), (200, 200));
        let port: u16 = endpoint.addr.rsplit(':').next().unwrap().parse().unwrap();
        let elsewhere = std::net::UdpSocket::bind("0.0.0.0:0").and_then(|s| s.connect("192.0.2.1:9").and_then(|()| s.local_addr())).ok().map(|a| a.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified());
        if let Some(ip) = elsewhere {
            assert!(std::net::TcpStream::connect_timeout(&std::net::SocketAddr::new(ip, port), Duration::from_secs(3)).is_err(), "it answers on {ip}");
        }
        // A start by hand of the model it has changes nothing; of the other, while this one is being used, is
        // refused and cuts nothing off.
        let other = if held == "x" { "y" } else { "x" };
        assert_eq!(studio.egpu.start(&cfg, &dir, &held), Ok(()));
        let refused = studio.egpu.start(&cfg, &dir, other).unwrap_err();
        assert!(refused.contains(&format!("answering with {held}")), "{refused}");
        assert_eq!((starts(&studio), studio.egpu.state()), (2, crate::llm::State::Ready));
        drop(lease);
        // Let go of without being stopped (as when the studio dies): its standard input closes and it goes.
        assert!(studio.egpu.let_go(Duration::from_secs(15)), "the server outlived the studio");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stop_while_the_server_is_starting_holds_and_the_waiting_request_is_answered_here() {
        let dir = fake_dir("stop");
        let Some(studio) = fake_tinygrad(&dir, "4", "small") else { return };
        let engine = StandIn::new(ENGINE_SAYS);
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        let asking = {
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || chat(&studio, r#"{"model": "x", "messages": [{"role": "user", "content": "hi"}]}"#))
        };
        until("the server to be starting", || studio.egpu.state() == crate::llm::State::Starting);
        studio.egpu.stop();
        // The request that was waiting for it is answered by this computer's engine, with the model that stands
        // in, and does not start the server again.
        let (status, _, body) = asking.join().unwrap();
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("on this computer"), "{body}");
        assert_eq!(engine.requests().last().unwrap().1.get("model").and_then(Json::as_str), Some("small"));
        std::thread::sleep(Duration::from_secs(1));
        assert_eq!((starts(&studio), studio.egpu.state()), (1, crate::llm::State::Stopped), "{}", studio.egpu.log.tail(20));
        // The next request starts it afresh.
        let (status, _, body) = chat(&studio, r#"{"model": "x", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(status, 200, "{body}\n{}", studio.egpu.log.tail(20));
        assert!(body.contains("x.gguf"), "{body}");
        studio.egpu.stop();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stream_cut_short_is_not_passed_on_as_a_finished_reply() {
        let dir = fake_dir("cut");
        let Some(studio) = fake_tinygrad(&dir, "0.2", "small") else { return };
        let engine = StandIn::new(ENGINE_SAYS);
        studio.llm.adopt(&engine.addr, "sk-studio-test");
        // A stream that runs to its end arrives whole.
        let (status, kind, body) = chat(&studio, r#"{"model": "x", "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!((status, kind.as_str()), (200, "text/event-stream"), "{body}\n{}", studio.egpu.log.tail(20));
        assert!(body.contains("x.gguf") && body.trim_end().ends_with("data: [DONE]"), "{body}");
        // One whose server dies mid-reply (the card unplugged): the client's connection breaks; it is not handed
        // the first words as a whole reply.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let served = Arc::clone(&studio);
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            oaiy_engine::http::serve(stream, |req, w| handle(&served, req, w, Matched { target: "chat".into(), spec: "openai".into(), rest: String::new() }, true));
        });
        let reply = fetch(&addr, "POST", "/v1/chat/completions", &[("Content-Type", "application/json")], br#"{"model": "x", "stream": true, "messages": [{"role": "user", "content": "cut"}]}"#, Duration::from_secs(60)).unwrap();
        assert_eq!(reply.status, 200);
        let read = reply.body(8 << 20);
        assert!(read.is_err(), "a broken stream, not a reply: {:?}", read.map(|b| String::from_utf8_lossy(&b).into_owned()));
        // Its process has gone: the next chats are this computer's engine's.
        until("the server to be seen gone", || studio.egpu.state() == crate::llm::State::Failed);
        let (status, _, body) = chat(&studio, r#"{"model": "x", "messages": [{"role": "user", "content": "hi"}]}"#);
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("on this computer"), "{body}");
        studio.egpu.stop();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn byte_ranges_follow_rfc_7233() {
        assert_eq!(byte_range("bytes=0-99", 1000), Range::Part(0, 99));
        assert_eq!(byte_range("bytes=500-", 1000), Range::Part(500, 999));
        assert_eq!(byte_range("bytes=-100", 1000), Range::Part(900, 999));
        assert_eq!(byte_range("bytes=-5000", 1000), Range::Part(0, 999));
        assert_eq!(byte_range("bytes=990-2000", 1000), Range::Part(990, 999));
        assert_eq!(byte_range("bytes=1000-", 1000), Range::Unsatisfiable);
        assert_eq!(byte_range("bytes=-0", 1000), Range::Unsatisfiable);
        // Several ranges, other units and nonsense are ignored: the whole file.
        assert_eq!(byte_range("bytes=0-1,5-9", 1000), Range::Whole);
        assert_eq!(byte_range("items=0-1", 1000), Range::Whole);
        assert_eq!(byte_range("bytes=9-3", 1000), Range::Whole);
        assert_eq!(byte_range("bytes=x-", 1000), Range::Whole);
    }

    #[test]
    fn jobs_render_as_openai_video_objects() {
        let m = crate::media::Media::new();
        let job = m.submit(Kind::Video, Json::obj([("prompt", Json::str("waves"))]), "ltx".into(), "1024x576".into(), 1, 4.0, 10, None);
        let v = video_object(&job);
        assert_eq!(v.get("object").and_then(Json::as_str), Some("video"));
        assert_eq!(v.get("status").and_then(Json::as_str), Some("queued"));
        assert_eq!(v.get("seconds").and_then(Json::as_str), Some("4"));
        assert_eq!(v.get("error"), Some(&Json::Null));
        m.cancel(&job.id);
        let v = video_object(&m.get(&job.id).unwrap());
        assert_eq!(v.get("status").and_then(Json::as_str), Some("failed"));
        assert_eq!(v.get("error").unwrap().get("code").and_then(Json::as_str), Some("cancelled"));
    }

    #[test]
    fn jobs_render_as_3d_model_objects() {
        let m = crate::media::Media::new();
        let request = Json::obj([("kind", Json::str("model3d")), ("seed", Json::Int(7)), ("resolution", Json::Int(1536))]);
        let job = m.submit(Kind::Model3d, request, "pixal3d".into(), "1536, 200000 faces".into(), 1, 0.0, 10, None);
        assert!(job.id.starts_with("m3d_"));
        let v = model3d_object(&job);
        assert_eq!(v.get("object").and_then(Json::as_str), Some("model3d"));
        assert_eq!(v.get("status").and_then(Json::as_str), Some("queued"));
        assert_eq!(v.get("format").and_then(Json::as_str), Some("glb"));
        assert_eq!((v.get("seed").and_then(Json::as_i64), v.get("resolution").and_then(Json::as_i64)), (Some(7), Some(1536)));
        assert_eq!(v.get("error"), Some(&Json::Null));
        m.cancel(&job.id);
        let v = model3d_object(&m.get(&job.id).unwrap());
        assert_eq!(v.get("status").and_then(Json::as_str), Some("failed"));
        assert_eq!(v.get("error").unwrap().get("code").and_then(Json::as_str), Some("cancelled"));
        assert_eq!(content_type(std::path::Path::new("a/model.glb")), "model/gltf-binary");
    }
}
