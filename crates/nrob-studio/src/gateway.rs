//! The public API: a route table from the configuration, each route a path, a
//! method, a target (what serves it) and a spec (the dialect it speaks).
//!
//! * `chat`, `completions`, `models`: proxied to nrob-server (OpenAI chat,
//!   streamed as it arrives -- event streams are never buffered).
//! * `images` + `openai`: OpenAI Images, synchronous: `{created, data:[{b64_json}|{url}]}`.
//! * `edits` + `openai`: OpenAI image edits, as `multipart/form-data` (what the
//!   SDKs send) or JSON with `images: [{image_url}]`; same reply as `images`.
//! * `videos` + `openai`: OpenAI Videos, asynchronous: `POST P` creates,
//!   `GET P` lists, `GET P/{id}` polls, `GET P/{id}/content` downloads,
//!   `DELETE P/{id}` forgets.
//! * `images`/`videos` + `nrob`: nrob-server's job API (`202 {id, status_url}`,
//!   `GET P/status`, `POST P/cancel`), for clients such as coder-cli.
//! * `files`: generated media under the output root.

use crate::media::{self, Job, Kind};
use crate::util::{base64_encode, bool_or, error_json, int_or, now, str_or};
use crate::Studio;
use nrob::http::{fetch, respond, respond_with, Request, Stream};
use nrob::json::Json;
use std::io;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// How long a chat request waits for a media job holding the LLM's GPU.
const CHAT_WAIT: Duration = Duration::from_secs(15 * 60);
/// How long a request waits for the LLM to load.
const LOAD_WAIT: Duration = Duration::from_secs(10 * 60);
/// Longest silence from the LLM (nrob-server sends keepalives every 10 s while streaming).
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
        let prefix_owner = matches!(target, "images" | "videos" | "files" | "voices" | "music");
        let rest = if path == base {
            ""
        } else if prefix_owner && path.starts_with(base) && path[base.len()..].starts_with('/') {
            &path[base.len()..]
        } else {
            continue;
        };
        // The configured method is the primary route's; sub-routes set their own.
        if rest.is_empty() && !matches!(target, "videos" | "files" | "voices" | "music") && !str_or(r, "method", "POST").eq_ignore_ascii_case(method) {
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
        ("images", "nrob") | ("videos", "nrob") => {
            let kind = if m.target == "images" { Kind::Image } else { Kind::Video };
            send(w, nrob_jobs(studio, req, kind, &m.rest, trusted))
        }
        ("videos", "openai") => videos_openai(studio, req, w, method, &m.rest, trusted),
        ("speech", _) if m.rest.is_empty() => speech_openai(studio, req, w),
        ("voices", _) => voices(studio, req, w, &m.rest),
        ("music", _) => music_openai(studio, req, w, method, &m.rest),
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

/// Forward to nrob-server, starting it if needed, and stream the reply back.
fn proxy(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, upstream: &str) -> io::Result<bool> {
    let cfg = studio.config();
    if !cfg.get("llm").is_some_and(|l| bool_or(l, "enabled", true)) {
        return send(w, Err(fail(503, "the language model is disabled")));
    }
    let lease = match studio.media.chat_lease(media::pauses_llm(&cfg), CHAT_WAIT) {
        Ok(l) => l,
        Err(e) => return send(w, Err(fail(503, e))),
    };
    let endpoint = match studio.llm.ensure_ready(&cfg, &studio.root, LOAD_WAIT) {
        Ok(e) => e,
        Err(e) => return send(w, Err(fail(503, e))),
    };
    studio.llm.touch();
    let auth = format!("Bearer {}", endpoint.key);
    let ctype = req.header("content-type").unwrap_or("application/json").to_string();
    let mut headers = vec![("Authorization", auth.as_str()), ("Content-Type", ctype.as_str())];
    if let Some(a) = req.header("accept") {
        headers.push(("Accept", a));
    }
    if incognito_mode(studio) || incognito_header(req) {
        headers.push(("X-NROB-Incognito", "1"));
    }
    let response = match fetch(&endpoint.addr, &req.method, upstream, &headers, &req.body, UPSTREAM_READ) {
        Ok(r) => r,
        Err(e) => return send(w, Err(fail(502, format!("the language model did not answer: {e}")))),
    };
    let status = response.status;
    let rtype = response.header("content-type").unwrap_or("application/json").to_string();
    let streamed = rtype.contains("event-stream") || response.header("transfer-encoding").is_some_and(|t| t.contains("chunked"));
    let result = if streamed {
        let mut out = Stream::start_status(w, status, &rtype)?;
        // A client that leaves makes `send` fail, which drops the upstream
        // connection, which nrob-server takes as a cancel.
        let r = response.for_each_chunk(|chunk| out.send(chunk));
        if r.is_ok() {
            out.finish()?;
        }
        // The stream said keep-alive and ended cleanly: the client may send its
        // next request on this connection, so keep reading it rather than close
        // it under a request already on its way.
        r.map(|_| true)
    } else {
        let body = response.body(64 << 20)?;
        respond(w, status, &rtype, &body, true).map(|_| true)
    };
    studio.llm.touch();
    drop(lease);
    result
}

/// OpenAI model list: nrob-server's own when it runs, the configured names otherwise,
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
    for kind in ["image", "video", "speech", "music"] {
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
        ("owned_by", Json::str("nrob-studio")),
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
    nrob::http::respond_head(w, status, content_type(path), &extra, count, true)?;
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

/// `X-NROB-Incognito: 1` (or `true`, `yes`).
fn incognito_header(req: &Request) -> bool {
    req.header("x-nrob-incognito").is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Whether the studio runs in incognito mode (`privacy.incognito`).
pub fn incognito_mode(studio: &Studio) -> bool {
    studio.config().get("privacy").is_some_and(|p| bool_or(p, "incognito", false))
}

/// For an incognito request (global mode, `X-NROB-Incognito: 1`, or
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

/// Saved voices: list, design and save, show, sample clip, delete.
fn voices(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, rest: &str) -> io::Result<bool> {
    let cfg = studio.config();
    let dir = crate::speech::voices_dir(&cfg, &studio.root);
    let parts: Vec<String> = rest.split('/').filter(|s| !s.is_empty()).map(nrob::http::percent_decode).collect();
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

/// nrob-server's asynchronous job API.
fn nrob_jobs(studio: &Arc<Studio>, req: &Request, kind: Kind, rest: &str, trusted: bool) -> Result<Reply, Reply> {
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
        let custom = vec![Json::parse(br#"{"path":"/api/gen","method":"POST","target":"images","spec":"nrob"}"#).unwrap()];
        let m = route(&custom, "GET", "/api/gen/status").unwrap();
        assert_eq!((m.spec.as_str(), m.rest.as_str()), ("nrob", "/status"));
        let disabled = vec![Json::parse(br#"{"path":"/x","method":"POST","target":"chat","enabled":false}"#).unwrap()];
        assert!(route(&disabled, "POST", "/x").is_none());
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
}
