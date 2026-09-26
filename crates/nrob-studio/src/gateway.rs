//! The public API: a route table from the configuration, each route a path, a
//! method, a target (what serves it) and a spec (the dialect it speaks).
//!
//! * `chat`, `completions`, `models`: proxied to nrob-server (OpenAI chat,
//!   streamed as it arrives -- event streams are never buffered).
//! * `images` + `openai`: OpenAI Images, synchronous: `{created, data:[{b64_json}|{url}]}`.
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
        let prefix_owner = matches!(target, "images" | "videos" | "files");
        let rest = if path == base {
            ""
        } else if prefix_owner && path.starts_with(base) && path[base.len()..].starts_with('/') {
            &path[base.len()..]
        } else {
            continue;
        };
        // The configured method is the primary route's; sub-routes set their own.
        if rest.is_empty() && target != "videos" && target != "files" && !str_or(r, "method", "POST").eq_ignore_ascii_case(method) {
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
        ("files", _) => files(studio, w, m.rest.trim_start_matches('/')),
        ("images", "openai") if m.rest.is_empty() => send(w, images_openai(studio, req, trusted)),
        ("images", "nrob") | ("videos", "nrob") => {
            let kind = if m.target == "images" { Kind::Image } else { Kind::Video };
            send(w, nrob_jobs(studio, req, kind, &m.rest, trusted))
        }
        ("videos", "openai") => videos_openai(studio, req, w, method, &m.rest, trusted),
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
        r.map(|_| false)
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
    for kind in ["image", "video"] {
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
        Some("json" | "jsonl") => "application/json",
        _ => "application/octet-stream",
    }
}

pub fn serve_file(w: &mut TcpStream, path: &std::path::Path, download: bool) -> io::Result<bool> {
    let bytes = std::fs::read(path)?;
    let name = path.file_name().map(|n| n.to_string_lossy().replace('"', "")).unwrap_or_default();
    let disposition = format!("{}; filename=\"{name}\"", if download { "attachment" } else { "inline" });
    respond_with(w, 200, content_type(path), &[("Content-Disposition", &disposition), ("Cache-Control", "max-age=3600")], &bytes, true)?;
    Ok(true)
}

pub fn files(studio: &Studio, w: &mut TcpStream, rel: &str) -> io::Result<bool> {
    match media::resolve_file(&studio.output_root(), rel) {
        Some(p) => serve_file(w, &p, false),
        None => send(w, Err(fail(404, "no such file"))),
    }
}

/// OpenAI Images: generate, wait, and answer with base64 PNGs or URLs.
fn images_openai(studio: &Arc<Studio>, req: &Request, trusted: bool) -> Result<Reply, Reply> {
    if req.method != "POST" {
        return Err(fail(405, "use POST"));
    }
    let body = parse_body(req)?;
    let format = match body.get("response_format").and_then(Json::as_str) {
        None => "b64_json",
        Some(f @ ("b64_json" | "url")) => f,
        Some(_) => return Err(fail(400, "response_format must be b64_json or url")),
    };
    if body.get("output_format").and_then(Json::as_str).is_some_and(|f| f != "png") {
        return Err(fail(400, "output_format: this server writes png"));
    }
    let job = submit_image(studio, &body)?;
    let _ = trusted;
    let done = studio.media.wait(&job.id, IMAGE_WAIT).ok_or_else(|| fail(500, "the job disappeared"))?;
    if done.status != "completed" {
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
    ok(Json::obj([
        ("created", Json::Int(now() as i64)),
        ("data", Json::Arr(data)),
        ("output_format", Json::str("png")),
        ("size", Json::str(&done.size)),
        ("model", Json::str(&done.model)),
    ]))
}

fn submit_image(studio: &Arc<Studio>, body: &Json) -> Result<Job, Reply> {
    let cfg = studio.config();
    let root = studio.output_root();
    let (request, model, size, n) = media::image_request(&cfg, &studio.root, &root, body).map_err(|e| fail(400, e))?;
    Ok(studio.media.submit(Kind::Image, request, model, size, n, 0.0, keep_jobs(&cfg)))
}

fn submit_video(studio: &Arc<Studio>, body: &Json, trusted: bool) -> Result<Job, Reply> {
    let cfg = studio.config();
    let root = studio.output_root();
    let (request, model, size, seconds) = media::video_request(&cfg, &studio.root, &root, body, trusted).map_err(|e| fail(400, e))?;
    Ok(studio.media.submit(Kind::Video, request, model, size, 1, seconds, keep_jobs(&cfg)))
}

fn keep_jobs(cfg: &Json) -> usize {
    cfg.get("media").map_or(200, |m| int_or(m, "keep_jobs", 200)).clamp(1, 100_000) as usize
}

/// Where clients reach this gateway: `gateway.public_url`, else the Host they used.
fn public_base(studio: &Studio, req: &Request) -> String {
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
    let seconds = if job.seconds.fract().abs() < 0.05 { format!("{}", job.seconds.round() as i64) } else { format!("{:.1}", job.seconds) };
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
        ("POST", []) => send(w, parse_body(req).and_then(|b| submit_video(studio, &b, trusted)).map(|j| Reply { status: 200, body: video_object(&j) })),
        ("GET", []) => {
            let limit = req.query("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
            let data: Vec<Json> = studio.media.list().iter().filter(|j| j.kind == Kind::Video).take(limit).map(video_object).collect();
            let id = |i: Option<&Json>| i.and_then(|v| v.get("id")).cloned().unwrap_or(Json::Null);
            json_reply(w, 200, &Json::obj([
                ("object", Json::str("list")),
                ("first_id", id(data.first())),
                ("last_id", id(data.last())),
                ("has_more", Json::Bool(false)),
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
                Some(f) => serve_file(w, &f, true),
                None => send(w, Err(fail(404, "the file is gone"))),
            }
        }
        _ => send(w, Err(fail(405, format!("{method} is not supported here")))),
    }
}

/// nrob-server's asynchronous job API.
fn nrob_jobs(studio: &Arc<Studio>, req: &Request, kind: Kind, rest: &str, trusted: bool) -> Result<Reply, Reply> {
    let latest = || {
        let id = req.query("id");
        studio.media.list().into_iter().find(|j| j.kind == kind && id.as_deref().is_none_or(|i| i == j.id))
    };
    match (req.method.as_str(), rest) {
        ("POST", "") => {
            let body = parse_body(req)?;
            let job = if kind == Kind::Image { submit_image(studio, &body)? } else { submit_video(studio, &body, trusted)? };
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
            let job = studio.media.list().into_iter().find(|j| j.kind == kind && !j.finished());
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
    fn jobs_render_as_openai_video_objects() {
        let m = crate::media::Media::new();
        let job = m.submit(Kind::Video, Json::obj([("prompt", Json::str("waves"))]), "ltx".into(), "1024x576".into(), 1, 4.0, 10);
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
