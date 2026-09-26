//! Image and video jobs: validated into worker requests when submitted, run one
//! at a time by `nrob-diffusion` (a subprocess, so its exit returns every byte of
//! VRAM), with progress read from the worker's event lines.
//!
//! Weight residency is the worker's: each job carries `memory`
//! (auto|gpu|ram|ssd), `ram_gb` and `vram_gb`, and the worker places every block
//! of the model on the GPU, in RAM or on the SSD accordingly. The studio decides
//! only what a job may use and whether the LLM must leave the GPU first.

use crate::config::{self, VIDEO_FAMILIES};
use crate::util::{base64_decode, bool_or, int_or, now, num_or, random_id, str_or, LogRing};
use nrob::json::Json;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Image,
    Video,
    /// Text to speech, and designing a voice to save.
    Speech,
    /// Songs, and making the music model's smaller copy.
    Music,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Video => "video",
            Kind::Speech => "speech",
            Kind::Music => "music",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Job {
    pub id: String,
    pub kind: Kind,
    /// queued | in_progress | completed | failed | cancelled
    pub status: String,
    pub progress: f64,
    pub stage: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub prompt: String,
    pub model: String,
    pub size: String,
    /// Clip length in seconds, for videos.
    pub seconds: f64,
    pub n: usize,
    pub request: Json,
    pub result: Json,
    pub files: Vec<PathBuf>,
    pub preview: Option<PathBuf>,
    pub error: Option<String>,
    cancel: Arc<AtomicBool>,
    /// Incognito: hidden from lists and logs; its folder is deleted after
    /// delivery or at `expires_at`.
    pub incognito: bool,
    /// The job's private folder (incognito only), removed with the job.
    pub scratch: Option<PathBuf>,
    pub expires_at: Option<u64>,
    /// Deleted while running: forget it as soon as it stops.
    forget: bool,
}

impl Job {
    pub fn finished(&self) -> bool {
        matches!(self.status.as_str(), "completed" | "failed" | "cancelled")
    }

    pub fn to_json(&self, output_root: &Path) -> Json {
        let urls = |files: &[PathBuf]| Json::Arr(files.iter().filter_map(|f| file_url(output_root, f)).map(Json::str).collect());
        Json::obj([
            ("id", Json::str(&self.id)),
            ("kind", Json::str(self.kind.name())),
            ("status", Json::str(&self.status)),
            ("progress", Json::Num((self.progress * 10.0).round() / 10.0)),
            ("stage", Json::str(&self.stage)),
            ("created_at", Json::Int(self.created_at as i64)),
            ("completed_at", self.completed_at.map_or(Json::Null, |t| Json::Int(t as i64))),
            ("prompt", Json::str(&self.prompt)),
            ("model", Json::str(&self.model)),
            ("size", Json::str(&self.size)),
            ("n", Json::Int(self.n as i64)),
            ("files", urls(&self.files)),
            ("preview", self.preview.as_ref().and_then(|p| file_url(output_root, p)).map_or(Json::Null, Json::str)),
            ("error", self.error.as_ref().map_or(Json::Null, Json::str)),
            ("residency", self.result.get("residency").cloned().unwrap_or(Json::Null)),
            ("seconds_taken", self.result.get("seconds").cloned().unwrap_or(Json::Null)),
            ("seconds", match self.kind {
                Kind::Video => Json::Num(self.seconds),
                Kind::Speech | Kind::Music => self.result.get("duration").cloned().unwrap_or(Json::Null),
                Kind::Image => Json::Null,
            }),
            // What a viewer needs to reproduce it; never paths or references.
            ("settings", Json::obj(["seed", "steps", "memory", "cfg", "negative_prompt", "fps", "frames", "lyrics"].map(|k| (k, self.request.get(k).cloned().unwrap_or(Json::Null))))),
        ])
    }
}

/// `/files/...` for a file under the output root (forward slashes, no escape).
pub fn file_url(root: &Path, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(root).ok()?;
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    if parts.iter().any(|p| p == ".." || p.contains(['?', '#'])) {
        return None;
    }
    Some(format!("/files/{}", parts.join("/")))
}

/// The file a `/files/...` path names, if it stays under `root` (links included).
pub fn resolve_file(root: &Path, rel: &str) -> Option<PathBuf> {
    let rel = nrob::http::percent_decode(rel);
    // Plain names separated by `/` only. A backslash, drive or UNC prefix would
    // make `join` replace the root (on Windows `\\\\host\\share` then opens over
    // SMB and hands the machine's NTLM hash to that host) before any check runs.
    let parts: Vec<&str> = rel.split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() || parts.iter().any(|p| matches!(*p, "." | "..") || p.contains(['\\', ':', '\0'])) {
        return None;
    }
    let root = root.canonicalize().ok()?;
    let mut joined = root.clone();
    for p in &parts {
        joined.push(p);
    }
    let path = joined.canonicalize().ok()?;
    (path.starts_with(&root) && path.is_file()).then_some(path)
}

/// How media shares the GPU with chat: while a media job holds a device the LLM
/// also uses, the LLM is stopped, so new chat requests wait (bounded) and a job
/// waits for chats already running to end.
struct Broker {
    chats: usize,
    media: bool,
    waiting_media: bool,
}

pub struct Media {
    jobs: Mutex<VecDeque<Job>>,
    changed: Condvar,
    broker: Mutex<Broker>,
    broker_changed: Condvar,
    queue_signal: Condvar,
    pub log: Arc<LogRing>,
}

/// A chat request's hold on the LLM; released on drop.
pub struct ChatLease<'a>(&'a Media);

impl Drop for ChatLease<'_> {
    fn drop(&mut self) {
        let mut b = self.0.broker.lock().unwrap_or_else(|p| p.into_inner());
        b.chats -= 1;
        self.0.broker_changed.notify_all();
    }
}

fn gib_field(request: Option<&Json>, section: &Json, model: &Json, key: &str, max: i64) -> Result<Option<i64>, String> {
    // The section value is the cap; the model may lower it; a request may ask for less.
    let cap = section.get(key).and_then(Json::as_i64).map(|n| n.clamp(0, max));
    let base = model.get(key).and_then(Json::as_i64).map(|n| n.clamp(0, cap.unwrap_or(max))).or(cap);
    match request.and_then(|r| r.get(key)) {
        None | Some(Json::Null) => Ok(base),
        Some(v) => {
            let n = v.as_i64().ok_or_else(|| format!("{key} must be an integer"))?;
            let limit = cap.unwrap_or(max);
            if !(0..=limit).contains(&n) {
                return Err(format!("{key} must be between 0 and {limit}"));
            }
            Ok(Some(n))
        }
    }
}

/// `memory`, `ram_gb`, `vram_gb` for the worker: request, then model, then section.
pub(crate) fn residency(body: &Json, section: &Json, model: &Json) -> Result<Vec<(String, Json)>, String> {
    let memory = body.get("memory").or_else(|| model.get("memory")).or_else(|| section.get("memory"));
    let memory = match memory {
        None | Some(Json::Null) => "auto",
        Some(v) => v.as_str().ok_or("memory must be a string")?,
    };
    if !config::MEMORY.contains(&memory) {
        return Err("memory must be auto, gpu, ram or ssd".into());
    }
    let mut out = vec![("memory".to_string(), Json::str(memory))];
    if let Some(n) = gib_field(Some(body), section, model, "ram_gb", 512)? {
        out.push(("ram_gb".into(), Json::Int(n)));
    }
    if let Some(n) = gib_field(Some(body), section, model, "vram_gb", 192)? {
        out.push(("vram_gb".into(), Json::Int(n)));
    }
    Ok(out)
}

pub(crate) fn pick_model<'a>(section: &'a Json, body: &Json, kind: &str) -> Result<(String, &'a Json), String> {
    let models = section.get("models").filter(|m| m.as_object().is_some_and(|o| !o.is_empty()))
        .ok_or_else(|| format!("no {kind} model is configured; add one under Models"))?;
    let asked = body.get("model").and_then(Json::as_str).filter(|s| !s.is_empty());
    let name = asked.or_else(|| section.get("default_model").and_then(Json::as_str).filter(|s| !s.is_empty()));
    let found = match name {
        Some(n) => models.members().find(|(k, _)| *k == n),
        None => models.members().next(),
    };
    // An OpenAI client names OpenAI's models ("dall-e-3", "sora-2"): use the default.
    let found = found.or_else(|| {
        asked.filter(|a| a.starts_with("dall-e") || a.starts_with("gpt-image") || a.starts_with("sora") || a.starts_with("tts-") || a.ends_with("-tts"))
            .and_then(|_| section.get("default_model").and_then(Json::as_str).and_then(|d| models.members().find(|(k, _)| *k == d)).or_else(|| models.members().next()))
    });
    let (name, model) = found.ok_or_else(|| format!("unknown {kind} model {:?}", name.unwrap_or("")))?;
    if !bool_or(model, "enabled", true) {
        return Err(format!("{kind} model {name} is disabled"));
    }
    Ok((name.to_string(), model))
}

/// `WxH` (or `auto`) into a size, or the model's default.
fn size(body: &Json, model: &Json, default: (i64, i64)) -> Result<(i64, i64), String> {
    let (mut w, mut h) = (int_or(model, "width", default.0), int_or(model, "height", default.1));
    match body.get("size").and_then(Json::as_str) {
        None | Some("auto") | Some("") => {}
        Some(s) => {
            let (a, b) = s.split_once(['x', 'X']).ok_or("size must look like 1024x1024")?;
            w = a.trim().parse().map_err(|_| "size width is not a number")?;
            h = b.trim().parse().map_err(|_| "size height is not a number")?;
        }
    }
    if let Some(v) = body.get("width").and_then(Json::as_i64) {
        w = v;
    }
    if let Some(v) = body.get("height").and_then(Json::as_i64) {
        h = v;
    }
    Ok((w, h))
}

/// Scale into `max` on the longer side (keeping the aspect), then round to `step`.
pub fn fit(w: i64, h: i64, max: i64, min: i64, step: i64) -> (i64, i64) {
    let scale = (max as f64 / w.max(h) as f64).min(1.0);
    let round = |v: i64| (((v as f64 * scale) / step as f64).round() as i64 * step).clamp(min, max);
    (round(w), round(h))
}

fn seed(body: &Json) -> Result<i64, String> {
    match body.get("seed") {
        None | Some(Json::Null) => Ok((crate::util::now_millis() % 2_147_483_647) as i64),
        Some(v) => v.as_i64().filter(|n| (0..i64::MAX / 2).contains(n)).ok_or_else(|| "seed must be a nonnegative integer".into()),
    }
}

fn prompt(body: &Json) -> Result<String, String> {
    body.get("prompt").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty() && s.len() <= 16384)
        .map(str::to_owned).ok_or_else(|| "prompt must be a nonempty string of at most 16384 bytes".into())
}

/// A day folder under the output root: `images/2026-09-26`.
fn output_dir(root: &Path, kind: &str) -> PathBuf {
    day_dir(root, &format!("{kind}s"))
}

/// `folder/2026-09-26` under the output root.
pub(crate) fn day_dir(root: &Path, folder: &str) -> PathBuf {
    let days = now() / 86_400;
    // Civil date from days since 1970 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    root.join(folder).join(format!("{y:04}-{m:02}-{d:02}"))
}

fn path_field(root: &Path, model: &Json, key: &str) -> Option<String> {
    let v = str_or(model, key, "").trim();
    (!v.is_empty()).then(|| config::resolve(root, v).to_string_lossy().into_owned())
}

/// Reference images for editing: `images` (an array) or `image` (one or an
/// array), each a `data:` URL, `{image_url}`, or (from the local UI) a path.
fn references(body: &Json, output_root: &Path, allow_local: bool) -> Result<Vec<String>, String> {
    let mut values: Vec<&Json> = Vec::new();
    for key in ["images", "image"] {
        match body.get(key) {
            None | Some(Json::Null) => {}
            Some(Json::Arr(items)) => values.extend(items.iter()),
            Some(one) => values.push(one),
        }
    }
    if values.len() > 3 {
        return Err("at most three reference images".into());
    }
    let mut paths = Vec::new();
    for v in values {
        if let Some(p) = reference_image(v, output_root, allow_local)? {
            paths.push(p);
        }
    }
    Ok(paths)
}

/// The worker request for an image job, and a summary for the job list. With
/// reference images (`images`/`image`) the job edits them: Qwen Image conditions
/// on up to three.
pub fn image_request(cfg: &Json, root: &Path, output_root: &Path, body: &Json, allow_local: bool) -> Result<(Json, String, String, usize), String> {
    let media = cfg.get("media").ok_or("no media section")?;
    let section = media.get("image").ok_or("no image section")?;
    if !bool_or(section, "enabled", true) {
        return Err("image generation is disabled".into());
    }
    let (name, model) = pick_model(section, body, "image")?;
    let prompt = prompt(body)?;
    let n = match body.get("n") {
        None | Some(Json::Null) => 1,
        Some(v) => v.as_i64().filter(|n| (1..=16).contains(n)).ok_or("n must be between 1 and 16")?,
    };
    let arch = str_or(model, "architecture", "qwen-image");
    let step = if arch == "sdxl" { 64 } else { 32 };
    let refs = references(body, output_root, allow_local)?;
    if !refs.is_empty() && arch == "sdxl" {
        return Err(format!("{name} is an SDXL model, which cannot edit images; pick a Qwen Image model"));
    }
    // Editing with `size: auto` keeps the first reference's shape at about a megapixel.
    let auto = matches!(body.get("size").and_then(Json::as_str), None | Some("auto" | ""));
    let first = refs.first().and_then(|p| std::fs::read(p).ok()).and_then(|b| crate::multipart::image_info(&b));
    let (w, h) = match (auto && body.get("width").is_none(), first) {
        (true, Some((_, iw, ih))) => crate::multipart::size_like(iw, ih, step),
        _ => size(body, model, (1024, 1024))?,
    };
    if !(256..=2048).contains(&w) || !(256..=2048).contains(&h) {
        return Err("width and height must be between 256 and 2048".into());
    }
    // OpenAI's fixed sizes (1536x1024, 1792x1024) round to the model's grid.
    let (w, h) = ((w / step).max(1) * step, (h / step).max(1) * step);
    let out = output_dir(output_root, "image");
    let mut f: Vec<(String, Json)> = vec![
        ("prompt".into(), Json::str(&prompt)),
        ("n".into(), Json::Int(n)),
        ("width".into(), Json::Int(w)),
        ("height".into(), Json::Int(h)),
        ("seed".into(), Json::Int(seed(body)?)),
        ("device".into(), Json::Int(int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(out.to_string_lossy())),
        ("model".into(), Json::str(&name)),
    ];
    let steps = |default: i64, lo: i64, hi: i64| -> Result<i64, String> {
        let n = body.get("steps").and_then(Json::as_i64).unwrap_or(int_or(model, "steps", default));
        if !(lo..=hi).contains(&n) {
            return Err(format!("steps must be between {lo} and {hi}"));
        }
        Ok(n)
    };
    if arch == "sdxl" {
        f.push(("architecture".into(), Json::str("sdxl")));
        f.push(("checkpoint".into(), Json::str(path_field(root, model, "checkpoint").ok_or("SDXL model needs a checkpoint")?)));
        f.push(("tokenizer".into(), Json::str(path_field(root, model, "tokenizer").ok_or("SDXL model needs the CLIP tokenizer.json")?)));
        f.push(("steps".into(), Json::Int(steps(20, 2, 100)?)));
        let cfg_scale = body.get("cfg").and_then(Json::as_f64).unwrap_or(num_or(model, "cfg", 4.0));
        f.push(("cfg".into(), Json::Num(cfg_scale)));
        let negative = body.get("negative_prompt").and_then(Json::as_str).unwrap_or(str_or(model, "negative_prompt", ""));
        f.push(("negative_prompt".into(), Json::str(negative)));
        f.push(("clip_skip".into(), Json::Int(int_or(model, "clip_skip", 1))));
    } else {
        f.push(("base".into(), Json::str(path_field(root, model, "base").ok_or("Qwen Image model needs its base folder")?)));
        let gguf = path_field(root, model, "transformer");
        let safetensors = path_field(root, model, "safetensors_transformer");
        let weights = body.get("weights").and_then(Json::as_str).or(model.get("weights").and_then(Json::as_str));
        let transformer = match (weights, &gguf, &safetensors) {
            (Some("safetensors"), _, Some(s)) | (None, None, Some(s)) => s.clone(),
            (_, Some(g), _) => g.clone(),
            (_, None, Some(s)) => s.clone(),
            _ => return Err("Qwen Image model needs a transformer".into()),
        };
        f.push(("transformer".into(), Json::str(transformer)));
        let turbo = body.get("turbo").and_then(Json::as_bool).unwrap_or(true);
        let adapter = path_field(root, model, "adapter").filter(|_| turbo);
        let default_steps = if adapter.is_some() { 6 } else { 40 };
        let steps = steps(default_steps, 2, 100)?;
        if adapter.is_some() && !matches!(steps, 4 | 6) {
            return Err("turbo (adapter) models take 4 or 6 steps".into());
        }
        f.push(("steps".into(), Json::Int(steps)));
        if adapter.is_none() {
            let cfg_scale = body.get("cfg").and_then(Json::as_f64).unwrap_or(num_or(model, "cfg", 4.0));
            f.push(("cfg".into(), Json::Num(cfg_scale)));
        }
        f.push(("adapter".into(), adapter.map_or(Json::Null, Json::str)));
        if let Some(te) = path_field(root, model, "text_encoder") {
            f.push(("text_encoder".into(), Json::str(te)));
        }
        if !refs.is_empty() {
            f.push(("images".into(), Json::Arr(refs.iter().map(Json::str).collect())));
            if let Some(n) = body.get("reference_size").and_then(Json::as_i64) {
                f.push(("reference_size".into(), Json::Int(n)));
            }
        }
    }
    f.extend(residency(body, section, model)?);
    Ok((Json::Obj(f), name, format!("{w}x{h}"), n as usize))
}

/// Decode an `input_reference` / `image` value (a `data:` URL, or a local path
/// when `allow_local`) into a file the worker can read.
fn reference_image(value: &Json, output_root: &Path, allow_local: bool) -> Result<Option<String>, String> {
    let text = match value {
        Json::Null => return Ok(None),
        Json::Str(s) => s.clone(),
        other => match other.get("image_url").or_else(|| other.get("url")) {
            Some(Json::Str(s)) => s.clone(),
            Some(inner) => inner.get("url").and_then(Json::as_str).ok_or("input_reference needs image_url")?.to_string(),
            None if other.get("file_id").is_some() => return Err("file_id references need the Files API, which the studio does not host; send image_url as a data: URL".into()),
            None => return Err("input_reference needs image_url".into()),
        },
    };
    if let Some(rest) = text.strip_prefix("data:") {
        let (head, data) = rest.split_once(',').ok_or("bad data URL")?;
        let ext = if head.contains("png") { "png" } else if head.contains("webp") { "webp" } else { "jpg" };
        let bytes = base64_decode(data)?;
        if bytes.len() > 32 << 20 {
            return Err("reference images are limited to 32 MiB".into());
        }
        let dir = output_root.join("inputs");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.{ext}", random_id("ref_")));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        return Ok(Some(path.to_string_lossy().into_owned()));
    }
    if text.starts_with("http://") || text.starts_with("https://") {
        return Err("remote image URLs are not fetched; send a data: URL".into());
    }
    if !allow_local {
        return Err("local image paths are only accepted from this machine; send a data: URL".into());
    }
    let p = PathBuf::from(text.strip_prefix("file://").unwrap_or(&text));
    let meta = std::fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    if !p.is_absolute() || !meta.is_file() || meta.len() > 32 << 20 {
        return Err("reference images must be absolute paths to files of at most 32 MiB".into());
    }
    Ok(Some(p.to_string_lossy().into_owned()))
}

/// Decode an `input_audio` value into a file the worker can read: a `data:`
/// URL, `{"data": base64, "format": "wav"}` (OpenAI's audio input), `{"url"}`
/// / `{"audio_url"}`, or a local path when `allow_local`.
fn reference_audio(value: &Json, output_root: &Path, allow_local: bool) -> Result<Option<String>, String> {
    const FORMATS: [&str; 8] = ["wav", "mp3", "ogg", "opus", "flac", "m4a", "aac", "webm"];
    let save = |bytes: Vec<u8>, ext: &str| -> Result<Option<String>, String> {
        // A request is at most 64 MiB, and base64 grows it by a third.
        if bytes.len() > 45 << 20 {
            return Err("uploaded soundtracks are limited to 45 MiB".into());
        }
        let dir = output_root.join("inputs");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.{ext}", random_id("audio_")));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        Ok(Some(path.to_string_lossy().into_owned()))
    };
    let text = match value {
        Json::Null => return Ok(None),
        Json::Str(s) => s.clone(),
        other => {
            if let Some(data) = other.get("data").and_then(Json::as_str) {
                let format = other.get("format").and_then(Json::as_str).unwrap_or("wav").to_lowercase();
                if !FORMATS.contains(&format.as_str()) {
                    return Err(format!("input_audio format must be one of {}", FORMATS.join(", ")));
                }
                return save(base64_decode(data)?, &format);
            }
            match other.get("audio_url").or_else(|| other.get("url")) {
                Some(Json::Str(s)) => s.clone(),
                Some(inner) => inner.get("url").and_then(Json::as_str).ok_or("input_audio needs data or url")?.to_string(),
                None => return Err("input_audio needs data (base64) or url".into()),
            }
        }
    };
    if let Some(rest) = text.strip_prefix("data:") {
        let (head, data) = rest.split_once(',').ok_or("bad data URL")?;
        let ext = FORMATS.iter().find(|f| head.contains(*f)).copied().unwrap_or(if head.contains("mpeg") { "mp3" } else { "wav" });
        return save(base64_decode(data)?, ext);
    }
    if text.starts_with("http://") || text.starts_with("https://") {
        return Err("remote audio URLs are not fetched; send a data: URL".into());
    }
    if !allow_local {
        return Err("local audio paths are only accepted from this machine; send a data: URL".into());
    }
    let p = PathBuf::from(text.strip_prefix("file://").unwrap_or(&text));
    let meta = std::fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    if !p.is_absolute() || !meta.is_file() || meta.len() > 256 << 20 {
        return Err("soundtracks must be absolute paths to files of at most 256 MiB".into());
    }
    Ok(Some(p.to_string_lossy().into_owned()))
}

/// The worker request for a video job. `seconds` (OpenAI) or `frames` sets the
/// length; LTX takes 8k+1 frames, at most 121. With a soundtrack to follow
/// (`input_audio`, or `speech` to make first) and no length, the clip is as
/// long as the soundtrack.
pub fn video_request(cfg: &Json, root: &Path, output_root: &Path, body: &Json, allow_local: bool) -> Result<(Json, String, String, f64), String> {
    let media = cfg.get("media").ok_or("no media section")?;
    let section = media.get("video").ok_or("no video section")?;
    if !bool_or(section, "enabled", true) {
        return Err("video generation is disabled".into());
    }
    let (name, model) = pick_model(section, body, "video")?;
    let family = str_or(model, "family", &name).to_string();
    if !VIDEO_FAMILIES.contains(&family.as_str()) {
        return Err(format!("video model {name} has no LTX family"));
    }
    let prompt = prompt(body)?;
    let fps = body.get("fps").and_then(Json::as_i64).unwrap_or(int_or(section, "fps", 24));
    if !(1..=60).contains(&fps) {
        return Err("fps must be between 1 and 60".into());
    }
    let follows_audio = body.get("input_audio").is_some_and(|v| !matches!(v, Json::Null)) || body.get("speech").is_some_and(|v| !matches!(v, Json::Null));
    let explicit_length = body.get("frames").and_then(Json::as_i64).is_some() || body.get("seconds").is_some_and(|v| !matches!(v, Json::Null));
    let frames = match (body.get("frames").and_then(Json::as_i64), body.get("seconds")) {
        (Some(f), _) => f,
        (None, Some(s)) => {
            let secs = s.as_f64().or_else(|| s.as_str().and_then(|t| t.trim().parse().ok())).ok_or("seconds must be a number")?;
            if !(0.0..=60.0).contains(&secs) {
                return Err("seconds must be between 0 and 60".into());
            }
            ((secs * fps as f64 - 1.0) / 8.0).round() as i64 * 8 + 1
        }
        (None, None) => int_or(model, "frames", 49),
    }
    .clamp(9, 121);
    let frames = (frames - 1) / 8 * 8 + 1;
    let (w, h) = size(body, model, (768, 512))?;
    if w <= 0 || h <= 0 {
        return Err("size must be positive".into());
    }
    // OpenAI sizes (1280x720, 1792x1024) exceed LTX's 1024 cap: scale, keep aspect.
    let (w, h) = fit(w, h, 1024, 128, 32);
    let mut f: Vec<(String, Json)> = vec![
        ("kind".into(), Json::str("video")),
        ("model".into(), Json::str(&family)),
        ("prompt".into(), Json::str(&prompt)),
        ("width".into(), Json::Int(w)),
        ("height".into(), Json::Int(h)),
        ("fps".into(), Json::Int(fps)),
        ("seed".into(), Json::Int(seed(body)?)),
        ("device".into(), Json::Int(int_or(media, "device", 0))),
        ("output_dir".into(), Json::str(output_dir(output_root, "video").to_string_lossy())),
        ("ffmpeg".into(), Json::str(config::program(root, str_or(section, "ffmpeg", "ffmpeg")).to_string_lossy())),
    ];
    for key in ["transformer", "text_encoder", "vae", "tokenizer"] {
        match path_field(root, model, key) {
            Some(p) => f.push((key.into(), Json::str(p))),
            None if key == "tokenizer" && family == "ltx-2.5" => {}
            None => return Err(format!("video model {name} needs {key}")),
        }
    }
    // A soundtrack comes with models that have their audio VAE (LTX 2.3 and
    // Sulphur checkpoints are their own), unless the request says `"audio": false`.
    let audio_vae = path_field(root, model, "audio_vae");
    match body.get("audio") {
        None | Some(Json::Null) => {}
        Some(Json::Bool(false)) => f.push(("audio".into(), Json::Bool(false))),
        Some(Json::Bool(true)) if audio_vae.is_some() => f.push(("audio".into(), Json::Bool(true))),
        Some(Json::Bool(true)) => return Err(format!("video model {name} has no audio VAE, so it cannot make sound")),
        Some(_) => return Err("audio must be true or false".into()),
    }
    if follows_audio {
        if audio_vae.is_none() {
            return Err(format!("video model {name} has no audio VAE, so it cannot follow a soundtrack"));
        }
        if body.get("audio") == Some(&Json::Bool(false)) {
            return Err("the soundtrack is the clip's audio; leave audio on".into());
        }
        if body.get("input_audio").is_some_and(|v| !matches!(v, Json::Null)) && body.get("speech").is_some_and(|v| !matches!(v, Json::Null)) {
            return Err("send input_audio or speech, not both".into());
        }
    }
    if let Some(p) = audio_vae {
        f.push(("audio_vae".into(), Json::str(p)));
    }
    if let Some(t) = body.get("transcript").and_then(Json::as_str).map(str::trim).filter(|t| !t.is_empty()) {
        if t.len() > 4000 {
            return Err("transcript is limited to 4000 bytes".into());
        }
        f.push(("transcript".into(), Json::str(t)));
    }
    if let Some(g) = body.get("a2v_guidance").filter(|v| !matches!(v, Json::Null)) {
        let g = g.as_f64().filter(|g| (1.0..=10.0).contains(g)).ok_or("a2v_guidance must be a number from 1 to 10")?;
        f.push(("a2v_guidance".into(), Json::Num(g)));
    }
    // Without a length, the worker makes the clip as long as the soundtrack.
    let mut seconds = (frames - 1) as f64 / fps as f64;
    if !follows_audio || explicit_length {
        f.push(("frames".into(), Json::Int(frames)));
    }
    if let Some(v) = body.get("input_audio") {
        if let Some(p) = reference_audio(v, output_root, allow_local)? {
            f.push(("audio_file".into(), Json::str(p)));
            if !explicit_length {
                seconds = 120.0 / fps as f64;
            }
        }
    }
    if let Some(s) = body.get("speech").filter(|v| !matches!(v, Json::Null)) {
        // What to say and in which voice: `input` (or `text`), `voice` (a saved
        // voice or an OpenAI name), `instructions`, `language`, `seed`, `model`.
        let mut sb = s.clone();
        if sb.get("input").is_none() {
            if let Some(t) = s.get("text").cloned() {
                crate::util::set(&mut sb, "input", t);
            }
        }
        let (request, _, _, speech_frames) = crate::speech::speech_request(cfg, root, &day_dir(output_root, "speech"), &sb).map_err(|e| format!("speech: {e}"))?;
        f.push(("speech".into(), request));
        if !explicit_length {
            // Speech runs at 12.5 frames a second; the clip stops at 121 video frames.
            seconds = (speech_frames as f64 / 12.5).min(120.0 / fps as f64);
        }
    }
    let start = body.get("input_reference").or_else(|| body.get("image"));
    if let Some(v) = start {
        if let Some(p) = reference_image(v, output_root, allow_local)? {
            f.push(("image".into(), Json::str(p)));
        }
    }
    if let Some(v) = body.get("end_image") {
        if let Some(p) = reference_image(v, output_root, allow_local)? {
            f.push(("end_image".into(), Json::str(p)));
        }
    }
    if body.get("prompt_cache").and_then(Json::as_bool).unwrap_or(true) {
        // Incognito jobs build under `.incognito/`: no prompt cache for them.
        if !output_root.components().any(|c| c.as_os_str() == ".incognito") {
            f.push(("cache_dir".into(), Json::str(output_root.join(".ltx-prompt-cache").to_string_lossy())));
        }
    }
    let mut memory = residency(body, section, model)?;
    // The video worker reads vram_gb as a cap it lowers to what is free.
    if !memory.iter().any(|(k, _)| k == "vram_gb") {
        memory.push(("vram_gb".into(), Json::Int(192)));
    }
    f.extend(memory);
    Ok((Json::Obj(f), name, format!("{w}x{h}"), seconds))
}

/// Whether a media job must have the GPU to itself: `llm_policy` `pause_llm`
/// always, `coexist` never, `auto` when the media device is one the LLM uses.
pub fn pauses_llm(cfg: &Json) -> bool {
    let media = cfg.get("media").cloned().unwrap_or(Json::Null);
    let device = int_or(&media, "device", 0);
    let devices: Vec<i64> = cfg.get("llm").and_then(|l| l.get("devices")).and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_i64).collect();
    match str_or(&media, "llm_policy", "auto") {
        "coexist" => false,
        "pause_llm" => true,
        // Without explicit devices nrob-server takes the first two GPUs.
        _ => if devices.is_empty() { device < 2 } else { devices.contains(&device) },
    }
}

/// Progress (0-100) and a stage label from one worker event line.
fn progress_of(kind: Kind, n: usize, e: &Json) -> Option<(f64, String)> {
    let stage = e.get("stage").and_then(Json::as_str)?.to_string();
    let i = |k: &str| e.get(k).and_then(Json::as_i64).unwrap_or(0) as f64;
    let n = n.max(1) as f64;
    let p = match (kind, stage.as_str()) {
        (Kind::Image, "loading_transformer") if i("blocks") > 0.0 => 5.0 + 10.0 * i("block") / i("blocks"),
        (Kind::Image, "sampling") if i("steps") > 0.0 => 15.0 + 80.0 * ((i("image") - 1.0) + i("step") / i("steps")) / n,
        (Kind::Image, "image_saved") => 15.0 + 80.0 * i("image") / n,
        (Kind::Image, "decoding") => 15.0 + 80.0 * i("image") / n - 1.0,
        (Kind::Image, "loading_unet" | "encoding_prompt") => 5.0 + 80.0 * (i("image") - 1.0).max(0.0) / n,
        (Kind::Image, _) => 3.0,
        // Speech to follow comes first, then its reading.
        (Kind::Video, "speaking") if i("total") > 0.0 => 1.0 + 3.0 * (1.0 - (-3.0 * i("current") / i("total")).exp()),
        (Kind::Video, "decoding_speech" | "reading_soundtrack") => 4.0,
        (Kind::Video, "encoding_soundtrack") => 5.0,
        (Kind::Video, "encoding_video_prompt") if i("total") > 0.0 => 3.0 + 12.0 * i("current") / i("total"),
        (Kind::Video, "video_text_connector") => 16.0,
        (Kind::Video, "video_denoising") if i("total") > 0.0 => 18.0 + 67.0 * i("current") / i("total"),
        (Kind::Video, "decoding_video") => 87.0,
        (Kind::Video, "encoding_mp4") => 95.0,
        (Kind::Video, _) => 2.0,
        // `n` holds the expected frame count; the curve never quite reaches it.
        (Kind::Speech, "speaking") => 8.0 + 84.0 * (1.0 - (-i("current") / n).exp()),
        (Kind::Speech, "designing_voice") => 5.0 + 30.0 * i("current"),
        (Kind::Speech, "decoding_speech") => 95.0,
        (Kind::Speech, _) => 3.0,
        // `n` holds the most frames the song may have; it may end sooner.
        (Kind::Music, "loading_music_model") if i("total") > 0.0 => 2.0 + 6.0 * i("current") / i("total"),
        (Kind::Music, "composing") => 8.0 + 52.0 * i("current") / n,
        (Kind::Music, "loading_renderer") if i("total") > 0.0 => 60.0 + 4.0 * i("current") / i("total"),
        (Kind::Music, "rendering") if i("total") > 0.0 => 64.0 + 34.0 * i("current") / i("total"),
        (Kind::Music, "quantizing") if i("total") > 0.0 => 2.0 + 96.0 * i("current") / i("total"),
        (Kind::Music, _) => 2.0,
    };
    Some((p, stage))
}

impl Media {
    pub fn new() -> Media {
        Media {
            jobs: Mutex::new(VecDeque::new()),
            changed: Condvar::new(),
            broker: Mutex::new(Broker { chats: 0, media: false, waiting_media: false }),
            broker_changed: Condvar::new(),
            queue_signal: Condvar::new(),
            log: Arc::new(LogRing::new(2000)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Job>> {
        self.jobs.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Queue a prepared job; returns its id.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(&self, kind: Kind, request: Json, model: String, size: String, n: usize, seconds: f64, keep: usize, scratch: Option<PathBuf>) -> Job {
        let prompt = str_or(&request, "prompt", "").to_string();
        let job = Job {
            id: random_id(match kind {
                Kind::Image => "img_",
                Kind::Video => "video_",
                Kind::Speech => "speech_",
                Kind::Music => "music_",
            }),
            kind,
            status: "queued".into(),
            progress: 0.0,
            stage: "queued".into(),
            created_at: now(),
            completed_at: None,
            prompt,
            model,
            size,
            seconds,
            n,
            request,
            result: Json::Null,
            files: Vec::new(),
            preview: None,
            error: None,
            cancel: Arc::new(AtomicBool::new(false)),
            incognito: scratch.is_some(),
            scratch,
            expires_at: None,
            forget: false,
        };
        let mut jobs = self.lock();
        // Forget the oldest finished jobs beyond `keep` (their files stay on disk).
        // Not ones that just finished: a synchronous request may still be about
        // to read its result. Incognito jobs expire on their own.
        let settled = |j: &Job| j.finished() && !j.incognito && j.completed_at.is_some_and(|t| now().saturating_sub(t) > 120);
        while jobs.len() >= keep.max(1) {
            match jobs.iter().position(settled) {
                Some(i) => {
                    jobs.remove(i);
                }
                None => break,
            }
        }
        jobs.push_back(job.clone());
        self.queue_signal.notify_all();
        self.changed.notify_all();
        job
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.lock().iter().find(|j| j.id == id).cloned()
    }

    /// Jobs for lists and the gallery: incognito ones never appear.
    pub fn list(&self) -> Vec<Job> {
        self.lock().iter().rev().filter(|j| !j.incognito).cloned().collect()
    }

    /// Unfinished incognito jobs as progress only: no prompt, model or files,
    /// so the UI can show that the GPU is busy (and offer Cancel) without
    /// listing what the job is.
    pub fn private_activity(&self) -> Json {
        Json::Arr(
            self.lock()
                .iter()
                .filter(|j| j.incognito && !j.finished())
                .map(|j| {
                    Json::obj([
                        ("id", Json::str(&j.id)),
                        ("kind", Json::str(j.kind.name())),
                        ("status", Json::str(&j.status)),
                        ("progress", Json::Num(j.progress)),
                        ("stage", Json::str(&j.stage)),
                    ])
                })
                .collect(),
        )
    }

    /// The job on the GPU now, if any (incognito included): kind and progress.
    pub fn running(&self) -> Option<(Kind, f64)> {
        self.lock().iter().find(|j| j.status == "in_progress").map(|j| (j.kind, j.progress))
    }

    /// Chat requests holding the LLM right now.
    pub fn chats_active(&self) -> usize {
        self.broker.lock().unwrap_or_else(|p| p.into_inner()).chats
    }

    /// Remove a job and, for incognito ones, everything it wrote.
    pub fn purge(&self, id: &str) {
        let gone: Vec<Job> = {
            let mut jobs = self.lock();
            let (gone, keep): (Vec<Job>, Vec<Job>) = jobs.drain(..).partition(|j| j.id == id);
            jobs.extend(keep);
            gone
        };
        for job in gone {
            if let Some(dir) = &job.scratch {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    /// Delete incognito jobs past their expiry, and `forget`-marked ones that
    /// have finished. Call periodically.
    pub fn sweep(&self) {
        let due: Vec<String> = self
            .lock()
            .iter()
            // An incognito job cancelled while queued never reached the runner
            // that stamps `expires_at`: nothing is left to deliver, so it goes.
            .filter(|j| j.finished() && (j.forget || j.expires_at.map_or(j.incognito, |t| now() >= t)))
            .map(|j| j.id.clone())
            .collect();
        for id in due {
            self.purge(&id);
        }
    }

    pub fn busy(&self) -> bool {
        self.lock().iter().any(|j| !j.finished())
    }

    /// Forget a job (its files stay, unless incognito). A running one is
    /// cancelled and forgotten once it stops.
    pub fn remove(&self, id: &str) -> bool {
        self.cancel(id);
        let finished = {
            let mut jobs = self.lock();
            let Some(j) = jobs.iter_mut().find(|j| j.id == id) else { return false };
            j.forget = true;
            j.finished()
        };
        if finished {
            self.purge(id);
        }
        true
    }

    pub fn cancel(&self, id: &str) -> bool {
        let mut jobs = self.lock();
        let Some(j) = jobs.iter_mut().find(|j| j.id == id) else { return false };
        if j.finished() {
            return false;
        }
        j.cancel.store(true, Ordering::Relaxed);
        if j.status == "queued" {
            j.status = "cancelled".into();
            j.stage = "cancelled".into();
            j.completed_at = Some(now());
        }
        self.changed.notify_all();
        // The runner re-checks the queue: if this was the last job, the LLM it
        // paused comes back.
        self.queue_signal.notify_all();
        true
    }

    /// Wait until the job finishes (or `timeout`); the latest state either way.
    pub fn wait(&self, id: &str, timeout: Duration) -> Option<Job> {
        let deadline = std::time::Instant::now() + timeout;
        let mut jobs = self.lock();
        loop {
            let job = jobs.iter().find(|j| j.id == id)?.clone();
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if job.finished() || left.is_zero() {
                return Some(job);
            }
            jobs = self.changed.wait_timeout(jobs, left.min(Duration::from_secs(1))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        let mut jobs = self.lock();
        if let Some(j) = jobs.iter_mut().find(|j| j.id == id) {
            f(j);
        }
        self.changed.notify_all();
    }

    /// Hold the LLM for one chat request. Waits (up to `timeout`) while a media
    /// job that needs the LLM's GPU is queued to start or running.
    pub fn chat_lease(&self, exclusive: bool, timeout: Duration) -> Result<ChatLease<'_>, String> {
        let mut b = self.broker.lock().unwrap_or_else(|p| p.into_inner());
        let deadline = std::time::Instant::now() + timeout;
        while exclusive && (b.media || b.waiting_media) {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return Err("a media job is using the GPU the language model needs; try again when it finishes".into());
            }
            b = self.broker_changed.wait_timeout(b, left).unwrap_or_else(|p| p.into_inner()).0;
        }
        b.chats += 1;
        Ok(ChatLease(self))
    }

    /// The runner: one job at a time, forever. `studio` supplies configuration
    /// and the LLM supervisor.
    pub fn run(self: &Arc<Self>, studio: &crate::Studio) {
        loop {
            let next = {
                let mut jobs = self.lock();
                loop {
                    if let Some(j) = jobs.iter_mut().find(|j| j.status == "queued") {
                        j.status = "in_progress".into();
                        j.stage = "starting".into();
                        break j.clone();
                    }
                    // Nothing queued: bring back an LLM that media paused. Here
                    // rather than after each job, so a batch does not reload it
                    // between jobs, and a cancelled last job still resumes it.
                    if studio.llm.paused() {
                        drop(jobs);
                        self.resume_llm(studio);
                        jobs = self.lock();
                        continue;
                    }
                    jobs = self.queue_signal.wait(jobs).unwrap_or_else(|p| p.into_inner());
                }
            };
            self.changed.notify_all();
            let outcome = self.run_one(studio, &next);
            let cancelled = next.cancel.load(Ordering::Relaxed);
            self.update(&next.id, |j| {
                j.completed_at = Some(now());
                if j.incognito {
                    // Delivered and deleted on read, or deleted when this passes.
                    j.expires_at = Some(now() + if j.kind == Kind::Video { 30 * 60 } else { 10 * 60 });
                }
                match outcome {
                    Ok(result) => {
                        j.status = "completed".into();
                        j.progress = 100.0;
                        j.stage = "completed".into();
                        j.files = match j.kind {
                            Kind::Image => result.get("data").and_then(Json::as_array).unwrap_or(&[]).iter()
                                .filter_map(|d| d.get("path").and_then(Json::as_str)).map(PathBuf::from).collect(),
                            Kind::Video | Kind::Speech => result.get("path").and_then(Json::as_str).map(PathBuf::from).into_iter().collect(),
                            // A song; a smaller copy of the model is not a file to show.
                            Kind::Music => result.get("path").and_then(Json::as_str).map(PathBuf::from).filter(|p| p.extension().is_some_and(|e| e == "wav")).into_iter().collect(),
                        };
                        j.preview = match j.kind {
                            Kind::Image => j.files.first().cloned(),
                            Kind::Video => result.get("preview").and_then(Json::as_str).map(PathBuf::from),
                            Kind::Speech | Kind::Music => None,
                        };
                        j.result = result;
                    }
                    Err(e) => {
                        j.status = if cancelled { "cancelled" } else { "failed" }.into();
                        j.stage = j.status.clone();
                        j.error = Some(e);
                    }
                }
            });
            self.sweep();
        }
    }

    fn resume_llm(&self, studio: &crate::Studio) {
        studio.llm.set_paused(false);
        let cfg = studio.config();
        if cfg.get("media").is_none_or(|m| bool_or(m, "resume_llm", true)) {
            self.log.push("restarting the LLM after media jobs");
            if let Err(e) = studio.llm.start(&cfg, &studio.root) {
                self.log.push(format!("LLM restart failed: {e}"));
            }
        }
    }

    fn run_one(&self, studio: &crate::Studio, job: &Job) -> Result<Json, String> {
        let cfg = studio.config();
        let media = cfg.get("media").cloned().unwrap_or(Json::Null);
        let device = int_or(&media, "device", 0);
        let pause = pauses_llm(&cfg);
        {
            let mut b = self.broker.lock().unwrap_or_else(|p| p.into_inner());
            if pause {
                b.waiting_media = true;
                // Chats finish, or their streams time out on a client that
                // stopped reading; either way, not forever.
                let deadline = std::time::Instant::now() + Duration::from_secs(10 * 60);
                while b.chats > 0 {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        self.log.push("chat requests still running after 10 minutes; starting the media job anyway");
                        break;
                    }
                    b = self.broker_changed.wait_timeout(b, left).unwrap_or_else(|p| p.into_inner()).0;
                }
            }
            b.media = pause;
            b.waiting_media = false;
        }
        let paused_llm = pause && studio.llm.is_running();
        if paused_llm {
            self.log.push(format!("stopping the LLM to free GPU {device} for a media job"));
            studio.llm.stop();
            studio.llm.set_paused(true);
        }
        let result = if job.cancel.load(Ordering::Relaxed) { Err("cancelled".into()) } else { self.worker(studio, &cfg, job) };
        {
            let mut b = self.broker.lock().unwrap_or_else(|p| p.into_inner());
            b.media = false;
            self.broker_changed.notify_all();
        }
        result
    }

    fn worker(&self, studio: &crate::Studio, cfg: &Json, job: &Job) -> Result<Json, String> {
        let media = cfg.get("media").ok_or("no media section")?;
        let program = config::program(&studio.root, str_or(media, "worker", "nrob-diffusion"));
        let output_root = studio.output_root();
        std::fs::create_dir_all(&output_root).map_err(|e| e.to_string())?;
        let mut command = Command::new(&program);
        if std::env::var_os("CUDA_CACHE_PATH").is_none() {
            // Share compiled CUDA kernels across jobs.
            let cache = output_root.join(".cuda-cache");
            let _ = std::fs::create_dir_all(&cache);
            command.env("CUDA_CACHE_PATH", cache);
        }
        command.arg("--stdin").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        let label = if job.incognito { "incognito job".to_string() } else { job.id.clone() };
        if job.incognito {
            self.log.push(format!("incognito {} job on GPU {}", job.kind.name(), int_or(&job.request, "device", 0)));
        } else {
            self.log.push(format!("{}: {} job on GPU {} ({} {})", job.id, job.kind.name(), int_or(&job.request, "device", 0), job.model, str_or(&job.request, "memory", "auto")));
        }
        let mut child = command.spawn().map_err(|e| format!("could not start {}: {e} (is nrob-diffusion built with --features flash-attn? set media.worker)", program.display()))?;
        let payload = job.request.to_json();
        let sent = child.stdin.take().ok_or("worker stdin missing".to_string()).and_then(|mut s| s.write_all(payload.as_bytes()).map_err(|e| e.to_string()));
        if let Err(e) = sent {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("sending the request to the worker: {e}"));
        }
        let stdout = child.stdout.take().ok_or("worker stdout missing")?;
        let stderr = child.stderr.take().ok_or("worker stderr missing")?;
        let output = std::thread::spawn(move || {
            let mut b = Vec::new();
            stdout.take(8 << 20).read_to_end(&mut b).map(|_| b)
        });
        let (id, kind, n, incognito) = (job.id.clone(), job.kind, job.n, job.incognito);
        let errors = std::thread::scope(|s| {
            let events = s.spawn(|| {
                let mut last_error = None;
                let mut tail = String::new();
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if let Ok(e) = Json::parse(line.as_bytes()) {
                        if let Some(msg) = e.get("error").and_then(Json::as_str) {
                            last_error = Some(msg.to_string());
                        }
                        if let Some((p, stage)) = progress_of(kind, n, &e) {
                            self.update(&id, |j| {
                                j.progress = j.progress.max(p.min(99.0));
                                j.stage = stage;
                            });
                        }
                    } else if !incognito { self.log.push(format!("{id}: {line}")); }
                    tail = line.chars().take(4096).collect();
                }
                last_error.unwrap_or(tail)
            });
            let status = loop {
                if job.cancel.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err("cancelled".to_string());
                }
                match child.try_wait() {
                    Ok(Some(s)) => break Ok(s),
                    Ok(None) => std::thread::sleep(Duration::from_millis(200)),
                    Err(e) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break Err(e.to_string());
                    }
                }
            };
            (status, events.join().unwrap_or_default())
        });
        let (status, tail) = errors;
        let status = status?;
        let bytes = output.join().map_err(|_| "worker output reader failed")?.map_err(|e| e.to_string())?;
        if !status.success() {
            self.log.push(format!("{label}: worker failed{}", if job.incognito { String::new() } else { format!(": {tail}") }));
            return Err(if tail.is_empty() { format!("worker exited {status}") } else { tail });
        }
        let result = Json::parse(&bytes).map_err(|e| format!("invalid worker result: {e}"))?;
        self.log.push(format!("{label}: done in {:.1}s", num_or(&result, "seconds", 0.0)));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(extra_image: &str, extra_video: &str) -> Json {
        let mut v = config::default_json();
        let image = Json::parse(format!(r#"{{"enabled":true,"default_model":"qwen","memory":"auto","ram_gb":32,"vram_gb":20,"models":{{
            "qwen":{{"architecture":"qwen-image","base":"base","transformer":"q.gguf","adapter":"turbo.safetensors"{extra_image}}},
            "anime":{{"architecture":"sdxl","checkpoint":"/abs/anime.safetensors","tokenizer":"tok.json","steps":24,"memory":"ssd"}}}}}}"#).as_bytes()).unwrap();
        let video = Json::parse(format!(r#"{{"enabled":true,"default_model":"sulphur","ram_gb":48,"vram_gb":null,"fps":24,"ffmpeg":"ffmpeg","models":{{
            "sulphur":{{"family":"sulphur-2","transformer":"s.safetensors","vae":"s.safetensors","text_encoder":"g.safetensors","tokenizer":"t.json"{extra_video}}}}}}}"#).as_bytes()).unwrap();
        let Json::Obj(top) = &mut v else { unreachable!() };
        let media = &mut top.iter_mut().find(|(k, _)| k == "media").unwrap().1;
        crate::util::set(media, "image", image);
        crate::util::set(media, "video", video);
        crate::util::set(media, "device", Json::Int(1));
        config::validate(&v).unwrap();
        v
    }

    fn body(s: &str) -> Json {
        Json::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn image_requests_map_openai_fields_onto_the_worker() {
        let c = cfg("", "");
        let root = Path::new("/install");
        let out = Path::new("/install/outputs");
        let (r, name, size, n) = image_request(&c, root, out, &body(r#"{"prompt":"a fox","n":2,"size":"1536x1024","model":"dall-e-3","seed":5}"#), false).unwrap();
        assert_eq!((name.as_str(), size.as_str(), n), ("qwen", "1536x1024", 2));
        assert_eq!(r.get("steps").and_then(Json::as_i64), Some(6));
        assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
        assert_eq!(r.get("transformer").and_then(Json::as_str), Some(root.join("q.gguf").to_str().unwrap()));
        assert_eq!(r.get("memory").and_then(Json::as_str), Some("auto"));
        assert_eq!((r.get("ram_gb").and_then(Json::as_i64), r.get("vram_gb").and_then(Json::as_i64)), (Some(32), Some(20)));
        assert!(r.get("output_dir").and_then(Json::as_str).unwrap().contains("images"));
        // The SDXL model's own residency applies; requests may lower but not raise caps.
        let (s, ..) = image_request(&c, root, out, &body(r#"{"prompt":"x","model":"anime","size":"1000x1000","ram_gb":8}"#), false).unwrap();
        assert_eq!(s.get("architecture").and_then(Json::as_str), Some("sdxl"));
        assert_eq!((s.get("width").and_then(Json::as_i64), s.get("steps").and_then(Json::as_i64)), (Some(960), Some(24)));
        assert_eq!(s.get("memory").and_then(Json::as_str), Some("ssd"));
        assert_eq!(s.get("ram_gb").and_then(Json::as_i64), Some(8));
        for bad in [r#"{"prompt":""}"#, r#"{"prompt":"x","n":17}"#, r#"{"prompt":"x","size":"big"}"#, r#"{"prompt":"x","ram_gb":33}"#,
            r#"{"prompt":"x","memory":"disk"}"#, r#"{"prompt":"x","model":"missing"}"#, r#"{"prompt":"x","steps":5}"#, r#"{"prompt":"x","size":"4096x4096"}"#] {
            assert!(image_request(&c, root, out, &body(bad), false).is_err(), "{bad}");
        }
    }

    #[test]
    fn edits_carry_references_and_take_their_shape() {
        let c = cfg("", "");
        let out = std::env::temp_dir().join(format!("nrob-studio-edit-{}", std::process::id()));
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png.extend(1920u32.to_be_bytes());
        png.extend(1080u32.to_be_bytes());
        let url = format!("data:image/png;base64,{}", crate::util::base64_encode(&png));
        let edit = body(&format!(r#"{{"prompt":"make it night","images":[{{"image_url":"{url}"}}]}}"#));
        let (r, _, size, _) = image_request(&c, Path::new("/install"), &out, &edit, false).unwrap();
        assert_eq!(size, "1376x768");
        let refs = r.get("images").and_then(Json::as_array).unwrap();
        assert_eq!(std::fs::read(refs[0].as_str().unwrap()).unwrap(), png);
        let four = body(&format!(r#"{{"prompt":"x","image":["{url}","{url}","{url}","{url}"]}}"#));
        assert!(image_request(&c, Path::new("/install"), &out, &four, false).unwrap_err().contains("three"));
        let sdxl = body(&format!(r#"{{"prompt":"x","model":"anime","image":"{url}"}}"#));
        assert!(image_request(&c, Path::new("/install"), &out, &sdxl, false).unwrap_err().contains("SDXL"));
        assert!(image_request(&c, Path::new("/install"), &out, &body(r#"{"prompt":"x","image":"C:/private.png"}"#), false).is_err());
        let _ = std::fs::remove_dir_all(out);
    }

    #[test]
    fn video_requests_fit_openai_sizes_and_seconds_to_ltx() {
        let c = cfg("", "");
        let root = Path::new("/install");
        let out = std::env::temp_dir().join(format!("nrob-studio-video-{}", std::process::id()));
        let (r, name, size, secs) = video_request(&c, root, &out, &body(r#"{"prompt":"waves","model":"sora-2","seconds":"4","size":"1280x720"}"#), false).unwrap();
        assert_eq!((name.as_str(), size.as_str()), ("sulphur", "1024x576"));
        assert_eq!(r.get("model").and_then(Json::as_str), Some("sulphur-2"));
        assert_eq!(r.get("frames").and_then(Json::as_i64), Some(97));
        assert!((secs - 4.0).abs() < 1e-9);
        assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(192));
        assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(48));
        // A data: URL reference becomes a file the worker reads; local paths need trust.
        let png = format!(r#"{{"prompt":"x","input_reference":{{"image_url":"data:image/png;base64,{}"}}}}"#, crate::util::base64_encode(b"\x89PNG"));
        let (r, ..) = video_request(&c, root, &out, &body(&png), false).unwrap();
        assert_eq!(std::fs::read(r.get("image").and_then(Json::as_str).unwrap()).unwrap(), b"\x89PNG");
        assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","image":"C:/secret.png"}"#), false).is_err());
        assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_reference":{"file_id":"f"}}"#), false).is_err());
        let (long, ..) = video_request(&c, root, &out, &body(r#"{"prompt":"x","seconds":30}"#), false).unwrap();
        assert_eq!(long.get("frames").and_then(Json::as_i64), Some(121));
        let _ = std::fs::remove_dir_all(out);
    }

    #[test]
    fn video_follows_a_soundtrack_or_speech() {
        let mut c = cfg("", r#","audio_vae":"s.safetensors""#);
        let speech = Json::parse(br#"{"enabled":true,"default_model":"tts","voices_dir":"voices","models":{"tts":{"design":"/m/design","base":"/m/base"}}}"#).unwrap();
        let Json::Obj(top) = &mut c else { unreachable!() };
        let media = &mut top.iter_mut().find(|(k, _)| k == "media").unwrap().1;
        crate::util::set(media, "speech", speech);
        let root = Path::new("/install");
        let out = std::env::temp_dir().join(format!("nrob-studio-a2v-{}", std::process::id()));
        // OpenAI's audio input shape: the file is written, the length comes from it.
        let wav = format!(r#"{{"prompt":"a woman talks","input_audio":{{"data":"{}","format":"wav"}}}}"#, crate::util::base64_encode(b"RIFF"));
        let (r, ..) = video_request(&c, root, &out, &body(&wav), false).unwrap();
        assert_eq!(std::fs::read(r.get("audio_file").and_then(Json::as_str).unwrap()).unwrap(), b"RIFF");
        assert!(r.get("frames").is_none(), "the worker sizes the clip to the soundtrack");
        // An explicit length still wins.
        let (r, ..) = video_request(&c, root, &out, &body(&wav.replace(r#""prompt""#, r#""seconds":2,"prompt""#)), false).unwrap();
        assert_eq!(r.get("frames").and_then(Json::as_i64), Some(49));
        // Speech is made first, in the same job, with a described voice.
        let (r, _, _, secs) = video_request(&c, root, &out, &body(r#"{"prompt":"a man speaks to camera","speech":{"text":"Hello there, welcome back.","voice":"onyx"}}"#), false).unwrap();
        let s = r.get("speech").unwrap();
        assert_eq!((str_or(s, "kind", ""), str_or(s, "text", "")), ("speech", "Hello there, welcome back."));
        assert!(secs > 0.5 && secs <= 5.0, "{secs}");
        // Refusals: no audio VAE, audio off, both sources, a local file from afar.
        assert!(video_request(&cfg("", ""), root, &out, &body(&wav), false).unwrap_err().contains("audio VAE"));
        assert!(video_request(&c, root, &out, &body(&wav.replace(r#""prompt""#, r#""audio":false,"prompt""#)), false).is_err());
        assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_audio":"C:/voice.wav"}"#), false).unwrap_err().contains("this machine"));
        assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_audio":"data:audio/wav;base64,UklGRg==","speech":{"text":"hi"}}"#), false).is_err());
        let _ = std::fs::remove_dir_all(out);
    }

    #[test]
    fn sizes_fit_and_urls_stay_inside_the_output_root() {
        assert_eq!(fit(1792, 1024, 1024, 128, 32), (1024, 576));
        assert_eq!(fit(720, 1280, 1024, 128, 32), (576, 1024));
        assert_eq!(fit(512, 320, 1024, 128, 32), (512, 320));
        let root = std::env::temp_dir().join(format!("nrob-studio-files-{}", std::process::id()));
        std::fs::create_dir_all(root.join("images")).unwrap();
        let file = root.join("images").join("a b.png");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(file_url(&root, &file).as_deref(), Some("/files/images/a b.png"));
        assert_eq!(resolve_file(&root, "images/a%20b.png"), Some(file.canonicalize().unwrap()));
        assert!(resolve_file(&root, "../secret").is_none());
        assert!(resolve_file(&root, "images/../../x").is_none());
        assert!(resolve_file(&root, "C:/Windows/win.ini").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn worker_events_become_monotonic_progress() {
        let e = |s: &str| body(s);
        let (p, stage) = progress_of(Kind::Image, 2, &e(r#"{"stage":"sampling","image":2,"step":3,"steps":6}"#)).unwrap();
        assert_eq!(stage, "sampling");
        assert!((p - 75.0).abs() < 1e-9);
        let (p, _) = progress_of(Kind::Video, 1, &e(r#"{"stage":"video_denoising","current":192,"total":384}"#)).unwrap();
        assert!((p - 51.5).abs() < 1e-9);
        assert!(progress_of(Kind::Image, 1, &e(r#"{"error":"x"}"#)).is_none());
    }

    #[test]
    fn jobs_queue_cancel_and_trim_to_the_kept_count() {
        let m = Media::new();
        let a = m.submit(Kind::Image, body(r#"{"prompt":"a"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 2, None);
        assert!(m.busy());
        assert!(m.cancel(&a.id));
        assert_eq!(m.get(&a.id).unwrap().status, "cancelled");
        m.submit(Kind::Image, body(r#"{"prompt":"b"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 2, None);
        // Just finished: a synchronous caller may still be reading it, so it stays.
        assert!(m.get(&a.id).is_some());
        m.update(&a.id, |j| j.completed_at = Some(0));
        m.submit(Kind::Video, body(r#"{"prompt":"c"}"#), "v".into(), "1x1".into(), 1, 1.0, 2, None);
        assert!(m.get(&a.id).is_none(), "a settled finished job made room");
        assert_eq!(m.list().len(), 2);
        assert_eq!(m.wait("missing", Duration::ZERO).map(|j| j.id), None);
    }

    #[test]
    fn incognito_jobs_stay_out_of_lists_and_leave_nothing_behind() {
        let m = Media::new();
        let dir = std::env::temp_dir().join(format!("nrob-studio-incognito-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::write(dir.join("images").join("a.png"), b"x").unwrap();
        let job = m.submit(Kind::Image, body(r#"{"prompt":"secret"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 10, Some(dir.clone()));
        assert!(job.incognito);
        assert!(m.list().is_empty(), "incognito jobs are not listed");
        assert!(m.get(&job.id).is_some(), "but their owner can poll them by id");
        let private = m.private_activity();
        assert_eq!(private.len(), 1, "the UI still learns that something runs");
        assert!(private.at(0).unwrap().get("prompt").is_none(), "but not what");
        // Cancelled while queued, it never reached the runner that stamps an
        // expiry; the next sweep takes it anyway.
        m.cancel(&job.id);
        assert!(m.get(&job.id).unwrap().expires_at.is_none());
        m.sweep();
        assert!(m.get(&job.id).is_none());
        assert!(!dir.exists(), "its folder is gone");
        // Deleting a running job forgets it once it stops.
        let run = m.submit(Kind::Image, body(r#"{"prompt":"x"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 10, None);
        m.update(&run.id, |j| j.status = "in_progress".into());
        assert!(m.remove(&run.id));
        m.update(&run.id, |j| j.status = "cancelled".into());
        m.sweep();
        assert!(m.get(&run.id).is_none());
    }

    #[test]
    fn chat_leases_wait_for_exclusive_media_and_time_out() {
        let m = Media::new();
        {
            let _a = m.chat_lease(true, Duration::ZERO).unwrap();
            assert_eq!(m.broker.lock().unwrap().chats, 1);
        }
        assert_eq!(m.broker.lock().unwrap().chats, 0);
        m.broker.lock().unwrap().media = true;
        assert!(m.chat_lease(true, Duration::from_millis(20)).is_err());
        assert!(m.chat_lease(false, Duration::ZERO).is_ok(), "coexisting devices never wait");
    }
}
