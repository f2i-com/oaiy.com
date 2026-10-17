//! The models loaded on each GPU, as one view (`GET /api/resources`: [`snapshot`]), and what loaded and let them go
//! ([`event`]). The studio's own parts hold the models and decide when they come and go: the language model's engine
//! on this computer (`crate::llm`), the eGPU's (`crate::llm::egpu`), and the media queue with the picture worker it
//! keeps (`crate::media`). This reads them all at once, each part's figures beside the memory each GPU has in use, so a
//! flow, a client or a person sees where everything is, what is using it now (its leases: requests answered, a job
//! running) and what may be let go.
//!
//! The policy those parts follow, on one GPU at a time (the card's own server holds one process, and 24 GB does not
//! hold a language model and a picture model beside each other):
//!
//! - A media job takes its GPU: on a Mac's eGPU the card is lent to it (its language model let go and brought back
//!   after), on this computer's GPU a language model sharing it is stopped first. Never one with a lease (a chat being
//!   answered finishes first), and never a pinned one (`llm.pinned`): media then go to this computer's GPU, or run
//!   beside it.
//! - One picture worker is kept between pictures, with the models it loaded, for `media.keep_minutes` (2 unless set),
//!   or until a chat wants its GPU, a job asks for its models to be let go after it (`release`), or another worker needs
//!   the GPU. A job that ran out of memory is tried once more with everything idle let go.
//! - One media job at a time (a GPU at a time), queued in order up to [`crate::media::MAX_QUEUED`]; a job is cancelled
//!   with `POST /api/jobs/{id}/cancel`.
use crate::util::{bool_or, str_or};
use crate::Studio;
use oaiy_engine::json::Json;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;

/// The events kept (the oldest let go first).
const EVENTS_KEPT: usize = 200;

static EVENTS: Mutex<VecDeque<Json>> = Mutex::new(VecDeque::new());

/// A model loaded or let go: `what` (`load`, `unload`, `evict`, `lend`, `give_back`, `retry`, `release`), the model, its
/// device (`egpu`, `gpu{n}`), and why.
pub fn event(what: &str, model: &str, device: &str, why: &str) {
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
    let mut ring = EVENTS.lock().unwrap_or_else(|p| p.into_inner());
    if ring.len() >= EVENTS_KEPT {
        ring.pop_front();
    }
    ring.push_back(Json::obj([("at_ms", Json::Int(at)), ("event", Json::str(what)), ("model", Json::str(model)), ("device", Json::str(device)), ("why", Json::str(why))]));
}

/// The events since `after_ms` (all where 0), oldest first.
pub fn events(after_ms: i64) -> Vec<Json> {
    EVENTS.lock().unwrap_or_else(|p| p.into_inner()).iter().filter(|e| e.get("at_ms").and_then(Json::as_i64).unwrap_or(0) > after_ms).cloned().collect()
}

/// The megabytes a model's files hold: a file's size, or a folder's files' (a few levels down), its weights' alone.
fn size_mb(path: &str) -> Option<i64> {
    fn walk(p: &Path, depth: usize) -> u64 {
        match std::fs::metadata(p) {
            Ok(m) if m.is_file() => {
                let weights = p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "safetensors" | "gguf" | "bin" | "pth"));
                if weights { m.len() } else { 0 }
            }
            Ok(m) if m.is_dir() && depth > 0 => std::fs::read_dir(p).map_or(0, |rd| rd.flatten().map(|e| walk(&e.path(), depth - 1)).sum()),
            _ => 0,
        }
    }
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    let bytes = walk(Path::new(path), 3);
    (bytes > 0).then_some((bytes >> 20) as i64)
}

/// A process's resident memory (MB), where the system says it (`ps`; none on Windows).
pub fn rss_mb(pid: u32) -> Option<i64> {
    #[cfg(unix)]
    {
        let out = std::process::Command::new("/bin/ps").args(["-o", "rss=", "-p", &pid.to_string()]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse::<i64>().ok().map(|kb| kb >> 10)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        None
    }
}

/// The weights' megabytes of the model the settings name `name` in `section` (`llm`, or a media section).
fn model_mb(cfg: &Json, section: &str, name: &str) -> Option<i64> {
    if section == "llm" {
        let m = cfg.get("llm")?.get("models")?.as_array()?.iter().find(|m| str_or(m, "name", "") == name)?;
        return size_mb(str_or(m, "path", ""));
    }
    let m = cfg.get("media")?.get(section)?.get("models")?.get(name)?;
    let parts: Vec<i64> = ["transformer", "text_encoder", "vae", "distill", "base", "checkpoint", "path", "safetensors_transformer"]
        .iter()
        .filter_map(|k| m.get(k).and_then(Json::as_str))
        .filter_map(size_mb)
        .collect();
    (!parts.is_empty()).then(|| parts.iter().sum())
}

/// Everything loaded, where, and what is using it: `{devices, models, queue, policy, events}`.
pub fn snapshot(studio: &Studio) -> Json {
    let cfg = studio.config();
    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    let pinned = bool_or(&llm, "pinned", false);
    let mut devices: Vec<Json> = studio.system.gpus().as_array().unwrap_or(&[]).iter().map(|g| {
        Json::obj([
            ("id", Json::str(format!("gpu{}", g.get("index").and_then(Json::as_i64).unwrap_or(0)))),
            ("name", g.get("name").cloned().unwrap_or(Json::Null)),
            ("unified", Json::Bool(bool_or(g, "unified", false))),
            ("memory_total_mb", g.get("memory_total_mb").cloned().unwrap_or(Json::Null)),
            ("memory_used_mb", g.get("memory_used_mb").cloned().unwrap_or(Json::Null)),
            ("load", g.get("utilization").cloned().unwrap_or(Json::Null)),
        ])
    }).collect();
    let card = studio.egpu.status(&llm);
    if bool_or(&card, "available", false) && bool_or(&card, "enabled", false) {
        let usage = studio.egpu.card_usage();
        devices.push(Json::obj([
            ("id", Json::str("egpu")),
            ("name", Json::str(format!("{} card (eGPU)", str_or(&card, "device", "NVIDIA")))),
            ("unified", Json::Bool(false)),
            ("memory_total_mb", usage.get("memory_total_mb").cloned().unwrap_or(Json::Null)),
            ("memory_used_mb", usage.get("memory_used_mb").cloned().unwrap_or(Json::Null)),
            ("memory_cached_mb", usage.get("memory_cached_mb").cloned().unwrap_or(Json::Null)),
            ("load", usage.get("load").cloned().unwrap_or(Json::Null)),
        ]));
    }
    let mut models = Vec::new();
    // the eGPU's language model: in use while it answers (its leases), let go while the card is lent; none where the
    // card's server was started only to lend the card to pictures (paused from the start, nothing loaded)
    let lent = bool_or(&card, "lent", false) || bool_or(&card, "paused", false);
    let never_loaded = lent && card.get("load_seconds").is_none_or(|s| matches!(s, Json::Null));
    if let Some(name) = card.get("model").and_then(Json::as_str).filter(|_| str_or(&card, "state", "stopped") != "stopped" && !never_loaded) {
        models.push(Json::obj([
            ("model", Json::str(name)),
            ("kind", Json::str("language")),
            ("device", Json::str("egpu")),
            ("state", Json::str(if lent { "lent" } else { str_or(&card, "state", "") })),
            ("leases", card.get("busy").cloned().unwrap_or(Json::Int(0))),
            ("pinned", Json::Bool(pinned)),
            ("estimated_mb", model_mb(&cfg, "llm", name).map_or(Json::Null, Json::Int)),
            ("load_seconds", card.get("load_seconds").cloned().unwrap_or(Json::Null)),
            ("idle_seconds", card.get("idle_seconds").cloned().unwrap_or(Json::Null)),
        ]));
    }
    // this computer's language model engine
    let local = studio.llm.status();
    if str_or(&local, "state", "stopped") != "stopped" {
        let name = local.get("resident").and_then(Json::as_str).or_else(|| local.get("models").and_then(|m| m.at(0)).and_then(Json::as_str)).unwrap_or("").to_string();
        models.push(Json::obj([
            ("model", Json::str(&name)),
            ("kind", Json::str("language")),
            ("device", Json::str(local.get("runs_on").and_then(Json::as_str).unwrap_or("this computer"))),
            ("state", local.get("state").cloned().unwrap_or(Json::Null)),
            ("leases", Json::Int(studio.media.chats() as i64)),
            ("pinned", Json::Bool(pinned)),
            ("estimated_mb", model_mb(&cfg, "llm", &name).map_or(Json::Null, Json::Int)),
            ("load_seconds", local.get("load_seconds").cloned().unwrap_or(Json::Null)),
            ("idle_seconds", local.get("idle_seconds").cloned().unwrap_or(Json::Null)),
        ]));
    }
    // the picture worker kept between pictures, and the job running (its lease)
    let keep = crate::media::keep_for(&cfg).as_secs_f64();
    let running = studio.media.running_job();
    // (each model the kept worker holds: its process's memory shared among them)
    let pins = crate::media::pins::pinned(&cfg);
    if let Some(k) = studio.media.kept_worker() {
        let busy_with = running.as_ref().map(|r| str_or(r, "model", "").to_string());
        for model in &k.models {
            let busy = busy_with.as_deref() == Some(model.as_str());
            let pinned = pins.contains(model);
            models.push(Json::obj([
                ("model", Json::str(model)),
                ("kind", Json::str("image")),
                ("device", Json::str(&k.device)),
                ("state", Json::str(if busy { "in_use" } else { "kept" })),
                ("leases", Json::Int(busy as i64)),
                ("pinned", Json::Bool(pinned)),
                ("estimated_mb", model_mb(&cfg, "image", model).map_or(Json::Null, Json::Int)),
                ("resident_ram_mb", rss_mb(k.pid).map_or(Json::Null, Json::Int)),
                ("idle_seconds", Json::Num((k.idle_seconds * 10.).round() / 10.)),
                // (a pinned model's worker is kept, whatever the keep time)
                ("keep_seconds_left", if busy_with.is_some() || pinned { Json::Null } else { Json::Num(((keep - k.idle_seconds).max(0.) * 10.).round() / 10.) }),
            ]));
        }
    }
    if let Some(r) = running.as_ref().filter(|r| studio.media.kept_worker().is_none_or(|k| !k.models.iter().any(|m| m == str_or(r, "model", "")))) {
        models.push(Json::obj([
            ("model", r.get("model").cloned().unwrap_or(Json::Null)),
            ("kind", r.get("kind").cloned().unwrap_or(Json::Null)),
            ("device", r.get("device").cloned().unwrap_or(Json::Null)),
            ("state", Json::str("in_use")),
            ("leases", Json::Int(1)),
            ("pinned", Json::Bool(false)),
        ]));
    }
    Json::obj([
        ("devices", Json::Arr(devices)),
        ("models", Json::Arr(models)),
        ("queue", Json::obj([
            ("running", running.unwrap_or(Json::Null)),
            ("queued", Json::Int(studio.media.queued() as i64)),
            ("max_queued", Json::Int(crate::media::MAX_QUEUED as i64)),
        ])),
        ("policy", Json::obj([
            ("keep_minutes", Json::Num(keep / 60.)),
            ("llm_pinned", Json::Bool(pinned)),
            ("one_job_at_a_time", Json::Bool(true)),
        ])),
        // (the pinned picture models and where each stands: `crate::media::pins`)
        ("pins", crate::media::pins::snapshot(&cfg, &studio.media)),
        ("events", Json::Arr(events(0).into_iter().rev().take(50).collect::<Vec<_>>().into_iter().rev().collect())),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_kept_in_order_and_the_oldest_let_go() {
        let before = events(0).len();
        event("load", "m", "egpu", "a test");
        let all = events(0);
        assert_eq!(all.len(), (before + 1).min(EVENTS_KEPT));
        let last = all.last().unwrap();
        assert_eq!((str_or(last, "event", ""), str_or(last, "model", ""), str_or(last, "why", "")), ("load", "m", "a test"));
        for i in 0..EVENTS_KEPT + 5 {
            event("unload", &format!("m{i}"), "gpu0", "many");
        }
        let all = events(0);
        assert_eq!(all.len(), EVENTS_KEPT);
        assert_eq!(str_or(all.last().unwrap(), "model", ""), format!("m{}", EVENTS_KEPT + 4));
    }

    #[test]
    fn a_models_size_is_its_weights() {
        let d = std::env::temp_dir().join(format!("oaiy-resources-{}", std::process::id()));
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("a.safetensors"), vec![0u8; 3 << 20]).unwrap();
        std::fs::write(d.join("sub").join("b.gguf"), vec![0u8; 2 << 20]).unwrap();
        std::fs::write(d.join("README.md"), vec![0u8; 5 << 20]).unwrap();
        assert_eq!(size_mb(d.to_str().unwrap()), Some(5));
        assert_eq!(size_mb(d.join("a.safetensors").to_str().unwrap()), Some(3));
        assert_eq!(size_mb(""), None);
        let _ = std::fs::remove_dir_all(&d);
    }
}
