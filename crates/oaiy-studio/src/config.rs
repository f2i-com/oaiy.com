//! The studio's one configuration file, `oaiy-studio.json`, kept beside the
//! executable so a copied folder is a complete, portable install.
//!
//! The configuration stays a JSON tree (the UI edits it whole and saves it back);
//! [`validate`] is the contract every save passes, and the typed getters below
//! read what the supervisors need from it. Relative paths are relative to the
//! file's own directory.

use crate::util::{bool_or, int_or, str_or};
use oaiy_engine::json::Json;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FILE_NAME: &str = "oaiy-studio.json";

/// What a gateway route serves.
pub const TARGETS: [&str; 16] = ["chat", "completions", "models", "images", "edits", "videos", "health", "files", "discovery", "speech", "voices", "music", "sound", "model3d", "background", "upscale"];
/// The request/response dialect a route speaks. `openai` is the OpenAI API;
/// `OAIY` is oaiy-llm-server's own asynchronous media job API (what coder-cli uses).
pub const SPECS: [&str; 2] = ["openai", "oaiy"];
pub const MEMORY: [&str; 4] = ["auto", "gpu", "ram", "ssd"];
pub const VIDEO_FAMILIES: [&str; 3] = ["ltx-2.3", "ltx-2.5", "sulphur-2"];
pub const IMAGE_ARCHITECTURES: [&str; 3] = ["qwen-image", "sdxl", "flux2-klein-4b"];
/// How media jobs share GPUs with the LLM: `auto` pauses the LLM only when the
/// media device is one of its devices; `pause_llm` always; `coexist` never.
pub const LLM_POLICIES: [&str; 3] = ["auto", "pause_llm", "coexist"];

pub const DEFAULT: &str = r#"{
  "ui": { "host": "127.0.0.1", "port": 7860, "open": "app" },
  "gateway": {
    "host": "127.0.0.1",
    "port": 8080,
    "api_key": "",
    "public_url": "",
    "cors_origins": ["https://bot.computer", "http://localhost:5317", "http://127.0.0.1:5317", "http://botcomputer.localhost", "botcomputer://localhost", "http://oaiy.localhost", "oaiy://localhost"],
    "origins_version": 2,
    "routes_version": 8,
    "routes": [
      { "path": "/v1/chat/completions", "method": "POST", "target": "chat", "spec": "openai", "enabled": true },
      { "path": "/v1/completions", "method": "POST", "target": "completions", "spec": "openai", "enabled": true },
      { "path": "/v1/models", "method": "GET", "target": "models", "spec": "openai", "enabled": true },
      { "path": "/v1/images/generations", "method": "POST", "target": "images", "spec": "openai", "enabled": true },
      { "path": "/v1/images/edits", "method": "POST", "target": "edits", "spec": "openai", "enabled": true },
      { "path": "/v1/videos", "method": "POST", "target": "videos", "spec": "openai", "enabled": true },
      { "path": "/v1/audio/speech", "method": "POST", "target": "speech", "spec": "openai", "enabled": true },
      { "path": "/v1/audio/voices", "method": "GET", "target": "voices", "spec": "openai", "enabled": true },
      { "path": "/v1/audio/music", "method": "POST", "target": "music", "spec": "openai", "enabled": true },
      { "path": "/v1/audio/sound_effects", "method": "POST", "target": "sound", "spec": "openai", "enabled": true },
      { "path": "/v1/3d/models", "method": "POST", "target": "model3d", "spec": "openai", "enabled": true },
      { "path": "/v1/images/background_removal", "method": "POST", "target": "background", "spec": "openai", "enabled": true },
      { "path": "/v1/images/upscale", "method": "POST", "target": "upscale", "spec": "openai", "enabled": true },
      { "path": "/files", "method": "GET", "target": "files", "spec": "openai", "enabled": true },
      { "path": "/health", "method": "GET", "target": "health", "spec": "openai", "enabled": true },
      { "path": "/v1/discovery", "method": "GET", "target": "discovery", "spec": "openai", "enabled": true }
    ]
  },
  "privacy": { "incognito": false },
  "downloads": { "dir": "", "hf_token": "", "curl": "" },
  "llm": {
    "enabled": true,
    "autostart": false,
    "server": "oaiy-llm-server",
    "server_webgpu": "oaiy-llm-server-webgpu",
    "backend": "auto",
    "webgpu_gb": null,
    "default_model": "",
    "models": [],
    "devices": [],
    "ctx": 0,
    "ram_gb": 0,
    "cpu_threads": null,
    "vram_headroom_gb": 2,
    "thinking": false,
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "prompt_cache": true,
    "prompt_cache_gb": 4,
    "park_gb": null,
    "vision": true,
    "idle_stop_minutes": 0,
    "extra_args": []
  },
  "media": {
    "worker": "oaiy-media",
    "output_dir": "outputs",
    "device": 0,
    "llm_policy": "auto",
    "resume_llm": true,
    "keep_jobs": 200,
    "image": {
      "enabled": true,
      "default_model": "",
      "memory": "auto",
      "ram_gb": 32,
      "vram_gb": null,
      "models": {}
    },
    "video": {
      "enabled": true,
      "default_model": "",
      "memory": "auto",
      "ram_gb": 48,
      "vram_gb": null,
      "ffmpeg": "ffmpeg",
      "fps": 24,
      "models": {}
    },
    "speech": {
      "enabled": true,
      "default_model": "",
      "voices_dir": "voices",
      "models": {}
    },
    "music": {
      "enabled": true,
      "default_model": "",
      "memory": "auto",
      "ram_gb": 48,
      "vram_gb": null,
      "models": {}
    },
    "sound": {
      "enabled": true,
      "default_model": "",
      "models": {}
    },
    "model3d": {
      "enabled": true,
      "default_model": "",
      "models": {}
    },
    "picture": {
      "enabled": true,
      "background": "",
      "upscaler": ""
    }
  }
}"#;

pub fn default_json() -> Json {
    Json::parse(DEFAULT.as_bytes()).expect("the built-in default configuration is valid JSON")
}

/// Fill anything `v` lacks from `defaults`, recursively for objects. Arrays and
/// scalars that are present are the user's and are kept as they are.
pub fn merge_defaults(v: &mut Json, defaults: &Json) {
    let (Json::Obj(fields), Json::Obj(base)) = (v, defaults) else { return };
    for (key, default) in base {
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, value)) => merge_defaults(value, default),
            None => fields.push((key.clone(), default.clone())),
        }
    }
}

/// Routes for targets added after a file was written (`routes_version` records
/// which it has seen). Returns whether anything changed.
pub fn upgrade_routes(v: &mut Json) -> bool {
    /// The target each routes_version introduced.
    const ADDED: [(i64, &str); 9] = [(2, "edits"), (3, "discovery"), (4, "speech"), (4, "voices"), (5, "music"), (6, "sound"), (7, "model3d"), (8, "background"), (8, "upscale")];
    const VERSION: i64 = 8;
    let Json::Obj(top) = v else { return false };
    let Some((_, gateway)) = top.iter_mut().find(|(k, _)| k == "gateway") else { return false };
    let from = int_or(gateway, "routes_version", 1);
    if from >= VERSION {
        return false;
    }
    let defaults = default_json();
    let new_targets: Vec<&str> = ADDED.iter().filter(|(v, _)| *v > from).map(|(_, t)| *t).collect();
    let added: Vec<Json> = defaults.get("gateway").and_then(|g| g.get("routes")).and_then(Json::as_array).unwrap_or(&[]).iter()
        .filter(|r| new_targets.contains(&str_or(r, "target", ""))).cloned().collect();
    if let Json::Obj(fields) = gateway {
        if let Some((_, Json::Arr(routes))) = fields.iter_mut().find(|(k, _)| k == "routes") {
            for route in added {
                let target = str_or(&route, "target", "").to_string();
                let taken = routes.iter().any(|r| str_or(r, "target", "") == target || str_or(r, "path", "") == str_or(&route, "path", ""));
                if !taken {
                    routes.push(route);
                }
            }
        }
        match fields.iter_mut().find(|(k, _)| k == "routes_version") {
            Some((_, n)) => *n = Json::Int(VERSION),
            None => fields.push(("routes_version".into(), Json::Int(VERSION))),
        }
    }
    true
}

/// Browser apps allowed by default (`gateway.cors_origins`): bot.computer on
/// the web, served on this machine (`npm start`, port 5317), and as the desktop
/// app (its own scheme: on Windows, then on macOS and Linux); and the agent in
/// OAIY's own window (its `oaiy` scheme).
pub const COMPANION_ORIGINS: [&str; 7] = ["https://bot.computer", "http://localhost:5317", "http://127.0.0.1:5317", "http://botcomputer.localhost", "botcomputer://localhost", "http://oaiy.localhost", "oaiy://localhost"];

/// The agent in OAIY's own window: offered once to files from before version 3.
const OAIY_WINDOW_ORIGINS: [&str; 2] = ["http://oaiy.localhost", "oaiy://localhost"];

/// The desktop app's origins in the first `origins_version`: Tauri's shared
/// ones, which every Tauri app has. Version 2 names bot.computer's own scheme.
const SHARED_TAURI_ORIGINS: [(&str, &str); 2] = [("http://tauri.localhost", "http://botcomputer.localhost"), ("tauri://localhost", "botcomputer://localhost")];

/// Companion origins for files written before they were defaults
/// (`origins_version` records that they were offered once, so an origin the
/// user removes stays removed). A file at version 1 has Tauri's shared origins
/// swapped for the desktop app's own; one before version 3 gains OAIY's window.
/// Returns whether anything changed.
pub fn upgrade_origins(v: &mut Json) -> bool {
    const VERSION: i64 = 3;
    let Json::Obj(top) = v else { return false };
    let Some((_, gateway)) = top.iter_mut().find(|(k, _)| k == "gateway") else { return false };
    let from = int_or(gateway, "origins_version", 0);
    if from >= VERSION {
        return false;
    }
    let Json::Obj(fields) = gateway else { return false };
    let has = |origins: &[Json], origin: &str| origins.iter().any(|o| o.as_str().is_some_and(|s| s.trim_end_matches('/').eq_ignore_ascii_case(origin)));
    // A file without the list gets the defaults' list when they are filled in.
    if let Some((_, Json::Arr(origins))) = fields.iter_mut().find(|(k, _)| k == "cors_origins") {
        if !origins.iter().any(|o| o.as_str() == Some("*")) {
            if from == 0 {
                for origin in COMPANION_ORIGINS {
                    if !has(origins, origin) {
                        origins.push(Json::str(origin));
                    }
                }
            } else {
                if from == 1 {
                    for (shared, own) in SHARED_TAURI_ORIGINS {
                        let at = origins.iter().position(|o| o.as_str().is_some_and(|s| s.trim_end_matches('/').eq_ignore_ascii_case(shared)));
                        if let Some(i) = at {
                            if has(origins, own) {
                                origins.remove(i);
                            } else {
                                origins[i] = Json::str(own);
                            }
                        }
                    }
                }
                for origin in OAIY_WINDOW_ORIGINS {
                    if !has(origins, origin) {
                        origins.push(Json::str(origin));
                    }
                }
            }
        }
    }
    match fields.iter_mut().find(|(k, _)| k == "origins_version") {
        Some((_, n)) => *n = Json::Int(VERSION),
        None => fields.push(("origins_version".into(), Json::Int(VERSION))),
    }
    true
}

/// Where the configuration lives: `--config`, else beside the executable.
pub fn default_path() -> PathBuf {
    exe_dir().join(FILE_NAME)
}

pub fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Read the file (creating it with defaults when it does not exist yet).
pub fn load(path: &Path) -> Result<Json, String> {
    if !path.exists() {
        let v = default_json();
        save(path, &v)?;
        return Ok(v);
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut v = Json::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    // Before the defaults fill in, so a file without `routes_version` is seen as old.
    let upgraded = upgrade_routes(&mut v);
    let upgraded = upgrade_origins(&mut v) || upgraded;
    merge_defaults(&mut v, &default_json());
    if upgraded {
        save(path, &v)?;
    }
    validate(&v).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(v)
}

/// Write atomically: a new file beside it, then a rename.
///
/// The file holds secrets (`gateway.api_key` and `downloads.hf_token`), so on unix the new file is
/// created owner-only (mode 0600) by the call that creates it, and the rename carries that to the
/// real file: nothing is written to a file the whole machine can read and narrowed afterwards. On
/// Windows there is no mode to set, and the file keeps the access rights of the folder it is in.
///
/// The name is this process's own (`.<file>.<pid>-<n>.tmp`), so two savers cannot share one. A name
/// that is taken (the leftover of a killed process) is skipped, never reused, and `create_new` refuses
/// to follow a link put there. The file is removed again if the save fails.
pub fn save(path: &Path, v: &Json) -> Result<(), String> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().ok_or_else(|| format!("{}: no file name", path.display()))?.to_string_lossy().into_owned();
    let (tmp, mut file) = create_private(dir, &name)?;
    let staged = file.write_all(pretty(v, 0).as_bytes()).and_then(|()| file.sync_all());
    drop(file);
    let saved = staged
        .map_err(|e| format!("{}: {e}", tmp.display()))
        .and_then(|()| std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display())));
    if saved.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    saved
}

/// A new file in `dir` for a save of `name`, readable and writable by its owner only (unix).
fn create_private(dir: &Path, name: &str) -> Result<(PathBuf, std::fs::File), String> {
    static STAGED: AtomicU64 = AtomicU64::new(0);
    create_private_numbered(dir, name, || STAGED.fetch_add(1, Ordering::Relaxed))
}

/// [`create_private`] with the numbers for the names from `next`, which the tests choose.
fn create_private_numbered(dir: &Path, name: &str, mut next: impl FnMut() -> u64) -> Result<(PathBuf, std::fs::File), String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    for _ in 0..32 {
        let tmp = dir.join(format!(".{name}.{}-{}.tmp", std::process::id(), next()));
        match options.open(&tmp) {
            Ok(file) => return Ok((tmp, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", tmp.display())),
        }
    }
    Err(format!("{}: every name for a temporary file is taken", dir.display()))
}

/// Indented JSON, so the file stays pleasant to edit by hand.
pub fn pretty(v: &Json, depth: usize) -> String {
    let pad = "  ".repeat(depth + 1);
    let end = "  ".repeat(depth);
    match v {
        Json::Arr(items) if !items.is_empty() && items.iter().any(|i| matches!(i, Json::Obj(_) | Json::Arr(_))) => {
            let inner: Vec<_> = items.iter().map(|i| format!("{pad}{}", pretty(i, depth + 1))).collect();
            format!("[\n{}\n{end}]", inner.join(",\n"))
        }
        Json::Obj(fields) if !fields.is_empty() => {
            let inner: Vec<_> = fields
                .iter()
                .map(|(k, v)| format!("{pad}{}: {}", Json::str(k).to_json(), pretty(v, depth + 1)))
                .collect();
            format!("{{\n{}\n{end}}}", inner.join(",\n"))
        }
        other => other.to_json(),
    }
}

fn object<'a>(v: &'a Json, key: &str) -> Result<&'a Json, String> {
    v.get(key).filter(|v| v.as_object().is_some()).ok_or_else(|| format!("{key} must be an object"))
}

fn port(v: &Json, what: &str) -> Result<(), String> {
    match v.get("port").and_then(Json::as_i64) {
        Some(p) if (0..=65535).contains(&p) => Ok(()),
        _ => Err(format!("{what}.port must be 0..65535")),
    }
}

fn gib(v: &Json, key: &str, what: &str, max: i64) -> Result<(), String> {
    match v.get(key) {
        None | Some(Json::Null) => Ok(()),
        Some(n) => match n.as_i64() {
            Some(n) if (0..=max).contains(&n) => Ok(()),
            _ => Err(format!("{what}.{key} must be a whole number of GiB in 0..{max}, or null")),
        },
    }
}

fn memory(v: &Json, what: &str) -> Result<(), String> {
    match v.get("memory") {
        None | Some(Json::Null) => Ok(()),
        Some(m) if m.as_str().is_some_and(|m| MEMORY.contains(&m)) => Ok(()),
        _ => Err(format!("{what}.memory must be one of {}", MEMORY.join(", "))),
    }
}

/// The contract every configuration (loaded or saved from the UI) must meet.
pub fn validate(v: &Json) -> Result<(), String> {
    let ui = object(v, "ui")?;
    port(ui, "ui")?;
    let gateway = object(v, "gateway")?;
    port(gateway, "gateway")?;
    let loopback = |host: &str| host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    // The control port can change which programs run: reachable from other
    // machines only behind the key (DNS rebinding defeats a name check alone).
    if !loopback(str_or(ui, "host", "127.0.0.1")) && str_or(gateway, "api_key", "").is_empty() {
        return Err("set gateway.api_key before serving the control UI beyond this machine (ui.host)".into());
    }
    if !gateway.get("cors_origins").is_none_or(|o| o.as_array().is_some_and(|a| a.iter().all(|x| x.as_str().is_some()))) {
        return Err("gateway.cors_origins must be a list of origins such as http://localhost:3000, or \"*\"".into());
    }
    if v.get("privacy").is_some_and(|p| p.as_object().is_none() || p.get("incognito").is_some_and(|i| i.as_bool().is_none())) {
        return Err("privacy.incognito must be true or false".into());
    }
    if int_or(ui, "port", 0) != 0 && int_or(ui, "port", 0) == int_or(gateway, "port", 1) && str_or(ui, "host", "") == str_or(gateway, "host", "") {
        return Err("the UI and the gateway need different ports".into());
    }
    let routes = gateway.get("routes").and_then(Json::as_array).ok_or("gateway.routes must be an array")?;
    let mut seen = Vec::new();
    for (i, r) in routes.iter().enumerate() {
        let what = format!("gateway.routes[{i}]");
        let path = r.get("path").and_then(Json::as_str).unwrap_or("");
        if !path.starts_with('/') || path.len() > 200 || path.contains(['?', '#', ' ', '{', '}']) {
            return Err(format!("{what}.path must start with / and hold no spaces, braces, ? or #"));
        }
        let method = str_or(r, "method", "POST").to_ascii_uppercase();
        if !["GET", "POST", "PUT", "DELETE"].contains(&method.as_str()) {
            return Err(format!("{what}.method must be GET, POST, PUT or DELETE"));
        }
        let target = str_or(r, "target", "");
        if !TARGETS.contains(&target) {
            return Err(format!("{what}.target must be one of {}", TARGETS.join(", ")));
        }
        let spec = str_or(r, "spec", "openai");
        if !SPECS.contains(&spec) {
            return Err(format!("{what}.spec must be one of {}", SPECS.join(", ")));
        }
        if spec == "oaiy" && !["images", "videos"].contains(&target) {
            return Err(format!("{what}: the OAIY spec applies to images and videos only"));
        }
        if bool_or(r, "enabled", true) {
            let key = (method, path.trim_end_matches('/').to_string());
            if seen.contains(&key) {
                return Err(format!("{what}: {} {} is routed twice", key.0, key.1));
            }
            seen.push(key);
        }
    }
    let llm = object(v, "llm")?;
    let models = llm.get("models").and_then(Json::as_array).ok_or("llm.models must be an array")?;
    let mut names = Vec::new();
    for (i, m) in models.iter().enumerate() {
        let name = str_or(m, "name", "");
        if name.is_empty() || name.len() > 100 || name.contains(['=', ' ', '/', '\\']) {
            return Err(format!("llm.models[{i}].name must be a short name without spaces, = or slashes"));
        }
        if names.contains(&name) {
            return Err(format!("llm model {name} is listed twice"));
        }
        names.push(name);
        if str_or(m, "path", "").trim().is_empty() {
            return Err(format!("llm model {name} needs a path (a .gguf, a folder holding one, or a checkpoint folder)"));
        }
    }
    if !["auto", "cuda", "webgpu", "cpu"].contains(&str_or(llm, "backend", "auto")) {
        return Err("llm.backend must be auto, cuda, webgpu or cpu".into());
    }
    gib(llm, "webgpu_gb", "llm", 1024)?;
    // Host RAM for the conversations Flash-Next sets aside; null (or absent) leaves the server's own default.
    if let Some(gb) = llm.get("park_gb").filter(|g| !matches!(g, Json::Null)) {
        if !gb.as_f64().is_some_and(|g| g.is_finite() && (0.0..=1024.0).contains(&g)) {
            return Err("llm.park_gb must be a number of GB in 0..1024, or null".into());
        }
    }
    let default = str_or(llm, "default_model", "");
    if !default.is_empty() && !names.contains(&default) {
        return Err(format!("llm.default_model {default} is not one of the listed models"));
    }
    if !llm.get("devices").and_then(Json::as_array).is_some_and(|d| d.iter().all(|d| d.as_i64().is_some_and(|d| (0..64).contains(&d)))) {
        return Err("llm.devices must be a list of GPU indices".into());
    }
    // A model's own GPUs (it runs there instead of on llm.devices).
    for m in llm.get("models").and_then(Json::as_array).unwrap_or(&[]) {
        if m.get("devices").is_some_and(|d| !matches!(d, Json::Null) && !d.as_array().is_some_and(|d| d.iter().all(|d| d.as_i64().is_some_and(|d| (0..64).contains(&d))))) {
            return Err(format!("model {}: devices must be a list of GPU indices", str_or(m, "name", "")));
        }
    }
    for (key, min, max) in [("ctx", 512, 1 << 20), ("max_tokens", 1, 1 << 20), ("cpu_threads", 0, 1024), ("ram_gb", 0, 4096)] {
        // `cpu_threads: null` means "this machine's core count".
        if key == "cpu_threads" && matches!(llm.get(key), Some(Json::Null)) {
            continue;
        }
        // `ctx: 0` means "the most the model allows".
        if key == "ctx" && llm.get(key).and_then(Json::as_i64) == Some(0) {
            continue;
        }
        if !llm.get(key).and_then(Json::as_i64).is_some_and(|n| (min..=max).contains(&n)) {
            return Err(format!("llm.{key} must be a whole number in {min}..{max}"));
        }
    }
    let media = object(v, "media")?;
    if !media.get("device").and_then(Json::as_i64).is_some_and(|d| (0..64).contains(&d)) {
        return Err("media.device must be a GPU index".into());
    }
    if !LLM_POLICIES.contains(&str_or(media, "llm_policy", "auto")) {
        return Err(format!("media.llm_policy must be one of {}", LLM_POLICIES.join(", ")));
    }
    let image = object(media, "image")?;
    memory(image, "media.image")?;
    gib(image, "ram_gb", "media.image", 512)?;
    gib(image, "vram_gb", "media.image", 192)?;
    let image_models = object(image, "models")?;
    for (name, m) in image_models.members() {
        let arch = str_or(m, "architecture", "qwen-image");
        if !IMAGE_ARCHITECTURES.contains(&arch) {
            return Err(format!("image model {name}: architecture must be one of {}", IMAGE_ARCHITECTURES.join(", ")));
        }
        // A disabled entry may be incomplete: a picked file waiting for its parts.
        if !bool_or(m, "enabled", true) {
            continue;
        }
        let required: &[&str] = match arch {
            "sdxl" => &["checkpoint", "tokenizer"],
            "flux2-klein-4b" => &["transformer", "text_encoder", "vae", "tokenizer"],
            _ => &["base"],
        };
        for key in required {
            if str_or(m, key, "").trim().is_empty() {
                return Err(format!("image model {name} needs {key}"));
            }
        }
        if arch == "qwen-image" && str_or(m, "transformer", "").is_empty() && str_or(m, "safetensors_transformer", "").is_empty() {
            return Err(format!("image model {name} needs a transformer (a .gguf or a .safetensors checkpoint)"));
        }
        if arch == "flux2-klein-4b" {
            let variant=str_or(m,"variant","distilled");
            if !["distilled","base"].contains(&variant) { return Err(format!("image model {name}: Klein variant must be distilled or base")); }
            let distilled=variant=="distilled";
            let steps=match m.get("steps") { None=>if distilled {4}else{50}, Some(v)=>v.as_i64().ok_or_else(||format!("image model {name}: steps must be an integer"))? };
            let cfg=match m.get("cfg") { None=>if distilled {1.0}else{4.0}, Some(v)=>v.as_f64().ok_or_else(||format!("image model {name}: cfg must be numeric"))? };
            if !(1..=100).contains(&steps) || !cfg.is_finite() || !(1.0..=10.0).contains(&cfg) || (distilled&&(steps!=4||cfg!=1.0)) { return Err(format!("image model {name}: distilled Klein requires four steps and cfg 1; base cfg must be 1..10")); }
        }
        memory(m, &format!("image model {name}"))?;
    }
    let video = object(media, "video")?;
    memory(video, "media.video")?;
    gib(video, "ram_gb", "media.video", 512)?;
    gib(video, "vram_gb", "media.video", 192)?;
    for (name, m) in object(video, "models")?.members() {
        let family = str_or(m, "family", name);
        if !VIDEO_FAMILIES.contains(&family) {
            return Err(format!("video model {name}: family must be one of {}", VIDEO_FAMILIES.join(", ")));
        }
        if !bool_or(m, "enabled", true) {
            continue;
        }
        for key in ["transformer", "text_encoder", "vae"] {
            if str_or(m, key, "").trim().is_empty() {
                return Err(format!("video model {name} needs {key}"));
            }
        }
        if family != "ltx-2.5" && str_or(m, "tokenizer", "").trim().is_empty() {
            return Err(format!("video model {name} needs the Gemma 3 tokenizer.json"));
        }
    }
    let speech = object(media, "speech")?;
    for (name, m) in object(speech, "models")?.members() {
        if bool_or(m, "enabled", true) && ["design", "base", "breeze"].iter().all(|k| str_or(m, k, "").trim().is_empty()) {
            return Err(format!("speech model {name} needs a VoiceDesign folder (design), a Base folder (base) or a Breeze TTS 2 folder (breeze)"));
        }
    }
    let music = object(media, "music")?;
    for (name, m) in object(music, "models")?.members() {
        if bool_or(m, "enabled", true) && str_or(m, "path", "").trim().is_empty() {
            return Err(format!("music model {name} needs its MiniMax-Music3 folder (path)"));
        }
        let precision = str_or(m, "precision", "bf16");
        if !["bf16", "f32"].contains(&precision) {
            return Err(format!("music model {name}: precision must be bf16 or f32"));
        }
    }
    let sound = object(media, "sound")?;
    for (name, m) in object(sound, "models")?.members() {
        if bool_or(m, "enabled", true) && str_or(m, "path", "").trim().is_empty() {
            return Err(format!("sound model {name} needs its MOSS-SoundEffect folder (path)"));
        }
    }
    let model3d = object(media, "model3d")?;
    for (name, m) in object(model3d, "models")?.members() {
        if bool_or(m, "enabled", true) && str_or(m, "path", "").trim().is_empty() {
            return Err(format!("3D model {name} needs its Pixal3D folder (path)"));
        }
    }
    for (kind, section, models) in [("image", image, image_models), ("video", video, object(video, "models")?), ("speech", speech, object(speech, "models")?), ("music", music, object(music, "models")?), ("sound", sound, object(sound, "models")?), ("model3d", model3d, object(model3d, "models")?)] {
        let d = str_or(section, "default_model", "");
        if !d.is_empty() && models.get(d).is_none() {
            return Err(format!("media.{kind}.default_model {d} is not one of its models"));
        }
    }
    Ok(())
}

/// `value` as a path: absolute as given, relative to `root` otherwise.
pub fn resolve(root: &Path, value: &str) -> PathBuf {
    let p = Path::new(value.trim());
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

/// A sibling program (`oaiy-llm-server`, `oaiy-media`): an explicit path when it
/// names one, else beside the studio executable, else the same folder as the
/// configuration, else left to the OS search path.
pub fn program(root: &Path, value: &str) -> PathBuf {
    let value = value.trim();
    if value.contains(['/', '\\']) {
        return resolve(root, value);
    }
    let file = if cfg!(windows) && !value.to_ascii_lowercase().ends_with(".exe") { format!("{value}.exe") } else { value.to_string() };
    for dir in [exe_dir(), root.to_path_buf()] {
        let candidate = dir.join(&file);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(edit: impl FnOnce(&mut Json)) -> Result<(), String> {
        let mut v = default_json();
        edit(&mut v);
        validate(&v)
    }

    fn field<'a>(v: &'a mut Json, path: &[&str]) -> &'a mut Json {
        let mut v = v;
        for key in path {
            let Json::Obj(fields) = v else { panic!("not an object at {key}") };
            v = &mut fields.iter_mut().find(|(k, _)| k == key).unwrap().1;
        }
        v
    }

    #[test]
    fn older_files_gain_new_routes_once() {
        let mut v = Json::parse(br#"{"gateway":{"routes":[{"path":"/v1/chat/completions","method":"POST","target":"chat"}]}}"#).unwrap();
        assert!(upgrade_routes(&mut v));
        merge_defaults(&mut v, &default_json());
        let routes = v.get("gateway").unwrap().get("routes").unwrap().as_array().unwrap();
        let paths: Vec<&str> = routes.iter().map(|r| str_or(r, "path", "")).collect();
        assert_eq!(paths, ["/v1/chat/completions", "/v1/images/edits", "/v1/audio/speech", "/v1/audio/voices", "/v1/audio/music", "/v1/audio/sound_effects", "/v1/3d/models", "/v1/images/background_removal", "/v1/images/upscale", "/v1/discovery"]);
        // A file already at version 3 gains only the speech, music, sound, 3D and picture tool routes.
        let mut v3 = Json::parse(br#"{"gateway":{"routes_version":3,"routes":[]}}"#).unwrap();
        assert!(upgrade_routes(&mut v3));
        assert_eq!(v3.get("gateway").unwrap().get("routes").unwrap().len(), 7);
        assert!(!upgrade_routes(&mut v), "a second load changes nothing");
        validate(&v).unwrap();
    }

    #[test]
    fn companion_origins_are_defaults_and_reach_older_files_once() {
        let defaults = default_json();
        let listed: Vec<&str> = defaults.get("gateway").unwrap().get("cors_origins").unwrap().as_array().unwrap().iter().filter_map(Json::as_str).collect();
        assert_eq!(listed, COMPANION_ORIGINS);
        // A file written with the old empty default gains them, beside the user's own.
        let mut v = Json::parse(br#"{"gateway":{"cors_origins":["http://localhost:3000","https://BOT.computer/"],"routes":[]}}"#).unwrap();
        assert!(upgrade_origins(&mut v));
        merge_defaults(&mut v, &default_json());
        let origins: Vec<&str> = v.get("gateway").unwrap().get("cors_origins").unwrap().as_array().unwrap().iter().filter_map(Json::as_str).collect();
        assert_eq!(origins, ["http://localhost:3000", "https://BOT.computer/", "http://localhost:5317", "http://127.0.0.1:5317", "http://botcomputer.localhost", "botcomputer://localhost", "http://oaiy.localhost", "oaiy://localhost"]);
        // Once offered, a removed origin stays removed.
        if let Json::Obj(top) = &mut v {
            if let Some((_, Json::Obj(g))) = top.iter_mut().find(|(k, _)| k == "gateway") {
                if let Some((_, Json::Arr(o))) = g.iter_mut().find(|(k, _)| k == "cors_origins") {
                    o.retain(|x| x.as_str() != Some("botcomputer://localhost"));
                }
            }
        }
        assert!(!upgrade_origins(&mut v));
        // "*" already allows everything: nothing is added to it.
        let mut any = Json::parse(br#"{"gateway":{"cors_origins":["*"]}}"#).unwrap();
        assert!(upgrade_origins(&mut any));
        assert_eq!(any.get("gateway").unwrap().get("cors_origins").unwrap().len(), 1);
        validate(&v).unwrap();
    }

    #[test]
    fn a_file_from_version_2_gains_oaiys_window_once() {
        let mut v = Json::parse(br#"{"gateway":{"origins_version":2,"cors_origins":["https://bot.computer","http://botcomputer.localhost"]}}"#).unwrap();
        assert!(upgrade_origins(&mut v));
        let origins: Vec<&str> = v.get("gateway").unwrap().get("cors_origins").unwrap().as_array().unwrap().iter().filter_map(Json::as_str).collect();
        assert_eq!(origins, ["https://bot.computer", "http://botcomputer.localhost", "http://oaiy.localhost", "oaiy://localhost"]);
        assert!(!upgrade_origins(&mut v));
    }

    #[test]
    fn a_file_from_the_first_origins_version_gets_the_desktop_apps_own_origins() {
        let origins = |v: &Json| -> Vec<String> { v.get("gateway").unwrap().get("cors_origins").unwrap().as_array().unwrap().iter().filter_map(Json::as_str).map(String::from).collect() };
        // What version 1 wrote, less an origin the user removed and plus one they added.
        let mut v = Json::parse(br#"{"gateway":{"origins_version":1,"cors_origins":["https://bot.computer","http://127.0.0.1:5317","http://tauri.localhost","tauri://localhost","http://localhost:3000"]}}"#).unwrap();
        assert!(upgrade_origins(&mut v));
        assert_eq!(origins(&v), ["https://bot.computer", "http://127.0.0.1:5317", "http://botcomputer.localhost", "botcomputer://localhost", "http://localhost:3000", "http://oaiy.localhost", "oaiy://localhost"]);
        assert_eq!(v.get("gateway").unwrap().get("origins_version").and_then(Json::as_i64), Some(3));
        assert!(!upgrade_origins(&mut v), "once only");
        // Version 1 without the shared origins (the user took them out): they are not added back (OAIY's window is new).
        let mut w = Json::parse(br#"{"gateway":{"origins_version":1,"cors_origins":["http://localhost:3000"]}}"#).unwrap();
        assert!(upgrade_origins(&mut w));
        assert_eq!(origins(&w), ["http://localhost:3000", "http://oaiy.localhost", "oaiy://localhost"]);
    }

    #[test]
    fn the_shipped_example_configuration_validates() {
        let mut v = Json::parse(include_bytes!("../../../config/studio.example.json")).unwrap();
        assert!(!upgrade_routes(&mut v), "the example is already at the current routes_version");
        merge_defaults(&mut v, &default_json());
        validate(&v).unwrap();
    }

    #[test]
    fn defaults_validate_and_survive_a_pretty_round_trip() {
        let v = default_json();
        validate(&v).unwrap();
        assert_eq!(Json::parse(pretty(&v, 0).as_bytes()).unwrap(), v);
    }

    #[test]
    fn park_gb_is_null_by_default_and_takes_zero_or_a_number_of_gb() {
        assert_eq!(field(&mut default_json(), &["llm", "park_gb"]), &Json::Null, "unset: the server's own default applies");
        for ok in [Json::Null, Json::Int(0), Json::Int(8), Json::Num(2.5), Json::Int(1024)] {
            assert!(with(|v| *field(v, &["llm", "park_gb"]) = ok.clone()).is_ok(), "{ok:?}");
        }
        for bad in [Json::Int(-1), Json::Num(f64::NAN), Json::Int(1025), Json::str("8"), Json::Bool(true)] {
            assert!(with(|v| *field(v, &["llm", "park_gb"]) = bad.clone()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn missing_sections_are_filled_but_user_values_kept() {
        let mut v = Json::parse(br#"{"ui":{"port":9000},"gateway":{"routes":[]}}"#).unwrap();
        merge_defaults(&mut v, &default_json());
        assert_eq!(v.get("ui").unwrap().get("port").and_then(Json::as_i64), Some(9000));
        assert_eq!(v.get("ui").unwrap().get("host").and_then(Json::as_str), Some("127.0.0.1"));
        assert_eq!(v.get("gateway").unwrap().get("routes").unwrap().len(), 0);
        validate(&v).unwrap();
    }

    #[test]
    fn invalid_settings_are_refused_with_the_field_named() {
        let routes = |v: &mut Json| field(v, &["gateway", "routes"]).clone();
        for (edit, needle) in [
            (Box::new(|v: &mut Json| *field(v, &["ui", "port"]) = Json::Int(70000)) as Box<dyn FnOnce(&mut Json)>, "ui.port"),
            (Box::new(|v: &mut Json| *field(v, &["media", "image", "memory"]) = Json::str("disk")), "memory"),
            (Box::new(|v: &mut Json| *field(v, &["media", "llm_policy"]) = Json::str("never")), "llm_policy"),
            (Box::new(|v: &mut Json| *field(v, &["llm", "default_model"]) = Json::str("ghost")), "default_model"),
            (Box::new(|v: &mut Json| *field(v, &["llm", "park_gb"]) = Json::Int(-1)), "llm.park_gb"),
            (Box::new(|v: &mut Json| *field(v, &["llm", "park_gb"]) = Json::str("8")), "llm.park_gb"),
            (Box::new(move |v: &mut Json| {
                let mut r = routes(v);
                if let Json::Arr(items) = &mut r { let first = items[0].clone(); items.push(first); }
                *field(v, &["gateway", "routes"]) = r;
            }), "routed twice"),
            (Box::new(|v: &mut Json| {
                *field(v, &["gateway", "routes"]) = Json::parse(br#"[{"path":"/x","method":"POST","target":"chat","spec":"oaiy"}]"#).unwrap();
            }), "OAIY spec"),
            (Box::new(|v: &mut Json| {
                *field(v, &["media", "video", "models"]) = Json::parse(br#"{"clip":{"transformer":"a","text_encoder":"b","vae":"c"}}"#).unwrap();
            }), "family"),
            (Box::new(|v: &mut Json| {
                *field(v, &["media", "image", "models"]) = Json::parse(br#"{"x":{"architecture":"sdxl","checkpoint":"a"}}"#).unwrap();
            }), "tokenizer"),
        ] {
            let err = with(edit).unwrap_err();
            assert!(err.contains(needle), "{err} lacks {needle}");
        }
    }

    #[test]
    fn programs_and_paths_resolve_against_the_install() {
        let root = Path::new("/portable");
        assert_eq!(resolve(root, "outputs"), root.join("outputs"));
        let abs = std::env::temp_dir();
        assert_eq!(resolve(root, abs.to_str().unwrap()), abs);
        assert_eq!(program(root, "bin/worker"), root.join("bin/worker"));
        assert_eq!(program(root, "surely-not-installed"), PathBuf::from("surely-not-installed"));
    }

    /// A folder of the test's own under the temp folder, removed when the test ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!("oaiy-studio-config-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Everything in `dir`, by name.
    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    }

    /// The permission bits of `path`.
    #[cfg(unix)]
    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn the_configuration_and_the_secrets_in_it_are_saved_whole_and_owner_only() {
        let dir = Scratch::new("private");
        let path = dir.0.join(FILE_NAME);
        // Loading a file that is not there makes it, from the defaults.
        let mut v = load(&path).unwrap();
        #[cfg(unix)]
        assert_eq!(mode(&path), 0o600, "the file made by load");

        *field(&mut v, &["downloads", "hf_token"]) = Json::str("hf_test_token");
        *field(&mut v, &["gateway", "api_key"]) = Json::str("gateway-test-key");
        save(&path, &v).unwrap();
        #[cfg(unix)]
        assert_eq!(mode(&path), 0o600, "the file made by save");

        let back = load(&path).unwrap();
        assert_eq!(back.get("downloads").and_then(|d| d.get("hf_token")).and_then(Json::as_str), Some("hf_test_token"));
        assert_eq!(back.get("gateway").and_then(|g| g.get("api_key")).and_then(Json::as_str), Some("gateway-test-key"));
        assert_eq!(names(&dir.0), [FILE_NAME], "no temporary file is left beside it");
    }

    /// A file an older build wrote readable by everyone is replaced by a private one, not changed in place.
    #[cfg(unix)]
    #[test]
    fn a_configuration_left_readable_by_everyone_is_replaced_by_a_private_one() {
        use std::os::unix::fs::PermissionsExt;
        let dir = Scratch::new("loose");
        let path = dir.0.join(FILE_NAME);
        std::fs::write(&path, pretty(&default_json(), 0)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        save(&path, &default_json()).unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(names(&dir.0), [FILE_NAME]);
    }

    /// The point of it: at the first moment the file exists it is already private.
    #[cfg(unix)]
    #[test]
    fn the_new_file_is_private_from_the_moment_it_exists() {
        use std::os::unix::fs::PermissionsExt;
        let dir = Scratch::new("window");
        let (tmp, file) = create_private(&dir.0, FILE_NAME).unwrap();
        let meta = file.metadata().unwrap();
        assert_eq!(meta.len(), 0, "nothing has been written to it yet");
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        assert_eq!(tmp.parent(), Some(dir.0.as_path()), "beside the real file, so the rename stays on one filesystem");
    }

    /// A leftover of a killed process, or a link somebody put there, is not this save's to use.
    #[test]
    fn a_temporary_name_that_is_taken_is_skipped_and_left_alone() {
        let dir = Scratch::new("taken");
        let staged = |n: u64| format!(".{FILE_NAME}.{}-{n}.tmp", std::process::id());
        std::fs::write(dir.0.join(staged(100)), "a leftover").unwrap();
        #[cfg(unix)]
        {
            let victim = dir.0.join("victim");
            std::fs::write(&victim, "not to be overwritten").unwrap();
            std::os::unix::fs::symlink(&victim, dir.0.join(staged(101))).unwrap();
        }
        #[cfg(not(unix))]
        std::fs::write(dir.0.join(staged(101)), "another leftover").unwrap();

        let mut numbers = 100..;
        let (tmp, _file) = create_private_numbered(&dir.0, FILE_NAME, || numbers.next().unwrap()).unwrap();
        assert_eq!(tmp, dir.0.join(staged(102)));
        assert_eq!(std::fs::read_to_string(dir.0.join(staged(100))).unwrap(), "a leftover");
        #[cfg(unix)]
        assert_eq!(std::fs::read_to_string(dir.0.join("victim")).unwrap(), "not to be overwritten");

        // Every name taken: the save fails rather than write anywhere else.
        let mut all = 0..;
        let full = Scratch::new("full");
        for n in 0..32 {
            std::fs::write(full.0.join(format!(".{FILE_NAME}.{}-{n}.tmp", std::process::id())), "x").unwrap();
        }
        assert!(create_private_numbered(&full.0, FILE_NAME, || all.next().unwrap()).is_err());
        assert_eq!(all.next(), Some(32), "and it tried no more than 32");
    }
}
