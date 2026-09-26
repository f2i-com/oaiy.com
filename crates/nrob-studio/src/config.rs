//! The studio's one configuration file, `nrob-studio.json`, kept beside the
//! executable so a copied folder is a complete, portable install.
//!
//! The configuration stays a JSON tree (the UI edits it whole and saves it back);
//! [`validate`] is the contract every save passes, and the typed getters below
//! read what the supervisors need from it. Relative paths are relative to the
//! file's own directory.

use crate::util::{bool_or, int_or, str_or};
use nrob::json::Json;
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "nrob-studio.json";

/// What a gateway route serves.
pub const TARGETS: [&str; 8] = ["chat", "completions", "models", "images", "edits", "videos", "health", "files"];
/// The request/response dialect a route speaks. `openai` is the OpenAI API;
/// `nrob` is nrob-server's own asynchronous media job API (what coder-cli uses).
pub const SPECS: [&str; 2] = ["openai", "nrob"];
pub const MEMORY: [&str; 4] = ["auto", "gpu", "ram", "ssd"];
pub const VIDEO_FAMILIES: [&str; 3] = ["ltx-2.3", "ltx-2.5", "sulphur-2"];
pub const IMAGE_ARCHITECTURES: [&str; 2] = ["qwen-image", "sdxl"];
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
    "routes_version": 2,
    "routes": [
      { "path": "/v1/chat/completions", "method": "POST", "target": "chat", "spec": "openai", "enabled": true },
      { "path": "/v1/completions", "method": "POST", "target": "completions", "spec": "openai", "enabled": true },
      { "path": "/v1/models", "method": "GET", "target": "models", "spec": "openai", "enabled": true },
      { "path": "/v1/images/generations", "method": "POST", "target": "images", "spec": "openai", "enabled": true },
      { "path": "/v1/images/edits", "method": "POST", "target": "edits", "spec": "openai", "enabled": true },
      { "path": "/v1/videos", "method": "POST", "target": "videos", "spec": "openai", "enabled": true },
      { "path": "/files", "method": "GET", "target": "files", "spec": "openai", "enabled": true },
      { "path": "/health", "method": "GET", "target": "health", "spec": "openai", "enabled": true }
    ]
  },
  "llm": {
    "enabled": true,
    "autostart": false,
    "server": "nrob-server",
    "server_webgpu": "nrob-server-webgpu",
    "backend": "auto",
    "webgpu_gb": null,
    "default_model": "",
    "models": [],
    "devices": [],
    "ctx": 32768,
    "ram_gb": 0,
    "cpu_threads": null,
    "vram_headroom_gb": 2,
    "thinking": false,
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "prompt_cache": true,
    "prompt_cache_gb": 4,
    "vision": true,
    "idle_stop_minutes": 0,
    "extra_args": []
  },
  "media": {
    "worker": "nrob-diffusion",
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
    const VERSION: i64 = 2;
    let Json::Obj(top) = v else { return false };
    let Some((_, gateway)) = top.iter_mut().find(|(k, _)| k == "gateway") else { return false };
    if int_or(gateway, "routes_version", 1) >= VERSION {
        return false;
    }
    let defaults = default_json();
    let added: Vec<Json> = defaults.get("gateway").and_then(|g| g.get("routes")).and_then(Json::as_array).unwrap_or(&[]).iter()
        .filter(|r| str_or(r, "target", "") == "edits").cloned().collect();
    if let Json::Obj(fields) = gateway {
        if let Some((_, Json::Arr(routes))) = fields.iter_mut().find(|(k, _)| k == "routes") {
            for route in added {
                let taken = routes.iter().any(|r| str_or(r, "target", "") == "edits" || str_or(r, "path", "") == str_or(&route, "path", ""));
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
    merge_defaults(&mut v, &default_json());
    if upgraded {
        save(path, &v)?;
    }
    validate(&v).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(v)
}

/// Write atomically: a temporary file beside it, then a rename.
pub fn save(path: &Path, v: &Json) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, pretty(v, 0)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
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
        if spec == "nrob" && !["images", "videos"].contains(&target) {
            return Err(format!("{what}: the nrob spec applies to images and videos only"));
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
    let default = str_or(llm, "default_model", "");
    if !default.is_empty() && !names.contains(&default) {
        return Err(format!("llm.default_model {default} is not one of the listed models"));
    }
    if !llm.get("devices").and_then(Json::as_array).is_some_and(|d| d.iter().all(|d| d.as_i64().is_some_and(|d| (0..64).contains(&d)))) {
        return Err("llm.devices must be a list of GPU indices".into());
    }
    for (key, min, max) in [("ctx", 512, 1 << 20), ("max_tokens", 1, 1 << 20), ("cpu_threads", 0, 1024), ("ram_gb", 0, 4096)] {
        // `cpu_threads: null` means "this machine's core count".
        if key == "cpu_threads" && matches!(llm.get(key), Some(Json::Null)) {
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
            return Err(format!("image model {name}: architecture must be qwen-image or sdxl"));
        }
        // A disabled entry may be incomplete: a picked file waiting for its parts.
        if !bool_or(m, "enabled", true) {
            continue;
        }
        let required: &[&str] = if arch == "sdxl" { &["checkpoint", "tokenizer"] } else { &["base"] };
        for key in required {
            if str_or(m, key, "").trim().is_empty() {
                return Err(format!("image model {name} needs {key}"));
            }
        }
        if arch == "qwen-image" && str_or(m, "transformer", "").is_empty() && str_or(m, "safetensors_transformer", "").is_empty() {
            return Err(format!("image model {name} needs a transformer (a .gguf or a .safetensors checkpoint)"));
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
    for (kind, section, models) in [("image", image, image_models), ("video", video, object(video, "models")?)] {
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

/// A sibling program (`nrob-server`, `nrob-diffusion`): an explicit path when it
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
    fn older_files_gain_the_edits_route_once() {
        let mut v = Json::parse(br#"{"gateway":{"routes":[{"path":"/v1/chat/completions","method":"POST","target":"chat"}]}}"#).unwrap();
        assert!(upgrade_routes(&mut v));
        merge_defaults(&mut v, &default_json());
        let routes = v.get("gateway").unwrap().get("routes").unwrap().as_array().unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(str_or(&routes[1], "path", ""), "/v1/images/edits");
        assert!(!upgrade_routes(&mut v), "a second load changes nothing");
        validate(&v).unwrap();
    }

    #[test]
    fn defaults_validate_and_survive_a_pretty_round_trip() {
        let v = default_json();
        validate(&v).unwrap();
        assert_eq!(Json::parse(pretty(&v, 0).as_bytes()).unwrap(), v);
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
            (Box::new(move |v: &mut Json| {
                let mut r = routes(v);
                if let Json::Arr(items) = &mut r { let first = items[0].clone(); items.push(first); }
                *field(v, &["gateway", "routes"]) = r;
            }), "routed twice"),
            (Box::new(|v: &mut Json| {
                *field(v, &["gateway", "routes"]) = Json::parse(br#"[{"path":"/x","method":"POST","target":"chat","spec":"nrob"}]"#).unwrap();
            }), "nrob spec"),
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
}
