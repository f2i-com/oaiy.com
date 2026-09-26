//! Adding a picked file to the configuration: detect what it is, register it in
//! the section whose endpoint serves it, fill the parts it needs from models
//! already configured (or from files beside it), and make sure a route for that
//! endpoint is enabled.

use crate::config;
use crate::detect::{self, Detected, Role};
use crate::util::{bool_or, set, str_or};
use nrob::json::Json;
use std::path::{Path, PathBuf};

/// Where an entry lives: `llm`, `image` or `video`, and its name.
#[derive(Debug, PartialEq)]
pub struct Added {
    pub section: &'static str,
    pub name: String,
    pub missing: Vec<String>,
    pub enabled: bool,
}

fn obj_mut<'a>(v: &'a mut Json, path: &[&str]) -> Option<&'a mut Json> {
    let mut v = v;
    for key in path {
        let Json::Obj(fields) = v else { return None };
        v = &mut fields.iter_mut().find(|(k, _)| k == key)?.1;
    }
    Some(v)
}

fn get<'a>(v: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(v, |v, k| v.get(k))
}

fn unique(taken: &[String], wanted: &str) -> String {
    if !taken.iter().any(|t| t == wanted) {
        return wanted.to_string();
    }
    (2..).map(|i| format!("{wanted}-{i}")).find(|n| !taken.iter().any(|t| t == n)).unwrap_or_default()
}

/// Enable a default route for `target` unless one is enabled already.
fn ensure_route(cfg: &mut Json, target: &str) {
    let Some(Json::Arr(routes)) = obj_mut(cfg, &["gateway", "routes"]) else { return };
    if routes.iter().any(|r| str_or(r, "target", "") == target && bool_or(r, "enabled", true)) {
        return;
    }
    let defaults = config::default_json();
    let fallback = get(&defaults, &["gateway", "routes"]).and_then(Json::as_array).and_then(|d| d.iter().find(|r| str_or(r, "target", "") == target).cloned());
    if let Some(mut route) = fallback {
        // Re-enable a disabled route at the same path rather than adding a twin.
        if let Some(existing) = routes.iter_mut().find(|r| str_or(r, "path", "") == str_or(&route, "path", "")) {
            set(existing, "enabled", Json::Bool(true));
            set(existing, "target", Json::str(target));
            return;
        }
        set(&mut route, "enabled", Json::Bool(true));
        routes.push(route);
    }
}

/// Add the default route for `target` only when the table has none at all. An
/// import uses this: a route the user turned off stays off.
fn add_route_if_absent(cfg: &mut Json, target: &str) {
    let present = get(cfg, &["gateway", "routes"]).and_then(Json::as_array).is_some_and(|r| r.iter().any(|r| str_or(r, "target", "") == target));
    if !present {
        ensure_route(cfg, target);
    }
}

/// Files near `path` (its folder, the parent's, and their subfolders, a few
/// levels) for which `accept` finds a part. Header reads only; bounded.
fn nearby(path: &Path, accept: impl Fn(&Path, &Detected) -> bool) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut at = if path.is_dir() { Some(path.to_path_buf()) } else { path.parent().map(Path::to_path_buf) };
    for _ in 0..3 {
        let Some(d) = at else { break };
        at = d.parent().map(Path::to_path_buf);
        dirs.push(d);
    }
    let mut seen = 0;
    for d in &dirs {
        let mut candidates = vec![d.clone()];
        if let Ok(rd) = std::fs::read_dir(d) {
            let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            candidates.extend(entries);
        }
        for c in candidates {
            if c == path {
                continue;
            }
            let ext = c.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let interesting = c.is_dir() || ext == "safetensors" || ext == "gguf" || c.file_name().is_some_and(|n| n == "tokenizer.json");
            if !interesting {
                continue;
            }
            // A subfolder's tokenizer is a common layout (`clip-tokenizer/tokenizer.json`).
            let probe = if c.is_dir() && c.join("tokenizer.json").is_file() && !c.join("model_index.json").is_file() { c.join("tokenizer.json") } else { c.clone() };
            seen += 1;
            if seen > 80 {
                return None;
            }
            if let Ok(found) = detect::detect(&probe) {
                if accept(&probe, &found) {
                    return Some(probe);
                }
            }
        }
    }
    None
}

fn field_of(d: &Detected, key: &str) -> Option<Json> {
    d.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// The part named by `missing` for a new entry: from another configured model
/// of the same kind first, then from files near the picked one.
fn companion(cfg: &Json, section: &str, entry: &Json, missing: &str, picked: &Path) -> Option<Json> {
    let arch = str_or(entry, "architecture", "");
    let family = str_or(entry, "family", "");
    // LTX 2.3 and Sulphur share Gemma 3; LTX 2.5 has its own encoder.
    let same_group = |f: &str| (f == "ltx-2.5") == (family == "ltx-2.5");
    for (_, other) in get(cfg, &["media", section, "models"]).map(|m| m.members().collect::<Vec<_>>()).unwrap_or_default() {
        let compatible = match section {
            "image" => str_or(other, "architecture", "qwen-image") == arch,
            _ => same_group(str_or(other, "family", "")),
        };
        if let (true, Some(v)) = (compatible, other.get(missing).filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))) {
            return Some(v.clone());
        }
    }
    let want: &dyn Fn(&Detected) -> bool = match (section, missing) {
        ("image", "base") => &|d| d.format == "diffusers" && field_of(d, "base").is_some(),
        ("image", "tokenizer") => &|d| d.kind() == "clip_tokenizer",
        ("video", "tokenizer") => &|d| d.kind() == "tokenizer",
        ("video", "vae") => &|d| d.kind() == "vae",
        ("video", "text_encoder") if family == "ltx-2.5" => &|d| d.kind() == "video_text_encoder" && d.summary.contains("2.5"),
        ("video", "text_encoder") => &|d| d.kind() == "video_text_encoder" && !d.summary.contains("2.5"),
        _ => return None,
    };
    let found = nearby(picked, |_, d| want(d))?;
    let d = detect::detect(&found).ok()?;
    let key = if missing == "base" { "base" } else { missing };
    field_of(&d, key).or_else(|| Some(Json::str(found.to_string_lossy())))
}

/// Components attach to the first model that lacks them.
fn attach(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    type Fits = Box<dyn Fn(&Json) -> bool>;
    let (section, field, fits): (&str, &str, Fits) = match d.kind() {
        "vision_projector" => ("llm", "vision_projector", Box::new(|_| true)),
        "adapter" => ("image", "adapter", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "image_base" => ("image", "base", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "text_encoder" => ("image", "text_encoder", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "clip_tokenizer" => ("image", "tokenizer", Box::new(|m| str_or(m, "architecture", "") == "sdxl")),
        "tokenizer" => ("video", "tokenizer", Box::new(|m| str_or(m, "family", "") != "ltx-2.5")),
        "vae" => ("video", "vae", Box::new(|_| true)),
        "video_text_encoder" => {
            let v25 = d.summary.contains("2.5");
            ("video", "text_encoder", Box::new(move |m| (str_or(m, "family", "") == "ltx-2.5") == v25))
        }
        "ffmpeg" => {
            let value = field_of(d, "ffmpeg").unwrap_or(Json::Null);
            set(obj_mut(cfg, &["media", "video"]).ok_or("no video section")?, "ffmpeg", value);
            return Ok(Added { section: "video", name: "ffmpeg".into(), missing: Vec::new(), enabled: true });
        }
        other => return Err(format!("do not know where a {other} goes")),
    };
    let value = field_of(d, field).ok_or("detected part has no path")?;
    let lacks = |m: &Json| str_or(m, field, "").trim().is_empty();
    if section == "llm" {
        let Some(Json::Arr(models)) = obj_mut(cfg, &["llm", "models"]) else { return Err("no llm models".into()) };
        let default = models.iter().position(|m| target.is_some_and(|(_, n)| str_or(m, "name", "") == n))
            .or_else(|| models.iter().position(lacks))
            .ok_or("add the language model first, then its vision projector")?;
        set(&mut models[default], field, value);
        return Ok(Added { section: "llm", name: str_or(&models[default], "name", "").into(), missing: Vec::new(), enabled: true });
    }
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", section, "models"]) else { return Err(format!("no {section} models")) };
    let chosen = models.iter().position(|(n, _)| target.is_some_and(|(_, t)| n == t))
        .or_else(|| models.iter().position(|(_, m)| fits(m) && lacks(m)))
        .ok_or_else(|| format!("{}: no {section} model is missing this part; add the model first, or pick the part from the model's own field", d.summary))?;
    let (name, model) = &mut models[chosen];
    set(model, field, value);
    let missing = missing_fields(section, model);
    let enabled = missing.is_empty();
    set(model, "enabled", Json::Bool(enabled));
    Ok(Added { section: if section == "image" { "image" } else { "video" }, name: name.clone(), missing, enabled })
}

/// Required fields an entry still lacks (mirrors `config::validate`).
pub fn missing_fields(section: &str, m: &Json) -> Vec<String> {
    let empty = |k: &str| str_or(m, k, "").trim().is_empty();
    let mut out = Vec::new();
    if section == "image" {
        if str_or(m, "architecture", "qwen-image") == "sdxl" {
            out.extend(["checkpoint", "tokenizer"].into_iter().filter(|k| empty(k)).map(String::from));
        } else {
            if empty("base") {
                out.push("base".into());
            }
            if empty("transformer") && empty("safetensors_transformer") {
                out.push("transformer".into());
            }
        }
    } else {
        out.extend(["transformer", "text_encoder", "vae"].into_iter().filter(|k| empty(k)).map(String::from));
        if str_or(m, "family", "") != "ltx-2.5" && empty("tokenizer") {
            out.push("tokenizer".into());
        }
    }
    out
}

/// Entries still waiting for a part the new entry carries get it now (a GGUF
/// transformer added before its base folder, a second video model before the
/// first one's encoder). Returns the names completed.
fn backfill(cfg: &mut Json, section: &str, new_name: &str) -> Vec<String> {
    let Some(new) = get(cfg, &["media", section, "models", new_name]).cloned() else { return Vec::new() };
    let keys: &[&str] = if section == "image" { &["base"] } else { &["text_encoder", "tokenizer", "vae"] };
    let family = str_or(&new, "family", "").to_string();
    let arch = str_or(&new, "architecture", "qwen-image").to_string();
    let mut done = Vec::new();
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", section, "models"]) else { return done };
    for (name, m) in models.iter_mut().filter(|(n, _)| n != new_name) {
        let compatible = if section == "image" {
            str_or(m, "architecture", "qwen-image") == arch
        } else {
            (str_or(m, "family", "") == "ltx-2.5") == (family == "ltx-2.5")
        };
        if !compatible || bool_or(m, "enabled", true) {
            continue;
        }
        for key in keys {
            // A combined LTX 2.3 file is its own VAE; only lend what was a separate part.
            let lend = new.get(key).and_then(Json::as_str).filter(|v| !v.is_empty() && Some(*v) != new.get("transformer").and_then(Json::as_str));
            if let (true, Some(v)) = (str_or(m, key, "").is_empty(), lend) {
                set(m, key, Json::str(v));
            }
        }
        if missing_fields(section, m).is_empty() {
            set(m, "enabled", Json::Bool(true));
            done.push(name.clone());
        }
    }
    done
}

/// Detect `path` and add it: a new model in its section, or a part attached to
/// the model that needs it (`target`: section and model name, when the UI says).
pub fn add(cfg: &mut Json, path: &Path, name: Option<&str>, target: Option<(&str, &str)>) -> Result<(Added, Detected), String> {
    let d = detect::detect(path)?;
    let wanted = name.map(str::to_string).filter(|n| !n.trim().is_empty()).unwrap_or_else(|| detect::model_name(path));
    let added = match &d.role {
        Role::Component { .. } => attach(cfg, &d, target)?,
        Role::Llm => {
            let Some(Json::Arr(models)) = obj_mut(cfg, &["llm", "models"]) else { return Err("no llm models".into()) };
            let taken: Vec<String> = models.iter().map(|m| str_or(m, "name", "").to_string()).collect();
            let name = unique(&taken, &wanted);
            let mut entry = Json::obj([("name", Json::str(&name)), ("enabled", Json::Bool(true))]);
            for (k, v) in &d.fields {
                set(&mut entry, k, v.clone());
            }
            models.push(entry);
            let llm = obj_mut(cfg, &["llm"]).ok_or("no llm section")?;
            if str_or(llm, "default_model", "").is_empty() {
                set(llm, "default_model", Json::str(&name));
            }
            for target in ["chat", "completions", "models"] {
                ensure_route(cfg, target);
            }
            Added { section: "llm", name, missing: Vec::new(), enabled: true }
        }
        Role::Image { .. } | Role::Video { .. } => {
            let section = if matches!(d.role, Role::Image { .. }) { "image" } else { "video" };
            let mut entry = Json::Obj(d.fields.clone());
            for m in d.missing.clone() {
                if let Some(v) = companion(cfg, section, &entry, &m, path) {
                    set(&mut entry, &m, v);
                }
            }
            if section == "image" && str_or(&entry, "architecture", "") == "qwen-image" && str_or(&entry, "adapter", "").is_empty() {
                // Reuse the turbo adapter another Qwen Image model already runs with.
                if let Some(a) = companion(cfg, "image", &entry, "adapter", path) {
                    set(&mut entry, "adapter", a);
                }
            }
            let missing = missing_fields(section, &entry);
            let enabled = missing.is_empty();
            set(&mut entry, "enabled", Json::Bool(enabled));
            let models = obj_mut(cfg, &["media", section, "models"]).ok_or("no media models")?;
            let taken: Vec<String> = models.members().map(|(k, _)| k.to_string()).collect();
            let name = unique(&taken, &wanted);
            if let Json::Obj(fields) = models {
                fields.push((name.clone(), entry));
            }
            let sec = obj_mut(cfg, &["media", section]).ok_or("no media section")?;
            if enabled && str_or(sec, "default_model", "").is_empty() {
                set(sec, "default_model", Json::str(&name));
            }
            backfill(cfg, section, &name);
            ensure_route(cfg, if section == "image" { "images" } else { "videos" });
            ensure_route(cfg, "files");
            Added { section: if section == "image" { "image" } else { "video" }, name, missing, enabled }
        }
    };
    config::validate(cfg)?;
    Ok((added, d))
}

/// Keys holding file paths, per section, for export and import.
const LLM_PATHS: [&str; 3] = ["path", "vision_projector", "lora"];
const IMAGE_PATHS: [&str; 7] = ["base", "transformer", "safetensors_transformer", "adapter", "text_encoder", "checkpoint", "tokenizer"];
const VIDEO_PATHS: [&str; 4] = ["transformer", "text_encoder", "vae", "tokenizer"];
/// Media settings that travel with the models.
const MEDIA_SETTINGS: [&str; 4] = ["memory", "ram_gb", "vram_gb", "default_model"];

fn absolute(root: &Path, m: &Json, keys: &[&str]) -> Json {
    let mut m = m.clone();
    for key in keys {
        if let Some(p) = m.get(key).and_then(Json::as_str).filter(|p| !p.trim().is_empty()) {
            let abs = config::resolve(root, p).to_string_lossy().into_owned();
            set(&mut m, key, Json::str(abs));
        }
    }
    m
}

/// Every configured model (language, image, video), with their parts, defaults
/// and memory settings, as a file another install can import. Paths are made
/// absolute so the file does not depend on where this studio lives.
pub fn export(cfg: &Json, root: &Path) -> Json {
    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    let llm_models: Vec<Json> = llm.get("models").and_then(Json::as_array).unwrap_or(&[]).iter().map(|m| absolute(root, m, &LLM_PATHS)).collect();
    let media = |kind: &str, keys: &[&str], extra: &[&str]| {
        let section = get(cfg, &["media", kind]).cloned().unwrap_or(Json::Null);
        let models: Vec<(String, Json)> = section.get("models").map(|m| m.members().map(|(n, v)| (n.to_string(), absolute(root, v, keys))).collect()).unwrap_or_default();
        let mut out: Vec<(String, Json)> = MEDIA_SETTINGS.iter().chain(extra).filter_map(|k| section.get(k).map(|v| (k.to_string(), v.clone()))).collect();
        out.push(("models".into(), Json::Obj(models)));
        Json::Obj(out)
    };
    Json::obj([
        ("nrob_models", Json::Int(1)),
        ("exported_at", Json::Int(crate::util::now() as i64)),
        ("llm", Json::obj([("default_model", llm.get("default_model").cloned().unwrap_or(Json::str(""))), ("models", Json::Arr(llm_models))])),
        ("image", media("image", &IMAGE_PATHS, &[])),
        ("video", media("video", &VIDEO_PATHS, &["fps", "ffmpeg"])),
    ])
}

/// Bring an exported file in. `replace` clears the current models first;
/// otherwise models merge by name (the file's version wins). Entries whose
/// files are not on this machine come in disabled and are reported, so a file
/// from another PC imports cleanly and shows what to repoint.
pub fn import(cfg: &mut Json, doc: &Json, replace: bool, root: &Path) -> Result<Json, String> {
    if doc.get("nrob_models").and_then(Json::as_i64) != Some(1) {
        return Err("not an NROB models file (missing \"nrob_models\": 1)".into());
    }
    let mut missing = Vec::new();
    let mut tools_missing = Vec::new();
    let mut counts = (0, 0, 0);
    // Relative paths mean what they mean everywhere else: beside the studio.
    let exists = |p: &str| config::resolve(root, p).exists();
    let mut check = |section: &str, name: &str, m: &mut Json, keys: &[&str]| {
        let mut ok = true;
        for key in keys {
            if let Some(p) = m.get(key).and_then(Json::as_str).filter(|p| !p.trim().is_empty() && !exists(p)) {
                missing.push(Json::obj([("section", Json::str(section)), ("model", Json::str(name)), ("field", Json::str(*key)), ("path", Json::str(p))]));
                ok = false;
            }
        }
        if !ok {
            set(m, "enabled", Json::Bool(false));
        }
    };
    // Language models: a list keyed by name.
    if let Some(incoming) = get(doc, &["llm", "models"]).and_then(Json::as_array) {
        let Some(Json::Arr(models)) = obj_mut(cfg, &["llm", "models"]) else { return Err("no llm section".into()) };
        if replace {
            models.clear();
        }
        for m in incoming {
            let name = str_or(m, "name", "").to_string();
            if name.is_empty() || str_or(m, "path", "").is_empty() {
                continue;
            }
            let mut m = m.clone();
            check("llm", &name, &mut m, &LLM_PATHS);
            models.retain(|x| str_or(x, "name", "") != name);
            models.push(m);
            counts.0 += 1;
        }
    }
    if let Some(d) = get(doc, &["llm", "default_model"]).and_then(Json::as_str).filter(|d| !d.is_empty()) {
        let known = get(cfg, &["llm", "models"]).and_then(Json::as_array).is_some_and(|m| m.iter().any(|x| str_or(x, "name", "") == d));
        if known {
            set(obj_mut(cfg, &["llm"]).ok_or("no llm section")?, "default_model", Json::str(d));
        }
    }
    for (kind, keys) in [("image", &IMAGE_PATHS[..]), ("video", &VIDEO_PATHS[..])] {
        let Some(section_in) = doc.get(kind) else { continue };
        {
            let Some(Json::Obj(models)) = obj_mut(cfg, &["media", kind, "models"]) else { return Err(format!("no {kind} section")) };
            if replace {
                models.clear();
            }
            for (name, m) in section_in.get("models").map(|m| m.members().collect::<Vec<_>>()).unwrap_or_default() {
                let mut m = m.clone();
                check(kind, name, &mut m, keys);
                models.retain(|(n, _)| n != name);
                models.push((name.to_string(), m));
                if kind == "image" { counts.1 += 1 } else { counts.2 += 1 }
            }
        }
        let section = obj_mut(cfg, &["media", kind]).ok_or("no media section")?;
        for key in ["memory", "ram_gb", "vram_gb", "fps"] {
            if let Some(v) = section_in.get(key) {
                set(section, key, v.clone());
            }
        }
        // ffmpeg is a program this machine runs: taken only when it is here,
        // otherwise the local setting stays and the path is reported.
        if let Some(f) = section_in.get("ffmpeg").and_then(Json::as_str).map(str::trim).filter(|f| !f.is_empty()) {
            // A bare `ffmpeg` means the one on the search path, wherever that is.
            if exists(f) || f.eq_ignore_ascii_case("ffmpeg") || f.eq_ignore_ascii_case("ffmpeg.exe") {
                set(section, "ffmpeg", Json::str(f));
            } else {
                tools_missing.push(Json::obj([("section", Json::str(kind)), ("model", Json::str("")), ("field", Json::str("ffmpeg")), ("path", Json::str(f))]));
            }
        }
        let d = str_or(section_in, "default_model", "").to_string();
        let known = section.get("models").is_some_and(|m| m.get(&d).is_some());
        if known {
            set(section, "default_model", Json::str(&d));
        } else if replace {
            set(section, "default_model", Json::str(""));
        }
    }
    if replace {
        let known: Vec<String> = get(cfg, &["llm", "models"]).and_then(Json::as_array).unwrap_or(&[]).iter().map(|m| str_or(m, "name", "").to_string()).collect();
        let llm = obj_mut(cfg, &["llm"]).ok_or("no llm section")?;
        if !known.iter().any(|n| n == str_or(llm, "default_model", "")) {
            set(llm, "default_model", Json::str(known.first().map_or("", String::as_str)));
        }
    }
    let mut needed = Vec::new();
    if counts.0 > 0 {
        needed.extend(["chat", "completions", "models"]);
    }
    if counts.1 > 0 {
        needed.extend(["images", "edits", "files"]);
    }
    if counts.2 > 0 {
        needed.extend(["videos", "files"]);
    }
    for target in needed {
        add_route_if_absent(cfg, target);
    }
    config::validate(cfg)?;
    missing.extend(tools_missing);
    Ok(Json::obj([
        ("llm", Json::Int(counts.0)),
        ("image", Json::Int(counts.1)),
        ("video", Json::Int(counts.2)),
        ("missing", Json::Arr(missing)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nrob-studio-registry-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn safetensors(path: &Path, header: &str) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    fn gguf(path: &Path, arch: &str) {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        for s in ["general.architecture"] {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        }
        b.extend(8u32.to_le_bytes());
        b.extend((arch.len() as u64).to_le_bytes());
        b.extend(arch.as_bytes());
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn picked_files_land_in_the_right_section_with_their_parts() {
        let d = tmp("add");
        let mut cfg = config::default_json();
        // An LLM becomes the default chat model; its projector attaches to it.
        gguf(&d.join("Qwen3-8B.gguf"), "qwen3");
        let (a, _) = add(&mut cfg, &d.join("Qwen3-8B.gguf"), None, None).unwrap();
        assert_eq!((a.section, a.name.as_str()), ("llm", "qwen3-8b"));
        assert_eq!(get(&cfg, &["llm", "default_model"]).and_then(Json::as_str), Some("qwen3-8b"));
        gguf(&d.join("mmproj.gguf"), "clip");
        let (p, _) = add(&mut cfg, &d.join("mmproj.gguf"), None, None).unwrap();
        assert_eq!((p.section, p.name.as_str()), ("llm", "qwen3-8b"));
        // SDXL finds the CLIP tokenizer in a subfolder beside it.
        std::fs::create_dir_all(d.join("clip-tokenizer")).unwrap();
        std::fs::write(d.join("clip-tokenizer").join("tokenizer.json"), r#"{"x":"<|startoftext|>"}"#).unwrap();
        safetensors(&d.join("anime.safetensors"), r#"{"__metadata__":{"modelspec.architecture":"stable-diffusion-xl-v1-base"}}"#);
        let (s, _) = add(&mut cfg, &d.join("anime.safetensors"), None, None).unwrap();
        assert_eq!((s.section, s.enabled), ("image", true), "{:?}", s.missing);
        assert_eq!(get(&cfg, &["media", "image", "default_model"]).and_then(Json::as_str), Some("anime"));
        // A video transformer without its encoder is kept, disabled, until the part arrives.
        safetensors(&d.join("ltx.safetensors"), r#"{"__metadata__":{"model_version":"2.3.0","config":"{\"transformer\":{},\"vae\":{}}"}}"#);
        let (v, _) = add(&mut cfg, &d.join("ltx.safetensors"), None, None).unwrap();
        assert_eq!((v.section, v.enabled), ("video", false));
        assert_eq!(v.missing, vec!["text_encoder".to_string(), "tokenizer".to_string()]);
        let gemma = r#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[262208,8],"data_offsets":[0,2]},"model.layers.0.mlp.down_proj.weight":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#;
        let enc_dir = d.join("encoders");
        std::fs::create_dir_all(&enc_dir).unwrap();
        safetensors(&enc_dir.join("gemma.safetensors"), gemma);
        let (e, _) = add(&mut cfg, &enc_dir.join("gemma.safetensors"), None, None).unwrap();
        assert_eq!((e.section, e.name.as_str(), e.missing.clone()), ("video", "ltx", vec!["tokenizer".to_string()]));
        std::fs::write(d.join("tokenizer.json"), r#"{"x":"<start_of_turn>"}"#).unwrap();
        let (t, _) = add(&mut cfg, &d.join("tokenizer.json"), None, None).unwrap();
        assert!(t.enabled, "complete video model is enabled");
        // A second video model of the same family borrows the first one's parts.
        safetensors(&d.join("ltx2.safetensors"), r#"{"__metadata__":{"model_version":"2.3.0","config":"{\"transformer\":{},\"vae\":{}}"}}"#);
        let (v2, _) = add(&mut cfg, &d.join("ltx2.safetensors"), None, None).unwrap();
        assert!(v2.enabled && v2.missing.is_empty());
        // A base folder added after a GGUF transformer completes it.
        let gguf_image = d.join("qimg.gguf");
        gguf(&gguf_image, "qwen_image21");
        let (q, _) = add(&mut cfg, &gguf_image, None, None).unwrap();
        assert_eq!(q.missing, vec!["base".to_string()]);
        let pipe = d.join("pipeline");
        std::fs::create_dir_all(pipe.join("transformer")).unwrap();
        std::fs::write(pipe.join("model_index.json"), r#"{"_class_name":"QwenImage21Pipeline"}"#).unwrap();
        safetensors(&pipe.join("transformer").join("t.safetensors"), "{}");
        add(&mut cfg, &pipe, None, None).unwrap();
        assert_eq!(get(&cfg, &["media", "image", "models", "qimg", "enabled"]), Some(&Json::Bool(true)));
        // Duplicate names get a suffix; unknown files are refused.
        let (again, _) = add(&mut cfg, &d.join("Qwen3-8B.gguf"), None, None).unwrap();
        assert_eq!(again.name, "qwen3-8b-2");
        std::fs::write(d.join("notes.txt"), b"x").unwrap();
        assert!(add(&mut cfg, &d.join("notes.txt"), None, None).is_err());
        config::validate(&cfg).unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn exports_round_trip_and_imports_flag_missing_files() {
        let d = tmp("export");
        let mut cfg = config::default_json();
        gguf(&d.join("Chat.gguf"), "qwen3");
        add(&mut cfg, &d.join("Chat.gguf"), None, None).unwrap();
        let file = export(&cfg, Path::new("/unused"));
        assert_eq!(get(&file, &["llm", "models"]).unwrap().len(), 1);
        // Into a fresh install: the model arrives, enabled, as the default.
        let mut other = config::default_json();
        let report = import(&mut other, &file, false, &d).unwrap();
        assert_eq!(report.get("llm").and_then(Json::as_i64), Some(1));
        assert_eq!(get(&other, &["llm", "default_model"]).and_then(Json::as_str), Some("chat"));
        assert!(report.get("missing").unwrap().as_array().unwrap().is_empty());
        // From a machine where the file is elsewhere: imported, disabled, reported.
        std::fs::remove_file(d.join("Chat.gguf")).unwrap();
        let mut third = config::default_json();
        // A route turned off here stays off, whatever the file brings.
        if let Some(Json::Arr(routes)) = obj_mut(&mut third, &["gateway", "routes"]) {
            for r in routes.iter_mut().filter(|r| str_or(r, "target", "") == "completions") {
                set(r, "enabled", Json::Bool(false));
            }
        }
        let report = import(&mut third, &file, true, &d).unwrap();
        assert_eq!(report.get("missing").unwrap().len(), 1);
        assert_eq!(get(&third, &["llm", "models"]).unwrap().at(0).unwrap().get("enabled"), Some(&Json::Bool(false)));
        let routes = get(&third, &["gateway", "routes"]).unwrap().as_array().unwrap();
        assert!(routes.iter().filter(|r| str_or(r, "target", "") == "completions").all(|r| !bool_or(r, "enabled", true)));
        // Another machine's ffmpeg is not taken when it is not here.
        let mut foreign = file.clone();
        if let Some(v) = obj_mut(&mut foreign, &["video"]) {
            set(v, "ffmpeg", Json::str("Z:\\nowhere\\ffmpeg.exe"));
        } else if let Json::Obj(f) = &mut foreign {
            f.push(("video".into(), Json::obj([("ffmpeg", Json::str("Z:\\nowhere\\ffmpeg.exe"))])));
        }
        let before = get(&third, &["media", "video", "ffmpeg"]).cloned();
        let report = import(&mut third, &foreign, false, &d).unwrap();
        assert!(report.get("missing").unwrap().as_array().unwrap().iter().any(|m| str_or(m, "field", "") == "ffmpeg"));
        assert_eq!(get(&third, &["media", "video", "ffmpeg"]).cloned(), before);
        assert!(import(&mut third, &Json::parse(b"{}").unwrap(), false, &d).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_disabled_route_is_re_enabled_rather_than_duplicated() {
        let mut cfg = config::default_json();
        if let Some(Json::Arr(routes)) = obj_mut(&mut cfg, &["gateway", "routes"]) {
            for r in routes.iter_mut().filter(|r| str_or(r, "target", "") == "images") {
                set(r, "enabled", Json::Bool(false));
            }
        }
        ensure_route(&mut cfg, "images");
        let routes = get(&cfg, &["gateway", "routes"]).unwrap().as_array().unwrap();
        let images: Vec<_> = routes.iter().filter(|r| str_or(r, "target", "") == "images").collect();
        assert_eq!(images.len(), 1);
        assert!(bool_or(images[0], "enabled", false));
    }
}
