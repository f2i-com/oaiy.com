//! Adding a picked file to the configuration: detect what it is, register it in
//! the section whose endpoint serves it, fill the parts it needs from models
//! already configured (or from files beside it), and make sure a route for that
//! endpoint is enabled.

use crate::config;
use crate::models::detect::{self, Detected, Role};
use crate::util::{bool_or, set, str_or};
use oaiy_engine::json::Json;
use std::path::{Path, PathBuf};

/// Where an entry lives: its section (`llm`, `image`, `video`, `speech`, `music`,
/// `sound` or `model3d`) and its name.
#[derive(Debug, PartialEq)]
pub struct Added {
    pub section: &'static str,
    pub name: String,
    pub missing: Vec<String>,
    pub enabled: bool,
}

pub(crate) fn obj_mut<'a>(v: &'a mut Json, path: &[&str]) -> Option<&'a mut Json> {
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

/// Files near `path` (its folder, and `levels - 1` folders above it, and the
/// folders in each) for which `accept` finds a part. Header reads only; bounded.
fn nearby(path: &Path, levels: usize, accept: impl Fn(&Path, &Detected) -> bool) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut at = if path.is_dir() { Some(path.to_path_buf()) } else { path.parent().map(Path::to_path_buf) };
    for _ in 0..levels {
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
            let interesting = c.is_dir() || ext == "safetensors" || ext == "gguf" || c.file_name().is_some_and(|n| n == "tokenizer.json" || detect::readable_pth(&n.to_string_lossy()));
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
            // (else the folder itself: a model folder keeps its tokenizer beside its weights, FLUX 3's text encoder's)
            if probe != c {
                if let Ok(found) = detect::detect(&c) {
                    if accept(&c, &found) {
                        return Some(c);
                    }
                }
            }
        }
    }
    None
}

fn field_of(d: &Detected, key: &str) -> Option<Json> {
    d.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// The part named by `missing` for a new entry: from the picked one's own folder first (what came with it: a
/// downloaded model's own VAE and text encoder), then from another configured model of the same kind, then from the
/// folders above it (where another downloaded model's part may be a different release's).
fn companion(cfg: &Json, section: &str, entry: &Json, missing: &str, picked: &Path) -> Option<Json> {
    let arch = str_or(entry, "architecture", "");
    let family = str_or(entry, "family", "");
    // LTX 2.3 and its fine-tunes share Gemma 3; LTX 2.5 has its own encoder.
    let same_group = |f: &str| (f == "ltx-2.5") == (family == "ltx-2.5");
    if let Some(v) = companion_nearby(section, &arch, family, missing, picked, 1) {
        return Some(v);
    }
    for (_, other) in get(cfg, &["media", section, "models"]).map(|m| m.members().collect::<Vec<_>>()).unwrap_or_default() {
        let compatible = match section {
            "image" => str_or(other, "architecture", "qwen-image") == arch,
            _ => same_group(str_or(other, "family", "")),
        };
        if let (true, Some(v)) = (compatible, other.get(missing).filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))) {
            return Some(v.clone());
        }
    }
    companion_nearby(section, &arch, family, missing, picked, 3)
}

/// The part named by `missing` among the files near `picked` (`levels`: its folder alone, or more above it).
fn companion_nearby(section: &str, arch: &str, family: &str, missing: &str, picked: &Path, levels: usize) -> Option<Json> {
    let want: &dyn Fn(&Detected) -> bool = match (section, missing) {
        ("image", "base") => &|d| d.format == "diffusers" && field_of(d, "base").is_some(),
        ("image", "tokenizer") if arch == "flux2-klein-4b" => &|d| d.kind() == "klein_tokenizer",
        ("image", "text_encoder") if arch == "flux2-klein-4b" => &|d| d.kind() == "klein_text_encoder",
        ("image", "vae") if arch == "flux2-klein-4b" || arch == "flux3-klein" => &|d| d.kind() == "flux2_vae",
        ("image", "vae") if arch == "moss-image" => &|d| matches!(d.kind(), "moss_vae" | "flux2_vae"),
        ("image", "text_encoder") if arch == "moss-image" => &|d| d.kind() == "moss_text_encoder",
        ("image", "tokenizer") if arch == "moss-image" => &|d| d.kind() == "moss_tokenizer",
        ("image", "text_encoder") if arch == "flux3-klein" => &|d| d.kind() == "flux3_text_encoder",
        ("image", "tokenizer") => &|d| d.kind() == "clip_tokenizer",
        ("video", "tokenizer") => &|d| d.kind() == "tokenizer",
        ("video", "vae") => &|d| d.kind() == "vae",
        ("video", "audio_vae") => &|d| d.kind() == "audio_vae",
        ("video", "text_encoder") if family == "ltx-2.5" => &|d| d.kind() == "video_text_encoder" && d.summary.contains("2.5"),
        ("video", "text_encoder") => &|d| d.kind() == "video_text_encoder" && !d.summary.contains("2.5"),
        _ => return None,
    };
    let found = nearby(picked, levels, |_, d| want(d))?;
    let d = detect::detect(&found).ok()?;
    let key = if missing == "base" { "base" } else { missing };
    field_of(&d, key).or_else(|| Some(Json::str(found.to_string_lossy())))
}

/// A Qwen3-TTS folder: the VoiceDesign or Base half of a speech model. It
/// completes the chosen (or the first) model lacking that half, else starts
/// one.
fn attach_speech(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    if d.kind() == "speech_breeze" {
        return attach_breeze(cfg, d, target);
    }
    let field = if d.kind() == "speech_design" { "design" } else { "base" };
    let value = field_of(d, field).ok_or("detected part has no path")?;
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "speech", "models"]) else { return Err("no speech section".into()) };
    let chosen = models
        .iter()
        .position(|(n, _)| target.is_some_and(|(_, t)| n == t))
        .or_else(|| models.iter().position(|(_, m)| str_or(m, field, "").trim().is_empty()));
    let name = match chosen {
        Some(i) => {
            set(&mut models[i].1, field, value);
            set(&mut models[i].1, "enabled", Json::Bool(true));
            models[i].0.clone()
        }
        None => {
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            let name = unique(&taken, "qwen3-tts");
            models.push((name.clone(), Json::obj([(field, value), ("enabled", Json::Bool(true))])));
            name
        }
    };
    let section = obj_mut(cfg, &["media", "speech"]).ok_or("no speech section")?;
    if str_or(section, "default_model", "").is_empty() {
        set(section, "default_model", Json::str(&name));
    }
    add_route_if_absent(cfg, "speech");
    add_route_if_absent(cfg, "voices");
    Ok(Added { section: "speech", name, missing: Vec::new(), enabled: true })
}

/// A Breeze TTS 2 folder is a speech model of its own (or fills the chosen one).
fn attach_breeze(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    let value = field_of(d, "breeze").ok_or("detected part has no path")?;
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "speech", "models"]) else { return Err("no speech section".into()) };
    let chosen = models.iter().position(|(n, _)| target.is_some_and(|(_, t)| n == t));
    let name = match chosen {
        Some(i) => {
            set(&mut models[i].1, "breeze", value);
            set(&mut models[i].1, "enabled", Json::Bool(true));
            models[i].0.clone()
        }
        None => {
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            let name = unique(&taken, "breeze-tts-2");
            models.push((name.clone(), Json::obj([("breeze", value), ("enabled", Json::Bool(true))])));
            name
        }
    };
    let section = obj_mut(cfg, &["media", "speech"]).ok_or("no speech section")?;
    if str_or(section, "default_model", "").is_empty() {
        set(section, "default_model", Json::str(&name));
    }
    add_route_if_absent(cfg, "speech");
    add_route_if_absent(cfg, "voices");
    Ok(Added { section: "speech", name, missing: Vec::new(), enabled: true })
}

/// A MiniMax Music 3 folder starts a music model (or fills the chosen one);
/// a quantized language model goes to the chosen model, else the first
/// without one.
fn attach_music(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    let field = if d.kind() == "music_model" { "path" } else { "language_model" };
    let value = field_of(d, field).ok_or("detected part has no path")?;
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "music", "models"]) else { return Err("no music section".into()) };
    let chosen = models.iter().position(|(n, _)| target.is_some_and(|(_, t)| n == t)).or_else(|| {
        if field == "path" {
            models.iter().position(|(_, m)| str_or(m, "path", "").trim().is_empty())
        } else {
            models.iter().position(|(_, m)| str_or(m, "language_model", "").trim().is_empty() && !str_or(m, "path", "").trim().is_empty())
        }
    });
    let name = match chosen {
        Some(i) => {
            set(&mut models[i].1, field, value);
            set(&mut models[i].1, "enabled", Json::Bool(true));
            models[i].0.clone()
        }
        None if field == "path" => {
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            let name = unique(&taken, "minimax-music3");
            models.push((name.clone(), Json::obj([("path", value), ("enabled", Json::Bool(true))])));
            name
        }
        None => return Err(format!("{}: add the MiniMax Music 3 folder first", d.summary)),
    };
    let section = obj_mut(cfg, &["media", "music"]).ok_or("no music section")?;
    if str_or(section, "default_model", "").is_empty() {
        set(section, "default_model", Json::str(&name));
    }
    add_route_if_absent(cfg, "music");
    Ok(Added { section: "music", name, missing: Vec::new(), enabled: true })
}

/// A MOSS-Image distillation: added to a MOSS-Image model, that model makes its pictures in the distillation's passes;
/// added by itself, it is a model of its own, a MOSS-Image model's parts (the image default's where that is one, else
/// the first's) with it, named after that model and its passes. The model's steps are its passes.
fn attach_moss_distill(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    let value = field_of(d, "distill").ok_or("detected distillation has no path")?;
    let passes = field_of(d, "steps").and_then(|v| v.as_i64()).filter(|n| *n > 0).unwrap_or(8);
    let default = get(cfg, &["media", "image", "default_model"]).and_then(Json::as_str).unwrap_or("").to_string();
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "image", "models"]) else { return Err("no image models".into()) };
    let moss = |m: &Json| str_or(m, "architecture", "") == "moss-image";
    let name = match target {
        Some((_, t)) => {
            let model = models.iter().find(|(n, _)| n == t).map(|(_, m)| m).ok_or_else(|| format!("no image model {t}"))?;
            if !moss(model) {
                return Err(format!("{} is for MOSS-Image models; {t} is not one", d.summary));
            }
            t.to_string()
        }
        None => {
            let base = models.iter().find(|(n, m)| *n == default && moss(m)).or_else(|| models.iter().find(|(_, m)| moss(m)))
                .ok_or("add a MOSS-Image model first (its transformer, text encoder and VAE), then its distillation")?;
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            let name = unique(&taken, &format!("{}-{passes}-step", base.0));
            // (its parts, not the guided sampler's settings: the distillation's passes take none of them)
            let mut entry = Json::Obj(base.1.members().filter(|(k, _)| !["cfg", "shift", "uncond_skip_layers", "negative_prompt", "steps"].contains(k)).map(|(k, v)| (k.to_string(), v.clone())).collect());
            set(&mut entry, "enabled", Json::Bool(true));
            models.push((name.clone(), entry));
            name
        }
    };
    let model = models.iter_mut().find(|(n, _)| *n == name).map(|(_, m)| m).ok_or("the model went")?;
    set(model, "distill", value);
    set(model, "steps", Json::Int(passes));
    let missing = missing_fields("image", model);
    let enabled = missing.is_empty() && bool_or(model, "enabled", true);
    set(model, "enabled", Json::Bool(enabled));
    add_route_if_absent(cfg, "images");
    Ok(Added { section: "image", name, missing, enabled })
}

/// A MOSS-SoundEffect folder is a sound model of its own (or fills the chosen one).
fn attach_sound(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>) -> Result<Added, String> {
    let value = field_of(d, "path").ok_or("detected part has no path")?;
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "sound", "models"]) else { return Err("no sound section".into()) };
    let chosen = models.iter().position(|(n, _)| target.is_some_and(|(_, t)| n == t)).or_else(|| models.iter().position(|(_, m)| str_or(m, "path", "").trim().is_empty()));
    let name = match chosen {
        Some(i) => {
            set(&mut models[i].1, "path", value);
            set(&mut models[i].1, "enabled", Json::Bool(true));
            models[i].0.clone()
        }
        None => {
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            let name = unique(&taken, "moss-soundeffect");
            models.push((name.clone(), Json::obj([("path", value), ("enabled", Json::Bool(true))])));
            name
        }
    };
    let section = obj_mut(cfg, &["media", "sound"]).ok_or("no sound section")?;
    if str_or(section, "default_model", "").is_empty() {
        set(section, "default_model", Json::str(&name));
    }
    add_route_if_absent(cfg, "sound");
    Ok(Added { section: "sound", name, missing: Vec::new(), enabled: true })
}

/// The fields a 3D model needs: the Pixal3D folder, DINOv3 and NAF.
const MODEL3D_PARTS: [&str; 3] = ["path", "dino", "naf"];
/// The fields that help it when they are there: BiRefNet (cuts objects out of
/// any background) and Real-ESRGAN (enlarges small pictures).
const MODEL3D_HELPERS: [&str; 2] = ["matte", "upscaler"];

/// A 3D model's part from files near `picked` (the release folders usually sit
/// side by side): a DINOv3 folder, NAF's folder or weights, BiRefNet's folder,
/// or Real-ESRGAN's folder or weights.
fn model3d_nearby(picked: &Path, key: &str) -> Option<Json> {
    let kind = match key {
        "dino" => "model3d_dino",
        "naf" => "model3d_naf",
        "matte" => "model3d_matte",
        _ => "model3d_upscaler",
    };
    let found = nearby(picked, 3, |_, d| d.kind() == kind)?;
    field_of(&detect::detect(&found).ok()?, key)
}

/// A Pixal3D folder is a 3D model of its own (or fills the chosen one, else one
/// still waiting for it); DINOv3 and NAF fill the chosen model, else the
/// default or only one, else one lacking them, else start one. Parts a model
/// lacks come from another 3D model, or from folders beside the picked one.
fn attach_model3d(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>, picked: &Path) -> Result<Added, String> {
    let field = match d.kind() {
        "model3d_dino" => "dino",
        "model3d_naf" => "naf",
        "model3d_matte" => "matte",
        "model3d_upscaler" => "upscaler",
        _ => "path",
    };
    let value = field_of(d, field).ok_or("detected part has no path")?;
    let default = get(cfg, &["media", "model3d", "default_model"]).and_then(Json::as_str).unwrap_or("").to_string();
    // What the other 3D models already use, lent to one that lacks it.
    let lent = |key: &str| -> Option<Json> {
        get(cfg, &["media", "model3d", "models"])?.members().find_map(|(_, m)| m.get(key).filter(|v| v.as_str().is_some_and(|s| !s.trim().is_empty())).cloned())
    };
    // The picture tools' BiRefNet and Real-ESRGAN are the 3D model's helpers too.
    let picture = |key: &str| -> Option<Json> {
        let field = if key == "matte" { "background" } else { "upscaler" };
        get(cfg, &["media", "picture", field]).filter(|v| v.as_str().is_some_and(|s| !s.trim().is_empty())).cloned()
    };
    let lend: Vec<(&str, Json)> = MODEL3D_PARTS
        .iter()
        .chain(&MODEL3D_HELPERS)
        .filter(|k| **k != field)
        .filter_map(|k| lent(k).or_else(|| if MODEL3D_HELPERS.contains(k) { picture(k) } else { None }).map(|v| (*k, v)))
        .collect();
    let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "model3d", "models"]) else { return Err("no 3D model section".into()) };
    let lacks = |m: &Json| str_or(m, field, "").trim().is_empty();
    let chosen = models.iter().position(|(n, _)| target.is_some_and(|(_, t)| n == t)).or_else(|| {
        if field == "path" {
            models.iter().position(|(_, m)| lacks(m))
        } else {
            models.iter().position(|(n, _)| *n == default).or_else(|| (models.len() == 1).then_some(0)).or_else(|| models.iter().position(|(_, m)| lacks(m)))
        }
    });
    let i = match chosen {
        Some(i) => i,
        None => {
            let taken: Vec<String> = models.iter().map(|(k, _)| k.clone()).collect();
            models.push((unique(&taken, "pixal3d"), Json::Obj(Vec::new())));
            models.len() - 1
        }
    };
    let (name, model) = &mut models[i];
    let name = name.clone();
    set(model, field, value);
    for key in MODEL3D_PARTS.iter().chain(&MODEL3D_HELPERS).filter(|k| **k != "path") {
        if str_or(model, key, "").trim().is_empty() {
            if let Some(v) = lend.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()).or_else(|| model3d_nearby(picked, key)) {
                set(model, key, v);
            }
        }
    }
    let missing: Vec<String> = MODEL3D_PARTS.iter().filter(|k| str_or(model, k, "").trim().is_empty()).map(|k| k.to_string()).collect();
    let enabled = missing.is_empty();
    set(model, "enabled", Json::Bool(enabled));
    let section = obj_mut(cfg, &["media", "model3d"]).ok_or("no 3D model section")?;
    if enabled && str_or(section, "default_model", "").is_empty() {
        set(section, "default_model", Json::str(&name));
    }
    add_route_if_absent(cfg, "model3d");
    Ok(Added { section: "model3d", name, missing, enabled })
}

/// Components attach to the first model that lacks them (`picked`: where the
/// part is, to look for others beside it).
fn attach(cfg: &mut Json, d: &Detected, target: Option<(&str, &str)>, picked: &Path) -> Result<Added, String> {
    type Fits = Box<dyn Fn(&Json) -> bool>;
    let (section, field, fits): (&str, &str, Fits) = match d.kind() {
        // (added to a Qwen Image model: Qwen3-VL's vision tower for its reference images, where its text encoder is
        // a GGUF of the language model alone)
        "vision_projector" if target.is_some_and(|(s, _)| s == "image") => ("image", "vision", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "vision_projector" => ("llm", "vision_projector", Box::new(|_| true)),
        // (not a distilled Turbo checkpoint: it takes no adapter)
        "adapter" => ("image", "adapter", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image" && !bool_or(m, "distilled", false))),
        kind @ ("klein_lora" | "flux3_lora") => {
            let (arch, what) = if kind == "klein_lora" { ("flux2-klein-4b", "a Klein 4B image model before its style LoRA") } else { ("flux3-klein", "a FLUX 3 Image Klein model before its LoRA") };
            let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "image", "models"]) else { return Err("no image models".into()) };
            let (name, model) = models.iter_mut().find(|(n,m)| str_or(m,"architecture","") == arch && target.is_none_or(|(_,t)| n == t))
                .ok_or(format!("add or select {what}"))?;
            let mut list = model.get("loras").and_then(Json::as_array).unwrap_or(&[]).to_vec();
            let added = field_of(d,"loras").ok_or("detected LoRA has no path")?;
            for value in added.as_array().ok_or("detected LoRA list is invalid")? { if !list.contains(value) { list.push(value.clone()); } }
            set(model,"loras",Json::Arr(list));
            let missing = missing_fields("image",model);
            return Ok(Added { section:"image", name:name.clone(), enabled:missing.is_empty(), missing });
        }
        "klein_text_encoder" => ("image", "text_encoder", Box::new(|m| str_or(m, "architecture", "") == "flux2-klein-4b")),
        "klein_tokenizer" => ("image", "tokenizer", Box::new(|m| str_or(m, "architecture", "") == "flux2-klein-4b")),
        "flux2_vae" => ("image", "vae", Box::new(|m| matches!(str_or(m, "architecture", ""), "flux2-klein-4b" | "flux3-klein" | "moss-image"))),
        "moss_vae" => ("image", "vae", Box::new(|m| str_or(m, "architecture", "") == "moss-image")),
        "moss_text_encoder" => ("image", "text_encoder", Box::new(|m| str_or(m, "architecture", "") == "moss-image")),
        "moss_tokenizer" => ("image", "tokenizer", Box::new(|m| str_or(m, "architecture", "") == "moss-image")),
        "flux3_text_encoder" => ("image", "text_encoder", Box::new(|m| str_or(m, "architecture", "") == "flux3-klein")),
        "image_base" => ("image", "base", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "text_encoder" => ("image", "text_encoder", Box::new(|m| str_or(m, "architecture", "qwen-image") == "qwen-image")),
        "clip_tokenizer" => ("image", "tokenizer", Box::new(|m| str_or(m, "architecture", "") == "sdxl")),
        "tokenizer" => ("video", "tokenizer", Box::new(|m| str_or(m, "family", "") != "ltx-2.5")),
        "vae" => ("video", "vae", Box::new(|_| true)),
        "audio_vae" => {
            // One vocoder serves every LTX model (their audio VAEs share one
            // layout): each that lacks it gets it.
            let value = field_of(d, "audio_vae").ok_or("detected part has no path")?;
            let Some(Json::Obj(models)) = obj_mut(cfg, &["media", "video", "models"]) else { return Err("no video models".into()) };
            let mut named = None;
            for (n, m) in models.iter_mut() {
                let chosen = target.is_some_and(|(_, t)| n == t);
                if (chosen || str_or(m, "audio_vae", "").trim().is_empty()) && !str_or(m, "transformer", "").is_empty() {
                    set(m, "audio_vae", value.clone());
                    named.get_or_insert_with(|| n.clone());
                }
            }
            let name = named.ok_or_else(|| format!("{}: add an LTX video model first", d.summary))?;
            return Ok(Added { section: "video", name, missing: Vec::new(), enabled: true });
        }
        "video_text_encoder" => {
            let v25 = d.summary.contains("2.5");
            ("video", "text_encoder", Box::new(move |m| (str_or(m, "family", "") == "ltx-2.5") == v25))
        }
        "speech_design" | "speech_base" | "speech_breeze" => return attach_speech(cfg, d, target),
        "music_model" | "music_lm" => return attach_music(cfg, d, target),
        "sound_model" => return attach_sound(cfg, d, target),
        "moss_distill" => return attach_moss_distill(cfg, d, target),
        "model3d_matte" | "model3d_upscaler" => {
            // Picture tools first (they need nothing else), then any 3D model.
            let (field, key) = if d.kind() == "model3d_matte" { ("background", "matte") } else { ("upscaler", "upscaler") };
            let value = field_of(d, key).ok_or("detected part has no path")?;
            if let Some(section) = obj_mut(cfg, &["media", "picture"]) {
                if target.is_none_or(|(s, _)| s == "picture") || str_or(section, field, "").trim().is_empty() {
                    set(section, field, value);
                }
            }
            add_route_if_absent(cfg, if field == "background" { "background" } else { "upscale" });
            // With no 3D model yet it serves the picture tools alone (a 3D model added later borrows it).
            let no_3d = get(cfg, &["media", "model3d", "models"]).is_none_or(|m| m.len() == 0);
            if target.is_some_and(|(s, _)| s == "picture") || (target.is_none() && no_3d) {
                return Ok(Added { section: "picture", name: field.to_string(), missing: Vec::new(), enabled: true });
            }
            return attach_model3d(cfg, d, target, picked);
        }
        "model3d" | "model3d_dino" | "model3d_naf" => return attach_model3d(cfg, d, target, picked),
        "ffmpeg" => {
            let value = field_of(d, "ffmpeg").unwrap_or(Json::Null);
            set(obj_mut(cfg, &["media", "video"]).ok_or("no video section")?, "ffmpeg", value);
            return Ok(Added { section: "video", name: "ffmpeg".into(), missing: Vec::new(), enabled: true });
        }
        other => return Err(format!("do not know where a {other} goes")),
    };
    // (a projector's path is detected as an LLM's `vision_projector`, a Qwen Image model's `vision` too)
    let value = field_of(d, field).or_else(|| field_of(d, "vision_projector").filter(|_| field == "vision")).ok_or("detected part has no path")?;
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
    if !fits(model) { return Err(format!("{} is incompatible with the selected model", d.summary)); }
    set(model, field, value);
    let missing = missing_fields(section, model);
    let enabled = missing.is_empty();
    set(model, "enabled", Json::Bool(enabled));
    Ok(Added { section: if section == "image" { "image" } else { "video" }, name: name.clone(), missing, enabled })
}

/// Whether MOSS-Image's text encoder at `te` brings its own tokenizer: a GGUF's, or a tokenizer.json beside its weights
/// (as the worker finds it).
fn moss_tokenizer_beside(te: &Path) -> bool {
    te.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        || (if te.is_dir() { te.join("tokenizer.json") } else { te.with_file_name("tokenizer.json") }).is_file()
}

/// Required fields an entry still lacks (mirrors `config::validate`).
pub fn missing_fields(section: &str, m: &Json) -> Vec<String> {
    let empty = |k: &str| str_or(m, k, "").trim().is_empty();
    let mut out = Vec::new();
    if section == "image" {
        if str_or(m, "architecture", "qwen-image") == "sdxl" {
            out.extend(["checkpoint", "tokenizer"].into_iter().filter(|k| empty(k)).map(String::from));
        } else if str_or(m, "architecture", "qwen-image") == "flux2-klein-4b" {
            out.extend(["transformer", "text_encoder", "vae", "tokenizer"].into_iter().filter(|k| empty(k)).map(String::from));
        } else if str_or(m, "architecture", "qwen-image") == "flux3-klein" {
            out.extend(["transformer", "text_encoder", "vae"].into_iter().filter(|k| empty(k)).map(String::from));
        } else if str_or(m, "architecture", "qwen-image") == "moss-image" {
            out.extend(["transformer", "text_encoder", "vae"].into_iter().filter(|k| empty(k)).map(String::from));
            if empty("tokenizer") && !empty("text_encoder") && !moss_tokenizer_beside(Path::new(str_or(m, "text_encoder", ""))) {
                out.push("tokenizer".into());
            }
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
    // (a Qwen3-VL GGUF added to a Qwen Image model is its text encoder, whatever the file is called)
    let to_image = target.is_some_and(|(s, m)| s == "image" && get(cfg, &["media", "image", "models", m]).is_some_and(|e| str_or(e, "architecture", "qwen-image") == "qwen-image"));
    let d = if to_image { detect::as_qwen_image_encoder(&d, path).unwrap_or(d) } else { d };
    let to_moss = target.is_some_and(|(s, m)| s == "image" && get(cfg, &["media", "image", "models", m]).is_some_and(|e| str_or(e, "architecture", "") == "moss-image"));
    let d = if to_moss { detect::as_moss_encoder(&d, path).unwrap_or(d) } else { d };
    let wanted = name.map(str::to_string).filter(|n| !n.trim().is_empty()).unwrap_or_else(|| detect::model_name(path));
    let added = match &d.role {
        Role::Component { .. } => attach(cfg, &d, target, path)?,
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
            if section == "video" && str_or(&entry, "audio_vae", "").is_empty() {
                // Optional: the soundtrack's decoder, from another LTX model or nearby.
                if let Some(a) = companion(cfg, "video", &entry, "audio_vae", path) {
                    set(&mut entry, "audio_vae", a);
                }
            }
            if section == "image" && str_or(&entry, "architecture", "") == "moss-image" && str_or(&entry, "tokenizer", "").is_empty()
                && !moss_tokenizer_beside(Path::new(str_or(&entry, "text_encoder", "")))
            {
                if let Some(t) = companion(cfg, "image", &entry, "tokenizer", path) {
                    set(&mut entry, "tokenizer", t);
                }
            }
            if section == "image" && str_or(&entry, "architecture", "") == "qwen-image" && str_or(&entry, "adapter", "").is_empty() && !bool_or(&entry, "distilled", false) {
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

/// Keys holding file paths, per section, for export and import. (A model's `loras`, a list of `{path, strength}`, are
/// paths too: [`LORAS`].)
const LLM_PATHS: [&str; 3] = ["path", "vision_projector", "lora"];
const IMAGE_PATHS: [&str; 10] = ["base", "transformer", "safetensors_transformer", "adapter", "text_encoder", "vision", "checkpoint", "tokenizer", "vae", "distill"];
const VIDEO_PATHS: [&str; 5] = ["transformer", "text_encoder", "vae", "tokenizer", "audio_vae"];
const SPEECH_PATHS: [&str; 3] = ["design", "base", "breeze"];
const MUSIC_PATHS: [&str; 2] = ["path", "language_model"];
const SOUND_PATHS: [&str; 1] = ["path"];
const MODEL3D_PATHS: [&str; 5] = ["path", "dino", "naf", "matte", "upscaler"];
const LORAS: &str = "loras";
/// The media sections a models file carries, each with its path keys, its settings beside `MEDIA_SETTINGS` and the
/// routes its models are served on.
const MEDIA_SECTIONS: [(&str, &[&str], &[&str], &[&str]); 6] = [
    ("image", &IMAGE_PATHS, &[], &["images", "edits", "files"]),
    ("video", &VIDEO_PATHS, &["fps", "ffmpeg"], &["videos", "files"]),
    ("speech", &SPEECH_PATHS, &[], &["speech", "voices"]),
    ("music", &MUSIC_PATHS, &[], &["music"]),
    ("sound", &SOUND_PATHS, &[], &["sound"]),
    ("model3d", &MODEL3D_PATHS, &[], &["model3d"]),
];
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
    if let Some(Json::Arr(loras)) = m.get(LORAS).cloned() {
        set(&mut m, LORAS, Json::Arr(loras.iter().map(|l| absolute(root, l, &["path"])).collect()));
    }
    m
}

/// Every configured model (language, image, video, speech, music, sound effect and 3D), with their parts, LoRAs,
/// settings, defaults and memory settings, as a file another install can import. Paths are made absolute so the file
/// does not depend on where this studio lives.
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
    let mut file = vec![
        ("oaiy_models".to_string(), Json::Int(1)),
        ("exported_at".to_string(), Json::Int(crate::util::now() as i64)),
        ("llm".to_string(), Json::obj([("default_model", llm.get("default_model").cloned().unwrap_or(Json::str(""))), ("models", Json::Arr(llm_models))])),
    ];
    file.extend(MEDIA_SECTIONS.iter().map(|(kind, keys, extra, _)| (kind.to_string(), media(kind, keys, extra))));
    Json::Obj(file)
}

/// Bring an exported file in. `replace` clears the current models of each section the file has first (a file from
/// before a section was exported leaves that section's models as they are); otherwise models merge by name (the
/// file's version wins). Entries whose files are not on this machine come in disabled and are reported, so a file
/// from another PC imports cleanly and shows what to repoint.
pub fn import(cfg: &mut Json, doc: &Json, replace: bool, root: &Path) -> Result<Json, String> {
    if doc.get("oaiy_models").and_then(Json::as_i64) != Some(1) {
        return Err("not an OAIY models file (missing \"oaiy_models\": 1)".into());
    }
    let mut missing = Vec::new();
    let mut tools_missing = Vec::new();
    let mut counts: Vec<(&str, i64)> = Vec::new();
    // Relative paths mean what they mean everywhere else: beside the studio.
    let exists = |p: &str| config::resolve(root, p).exists();
    let mut check = |section: &str, name: &str, m: &mut Json, keys: &[&str]| {
        let mut ok = true;
        let mut gone = |field: &str, p: &str| {
            missing.push(Json::obj([("section", Json::str(section)), ("model", Json::str(name)), ("field", Json::str(field)), ("path", Json::str(p))]));
            ok = false;
        };
        for key in keys {
            if let Some(p) = m.get(key).and_then(Json::as_str).filter(|p| !p.trim().is_empty() && !exists(p)) {
                gone(key, p);
            }
        }
        for l in m.get(LORAS).and_then(Json::as_array).unwrap_or(&[]) {
            if let Some(p) = l.get("path").and_then(Json::as_str).filter(|p| !p.trim().is_empty() && !exists(p)) {
                gone(LORAS, p);
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
        let mut n = 0;
        for m in incoming {
            let name = str_or(m, "name", "").to_string();
            if name.is_empty() || str_or(m, "path", "").is_empty() {
                continue;
            }
            let mut m = m.clone();
            check("llm", &name, &mut m, &LLM_PATHS);
            models.retain(|x| str_or(x, "name", "") != name);
            models.push(m);
            n += 1;
        }
        counts.push(("llm", n));
    }
    if let Some(d) = get(doc, &["llm", "default_model"]).and_then(Json::as_str).filter(|d| !d.is_empty()) {
        let known = get(cfg, &["llm", "models"]).and_then(Json::as_array).is_some_and(|m| m.iter().any(|x| str_or(x, "name", "") == d));
        if known {
            set(obj_mut(cfg, &["llm"]).ok_or("no llm section")?, "default_model", Json::str(d));
        }
    }
    let mut needed: Vec<&str> = Vec::new();
    if counts.iter().any(|&(_, n)| n > 0) {
        needed.extend(["chat", "completions", "models"]);
    }
    for (kind, keys, _, routes) in MEDIA_SECTIONS {
        let Some(section_in) = doc.get(kind) else { continue };
        let mut n = 0;
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
                n += 1;
            }
        }
        counts.push((kind, n));
        if n > 0 {
            needed.extend(routes);
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
    for target in needed {
        add_route_if_absent(cfg, target);
    }
    config::validate(cfg)?;
    missing.extend(tools_missing);
    // (every section's count, 0 where the file had none of it)
    let count = |kind: &str| counts.iter().find(|(k, _)| *k == kind).map_or(0, |&(_, n)| n);
    let mut report: Vec<(String, Json)> = std::iter::once("llm").chain(MEDIA_SECTIONS.iter().map(|s| s.0)).map(|k| (k.to_string(), Json::Int(count(k)))).collect();
    report.push(("missing".into(), Json::Arr(missing)));
    Ok(Json::Obj(report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oaiy-studio-registry-{tag}-{}", std::process::id()));
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

    /// A GGUF with `general.architecture` `arch` and the integer key/values `ints` (`{arch}.` before each key).
    fn gguf_with(path: &Path, arch: &str, ints: &[(&str, u32)]) {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((1 + ints.len() as u64).to_le_bytes());
        let string = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        };
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, arch);
        for (k, v) in ints {
            string(&mut b, &format!("{arch}.{k}"));
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        std::fs::write(path, b).unwrap();
    }

    /// MOSS-Image: OpenMOSS's release folder picked is a whole image model (its transformer, VAE, tokenizer and text
    /// encoder folders); its 4-bit set as the catalog adds it (a GGUF of its transformer, then its VAE, its tokenizer and
    /// a Qwen3.5-9B GGUF, each to that model) is one too, the Qwen3.5 GGUF its text encoder where alone it would be a
    /// chat model; Qwen's own 9B snapshot (shards, no tokenizer) its text encoder folder, the tokenizer then sought.
    #[test]
    fn moss_images_release_and_its_quantized_parts_are_one_model() {
        let d = tmp("moss-image");
        let tensor = |name: &str, shape: &str| format!(r#""{name}":{{"dtype":"BF16","shape":{shape},"data_offsets":[0,0]}}"#);
        let release = d.join("MOSS-Image-SFT");
        for sub in ["transformer", "vae", "tokenizer", "text_encoder"] {
            std::fs::create_dir_all(release.join(sub)).unwrap();
        }
        std::fs::write(release.join("model_index.json"), r#"{"_class_name":"MOSSImagePipeline"}"#).unwrap();
        std::fs::write(release.join("transformer/config.json"), r#"{"_class_name":"MOSSImageTransformer"}"#).unwrap();
        let transformer = [tensor("video_embedder.weight", "[3072,128]"), tensor("text_embedder.proj.weight", "[3072,4096]"), tensor("shared_block_modulation.weight", "[18432,3072]")].join(",");
        safetensors(&release.join("transformer/diffusion_pytorch_model-00001-of-00003.safetensors"), &format!("{{{transformer}}}"));
        safetensors(&release.join("vae/diffusion_pytorch_model.safetensors"), &format!("{{{},{}}}", tensor("bn.running_mean", "[128]"), tensor("decoder.conv_in.weight", "[384,32,3,3]")));
        std::fs::write(release.join("tokenizer/tokenizer.json"), r#"{"added_tokens":["<|im_start|>","<|audio_start|>"]}"#).unwrap();
        std::fs::write(release.join("text_encoder/config.json"), r#"{"model_type":"qwen3_5","text_config":{"hidden_size":4096}}"#).unwrap();
        safetensors(&release.join("text_encoder/model-00001-of-00004.safetensors"), &format!("{{{}}}", tensor("language_model.embed_tokens.weight", "[248320,4096]")));
        let mut cfg = config::default_json();
        let (a, found) = add(&mut cfg, &release, None, None).unwrap();
        assert!(matches!(found.role, detect::Role::Image { architecture: "moss-image" }), "{}", found.summary);
        assert_eq!((a.section, a.enabled), ("image", true), "{:?}", a.missing);
        let m = get(&cfg, &["media", "image", "models", &a.name]).unwrap();
        for (key, want) in [("transformer", release.join("transformer")), ("text_encoder", release.join("text_encoder")), ("vae", release.join("vae/diffusion_pytorch_model.safetensors")), ("tokenizer", release.join("tokenizer/tokenizer.json"))] {
            assert_eq!(Path::new(str_or(m, key, "")), want.as_path(), "{key}");
        }
        config::validate(&cfg).unwrap();
        // the 4-bit set, as the catalog adds it
        let parts = d.join("parts");
        std::fs::create_dir_all(&parts).unwrap();
        gguf(&parts.join("MOSS-Image-SFT-Q4_K_M.gguf"), "moss_image");
        gguf_with(&parts.join("Qwen3.5-9B-Q4_K_M.gguf"), "qwen35", &[("embedding_length", 4096), ("block_count", 32)]);
        let mut cfg = config::default_json();
        let mut first: Option<(&str, String)> = None;
        for part in [parts.join("MOSS-Image-SFT-Q4_K_M.gguf"), release.join("vae/diffusion_pytorch_model.safetensors"), release.join("tokenizer/tokenizer.json"), parts.join("Qwen3.5-9B-Q4_K_M.gguf")] {
            let (a, _) = add(&mut cfg, &part, None, first.as_ref().map(|(s, n)| (*s, n.as_str()))).unwrap();
            first.get_or_insert((a.section, a.name.clone()));
        }
        let (_, name) = first.unwrap();
        let m = get(&cfg, &["media", "image", "models", &name]).unwrap();
        assert_eq!((str_or(m, "architecture", ""), bool_or(m, "enabled", false)), ("moss-image", true));
        assert_eq!(Path::new(str_or(m, "text_encoder", "")), parts.join("Qwen3.5-9B-Q4_K_M.gguf").as_path());
        assert!(get(&cfg, &["llm", "models"]).and_then(Json::as_array).is_some_and(|l| l.is_empty()), "the Qwen3.5 GGUF is no chat model here");
        config::validate(&cfg).unwrap();
        // (the same GGUF alone: a chat model)
        let mut alone = config::default_json();
        assert_eq!(add(&mut alone, &parts.join("Qwen3.5-9B-Q4_K_M.gguf"), None, None).unwrap().0.section, "llm");
        // Qwen's own snapshot of the 9B: shards with no config or tokenizer of their own
        let snapshot = d.join("Qwen3.5-9B");
        std::fs::create_dir_all(&snapshot).unwrap();
        safetensors(&snapshot.join("model.safetensors-00001-of-00004.safetensors"), &format!("{{{}}}", tensor("model.language_model.embed_tokens.weight", "[248320,4096]")));
        let found = detect::detect(&snapshot).unwrap();
        assert_eq!(found.kind(), "moss_text_encoder");
        assert_eq!(found.fields.iter().find(|(k, _)| k == "text_encoder").and_then(|(_, v)| v.as_str()).map(PathBuf::from), Some(snapshot.clone()));
        // (beside a GGUF of the transformer, it is found as its text encoder; its tokenizer is not beside it, and
        // MOSS-Image's then picked is)
        let mut cfg = config::default_json();
        let (a, _) = add(&mut cfg, &parts.join("MOSS-Image-SFT-Q4_K_M.gguf"), None, None).unwrap();
        assert_eq!(a.missing, ["vae", "tokenizer"]);
        let m = get(&cfg, &["media", "image", "models", &a.name]).unwrap();
        assert_eq!(Path::new(str_or(m, "text_encoder", "")), snapshot.as_path());
        let (b, _) = add(&mut cfg, &release.join("tokenizer/tokenizer.json"), None, Some(("image", &a.name))).unwrap();
        assert_eq!(b.missing, ["vae"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A MOSS-Image distillation (its one file, or its folder of three, any of them picked): added by itself it is a
    /// model of its own, the image default MOSS-Image model's parts with it and its passes its steps, the guided
    /// sampler's settings left out; added to a MOSS-Image model, that model takes it. With no MOSS-Image model, or to
    /// another kind, it is refused; its path travels in an export.
    #[test]
    fn a_moss_image_distillation_is_a_model_of_its_own_or_the_chosen_ones() {
        let d = tmp("moss-distill");
        let tensor = |name: &str, shape: &str| format!(r#""{name}":{{"dtype":"F32","shape":{shape},"data_offsets":[0,0]}}"#);
        let factors = [tensor("layers.0.proj.ffn_down.lora_A", "[128,8192]"), tensor("layers.0.proj.ffn_down.lora_B", "[3072,128]")].join(",");
        let codes = [tensor("latent_mix.Q", "[128,128]"), tensor("step_codes", "[8]")].join(",");
        let whole = d.join("generator.safetensors");
        safetensors(&whole, &format!("{{{factors},{codes},{}}}", tensor("time_embedding.2.weight", "[3072,3072]")));
        let three = d.join("three");
        std::fs::create_dir_all(&three).unwrap();
        safetensors(&three.join("lora.safetensors"), &format!("{{{}}}", tensor("diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight", "[128,3072]")));
        safetensors(&three.join("time_embedding.safetensors"), &format!("{{{},{}}}", tensor("time_embedding.2.weight", "[3072,3072]"), tensor("video_head_modulation.bias", "[6144]")));
        safetensors(&three.join("dataspace.safetensors"), &format!("{{{codes}}}"));
        for (picked, at) in [(whole.clone(), whole.clone()), (d.clone(), whole.clone()), (three.join("lora.safetensors"), three.clone()), (three.join("dataspace.safetensors"), three.clone()), (three.clone(), three.clone())] {
            let found = detect::detect(&picked).unwrap();
            assert_eq!(found.kind(), "moss_distill", "{}: {}", picked.display(), found.summary);
            assert_eq!(found.fields.iter().find(|(k, _)| k == "distill").and_then(|(_, v)| v.as_str()).map(PathBuf::from), Some(at), "{}", picked.display());
            assert!(found.summary.contains("8 passes"), "{}", found.summary);
        }
        let mut cfg = config::default_json();
        assert!(add(&mut cfg, &whole, None, None).unwrap_err().contains("add a MOSS-Image model first"));
        let base = Json::parse(br#"{"architecture":"moss-image","transformer":"t","text_encoder":"te.gguf","vae":"vae.safetensors","steps":50,"cfg":4,"negative_prompt":"blurry","enabled":true}"#).unwrap();
        let other = Json::parse(br#"{"architecture":"sdxl","checkpoint":"x.safetensors","tokenizer":"t.json","enabled":true}"#).unwrap();
        let Some(Json::Obj(models)) = obj_mut(&mut cfg, &["media", "image", "models"]) else { panic!() };
        models.extend([("xl".to_string(), other), ("moss-image-sft".to_string(), base)]);
        set(obj_mut(&mut cfg, &["media", "image"]).unwrap(), "default_model", Json::str("xl"));
        // by itself: a model of its own, the first MOSS-Image model's parts (the default is none)
        let (a, _) = add(&mut cfg, &whole, None, None).unwrap();
        assert_eq!((a.section, a.name.as_str(), a.enabled), ("image", "moss-image-sft-8-step", true));
        let m = get(&cfg, &["media", "image", "models", "moss-image-sft-8-step"]).unwrap();
        assert_eq!((str_or(m, "transformer", ""), str_or(m, "text_encoder", ""), str_or(m, "distill", "")), ("t", "te.gguf", whole.to_str().unwrap()));
        assert_eq!(m.get("steps").and_then(Json::as_i64), Some(8));
        assert!(["cfg", "negative_prompt", "shift"].iter().all(|k| m.get(k).is_none()), "the guided sampler's settings stay with the other");
        assert_eq!(get(&cfg, &["media", "image", "models", "moss-image-sft", "steps"]).and_then(Json::as_i64), Some(50), "the base model as it was");
        config::validate(&cfg).unwrap();
        // to a chosen MOSS-Image model: that one takes it; to another kind, refused
        let (b, _) = add(&mut cfg, &three, None, Some(("image", "moss-image-sft"))).unwrap();
        assert_eq!(b.name, "moss-image-sft");
        let m = get(&cfg, &["media", "image", "models", "moss-image-sft"]).unwrap();
        assert_eq!((str_or(m, "distill", ""), m.get("steps").and_then(Json::as_i64)), (three.to_str().unwrap(), Some(8)));
        assert!(add(&mut cfg, &whole, None, Some(("image", "xl"))).unwrap_err().contains("for MOSS-Image models"));
        config::validate(&cfg).unwrap();
        let file = export(&cfg, Path::new("/unused"));
        assert_eq!(get(&file, &["image", "models", "moss-image-sft-8-step", "distill"]).and_then(Json::as_str), whole.to_str());
        // (a distillation on another kind of model is no setting)
        let mut bad = cfg.clone();
        set(obj_mut(&mut bad, &["media", "image", "models", "xl"]).unwrap(), "distill", Json::str("x"));
        assert!(config::validate(&bad).unwrap_err().contains("for MOSS-Image models"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// BFL's FLUX 3 Image Klein release: its model.safetensors picked (or its folder) is a whole image model, its text
    /// encoder's folder and its VAE found beside it.
    #[test]
    fn flux3s_release_brings_its_text_encoder_and_vae() {
        let d = tmp("flux3");
        let tensor = |name: &str, shape: &str| format!(r#""{name}":{{"dtype":"BF16","shape":{shape},"data_offsets":[0,0]}}"#);
        let transformer = [tensor("emb_in.image.weight", "[3072,128]"), tensor("txt_in.weight", "[3072,20480]"), tensor("single_blocks.27.q_proj.weight", "[3072,3072]"), tensor("ref_time_in.in_layer.weight", "[3072,256]")].join(",");
        safetensors(&d.join("model.safetensors"), &format!("{{{transformer}}}"));
        safetensors(&d.join("ae.safetensors"), &format!("{{{},{}}}", tensor("bn.running_mean", "[128]"), tensor("decoder.conv_in.weight", "[512,32,3,3]")));
        let te = d.join("text_encoder");
        std::fs::create_dir_all(&te).unwrap();
        std::fs::write(te.join("config.json"), r#"{"model_type":"qwen3_vl","text_config":{"hidden_size":2560}}"#).unwrap();
        std::fs::write(te.join("tokenizer.json"), "{}").unwrap();
        safetensors(&te.join("model-00001-of-00002.safetensors"), &format!("{{{}}}", tensor("model.language_model.embed_tokens.weight", "[151936,2560]")));
        for picked in [d.join("model.safetensors"), d.clone()] {
            let mut cfg = config::default_json();
            let (a, found) = add(&mut cfg, &picked, None, None).unwrap();
            assert!(matches!(found.role, detect::Role::Image { architecture: "flux3-klein" }), "{}", found.summary);
            assert!(found.summary.contains("distilled"), "{}", found.summary);
            assert_eq!((a.section, a.enabled), ("image", true), "{:?}", a.missing);
            let m = get(&cfg, &["media", "image", "models", &a.name]).unwrap();
            assert_eq!(str_or(m, "architecture", ""), "flux3-klein");
            assert_eq!(Path::new(str_or(m, "text_encoder", "")), te.as_path());
            assert_eq!(Path::new(str_or(m, "vae", "")), d.join("ae.safetensors").as_path());
            config::validate(&cfg).unwrap();
        }
        // a LoRA picked after it (diffusers' names here) joins its list, once
        let lora = d.join("tarot.safetensors");
        safetensors(&lora, &format!("{{{}}}", tensor("transformer.mode_blocks.image.0.attn.to_q.lora_A.weight", "[16,3072]")));
        let mut cfg = config::default_json();
        assert!(add(&mut cfg, &lora, None, None).unwrap_err().contains("add or select a FLUX 3 Image Klein model before its LoRA"));
        let (model, _) = add(&mut cfg, &d.join("model.safetensors"), None, None).unwrap();
        for _ in 0..2 {
            let (a, found) = add(&mut cfg, &lora, None, None).unwrap();
            assert_eq!((found.kind(), a.name.as_str()), ("flux3_lora", model.name.as_str()));
        }
        let m = get(&cfg, &["media", "image", "models", &model.name]).unwrap();
        assert_eq!(m.get("loras").map(Json::to_json), Some(format!("[{}]", Json::str(lora.to_string_lossy()).to_json())));
        config::validate(&cfg).unwrap();
        let _ = std::fs::remove_dir_all(&d);
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

    /// A Qwen3-VL GGUF added to a Qwen Image model is its text encoder, whatever the file is called (Qwen's own
    /// `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, the encoder Qwen Image 2.1 ships, as a chat model names it); added to nothing,
    /// or to another kind of model, it is a chat model as before.
    #[test]
    fn a_qwen3_vl_gguf_added_to_a_qwen_image_model_is_its_text_encoder() {
        let d = tmp("encoder");
        let mut cfg = config::default_json();
        let turbo = d.join("Qwen-Image-2.1-Turbo-AD-Q4_K.gguf");
        gguf(&turbo, "qwen_image21");
        let (image, _) = add(&mut cfg, &turbo, None, None).unwrap();
        assert_eq!(image.section, "image");
        let encoder = d.join("Qwen3VL-8B-Instruct-Q4_K_M.gguf");
        gguf(&encoder, "qwen3vl");
        let (a, detected) = add(&mut cfg, &encoder, None, Some(("image", &image.name))).unwrap();
        assert_eq!((a.section, a.name.as_str(), detected.kind()), ("image", image.name.as_str(), "text_encoder"));
        let path = get(&cfg, &["media", "image", "models", &image.name, "text_encoder"]).and_then(Json::as_str).unwrap_or_default().to_string();
        assert!(path.ends_with("Qwen3VL-8B-Instruct-Q4_K_M.gguf"), "{path}");
        assert!(get(&cfg, &["llm", "models"]).is_none_or(|m| m.len() == 0), "no chat model was added");
        // Without a Qwen Image model named: a chat model.
        let (chat, _) = add(&mut cfg, &encoder, None, None).unwrap();
        assert_eq!(chat.section, "llm");
        let (chat2, _) = add(&mut cfg, &encoder, None, Some(("image", "no-such-model"))).unwrap();
        assert_eq!(chat2.section, "llm");
        config::validate(&cfg).unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    /// Qwen's mmproj GGUF added to a Qwen Image model is its vision tower (for its reference images); added to nothing
    /// in particular it is a chat model's projector as before.
    #[test]
    fn an_mmproj_added_to_a_qwen_image_model_is_its_vision() {
        let d = tmp("mmproj");
        let mut cfg = config::default_json();
        let turbo = d.join("Qwen-Image-2.1-Turbo-AD-Q4_K.gguf");
        gguf(&turbo, "qwen_image21");
        let (image, _) = add(&mut cfg, &turbo, None, None).unwrap();
        let proj = d.join("mmproj-Qwen3VL-8B-Instruct-F16.gguf");
        gguf(&proj, "clip");
        let (a, _) = add(&mut cfg, &proj, None, Some(("image", &image.name))).unwrap();
        assert_eq!((a.section, a.name.as_str()), ("image", image.name.as_str()));
        let path = get(&cfg, &["media", "image", "models", &image.name, "vision"]).and_then(Json::as_str).unwrap_or_default().to_string();
        assert!(path.ends_with("mmproj-Qwen3VL-8B-Instruct-F16.gguf"), "{path}");
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

    /// Speech, music, sound effect and 3D models travel too, each with its settings, and LoRAs with their strengths:
    /// a LoRA not on the other machine switches its model off like any of its files.
    #[test]
    fn every_section_and_its_loras_travel_in_a_models_file() {
        let d = tmp("export-all");
        for dir in ["tts", "music", "sfx", "pixal3d", "dino"] {
            std::fs::create_dir_all(d.join(dir)).unwrap();
        }
        for file in ["naf_release.pth", "style.safetensors", "transformer.safetensors", "te.gguf", "vae.safetensors"] {
            std::fs::write(d.join(file), b"x").unwrap();
        }
        let mut cfg = config::default_json();
        let at = |f: &str| Json::str(d.join(f).to_string_lossy());
        let media = |cfg: &mut Json, kind: &str, name: &str, m: Json| {
            let Some(Json::Obj(models)) = obj_mut(cfg, &["media", kind, "models"]) else { panic!("no {kind}") };
            models.push((name.into(), m));
            set(obj_mut(cfg, &["media", kind]).unwrap(), "default_model", Json::str(name));
        };
        media(&mut cfg, "image", "moss", Json::obj([("architecture", Json::str("moss-image")), ("transformer", at("transformer.safetensors")), ("text_encoder", at("te.gguf")), ("vae", at("vae.safetensors")),
            ("steps", Json::Int(30)), ("loras", Json::Arr(vec![Json::obj([("path", at("style.safetensors")), ("strength", Json::Num(0.8))])]))]));
        media(&mut cfg, "speech", "tts", Json::obj([("design", at("tts"))]));
        media(&mut cfg, "music", "song", Json::obj([("path", at("music")), ("precision", Json::str("f32"))]));
        set(obj_mut(&mut cfg, &["media", "music"]).unwrap(), "ram_gb", Json::Int(20));
        media(&mut cfg, "sound", "sfx", Json::obj([("path", at("sfx"))]));
        media(&mut cfg, "model3d", "pixal", Json::obj([("path", at("pixal3d")), ("dino", at("dino")), ("naf", at("naf_release.pth"))]));
        config::validate(&cfg).unwrap();
        let file = export(&cfg, &d);
        for kind in ["image", "speech", "music", "sound", "model3d"] {
            assert_eq!(get(&file, &[kind, "models"]).map(|m| m.members().count()), Some(1), "{kind}");
        }
        assert_eq!(get(&file, &["music", "ram_gb"]).and_then(Json::as_i64), Some(20));
        // Into a fresh install: all of them, their settings, defaults and routes.
        let mut other = config::default_json();
        if let Some(Json::Arr(routes)) = obj_mut(&mut other, &["gateway", "routes"]) {
            routes.retain(|r| !["music", "model3d"].contains(&str_or(r, "target", "")));
        }
        let report = import(&mut other, &file, false, &d).unwrap();
        for kind in ["image", "speech", "music", "sound", "model3d"] {
            assert_eq!(report.get(kind).and_then(Json::as_i64), Some(1), "{kind}");
            assert!(get(&other, &["media", kind, "default_model"]).and_then(Json::as_str).is_some_and(|n| !n.is_empty()), "{kind}");
        }
        assert_eq!((report.get("llm").and_then(Json::as_i64), report.get("video").and_then(Json::as_i64)), (Some(0), Some(0)));
        assert!(report.get("missing").unwrap().as_array().unwrap().is_empty());
        let moss = get(&other, &["media", "image", "models", "moss"]).unwrap();
        assert_eq!(moss.get("steps").and_then(Json::as_i64), Some(30));
        assert_eq!(get(moss, &["loras"]).and_then(|l| l.at(0)).and_then(|l| l.get("strength")).and_then(Json::as_f64), Some(0.8));
        assert_eq!(get(&other, &["media", "music", "ram_gb"]).and_then(Json::as_i64), Some(20));
        let routes = get(&other, &["gateway", "routes"]).unwrap().as_array().unwrap();
        assert!(["music", "model3d"].iter().all(|t| routes.iter().any(|r| str_or(r, "target", "") == *t)), "their routes come back");
        // Elsewhere, without the LoRA: the model comes in switched off, the LoRA named.
        std::fs::remove_file(d.join("style.safetensors")).unwrap();
        let mut third = config::default_json();
        let report = import(&mut third, &file, true, &d).unwrap();
        let missing = report.get("missing").unwrap().as_array().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!((str_or(&missing[0], "model", ""), str_or(&missing[0], "field", "")), ("moss", "loras"));
        assert_eq!(get(&third, &["media", "image", "models", "moss", "enabled"]), Some(&Json::Bool(false)));
        assert_eq!(get(&third, &["media", "sound", "models", "sfx", "enabled"]), None, "the others as they were");
        // A file from before these sections were exported leaves them be, even replacing.
        let mut old = file.clone();
        if let Json::Obj(f) = &mut old {
            f.retain(|(k, _)| !["speech", "music", "sound", "model3d"].contains(&k.as_str()));
        }
        import(&mut third, &old, true, &d).unwrap();
        assert!(get(&third, &["media", "speech", "models", "tts"]).is_some());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn qwen3_tts_folders_pair_into_one_speech_model() {
        let d = tmp("tts");
        let make = |name: &str, kind: &str| {
            let dir = d.join(name);
            std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
            std::fs::write(dir.join("config.json"), format!(r#"{{"model_type":"qwen3_tts","tts_model_type":"{kind}"}}"#)).unwrap();
            std::fs::write(dir.join("speech_tokenizer").join("model.safetensors"), b"x").unwrap();
            dir
        };
        let (design, base) = (make("Qwen3-TTS-VoiceDesign", "voice_design"), make("Qwen3-TTS-Base", "base"));
        let mut cfg = config::default_json();
        let (a, _) = add(&mut cfg, &design, None, None).unwrap();
        assert_eq!((a.section, a.name.as_str()), ("speech", "qwen3-tts"));
        let (b, _) = add(&mut cfg, &base, None, None).unwrap();
        assert_eq!(b.name, "qwen3-tts", "the Base half completes the same model");
        let m = get(&cfg, &["media", "speech", "models", "qwen3-tts"]).unwrap();
        assert!(str_or(m, "design", "").ends_with("Qwen3-TTS-VoiceDesign") && str_or(m, "base", "").ends_with("Qwen3-TTS-Base"));
        assert_eq!(get(&cfg, &["media", "speech", "default_model"]).and_then(Json::as_str), Some("qwen3-tts"));
        let routes = get(&cfg, &["gateway", "routes"]).unwrap().as_array().unwrap();
        assert!(["speech", "voices"].iter().all(|t| routes.iter().any(|r| str_or(r, "target", "") == *t)));
        let mut bad = config::default_json();
        std::fs::write(design.join("config.json"), r#"{"model_type":"qwen3_tts","tts_model_type":"custom_voice"}"#).unwrap();
        assert!(add(&mut bad, &design, None, None).is_err());
        config::validate(&cfg).unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_moss_folder_makes_a_sound_model_and_its_route() {
        let d = tmp("moss");
        let dir = d.join("MOSS-SoundEffect-v2.0");
        for part in ["text_encoder", "tokenizer", "transformer", "vae"] {
            std::fs::create_dir_all(dir.join(part)).unwrap();
        }
        std::fs::write(dir.join("model_index.json"), r#"{"_class_name":"MossSoundEffectPipeline","dit_variant":"1.3B"}"#).unwrap();
        let found = crate::models::detect::detect(&dir).unwrap();
        assert_eq!(found.kind(), "sound_model");
        let mut cfg = crate::config::default_json();
        if let Some(Json::Obj(g)) = obj_mut(&mut cfg, &["gateway"]) {
            if let Some((_, Json::Arr(routes))) = g.iter_mut().find(|(k, _)| k == "routes") {
                routes.retain(|r| str_or(r, "target", "") != "sound");
            }
        }
        let a = attach(&mut cfg, &found, None, &dir).unwrap();
        assert_eq!((a.section, a.name.as_str()), ("sound", "moss-soundeffect"));
        assert_eq!(get(&cfg, &["media", "sound", "default_model"]).and_then(Json::as_str), Some("moss-soundeffect"));
        let routes = get(&cfg, &["gateway", "routes"]).and_then(Json::as_array).unwrap();
        assert!(routes.iter().any(|r| str_or(r, "target", "") == "sound"));
        crate::config::validate(&cfg).unwrap();
        let _ = std::fs::remove_dir_all(d);
    }

    /// A Pixal3D release as it is downloaded: its folder, DINOv3 and NAF side by side under `at`.
    fn pixal3d_parts(at: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let (pixal, dino, naf) = (at.join("Pixal3D"), at.join("dinov3-vitl16"), at.join("NAF"));
        for dir in [pixal.join("ckpts"), dino.clone(), naf.clone()] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(pixal.join("pipeline.json"), r#"{"name":"Trellis2ImageTo3DPipeline"}"#).unwrap();
        std::fs::write(dino.join("config.json"), r#"{"model_type":"dinov3_vit","hidden_size":1024,"patch_size":16}"#).unwrap();
        safetensors(&dino.join("model.safetensors"), r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
        std::fs::write(naf.join("naf_release.pth"), b"PK").unwrap();
        (pixal, dino, naf.join("naf_release.pth"))
    }

    #[test]
    fn a_pixal3d_folder_and_its_parts_make_one_3d_model_and_its_route() {
        let d = tmp("model3d");
        let model = |cfg: &Json, name: &str| get(cfg, &["media", "model3d", "models", name]).cloned().unwrap_or(Json::Null);
        // The Pixal3D folder finds DINOv3 and NAF beside it, and the route comes back.
        let (pixal, _, naf) = pixal3d_parts(&d.join("side-by-side"));
        let mut cfg = config::default_json();
        if let Some(Json::Arr(routes)) = obj_mut(&mut cfg, &["gateway", "routes"]) {
            routes.retain(|r| str_or(r, "target", "") != "model3d");
        }
        let (a, found) = add(&mut cfg, &pixal, None, None).unwrap();
        assert_eq!(found.kind(), "model3d");
        assert_eq!((a.section, a.name.as_str(), a.enabled), ("model3d", "pixal3d", true), "{:?}", a.missing);
        let m = model(&cfg, "pixal3d");
        assert!(str_or(&m, "path", "").ends_with("Pixal3D") && str_or(&m, "dino", "").ends_with("dinov3-vitl16") && str_or(&m, "naf", "").ends_with("naf_release.pth"));
        assert_eq!(get(&cfg, &["media", "model3d", "default_model"]).and_then(Json::as_str), Some("pixal3d"));
        let routes = get(&cfg, &["gateway", "routes"]).and_then(Json::as_array).unwrap();
        assert!(routes.iter().any(|r| str_or(r, "target", "") == "model3d" && str_or(r, "path", "") == "/v1/3d/models"));
        // A second Pixal3D folder is a model of its own, with the first one's parts.
        let (a2, _) = add(&mut cfg, &pixal, None, None).unwrap();
        assert_eq!((a2.name.as_str(), a2.enabled), ("pixal3d-2", true));
        assert_eq!(str_or(&model(&cfg, "pixal3d-2"), "naf", ""), str_or(&model(&cfg, "pixal3d"), "naf", ""));
        config::validate(&cfg).unwrap();
        // BiRefNet and Real-ESRGAN, the optional helpers: found beside Pixal3D.
        let with_helpers = d.join("with-helpers");
        let (pixal, _, _) = pixal3d_parts(&with_helpers);
        let birefnet = with_helpers.join("BiRefNet");
        let esrgan = with_helpers.join("Real-ESRGAN");
        std::fs::create_dir_all(&birefnet).unwrap();
        std::fs::create_dir_all(&esrgan).unwrap();
        std::fs::write(birefnet.join("config.json"), r#"{"architectures":["BiRefNet"]}"#).unwrap();
        safetensors(&birefnet.join("model.safetensors"), r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
        std::fs::write(esrgan.join("RealESRGAN_x4plus.pth"), b"PK").unwrap();
        let mut helped = config::default_json();
        let (h, _) = add(&mut helped, &pixal, None, None).unwrap();
        let m = model(&helped, &h.name);
        assert!(str_or(&m, "matte", "").ends_with("BiRefNet") && str_or(&m, "upscaler", "").ends_with("RealESRGAN_x4plus.pth"), "{m:?}");
        config::validate(&helped).unwrap();
        // Added to a model that has none yet, a helper fills it.
        let (a3, found) = add(&mut cfg, &birefnet, None, Some(("model3d", "pixal3d"))).unwrap();
        assert_eq!((found.kind(), a3.name.as_str()), ("model3d_matte", "pixal3d"));
        assert!(str_or(&model(&cfg, "pixal3d"), "matte", "").ends_with("BiRefNet"));
        // It is the picture tools' background remover too, and Real-ESRGAN their upscaler.
        assert!(get(&cfg, &["media", "picture", "background"]).and_then(Json::as_str).is_some_and(|p| p.ends_with("BiRefNet")));
        add(&mut cfg, &esrgan, None, None).unwrap();
        assert!(get(&cfg, &["media", "picture", "upscaler"]).and_then(Json::as_str).is_some_and(|p| p.ends_with("RealESRGAN_x4plus.pth")));
        config::validate(&cfg).unwrap();
        // Parts first, each from its own place: one model, switched on once it has all three.
        let apart = d.join("apart");
        let (pixal, _, _) = pixal3d_parts(&apart.join("a"));
        let (_, dino, _) = pixal3d_parts(&apart.join("b").join("c"));
        std::fs::remove_dir_all(apart.join("b").join("c").join("NAF")).unwrap();
        let mut cfg = config::default_json();
        let (a, _) = add(&mut cfg, &dino, None, None).unwrap();
        assert_eq!((a.name.as_str(), a.enabled, a.missing.clone()), ("pixal3d", false, vec!["path".to_string(), "naf".to_string()]));
        assert_eq!(get(&cfg, &["media", "model3d", "default_model"]).and_then(Json::as_str), Some(""), "an incomplete model is not the default");
        let (b, _) = add(&mut cfg, &naf, None, None).unwrap();
        assert_eq!((b.name.as_str(), b.enabled, b.missing.clone()), ("pixal3d", false, vec!["path".to_string()]));
        let (c, _) = add(&mut cfg, &pixal, None, None).unwrap();
        assert_eq!((c.name.as_str(), c.enabled), ("pixal3d", true), "{:?}", c.missing);
        assert_eq!(get(&cfg, &["media", "model3d", "models"]).unwrap().len(), 1);
        assert!(str_or(&model(&cfg, "pixal3d"), "dino", "").ends_with("dinov3-vitl16"));
        assert_eq!(get(&cfg, &["media", "model3d", "default_model"]).and_then(Json::as_str), Some("pixal3d"), "the default once it is complete");
        config::validate(&cfg).unwrap();
        // NAF's weights loose beside the Pixal3D folder, not in a folder of their own.
        let loose = d.join("loose");
        let (pixal, _, naf) = pixal3d_parts(&loose);
        std::fs::rename(&naf, loose.join("naf_release.pth")).unwrap();
        let mut cfg = config::default_json();
        let (a, _) = add(&mut cfg, &pixal, None, None).unwrap();
        assert!(a.enabled, "{:?}", a.missing);
        assert_eq!(str_or(&model(&cfg, "pixal3d"), "naf", ""), loose.join("naf_release.pth").to_string_lossy());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_music_folder_and_its_smaller_language_model_make_one_music_model() {
        let d = tmp("music");
        let dir = d.join("MiniMax-Music3");
        for part in ["language_model", "rvq_depth_decoder", "condition_encoder", "transformer", "vocoder", "tokenizer"] {
            std::fs::create_dir_all(dir.join(part)).unwrap();
        }
        std::fs::write(dir.join("config.json"), r#"{"model_type":"minimax_music3"}"#).unwrap();
        // A GGUF with only its metadata: general.architecture and music3.quant.
        let mut g = b"GGUF".to_vec();
        g.extend(3u32.to_le_bytes());
        g.extend(0u64.to_le_bytes());
        g.extend(2u64.to_le_bytes());
        for (k, v) in [("general.architecture", "music3-lm"), ("music3.quant", "q4_k")] {
            g.extend((k.len() as u64).to_le_bytes());
            g.extend(k.as_bytes());
            g.extend(8u32.to_le_bytes());
            g.extend((v.len() as u64).to_le_bytes());
            g.extend(v.as_bytes());
        }
        let gguf = dir.join("language_model-q4_k.gguf");
        std::fs::write(&gguf, g).unwrap();
        let mut cfg = config::default_json();
        // The smaller language model alone has nowhere to go yet.
        assert!(add(&mut cfg.clone(), &gguf, None, None).unwrap_err().contains("folder first"));
        let (a, _) = add(&mut cfg, &dir, None, None).unwrap();
        assert_eq!((a.section, a.name.as_str()), ("music", "minimax-music3"));
        let (b, det) = add(&mut cfg, &gguf, None, None).unwrap();
        assert_eq!(b.name, "minimax-music3");
        assert!(det.summary.contains("q4_k"), "{}", det.summary);
        let m = get(&cfg, &["media", "music", "models", "minimax-music3"]).unwrap();
        assert!(str_or(m, "path", "").ends_with("MiniMax-Music3") && str_or(m, "language_model", "").ends_with("language_model-q4_k.gguf"));
        assert_eq!(get(&cfg, &["media", "music", "default_model"]).and_then(Json::as_str), Some("minimax-music3"));
        let routes = get(&cfg, &["gateway", "routes"]).unwrap().as_array().unwrap();
        assert!(routes.iter().any(|r| str_or(r, "target", "") == "music"));
        config::validate(&cfg).unwrap();
        std::fs::remove_dir_all(dir.join("vocoder")).unwrap();
        assert!(add(&mut config::default_json(), &dir, None, None).is_err());
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
