//! Live, data-only media manifests. Reading these never loads model weights.
use crate::images::Config;
use nrob::json::Json;
use std::path::{Path, PathBuf};

fn document(path: &Path) -> Result<Json, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() > 1024 * 1024 {
        return Err("media manifest must be a file of at most 1 MiB".into());
    }
    let value = Json::parse(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if value.as_object().is_none() {
        return Err("media manifest must be an object".into());
    }
    match value.get("enabled") {
        None | Some(Json::Bool(true)) => Ok(value),
        Some(Json::Bool(false)) => Err(format!("{} is disabled", path.display())),
        _ => Err("enabled must be boolean".into()),
    }
}
fn text<'a>(j: &'a Json, key: &str) -> Result<&'a str, String> {
    j.get(key)
        .and_then(Json::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| format!("media manifest: missing {key}"))
}
fn path(root: &Path, value: &str) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() {
        p.into()
    } else {
        root.join(p)
    }
}
fn required_path(root: &Path, j: &Json, key: &str) -> Result<PathBuf, String> {
    let p = path(root, text(j, key)?);
    if !p.exists() {
        return Err(format!("{key} does not exist: {}", p.display()));
    }
    Ok(p)
}
fn optional_path(root: &Path, j: &Json, key: &str) -> Result<Option<PathBuf>, String> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(None),
        _ => required_path(root, j, key).map(Some),
    }
}
pub(crate) fn read_directory(root: &Path) -> Result<Config, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let j = document(&root.join("controller.json"))?;
    let device = |key| {
        j.get(key)
            .and_then(Json::as_i64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("invalid {key}"))
    };
    let c = Config {
        worker: required_path(&root, &j, "worker")?,
        controller_name: text(&j, "controller_name")?.into(),
        controller_path: required_path(&root, &j, "controller_path")?,
        controller_device: device("controller_device")?,
        image_device: device("image_device")?,
        output_root: path(&root, text(&j, "output_root")?),
        media_dir: Some(root),
        base: PathBuf::new(),
        transformer: PathBuf::new(),
        safetensors_transformer: None,
        adapter: None,
        loras: Vec::new(),
        video_config: None,
        default_weights: "gguf".into(),
        image_model: None,
        text_encoder: None,
        sdxl: None,
        image_memory: Json::obj([] as [(&str, Json); 0]),
    };
    if c.controller_device == c.image_device {
        return Err("image and controller devices must differ".into());
    }
    Ok(c)
}
pub(crate) fn image(c: &Config, selected: Option<&str>) -> Result<Config, String> {
    let Some(root) = &c.media_dir else {
        if selected.is_some() {
            return Err("named image models require a media catalog".into());
        }
        return Ok(c.clone());
    };
    let mut j = document(&root.join("image.json"))?;
    let mut snapshot = c.clone();
    if let Some(models) = j.get("models") {
        let models = models.as_object().ok_or("image models must be an object")?;
        let name = selected
            .or_else(|| j.get("default_model").and_then(Json::as_str))
            .ok_or("image catalog requires default_model")?
            .to_owned();
        let model = models
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, m)| m)
            .ok_or_else(|| format!("unknown image model: {name}"))?;
        let fields = model.as_object().ok_or("image model must be an object")?;
        match model.get("enabled") {
            None | Some(Json::Bool(true)) => (),
            Some(Json::Bool(false)) => return Err(format!("image model {name} is disabled")),
            _ => return Err("model enabled must be boolean".into()),
        }
        let overrides = fields.to_vec();
        if let Json::Obj(shared) = &mut j {
            for (key, value) in overrides {
                shared.retain(|(k, _)| k != &key);
                shared.push((key, value));
            }
        }
        snapshot.image_model = Some(name);
    } else if selected.is_some() {
        return Err("image catalog has no named models".into());
    }
    // Catalog-wide residency, overridable per model like every other field.
    snapshot.image_memory = Json::Obj(
        ["memory", "ram_gb", "vram_gb"]
            .into_iter()
            .filter_map(|k| j.get(k).map(|v| (k.to_string(), v.clone())))
            .collect(),
    );
    match j.get("architecture").and_then(Json::as_str).unwrap_or("qwen-image") {
        "sdxl" => {
            let checkpoint = required_path(root, &j, "checkpoint")?;
            let tokenizer = required_path(root, &j, "tokenizer")?;
            if !checkpoint.is_file() || !tokenizer.is_file() {
                return Err("SDXL checkpoint and tokenizer must be files".into());
            }
            let mut fields = vec![
                ("checkpoint".into(), Json::str(checkpoint.to_string_lossy())),
                ("tokenizer".into(), Json::str(tokenizer.to_string_lossy())),
            ];
            for key in ["steps", "cfg", "clip_skip", "sampler", "scheduler", "negative_prompt"] {
                if let Some(value) = j.get(key) { fields.push((key.into(), value.clone())); }
            }
            snapshot.sdxl = Some(Json::Obj(fields));
            snapshot.safetensors_transformer = Some(checkpoint);
            snapshot.default_weights = "safetensors".into();
            snapshot.adapter = None;
            snapshot.loras = Vec::new();
            snapshot.text_encoder = None;
            return Ok(snapshot);
        }
        "qwen-image" => snapshot.sdxl = None,
        _ => return Err("unsupported image architecture; use qwen-image or sdxl".into()),
    }
    snapshot.text_encoder = optional_path(root, &j, "text_encoder")?;
    snapshot.base = required_path(root, &j, "base")?;
    snapshot.transformer = optional_path(root, &j, "transformer")?.unwrap_or_default();
    snapshot.safetensors_transformer = optional_path(root, &j, "safetensors_transformer")?;
    snapshot.adapter = optional_path(root, &j, "adapter")?;
    snapshot.loras = crate::images::loras(Some(root), &j)?;
    snapshot.default_weights = match j.get("default_weights") {
        None => "gguf",
        Some(v) => v.as_str().ok_or("default_weights must be a string")?,
    }
    .into();
    if !["gguf", "safetensors"].contains(&snapshot.default_weights.as_str()) {
        return Err("default_weights must be gguf or safetensors".into());
    }
    let selected = if snapshot.default_weights == "gguf" {
        snapshot.transformer.clone()
    } else {
        snapshot
            .safetensors_transformer
            .clone()
            .unwrap_or_else(|| snapshot.base.join("transformer"))
    };
    if !selected.exists() {
        return Err(format!(
            "default image weights are missing: {}",
            selected.display()
        ));
    }
    Ok(snapshot)
}
pub(crate) fn video(c: &Config) -> Result<Json, String> {
    let manifest = c
        .media_dir
        .as_ref()
        .map(|p| p.join("video.json"))
        .or_else(|| c.video_config.clone())
        .ok_or("video generation is not configured")?;
    let mut j = document(&manifest)?;
    let root = manifest.parent().unwrap_or(Path::new("."));
    // Both old manifests and catalog files accept paths relative to the manifest.
    if let Json::Obj(fields) = &mut j {
        for (key, value) in fields {
            if key == "ffmpeg" {
                if let Some(s) = value.as_str() {
                    *value = Json::str(path(root, s).to_string_lossy());
                }
            }
            if key == "models" {
                if let Json::Obj(models) = value {
                    for (_, model) in models {
                        if let Json::Obj(paths) = model {
                            for (key, value) in paths {
                                if ["transformer", "text_encoder", "vae", "tokenizer"]
                                    .contains(&key.as_str())
                                {
                                    if let Some(s) = value.as_str() {
                                        *value = Json::str(path(root, s).to_string_lossy());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(j)
}
pub(crate) fn ready_video(name: &str, model: &Json) -> bool {
    if !["ltx-2.3", "ltx-2.5", "sulphur-2"].contains(&name) {
        return false;
    }
    ["transformer", "text_encoder", "vae", "tokenizer"]
        .iter()
        .filter(|&&k| k != "tokenizer" || name != "ltx-2.5")
        .all(|k| {
            model
                .get(k)
                .and_then(Json::as_str)
                .is_some_and(|p| Path::new(p).is_file())
        })
}
impl Config {
    pub fn capabilities(&self) -> Json {
        let image = match image(self, None) {
            Ok(c) => Json::obj([
                ("available", Json::Bool(true)),
                ("default_weights", Json::str(&c.default_weights)),
                (
                    "default_model",
                    c.image_model.as_ref().map(Json::str).unwrap_or(Json::Null),
                ),
                ("models", image_models(self)),
                (
                    "checkpoint",
                    Json::str(
                        if c.default_weights == "gguf" {
                            c.transformer
                        } else {
                            c.safetensors_transformer
                                .unwrap_or_else(|| c.base.join("transformer"))
                        }
                        .to_string_lossy(),
                    ),
                ),
            ]),
            Err(e) => Json::obj([("available", Json::Bool(false)), ("error", Json::str(e))]),
        };
        let video = match video(self) {
            Ok(j) => {
                let default = j
                    .get("default_model")
                    .and_then(Json::as_str)
                    .unwrap_or("ltx-2.3");
                let models = j.get("models").and_then(Json::as_object).unwrap_or(&[]);
                Json::obj([
                    (
                        "available",
                        Json::Bool(
                            models
                                .iter()
                                .any(|(n, m)| n == default && ready_video(n, m)),
                        ),
                    ),
                    ("default_model", Json::str(default)),
                    (
                        "models",
                        Json::Arr(
                            models
                                .iter()
                                .map(|(n, m)| {
                                    Json::obj([
                                        ("model", Json::str(n)),
                                        ("weights_ready", Json::Bool(ready_video(n, m))),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            }
            Err(e) => Json::obj([("available", Json::Bool(false)), ("error", Json::str(e))]),
        };
        Json::obj([("image", image), ("video", video)])
    }
}

fn image_models(c: &Config) -> Json {
    let entries = c
        .media_dir
        .as_ref()
        .and_then(|root| document(&root.join("image.json")).ok())
        .and_then(|j| {
            j.get("models")
                .and_then(Json::as_object)
                .map(|m| m.to_vec())
        })
        .unwrap_or_default();
    Json::Arr(
        entries
            .iter()
            .map(|(name, entry)| {
                let ready = image(c, Some(name));
                Json::obj([
                    ("model", Json::str(name)),
                    ("architecture", entry.get("architecture").cloned().unwrap_or_else(|| Json::str("qwen-image"))),
                    ("defaults", if entry.get("architecture").and_then(Json::as_str) == Some("sdxl") {
                        Json::obj([
                            ("steps", entry.get("steps").cloned().unwrap_or(Json::Int(16))),
                            ("cfg", entry.get("cfg").cloned().unwrap_or(Json::Num(2.5))),
                            ("sampler", Json::str("dpmpp_2m")), ("scheduler", Json::str("karras")),
                            ("clip_skip", entry.get("clip_skip").cloned().unwrap_or(Json::Int(1))),
                        ])
                    } else { Json::Null }),
                    ("weights_ready", Json::Bool(ready.is_ok())),
                    ("error", ready.err().map(Json::str).unwrap_or(Json::Null)),
                ])
            })
            .collect(),
    )
}
