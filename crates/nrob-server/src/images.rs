//! Native Rust diffusion subprocess supervision. The worker owns all media
//! allocations: exit (including cancellation) releases its CUDA context.
use nrob::json::Json;
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub media_dir: Option<PathBuf>,
    pub default_weights: String,
    pub image_model: Option<String>,
    pub text_encoder: Option<PathBuf>,
    pub sdxl: Option<Json>,
    /// Image weight residency defaults and caps from the catalog: `memory`
    /// (auto|gpu|ram|ssd), `ram_gb`, `vram_gb`. An object; empty when unset.
    pub image_memory: Json,
    pub worker: PathBuf,
    pub base: PathBuf,
    pub transformer: PathBuf,
    pub safetensors_transformer: Option<PathBuf>,
    pub adapter: Option<PathBuf>,
    pub output_root: PathBuf,
    pub controller_name: String,
    pub controller_path: PathBuf,
    pub controller_device: usize,
    pub image_device: usize,
    pub video_config: Option<PathBuf>,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self, String> {
        if path.is_dir() { return crate::media_catalog::read_directory(path); }
        let j = Json::parse(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let s = |k: &str| {
            j.get(k)
                .and_then(Json::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("image config: missing {k}"))
        };
        let n = |k: &str| {
            j.get(k)
                .and_then(Json::as_i64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("image config: invalid {k}"))
        };
        let c = Self {
            media_dir: None,
            default_weights: "gguf".into(),
            image_model: None,
            text_encoder: None,
            sdxl: None,
            image_memory: Json::obj([] as [(&str, Json); 0]),
            worker: s("worker")?.into(),
            base: s("base")?.into(),
            transformer: s("transformer")?.into(),
            safetensors_transformer: match j.get("safetensors_transformer") {
                None | Some(Json::Null) => None,
                Some(v) => Some(v.as_str().filter(|s| !s.trim().is_empty())
                    .ok_or("safetensors_transformer must be a nonempty path")?.into()),
            },
            adapter: j.get("adapter").and_then(Json::as_str).map(PathBuf::from),
            output_root: s("output_root")?.into(),
            controller_name: s("controller_name")?,
            controller_path: s("controller_path")?.into(),
            controller_device: n("controller_device")?,
            image_device: n("image_device")?,
            video_config: j
                .get("video_config")
                .and_then(Json::as_str)
                .map(PathBuf::from),
        };
        for p in [&c.worker, &c.base, &c.transformer, &c.controller_path]
            .into_iter()
            .chain(c.adapter.iter())
            .chain(c.safetensors_transformer.iter())
            .chain(c.video_config.iter())
        {
            if !p.is_absolute() || !p.exists() {
                return Err(format!(
                    "image config path must exist and be absolute: {}",
                    p.display()
                ));
            }
        }
        if c.controller_device == c.image_device {
            return Err(
                "image and controller devices must differ to reserve VRAM for both models".into(),
            );
        }
        if !c.output_root.is_absolute() {
            return Err("output_root must be absolute".into());
        }
        Ok(c)
    }
}

struct Job {
    id: u64,
    kind: &'static str,
    state: String,
    progress: Json,
    result: Json,
    error: Option<String>,
    cancel: Arc<AtomicBool>,
}
struct Worker(std::process::Child);
impl std::ops::Deref for Worker {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Worker {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
pub struct Images {
    models: Arc<crate::models::Models>,
    job: Mutex<Option<Job>>,
}
impl Images {
    pub fn new(models: Arc<crate::models::Models>) -> Self {
        Self {
            models,
            job: Mutex::new(None),
        }
    }
    pub fn busy(&self) -> bool {
        self.job
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|j| matches!(j.state.as_str(), "queued" | "running" | "cancelling"))
    }
    pub fn status(&self) -> Json {
        let job = self.job.lock().unwrap_or_else(|p| p.into_inner());
        let config = self.models.image_config();
        let capabilities = config.map(Config::capabilities).unwrap_or(Json::Null);
        Json::obj([
            ("configured", Json::Bool(config.is_some())),
            (
                "video_configured",
                Json::Bool(capabilities.get("video").and_then(|v| v.get("available")).and_then(Json::as_bool).unwrap_or(false)),
            ),
            ("video_models", video_models(config)),
            ("media", capabilities),
            (
                "controller_model",
                config.map_or(Json::Null, |c| Json::str(&c.controller_name)),
            ),
            (
                "job",
                job.as_ref().map_or(Json::Null, |j| {
                    Json::obj([
                        ("id", Json::Int(j.id as i64)),
                        ("kind", Json::str(j.kind)),
                        ("state", Json::str(&j.state)),
                        ("progress", j.progress.clone()),
                        ("result", j.result.clone()),
                        ("error", j.error.as_ref().map_or(Json::Null, Json::str)),
                    ])
                }),
            ),
        ])
    }
    pub fn cancel(&self) {
        if let Some(j) = self.job.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            if matches!(j.state.as_str(), "queued" | "running") {
                j.cancel.store(true, Ordering::Relaxed);
                j.state = "cancelling".into();
            }
        }
    }
    pub fn release(&self) -> Result<(), String> {
        let job = self.job.lock().unwrap_or_else(|p| p.into_inner());
        if job
            .as_ref()
            .is_some_and(|j| matches!(j.state.as_str(), "queued" | "running" | "cancelling"))
        {
            return Err("cancel or finish the media job before releasing the controller".into());
        }
        self.models.release_images();
        Ok(())
    }
    pub fn submit(self: &Arc<Self>, body: &Json) -> Result<Json, String> {
        self.submit_kind(body, false)
    }
    pub fn submit_video(self: &Arc<Self>, body: &Json) -> Result<Json, String> {
        self.submit_kind(body, true)
    }
    fn submit_kind(self: &Arc<Self>, body: &Json, video: bool) -> Result<Json, String> {
        let cfg = self
            .models
            .image_config()
            .ok_or("image generation is disabled; configure --image-config")?
            .clone();
        let request = if video {
            prepare_video(&cfg, body)?
        } else {
            prepare(&cfg, body)?
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis() as u64;
        {
            let mut job = self.job.lock().unwrap_or_else(|p| p.into_inner());
            if job
                .as_ref()
                .is_some_and(|j| matches!(j.state.as_str(), "queued" | "running" | "cancelling"))
            {
                return Err("an image or video job is already active".into());
            }
            *job = Some(Job {
                id,
                kind: if video { "video" } else { "image" },
                state: "queued".into(),
                progress: Json::Null,
                result: Json::Null,
                error: None,
                cancel: Arc::clone(&cancel),
            });
        }
        let this = Arc::clone(self);
        if let Err(e) = std::thread::Builder::new()
            .name("image-batch".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    this.run(&cfg, request, &cancel)
                }))
                .unwrap_or_else(|_| Err("image supervisor panicked".into()));
                let mut job = this.job.lock().unwrap_or_else(|p| p.into_inner());
                // run() has dropped/reaped the worker, freeing its CUDA memory.
                // Clear the temporary route before publishing completion; the
                // next chat request can return to its chosen model and context.
                this.models.release_images();
                if let Some(j) = job.as_mut() {
                    match result {
                        Ok(value) => {
                            j.state = "completed".into();
                            j.result = value;
                        }
                        Err(e) => {
                            j.state = if cancel.load(Ordering::Relaxed) {
                                "cancelled"
                            } else {
                                "failed"
                            }
                            .into();
                            j.error = Some(e);
                        }
                    }
                }
            })
        {
            self.job
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
                .map(|j| {
                    j.state = "failed".into();
                    j.error = Some(e.to_string());
                });
            return Err(e.to_string());
        }
        Ok(Json::obj([
            ("id", Json::Int(id as i64)),
            ("state", Json::str("queued")),
            (
                "status_url",
                Json::str(if video {
                    "/v1/videos/status"
                } else {
                    "/v1/images/status"
                }),
            ),
            (
                "controller_model",
                Json::str(&self.models.image_config().unwrap().controller_name),
            ),
        ]))
    }
    fn run(
        self: &Arc<Self>,
        cfg: &Config,
        request: Json,
        cancel: &AtomicBool,
    ) -> Result<Json, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled before loading".into());
        }
        self.models.begin_images()?;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled after controller handoff".into());
        }
        let mut command = Command::new(&cfg.worker);
        if std::env::var_os("CUDA_CACHE_PATH").is_none() {
            // Share compiled CUDA kernels across all output folders and jobs.
            let cache = cfg.output_root.join(".cuda-cache");
            std::fs::create_dir_all(&cache).map_err(|e| format!("CUDA cache: {e}"))?;
            command.env("CUDA_CACHE_PATH", cache);
        }
        command
            .arg("--stdin")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = Worker(
            command
                .spawn()
                .map_err(|e| format!("starting native diffusion worker: {e}"))?,
        );
        let payload = request.to_json();
        let sent = child
            .stdin
            .take()
            .ok_or("worker stdin missing")
            .and_then(|mut input| {
                input
                    .write_all(payload.as_bytes())
                    .map_err(|_| "writing worker request failed")
            });
        if let Err(e) = sent {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e.into());
        }
        let stdout = child.stdout.take().ok_or("worker stdout missing")?;
        let stderr = child.stderr.take().ok_or("worker stderr missing")?;
        let output = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout
                .take(4 * 1024 * 1024)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        });
        let this = Arc::clone(self);
        let errors = std::thread::spawn(move || {
            let mut tail = String::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(progress) = Json::parse(line.as_bytes()) {
                    if let Some(j) = this.job.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                        j.progress = progress;
                    }
                }
                tail = line.chars().take(8192).collect();
            }
            tail
        });
        if let Some(j) = self.job.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            if !cancel.load(Ordering::Relaxed) {
                j.state = "running".into();
            }
        }
        let status = loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                break child.wait().map_err(|e| e.to_string());
            }
            match child.try_wait() {
                Ok(Some(s)) => break Ok(s),
                Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(e.to_string());
                }
            }
        }?;
        let bytes = output
            .join()
            .map_err(|_| "worker output reader failed")?
            .map_err(|e| e.to_string())?;
        let tail = errors
            .join()
            .unwrap_or_else(|_| "worker progress reader failed".into());
        if !status.success() {
            return Err(format!("diffusion worker exited {status}: {tail}"));
        }
        Json::parse(&bytes).map_err(|e| format!("invalid worker result: {e}"))
    }
}

fn video_models(config: Option<&Config>) -> Json {
    let manifest = config.and_then(|c| crate::media_catalog::video(c).ok());
    Json::Arr(
        manifest
            .as_ref()
            .and_then(|j| j.get("models"))
            .and_then(Json::as_object)
            .unwrap_or(&[])
            .iter()
            .map(|(name, model)| {
                let mut keys = vec!["transformer", "text_encoder", "vae"];
                if name != "ltx-2.5" {
                    keys.push("tokenizer");
                }
                let ready = keys.iter().all(|k| {
                    model
                        .get(k)
                        .and_then(Json::as_str)
                        .is_some_and(|p| Path::new(p).is_file())
                });
                Json::obj([
                    ("model", Json::str(name)),
                    ("weights_ready", Json::Bool(ready)),
                ])
            })
            .collect(),
    )
}

fn output_directory(c: &Config, body: &Json, default: &str) -> Result<PathBuf, String> {
    let folder = match body.get("output_dir") {
        None => default,
        Some(v) => v
            .as_str()
            .ok_or("output_dir must be a relative folder name")?,
    };
    let folder = Path::new(folder);
    if folder.as_os_str().is_empty()
        || folder
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        return Err("output_dir must stay under the configured output root".into());
    }
    std::fs::create_dir_all(&c.output_root).map_err(|e| e.to_string())?;
    let root = c.output_root.canonicalize().map_err(|e| e.to_string())?;
    let mut out = root.clone();
    for part in folder.components() {
        out.push(part.as_os_str());
        if out.exists() {
            out = out.canonicalize().map_err(|e| e.to_string())?;
            if !out.starts_with(&root) {
                return Err("output directory escapes root through a link".into());
            }
        } else {
            std::fs::create_dir(&out).map_err(|e| e.to_string())?;
        }
    }
    Ok(out)
}

fn prepare_video(c: &Config, body: &Json) -> Result<Json, String> {
    let config = crate::media_catalog::video(c)?;
    let model = match body.get("model") {
        None => config.get("default_model").and_then(Json::as_str).unwrap_or("ltx-2.3"),
        Some(v) => v.as_str().ok_or("model must be a string")?,
    };
    if !["ltx-2.3", "ltx-2.5", "sulphur-2"].contains(&model) {
        return Err("unsupported video model".into());
    }
    let selected = config
        .get("models")
        .and_then(|j| j.get(model))
        .ok_or_else(|| format!("video model {model} is not configured"))?;
    let prompt = body
        .get("prompt")
        .and_then(Json::as_str)
        .filter(|s| !s.trim().is_empty() && s.len() <= 16384)
        .ok_or("prompt must contain 1..16384 bytes")?;
    let number = |key: &str, default: i64, min: i64, max: i64| -> Result<i64, String> {
        let value = match body.get(key) {
            None => default,
            Some(v) => v
                .as_i64()
                .ok_or_else(|| format!("{key} must be an integer"))?,
        };
        if !(min..=max).contains(&value) {
            return Err(format!("{key} must be in {min}..{max}"));
        }
        Ok(value)
    };
    let width = number("width", 512, 128, 1024)?;
    let height = number("height", 320, 128, 1024)?;
    let frames = number("frames", 49, 9, 121)?;
    if width % 32 != 0 || height % 32 != 0 || (frames - 1) % 8 != 0 {
        return Err("video dimensions must be multiples of 32; frames must be 8k+1".into());
    }
    let fps = number("fps", 24, 1, 60)?;
    let seed = number("seed", 0, 0, i64::MAX)?;
    number("steps", 8, 8, 8)?;
    number("n", 1, 1, 1)?;
    let image = video_reference_path(body, "image", true)?;
    let end_image = video_reference_path(body, "end_image", true)?;
    for key in ["images", "adapter"] {
        if body.get(key).is_some_and(|v| !matches!(v, Json::Null)) {
            return Err(format!("{key} is not supported for video; use image for one starting frame"));
        }
    }
    let audio = match body.get("audio") {
        None | Some(Json::Null) => None,
        Some(v) => Some(v.as_bool().ok_or("audio must be true or false")?),
    };
    let memory = match body.get("memory") {
        None => "auto",
        Some(v) => v.as_str().ok_or("memory must be a string")?,
    };
    if !["auto", "gpu", "ram", "ssd"].contains(&memory) {
        return Err("memory must be auto, gpu, ram or ssd".into());
    }
    let ram_limit = config
        .get("ram_gb")
        .and_then(Json::as_i64)
        .unwrap_or(48)
        .clamp(0, 512);
    let vram_limit = config
        .get("vram_gb")
        .and_then(Json::as_i64)
        .unwrap_or(26)
        .clamp(0, 192);
    let mut fields = vec![
        ("kind".into(), Json::str("video")),
        ("image".into(), image.unwrap_or(Json::Null)),
        ("end_image".into(), end_image.unwrap_or(Json::Null)),
        ("model".into(), Json::str(model)),
        ("prompt".into(), Json::str(prompt)),
        ("width".into(), Json::Int(width)),
        ("height".into(), Json::Int(height)),
        ("frames".into(), Json::Int(frames)),
        ("fps".into(), Json::Int(fps)),
        ("seed".into(), Json::Int(seed)),
        ("device".into(), Json::Int(c.image_device as i64)),
        ("memory".into(), Json::str(memory)),
        (
            "ram_gb".into(),
            Json::Int(number("ram_gb", ram_limit, 0, ram_limit)?),
        ),
        (
            "vram_gb".into(),
            Json::Int(number("vram_gb", vram_limit, 0, vram_limit)?),
        ),
    ];
    let prompt_cache = match body.get("prompt_cache") {
        None => true,
        Some(v) => v.as_bool().ok_or("prompt_cache must be a boolean")?,
    };
    if prompt_cache {
        let cache_request = Json::obj([("output_dir", Json::str(".ltx-prompt-cache"))]);
        fields.push(("cache_dir".into(), Json::str(output_directory(c, &cache_request, "videos")?.to_string_lossy())));
    }
    for key in ["transformer", "text_encoder", "vae", "tokenizer"] {
        let value = selected.get(key).and_then(Json::as_str);
        if key == "tokenizer" && model == "ltx-2.5" && value.is_none() {
            continue;
        }
        let value = value.ok_or_else(|| format!("video config missing {model}.{key}"))?;
        let path = Path::new(value);
        if !path.is_absolute() || !path.is_file() {
            return Err(format!("video weights are not ready: {}", path.display()));
        }
        fields.push((key.into(), Json::str(value)));
    }
    // LTX 2.5 models with an `audio_vae` generate a soundtrack unless asked not to.
    let audio_vae = selected.get("audio_vae").and_then(Json::as_str).filter(|_| model == "ltx-2.5");
    if let Some(value) = audio_vae {
        let path = Path::new(value);
        if !path.is_absolute() || !path.is_file() {
            return Err(format!("audio weights are not ready: {}", path.display()));
        }
        fields.push(("audio_vae".into(), Json::str(value)));
    }
    match audio {
        Some(true) if audio_vae.is_none() => return Err(format!("video model {model} has no audio_vae, so it cannot make sound")),
        Some(on) => fields.push(("audio".into(), Json::Bool(on))),
        None => {}
    }
    if let Some(ffmpeg) = config.get("ffmpeg").and_then(Json::as_str) {
        fields.push(("ffmpeg".into(), Json::str(ffmpeg)));
    }
    fields.push((
        "output_dir".into(),
        Json::str(output_directory(c, body, "videos")?.to_string_lossy()),
    ));
    Ok(Json::Obj(fields))
}

/// Where the worker may keep image weights: the request's `memory`, `ram_gb` and
/// `vram_gb`, defaulting to the catalog's and never above its caps (the same
/// contract as video). The worker places each block on the GPU, in RAM or on the
/// SSD accordingly; see nrob-diffusion's `residency`.
fn image_memory(c: &Config, body: &Json) -> Result<Vec<(String, Json)>, String> {
    let settings = &c.image_memory;
    let memory = match body.get("memory").or_else(|| settings.get("memory")) {
        None | Some(Json::Null) => "auto",
        Some(v) => v.as_str().ok_or("memory must be a string")?,
    };
    if !["auto", "gpu", "ram", "ssd"].contains(&memory) {
        return Err("memory must be auto, gpu, ram or ssd".into());
    }
    let mut fields = vec![("memory".to_string(), Json::str(memory))];
    for (key, default, max) in [("ram_gb", Some(32), 512), ("vram_gb", None, 192)] {
        let cap = settings.get(key).and_then(Json::as_i64).map_or(max, |n| n.clamp(0, max));
        let value = match body.get(key) {
            None | Some(Json::Null) => settings.get(key).and_then(Json::as_i64).map(|n| n.clamp(0, max)).or(default),
            Some(v) => {
                let n = v.as_i64().ok_or_else(|| format!("{key} must be an integer"))?;
                if !(0..=cap).contains(&n) {
                    return Err(format!("{key} must be in 0..{cap}"));
                }
                Some(n)
            }
        };
        if let Some(n) = value {
            fields.push((key.into(), Json::Int(n.min(cap))));
        }
    }
    Ok(fields)
}

fn prepare_sdxl(c: &Config, settings: &Json, body: &Json) -> Result<Json, String> {
    // Only catalog paths can select weights. Validate before unloading the LLM.
    let value = |key| body.get(key).or_else(|| settings.get(key));
    let integer = |key, default, min, max| -> Result<i64, String> {
        let n = match value(key) { None => default, Some(v) => v.as_i64().ok_or_else(|| format!("{key} must be an integer"))? };
        if !(min..=max).contains(&n) { return Err(format!("{key} must be between {min} and {max}")); }
        Ok(n)
    };
    let n = integer("n", 1, 1, 1000)?;
    let width = integer("width", 1024, 256, 2048)?;
    let height = integer("height", 1024, 256, 2048)?;
    if width % 64 != 0 || height % 64 != 0 { return Err("SDXL dimensions must be multiples of 64".into()); }
    let valid_prompt = |v: &Json| v.as_str().is_some_and(|s| !s.trim().is_empty() && s.len() <= 16384);
    match body.get("prompts") {
        Some(v) => {
            let a = v.as_array().ok_or("prompts must be an array")?;
            if (a.len() != 1 && a.len() != n as usize) || a.iter().any(|v| !valid_prompt(v)) {
                return Err("provide one prompt or n nonempty prompts up to 16384 bytes each".into());
            }
        }
        None => if !body.get("prompt").is_some_and(valid_prompt) { return Err("prompt must be a nonempty string up to 16384 bytes".into()); },
    }
    for key in ["image", "images", "adapter"] {
        if body.get(key).is_some_and(|v| !matches!(v, Json::Null) && !v.as_array().is_some_and(|a| a.is_empty())) {
            return Err("SDXL currently supports text-to-image without reference images or LoRA".into());
        }
    }
    if body.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) { return Err("SDXL does not use Qwen turbo".into()); }
    if body.get("weights").is_some_and(|v| v.as_str() != Some("safetensors")) { return Err("SDXL requires safetensors weights".into()); }
    for (key, expected) in [("sampler", "dpmpp_2m"), ("scheduler", "karras")] {
        if value(key).is_some_and(|v| v.as_str() != Some(expected)) { return Err(format!("SDXL {key} must be {expected}")); }
    }
    let cfg = match value("cfg") { None => 2.5, Some(v) => v.as_f64().ok_or("cfg must be numeric")? };
    if !cfg.is_finite() || !(1.0..=30.0).contains(&cfg) { return Err("cfg must be between 1 and 30".into()); }
    let negative = match value("negative_prompt") { None => "", Some(v) => v.as_str().filter(|s| s.len() <= 16384).ok_or("negative_prompt must be a string up to 16384 bytes")? };
    let mut fields = vec![
        ("architecture".into(), Json::str("sdxl")),
        ("checkpoint".into(), settings.get("checkpoint").cloned().ok_or("missing SDXL checkpoint")?),
        ("tokenizer".into(), settings.get("tokenizer").cloned().ok_or("missing SDXL tokenizer")?),
        ("device".into(), Json::Int(c.image_device as i64)),
        ("n".into(), Json::Int(n)), ("width".into(), Json::Int(width)), ("height".into(), Json::Int(height)),
        ("steps".into(), Json::Int(integer("steps", 16, 2, 100)?)),
        ("seed".into(), Json::Int(integer("seed", 0, 0, i64::MAX - n)?)),
        ("clip_skip".into(), Json::Int(integer("clip_skip", 1, 1, 11)?)),
        ("cfg".into(), Json::Num(cfg)), ("negative_prompt".into(), Json::str(negative)),
        ("sampler".into(), Json::str("dpmpp_2m")), ("scheduler".into(), Json::str("karras")),
        ("output_dir".into(), Json::str(output_directory(c, body, "images")?.to_string_lossy())),
    ];
    if let Some(model) = &c.image_model { fields.push(("model".into(), Json::str(model))); }
    fields.extend(image_memory(c, body)?);
    for key in ["prompt", "prompts"] {
        if let Some(v) = body.get(key) { fields.push((key.into(), v.clone())); }
    }
    let request = Json::Obj(fields);
    if request.to_json().len() > 2 * 1024 * 1024 { return Err("image request exceeds 2 MiB".into()); }
    Ok(request)
}

fn prepare(c: &Config, body: &Json) -> Result<Json, String> {
    let selected = match body.get("model") {
        None => None,
        Some(v) => Some(v.as_str().filter(|s| !s.trim().is_empty()).ok_or("model must be a nonempty configured name")?),
    };
    let current = crate::media_catalog::image(c, selected)?;
    let c = &current;
    if let Some(settings) = &c.sdxl {
        return prepare_sdxl(c, settings, body);
    }
    let images = reference_paths(body, true)?;
    let prompt = body.get("prompt").cloned();
    let prompts = body.get("prompts").cloned();
    if prompt.is_none() && prompts.is_none() {
        return Err("prompt or prompts required".into());
    }
    if let Some(p) = &prompt {
        if !p
            .as_str()
            .is_some_and(|s| !s.trim().is_empty() && s.len() <= 16384)
        {
            return Err("prompt must be a nonempty string up to 16384 bytes".into());
        }
    }
    let number = |key: &str, default: i64, min: i64, max: i64| -> Result<i64, String> {
        let n = match body.get(key) {
            None => default,
            Some(v) => v
                .as_i64()
                .ok_or_else(|| format!("{key} must be an integer"))?,
        };
        if !(min..=max).contains(&n) {
            return Err(format!("{key} must be between {min} and {max}"));
        }
        Ok(n)
    };
    let n = number("n", 1, 1, 1000)?;
    let width = number("width", 1024, 256, 2048)?;
    let height = number("height", 1024, 256, 2048)?;
    let reference_size = number("reference_size", 1024, 256, 1024)?;
    if reference_size % 32 != 0 {
        return Err("reference_size must be a multiple of 32".into());
    }
    if width % 32 != 0 || height % 32 != 0 {
        return Err("image dimensions must be multiples of 32".into());
    }
    if let Some(p) = &prompts {
        let a = p.as_array().ok_or("prompts must be an array")?;
        if (a.len() != 1 && a.len() != n as usize)
            || a.iter().any(|v| {
                !v.as_str()
                    .is_some_and(|s| !s.trim().is_empty() && s.len() <= 16384)
            })
        {
            return Err("provide one prompt or n nonempty prompts up to 16384 bytes each".into());
        }
    }
    let turbo = match body.get("turbo") {
        None => true,
        Some(v) => v.as_bool().ok_or("turbo must be boolean")?,
    };
    let weights = match body.get("weights") {
        None => &c.default_weights,
        Some(v) => v.as_str().ok_or("weights must be a string")?,
    };
    let transformer = match weights {
        "gguf" => c.transformer.clone(),
        "safetensors" => c.safetensors_transformer.clone().unwrap_or_else(|| c.base.join("transformer")),
        _ => return Err("weights must be gguf or safetensors".into()),
    };
    if c.media_dir.is_some() && !transformer.exists() {
        return Err(format!("selected image weights are missing: {}", transformer.display()));
    }
    if turbo && c.adapter.is_none() {
        return Err("turbo adapter is not configured".into());
    }
    let steps = number("steps", if turbo { 6 } else { 40 }, 2, 100)?;
    if turbo && !matches!(steps, 4 | 6) {
        return Err("turbo steps must be 4 or 6".into());
    }
    let seed = number("seed", 0, 0, i64::MAX - n)?;
    let out = output_directory(c, body, "images")?;
    let mut fields = vec![
        ("base".into(), Json::str(c.base.to_string_lossy())),
        (
            "transformer".into(),
            Json::str(transformer.to_string_lossy()),
        ),
        (
            "adapter".into(),
            if turbo {
                Json::str(c.adapter.as_ref().unwrap().to_string_lossy())
            } else {
                Json::Null
            },
        ),
        ("output_dir".into(), Json::str(out.to_string_lossy())),
        ("device".into(), Json::Int(c.image_device as i64)),
        ("n".into(), Json::Int(n)),
        ("width".into(), Json::Int(width)),
        ("height".into(), Json::Int(height)),
        ("steps".into(), Json::Int(steps)),
        ("seed".into(), Json::Int(seed)),
        ("images".into(), Json::Arr(images)),
        ("reference_size".into(), Json::Int(reference_size)),
    ];
    if let Some(p) = &c.text_encoder { fields.push(("text_encoder".into(), Json::str(p.to_string_lossy()))); }
    fields.extend(image_memory(c, body)?);
    if let Some(name) = &c.image_model { fields.push(("model".into(), Json::str(name))); }
    if let Some(p) = prompt {
        fields.push(("prompt".into(), p));
    }
    if let Some(p) = prompts {
        fields.push(("prompts".into(), p));
    }
    let request = Json::Obj(fields);
    if request.to_json().len() > 2 * 1024 * 1024 {
        return Err("image request exceeds 2 MiB".into());
    }
    Ok(request)
}

/// Shares the server's existing opt-in policy for reading local image paths.
pub(crate) fn video_reference_path(body: &Json, key: &str, allow_local: bool) -> Result<Option<Json>, String> {
    match body.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(image) => {
            let wrapper = Json::obj([("images", Json::Arr(vec![image.clone()]))]);
            Ok(reference_paths(&wrapper, allow_local)?.into_iter().next())
        }
    }
}

pub(crate) fn reference_paths(body: &Json, allow_local: bool) -> Result<Vec<Json>, String> {
    let Some(value) = body.get("images") else {
        return Ok(Vec::new());
    };
    let paths = value
        .as_array()
        .ok_or("images must be an array of up to three local paths")?;
    if paths.len() > 3 {
        return Err("images accepts at most three reference images".into());
    }
    if !paths.is_empty() && !allow_local {
        return Err(
            "local reference images are disabled; enable --local-images on a trusted server".into(),
        );
    }
    paths
        .iter()
        .map(|p| {
            let p = p
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or("reference paths must be nonempty strings")?;
            let path = Path::new(p);
            if !path.is_absolute() {
                return Err("reference images must use absolute local paths".into());
            }
            let meta = std::fs::metadata(path).map_err(|e| format!("reference image {p}: {e}"))?;
            if !meta.is_file() || meta.len() > 32 * 1024 * 1024 {
                return Err("each reference must be a regular file of at most 32 MiB".into());
            }
            Ok(Json::str(
                path.canonicalize()
                    .map_err(|e| e.to_string())?
                    .to_string_lossy(),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sdxl_catalog_routes_checkpoint_and_validates_before_handoff() {
        let root = std::env::temp_dir().join(format!("nrob-sdxl-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("checkpoint.safetensors");
        std::fs::write(&fixture, b"fixture").unwrap();
        let write = |name: &str, j: Json| std::fs::write(root.join(name), j.to_json()).unwrap();
        write("controller.json", Json::obj([
            ("worker", Json::str("checkpoint.safetensors")), ("controller_path", Json::str("checkpoint.safetensors")),
            ("controller_name", Json::str("qwen")), ("controller_device", Json::Int(0)),
            ("image_device", Json::Int(1)), ("output_root", Json::str("outputs")),
        ]));
        write("image.json", Json::obj([
            // Missing shared Qwen files must not prevent selecting SDXL.
            ("base", Json::str("missing-qwen")), ("adapter", Json::str("missing-adapter")),
            ("default_model", Json::str("anime")),
            ("models", Json::obj([("anime", Json::obj([
                ("architecture", Json::str("sdxl")), ("checkpoint", Json::str("checkpoint.safetensors")),
                ("tokenizer", Json::str("checkpoint.safetensors")), ("steps", Json::Int(20)), ("cfg", Json::Num(3.0)),
            ]))])),
        ]));
        let cfg = Config::read(&root).unwrap();
        let request = |extra: &str| Json::parse(format!(r#"{{"prompt":"anime fox"{extra}}}"#).as_bytes()).unwrap();
        let prepared = prepare(&cfg, &request("")).unwrap();
        assert_eq!(prepared.get("architecture").and_then(Json::as_str), Some("sdxl"));
        assert_eq!(prepared.get("steps").and_then(Json::as_i64), Some(20));
        assert_eq!(prepared.get("cfg").and_then(Json::as_f64), Some(3.0));
        assert!(prepared.get("adapter").is_none());
        assert_eq!(prepared.get("device").and_then(Json::as_i64), Some(1));
        let overridden = prepare(&cfg, &request(r#", "steps":16,"cfg":2.5,"negative_prompt":"blurry","checkpoint":"untrusted.safetensors""#)).unwrap();
        assert_eq!(overridden.get("steps").and_then(Json::as_i64), Some(16));
        assert_eq!(overridden.get("checkpoint"), prepared.get("checkpoint"));
        assert_eq!(overridden.get("negative_prompt").and_then(Json::as_str), Some("blurry"));
        assert_eq!(cfg.capabilities().get("image").unwrap().get("available").and_then(Json::as_bool), Some(true));
        for extra in [r#", "width":544"#, r#", "cfg":0"#, r#", "cfg":"2.5""#, r#", "turbo":true"#,
            r#", "images":["x.png"]"#, r#", "weights":"gguf""#, r#", "sampler":"euler""#,
            r#", "scheduler":"normal""#, r#", "clip_skip":0"#, r#", "steps":1"#, r#", "prompts":[]"#,
            r#", "negative_prompt":3"#, r#", "output_dir":"../escape""#,
        ] { assert!(prepare(&cfg, &request(extra)).is_err(), "{extra}"); }
        std::fs::remove_file(fixture).unwrap();
        assert!(prepare(&cfg, &request("")).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn live_catalog_add_remove_disable_and_defaults() {
        let root = std::env::temp_dir().join(format!("nrob-live-media-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("weights.safetensors"), b"fixture").unwrap();
        std::fs::write(root.join("other.gguf"), b"fixture").unwrap();
        let write = |name: &str, j: Json| std::fs::write(root.join(name), j.to_json()).unwrap();
        write("controller.json", Json::obj([
            ("worker", Json::str("other.gguf")), ("controller_path", Json::str("other.gguf")),
            ("controller_name", Json::str("qwen")), ("controller_device", Json::Int(0)),
            ("image_device", Json::Int(1)), ("output_root", Json::str("outputs")),
        ]));
        let cfg = Config::read(&root).unwrap();
        let request = Json::obj([("prompt", Json::str("a fox"))]);
        let available = |kind: &str| cfg.capabilities().get(kind).unwrap().get("available").unwrap().as_bool().unwrap();
        assert!(!available("image") && !available("video"));
        assert!(prepare(&cfg, &request).is_err());
        write("image.json", Json::obj([
            ("base", Json::str(".")), ("transformer", Json::str("other.gguf")),
            ("safetensors_transformer", Json::str("weights.safetensors")),
            ("adapter", Json::str("weights.safetensors")), ("default_weights", Json::str("safetensors")),
        ]));
        assert!(available("image"));
        let queued = prepare(&cfg, &request).unwrap();
        assert!(queued.get("transformer").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
        assert_eq!(queued.get("steps").and_then(Json::as_i64), Some(6));
        write("image.json", Json::obj([
            ("base", Json::str(".")), ("default_weights", Json::str("safetensors")),
            ("adapter", Json::str("weights.safetensors")), ("default_model", Json::str("red")),
            ("models", Json::obj([
                ("red", Json::obj([("safetensors_transformer", Json::str("weights.safetensors"))])),
                ("realism", Json::obj([("safetensors_transformer", Json::str("other.gguf")), ("text_encoder", Json::str("weights.safetensors"))])),
                ("off", Json::obj([("enabled", Json::Bool(false))])),
            ])),
        ]));
        let default = prepare(&cfg, &request).unwrap();
        assert_eq!(default.get("model").and_then(Json::as_str), Some("red"));
        let named = prepare(&cfg, &Json::obj([
            ("prompt",Json::str("fox")), ("model",Json::str("realism")),
            ("text_encoder",Json::str("untrusted")),
        ])).unwrap();
        assert!(named.get("transformer").unwrap().as_str().unwrap().ends_with("other.gguf"));
        assert!(named.get("text_encoder").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
        assert_eq!(cfg.capabilities().get("image").unwrap().get("models").unwrap().as_array().unwrap().len(),3);
        for name in ["off", "missing", "../../weights.safetensors", ""] {
            assert!(prepare(&cfg,&Json::obj([("prompt",Json::str("fox")),("model",Json::str(name))])).is_err());
        }
        write("image.json", Json::obj([("enabled", Json::Bool(false))]));
        assert!(!available("image"));
        assert!(prepare(&cfg, &request).is_err());
        // Prepared jobs retain their snapshot even when the catalog changes.
        assert!(queued.get("transformer").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
        let weights = Json::obj(["transformer", "text_encoder", "vae", "tokenizer"].map(|k| (k, Json::str("weights.safetensors"))));
        write("video.json", Json::obj([
            ("default_model", Json::str("sulphur-2")),
            ("models", Json::obj([("sulphur-2", weights)])),
        ]));
        assert!(available("video"));
        let video = prepare_video(&cfg, &request).unwrap();
        assert_eq!(video.get("model").and_then(Json::as_str), Some("sulphur-2"));
        assert!(Path::new(video.get("transformer").unwrap().as_str().unwrap()).is_absolute());
        std::fs::remove_file(root.join("video.json")).unwrap();
        assert!(!available("video"));
        assert!(prepare_video(&cfg, &request).is_err());
        std::fs::write(root.join("image.json"), b"invalid JSON").unwrap();
        assert!(!available("image"));
        assert!(Config::read(&root).is_ok()); // a broken optional manifest does not prevent chat startup
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn video_requests_use_trusted_paths_and_bounded_memory() {
        let root = std::env::temp_dir().join(format!("nrob-video-config-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let weight = root.join("fixture.safetensors");
        std::fs::write(&weight, b"fixture").unwrap();
        let model = Json::obj(
            ["transformer", "text_encoder", "vae", "tokenizer"]
                .map(|k| (k, Json::str(weight.to_string_lossy()))),
        );
        let manifest = root.join("video.json");
        std::fs::write(
            &manifest,
            Json::obj([
                ("models", Json::obj([("sulphur-2", model)])),
                ("ram_gb", Json::Int(4)),
                ("vram_gb", Json::Int(8)),
            ])
            .to_json(),
        )
        .unwrap();
        let mut cfg = config();
        cfg.output_root = root.join("output");
        cfg.video_config = Some(manifest);
        let valid = |extra: &str| {
            Json::parse(
                format!(r#"{{"model":"sulphur-2","prompt":"A bird in flight"{extra}}}"#).as_bytes(),
            )
            .unwrap()
        };
        let r = prepare_video(
            &cfg,
            &valid(r#", "memory":"ssd", "transformer":"untrusted", "device":99"#),
        )
        .unwrap();
        assert_eq!(r.get("kind").and_then(Json::as_str), Some("video"));
        assert_eq!(r.get("memory").and_then(Json::as_str), Some("ssd"));
        assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
        assert!(r.get("cache_dir").and_then(Json::as_str).unwrap().ends_with(".ltx-prompt-cache"));
        assert!(prepare_video(&cfg, &valid(r#", "prompt_cache":false"#)).unwrap().get("cache_dir").is_none());
        for key in ["image", "end_image"] {
            let reference = Json::obj([(key, Json::str(weight.to_string_lossy()))]);
            assert!(video_reference_path(&reference, key, false).is_err());
            let expected = video_reference_path(&reference, key, true).unwrap().unwrap();
            let mut body = valid("");
            if let Json::Obj(fields) = &mut body { fields.push((key.into(), Json::str(weight.to_string_lossy()))); }
            assert_eq!(prepare_video(&cfg, &body).unwrap().get(key).unwrap().to_json(), expected.to_json());
            assert!(video_reference_path(&Json::obj([(key, Json::Null)]), key, false).unwrap().is_none());
            for invalid in [Json::Int(1), Json::str("relative.png"), Json::str("")] {
                assert!(video_reference_path(&Json::obj([(key, invalid)]), key, true).is_err());
            }
        }
        assert_eq!(
            r.get("transformer").and_then(Json::as_str),
            Some(weight.to_str().unwrap())
        );
        for extra in [
            r#", "ram_gb":5"#,
            r#", "vram_gb":9"#,
            r#", "frames":48"#,
            r#", "steps":6"#,
            r#", "output_dir":"../outside""#,
            r#", "images":["x.png"]"#,
        ] {
            assert!(prepare_video(&cfg, &valid(extra)).is_err(), "{extra}");
        }
        std::fs::remove_file(weight).unwrap();
        assert!(prepare_video(&cfg, &valid(""))
            .unwrap_err()
            .contains("not ready"));
        std::fs::remove_dir_all(root).unwrap();
    }
    fn config() -> Config {
        Config {
            media_dir: None,
            default_weights: "gguf".into(),
            image_model: None,
            text_encoder: None,
            sdxl: None,
            image_memory: Json::obj([] as [(&str, Json); 0]),
            worker: "worker".into(),
            base: "base".into(),
            transformer: "model.gguf".into(),
            safetensors_transformer: None,
            adapter: Some("turbo.safetensors".into()),
            output_root: std::env::temp_dir()
                .join(format!("nrob-image-test-{}", std::process::id())),
            controller_name: "controller".into(),
            controller_path: "controller.gguf".into(),
            controller_device: 0,
            image_device: 1,
            video_config: None,
        }
    }
    #[test]
    fn rejects_invalid_batches_before_creating_output() {
        let cfg = config();
        for json in [
            r#"{"prompt":"x","n":0}"#,
            r#"{"prompt":"x","n":1001}"#,
            r#"{"prompt":"x","steps":5}"#,
            r#"{"prompt":"x","width":257}"#,
            r#"{"prompt":"x","output_dir":"../escape"}"#,
            r#"{"prompts":["a","b"],"n":3}"#,
            r#"{"prompt":"x","weights":3}"#,
            r#"{"prompt":"x","images":["a","b","c","d"]}"#,
            r#"{"prompt":"x","images":"file.png"}"#,
            r#"{"prompt":"x","images":[null]}"#,
            r#"{"prompt":"x","images":["relative.png"]}"#,
            r#"{"prompt":"x","reference_size":1025}"#,
        ] {
            assert!(
                prepare(&cfg, &Json::parse(json.as_bytes()).unwrap()).is_err(),
                "{json}"
            );
        }
    }
    #[test]
    fn optional_references_obey_local_file_policy_and_preserve_order() {
        assert!(reference_paths(&Json::obj([] as [(&str, Json); 0]), false)
            .unwrap()
            .is_empty());
        assert!(
            reference_paths(&Json::obj([("images", Json::Arr(vec![]))]), false)
                .unwrap()
                .is_empty()
        );
        let path = std::env::temp_dir().join(format!("nrob-reference-{}.png", std::process::id()));
        std::fs::write(&path, b"fixture").unwrap();
        for count in 1..=3 {
            let body = Json::obj([(
                "images",
                Json::Arr(
                    (0..count)
                        .map(|_| Json::str(path.to_string_lossy()))
                        .collect(),
                ),
            )]);
            assert!(reference_paths(&body, false).is_err());
            assert_eq!(reference_paths(&body, true).unwrap().len(), count);
        }
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn batch_parameters_and_server_owned_paths_are_preserved() {
        let mut cfg = config();
        let turbo = prepare(&cfg, &Json::parse(br#"{"prompt":"x"}"#).unwrap()).unwrap();
        assert_eq!(turbo.get("steps").and_then(Json::as_i64), Some(6));
        assert_eq!(
            turbo.get("adapter").and_then(Json::as_str),
            Some("turbo.safetensors")
        );
        let r=prepare(&cfg,&Json::parse(br#"{"prompt":"x","n":100,"steps":4,"weights":"safetensors","base":"untrusted","device":99}"#).unwrap()).unwrap();
        assert_eq!(r.get("n").and_then(Json::as_i64), Some(100));
        assert_eq!(r.get("steps").and_then(Json::as_i64), Some(4));
        assert_eq!(r.get("base").and_then(Json::as_str), Some("base"));
        assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
        assert_eq!(
            Path::new(r.get("transformer").unwrap().as_str().unwrap()),
            Path::new("base").join("transformer")
        );
        let base = prepare(
            &cfg,
            &Json::parse(br#"{"prompt":"x","turbo":false}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(base.get("adapter"), Some(&Json::Null));
        assert_eq!(base.get("steps").and_then(Json::as_i64), Some(40));
        cfg.safetensors_transformer = Some("custom.safetensors".into());
        let custom = prepare(&cfg, &Json::parse(br#"{"prompt":"x","weights":"safetensors","transformer":"untrusted","safetensors_transformer":"untrusted"}"#).unwrap()).unwrap();
        assert_eq!(custom.get("transformer").and_then(Json::as_str), Some("custom.safetensors"));
        let gguf = prepare(&cfg, &Json::parse(br#"{"prompt":"x","weights":"gguf"}"#).unwrap()).unwrap();
        assert_eq!(gguf.get("transformer").and_then(Json::as_str), Some("model.gguf"));
        std::fs::remove_dir_all(&cfg.output_root).unwrap();
    }
    #[test]
    fn image_memory_defaults_to_auto_and_respects_catalog_caps() {
        let mut cfg = config();
        cfg.output_root = std::env::temp_dir().join(format!("nrob-image-memory-{}", std::process::id()));
        let body = |s: &str| Json::parse(s.as_bytes()).unwrap();
        let r = prepare(&cfg, &body(r#"{"prompt":"x"}"#)).unwrap();
        assert_eq!(r.get("memory").and_then(Json::as_str), Some("auto"));
        assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(32));
        assert!(r.get("vram_gb").is_none());
        cfg.image_memory = body(r#"{"memory":"ssd","ram_gb":8,"vram_gb":12}"#);
        let r = prepare(&cfg, &body(r#"{"prompt":"x"}"#)).unwrap();
        assert_eq!(r.get("memory").and_then(Json::as_str), Some("ssd"));
        assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(8));
        assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(12));
        let r = prepare(&cfg, &body(r#"{"prompt":"x","memory":"ram","ram_gb":4,"vram_gb":0}"#)).unwrap();
        assert_eq!(r.get("memory").and_then(Json::as_str), Some("ram"));
        assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(4));
        assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(0));
        for extra in [r#","ram_gb":9"#, r#","vram_gb":13"#, r#","memory":"disk""#, r#","ram_gb":"4""#] {
            assert!(prepare(&cfg, &body(&format!(r#"{{"prompt":"x"{extra}}}"#))).is_err(), "{extra}");
        }
        std::fs::remove_dir_all(&cfg.output_root).unwrap();
    }
    #[test]
    fn configured_safetensors_override_is_optional_and_validated() {
        let root = std::env::temp_dir().join(format!("nrob-safe-config-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let checkpoint = root.join("custom.safetensors");
        std::fs::write(&checkpoint, b"fixture").unwrap();
        let config_path = root.join("config.json");
        for (value, valid) in [
            (Json::Null, true),
            (Json::str(checkpoint.to_string_lossy()), true),
            (Json::Int(1), false),
            (Json::str(""), false),
            (Json::str("relative.safetensors"), false),
            (Json::str(root.join("missing.safetensors").to_string_lossy()), false),
        ] {
            let body = Json::obj([
                ("worker", Json::str(checkpoint.to_string_lossy())),
                ("base", Json::str(root.to_string_lossy())),
                ("transformer", Json::str(checkpoint.to_string_lossy())),
                ("safetensors_transformer", value),
                ("output_root", Json::str(root.to_string_lossy())),
                ("controller_name", Json::str("controller")),
                ("controller_path", Json::str(checkpoint.to_string_lossy())),
                ("controller_device", Json::Int(0)),
                ("image_device", Json::Int(1)),
            ]);
            std::fs::write(&config_path, body.to_json()).unwrap();
            assert_eq!(Config::read(&config_path).is_ok(), valid);
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
