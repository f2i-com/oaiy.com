//! Native Rust diffusion subprocess supervision. The worker owns all media
//! allocations: exit (including cancellation) releases its CUDA context.
use oaiy_engine::json::Json;
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
    pub klein: Option<Json>,
    /// Image weight residency defaults and caps from the catalog: `memory`
    /// (auto|gpu|ram|ssd), `ram_gb`, `vram_gb`. An object; empty when unset.
    pub image_memory: Json,
    pub worker: PathBuf,
    pub base: PathBuf,
    pub transformer: PathBuf,
    pub safetensors_transformer: Option<PathBuf>,
    pub adapter: Option<PathBuf>,
    /// More Qwen Image LoRA adapters, each with its strength, applied with the turbo one.
    pub loras: Vec<(PathBuf, f64)>,
    pub output_root: PathBuf,
    pub controller_name: String,
    pub controller_path: PathBuf,
    pub controller_device: usize,
    pub image_device: usize,
    pub video_config: Option<PathBuf>,
}
/// A model's `loras`: `[{"path": …, "strength": 0.8}]` (strength 1 when left out) or plain paths,
/// relative ones resolved against `root` when there is one.
pub(crate) fn loras(root: Option<&Path>, j: &Json) -> Result<Vec<(PathBuf, f64)>, String> {
    let Some(list) = j.get("loras") else { return Ok(Vec::new()) };
    if matches!(list, Json::Null) {
        return Ok(Vec::new());
    }
    list.as_array()
        .ok_or("loras must be an array of {path, strength}")?
        .iter()
        .map(|l| {
            let (path, strength) = match l {
                Json::Str(p) => (Some(p.as_str()), None),
                _ => (l.get("path").and_then(Json::as_str), l.get("strength")),
            };
            let path = path.filter(|p| !p.trim().is_empty()).ok_or("each LoRA needs a path")?;
            let strength = match strength {
                None | Some(Json::Null) => 1.,
                Some(v) => v.as_f64().filter(|s| s.is_finite() && (-4. ..=4.).contains(s)).ok_or("a LoRA's strength must be a number between -4 and 4")?,
            };
            let path = PathBuf::from(path);
            let path = match root {
                Some(r) if path.is_relative() => r.join(path),
                _ => path,
            };
            if !path.is_file() {
                return Err(format!("LoRA not found: {}", path.display()));
            }
            Ok((path, strength))
        })
        .collect()
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
            klein: None,
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
            loras: loras(None, &j)?,
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
            .chain(c.loras.iter().map(|(p, _)| p))
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
    // Models with an `audio_vae` generate a soundtrack unless asked not to
    // (for LTX 2.3 and Sulphur it is the checkpoint itself).
    let audio_vae = selected.get("audio_vae").and_then(Json::as_str);
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
    // the model's backend, else the catalog's (a request's own is not taken: where a model runs is the operator's)
    if let Some(backend) = selected.get("backend").or_else(|| config.get("backend")) {
        let b = backend.as_str().filter(|b| ["cpu", "webgpu"].contains(b)).ok_or("video backend must be webgpu or cpu (there is no CUDA backend any more)")?;
        fields.push(("backend".into(), Json::str(b)));
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
/// SSD accordingly; see oaiy-media's `residency`.
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
    // the catalog's backend (a request's own is not taken: where a model runs is the operator's): webgpu, which is
    // also what a catalog naming none gets, or cpu
    if let Some(backend) = settings.get("backend") {
        let b = backend.as_str().filter(|b| ["cpu", "webgpu"].contains(b)).ok_or("image backend must be webgpu or cpu (there is no CUDA backend any more)")?;
        fields.push(("backend".into(), Json::str(b)));
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

fn prepare_klein(c: &Config, settings: &Json, body: &Json) -> Result<Json, String> {
    let value = |key| body.get(key).or_else(|| settings.get(key));
    let integer = |key, default, min, max| -> Result<i64, String> {
        let n = match value(key) { None => default, Some(v) => v.as_i64().ok_or_else(|| format!("{key} must be an integer"))? };
        if !(min..=max).contains(&n) { return Err(format!("{key} must be between {min} and {max}")); }
        Ok(n)
    };
    let n = integer("n", 1, 1, 1000)?;
    let width = integer("width", 1024, 256, 2048)?;
    let height = integer("height", 1024, 256, 2048)?;
    if width % 16 != 0 || height % 16 != 0 { return Err("Klein dimensions must be multiples of 16".into()); }
    let valid_prompt = |v: &Json| v.as_str().is_some_and(|s| !s.trim().is_empty() && s.len() <= 16384);
    match body.get("prompts") {
        Some(v) => {
            let a = v.as_array().ok_or("prompts must be an array")?;
            if (a.len() != 1 && a.len() != n as usize) || a.iter().any(|v| !valid_prompt(v)) { return Err("provide one prompt or n nonempty prompts up to 16384 bytes each".into()); }
        }
        None => if !body.get("prompt").is_some_and(valid_prompt) { return Err("prompt must be nonempty and at most 16384 bytes".into()); },
    }
    for key in ["image", "images", "input_image", "input_reference", "adapter"] {
        if body.get(key).is_some_and(|v| !matches!(v, Json::Null) && !v.as_array().is_some_and(|a| a.is_empty())) { return Err("native Klein currently supports text-to-image without reference images or negative prompts".into()); }
    }
    if body.get("negative_prompt").is_some_and(|v| !matches!(v, Json::Null) && !v.as_str().is_some_and(|s| s.trim().is_empty())) {
        return Err("native Klein does not support negative prompt conditioning".into());
    }
    if body.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) { return Err("Klein does not use Qwen turbo".into()); }
    let variant = settings.get("variant").and_then(Json::as_str).unwrap_or("distilled");
    if !["distilled", "base"].contains(&variant) { return Err("Klein variant must be distilled or base".into()); }
    let distilled = variant == "distilled";
    let steps = integer("steps", if distilled { 4 } else { 50 }, 1, 100)?;
    let cfg = match value("cfg") { None => if distilled { 1.0 } else { 4.0 }, Some(v) => v.as_f64().ok_or("cfg must be numeric")? };
    if !cfg.is_finite() || !(1.0..=10.0).contains(&cfg) || (distilled && (steps != 4 || cfg != 1.0)) { return Err("distilled Klein requires four steps and cfg 1; base cfg must be between 1 and 10".into()); }
    let mut fields = vec![
        ("architecture".into(), Json::str("flux2-klein-4b")), ("variant".into(), Json::str(variant)),
        ("n".into(), Json::Int(n)), ("width".into(), Json::Int(width)), ("height".into(), Json::Int(height)),
        ("steps".into(), Json::Int(steps)), ("cfg".into(), Json::Num(cfg)),
        ("seed".into(), Json::Int(integer("seed", 0, 0, i64::MAX - n)?)),
        ("device".into(), Json::Int(c.image_device as i64)),
        ("output_dir".into(), Json::str(output_directory(c, body, "images")?.to_string_lossy())),
    ];
    for key in ["transformer", "text_encoder", "vae", "tokenizer"] { fields.push((key.into(), settings.get(key).cloned().ok_or_else(|| format!("missing Klein {key}"))?)); }
    let enabled = match body.get("use_loras") { None => true, Some(v) => v.as_bool().ok_or("use_loras must be boolean")? };
    fields.push(("loras".into(), if enabled { Json::Arr(c.loras.iter().map(|(p, s)| Json::obj([("path", Json::str(p.to_string_lossy())), ("strength", Json::Num(*s))])).collect()) } else { Json::Arr(Vec::new()) }));
    if let Some(model) = &c.image_model { fields.push(("model".into(), Json::str(model))); }
    fields.extend(image_memory(c, body)?);
    for key in ["prompt", "prompts"] { if let Some(v) = body.get(key) { fields.push((key.into(), v.clone())); } }
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
    if let Some(settings) = &c.klein { return prepare_klein(c, settings, body); }
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
    if !c.loras.is_empty() {
        fields.push(("loras".into(), Json::Arr(c.loras.iter().map(|(p, s)| Json::obj([("path", Json::str(p.to_string_lossy())), ("strength", Json::Num(*s))])).collect())));
    }
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
mod tests;
