//! Native Rust diffusion subprocess supervision. The worker owns all image
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
    pub worker: PathBuf,
    pub base: PathBuf,
    pub transformer: PathBuf,
    pub adapter: Option<PathBuf>,
    pub output_root: PathBuf,
    pub controller_name: String,
    pub controller_path: PathBuf,
    pub controller_device: usize,
    pub image_device: usize,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self, String> {
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
            worker: s("worker")?.into(),
            base: s("base")?.into(),
            transformer: s("transformer")?.into(),
            adapter: j.get("adapter").and_then(Json::as_str).map(PathBuf::from),
            output_root: s("output_root")?.into(),
            controller_name: s("controller_name")?,
            controller_path: s("controller_path")?.into(),
            controller_device: n("controller_device")?,
            image_device: n("image_device")?,
        };
        for p in [&c.worker, &c.base, &c.transformer, &c.controller_path]
            .into_iter()
            .chain(c.adapter.iter())
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
        Json::obj([
            ("configured", Json::Bool(config.is_some())),
            (
                "controller_model",
                config.map_or(Json::Null, |c| Json::str(&c.controller_name)),
            ),
            (
                "job",
                job.as_ref().map_or(Json::Null, |j| {
                    Json::obj([
                        ("id", Json::Int(j.id as i64)),
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
            return Err("cancel or finish the image batch before releasing the controller".into());
        }
        self.models.release_images();
        Ok(())
    }
    pub fn submit(self: &Arc<Self>, body: &Json) -> Result<Json, String> {
        let cfg = self
            .models
            .image_config()
            .ok_or("image generation is disabled; configure --image-config")?
            .clone();
        let request = prepare(&cfg, body)?;
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
                return Err("an image batch is already active".into());
            }
            *job = Some(Job {
                id,
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
            ("status_url", Json::str("/v1/images/status")),
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

fn prepare(c: &Config, body: &Json) -> Result<Json, String> {
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
        None => "gguf",
        Some(v) => v.as_str().ok_or("weights must be a string")?,
    };
    let transformer = match weights {
        "gguf" => c.transformer.clone(),
        "safetensors" => c.base.join("transformer"),
        _ => return Err("weights must be gguf or safetensors".into()),
    };
    if turbo && c.adapter.is_none() {
        return Err("turbo adapter is not configured".into());
    }
    let steps = number("steps", if turbo { 6 } else { 40 }, 2, 100)?;
    if turbo && !matches!(steps, 4 | 6) {
        return Err("turbo steps must be 4 or 6".into());
    }
    let seed = number("seed", 0, 0, i64::MAX - n)?;
    let folder = match body.get("output_dir") {
        None => "images",
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
    fn config() -> Config {
        Config {
            worker: "worker".into(),
            base: "base".into(),
            transformer: "model.gguf".into(),
            adapter: Some("turbo.safetensors".into()),
            output_root: std::env::temp_dir()
                .join(format!("nrob-image-test-{}", std::process::id())),
            controller_name: "controller".into(),
            controller_path: "controller.gguf".into(),
            controller_device: 0,
            image_device: 1,
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
        let cfg = config();
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
        std::fs::remove_dir_all(&cfg.output_root).unwrap();
    }
}
