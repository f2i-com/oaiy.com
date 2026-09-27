//! Supervision of the language model: `nrob-server` as a child process on a
//! private loopback port with a random key. The gateway proxies to it; nothing
//! else can reach it.
//!
//! A separate process rather than a linked library: nrob-server needs CUDA at
//! build time and holds GPU memory that only process exit reliably returns, so
//! stopping it (for a media job that needs its GPU, or after idling) is a kill,
//! and a crash takes down the model, not the studio.

use crate::config;
use crate::util::{bool_or, int_or, num_or, random_id, str_or, LogRing};
use nrob::json::Json;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Stopped,
    Starting,
    Ready,
    Failed,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            State::Stopped => "stopped",
            State::Starting => "starting",
            State::Ready => "ready",
            State::Failed => "failed",
        }
    }
}

struct Inner {
    state: State,
    error: Option<String>,
    child: Option<Child>,
    /// Held open for `--watch-stdin`: when the studio exits, however it exits,
    /// the pipe closes and nrob-server stops, returning its GPU memory.
    lifeline: Option<std::process::ChildStdin>,
    addr: String,
    key: String,
    /// Bumped on every start and stop, so threads of an older process stand down.
    generation: u64,
    started: Option<Instant>,
    ready_after: Option<f64>,
    /// Stopped by the broker for a media job (and so restarted after it).
    paused: bool,
    last_used: Instant,
    /// Model names served by the running process, default first.
    models: Vec<String>,
    /// The context the loaded model was opened with, as nrob-server reported it.
    context: Option<i64>,
    command: String,
    /// Launches still to try if this one dies while loading (auto: CUDA, then WebGPU).
    fallbacks: Vec<Launch>,
    /// Arguments, key and folder of the current launch, for a fallback.
    relaunch: Option<(Vec<String>, String, PathBuf)>,
}

/// One way to start the server: the program and its `--backend`.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub backend: &'static str,
}

/// Whether an NVIDIA driver answers (`nvidia-smi -L` lists a GPU).
fn nvidia_present() -> bool {
    let mut c = Command::new("nvidia-smi");
    c.arg("-L").stdin(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c.output().is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GPU"))
}

/// Which server programs to try, in order, for `llm.backend`:
/// - `cuda`: `llm.server` (the CUDA build);
/// - `webgpu` / `cpu`: `llm.server_webgpu` (built without CUDA, so it starts on
///   any machine);
/// - `auto`: the CUDA build when an NVIDIA GPU answers and the program exists,
///   falling back to the WebGPU build if it dies while loading; the WebGPU build
///   (WebGPU, else CPU) otherwise.
pub fn launches(llm: &Json, root: &Path, nvidia: bool) -> Vec<Launch> {
    let cuda = config::program(root, str_or(llm, "server", "nrob-server"));
    let portable = config::program(root, str_or(llm, "server_webgpu", "nrob-server-webgpu"));
    match str_or(llm, "backend", "auto") {
        "cuda" => vec![Launch { program: cuda, backend: "cuda" }],
        "webgpu" => vec![Launch { program: portable, backend: "webgpu" }],
        "cpu" => vec![Launch { program: portable, backend: "cpu" }],
        _ => {
            let mut plan = Vec::new();
            if nvidia && cuda.is_file() {
                plan.push(Launch { program: cuda.clone(), backend: "auto" });
            }
            if portable.is_file() || plan.is_empty() {
                plan.push(Launch { program: portable, backend: "auto" });
            }
            // Neither built beside the studio: let the OS path find the CUDA build.
            if !plan.iter().any(|l| l.program.is_file()) && !cuda.is_file() {
                plan.insert(0, Launch { program: cuda, backend: "auto" });
            }
            plan
        }
    }
}

pub struct Llm {
    inner: Mutex<Inner>,
    changed: Condvar,
    pub log: Arc<LogRing>,
}

/// Where a running server can be reached.
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub addr: String,
    pub key: String,
}

fn free_port() -> Result<u16, String> {
    // Bind port 0, read the port the OS picked, release it for the child. The
    // window between is small, and a clash shows up as a failed start.
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    l.local_addr().map(|a| a.port()).map_err(|e| e.to_string())
}

/// The nrob-server command line for the `llm` section. `root` resolves relative paths.
pub fn arguments(llm: &Json, root: &Path, port: u16, key: &str, local_images: bool) -> Result<(Vec<String>, Vec<String>), String> {
    let models: Vec<&Json> = llm
        .get("models")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .iter()
        .filter(|m| bool_or(m, "enabled", true))
        .collect();
    if models.is_empty() {
        return Err("no language model configured: add a GGUF (or EXL3/DeepSeek folder) under Models".into());
    }
    let default = str_or(llm, "default_model", "");
    let primary = models.iter().position(|m| str_or(m, "name", "") == default).unwrap_or(0);
    let mut ordered = vec![models[primary]];
    ordered.extend(models.iter().enumerate().filter(|(i, _)| *i != primary).map(|(_, m)| *m));
    let path = |m: &Json, key: &str| config::resolve(root, str_or(m, key, "")).to_string_lossy().into_owned();
    let mut a: Vec<String> = Vec::new();
    let mut push = |k: &str, v: String| {
        a.push(k.into());
        a.push(v);
    };
    push("--model", path(ordered[0], "path"));
    push("--name", str_or(ordered[0], "name", "model").into());
    for m in &ordered[1..] {
        push("--also", format!("{}={}", str_or(m, "name", ""), path(m, "path")));
    }
    for m in &ordered {
        let name = str_or(m, "name", "");
        if !str_or(m, "vision_projector", "").trim().is_empty() {
            push("--vision-projector", format!("{name}={}", path(m, "vision_projector")));
        }
        if !str_or(m, "lora", "").trim().is_empty() {
            push("--lora", format!("{name}={}", path(m, "lora")));
            // How strongly it applies: 1 (as trained) unless set.
            let strength = m.get("lora_strength").and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())));
            if let Some(s) = strength.filter(|s| s.is_finite() && *s != 1.0) {
                push("--lora-strength", format!("{name}={s}"));
            }
        }
        // More LoRAs (`loras`: [{path, strength}] or paths), stacked with it, each at its own strength.
        for l in m.get("loras").and_then(Json::as_array).into_iter().flatten() {
            let (file, strength) = match l {
                Json::Str(p) => (Some(p.as_str()), None),
                _ => (l.get("path").and_then(Json::as_str), l.get("strength").and_then(Json::as_f64)),
            };
            let Some(file) = file.map(str::trim).filter(|p| !p.is_empty()) else { continue };
            let file = config::resolve(root, file).to_string_lossy().into_owned();
            push("--lora", match strength.filter(|s| s.is_finite() && *s != 1.0) {
                Some(s) => format!("{name}={file}@{s}"),
                None => format!("{name}={file}"),
            });
        }
    }
    push("--host", "127.0.0.1".into());
    push("--port", port.to_string());
    push("--api-key", key.into());
    push("--ctx", match int_or(llm, "ctx", 0) { 0 => "auto".into(), n => n.to_string() });
    push("--max-tokens", int_or(llm, "max_tokens", 8192).to_string());
    push("--temperature", num_or(llm, "temperature", 0.6).to_string());
    push("--top-p", num_or(llm, "top_p", 0.95).to_string());
    // Against a reply that loops on a phrase (prose only; not thinking or tool calls).
    let repeat = num_or(llm, "repeat_penalty", 1.0);
    if repeat > 1.0 {
        push("--repeat-penalty", repeat.clamp(1.0, 2.0).to_string());
    }
    // Unset: every core the machine has (nrob-server's 24 suits one machine only).
    let cores = std::thread::available_parallelism().map_or(8, |n| n.get()) as i64;
    push("--cpu-threads", llm.get("cpu_threads").and_then(Json::as_i64).unwrap_or(cores).to_string());
    push("--vram-headroom-gb", num_or(llm, "vram_headroom_gb", 2.0).to_string());
    // Off whatever the host: anything that can reach the gateway (a web page
    // included) could otherwise have the vision model read this machine's files.
    let _ = local_images;
    push("--local-images", "off".into());
    if int_or(llm, "ram_gb", 0) > 0 {
        push("--ram-gb", int_or(llm, "ram_gb", 0).to_string());
    }
    let devices: Vec<String> = llm.get("devices").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_i64).map(|d| d.to_string()).collect();
    if !devices.is_empty() {
        push("--devices", devices.join(","));
    }
    if bool_or(llm, "prompt_cache", true) {
        push("--prompt-cache", root.join("cache").join("prompt-states").to_string_lossy().into_owned());
        push("--prompt-cache-gb", num_or(llm, "prompt_cache_gb", 4.0).to_string());
    }
    a.push("--watch-stdin".into());
    if bool_or(llm, "thinking", false) {
        a.push("--thinking".into());
    }
    if !bool_or(llm, "vision", true) {
        a.push("--no-vision".into());
    }
    for extra in llm.get("extra_args").and_then(Json::as_array).unwrap_or(&[]) {
        if let Some(s) = extra.as_str() {
            a.push(s.into());
        }
    }
    let names = ordered.iter().map(|m| str_or(m, "name", "").to_string()).collect();
    Ok((a, names))
}

impl Llm {
    pub fn new() -> Llm {
        Llm {
            inner: Mutex::new(Inner {
                state: State::Stopped,
                error: None,
                child: None,
                lifeline: None,
                addr: String::new(),
                key: String::new(),
                generation: 0,
                started: None,
                ready_after: None,
                paused: false,
                last_used: Instant::now(),
                models: Vec::new(),
                context: None,
                command: String::new(),
                fallbacks: Vec::new(),
                relaunch: None,
            }),
            changed: Condvar::new(),
            log: Arc::new(LogRing::new(2000)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn state(&self) -> State {
        self.lock().state
    }

    pub fn is_running(&self) -> bool {
        matches!(self.lock().state, State::Starting | State::Ready)
    }

    pub fn touch(&self) {
        self.lock().last_used = Instant::now();
    }

    pub fn idle_for(&self) -> Duration {
        self.lock().last_used.elapsed()
    }

    pub fn paused(&self) -> bool {
        self.lock().paused
    }

    pub fn set_paused(&self, paused: bool) {
        self.lock().paused = paused;
    }

    pub fn endpoint(&self) -> Option<Endpoint> {
        let g = self.lock();
        (g.state == State::Ready).then(|| Endpoint { addr: g.addr.clone(), key: g.key.clone() })
    }

    pub fn models(&self) -> Vec<String> {
        self.lock().models.clone()
    }

    /// Start the server for `cfg` (the whole configuration) unless it runs.
    pub fn start(self: &Arc<Self>, cfg: &Json, root: &Path) -> Result<(), String> {
        let llm = cfg.get("llm").ok_or("no llm section")?;
        if !bool_or(llm, "enabled", true) {
            return Err("the LLM is disabled in the configuration".into());
        }
        if self.is_running() {
            return Ok(());
        }
        // Probed before taking the lock: nvidia-smi can take a second, and the
        // UI's status reads wait on this lock.
        let mut plan = launches(llm, root, nvidia_present());
        let mut g = self.lock();
        if matches!(g.state, State::Starting | State::Ready) {
            return Ok(());
        }
        let port = free_port()?;
        let key = random_id("sk-studio-");
        let gateway_host = cfg.get("gateway").map_or("127.0.0.1", |g| str_or(g, "host", "127.0.0.1"));
        let loopback = gateway_host == "localhost" || gateway_host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
        let (mut args, names) = arguments(llm, root, port, &key, loopback)?;
        if let Some(gb) = llm.get("webgpu_gb").and_then(Json::as_i64).filter(|n| *n > 0) {
            args.push("--webgpu-gb".into());
            args.push(gb.to_string());
        }
        if cfg.get("privacy").is_some_and(|p| bool_or(p, "incognito", false)) {
            // No prompt states on disk, no request log, each request forgotten.
            args.push("--incognito".into());
        }
        let mut errors = Vec::new();
        let mut child = loop {
            if plan.is_empty() {
                let msg = format!("could not start the LLM server: {} (build nrob-server or nrob-server-webgpu, or set llm.server)", errors.join("; "));
                g.state = State::Failed;
                g.error = Some(msg.clone());
                self.changed.notify_all();
                return Err(msg);
            }
            let launch = plan.remove(0);
            match self.spawn(&launch, &args, &key, root) {
                Ok((child, shown)) => {
                    g.command = shown;
                    break child;
                }
                Err(e) => errors.push(e),
            }
        };
        g.fallbacks = plan;
        g.relaunch = Some((args.clone(), key.clone(), root.to_path_buf()));
        g.generation += 1;
        let generation = g.generation;
        g.lifeline = child.stdin.take();
        g.child = Some(child);
        g.addr = format!("127.0.0.1:{port}");
        g.key = key;
        g.state = State::Starting;
        g.error = None;
        g.started = Some(Instant::now());
        g.ready_after = None;
        g.context = None;
        g.models = names;
        g.last_used = Instant::now();
        drop(g);
        self.changed.notify_all();
        let this = Arc::clone(self);
        std::thread::Builder::new()
            .name("llm-watch".into())
            .spawn(move || Arc::clone(&this).watch(generation))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Start one launch: the process with piped output (read into the log) and
    /// the stdin lifeline. Returns the child and the command line shown in logs.
    fn spawn(&self, launch: &Launch, args: &[String], key: &str, root: &Path) -> Result<(Child, String), String> {
        let mut command = Command::new(&launch.program);
        command.args(args).args(["--backend", launch.backend]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).current_dir(root);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let shown: Vec<String> = args.iter().map(|a| if a == key { "<key>".into() } else { a.clone() }).collect();
        let shown = format!("{} {} --backend {}", launch.program.display(), shown.join(" "), launch.backend);
        self.log.push(format!("studio: starting {shown}"));
        let mut child = command.spawn().map_err(|e| format!("{}: {e}", launch.program.display()))?;
        for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten() {
            let log = Arc::clone(&self.log);
            std::thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    log.push(line);
                }
            });
        }
        Ok((child, shown))
    }

    /// Wait for the process to answer `/health` (which it does once its model has
    /// loaded), then keep watching for an exit nobody asked for.
    fn watch(self: Arc<Self>, generation: u64) {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let (addr, ready) = {
                let mut g = self.lock();
                if g.generation != generation {
                    return;
                }
                let exited = g.child.as_mut().and_then(|c| c.try_wait().ok().flatten());
                if let Some(status) = exited {
                    g.child = None;
                    g.lifeline = None;
                    // Died while loading with another launch to try: e.g. the CUDA
                    // build on a machine whose driver it cannot load.
                    let mut relaunched = false;
                    while g.state == State::Starting && !g.fallbacks.is_empty() && !relaunched {
                        let next = g.fallbacks.remove(0);
                        self.log.push(format!("studio: nrob-server exited ({status}) while loading; trying {}", next.program.display()));
                        if let Some((args, key, root)) = g.relaunch.clone() {
                            if let Ok((mut child, shown)) = self.spawn(&next, &args, &key, &root) {
                                g.lifeline = child.stdin.take();
                                g.child = Some(child);
                                g.command = shown;
                                g.started = Some(Instant::now());
                                relaunched = true;
                            }
                        }
                    }
                    if relaunched {
                        continue;
                    }
                    g.state = State::Failed;
                    let tail = self.log.tail(12);
                    g.error = Some(format!("nrob-server exited ({status}):\n{tail}"));
                    self.log.push(format!("studio: nrob-server exited ({status})"));
                    self.changed.notify_all();
                    return;
                }
                (g.addr.clone(), g.state == State::Ready)
            };
            if ready {
                continue;
            }
            let ok = nrob::http::fetch(&addr, "GET", "/health", &[], b"", Duration::from_secs(2)).is_ok_and(|r| r.status == 200);
            if ok {
                let mut g = self.lock();
                if g.generation == generation && g.state == State::Starting {
                    g.state = State::Ready;
                    g.ready_after = g.started.map(|s| s.elapsed().as_secs_f64());
                    g.context = self.loaded_context();
                    self.log.push(format!("studio: nrob-server ready in {:.1}s", g.ready_after.unwrap_or(0.0)));
                    self.changed.notify_all();
                }
            }
        }
    }

    pub fn stop(&self) {
        let mut g = self.lock();
        g.generation += 1;
        g.lifeline = None;
        if let Some(mut child) = g.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            self.log.push("studio: nrob-server stopped");
        }
        g.state = State::Stopped;
        g.error = None;
        g.ready_after = None;
        self.changed.notify_all();
    }

    /// Start if needed and wait until it serves (or fails, or `timeout` passes).
    pub fn ensure_ready(self: &Arc<Self>, cfg: &Json, root: &Path, timeout: Duration) -> Result<Endpoint, String> {
        if matches!(self.state(), State::Stopped | State::Failed) {
            self.start(cfg, root)?;
        }
        let deadline = Instant::now() + timeout;
        let mut g = self.lock();
        loop {
            match g.state {
                State::Ready => return Ok(Endpoint { addr: g.addr.clone(), key: g.key.clone() }),
                State::Failed => return Err(g.error.clone().unwrap_or_else(|| "nrob-server failed".into())),
                State::Stopped => return Err("the language model was stopped".into()),
                State::Starting => {}
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("the language model is still loading; try again shortly".into());
            }
            g = self.changed.wait_timeout(g, left.min(Duration::from_millis(500))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// "… loaded in 8.1s (262144 token context)", as nrob-server reported it.
    fn loaded_context(&self) -> Option<i64> {
        let tail = self.log.tail(400);
        tail.lines().rev().find_map(|l| l.strip_suffix(" token context)")?.rsplit_once('(')?.1.parse().ok())
    }

    /// "WebGPU on …", "CUDA (2 card(s))" or "the CPU", as nrob-server reported it.
    fn runs_on(&self) -> Option<String> {
        let tail = self.log.tail(400);
        tail.lines().rev().find_map(|l| l.split_once(" runs on ").map(|(_, on)| on.trim().to_string()))
    }

    pub fn status(&self) -> Json {
        let g = self.lock();
        Json::obj([
            ("state", Json::str(g.state.name())),
            ("error", g.error.as_ref().map_or(Json::Null, Json::str)),
            ("models", Json::Arr(g.models.iter().map(Json::str).collect())),
            ("paused_for_media", Json::Bool(g.paused)),
            ("uptime_seconds", g.started.filter(|_| g.state != State::Stopped).map_or(Json::Null, |s| Json::Int(s.elapsed().as_secs() as i64))),
            ("load_seconds", g.ready_after.map_or(Json::Null, Json::Num)),
            ("idle_seconds", Json::Int(g.last_used.elapsed().as_secs() as i64)),
            ("command", Json::str(&g.command)),
            ("runs_on", self.runs_on().map_or(Json::Null, Json::str)),
            ("context_tokens", g.context.map_or(Json::Null, Json::Int)),
        ])
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_model_leads_and_extras_follow() {
        let llm = Json::parse(br#"{
            "default_model": "b", "ctx": 4096, "devices": [1], "thinking": true, "vision": false, "prompt_cache": false,
            "extra_args": ["--quiet"],
            "models": [
                {"name": "a", "path": "models/a.gguf"},
                {"name": "b", "path": "C:/abs/b.gguf", "vision_projector": "mmproj.gguf"},
                {"name": "off", "path": "x.gguf", "enabled": false}
            ]}"#).unwrap();
        let root = Path::new("/install");
        let (args, names) = arguments(&llm, root, 9000, "secret", true).unwrap();
        assert_eq!(names, ["b", "a"]);
        let joined = args.join(" ");
        assert!(joined.starts_with("--model C:/abs/b.gguf --name b --also a="), "{joined}");
        assert!(joined.contains(&format!("a={}", root.join("models/a.gguf").display())));
        assert!(joined.contains("--vision-projector b="));
        for flag in ["--port 9000", "--api-key secret", "--ctx 4096", "--devices 1", "--thinking", "--no-vision", "--local-images off", "--quiet"] {
            assert!(joined.contains(flag), "{flag} missing from {joined}");
        }
        assert!(!joined.contains("--prompt-cache"));
        let none = Json::parse(br#"{"models": []}"#).unwrap();
        assert!(arguments(&none, root, 1, "k", true).is_err());
        let auto = Json::parse(br#"{"models": [{"name": "a", "path": "a.gguf"}]}"#).unwrap();
        let joined = arguments(&auto, root, 1, "k", true).unwrap().0.join(" ");
        assert!(joined.contains("--ctx auto"), "{joined}");
        // LoRAs stack: the adapter folder, then each of `loras` at its own strength.
        let stacked = Json::parse(br#"{"models": [{"name": "q", "path": "C:/q", "lora": "C:/yes", "loras": [{"path": "C:/heresy", "strength": 0.5}, "C:/plain"]}]}"#).unwrap();
        let args = arguments(&stacked, root, 1, "k", true).unwrap().0;
        let loras: Vec<&str> = args.windows(2).filter(|w| w[0] == "--lora").map(|w| w[1].as_str()).collect();
        assert_eq!(loras, ["q=C:/yes", "q=C:/heresy@0.5", "q=C:/plain"]);
    }

    #[test]
    fn auto_prefers_cuda_with_an_nvidia_gpu_and_falls_back_to_webgpu() {
        let dir = std::env::temp_dir().join(format!("nrob-studio-launch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = |n: &str| dir.join(if cfg!(windows) { format!("{n}.exe") } else { n.to_string() });
        std::fs::write(exe("cuda-srv"), b"x").unwrap();
        std::fs::write(exe("gpu-srv"), b"x").unwrap();
        let llm = |backend: &str| Json::parse(format!(r#"{{"backend":"{backend}","server":"cuda-srv","server_webgpu":"gpu-srv"}}"#).as_bytes()).unwrap();
        let names = |plan: Vec<Launch>| plan.into_iter().map(|l| (l.program.file_stem().unwrap().to_string_lossy().into_owned(), l.backend)).collect::<Vec<_>>();
        assert_eq!(names(launches(&llm("auto"), &dir, true)), [("cuda-srv".into(), "auto"), ("gpu-srv".into(), "auto")]);
        assert_eq!(names(launches(&llm("auto"), &dir, false)), [("gpu-srv".into(), "auto")]);
        assert_eq!(names(launches(&llm("cpu"), &dir, true)), [("gpu-srv".into(), "cpu")]);
        assert_eq!(names(launches(&llm("cuda"), &dir, false)), [("cuda-srv".into(), "cuda")]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_server_fails_clearly_instead_of_hanging() {
        let llm = Arc::new(Llm::new());
        let cfg = Json::parse(br#"{"gateway":{"host":"127.0.0.1"},"llm":{"backend":"cuda","server":"definitely-missing/nrob-server-x","models":[{"name":"m","path":"m.gguf"}]}}"#).unwrap();
        let err = llm.ensure_ready(&cfg, &std::env::temp_dir(), Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("could not start the LLM server"), "{err}");
        assert_eq!(llm.state(), State::Failed);
    }
}
