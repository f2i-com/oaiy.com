//! Supervision of the language model: `oaiy-llm-server` as a child process on a
//! private loopback port with a random key. The gateway proxies to it; nothing
//! else can reach it.
//!
//! A separate process rather than a linked library: oaiy-llm-server holds GPU
//! memory that only process exit reliably returns, so stopping it (for a media
//! job that needs its GPU, or after idling) is a kill, and a crash takes down
//! the model, not the studio.

use crate::config;
use crate::util::{bool_or, int_or, num_or, random_id, str_or, LogRing};
use oaiy_engine::json::Json;
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
    /// the pipe closes and oaiy-llm-server stops, returning its GPU memory.
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
    /// The model it holds now: the default when it starts, then the last one asked for.
    resident: Option<String>,
    /// The context the loaded model was opened with, as oaiy-llm-server reported it.
    context: Option<i64>,
    command: String,
}

/// One way to start the server: the program and its `--backend`.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub backend: &'static str,
}

/// What `llm.backend` asks of the server: `webgpu`, `cpu`, else `auto` (a GPU through WebGPU when there is one,
/// else the CPU). `cuda`, which a configuration from the CUDA build may still say, is `auto`: the GPU it meant.
pub fn backend(llm: &Json) -> &'static str {
    match str_or(llm, "backend", "auto") {
        "webgpu" => "webgpu",
        "cpu" => "cpu",
        _ => "auto",
    }
}

/// The server programs to try, in order: `llm.server` (`oaiy-llm-server`), then `llm.server_webgpu`
/// (`oaiy-llm-server-webgpu`: the same server, under the name it had while a CUDA build stood beside it, which older
/// configurations and installs still name). The one that is a file comes first; the other is tried only when the
/// first cannot be started.
pub fn launches(llm: &Json, root: &Path) -> Vec<Launch> {
    let backend = backend(llm);
    let server = config::program(root, str_or(llm, "server", "oaiy-llm-server"));
    let other = config::program(root, str_or(llm, "server_webgpu", "oaiy-llm-server-webgpu"));
    let mut programs = vec![server];
    if other != programs[0] {
        if other.is_file() && !programs[0].is_file() {
            programs.insert(0, other);
        } else {
            programs.push(other);
        }
    }
    programs.into_iter().map(|program| Launch { program, backend }).collect()
}

/// The weights the engine may put on the GPU when the configuration does not say (`llm.webgpu_gb`): the
/// largest GPU's memory, less `llm.vram_headroom_gb` and 2 GB for the cache and the work buffers. WebGPU cannot report
/// free memory, so the engine otherwise assumes 8 GiB of any discrete card, and a 27B model ran mostly on the CPU of a
/// 32 GB card. None (the engine's own default) when no GPU says, or the result would be under 4 GB (an integrated GPU,
/// whose memory is the computer's).
pub fn auto_webgpu_gb(gpus: &Json, llm: &Json) -> Option<i64> {
    let largest_mb = gpus.as_array()?.iter().filter_map(|g| g.get("memory_total_mb").and_then(Json::as_i64)).max()?;
    let headroom = int_or(llm, "vram_headroom_gb", 2).max(0);
    let gb = largest_mb / 1024 - headroom - 2;
    (gb >= 4).then_some(gb)
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

/// The oaiy-llm-server command line for the `llm` section. `root` resolves relative paths.
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
        // Drafting with its multi-token-prediction layer (a Qwen3.5 GGUF with one), where ticked.
        if bool_or(m, "mtp", false) {
            push("--mtp", name.into());
        }
        // Its own GPUs (a model too big for one), instead of the LLM's `devices`.
        let own: Vec<String> = m.get("devices").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_i64).map(|d| d.to_string()).collect();
        if !own.is_empty() {
            push("--devices-for", format!("{name}={}", own.join(",")));
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
    // Unset: every core the machine has (oaiy-llm-server's 24 suits one machine only).
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
        // DeepSeek-V4.1's experts' counts of uses, kept with what else the server keeps between runs: its next start
        // reads the most used first, onto the GPUs and then into RAM (no other model reads it).
        push("--usage", root.join("cache").join("expert-usage.bin").to_string_lossy().into_owned());
    }
    // Host RAM where Qwen3.8-Flash-Next sets aside the conversation another displaces (it keeps one
    // on the GPUs and nothing on disk); 0 = off. Passed only when `park_gb` is set: otherwise the
    // server's own default (8 GB) applies, and a server from before the option, which refuses the
    // flag, still starts.
    if let Some(gb) = llm.get("park_gb").and_then(Json::as_f64).filter(|gb| gb.is_finite()) {
        push("--park-gb", gb.max(0.0).to_string());
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
                resident: None,
                context: None,
                command: String::new(),
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
        let mut plan = launches(llm, root);
        let mut g = self.lock();
        if matches!(g.state, State::Starting | State::Ready) {
            return Ok(());
        }
        let port = free_port()?;
        let key = random_id("sk-studio-");
        let gateway_host = cfg.get("gateway").map_or("127.0.0.1", |g| str_or(g, "host", "127.0.0.1"));
        let loopback = gateway_host == "localhost" || gateway_host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
        let (mut args, names) = arguments(llm, root, port, &key, loopback)?;
        let webgpu_gb = llm.get("webgpu_gb").and_then(Json::as_i64).filter(|n| *n > 0).or_else(|| auto_webgpu_gb(&crate::system::System::new().gpus(), llm));
        if let Some(gb) = webgpu_gb {
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
                let msg = format!("could not start the LLM server: {} (build oaiy-llm-server or oaiy-llm-server-webgpu, or set llm.server)", errors.join("; "));
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
        // oaiy-llm-server loads its default model as it starts.
        g.resident = names.first().cloned();
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
                    g.state = State::Failed;
                    let tail = self.log.tail(12);
                    g.error = Some(format!("oaiy-llm-server exited ({status}):\n{tail}"));
                    self.log.push(format!("studio: oaiy-llm-server exited ({status})"));
                    self.changed.notify_all();
                    return;
                }
                (g.addr.clone(), g.state == State::Ready)
            };
            if ready {
                continue;
            }
            let ok = oaiy_engine::http::fetch(&addr, "GET", "/health", &[], b"", Duration::from_secs(2)).is_ok_and(|r| r.status == 200);
            if ok {
                let mut g = self.lock();
                if g.generation == generation && g.state == State::Starting {
                    g.state = State::Ready;
                    g.ready_after = g.started.map(|s| s.elapsed().as_secs_f64());
                    g.context = self.loaded_context();
                    self.log.push(format!("studio: oaiy-llm-server ready in {:.1}s", g.ready_after.unwrap_or(0.0)));
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
            self.log.push("studio: oaiy-llm-server stopped");
        }
        g.state = State::Stopped;
        g.error = None;
        g.ready_after = None;
        g.resident = None;
        self.changed.notify_all();
    }

    /// The model the running server holds (None when it is not running).
    pub fn resident(&self) -> Option<String> {
        let g = self.lock();
        matches!(g.state, State::Starting | State::Ready).then(|| g.resident.clone()).flatten()
    }

    /// A request is about to switch the server to `name`.
    pub fn set_resident(&self, name: String) {
        self.lock().resident = Some(name);
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
                State::Failed => return Err(g.error.clone().unwrap_or_else(|| "oaiy-llm-server failed".into())),
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

    /// "… loaded in 8.1s (262144 token context)", as oaiy-llm-server reported it.
    fn loaded_context(&self) -> Option<i64> {
        let tail = self.log.tail(400);
        tail.lines().rev().find_map(|l| l.strip_suffix(" token context)")?.rsplit_once('(')?.1.parse().ok())
    }

    /// "WebGPU on …" or "the CPU", as oaiy-llm-server reported it.
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
            ("resident", g.resident.as_ref().filter(|_| matches!(g.state, State::Starting | State::Ready)).map_or(Json::Null, Json::str)),
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
    fn the_portable_engines_gpu_budget_follows_the_largest_gpu_unless_configured() {
        let gpus = |mbs: &[i64]| Json::Arr(mbs.iter().map(|m| Json::obj([("memory_total_mb", Json::Int(*m))])).collect());
        let llm = |headroom: Option<i64>| match headroom {
            Some(h) => Json::obj([("vram_headroom_gb", Json::Int(h))]),
            None => Json::Obj(Vec::new()),
        };
        // A 32 GB RTX 5090 beside a 2 GB Radeon: the 5090's 31 GiB, less 2 of headroom and 2 for the cache.
        assert_eq!(auto_webgpu_gb(&gpus(&[2048, 32607]), &llm(None)), Some(27));
        assert_eq!(auto_webgpu_gb(&gpus(&[16288]), &llm(Some(1))), Some(12));
        assert_eq!(auto_webgpu_gb(&gpus(&[8192]), &llm(None)), Some(4));
        // An integrated GPU, or none that says: the engine's own default.
        assert_eq!(auto_webgpu_gb(&gpus(&[2048]), &llm(None)), None);
        assert_eq!(auto_webgpu_gb(&gpus(&[]), &llm(None)), None);
        assert_eq!(auto_webgpu_gb(&Json::Arr(vec![Json::obj([("memory_total_mb", Json::Null)])]), &llm(None)), None);
    }

    #[test]
    fn a_model_ticked_to_draft_is_named_for_mtp() {
        let llm = Json::parse(br#"{"models": [{"name": "q", "path": "q.gguf", "mtp": true}, {"name": "p", "path": "p.gguf"}, {"name": "r", "path": "r.gguf", "mtp": false}]}"#).unwrap();
        let args = arguments(&llm, Path::new("/install"), 1, "k", true).unwrap().0;
        let mtp: Vec<&str> = args.windows(2).filter(|w| w[0] == "--mtp").map(|w| w[1].as_str()).collect();
        assert_eq!(mtp, ["q"]);
    }

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
        assert!(!joined.contains("--usage"), "nothing kept between runs where the prompt states are not");
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
    fn park_gb_is_passed_only_when_it_is_configured() {
        let root = Path::new("/install");
        let park = |llm: &Json| {
            let args = arguments(llm, root, 1, "k", true).unwrap().0;
            args.windows(2).find(|w| w[0] == "--park-gb").map(|w| w[1].clone())
        };
        let parse = |llm: &str| Json::parse(llm.as_bytes()).unwrap();
        let model = r#""models": [{"name": "a", "path": "a.gguf"}]"#;
        // Not configured: no flag, so the server's own default of 8 GB applies and a server from before the
        // option (which refuses the flag) still starts. A section without the key, a null, and a
        // configuration file written before the option, once the defaults have filled it in.
        assert_eq!(park(&parse(&format!("{{{model}}}"))), None);
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": null, {model}}}"#))), None);
        let mut old = parse(&format!("{{\"llm\": {{{model}}}}}"));
        config::merge_defaults(&mut old, &config::default_json());
        assert_eq!(old.get("llm").and_then(|l| l.get("park_gb")), Some(&Json::Null));
        assert_eq!(park(old.get("llm").unwrap()), None);
        // Configured: it is passed, 0 (off) included, whatever the disk cache does.
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": 2.5, {model}}}"#))).as_deref(), Some("2.5"));
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": 8, {model}}}"#))).as_deref(), Some("8"));
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": 0, {model}}}"#))).as_deref(), Some("0"));
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": 4, "prompt_cache": false, {model}}}"#))).as_deref(), Some("4"));
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": -3, {model}}}"#))).as_deref(), Some("0"), "never negative");
        // A value that is not a number leaves it out (the configuration check refuses it before it gets here).
        assert_eq!(park(&parse(&format!(r#"{{"park_gb": "8", {model}}}"#))), None);
    }

    #[test]
    fn the_server_is_the_one_program_that_is_there_under_either_name() {
        let dir = std::env::temp_dir().join(format!("oaiy-studio-launch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let exe = |n: &str| dir.join(if cfg!(windows) { format!("{n}.exe") } else { n.to_string() });
        let llm = |backend: &str| Json::parse(format!(r#"{{"backend":"{backend}","server":"srv","server_webgpu":"srv-webgpu"}}"#).as_bytes()).unwrap();
        let names = |plan: Vec<Launch>| plan.into_iter().map(|l| (l.program.file_stem().unwrap().to_string_lossy().into_owned(), l.backend)).collect::<Vec<_>>();
        // Neither is a file: both by name, llm.server first (the OS search path may find one).
        assert_eq!(names(launches(&llm("auto"), &dir)), [("srv".into(), "auto"), ("srv-webgpu".into(), "auto")]);
        // Only the name the WebGPU build had is there (an install that staged that one): it is the server.
        std::fs::write(exe("srv-webgpu"), b"x").unwrap();
        assert_eq!(names(launches(&llm("auto"), &dir)), [("srv-webgpu".into(), "auto"), ("srv".into(), "auto")]);
        // Both there: llm.server, whatever the backend, and no NVIDIA card is asked about.
        std::fs::write(exe("srv"), b"x").unwrap();
        assert_eq!(names(launches(&llm("auto"), &dir)), [("srv".into(), "auto"), ("srv-webgpu".into(), "auto")]);
        assert_eq!(names(launches(&llm("webgpu"), &dir))[0], ("srv".into(), "webgpu"));
        assert_eq!(names(launches(&llm("cpu"), &dir))[0], ("srv".into(), "cpu"));
        // A configuration from the CUDA build: the GPU it meant, which is WebGPU's now.
        assert_eq!(names(launches(&llm("cuda"), &dir))[0], ("srv".into(), "auto"));
        // One program named twice is tried once.
        let same = Json::parse(br#"{"server":"srv","server_webgpu":"srv"}"#).unwrap();
        assert_eq!(names(launches(&same, &dir)), [("srv".into(), "auto")]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_server_fails_clearly_instead_of_hanging() {
        let llm = Arc::new(Llm::new());
        let cfg = Json::parse(br#"{"gateway":{"host":"127.0.0.1"},"llm":{"backend":"auto","server":"definitely-missing/oaiy-llm-server-x","server_webgpu":"definitely-missing/oaiy-llm-server-y","models":[{"name":"m","path":"m.gguf"}]}}"#).unwrap();
        let err = llm.ensure_ready(&cfg, &std::env::temp_dir(), Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("could not start the LLM server"), "{err}");
        assert_eq!(llm.state(), State::Failed);
    }
}
