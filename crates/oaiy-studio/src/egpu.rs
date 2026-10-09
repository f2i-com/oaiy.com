//! A graphics card OAIY's own engine cannot reach, through tinygrad's LLM server. For macOS only.
//!
//! macOS has no graphics driver for a card in a Thunderbolt enclosure, so WebGPU never sees one. tinygrad reaches it
//! with a driver of its own (its TinyGPU app), and serves a GGUF model on it behind an OpenAI-style API. This module
//! runs that server for the models set to it (`llm.models[].egpu`), one model at a time: `python egpu_serve.py
//! --model FILE --serve PORT` (that file holds tinygrad's server to this computer and to a key) on a private
//! loopback port, started when a request first names such a model.
//!
//! The gateway sends those models' chat requests there while the server answers. When it cannot be started (the
//! card unplugged, tinygrad not installed) or has exited, this computer's own engine answers instead: every model is
//! in `oaiy-llm-server`'s list too. A server that is still loading is waited for, never given up on: tinygrad's
//! first start compiles its kernels, which takes minutes.
//!
//! Only on a Mac ([`available`]): everywhere else nothing is offered and nothing is started, whatever the
//! configuration says.

use crate::config;
use crate::llm::{Endpoint, State};
use crate::util::{bool_or, int_or, random_id, str_or, LogRing};
use oaiy_engine::json::Json;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The launcher, written beside the configuration before each start. Carried by a Mac's build only.
#[cfg(any(target_os = "macos", test))]
const SERVE_PY: &str = include_str!("egpu_serve.py");
#[cfg(not(any(target_os = "macos", test)))]
const SERVE_PY: &str = "";
/// After a start that failed, how long requests go to this computer's engine before it is tried again.
const RETRY_AFTER: Duration = Duration::from_secs(60);
/// tinygrad's own default is 4,096 tokens, less than an agent's prompt with its tools.
const DEFAULT_CTX: i64 = 8192;
/// How long a Python has to say what it has installed.
const CHECK_WAIT: Duration = Duration::from_secs(30);

/// Whether this build offers it at all: a Mac's (and the tests', which run its logic on any computer).
pub fn available() -> bool {
    cfg!(any(target_os = "macos", test))
}

fn section(llm: &Json) -> &Json {
    llm.get("egpu").unwrap_or(&Json::Null)
}

/// Switched on (`llm.egpu.enabled`), on a computer that offers it.
pub fn enabled(llm: &Json) -> bool {
    available() && bool_or(section(llm), "enabled", false)
}

fn model<'a>(llm: &'a Json, name: &str) -> Option<&'a Json> {
    llm.get("models").and_then(Json::as_array)?.iter().find(|m| str_or(m, "name", "") == name && bool_or(m, "enabled", true))
}

/// Whether `name` is an enabled model set to run on the eGPU, and the eGPU is switched on.
pub fn assigned(llm: &Json, name: &str) -> bool {
    enabled(llm) && model(llm, name).is_some_and(|m| bool_or(m, "egpu", false))
}

/// The enabled models set to it, in the configuration's order.
pub fn models(llm: &Json) -> Vec<String> {
    if !enabled(llm) {
        return Vec::new();
    }
    let all = llm.get("models").and_then(Json::as_array).unwrap_or(&[]);
    all.iter().filter(|m| bool_or(m, "enabled", true) && bool_or(m, "egpu", false)).map(|m| str_or(m, "name", "").to_string()).collect()
}

/// The model that answers on this computer when the eGPU does not: `llm.egpu.fallback_model` where it names another
/// enabled model that is not itself on the eGPU, else `name` (the same model, on this computer's engine).
pub fn fallback(llm: &Json, name: &str) -> String {
    let other = str_or(section(llm), "fallback_model", "").trim();
    if !other.is_empty() && other != name && model(llm, other).is_some() && !assigned(llm, other) {
        return other.to_string();
    }
    name.to_string()
}

/// The context tinygrad's server is started with (`llm.egpu.ctx`; it sets its cache aside for all of it at once).
pub fn context(llm: &Json) -> i64 {
    match int_or(section(llm), "ctx", DEFAULT_CTX) {
        n if n >= 512 => n,
        _ => DEFAULT_CTX,
    }
}

/// The Pythons to ask, in order: the one `llm.egpu.python` names and no other; else a virtual environment inside the
/// tinygrad folder (`llm.egpu.tinygrad`), Homebrew's, the system's, and whatever `python3` the PATH finds. An app
/// opened from the Finder has a PATH without Homebrew's folders, so those are named outright.
pub fn pythons(llm: &Json, root: &Path) -> Vec<PathBuf> {
    let egpu = section(llm);
    let named = str_or(egpu, "python", "").trim();
    if !named.is_empty() {
        return vec![typed(root, named, std::env::var_os("HOME").as_deref())];
    }
    let mut found = Vec::new();
    if let Some(folder) = tinygrad_folder(llm, root) {
        for venv in [".venv", "venv"] {
            found.push(folder.join(venv).join("bin").join("python3"));
            found.push(folder.join(venv).join("Scripts").join("python.exe"));
        }
    }
    found.extend(["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"].map(PathBuf::from));
    found.retain(|p| p.is_file());
    found.push(PathBuf::from(if cfg!(windows) { "python" } else { "python3" }));
    found
}

/// A checkout of tinygrad that is not installed into the Python (`llm.egpu.tinygrad`): its folder goes on the
/// Python's path, as tinygrad's own instructions run it (`PYTHONPATH=.`).
fn tinygrad_folder(llm: &Json, root: &Path) -> Option<PathBuf> {
    let folder = str_or(section(llm), "tinygrad", "").trim();
    (!folder.is_empty()).then(|| typed(root, folder, std::env::var_os("HOME").as_deref()))
}

/// A path as it is typed into a settings field: `~/tinygrad` is in the home folder, as a shell would have it
/// (a field is no shell, and tinygrad's own instructions write such paths); anything else as every path here.
fn typed(root: &Path, path: &str, home: Option<&std::ffi::OsStr>) -> PathBuf {
    match (path.strip_prefix("~/"), home.filter(|h| !h.is_empty())) {
        (Some(rest), Some(home)) => Path::new(home).join(rest),
        _ => config::resolve(root, path),
    }
}

/// The folders a Mac's programs are found in that an app opened from the Finder does not have on its PATH:
/// `~/.local/bin` (where tinygrad's setup puts NVIDIA's compiler), Homebrew's, and Docker's (that compiler runs in
/// it), before whatever the PATH already holds.
pub fn search_path(home: Option<&str>, current: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(home) = home.filter(|h| !h.is_empty()) {
        parts.push(format!("{home}/.local/bin"));
    }
    parts.extend(["/opt/homebrew/bin", "/usr/local/bin", "/Applications/Docker.app/Contents/Resources/bin"].map(String::from));
    for p in current.split(':').filter(|p| !p.is_empty()) {
        if !parts.iter().any(|have| have == p) {
            parts.push(p.to_string());
        }
    }
    parts.join(":")
}

/// What a start adds to the server's environment: the device tinygrad uses (`llm.egpu.device`, `NV` unless set: an
/// NVIDIA card; `AMD` for a Radeon), the tinygrad folder on Python's path, and whatever `llm.egpu.env` names
/// (`JITBEAM` and the like).
fn environment(llm: &Json, root: &Path) -> Vec<(String, String)> {
    let egpu = section(llm);
    let device = str_or(egpu, "device", "NV").trim();
    let mut env = vec![("DEV".to_string(), if device.is_empty() { "NV".to_string() } else { device.to_string() }), ("PYTHONUNBUFFERED".into(), "1".into())];
    if let Some(folder) = tinygrad_folder(llm, root) {
        let folder = folder.to_string_lossy().into_owned();
        let sep = if cfg!(windows) { ";" } else { ":" };
        env.push(("PYTHONPATH".into(), match std::env::var("PYTHONPATH") {
            Ok(have) if !have.is_empty() => format!("{folder}{sep}{have}"),
            _ => folder,
        }));
    }
    if cfg!(target_os = "macos") {
        env.push(("PATH".into(), search_path(std::env::var("HOME").ok().as_deref(), &std::env::var("PATH").unwrap_or_default())));
    }
    for (k, v) in egpu.get("env").map(|e| e.members().collect::<Vec<_>>()).unwrap_or_default() {
        let value = match v {
            Json::Str(s) => s.clone(),
            Json::Int(n) => n.to_string(),
            _ => continue,
        };
        env.retain(|(have, _)| have != k);
        env.push((k.to_string(), value));
    }
    env
}

/// tinygrad's server's command line for `name`: the launcher, the model's file, the port and the context, then
/// `llm.egpu.extra_args`. A model that is not one GGUF file is refused: tinygrad's server reads nothing else.
pub fn arguments(llm: &Json, root: &Path, name: &str, script: &Path, port: u16) -> Result<Vec<String>, String> {
    let m = model(llm, name).ok_or_else(|| format!("{name} is not one of the enabled language models"))?;
    let path = config::resolve(root, str_or(m, "path", ""));
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        return Err(format!("{name} is not a GGUF file, and tinygrad's server reads nothing else"));
    }
    let mut a = vec!["-u".to_string(), script.to_string_lossy().into_owned(), "--watch-stdin".into()];
    a.extend(["--model".to_string(), path.to_string_lossy().into_owned()]);
    a.extend(["--serve".to_string(), port.to_string()]);
    a.extend(["--max_context".to_string(), context(llm).to_string()]);
    for extra in section(llm).get("extra_args").and_then(Json::as_array).unwrap_or(&[]) {
        if let Some(s) = extra.as_str() {
            a.push(s.into());
        }
    }
    Ok(a)
}

/// A chat request's body as tinygrad's server is sent it: `model` as OAIY names it, and what tinygrad would
/// otherwise decide differently from OAIY's own engine filled in from the language model's settings: its
/// temperature (tinygrad's default is 0) and a reply limit (it has none). None: not a JSON object.
pub fn request_body(llm: &Json, name: &str, body: &[u8]) -> Option<Vec<u8>> {
    let mut v = Json::parse(body).ok().filter(|v| v.as_object().is_some())?;
    crate::util::set(&mut v, "model", Json::str(name));
    if v.get("temperature").and_then(Json::as_f64).is_none() {
        crate::util::set(&mut v, "temperature", Json::Num(crate::util::num_or(llm, "temperature", 0.6)));
    }
    let limited = ["max_tokens", "max_completion_tokens"].iter().any(|k| v.get(k).and_then(Json::as_i64).is_some_and(|n| n > 0));
    if !limited {
        crate::util::set(&mut v, "max_tokens", Json::Int(int_or(llm, "max_tokens", 8192).max(1)));
    }
    Some(v.to_json().into_bytes())
}

/// Whether a request is to be thought about first, as OAIY's own engine decides it: `reasoning_effort` (`none` and
/// `minimal` say no, anything else yes), else `thinking: {type: enabled | disabled}`, else the language model's
/// *Think by default*. tinygrad's server reads none of these, and a Qwen model's chat format thinks unless told
/// not to: the launcher is told with the request (`X-OAIY-Thinking`), and tells the model's chat format.
pub fn thinks(llm: &Json, body: &Json) -> bool {
    if let Some(effort) = body.get("reasoning_effort").filter(|v| !matches!(v, Json::Null)) {
        return !matches!(effort.as_str(), Some("none" | "minimal"));
    }
    match body.get("thinking").and_then(|t| t.get("type")).and_then(Json::as_str) {
        Some("enabled") => true,
        Some("disabled") => false,
        _ => bool_or(llm, "thinking", false),
    }
}

/// Whether a chat request carries a picture: tinygrad's server reads text only, so such a request is this
/// computer's engine's.
pub fn has_picture(body: &Json) -> bool {
    let parts = |m: &Json| m.get("content").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default();
    body.get("messages").and_then(Json::as_array).unwrap_or(&[]).iter().flat_map(parts).any(|p| str_or(&p, "type", "text") != "text")
}

/// Why a model set to the eGPU is not answered there now.
#[derive(Debug, PartialEq)]
pub enum Unready {
    /// It cannot serve (why): this computer's engine answers instead.
    Gone(String),
    /// It is still loading after the wait, or answering with another model: the client tries again.
    Loading(String),
}

struct Inner {
    state: State,
    error: Option<String>,
    child: Option<Child>,
    /// Held open for the launcher's `--watch-stdin`: when the studio exits, however it exits, the pipe closes and
    /// tinygrad's server stops, giving the card back.
    lifeline: Option<std::process::ChildStdin>,
    addr: String,
    key: String,
    /// Bumped on every start and stop, so the threads of an older process stand down.
    generation: u64,
    started: Option<Instant>,
    ready_after: Option<f64>,
    /// The model it holds, or is loading, or failed to load.
    model: Option<String>,
    failed_at: Option<Instant>,
    /// Requests it is answering now: a model is not swapped out from under them.
    busy: usize,
    last_used: Instant,
    command: String,
    /// The Python that has tinygrad, for the configuration it was found under.
    python: Option<(String, PathBuf)>,
    /// The last reason a request went to this computer's engine, so the log says it once.
    said: Option<String>,
}

pub struct Egpu {
    inner: Mutex<Inner>,
    changed: Condvar,
    /// One start at a time: finding the Python runs processes, outside the state's lock.
    launching: Mutex<()>,
    pub log: Arc<LogRing>,
}

/// A request being answered by tinygrad's server: while one is held, the model is not swapped.
pub struct Lease(Arc<Egpu>);

impl Drop for Lease {
    fn drop(&mut self) {
        let mut g = self.0.lock();
        g.busy = g.busy.saturating_sub(1);
        g.last_used = Instant::now();
        drop(g);
        self.0.changed.notify_all();
    }
}

/// What a Python said of itself (`egpu_serve.py --check`), or why it could not be asked.
fn probe(python: &Path, script: &Path, env: &[(String, String)], cwd: Option<&Path>) -> Result<Json, String> {
    let mut command = Command::new(python);
    command.arg(script).arg("--check").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    command.envs(env.iter().map(|(k, v)| (k, v)));
    if let Some(dir) = cwd.filter(|d| d.is_dir()) {
        command.current_dir(dir);
    }
    hide_window(&mut command);
    let mut child = command.spawn().map_err(|e| format!("{}: {e}", python.display()))?;
    let deadline = Instant::now() + CHECK_WAIT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{}: no answer within {} s", python.display(), CHECK_WAIT.as_secs()));
            }
        }
    }
    let mut out = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut out);
    }
    let line = out.lines().rev().find(|l| l.trim_start().starts_with('{')).unwrap_or("");
    Json::parse(line.as_bytes()).map_err(|_| format!("{}: it did not answer as Python 3 does", python.display()))
}

fn hide_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    let _ = command;
}

fn free_port() -> Result<u16, String> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    l.local_addr().map(|a| a.port()).map_err(|e| e.to_string())
}

impl Egpu {
    pub fn new() -> Egpu {
        Egpu {
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
                model: None,
                failed_at: None,
                busy: 0,
                last_used: Instant::now(),
                command: String::new(),
                python: None,
                said: None,
            }),
            changed: Condvar::new(),
            launching: Mutex::new(()),
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

    /// A server that is already there (a test's stand-in), taken as tinygrad's, ready with `model`; `process` is
    /// what stands for its process (None: one that has gone).
    #[cfg(test)]
    pub(crate) fn adopt(&self, addr: &str, key: &str, model: &str, process: Option<Child>) {
        let mut g = self.lock();
        g.addr = addr.into();
        g.key = key.into();
        g.model = Some(model.into());
        g.child = process;
        g.state = State::Ready;
    }

    /// Let go of the server without stopping it, as a studio that dies does: its standard input closes. Whether it
    /// then went by itself within `wait` (it is killed if not).
    #[cfg(test)]
    pub(crate) fn let_go(&self, wait: Duration) -> bool {
        let child = {
            let mut g = self.lock();
            g.generation += 1;
            g.lifeline = None;
            g.state = State::Stopped;
            g.child.take()
        };
        let Some(mut child) = child else { return false };
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            if child.try_wait().ok().flatten().is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        false
    }

    /// The model it holds, is loading, or failed to load.
    pub fn held(&self) -> Option<String> {
        self.lock().model.clone()
    }

    /// How long since it last answered (None while it is answering).
    pub fn idle_for(&self) -> Option<Duration> {
        let g = self.lock();
        (g.busy == 0).then(|| g.last_used.elapsed())
    }

    /// The launcher's file, written again at each use (a small file; an older OAIY's copy must not linger).
    fn script(root: &Path) -> Result<PathBuf, String> {
        let dir = root.join("cache").join("egpu");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let file = dir.join("egpu_serve.py");
        std::fs::write(&file, SERVE_PY).map_err(|e| format!("{}: {e}", file.display()))?;
        Ok(file)
    }

    /// Ask each Python in turn what it has: `{ok, python, version, tinygrad, commit, server, jinja2, tried}`, `ok`
    /// when one has tinygrad's LLM server (the first that does is the one used).
    pub fn check(&self, cfg: &Json, root: &Path) -> Json {
        let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
        if !available() {
            return Json::obj([("ok", Json::Bool(false)), ("error", Json::str("tinygrad's server is used on a Mac only"))]);
        }
        let script = match Self::script(root) {
            Ok(s) => s,
            Err(e) => return Json::obj([("ok", Json::Bool(false)), ("error", Json::str(e))]),
        };
        let env = environment(&llm, root);
        let cwd = tinygrad_folder(&llm, root);
        let mut tried = Vec::new();
        for python in pythons(&llm, root) {
            let shown = python.to_string_lossy().into_owned();
            match probe(&python, &script, &env, cwd.as_deref()) {
                Ok(mut said) if said.get("server").and_then(Json::as_str).is_some() => {
                    crate::util::set(&mut said, "ok", Json::Bool(true));
                    crate::util::set(&mut said, "tried", Json::Arr(tried));
                    self.lock().python = Some((python_key(&llm), python));
                    return said;
                }
                Ok(said) => tried.push(Json::obj([("python", Json::str(shown)), ("error", Json::str(str_or(&said, "error", "it has no tinygrad")))])),
                Err(e) => tried.push(Json::obj([("python", Json::str(shown)), ("error", Json::str(e))])),
            }
        }
        let why = tried.iter().map(|t| format!("{}: {}", str_or(t, "python", ""), str_or(t, "error", ""))).collect::<Vec<_>>().join("; ");
        Json::obj([("ok", Json::Bool(false)), ("error", Json::str(format!("no Python here has tinygrad's LLM server ({why})"))), ("tried", Json::Arr(tried))])
    }

    /// The Python to start: the one a check found under this configuration, else found now.
    fn python(&self, cfg: &Json, root: &Path) -> Result<PathBuf, String> {
        let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
        if let Some((key, python)) = self.lock().python.clone() {
            if key == python_key(&llm) {
                return Ok(python);
            }
        }
        let said = self.check(cfg, root);
        match self.lock().python.clone() {
            Some((key, python)) if key == python_key(&llm) && bool_or(&said, "ok", false) => Ok(python),
            _ => Err(str_or(&said, "error", "no Python with tinygrad was found").to_string()),
        }
    }

    /// Start tinygrad's server with `name`, replacing whatever it held.
    pub fn start(self: &Arc<Self>, cfg: &Json, root: &Path, name: &str) -> Result<(), String> {
        let llm = cfg.get("llm").ok_or("no llm section")?;
        if !enabled(llm) {
            return Err(if available() { "the eGPU is switched off (Settings)".into() } else { "tinygrad's server is used on a Mac only".into() });
        }
        if !assigned(llm, name) {
            return Err(format!("{name} is not set to run on the eGPU"));
        }
        let _one = self.launching.lock().unwrap_or_else(|p| p.into_inner());
        {
            let g = self.lock();
            if matches!(g.state, State::Starting | State::Ready) && g.model.as_deref() == Some(name) {
                return Ok(());
            }
        }
        self.stop();
        let started = self.launch(cfg, llm, root, name);
        if let Err(e) = &started {
            let mut g = self.lock();
            g.state = State::Failed;
            g.error = Some(e.clone());
            g.model = Some(name.to_string());
            g.failed_at = Some(Instant::now());
            drop(g);
            self.log.push(format!("studio: tinygrad's server was not started: {e}"));
            self.changed.notify_all();
        }
        started
    }

    fn launch(self: &Arc<Self>, cfg: &Json, llm: &Json, root: &Path, name: &str) -> Result<(), String> {
        let python = self.python(cfg, root)?;
        let script = Self::script(root)?;
        let port = free_port()?;
        let key = random_id("sk-egpu-");
        let args = arguments(llm, root, name, &script, port)?;
        let file = config::resolve(root, str_or(model(llm, name).unwrap_or(&Json::Null), "path", ""));
        if !file.is_file() {
            return Err(format!("{} is not there", file.display()));
        }
        let env = environment(llm, root);
        let mut command = Command::new(&python);
        command.args(&args).envs(env.iter().map(|(k, v)| (k, v))).env("OAIY_EGPU_KEY", &key);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        command.current_dir(tinygrad_folder(llm, root).filter(|d| d.is_dir()).unwrap_or_else(|| root.to_path_buf()));
        hide_window(&mut command);
        let device = env.iter().find(|(k, _)| k == "DEV").map_or("NV", |(_, v)| v.as_str());
        let shown = format!("DEV={device} {} {}", python.display(), args.join(" "));
        self.log.push(format!("studio: starting {shown}"));
        let mut child = command.spawn().map_err(|e| format!("{}: {e}", python.display()))?;
        for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten() {
            let log = Arc::clone(&self.log);
            std::thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    log.push(line);
                }
            });
        }
        let mut g = self.lock();
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
        g.model = Some(name.to_string());
        g.failed_at = None;
        g.last_used = Instant::now();
        g.command = shown;
        drop(g);
        self.changed.notify_all();
        let this = Arc::clone(self);
        std::thread::Builder::new().name("egpu-watch".into()).spawn(move || this.watch(generation)).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Wait for the server to list its model (it does once the model has loaded and its kernels are compiled), then
    /// keep watching for an exit nobody asked for. Only `/v1/models` says it is ready: any other address answers
    /// with tinygrad's chat page, loaded or not.
    fn watch(self: Arc<Self>, generation: u64) {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let (addr, key, ready) = {
                let mut g = self.lock();
                if g.generation != generation {
                    return;
                }
                let exited = g.child.as_mut().and_then(|c| c.try_wait().ok().flatten());
                if let Some(status) = exited {
                    g.child = None;
                    g.lifeline = None;
                    g.state = State::Failed;
                    g.failed_at = Some(Instant::now());
                    let tail = self.log.tail(12);
                    g.error = Some(format!("tinygrad's server exited ({status}):\n{tail}"));
                    self.log.push(format!("studio: tinygrad's server exited ({status})"));
                    self.changed.notify_all();
                    return;
                }
                (g.addr.clone(), g.key.clone(), g.state == State::Ready)
            };
            if ready {
                continue;
            }
            let auth = format!("Bearer {key}");
            let listed = oaiy_engine::http::fetch(&addr, "GET", "/v1/models", &[("Authorization", &auth)], b"", Duration::from_secs(2))
                .ok()
                .filter(|r| r.status == 200)
                .and_then(|r| r.body(1 << 20).ok())
                .and_then(|b| Json::parse(&b).ok())
                .is_some_and(|v| v.get("data").and_then(Json::as_array).is_some());
            if listed {
                let mut g = self.lock();
                if g.generation == generation && g.state == State::Starting {
                    g.state = State::Ready;
                    g.ready_after = g.started.map(|s| s.elapsed().as_secs_f64());
                    g.said = None;
                    self.log.push(format!("studio: tinygrad's server ready in {:.1}s", g.ready_after.unwrap_or(0.0)));
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
            self.log.push("studio: tinygrad's server stopped");
        }
        g.state = State::Stopped;
        g.error = None;
        g.ready_after = None;
        g.model = None;
        g.failed_at = None;
        drop(g);
        self.changed.notify_all();
    }

    /// Where `name` is answered on the eGPU, starting tinygrad's server with it if it is not held, and waiting up
    /// to `timeout` for it to load. The lease is held for as long as the request is being answered.
    pub fn ensure(self: &Arc<Self>, cfg: &Json, root: &Path, name: &str, timeout: Duration) -> Result<(Endpoint, Lease), Unready> {
        let deadline = Instant::now() + timeout;
        let mut g = self.lock();
        loop {
            let holds = g.model.as_deref() == Some(name);
            match g.state {
                State::Ready if holds => {
                    g.busy += 1;
                    g.last_used = Instant::now();
                    let endpoint = Endpoint { addr: g.addr.clone(), key: g.key.clone() };
                    drop(g);
                    return Ok((endpoint, Lease(Arc::clone(self))));
                }
                State::Starting if holds => {}
                State::Failed if holds && g.failed_at.is_some_and(|t| t.elapsed() < RETRY_AFTER) => {
                    return Err(Unready::Gone(g.error.clone().unwrap_or_else(|| "tinygrad's server failed".into())));
                }
                // Another model's request is being answered: it is not cut off.
                State::Ready | State::Starting if g.busy > 0 => {}
                _ => {
                    drop(g);
                    self.start(cfg, root, name).map_err(Unready::Gone)?;
                    g = self.lock();
                    continue;
                }
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Unready::Loading(if holds { format!("{name} is still loading on the eGPU; try again shortly") } else { "the eGPU is answering with another model; try again shortly".into() }));
            }
            g = self.changed.wait_timeout(g, left.min(Duration::from_millis(500))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// A request got no reply from the server (the connection refused, or closed before an answer). Where its
    /// process has exited, the server is taken as gone (the reason, as the next requests will be given it): they go
    /// to this computer's engine until it is tried again. A process that is still there is left serving (None):
    /// tinygrad closes a connection without a word on a request it cannot render, and one such request must not
    /// cost a model that took minutes to load.
    pub fn unanswered(&self, why: &str) -> Option<String> {
        let said = format!("tinygrad's server stopped answering: {why}");
        let mut g = self.lock();
        if g.state != State::Ready {
            return Some(g.error.clone().unwrap_or(said));
        }
        if g.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None))) {
            return None;
        }
        g.generation += 1;
        g.lifeline = None;
        g.child = None;
        g.state = State::Failed;
        g.failed_at = Some(Instant::now());
        g.error = Some(said.clone());
        drop(g);
        self.log.push(format!("studio: tinygrad's server has gone ({why})"));
        self.changed.notify_all();
        Some(said)
    }

    /// Whether `why` is news: the reason a request went to this computer's engine, said once and not per request.
    pub fn news(&self, why: &str) -> bool {
        let mut g = self.lock();
        let first_line = why.lines().next().unwrap_or("").to_string();
        if g.said.as_deref() == Some(first_line.as_str()) {
            return false;
        }
        g.said = Some(first_line);
        true
    }

    /// "… on NV", as tinygrad's server reported where the model's weights are.
    fn runs_on(&self) -> Option<String> {
        let tail = self.log.tail(400);
        tail.lines().rev().find_map(|l| l.starts_with("loaded model").then(|| l.rsplit_once(" on ").map(|(_, on)| on.trim().to_string())).flatten())
    }

    pub fn status(&self, llm: &Json) -> Json {
        let g = self.lock();
        let egpu = section(llm);
        let retry = g.failed_at.filter(|_| g.state == State::Failed).map(|t| RETRY_AFTER.saturating_sub(t.elapsed()).as_secs() as i64);
        Json::obj([
            ("available", Json::Bool(available())),
            ("enabled", Json::Bool(enabled(llm))),
            ("state", Json::str(g.state.name())),
            ("error", g.error.as_ref().map_or(Json::Null, Json::str)),
            ("model", g.model.as_ref().map_or(Json::Null, Json::str)),
            ("models", Json::Arr(models(llm).iter().map(Json::str).collect())),
            ("device", Json::str(match str_or(egpu, "device", "NV").trim() { "" => "NV", d => d })),
            ("context_tokens", Json::Int(context(llm))),
            ("busy", Json::Int(g.busy as i64)),
            ("load_seconds", g.ready_after.map_or(Json::Null, Json::Num)),
            ("idle_seconds", Json::Int(g.last_used.elapsed().as_secs() as i64)),
            ("retry_seconds", retry.map_or(Json::Null, Json::Int)),
            ("command", Json::str(&g.command)),
            ("python", g.python.as_ref().map_or(Json::Null, |(_, p)| Json::str(p.to_string_lossy()))),
            ("runs_on", self.runs_on().filter(|_| g.state == State::Ready).map_or(Json::Null, Json::str)),
        ])
    }
}

impl Drop for Egpu {
    fn drop(&mut self) {
        self.stop();
    }
}

/// What a found Python stands for: the settings it was looked for under.
fn python_key(llm: &Json) -> String {
    let egpu = section(llm);
    format!("{}|{}", str_or(egpu, "python", ""), str_or(egpu, "tinygrad", ""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llm(text: &str) -> Json {
        Json::parse(text.as_bytes()).unwrap()
    }

    const TWO: &str = r#"{"temperature": 0.7, "max_tokens": 4096,
        "egpu": {"enabled": true, "ctx": 16384, "fallback_model": "small", "env": {"JITBEAM": 2, "DEV": "AMD"}, "extra_args": ["--shard", "1"]},
        "models": [{"name": "big", "path": "models/big.gguf", "egpu": true}, {"name": "small", "path": "models/small.gguf"},
                   {"name": "off", "path": "models/off.gguf", "egpu": true, "enabled": false}, {"name": "folder", "path": "models/exl3", "egpu": true}]}"#;

    #[test]
    fn a_model_runs_on_the_egpu_only_where_it_is_set_to_and_the_egpu_is_switched_on() {
        let on = llm(TWO);
        assert!(enabled(&on) && assigned(&on, "big"));
        assert!(!assigned(&on, "small"), "not set to it");
        assert!(!assigned(&on, "off"), "a model that is switched off");
        assert!(!assigned(&on, "ghost"));
        assert_eq!(models(&on), ["big", "folder"]);
        // Switched off, or with no section at all (every configuration from before it): nothing runs there.
        let off = llm(r#"{"egpu": {"enabled": false}, "models": [{"name": "big", "path": "big.gguf", "egpu": true}]}"#);
        assert!(!assigned(&off, "big") && models(&off).is_empty());
        let none = llm(r#"{"models": [{"name": "big", "path": "big.gguf", "egpu": true}]}"#);
        assert!(!enabled(&none) && !assigned(&none, "big"));
    }

    #[test]
    fn when_the_egpu_does_not_answer_another_model_does_where_one_is_named_else_the_same_one() {
        let on = llm(TWO);
        assert_eq!(fallback(&on, "big"), "small");
        // None named, one that is not there, the model itself, or one that is on the eGPU too: the same model.
        for other in ["", "ghost", "big", "folder", "off"] {
            let text = TWO.replace(r#""fallback_model": "small""#, &format!(r#""fallback_model": "{other}""#));
            assert_eq!(fallback(&llm(&text), "big"), "big", "{other:?}");
        }
    }

    #[test]
    fn tinygrads_server_is_started_through_the_launcher_with_the_models_file_a_port_and_the_context() {
        let on = llm(TWO);
        let root = Path::new("/data/engines");
        let script = root.join("cache").join("egpu").join("egpu_serve.py");
        let a = arguments(&on, root, "big", &script, 18000).unwrap();
        let file = root.join("models/big.gguf").to_string_lossy().into_owned();
        assert_eq!(a, ["-u", script.to_string_lossy().as_ref(), "--watch-stdin", "--model", file.as_str(), "--serve", "18000", "--max_context", "16384", "--shard", "1"]);
        // tinygrad's own default context is too small for an agent's prompt: 8,192 unless set, and nothing silly.
        for (ctx, want) in [("16384", 16384), ("0", 8192), ("12", 8192), ("\"big\"", 8192), ("4096", 4096)] {
            assert_eq!(context(&llm(&format!(r#"{{"egpu": {{"ctx": {ctx}}}}}"#))), want, "{ctx}");
        }
        assert_eq!(context(&llm("{}")), 8192);
        // A folder of weights (an EXL3 checkpoint) is not something tinygrad's server reads.
        assert!(arguments(&on, root, "folder", &script, 1).unwrap_err().contains("not a GGUF file"));
        assert!(arguments(&on, root, "ghost", &script, 1).unwrap_err().contains("not one of the enabled"));
    }

    #[test]
    fn the_servers_environment_names_the_device_and_what_the_settings_add() {
        let env = environment(&llm(TWO), Path::new("/data"));
        let get = |k: &str| env.iter().find(|(have, _)| have == k).map(|(_, v)| v.as_str());
        assert_eq!(get("DEV"), Some("AMD"), "the settings' own value wins over the device");
        assert_eq!(get("JITBEAM"), Some("2"));
        assert_eq!(get("PYTHONUNBUFFERED"), Some("1"));
        assert_eq!(get("PYTHONPATH"), None, "no tinygrad folder named");
        assert_eq!(env.iter().filter(|(k, _)| k == "DEV").count(), 1);
        // An NVIDIA card unless told, and a checkout's folder on Python's path.
        let env = environment(&llm(r#"{"egpu": {"tinygrad": "/src/tinygrad"}}"#), Path::new("/data"));
        assert!(env.contains(&("DEV".into(), "NV".into())));
        assert!(env.iter().any(|(k, v)| k == "PYTHONPATH" && v.replace('\\', "/").starts_with("/src/tinygrad")), "{env:?}");
    }

    #[test]
    fn an_app_opened_from_the_finder_still_finds_homebrew_docker_and_nvidias_compiler() {
        let path = search_path(Some("/Users/sam"), "/usr/bin:/bin:/usr/local/bin");
        assert_eq!(path, "/Users/sam/.local/bin:/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:/usr/bin:/bin");
        assert!(search_path(None, "").starts_with("/opt/homebrew/bin:"));
    }

    #[test]
    fn a_path_typed_with_a_tilde_is_in_the_home_folder() {
        let (root, home) = (Path::new("/data"), std::ffi::OsStr::new("/Users/sam"));
        assert_eq!(typed(root, "~/tinygrad", Some(home)), Path::new("/Users/sam").join("tinygrad"));
        assert_eq!(typed(root, "~/tinygrad", None), config::resolve(root, "~/tinygrad"), "no home known: as any path");
        assert_eq!(typed(root, "tinygrad", Some(home)), config::resolve(root, "tinygrad"));
    }

    #[test]
    fn a_named_python_is_the_only_one_asked() {
        let named = llm(r#"{"egpu": {"python": "/opt/venv/bin/python3", "tinygrad": "/src/tinygrad"}}"#);
        assert_eq!(pythons(&named, Path::new("/data")).len(), 1);
        // None named: whatever is installed, and the PATH's own last.
        let found = pythons(&llm("{}"), Path::new("/data"));
        assert_eq!(found.last().map(|p| p.to_string_lossy().into_owned()).as_deref(), Some(if cfg!(windows) { "python" } else { "python3" }));
    }

    #[test]
    fn a_request_goes_to_tinygrad_with_ooiys_name_temperature_and_reply_limit() {
        let on = llm(TWO);
        let sent = |body: &str| Json::parse(&request_body(&on, "big", body.as_bytes()).unwrap()).unwrap();
        let bare = sent(r#"{"messages": [{"role": "user", "content": "hi"}], "stream": true}"#);
        assert_eq!(bare.get("model").and_then(Json::as_str), Some("big"));
        assert_eq!(bare.get("temperature").and_then(Json::as_f64), Some(0.7), "tinygrad's own default is 0");
        assert_eq!(bare.get("max_tokens").and_then(Json::as_i64), Some(4096), "tinygrad has no limit of its own");
        assert_eq!(bare.get("stream").and_then(Json::as_bool), Some(true));
        // What the request says is kept.
        let said = sent(r#"{"model": "other", "temperature": 0, "max_completion_tokens": 64, "messages": []}"#);
        assert_eq!(said.get("model").and_then(Json::as_str), Some("big"));
        assert_eq!(said.get("temperature").and_then(Json::as_f64), Some(0.0));
        assert_eq!(said.get("max_tokens"), None);
        assert!(request_body(&on, "big", b"[1, 2]").is_none() && request_body(&on, "big", b"not json").is_none());
    }

    #[test]
    fn a_request_is_thought_about_as_ooiys_own_engine_would_decide() {
        let asks = |settings: &str, body: &str| thinks(&llm(settings), &llm(body));
        // Nothing said: the language model's own setting, which is not to.
        assert!(!asks("{}", "{}") && !asks(r#"{"thinking": false}"#, "{}") && asks(r#"{"thinking": true}"#, "{}"));
        // The request's own word wins either way.
        assert!(asks("{}", r#"{"reasoning_effort": "low"}"#) && asks("{}", r#"{"reasoning_effort": 50}"#));
        assert!(!asks(r#"{"thinking": true}"#, r#"{"reasoning_effort": "none"}"#) && !asks(r#"{"thinking": true}"#, r#"{"reasoning_effort": "minimal"}"#));
        assert!(asks("{}", r#"{"thinking": {"type": "enabled"}}"#) && !asks(r#"{"thinking": true}"#, r#"{"thinking": {"type": "disabled"}}"#));
        assert!(asks(r#"{"thinking": true}"#, r#"{"reasoning_effort": null}"#));
    }

    #[test]
    fn a_request_with_a_picture_is_not_tinygrads() {
        let text = llm(r#"{"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "and"}]}]}"#);
        assert!(!has_picture(&text));
        let picture = llm(r#"{"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}"#);
        assert!(has_picture(&picture));
    }

    #[test]
    fn a_model_that_cannot_be_started_is_gone_until_it_is_tried_again_and_says_why() {
        let dir = std::env::temp_dir().join(format!("oaiy-egpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), b"GGUF").unwrap();
        // A Python that is not there: the start fails, and so does the next request, without asking again.
        let cfg = llm(r#"{"llm": {"egpu": {"enabled": true, "python": "no-such-python-here"}, "models": [{"name": "m", "path": "m.gguf", "egpu": true}]}}"#);
        let egpu = Arc::new(Egpu::new());
        let why = match egpu.ensure(&cfg, &dir, "m", Duration::from_secs(5)) {
            Err(Unready::Gone(why)) => why,
            other => panic!("{:?}", other.map(|_| ())),
        };
        assert!(why.contains("no Python here has tinygrad"), "{why}");
        assert_eq!(egpu.state(), State::Failed);
        let status = egpu.status(cfg.get("llm").unwrap());
        assert_eq!(status.get("model").and_then(Json::as_str), Some("m"));
        assert!(status.get("retry_seconds").and_then(Json::as_i64).is_some_and(|s| s > 50));
        assert!(matches!(egpu.ensure(&cfg, &dir, "m", Duration::from_secs(5)), Err(Unready::Gone(_))));
        // Said once in the log, not per request.
        assert!(egpu.news(&why) && !egpu.news(&why));
        // Stopped by hand: the next request tries again.
        egpu.stop();
        assert_eq!(egpu.state(), State::Stopped);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
