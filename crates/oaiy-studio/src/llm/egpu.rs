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
//!
//! With `llm.egpu.engine` set to `webgpu`, the card answers with OAIY's own engine in place of tinygrad's server: the
//! launcher holds the card through tinygrad as a WebGPU adapter (`webgpu_server.py`, written beside it) and runs this
//! computer's `oaiy-llm-server` on it with the model (docs/TINYGPU.md). Every model the engine runs runs there, at the
//! card's speed (a 27B model ten times tinygrad's server's), and nothing else of the arrangement changes.
//!
//! The card can then be lent to OAIY's image worker ([`Egpu::lend`], `crate::media`): the launcher stops the engine,
//! keeps the card (opening it again would reset it), and the worker takes the card's socket as its WebGPU adapter, with
//! the memory the engine had. Requests for the model wait meanwhile; given back, the engine starts again.

use crate::config;
use crate::llm::{Endpoint, State};
use crate::util::{bool_or, int_or, random_key, str_or, LogRing};
use oaiy_engine::json::Json;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The launcher, written beside the configuration before each start. Carried by a Mac's build only.
#[cfg(any(target_os = "macos", test))]
const SERVE_PY: &str = include_str!("egpu_serve.py");
#[cfg(not(any(target_os = "macos", test)))]
const SERVE_PY: &str = "";
/// The card's WebGPU server, written beside the launcher for `llm.egpu.engine` `webgpu`.
#[cfg(any(target_os = "macos", test))]
const CARD_PY: &str = include_str!("../../../../tools/tinygpu/webgpu_server.py");
#[cfg(not(any(target_os = "macos", test)))]
const CARD_PY: &str = "";
/// After a start that failed, how long requests go to this computer's engine before it is tried again.
const RETRY_AFTER: Duration = Duration::from_secs(60);
/// tinygrad's own default is 4,096 tokens, less than an agent's prompt with its tools.
const DEFAULT_CTX: i64 = 8192;
/// How long a Python has to say what it has installed.
const CHECK_WAIT: Duration = Duration::from_secs(30);
/// What a start, or a request waiting for one, is told when the server was stopped by someone meanwhile.
const STOPPED: &str = "the eGPU was stopped, or its settings changed, while it was being started";

/// What runs the card when a plugin provides it (OAIY Desktop's eGPU module: tools/tinygpu/plugin): its own Python,
/// the folder put on that Python's path (tinygrad and what it needs), and the folder with the launcher and the card's
/// server (and the card's kernels and firmware beside them). In place of the settings' Python and tinygrad
/// (`llm.egpu.python`, `llm.egpu.tinygrad`) and of the launcher this build carries ([`Egpu::provide`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Runtime {
    pub python: PathBuf,
    pub pythonpath: PathBuf,
    pub scripts: PathBuf,
    /// Who provides it, as the log and the status say it ("eGPU 0.1.0").
    pub by: String,
}

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

/// Whether the card answers with OAIY's own engine (`llm.egpu.engine`: `webgpu`) rather than tinygrad's LLM server
/// (`tinygrad`, the default).
pub fn webgpu(llm: &Json) -> bool {
    str_or(section(llm), "engine", "tinygrad").trim() == "webgpu"
}

/// The context tinygrad's server is started with (`llm.egpu.ctx`; it sets its cache aside for all of it at once).
pub fn context(llm: &Json) -> i64 {
    match int_or(section(llm), "ctx", DEFAULT_CTX) {
        n if n >= 512 => n,
        _ => DEFAULT_CTX,
    }
}

/// The context model `name` is started with on the card: its own (`llm.models[].egpu_ctx`) where it has one, else the
/// eGPU's ([`context`]). A model whose cache can be the host's past what the card holds (Qwen3.8-Flash-Next's) may be
/// given far more than one whose cache must be all on the card (a dense 27B's: 32,768 fills a 24 GB card beside it).
pub fn model_context(llm: &Json, name: &str) -> i64 {
    match model(llm, name).map(|m| int_or(m, "egpu_ctx", 0)) {
        Some(n) if n >= 512 => n,
        _ => context(llm),
    }
}

/// The Pythons to ask, in order: a provider's ([`Runtime`]) and no other; else the one `llm.egpu.python` names and no
/// other; else a virtual environment inside the tinygrad folder (`llm.egpu.tinygrad`), Homebrew's, the system's, and
/// whatever `python3` the PATH finds. An app opened from the Finder has a PATH without Homebrew's folders, so those are
/// named outright.
pub fn pythons(llm: &Json, root: &Path, runtime: Option<&Runtime>) -> Vec<PathBuf> {
    if let Some(r) = runtime {
        return vec![r.python.clone()];
    }
    let egpu = section(llm);
    let named = str_or(egpu, "python", "").trim();
    if !named.is_empty() {
        // (a bare name, `python3.12`, is one for the PATH to find, not a file beside the configuration)
        return vec![if named.contains(['/', '\\']) { typed(root, named, std::env::var_os("HOME").as_deref()) } else { PathBuf::from(named) }];
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
/// NVIDIA card; `AMD` for a Radeon), the tinygrad folder on Python's path (a provider's, else the settings'), and
/// whatever `llm.egpu.env` names (`JITBEAM` and the like).
fn environment(llm: &Json, root: &Path, runtime: Option<&Runtime>) -> Vec<(String, String)> {
    let egpu = section(llm);
    let device = str_or(egpu, "device", "NV").trim();
    let mut env = vec![("DEV".to_string(), if device.is_empty() { "NV".to_string() } else { device.to_string() }), ("PYTHONUNBUFFERED".into(), "1".into())];
    // (the launcher of OAIY's engine on the card says the card's memory and load each second, for the page)
    env.push(("OAIY_EGPU_STATS".into(), "1".into()));
    if runtime.is_some() {
        // (a plugin's files are trusted by their digest: its Python writes no bytecode beside them)
        env.push(("PYTHONDONTWRITEBYTECODE".into(), "1".into()));
    }
    if let Some(folder) = runtime.map(|r| r.pythonpath.clone()).or_else(|| tinygrad_folder(llm, root)) {
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

/// Whether the card's engine runs in incognito (`privacy.incognito`, with OAIY's own engine on the card: `--incognito`
/// is its flag). tinygrad's server keeps nothing of a request on disk (its prompt's cache is in memory) and has no such
/// flag, which its argument parser would refuse: it is started as it is.
pub fn private(cfg: &Json) -> bool {
    cfg.get("llm").is_some_and(webgpu) && cfg.get("privacy").is_some_and(|p| bool_or(p, "incognito", false))
}

/// tinygrad's server's command line for `name`: the launcher, the model's file, the port and the context, then
/// `llm.egpu.extra_args`. A model that is not one GGUF file is refused: tinygrad's server reads nothing else.
pub fn arguments(llm: &Json, root: &Path, name: &str, script: &Path, port: u16) -> Result<Vec<String>, String> {
    let m = model(llm, name).ok_or_else(|| format!("{name} is not one of the enabled language models"))?;
    if webgpu(llm) {
        return engine_arguments(llm, root, m, script, port);
    }
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

/// The launcher's command line for OAIY's own engine on the card: `--webgpu`, the engine's program (the one this
/// computer's engine runs), and the engine's arguments as this computer's are made, for this model alone, with the
/// eGPU's context. Not among them: the key (the launcher passes it in the environment), the GPUs to use (the card is
/// the engine's one adapter) and the stdin watch (the launcher's own, which gives the engine one of its own pipe);
/// the prompt cache and DeepSeek's experts' counts of uses are each a folder or a file of the card's own. And the backend is the
/// GPU's and nothing else (`--backend webgpu`): an engine that cannot reach the card ends, saying so, where one left
/// to choose ran the model on this computer's CPU under the eGPU's name (Flash-Next at a token a second).
fn engine_arguments(llm: &Json, root: &Path, m: &Json, script: &Path, port: u16) -> Result<Vec<String>, String> {
    let launches = crate::llm::launches(llm, root);
    let program = launches.iter().map(|l| l.program.clone()).find(|p| p.is_file()).or_else(|| launches.first().map(|l| l.program.clone())).ok_or("no engine program")?;
    let mut alone = llm.clone();
    let mut one = m.clone();
    crate::util::set(&mut one, "egpu", Json::Bool(false));
    crate::util::set(&mut one, "devices", Json::Arr(Vec::new()));
    crate::util::set(&mut alone, "models", Json::Arr(vec![one]));
    crate::util::set(&mut alone, "ctx", Json::Int(model_context(llm, str_or(m, "name", ""))));
    crate::util::set(&mut alone, "devices", Json::Arr(Vec::new()));
    let (engine, _) = crate::llm::arguments_for(&alone, root, port, false, Some(str_or(m, "name", "")))?;
    let mut a = vec!["-u".to_string(), script.to_string_lossy().into_owned(), "--watch-stdin".into(), "--webgpu".into(), program.to_string_lossy().into_owned()];
    let mut it = engine.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--api-key" | "--devices" => {
                it.next();
            }
            "--watch-stdin" => {}
            "--prompt-cache" => {
                it.next();
                a.push(flag);
                a.push(root.join("cache").join("prompt-states-egpu").to_string_lossy().into_owned());
            }
            // (DeepSeek-V4.1's experts' counts of uses: the card's engine keeps its own, as its prompt states, since
            // what its card and this computer's memory hold is not what this computer's engine's tiers do. With none
            // its card began each start with whichever experts the first prompt met: 7% of a reply's on the card
            // where 56% with the counts of a session before, the rest read from the drive)
            "--usage" => {
                it.next();
                a.push(flag);
                a.push(root.join("cache").join("expert-usage-egpu.bin").to_string_lossy().into_owned());
            }
            _ => a.push(flag),
        }
    }
    a.push("--backend".into());
    a.push("webgpu".into());
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
/// computer's engine's (OAIY's own engine on the card takes it as any other).
pub fn has_picture(body: &Json) -> bool {
    let parts = |m: &Json| m.get("content").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default();
    body.get("messages").and_then(Json::as_array).unwrap_or(&[]).iter().flat_map(parts).any(|p| str_or(&p, "type", "text") != "text")
}

/// Why the card's server is to stop as the card is given back, a stop asked for while it was lent
/// ([`Egpu::stop_unless_lent`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Deferred {
    /// Its launch changed (the eGPU's settings, its model's file): the server that holds the card is not the one the
    /// settings start now, and it goes whatever is asked of it after.
    Changed,
    /// It is not wanted now: a default model chosen on this computer, a Stop by hand. A start for the model it holds,
    /// asked for after (the card's model chosen as the default again), keeps it: the card is given back to it.
    Unwanted,
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
    /// Requests waiting for the model it is loading: it is not swapped out from under them either (two models
    /// asked for at once would otherwise stop each other's load in turn, and neither would ever be ready).
    waiting: usize,
    /// Stops someone asked for (Stop, a change of settings, the studio going), counted: a start that began before
    /// one stands down, and a request that waited through one does not start the server again.
    epoch: u64,
    last_used: Instant,
    command: String,
    /// The Python that has tinygrad, for the configuration it was found under.
    python: Option<(String, PathBuf)>,
    /// The last reason a request went to this computer's engine, so the log says it once.
    said: Option<String>,
    /// The launcher's word on the card (OAIY's engine on it).
    card: Arc<Card>,
    /// The card is lent to OAIY's image worker: the engine on it paused, requests for its model waiting.
    lent: bool,
    /// The engine on the card is paused: asked to be, and not yet asked back.
    paused: bool,
    /// Bring the engine back as the card is given back (it was running when it was lent).
    resume_after: bool,
    /// A request waits for the lent card: whoever has it gives it back as soon as it may (`crate::media`).
    wanted: bool,
    /// The pinned picture model whose worker keeps the lent card (`media.pinned`): requests for the card's language
    /// model are told so rather than waiting for a card that is not given back.
    pin_holder: Option<String>,
    /// The card's Python, tinygrad and launcher from a provider ([`Egpu::provide`]); None: the settings'.
    runtime: Option<Runtime>,
    /// A stop asked for while the card was lent (a change of the eGPU's settings, a default model chosen on this
    /// computer, a Stop by hand), and why: done as the card is given back ([`Egpu::stop_unless_lent`]), not under the
    /// picture worker on its socket; one that is not wanted any more by then is not ([`Deferred`]).
    stop_when_given_back: Option<Deferred>,
    /// Stopped for good, as the studio stops ([`Egpu::close`]): no server is started on the card again.
    closed: bool,
}

pub struct Egpu {
    inner: Mutex<Inner>,
    changed: Condvar,
    /// One start at a time: finding the Python runs processes, outside the state's lock.
    launching: Mutex<()>,
    /// Servers stopped and not gone yet. Only one process may hold the card: a server started while the last still
    /// lets it go finds tinygrad's lock taken and fails, and the request that started it goes to this computer's
    /// engine (a change of settings, then a request at once, did that). So a start waits for them ([`GONE_WAIT`]).
    /// Every Egpu of this process counts on the same ([`stopping_servers`]): the engines restarted in OAIY Desktop's
    /// own process are a new studio with a new Egpu, which counted none of the old one's and opened the card at once.
    going: Arc<(Mutex<usize>, Condvar)>,
    pub log: Arc<LogRing>,
}

/// How long a start waits for a stopped server to let the card go: a stop that takes longer kills it at 20 s.
pub(crate) const GONE_WAIT: Duration = Duration::from_secs(25);

/// What the card's start is told once the studio has stopped ([`Egpu::close`]).
const CLOSED: &str = "the studio is stopping: no server is started on the eGPU again";

/// The count of the card's servers stopped and not gone yet ([`Egpu::going`]), one for the whole process: there is
/// one card, whichever studio stopped its server.
fn stopping_servers() -> Arc<(Mutex<usize>, Condvar)> {
    static GOING: std::sync::OnceLock<Arc<(Mutex<usize>, Condvar)>> = std::sync::OnceLock::new();
    Arc::clone(GOING.get_or_init(Arc::default))
}

/// Wait (`wait` at most) until every server of the card this process stopped has let it go, whichever studio stopped
/// it: for OAIY Desktop's update, as it hands off to the installer once its engines have stopped (`crate::Studio::shutdown`
/// waits a few seconds alone). Whether they had.
pub fn await_let_go(wait: Duration) -> bool {
    let (count, gone) = &*stopping_servers();
    let deadline = Instant::now() + wait;
    let mut n = count.lock().unwrap_or_else(|p| p.into_inner());
    while *n > 0 {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        n = gone.wait_timeout(n, left).unwrap_or_else(|p| p.into_inner()).0;
    }
    true
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

/// What the launcher has said of the card (`llm.egpu.engine` `webgpu`), one for each of its processes: where the
/// card's socket is, whether OAIY's engine on it is paused (its memory free for another program on the socket), and
/// the card's memory and load as it last said them (each second).
#[derive(Default)]
struct Card {
    said: Mutex<CardSaid>,
    changed: Condvar,
}

#[derive(Default)]
struct CardSaid {
    socket: Option<String>,
    paused: bool,
    /// `oaiy-egpu: usage {..}`'s word, and when it came.
    usage: Option<(Json, Instant)>,
}

/// How long the card's last word on its memory and load is shown: the launcher says it each second.
const USAGE_FRESH: Duration = Duration::from_secs(5);
/// The card's memory in use (not counting tinygrad's cached buffers) under which a departed worker's has been let go.
const CARD_FREE_MB: i64 = 2048;

impl Card {
    /// A line of the launcher's output, taken where it is one of its words on the card. True for its word on the
    /// card's memory and load, which is said each second and kept out of the log.
    fn heard(&self, line: &str) -> bool {
        let mut s = self.said.lock().unwrap_or_else(|p| p.into_inner());
        match line.trim_end() {
            "oaiy-egpu: paused" => s.paused = true,
            "oaiy-egpu: resumed" => s.paused = false,
            l => {
                if let Some(said) = l.strip_prefix("oaiy-egpu: usage ") {
                    s.usage = Json::parse(said.as_bytes()).ok().filter(|u| u.as_object().is_some()).map(|u| (u, Instant::now()));
                    return true;
                }
                match l.strip_prefix("oaiy-egpu: card at ") {
                    Some(path) => s.socket = Some(path.to_string()),
                    None => return false,
                }
            }
        }
        drop(s);
        self.changed.notify_all();
        false
    }

    /// Whether the card's memory in use, less tinygrad's cached buffers (which a load lets go), is under `limit_mb` in a
    /// word the launcher said after `since`, waiting up to `wait` for one: None where it says nothing of it.
    fn freed_since(&self, since: Instant, limit_mb: i64, wait: Duration) -> Option<bool> {
        let deadline = Instant::now() + wait;
        let mut heard_any = false;
        loop {
            {
                let s = self.said.lock().unwrap_or_else(|p| p.into_inner());
                if let Some((u, at)) = s.usage.as_ref() {
                    heard_any = true;
                    let mb = |k: &str| u.get(k).and_then(Json::as_i64).unwrap_or(0) >> 20;
                    if *at > since && mb("used") - mb("cached") < limit_mb {
                        return Some(true);
                    }
                }
            }
            if Instant::now() >= deadline {
                return heard_any.then_some(false);
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// The card's memory (MB) and load (percent) as the launcher last said them, where it said them lately: for the
    /// page's hardware, as `crate::system` says a GPU's.
    fn usage(&self) -> Json {
        let s = self.said.lock().unwrap_or_else(|p| p.into_inner());
        let Some((u, _)) = s.usage.as_ref().filter(|(_, at)| at.elapsed() < USAGE_FRESH) else {
            return Json::Null;
        };
        let mb = |k: &str| u.get(k).and_then(Json::as_i64).map_or(Json::Null, |b| Json::Int(b >> 20));
        Json::obj([
            ("memory_total_mb", mb("total")),
            ("memory_used_mb", mb("used")),
            ("memory_cached_mb", mb("cached")),
            ("load", u.get("load").and_then(Json::as_i64).map_or(Json::Null, |p| Json::Int(p.clamp(0, 100)))),
        ])
    }

    /// The card's socket once the launcher has said where it is and that the engine is paused, waiting up to `wait`
    /// (in slices, `gone` asked between them: whether to stop waiting, the launcher gone).
    fn paused_at(&self, wait: Duration, gone: impl Fn() -> bool) -> Option<String> {
        let deadline = Instant::now() + wait;
        let mut s = self.said.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if s.paused && s.socket.is_some() {
                return s.socket.clone();
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || gone() {
                return None;
            }
            s = self.changed.wait_timeout(s, left.min(Duration::from_millis(500))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }
}

/// What a start came to.
enum Begun {
    /// The server was started with the model (and the caller counted as waiting for it, where it asked to be).
    Launched,
    /// It has the model already, loading or ready.
    Held,
    /// It is answering, or awaited, with another model (which): not cut off.
    InUse(String),
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
        // (a test's Egpu counts its own, for tests run side by side: one's stopped stand-in, killed only after 20 s,
        // would hold up every other's start. The test of the count itself counts on the process's)
        Self::counting_on(if cfg!(test) { Arc::default() } else { stopping_servers() })
    }

    /// An Egpu whose stopped servers are counted on `going` ([`stopping_servers`] but in tests).
    fn counting_on(going: Arc<(Mutex<usize>, Condvar)>) -> Egpu {
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
                waiting: 0,
                epoch: 0,
                last_used: Instant::now(),
                command: String::new(),
                python: None,
                said: None,
                card: Arc::default(),
                lent: false,
                paused: false,
                resume_after: false,
                wanted: false,
                pin_holder: None,
                runtime: None,
                stop_when_given_back: None,
                closed: false,
            }),
            changed: Condvar::new(),
            launching: Mutex::new(()),
            going,
            log: Arc::new(LogRing::new(2000)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Wait (`wait` at most) until every stopped server has let the card go.
    pub(crate) fn wait_gone(&self, wait: Duration) -> Result<(), String> {
        let (count, gone) = &*self.going;
        let deadline = Instant::now() + wait;
        let mut n = count.lock().unwrap_or_else(|p| p.into_inner());
        while *n > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("the eGPU's last server has not let the card go yet".into());
            }
            n = gone.wait_timeout(n, left).unwrap_or_else(|p| p.into_inner()).0;
        }
        Ok(())
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

    /// The card taken as lent to the image worker, as [`Egpu::lend`] leaves it (a test's: no engine to pause).
    #[cfg(test)]
    pub(crate) fn lent_for_test(&self) {
        self.lock().lent = true;
    }

    /// [`Self::adopt`] with a process that ends when its standard input does, that input held as the lifeline (a
    /// launcher's `--watch-stdin`): stopped, it goes at once, as a server that lets the card go does.
    #[cfg(test)]
    pub(crate) fn adopt_watching(&self, addr: &str, key: &str, model: &str) {
        let mut c = if cfg!(windows) {
            let mut c = Command::new("findstr");
            c.arg("x");
            c
        } else {
            Command::new("cat")
        };
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        let lifeline = child.stdin.take();
        self.adopt(addr, key, model, Some(child));
        self.lock().lifeline = lifeline;
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

    /// The launcher's file: a provider's own (beside its card's server, kernels and firmware), else made what this OAIY
    /// carries at each use (an older OAIY's copy must not linger). Written only where it differs: a Python that is
    /// starting on it must not read half of it.
    fn script(root: &Path, runtime: Option<&Runtime>) -> Result<PathBuf, String> {
        if let Some(r) = runtime {
            let launcher = r.scripts.join("egpu_serve.py");
            return if launcher.is_file() { Ok(launcher) } else { Err(format!("{} has no egpu_serve.py ({} provides the eGPU)", r.scripts.display(), r.by)) };
        }
        let dir = root.join("cache").join("egpu");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, text) in [("webgpu_server.py", CARD_PY), ("egpu_serve.py", SERVE_PY)] {
            let file = dir.join(name);
            if std::fs::read(&file).ok().as_deref() != Some(text.as_bytes()) {
                std::fs::write(&file, text).map_err(|e| format!("{}: {e}", file.display()))?;
            }
        }
        Ok(dir.join("egpu_serve.py"))
    }

    /// Ask each Python in turn what it has: `{ok, python, version, tinygrad, commit, server, jinja2, tried}`, `ok`
    /// when one has tinygrad's LLM server (the first that does is the one used).
    pub fn check(&self, cfg: &Json, root: &Path) -> Json {
        let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
        if !available() {
            return Json::obj([("ok", Json::Bool(false)), ("error", Json::str("tinygrad's server is used on a Mac only"))]);
        }
        let runtime = self.lock().runtime.clone();
        let runtime = runtime.as_ref();
        let script = match Self::script(root, runtime) {
            Ok(s) => s,
            Err(e) => return Json::obj([("ok", Json::Bool(false)), ("error", Json::str(e))]),
        };
        let env = environment(&llm, root, runtime);
        let cwd = working_dir(&llm, root, runtime);
        let mut tried = Vec::new();
        for python in pythons(&llm, root, runtime) {
            let shown = python.to_string_lossy().into_owned();
            match probe(&python, &script, &env, Some(&cwd)) {
                Ok(mut said) if said.get("server").and_then(Json::as_str).is_some() => {
                    crate::util::set(&mut said, "ok", Json::Bool(true));
                    crate::util::set(&mut said, "tried", Json::Arr(tried));
                    if let Some(r) = runtime {
                        crate::util::set(&mut said, "provided_by", Json::str(&r.by));
                    }
                    self.lock().python = Some((python_key(&llm, runtime), python));
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
        let key_now = || {
            let g = self.lock();
            (python_key(&llm, g.runtime.as_ref()), g.python.clone())
        };
        if let (key, Some((found, python))) = key_now() {
            if key == found {
                return Ok(python);
            }
        }
        let said = self.check(cfg, root);
        match key_now() {
            (key, Some((found, python))) if key == found && bool_or(&said, "ok", false) => Ok(python),
            _ => Err(str_or(&said, "error", "no Python with tinygrad was found").to_string()),
        }
    }

    /// Start tinygrad's server with `name` by hand. A server that has `name` already is left as it is, and one that
    /// is answering (or awaited) with another model is not cut off: that is said instead.
    pub fn start(self: &Arc<Self>, cfg: &Json, root: &Path, name: &str) -> Result<(), String> {
        let epoch = self.lock().epoch;
        match self.begin(cfg, root, name, epoch, true, false, false)? {
            Begun::Launched | Begun::Held => Ok(()),
            Begun::InUse(other) => Err(format!("the eGPU is answering with {other}: wait for it, or stop it first")),
        }
    }

    /// Start the server as the studio starts (`llm.autostart`), once the card's Python, tinygrad and launcher are known:
    /// a provider's (OAIY Desktop's eGPU plugin says them a few seconds after the studio starts), else after `wait`
    /// the settings'. A server started or stopped meanwhile (a request, Start or Stop, a change of settings) has its
    /// way, and this stands down: Ok(None). `still` is asked last whether the settings still want it, and with what:
    /// the configuration, its folder and the model. Ok(Some(model)) once it is being started.
    pub fn start_when_provided(self: &Arc<Self>, wait: Duration, still: impl FnOnce() -> Option<(Json, PathBuf, String)>) -> Result<Option<String>, String> {
        let deadline = Instant::now() + wait;
        let mut g = self.lock();
        let epoch = g.epoch;
        while g.runtime.is_none() && g.epoch == epoch && g.state == State::Stopped {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            g = self.changed.wait_timeout(g, left).unwrap_or_else(|p| p.into_inner()).0;
        }
        if g.epoch != epoch || g.state != State::Stopped {
            return Ok(None);
        }
        drop(g);
        let Some((cfg, root, name)) = still() else { return Ok(None) };
        match self.begin(&cfg, &root, &name, epoch, true, false, false)? {
            Begun::Launched | Begun::Held => Ok(Some(name)),
            Begun::InUse(other) => Err(format!("the eGPU is answering with {other}")),
        }
    }

    /// Start the server with `name` unless it has it, in place of a model nobody is using. `epoch`: the stops
    /// someone asked for that the caller had seen; one more since, and this start stands down. `again`: start even
    /// where a start with `name` has just failed (a start by hand: a request waits out [`RETRY_AFTER`], or every
    /// request queued behind a failure would try all the Pythons once more). `count`: the caller is a request, to
    /// be counted as waiting for the model from the moment the server is set to starting, so that another model's
    /// request arriving just then does not find it unawaited and stop it. `paused`: the card held and the engine not
    /// started (OAIY's engine only: the card is wanted for the image worker).
    #[allow(clippy::too_many_arguments)]
    fn begin(self: &Arc<Self>, cfg: &Json, root: &Path, name: &str, epoch: u64, again: bool, count: bool, paused: bool) -> Result<Begun, String> {
        let llm = cfg.get("llm").ok_or("no llm section")?;
        if !enabled(llm) {
            return Err(if available() { "the eGPU is switched off (Settings)".into() } else { "tinygrad's server is used on a Mac only".into() });
        }
        if !assigned(llm, name) {
            return Err(format!("{name} is not set to run on the eGPU"));
        }
        let _one = self.launching.lock().unwrap_or_else(|p| p.into_inner());
        {
            let mut g = self.lock();
            if g.closed {
                return Err(CLOSED.into());
            }
            if g.epoch != epoch {
                return Err(STOPPED.into());
            }
            let holds = g.model.as_deref() == Some(name);
            match g.state {
                State::Starting | State::Ready if holds => {
                    if g.lent {
                        // (its model asked for again while the card is lent, after a stop that did not want it any
                        // more: it is wanted now, and is given the card back rather than stopped then)
                        if g.stop_when_given_back == Some(Deferred::Unwanted) {
                            g.stop_when_given_back = None;
                            self.log.push(format!("studio: {name} is wanted on the card again: its server is not stopped as the card is given back"));
                        }
                    } else if g.paused {
                        // (an engine paused since a picture, its card free: started is loaded again, now)
                        self.resume(&mut g);
                        drop(g);
                        self.changed.notify_all();
                    }
                    return Ok(Begun::Held);
                }
                // (the card lent to the image worker is not taken from under its job: a start here stopped the launcher
                // that holds it, mid-picture. Requests for a model on the card wait for it to be given back)
                State::Starting | State::Ready if g.lent => {
                    return Err(format!("the eGPU's card is lent to a picture, a video or a sound: {name} can be loaded there once it is given back"));
                }
                State::Starting | State::Ready if g.busy > 0 || g.waiting > 0 => return Ok(Begun::InUse(g.model.clone().unwrap_or_default())),
                State::Failed if holds && !again && g.failed_at.is_some_and(|t| t.elapsed() < RETRY_AFTER) => {
                    return Err(g.error.clone().unwrap_or_else(|| "tinygrad's server failed".into()));
                }
                _ => {}
            }
            // (under the lock the check was made under: no request can take the old model between the two)
            self.halt(&mut g);
        }
        self.changed.notify_all();
        match self.launch(cfg, llm, root, name, epoch, count, paused) {
            Ok(true) => Ok(Begun::Launched),
            Ok(false) => Err(STOPPED.into()),
            Err(e) => {
                let mut g = self.lock();
                if g.epoch == epoch {
                    g.state = State::Failed;
                    g.error = Some(e.clone());
                    g.model = Some(name.to_string());
                    g.failed_at = Some(Instant::now());
                }
                drop(g);
                self.log.push(format!("studio: tinygrad's server was not started: {e}"));
                self.changed.notify_all();
                Err(e)
            }
        }
    }

    /// Whether the server was started (false: someone asked for a stop meanwhile, and it stood down).
    #[allow(clippy::too_many_arguments)]
    fn launch(self: &Arc<Self>, cfg: &Json, llm: &Json, root: &Path, name: &str, epoch: u64, count: bool, paused: bool) -> Result<bool, String> {
        self.wait_gone(GONE_WAIT)?;
        let python = self.python(cfg, root)?;
        let runtime = self.lock().runtime.clone();
        let runtime = runtime.as_ref();
        let script = Self::script(root, runtime)?;
        let port = free_port()?;
        let key = random_key("sk-egpu-");
        let mut args = arguments(llm, root, name, &script, port)?;
        if private(cfg) {
            // (the engine's own arguments, which the launcher passes on: no prompt states on disk, no request log, each
            // request forgotten, as this computer's engine runs in incognito)
            args.push("--incognito".into());
        }
        let paused = paused && webgpu(llm);
        if paused {
            // (before --webgpu, where the launcher reads its own flags)
            let at = args.iter().position(|a| a == "--webgpu").ok_or("the launcher's command line has no --webgpu")?;
            args.insert(at, "--paused".into());
        }
        // (a file, or a folder of weights: an EXL3 checkpoint, which OAIY's engine reads on the card too)
        let file = config::resolve(root, str_or(model(llm, name).unwrap_or(&Json::Null), "path", ""));
        if !file.exists() {
            return Err(format!("{} is not there", file.display()));
        }
        let env = environment(llm, root, runtime);
        let mut command = Command::new(&python);
        command.args(&args).envs(env.iter().map(|(k, v)| (k, v))).env("OAIY_EGPU_KEY", &key);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        command.current_dir(working_dir(llm, root, runtime));
        hide_window(&mut command);
        let device = env.iter().find(|(k, _)| k == "DEV").map_or("NV", |(_, v)| v.as_str());
        let shown = format!("DEV={device} {} {}", python.display(), args.join(" "));
        self.log.push(format!("studio: starting {shown}"));
        let mut child = command.spawn().map_err(|e| {
            // (a Python that was found earlier and is not there now: looked for again at the next start)
            self.lock().python = None;
            format!("{}: {e}", python.display())
        })?;
        let card = Arc::new(Card::default());
        for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten() {
            let log = Arc::clone(&self.log);
            let card = Arc::clone(&card);
            std::thread::spawn(move || {
                // Bytes, not text: a line that is not UTF-8 (a compiler's, a progress bar's) must not end the reading,
                // or the pipe fills and the server blocks on its next word. A line with no end (a bar redrawn in
                // place) is passed on in pieces.
                let (mut reader, mut line) = (BufReader::new(pipe), Vec::new());
                while reader.by_ref().take(16 << 10).read_until(b'\n', &mut line).is_ok_and(|n| n > 0) {
                    let text = String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string();
                    if !card.heard(&text) {
                        log.push(text);
                    }
                    line.clear();
                }
            });
        }
        let mut g = self.lock();
        if g.epoch != epoch {
            drop(g);
            let _ = child.kill();
            let _ = child.wait();
            self.log.push("studio: tinygrad's server was stopped as it started");
            return Ok(false);
        }
        g.generation += 1;
        let generation = g.generation;
        g.lifeline = child.stdin.take();
        g.child = Some(child);
        g.addr = format!("127.0.0.1:{port}");
        g.key = key;
        g.state = State::Starting;
        g.waiting += count as usize;
        g.error = None;
        g.started = Some(Instant::now());
        g.ready_after = None;
        g.model = Some(name.to_string());
        g.failed_at = None;
        g.last_used = Instant::now();
        g.command = shown;
        g.card = card;
        g.lent = false;
        g.paused = paused;
        g.resume_after = false;
        drop(g);
        self.changed.notify_all();
        let this = Arc::clone(self);
        std::thread::Builder::new().name("egpu-watch".into()).spawn(move || this.watch(generation)).map_err(|e| e.to_string())?;
        Ok(true)
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

    /// The server stopped and forgotten, the state's lock held.
    ///
    /// Its standard input closed first, which has it release the card and exit (`egpu_serve.py`'s lifeline), and only
    /// past a wait killed: a server killed outright leaves the card's firmware up, and the next start resets the card,
    /// which over Thunderbolt has taken its link down until it was powered off and on. The wait is a thread's, not
    /// the lock's.
    fn halt(&self, g: &mut Inner) {
        g.generation += 1;
        g.lifeline = None;
        if let Some(mut child) = g.child.take() {
            let (log, going) = (self.log.clone(), Arc::clone(&self.going));
            *going.0.lock().unwrap_or_else(|p| p.into_inner()) += 1;
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(20);
                let mut gone = false;
                while Instant::now() < deadline {
                    if child.try_wait().is_ok_and(|done| done.is_some()) {
                        log.push("studio: tinygrad's server stopped");
                        gone = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                if !gone {
                    let _ = child.kill();
                    let _ = child.wait();
                    log.push("studio: tinygrad's server stopped (killed after 20 s)");
                }
                *going.0.lock().unwrap_or_else(|p| p.into_inner()) -= 1;
                going.1.notify_all();
            });
        }
        g.state = State::Stopped;
        g.error = None;
        g.ready_after = None;
        g.model = None;
        g.failed_at = None;
        g.card = Arc::default();
        g.lent = false;
        g.paused = false;
        g.resume_after = false;
        g.wanted = false;
        g.stop_when_given_back = None;
    }

    /// The card's Python, tinygrad and launcher from a provider (OAIY Desktop's eGPU plugin), in place of the
    /// settings' and of the launcher this build carries; None: the settings' again. It is used from the next start. A
    /// server that is idle on the other is stopped, and its model said, for the caller to start it again on this one;
    /// one that is answering, or lent, goes on as it is.
    pub fn provide(&self, runtime: Option<Runtime>) -> Option<String> {
        let mut g = self.lock();
        if g.runtime == runtime {
            return None;
        }
        self.log.push(match &runtime {
            Some(r) => format!("studio: the eGPU's Python, tinygrad and launcher are {}'s ({})", r.by, r.scripts.display()),
            None => "studio: the eGPU's Python and tinygrad are the settings' again".into(),
        });
        g.runtime = runtime;
        g.python = None;
        // (a start that failed on the other is tried again at once)
        g.failed_at = None;
        let idle = matches!(g.state, State::Starting | State::Ready) && g.busy == 0 && g.waiting == 0 && !g.lent;
        let again = if idle { g.model.clone() } else { None };
        if again.is_some() {
            g.epoch += 1;
            self.halt(&mut g);
        }
        drop(g);
        self.changed.notify_all();
        again
    }

    /// Stop the server because someone asked: Stop, a change of settings, the studio going. A start that is under
    /// way stands down, and requests waiting for the server go to this computer's engine: they do not start it
    /// again.
    pub fn stop(&self) {
        let mut g = self.lock();
        g.epoch += 1;
        self.halt(&mut g);
        drop(g);
        self.changed.notify_all();
    }

    /// [`Self::stop`], unless the card is lent to the image worker: then it is stopped as the card is given back
    /// ([`Self::give_back`]), and false is answered. For a stop the settings ask for (the eGPU's changed, a default
    /// model chosen on this computer): the lent card's server is the one whose socket a pinned picture model's worker
    /// is on, and stopping it took the card from under that worker. (The model is not on the card meanwhile: its
    /// engine is paused while the card is lent.)
    pub fn stop_unless_lent(&self, why: Deferred) -> bool {
        let mut g = self.lock();
        if g.lent {
            // (a change of its launch stays asked for, whatever is asked after it)
            if g.stop_when_given_back != Some(Deferred::Changed) {
                g.stop_when_given_back = Some(why);
            }
            return false;
        }
        g.epoch += 1;
        self.halt(&mut g);
        drop(g);
        self.changed.notify_all();
        true
    }

    /// Stopped for good, as the studio stops (`crate::Studio::shutdown`): the server stopped and no other started on
    /// the card after it (a request still being answered, a picture's lending). [`Self::await_gone`] then waits for it
    /// to let the card go.
    pub fn close(&self) {
        let mut g = self.lock();
        g.closed = true;
        g.epoch += 1;
        self.halt(&mut g);
        drop(g);
        self.changed.notify_all();
    }

    /// Wait (`wait` at most) for the servers stopped to have let the card go, as the studio stops for good:
    /// `egpu_serve.py` takes up to 15 s to, and a stop returns at once. Whether they had. An ordinary quit waits a few
    /// seconds of that (its server lets the card go by itself as OAIY ends); an update's hand-off waits for all of it
    /// ([`await_let_go`]), or the OAIY it started found tinygrad's lock still taken.
    pub fn await_gone(&self, wait: Duration) -> bool {
        match self.wait_gone(wait) {
            Ok(()) => true,
            Err(e) => {
                self.log.push(format!("studio: {e}"));
                false
            }
        }
    }

    /// Stop a ready server that has answered nothing for `idle`, or a card held with its engine paused (since a
    /// picture) for as long, and say whether it was. Decided and done under one lock: a request that comes meanwhile
    /// is counted first, and keeps the server, or finds it stopped.
    pub fn stop_if_idle(&self, idle: Duration) -> bool {
        let mut g = self.lock();
        let held = g.state == State::Ready || (g.paused && matches!(g.state, State::Starting | State::Ready));
        if !held || g.busy > 0 || g.waiting > 0 || g.lent || g.last_used.elapsed() <= idle {
            return false;
        }
        self.halt(&mut g);
        drop(g);
        self.changed.notify_all();
        true
    }

    /// The card lent to OAIY's image worker (`llm.egpu.engine` `webgpu`): the engine on it paused once the requests it
    /// is answering are done, its memory freed, the card still held (opened again, it would be reset). The answer is
    /// the card's socket, for the worker to take as its WebGPU adapter. Where the card is not held, it is opened for
    /// this, the engine paused from the start (with the model set to the eGPU, which loads at the first request for
    /// it). Requests for the model wait until [`Self::give_back`]. `wait`: for the requests being answered.
    pub fn lend(self: &Arc<Self>, cfg: &Json, root: &Path, wait: Duration) -> Result<String, String> {
        self.lend_unless(cfg, root, wait, &std::sync::atomic::AtomicBool::new(false))
    }

    /// [`Self::lend`] for a job that may be cancelled meanwhile (`cancel`, heard each half second): the card given back
    /// and `cancelled` answered, rather than the job waiting minutes for a card it no longer wants (the requests being
    /// answered, the card's firmware booting) to be told then.
    pub fn lend_unless(self: &Arc<Self>, cfg: &Json, root: &Path, wait: Duration, cancel: &std::sync::atomic::AtomicBool) -> Result<String, String> {
        let cancelled = || cancel.load(std::sync::atomic::Ordering::Relaxed);
        let llm = cfg.get("llm").ok_or("no llm section")?;
        if !enabled(llm) || !webgpu(llm) {
            return Err("the eGPU does not run OAIY's engine".into());
        }
        let name = models(llm).into_iter().next().ok_or("no language model is set to run on the eGPU, so the card is not held")?;
        let deadline = Instant::now() + wait;
        let mut g = self.lock();
        let epoch = g.epoch;
        // (whether this lent it, and so gives it back where the job is cancelled: a card already lent, to the picture
        // worker kept on it, stays so)
        let mut lent_here = false;
        loop {
            if g.epoch != epoch {
                return Err(STOPPED.into());
            }
            // (not the card opened, or its engine paused, for a job no longer wanted)
            if cancelled() && !lent_here {
                return Err("cancelled".into());
            }
            if matches!(g.state, State::Starting | State::Ready) {
                if !g.lent {
                    g.lent = true;
                    g.resume_after = !g.paused;
                    lent_here = true;
                }
                if g.busy == 0 {
                    break;
                }
            } else {
                drop(g);
                self.begin(cfg, root, &name, epoch, false, false, true)?;
                g = self.lock();
                continue;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if cancelled() {
                drop(g);
                if lent_here {
                    self.give_back();
                }
                return Err("cancelled".into());
            }
            if left.is_zero() {
                drop(g);
                self.give_back();
                return Err("the eGPU's language model is still answering".into());
            }
            g = self.changed.wait_timeout(g, left.min(Duration::from_millis(500))).unwrap_or_else(|p| p.into_inner()).0;
        }
        // (when this pauses a running engine: a picture's worker started while its model's memory was still being let
        // go thrashed the card, a picture of 18 s taking 128)
        let paused_now = (!g.paused && matches!(g.state, State::Ready)).then(Instant::now);
        if !g.paused {
            g.paused = true;
            if let Some(pipe) = g.lifeline.as_mut() {
                let _ = pipe.write_all(b"pause\n").and_then(|_| pipe.flush());
            }
            self.log.push("studio: the engine on the card pauses for a picture");
        }
        let (card, generation) = (Arc::clone(&g.card), g.generation);
        drop(g);
        // (opening the card takes a while the first time: its firmware boots)
        let gone = || {
            let g = self.lock();
            cancelled() || g.generation != generation || !g.lent || !matches!(g.state, State::Starting | State::Ready)
        };
        match card.paused_at(Duration::from_secs(180), gone) {
            Some(socket) if !cancelled() => {
                if let Some(since) = paused_now {
                    if card.freed_since(since, CARD_FREE_MB, Duration::from_secs(30)) == Some(false) {
                        self.log.push("studio: the card's memory was not let go within 30 s of its engine's pause; the picture starts anyway");
                    }
                }
                Ok(socket)
            }
            _ if cancelled() => {
                if lent_here {
                    self.give_back();
                }
                Err("cancelled".into())
            }
            _ => {
                self.give_back();
                Err("the eGPU's launcher did not say the card was free".into())
            }
        }
    }

    /// The card back from the image worker: the engine on it brought back where it was running when it was lent
    /// (its model loads again), else left paused until a request for its model comes.
    pub fn give_back(&self) {
        let mut g = self.lock();
        if !g.lent {
            return;
        }
        g.lent = false;
        g.wanted = false;
        g.pin_holder = None;
        g.last_used = Instant::now();
        if g.stop_when_given_back.is_some() {
            // (the stop asked for while it was lent: the server goes now, and the next request starts it as the
            // settings are)
            self.log.push("studio: the card is given back: tinygrad's server stops, as the settings asked while it was lent");
            g.epoch += 1;
            self.halt(&mut g);
        } else if g.paused && g.resume_after {
            self.resume(&mut g);
        }
        drop(g);
        self.changed.notify_all();
    }

    /// Whether the card is lent to the image worker now.
    pub fn is_lent(&self) -> bool {
        self.lock().lent
    }

    /// The pinned picture model whose worker keeps the lent card, or none (the media runner says, as it keeps it).
    pub fn set_pin_holder(&self, model: Option<String>) {
        self.lock().pin_holder = model;
    }

    /// Whether a request waits for the lent card (its model's): the image worker is to give it back once its job is done.
    pub fn wanted(&self) -> bool {
        let g = self.lock();
        g.lent && g.wanted
    }

    /// The engine on the card asked back from a pause, the state's lock held: starting again, its model loading.
    fn resume(&self, g: &mut Inner) {
        g.paused = false;
        g.resume_after = false;
        if let Some(pipe) = g.lifeline.as_mut() {
            let _ = pipe.write_all(b"resume\n").and_then(|_| pipe.flush());
        }
        g.state = State::Starting;
        g.started = Some(Instant::now());
        g.ready_after = None;
        self.log.push("studio: the engine on the card starts again");
    }

    /// Where `name` is answered on the eGPU, starting tinygrad's server with it if it is not held, and waiting up
    /// to `timeout` for it to load. The lease is held for as long as the request is being answered.
    pub fn ensure(self: &Arc<Self>, cfg: &Json, root: &Path, name: &str, timeout: Duration) -> Result<(Endpoint, Lease), Unready> {
        let deadline = Instant::now() + timeout;
        let mut g = self.lock();
        let epoch = g.epoch;
        // (whether this request is counted among those the loading model is awaited by: for as long as it sleeps)
        let mut counted = false;
        loop {
            if std::mem::take(&mut counted) {
                g.waiting = g.waiting.saturating_sub(1);
            }
            if g.epoch != epoch {
                return Err(Unready::Gone(STOPPED.into()));
            }
            let holds = g.model.as_deref() == Some(name);
            // While the card is lent to the image worker, requests wait for it as for a model loading; one for the
            // model of an engine that is paused brings it back.
            if g.paused && !g.lent && holds && matches!(g.state, State::Starting | State::Ready) {
                self.resume(&mut g);
            }
            // (one waiting for the lent card says so: the image worker keeps it no longer than its job, unless a pinned
            // model's worker keeps it, which is said at once)
            if g.lent {
                if let Some(pin) = g.pin_holder.clone() {
                    return Err(Unready::Gone(format!("the eGPU's card keeps the pinned picture model {pin} loaded: unpin it to use the card's language model")));
                }
                g.wanted = true;
            }
            if !g.lent { match g.state {
                State::Ready if holds => {
                    g.busy += 1;
                    g.last_used = Instant::now();
                    let endpoint = Endpoint { addr: g.addr.clone(), key: g.key.clone() };
                    drop(g);
                    return Ok((endpoint, Lease(Arc::clone(self))));
                }
                State::Starting if holds => {
                    g.waiting += 1;
                    counted = true;
                }
                State::Failed if holds && g.failed_at.is_some_and(|t| t.elapsed() < RETRY_AFTER) => {
                    return Err(Unready::Gone(g.error.clone().unwrap_or_else(|| "tinygrad's server failed".into())));
                }
                // Another model's requests are being answered, or waiting for it to load: it is not cut off.
                State::Ready | State::Starting if g.busy > 0 || g.waiting > 0 => {}
                _ => {
                    drop(g);
                    // (in use after all, taken between this check and the start's: the next turn waits for it)
                    let begun = self.begin(cfg, root, name, epoch, false, true, false).map_err(Unready::Gone)?;
                    g = self.lock();
                    counted = matches!(begun, Begun::Launched);
                    continue;
                }
            } }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                if counted {
                    g.waiting = g.waiting.saturating_sub(1);
                }
                return Err(Unready::Loading(if g.lent {
                    "the eGPU is making a picture, a video or a sound; try again shortly".into()
                } else if holds {
                    format!("{name} is still loading on the eGPU; try again shortly")
                } else {
                    "the eGPU is answering with another model; try again shortly".into()
                }));
            }
            g = self.changed.wait_timeout(g, left.min(Duration::from_millis(500))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// Whether the server a request was sent to (`endpoint`, as [`Self::ensure`] gave it) is still the one serving:
    /// ready, its process there, not stopped or started afresh since.
    pub fn serves(&self, endpoint: &Endpoint) -> bool {
        let mut g = self.lock();
        g.state == State::Ready && g.addr == endpoint.addr && g.key == endpoint.key && g.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)))
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

    /// The context the card's oaiy-llm-server said it opened its model with (`crate::llm::context_said`): none from
    /// tinygrad's server, which says none.
    fn loaded_context(&self, llm: &Json) -> Option<i64> {
        webgpu(llm).then(|| crate::llm::context_said(&self.log.tail(400))).flatten()
    }

    /// "… on NV", as tinygrad's server reported where the model's weights are.
    fn runs_on(&self) -> Option<String> {
        let tail = self.log.tail(400);
        tail.lines().rev().find_map(|l| l.starts_with("loaded model").then(|| l.rsplit_once(" on ").map(|(_, on)| on.trim().to_string())).flatten())
    }

    /// The card's memory and load as its launcher last said them (`{memory_total_mb, memory_used_mb, memory_cached_mb,
    /// load}`), or null where it has not said lately.
    pub fn card_usage(&self) -> Json {
        let card = Arc::clone(&self.lock().card);
        card.usage()
    }

    /// Wait, up to `wait`, for the card's memory to be let go after the image worker that had it has gone (`since`):
    /// the launcher frees a departed worker's buffers once it notices, and the language model brought back before then
    /// ran out of memory loading and its server ended (a flow's picture, then its review). True once it is free, or
    /// where the launcher says nothing of its memory (then after a moment's grace).
    pub fn await_freed(&self, since: Instant, wait: Duration) -> bool {
        let card = Arc::clone(&self.lock().card);
        match card.freed_since(since, CARD_FREE_MB, wait) {
            Some(freed) => {
                if !freed {
                    self.log.push(format!("studio: the card's memory was not let go within {} s; its language model loads anyway", wait.as_secs()));
                }
                freed
            }
            None => {
                std::thread::sleep(Duration::from_secs(2));
                true
            }
        }
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
            // the default model is set to it (and it is on): "the language model" is then its server's, for the page's
            // own card of that name (this computer's engine holds none ahead of time; its other models still run here)
            ("holds_all", Json::Bool(!crate::llm::holds_a_model(llm))),
            ("device", Json::str(match str_or(egpu, "device", "NV").trim() { "" => "NV", d => d })),
            // (the engine's own once it has loaded the model: the card may hold the cache for fewer positions than asked)
            ("context_tokens", Json::Int(self.loaded_context(llm).filter(|_| g.state == State::Ready).unwrap_or_else(|| g.model.as_deref().map_or_else(|| context(llm), |m| model_context(llm, m))))),
            ("busy", Json::Int(g.busy as i64)),
            ("load_seconds", g.ready_after.map_or(Json::Null, Json::Num)),
            ("idle_seconds", Json::Int(g.last_used.elapsed().as_secs() as i64)),
            ("retry_seconds", retry.map_or(Json::Null, Json::Int)),
            ("command", Json::str(&g.command)),
            ("python", g.python.as_ref().map_or(Json::Null, |(_, p)| Json::str(p.to_string_lossy()))),
            ("provided_by", g.runtime.as_ref().map_or(Json::Null, |r| Json::str(&r.by))),
            ("runs_on", self.runs_on().filter(|_| g.state == State::Ready).map_or(Json::Null, Json::str)),
            ("engine", Json::str(if webgpu(llm) { "webgpu" } else { "tinygrad" })),
            // the card's memory and load, as the launcher of OAIY's engine on it says them (none from tinygrad's server)
            ("usage", g.card.usage()),
            // the card lent to the image worker, and the engine on it paused (for that, or since)
            ("lent", Json::Bool(g.lent)),
            ("paused", Json::Bool(g.paused)),
        ])
    }
}

impl Drop for Egpu {
    fn drop(&mut self) {
        self.stop();
    }
}

/// What a found Python stands for: the settings it was looked for under, or the provider's.
fn python_key(llm: &Json, runtime: Option<&Runtime>) -> String {
    let egpu = section(llm);
    match runtime {
        Some(r) => format!("provided|{}", r.python.display()),
        None => format!("{}|{}", str_or(egpu, "python", ""), str_or(egpu, "tinygrad", "")),
    }
}

/// Where the server is started: the tinygrad folder the settings name (tinygrad's own instructions run it from
/// there), else the configuration's folder. A provider's runtime needs no folder of its own.
fn working_dir(llm: &Json, root: &Path, runtime: Option<&Runtime>) -> PathBuf {
    runtime.is_none().then(|| tinygrad_folder(llm, root)).flatten().filter(|d| d.is_dir()).unwrap_or_else(|| root.to_path_buf())
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
    fn the_context_said_is_the_card_engines_own_once_it_has_loaded() {
        let egpu = Egpu::new();
        let cfg = llm(&TWO.replace(r#""ctx": 16384,"#, r#""ctx": 16384, "engine": "webgpu","#).replace(r#""path": "models/big.gguf", "egpu": true"#, r#""path": "models/big.gguf", "egpu": true, "egpu_ctx": 131072"#));
        let said = |egpu: &Egpu, llm: &Json| egpu.status(llm).get("context_tokens").and_then(Json::as_i64);
        egpu.log.push("oaiy-llm-server: small loaded in 3.0s (16384 token context)");
        egpu.lock().model = Some("big".into());
        assert_eq!(said(&egpu, &cfg), Some(131072), "loading: the model's setting, not the last engine's");
        egpu.adopt("127.0.0.1:1", "k", "big", None);
        egpu.log.push("m27: the GPU holds its attention cache for 81920 positions beside its weights and a prompt's work");
        egpu.log.push("oaiy-llm-server: big loaded in 35.3s (81920 token context)");
        assert_eq!(said(&egpu, &cfg), Some(81920), "loaded: what the card holds");
        assert_eq!(said(&egpu, &llm(TWO)), Some(16384), "tinygrad's server says none: the setting");
    }

    #[test]
    fn a_start_waits_for_the_last_server_to_let_the_card_go() {
        let egpu = Egpu::new();
        assert!(egpu.wait_gone(Duration::from_millis(10)).is_ok(), "nothing stopped: no wait");
        // a server stopped, still letting the card go
        *egpu.going.0.lock().unwrap() += 1;
        let going = Arc::clone(&egpu.going);
        let started = Instant::now();
        let gone = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            *going.0.lock().unwrap() -= 1;
            going.1.notify_all();
        });
        assert!(egpu.wait_gone(Duration::from_secs(5)).is_ok());
        assert!(started.elapsed() >= Duration::from_millis(150), "the start went before the card was let go");
        gone.join().unwrap();
        // one that never goes: the start gives up, and says why
        *egpu.going.0.lock().unwrap() += 1;
        let err = egpu.wait_gone(Duration::from_millis(50)).unwrap_err();
        assert!(err.contains("not let the card go"), "{err}");
    }

    /// The count of servers stopped and not gone is the process's, not one Egpu's: the engines restarted in OAIY
    /// Desktop's own process are a new studio with a new Egpu, which waited for none of the old one's servers. The
    /// process's own count, as a studio's Egpu has it outside the tests (the only test that counts on it).
    #[test]
    fn a_new_egpu_waits_for_the_servers_another_stopped() {
        let (old, new) = (Egpu::counting_on(stopping_servers()), Egpu::counting_on(stopping_servers()));
        assert!(Arc::ptr_eq(&old.going, &new.going), "one count for the process");
        old.adopt_watching("127.0.0.1:1", "k", "big");
        old.stop();
        assert!(new.wait_gone(Duration::from_secs(10)).is_ok());
        assert!(old.log.tail(10).contains("tinygrad's server stopped"), "the new one's start waited for the old one's server: {}", old.log.tail(10));
        // and the update's wait, which no studio is left to ask, waits on the same
        old.adopt_watching("127.0.0.1:1", "k", "big");
        old.stop();
        assert!(await_let_go(Duration::from_secs(10)));
        assert!(old.log.tail(10).matches("tinygrad's server stopped").count() >= 2, "{}", old.log.tail(10));
    }

    /// Stopped for good, as the studio stops: its server let go of the card before the stop returns (an update's
    /// hand-off came first, and the OAIY it started found tinygrad's lock taken), and none is started after it.
    #[test]
    fn closed_it_waits_for_the_card_to_be_let_go_and_starts_nothing() {
        let egpu = Arc::new(Egpu::new());
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        egpu.close();
        assert!(egpu.await_gone(GONE_WAIT));
        assert!(egpu.log.tail(10).contains("tinygrad's server stopped"), "{}", egpu.log.tail(10));
        let cfg = Json::parse(br#"{"llm": {"egpu": {"enabled": true}, "models": [{"name": "m", "path": "m.gguf", "egpu": true}]}}"#).unwrap();
        let err = egpu.start(&cfg, Path::new("/install"), "m").unwrap_err();
        assert!(err.contains("the studio is stopping"), "{err}");
        assert_eq!(egpu.state(), State::Stopped);
    }

    /// A stop the settings ask for while the card is lent (a pinned picture model's worker on its socket) is done as
    /// the card is given back: choosing a default model on this computer stopped the card's server under the picture.
    #[test]
    fn a_stop_asked_while_the_card_is_lent_waits_for_it_to_be_given_back() {
        let egpu = Egpu::new();
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        egpu.lent_for_test();
        assert!(!egpu.stop_unless_lent(Deferred::Unwanted), "not stopped under the lent card");
        assert_eq!(egpu.state(), State::Ready);
        assert!(egpu.is_lent());
        egpu.give_back();
        assert_eq!(egpu.state(), State::Stopped, "stopped as it is given back");
        assert!(egpu.await_gone(GONE_WAIT));
        // Not lent: stopped at once, as ever.
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        assert!(egpu.stop_unless_lent(Deferred::Changed));
        assert_eq!(egpu.state(), State::Stopped);
        // And a card lent and given back with no stop asked for keeps its server.
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        egpu.lent_for_test();
        egpu.give_back();
        assert_eq!(egpu.state(), State::Ready);
        egpu.stop();
        assert!(egpu.await_gone(GONE_WAIT));
    }

    /// A stop deferred while the card is lent is decided again by what is asked after it: the card's model chosen as
    /// the default again (a start for the model it holds) keeps the server as the card is given back, where the
    /// default chosen on this computer meanwhile had asked for it to stop. A change of its launch stops it whatever.
    #[test]
    fn a_stop_deferred_while_lent_is_not_done_once_the_card_s_model_is_wanted_again() {
        let egpu = Arc::new(Egpu::new());
        let cfg = Json::parse(br#"{"llm": {"egpu": {"enabled": true, "engine": "webgpu"}, "models": [{"name": "m", "path": "m.gguf", "egpu": true}]}}"#).unwrap();
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        egpu.lent_for_test();
        // a picture on the lent card; a default chosen on this computer; the card's model chosen again
        assert!(!egpu.stop_unless_lent(Deferred::Unwanted));
        egpu.start(&cfg, Path::new("/install"), "m").unwrap();
        egpu.give_back();
        assert_eq!(egpu.state(), State::Ready, "kept: its model is the default again");
        // A change of its launch, then the same start: stopped as it is given back all the same.
        egpu.lent_for_test();
        assert!(!egpu.stop_unless_lent(Deferred::Changed));
        assert!(!egpu.stop_unless_lent(Deferred::Unwanted), "a later stop does not make the change one a start undoes");
        egpu.start(&cfg, Path::new("/install"), "m").unwrap();
        egpu.give_back();
        assert_eq!(egpu.state(), State::Stopped);
        assert!(egpu.await_gone(GONE_WAIT));
    }

    /// An engine on the card paused since a picture (the card given back, not lent) is started by Start: its model
    /// loads again now, not only at the next chat.
    #[test]
    fn a_paused_engine_is_started_again_by_start() {
        let egpu = Arc::new(Egpu::new());
        let cfg = Json::parse(br#"{"llm": {"egpu": {"enabled": true, "engine": "webgpu"}, "models": [{"name": "m", "path": "m.gguf", "egpu": true}]}}"#).unwrap();
        egpu.adopt_watching("127.0.0.1:1", "k", "m");
        egpu.lock().paused = true;
        egpu.start(&cfg, Path::new("/install"), "m").unwrap();
        let g = egpu.lock();
        assert!(!g.paused && g.state == State::Starting, "resumed, its model loading");
        drop(g);
        egpu.stop();
        assert!(egpu.await_gone(GONE_WAIT));
    }

    fn provided(scripts: &Path) -> Runtime {
        Runtime { python: "/plugins/egpu/python/bin/python3.12".into(), pythonpath: "/plugins/egpu/lib".into(), scripts: scripts.into(), by: "eGPU 0.1.0".into() }
    }

    #[test]
    fn a_provided_runtime_is_the_python_the_path_and_the_launcher_the_card_is_run_with() {
        let dir = std::env::temp_dir().join(format!("oaiy-egpu-provided-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let runtime = provided(&dir);
        let settings = llm(r#"{"egpu": {"python": "/usr/bin/python3", "tinygrad": "/src/tinygrad"}}"#);
        assert_eq!(pythons(&settings, Path::new("/data"), Some(&runtime)), [runtime.python.clone()], "the provider's Python and no other");
        let env = environment(&settings, Path::new("/data"), Some(&runtime));
        assert!(env.iter().any(|(k, v)| k == "PYTHONPATH" && v.replace('\\', "/").starts_with("/plugins/egpu/lib")), "{env:?}");
        assert!(!env.iter().any(|(_, v)| v.contains("/src/tinygrad")), "not the settings' tinygrad: {env:?}");
        assert!(env.contains(&("PYTHONDONTWRITEBYTECODE".into(), "1".into())), "a plugin's folder is not written to");
        assert!(!environment(&settings, Path::new("/data"), None).iter().any(|(k, _)| k == "PYTHONDONTWRITEBYTECODE"));
        assert_eq!(working_dir(&settings, Path::new("/data"), Some(&runtime)), Path::new("/data"));
        assert_ne!(python_key(&settings, Some(&runtime)), python_key(&settings, None), "a Python found for the settings is not the provider's");
        // its own launcher, not this build's copy; and none there is said
        let err = Egpu::script(Path::new("/data"), Some(&runtime)).unwrap_err();
        assert!(err.contains("egpu_serve.py") && err.contains("eGPU 0.1.0"), "{err}");
        std::fs::write(dir.join("egpu_serve.py"), "").unwrap();
        assert_eq!(Egpu::script(Path::new("/data"), Some(&runtime)).unwrap(), dir.join("egpu_serve.py"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_provider_stops_an_idle_server_for_its_model_to_be_started_again_and_leaves_a_busy_one() {
        let egpu = Egpu::new();
        let runtime = provided(Path::new("/plugins/egpu/card"));
        assert_eq!(egpu.provide(Some(runtime.clone())), None, "nothing running: nothing to start again");
        assert_eq!(egpu.provide(Some(runtime.clone())), None, "the same again changes nothing");
        assert_eq!(egpu.status(&Json::Null).get("provided_by").and_then(Json::as_str), Some("eGPU 0.1.0"));
        egpu.adopt("127.0.0.1:1", "k", "big", None);
        egpu.lock().busy = 1;
        assert_eq!(egpu.provide(None), None, "a server answering a request goes on");
        assert_eq!(egpu.lock().state, State::Ready);
        assert!(egpu.lock().runtime.is_none(), "and the next start uses the settings'");
        egpu.lock().busy = 0;
        let epoch = egpu.lock().epoch;
        assert_eq!(egpu.provide(Some(runtime)).as_deref(), Some("big"), "an idle one is stopped, and its model said");
        let g = egpu.lock();
        assert_eq!((g.state, g.epoch), (State::Stopped, epoch + 1), "a start under way stands down");
    }

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
    fn oaiys_own_engine_is_started_on_the_card_through_the_launcher_with_this_model_alone() {
        let text = TWO.replace(r#""egpu": {"enabled": true,"#, r#""devices": [1], "egpu": {"enabled": true, "engine": "webgpu","#);
        let on = llm(&text);
        assert!(webgpu(&on) && !webgpu(&llm(TWO)));
        let root = Path::new("/data/engines");
        let script = root.join("cache").join("egpu").join("egpu_serve.py");
        let a = arguments(&on, root, "big", &script, 18000).unwrap();
        assert_eq!(a[..4], ["-u", script.to_string_lossy().as_ref(), "--watch-stdin", "--webgpu"]);
        assert!(a[4].ends_with("oaiy-llm-server") || a[4].ends_with("oaiy-llm-server-webgpu"), "{}", a[4]);
        let after = |flag: &str| a.iter().position(|x| x == flag).map(|i| a[i + 1].clone());
        assert_eq!(after("--model"), Some(root.join("models/big.gguf").to_string_lossy().into_owned()));
        assert_eq!((after("--name").as_deref(), after("--port").as_deref(), after("--ctx").as_deref()), (Some("big"), Some("18000"), Some("16384")));
        assert_eq!(after("--prompt-cache"), Some(root.join("cache").join("prompt-states-egpu").to_string_lossy().into_owned()));
        // (DeepSeek's experts' counts of uses: the card's engine's own, not this computer's engine's)
        assert_eq!(after("--usage"), Some(root.join("cache").join("expert-usage-egpu.bin").to_string_lossy().into_owned()));
        assert_eq!(a.iter().filter(|x| *x == "--usage").count(), 1);
        // This model alone, the key out of sight, and nothing of this computer's GPUs or tinygrad's own arguments.
        for absent in ["--also", "--api-key", "--devices", "--devices-for", "--shard", "--serve", "--max_context"] {
            assert!(!a.contains(&absent.to_string()), "{absent} in {a:?}");
        }
        assert_eq!(a.iter().filter(|x| *x == "--watch-stdin").count(), 1, "the launcher's own");
        // On the card or not at all: never this computer's CPU under the eGPU's name.
        assert_eq!(after("--backend").as_deref(), Some("webgpu"));
        assert_eq!(a.iter().filter(|x| *x == "--backend").count(), 1);
        // Any model the engine reads, not only a GGUF file.
        assert!(arguments(&on, root, "folder", &script, 1).is_ok());
        // A model's own context on the card, where it has one; the eGPU's for the others.
        let own = llm(&text.replace(r#"{"name": "folder", "path": "models/exl3", "egpu": true}"#, r#"{"name": "folder", "path": "models/exl3", "egpu": true, "egpu_ctx": 262144}"#));
        let ctx_of = |name: &str| {
            let a = arguments(&own, root, name, &script, 1).unwrap();
            a.iter().position(|x| x == "--ctx").map(|i| a[i + 1].clone())
        };
        assert_eq!((ctx_of("folder").as_deref(), ctx_of("big").as_deref()), (Some("262144"), Some("16384")));
        assert_eq!((model_context(&own, "folder"), model_context(&own, "big"), model_context(&own, "nobody")), (262144, 16384, 16384));
    }

    #[test]
    fn the_servers_environment_names_the_device_and_what_the_settings_add() {
        let env = environment(&llm(TWO), Path::new("/data"), None);
        let get = |k: &str| env.iter().find(|(have, _)| have == k).map(|(_, v)| v.as_str());
        assert_eq!(get("DEV"), Some("AMD"), "the settings' own value wins over the device");
        assert_eq!(get("JITBEAM"), Some("2"));
        assert_eq!(get("PYTHONUNBUFFERED"), Some("1"));
        assert_eq!(get("PYTHONPATH"), None, "no tinygrad folder named");
        assert_eq!(env.iter().filter(|(k, _)| k == "DEV").count(), 1);
        // An NVIDIA card unless told, and a checkout's folder on Python's path.
        let env = environment(&llm(r#"{"egpu": {"tinygrad": "/src/tinygrad"}}"#), Path::new("/data"), None);
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
    fn a_named_python_that_is_a_bare_name_is_the_paths_to_find() {
        let root = Path::new("/conf");
        let named = |python: &str| pythons(&Json::parse(format!(r#"{{"egpu": {{"python": "{python}"}}}}"#).as_bytes()).unwrap(), root, None);
        assert_eq!(named("python3.12"), [PathBuf::from("python3.12")], "not a file beside the configuration");
        assert_eq!(named("bin/python3"), [config::resolve(root, "bin/python3")]);
    }

    #[test]
    fn a_named_python_is_the_only_one_asked() {
        let named = llm(r#"{"egpu": {"python": "/opt/venv/bin/python3", "tinygrad": "/src/tinygrad"}}"#);
        assert_eq!(pythons(&named, Path::new("/data"), None).len(), 1);
        // None named: whatever is installed, and the PATH's own last.
        let found = pythons(&llm("{}"), Path::new("/data"), None);
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

    /// At launch the card's model waits for the provider (or `wait`), and stands down for anything that started or
    /// stopped the card meanwhile.
    #[test]
    fn a_start_at_launch_waits_for_the_provider_and_gives_way_to_a_start_or_stop_meanwhile() {
        let asked = |egpu: &Arc<Egpu>, wait: Duration| {
            let (egpu, started) = (Arc::clone(egpu), Instant::now());
            std::thread::spawn(move || {
                let mut when = None;
                let r = egpu.start_when_provided(wait, || {
                    when = Some(started.elapsed());
                    None
                });
                (r, when)
            })
        };
        // The provider says within the wait: asked as it says, not at the end of the wait.
        let egpu = Arc::new(Egpu::new());
        let waiting = asked(&egpu, Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(150));
        egpu.provide(Some(provided(Path::new("/plugins/egpu/card"))));
        let (r, when) = waiting.join().unwrap();
        assert_eq!(r, Ok(None));
        assert!(when.is_some_and(|w| w >= Duration::from_millis(150) && w < Duration::from_secs(5)), "{when:?}");
        // None says: the settings' at the end of the wait.
        let egpu = Arc::new(Egpu::new());
        let (_, when) = asked(&egpu, Duration::from_millis(100)).join().unwrap();
        assert!(when.is_some_and(|w| w >= Duration::from_millis(100)), "{when:?}");
        // Stopped by hand while it waited: not started, not asked.
        let waiting = asked(&egpu, Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(50));
        egpu.stop();
        assert_eq!(waiting.join().unwrap(), (Ok(None), None));
        // A request started it meanwhile: left to it.
        let egpu = Arc::new(Egpu::new());
        let waiting = asked(&egpu, Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(50));
        egpu.adopt("127.0.0.1:1", "k", "big", None);
        egpu.changed.notify_all();
        assert_eq!(waiting.join().unwrap(), (Ok(None), None));
        // Wanted still: started, and a start that fails says why.
        let dir = std::env::temp_dir().join(format!("oaiy-egpu-launch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), b"GGUF").unwrap();
        let cfg = llm(r#"{"llm": {"egpu": {"enabled": true, "python": "no-such-python-here"}, "models": [{"name": "m", "path": "m.gguf", "egpu": true}]}}"#);
        let egpu = Arc::new(Egpu::new());
        let err = egpu.start_when_provided(Duration::from_millis(10), || Some((cfg, dir.clone(), "m".into()))).unwrap_err();
        assert!(err.contains("no Python here has tinygrad"), "{err}");
        assert_eq!((egpu.state(), egpu.held().as_deref()), (State::Failed, Some("m")));
        let _ = std::fs::remove_dir_all(&dir);
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

    /// A card lent to a picture is not taken from under it by a start for another model; a job cancelled before the
    /// card is lent opens nothing, one cancelled while the card's model answers gives the card back, and one cancelled
    /// where the card was lent already (to the picture worker kept there) leaves it so.
    #[test]
    fn a_lent_card_is_kept_from_a_start_and_a_cancelled_job_lets_it_be() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let cfg = llm(r#"{"llm": {"egpu": {"enabled": true, "engine": "webgpu"}, "models": [{"name": "a", "path": "a.gguf", "egpu": true}, {"name": "b", "path": "b.gguf", "egpu": true}]}}"#);
        let root = std::env::temp_dir().join(format!("oaiy-egpu-lent-{}", std::process::id()));
        let egpu = Arc::new(Egpu::new());
        egpu.adopt("127.0.0.1:1", "k", "a", None);
        egpu.lock().lent = true;
        let err = egpu.start(&cfg, &root, "b").unwrap_err();
        assert!(err.contains("lent"), "{err}");
        assert_eq!((egpu.held().as_deref(), egpu.state(), egpu.is_lent()), (Some("a"), State::Ready, true), "not stopped");
        assert!(egpu.start(&cfg, &root, "a").is_ok(), "its own model: held");
        // the card already lent (its kept worker's): a cancelled job leaves it lent
        let cancel = AtomicBool::new(true);
        assert_eq!(egpu.lend_unless(&cfg, &root, Duration::from_secs(5), &cancel).unwrap_err(), "cancelled");
        assert!(egpu.is_lent());
        // not held: nothing opened for a cancelled job
        egpu.stop();
        assert_eq!(egpu.lend_unless(&cfg, &root, Duration::from_secs(5), &cancel).unwrap_err(), "cancelled");
        assert_eq!(egpu.state(), State::Stopped);
        // cancelled as it waits for the card's model to finish a reply: the card is given back, at once
        egpu.adopt("127.0.0.1:1", "k", "a", None);
        egpu.lock().busy = 1;
        let cancel = Arc::new(AtomicBool::new(false));
        let later = Arc::clone(&cancel);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            later.store(true, Ordering::Relaxed);
        });
        let asked = Instant::now();
        assert_eq!(egpu.lend_unless(&cfg, &root, Duration::from_secs(60), &cancel).unwrap_err(), "cancelled");
        assert!(asked.elapsed() < Duration::from_secs(2), "{:?}", asked.elapsed());
        assert!(!egpu.is_lent());
    }

    #[test]
    fn the_cards_engine_is_incognito_as_this_computers_is_and_tinygrad_is_left_as_it_is() {
        let with = |engine: &str, incognito: bool| llm(&format!(r#"{{"llm": {{"egpu": {{"enabled": true, "engine": "{engine}"}}}}, "privacy": {{"incognito": {incognito}}}}}"#));
        assert!(private(&with("webgpu", true)));
        assert!(!private(&with("webgpu", false)));
        assert!(!private(&with("tinygrad", true)), "tinygrad's server has no --incognito");
        assert!(!private(&llm(r#"{"llm": {"egpu": {"enabled": true, "engine": "webgpu"}}}"#)));
    }

    /// The launcher itself with a card on the CPU (`OAIY_EGPU_EMULATE`) and a stand-in for the engine that writes its
    /// pid down and stays; `heeds` its standard input closing as OAIY's engine does (--watch-stdin), or not at all.
    #[cfg(unix)]
    struct Launched {
        dir: PathBuf,
        python: &'static str,
    }

    #[cfg(unix)]
    impl Launched {
        fn new(name: &str, heeds: bool) -> Option<Launched> {
            use std::os::unix::fs::PermissionsExt;
            use std::process::{Command, Stdio};
            let runs = |p: &&str| Command::new(p).arg("--version").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success());
            let python = ["python3", "python"].into_iter().find(runs)?;
            let dir = std::env::temp_dir().join(format!("oaiy-egpu-launched-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("egpu_serve.py"), SERVE_PY).unwrap();
            std::fs::write(dir.join("webgpu_server.py"), CARD_PY).unwrap();
            let heed = if heeds { "import threading\ndef gone():\n    while os.read(0, 256): pass\n    os._exit(0)\nif '--watch-stdin' in sys.argv: threading.Thread(target=gone, daemon=True).start()\n" } else { "" };
            let engine = format!("#!/usr/bin/env python3\nimport os, sys, time\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pids'), 'a').write(str(os.getpid()) + '\\n')\n{heed}time.sleep(120)\n");
            std::fs::write(dir.join("fake-engine"), engine).unwrap();
            std::fs::set_permissions(dir.join("fake-engine"), std::fs::Permissions::from_mode(0o755)).unwrap();
            Some(Launched { dir, python })
        }

        /// The launcher started (`paused`: holding the card, its engine not started), once it has said `until`.
        fn start(&self, paused: bool, until: &str) -> std::process::Child {
            self.start_saying(paused, until).0
        }

        /// As `start`, with the line that said `until`.
        fn start_saying(&self, paused: bool, until: &str) -> (std::process::Child, String) {
            use std::io::BufRead;
            use std::process::{Command, Stdio};
            let _ = std::fs::remove_file(self.dir.join("pids"));
            let mut command = Command::new(self.python);
            command.arg("-u").arg(self.dir.join("egpu_serve.py")).arg("--watch-stdin");
            if paused {
                command.arg("--paused");
            }
            command.arg("--webgpu").arg(self.dir.join("fake-engine")).args(["--port", "1"]);
            command.env("OAIY_EGPU_KEY", "k").env("OAIY_EGPU_EMULATE", "1").env("PYTHONDONTWRITEBYTECODE", "1").env("OAIY_EGPU_STATS", "1");
            let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
            let mut said = std::io::BufReader::new(child.stdout.take().unwrap());
            let mut line = String::new();
            while said.read_line(&mut line).is_ok_and(|n| n > 0) && !line.contains(until) {
                line.clear();
            }
            assert!(line.contains(until), "the launcher never said {until:?}");
            // (what it says after is still read, as OAIY reads it: with nobody reading, its next line would end it by
            // a broken pipe, and not by what the test is of)
            std::thread::spawn(move || {
                let mut rest = String::new();
                while said.read_line(&mut rest).is_ok_and(|n| n > 0) {
                    rest.clear();
                }
            });
            (child, line.trim_end().to_string())
        }

        /// The engines started since `start` that are still running: each is ended here, and counted.
        fn engines_left(&self) -> usize {
            // (running: one that has ended and waits to be reaped, where nothing reaps an orphan at once, is not)
            let alive = |pid: &&str| {
                let state = std::process::Command::new("ps").args(["-o", "stat=", "-p", pid]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
                !state.is_empty() && !state.starts_with('Z')
            };
            let pids = std::fs::read_to_string(self.dir.join("pids")).unwrap_or_default();
            let left: Vec<&str> = pids.split_whitespace().filter(alive).collect();
            for pid in &left {
                let _ = std::process::Command::new("kill").args(["-9", pid]).status();
            }
            left.len()
        }
    }

    #[cfg(unix)]
    impl Drop for Launched {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// OAIY asking for another model as a picture's pause ended: `resume`, and the launcher's standard input closed at
    /// once. The interrupt that ends the launcher then came while it was starting the engine, after the engine was
    /// running and before the launcher knew of it: nothing stopped that engine, which ran the model on this computer's
    /// CPU for as long as the Mac was up. The engine here does not heed its own standard input, so only the launcher
    /// can end it. (Where the interrupt lands is the scheduler's: of 30 such starts, 9 left an engine before the
    /// ending was held, and 7 were held after.)
    #[cfg(unix)]
    #[test]
    fn an_engine_started_as_its_launcher_is_told_to_end_is_ended_with_it() {
        use std::io::Write;
        let Some(launched) = Launched::new("ending", false) else { return };
        for round in 0..10 {
            let mut child = launched.start(true, "oaiy-egpu: paused");
            let mut pipe = child.stdin.take().unwrap();
            pipe.write_all(b"resume\n").unwrap();
            drop(pipe);
            child.wait().unwrap();
            std::thread::sleep(Duration::from_millis(300));
            assert_eq!(launched.engines_left(), 0, "round {round}: an engine outlived its launcher");
        }
    }

    /// The card is free for its language model once the launcher, after the worker has gone, says its memory in use
    /// (less tinygrad's cached buffers) is under the limit; a word from before then does not count.
    #[test]
    fn the_card_is_free_once_the_launcher_says_the_workers_memory_is_let_go() {
        let card = Card::default();
        let short = Duration::from_millis(300);
        // nothing said of its memory: nothing to go on
        assert_eq!(card.freed_since(Instant::now(), 2048, short), None);
        card.heard(r#"oaiy-egpu: usage {"total": 25757220864, "used": 1073741824, "cached": 0, "load": 0}"#);
        let gone = Instant::now();
        // low, but said before the worker went
        assert_eq!(card.freed_since(gone, 2048, short), Some(false));
        // the worker's 11 GB still held (1 GB of it tinygrad's cache)
        card.heard(r#"oaiy-egpu: usage {"total": 25757220864, "used": 12884901888, "cached": 1073741824, "load": 40}"#);
        assert_eq!(card.freed_since(gone, 2048, short), Some(false));
        // let go: 3 GB in use, 2 GB of it cache
        card.heard(r#"oaiy-egpu: usage {"total": 25757220864, "used": 3221225472, "cached": 2147483648, "load": 0}"#);
        assert_eq!(card.freed_since(gone, 2048, short), Some(true));
    }

    /// The launcher's word on the card's memory and load, each second, is the status's (in MB and percent) for as long
    /// as it is fresh, and kept out of the log; its other words are heard as before.
    #[test]
    fn the_cards_memory_and_load_are_heard_and_kept_out_of_the_log() {
        let card = Card::default();
        assert_eq!(card.usage(), Json::Null);
        assert!(!card.heard("oaiy-egpu: card at /tmp/x/card.sock") && !card.heard("oaiy-egpu: paused") && !card.heard("tinygpu-webgpu: ready"));
        let said = card.said.lock().unwrap();
        assert_eq!((said.socket.as_deref(), said.paused), (Some("/tmp/x/card.sock"), true));
        drop(said);
        assert!(card.heard(r#"oaiy-egpu: usage {"total": 25757220864, "used": 19327352832, "cached": 1073741824, "load": 87}"#));
        let usage = card.usage();
        let get = |k: &str| usage.get(k).and_then(Json::as_i64);
        assert_eq!((get("memory_total_mb"), get("memory_used_mb"), get("memory_cached_mb"), get("load")), (Some(24564), Some(18432), Some(1024), Some(87)));
        // (no load sampled yet: none said)
        assert!(card.heard(r#"oaiy-egpu: usage {"total": 25757220864, "used": 0, "cached": 0, "load": null}"#));
        assert_eq!(card.usage().get("load"), Some(&Json::Null));
        // A word that is not one is still kept out of the log, and nothing shown of it.
        assert!(card.heard("oaiy-egpu: usage {"));
        assert_eq!(card.usage(), Json::Null);
        // Stale: the launcher has stopped saying it.
        assert!(card.heard(r#"oaiy-egpu: usage {"total": 1048576, "used": 0, "cached": 0, "load": 0}"#));
        card.said.lock().unwrap().usage.as_mut().unwrap().1 -= USAGE_FRESH;
        assert_eq!(card.usage(), Json::Null);
        // The status has it, and which engine the card runs.
        let egpu = Egpu::new();
        let cfg = llm(r#"{"llm": {"egpu": {"enabled": true, "engine": "webgpu"}}}"#);
        let status = egpu.status(cfg.get("llm").unwrap());
        assert_eq!((status.get("engine").and_then(Json::as_str), status.get("usage")), (Some("webgpu"), Some(&Json::Null)));
        assert!(environment(cfg.get("llm").unwrap(), Path::new("/"), None).contains(&("OAIY_EGPU_STATS".into(), "1".into())));
    }

    /// The launcher says the card's memory and load each second, as the studio hears them (here the card's work on the
    /// CPU, whose memory is said as 16 GB).
    #[cfg(unix)]
    #[test]
    fn the_launcher_says_the_cards_memory_and_load() {
        use std::io::Write;
        let Some(launched) = Launched::new("usage", true) else { return };
        let (mut child, line) = launched.start_saying(true, "oaiy-egpu: usage ");
        let card = Card::default();
        assert!(card.heard(&line), "{line}");
        let usage = card.usage();
        assert_eq!(usage.get("memory_total_mb").and_then(Json::as_i64), Some(16 << 10), "{line}");
        assert!(usage.get("memory_used_mb").and_then(Json::as_i64).is_some_and(|mb| mb >= 0), "{line}");
        assert!(usage.get("load").and_then(Json::as_i64).is_some_and(|p| (0..=100).contains(&p)), "{line}");
        let mut pipe = child.stdin.take().unwrap();
        let _ = pipe.write_all(b"");
        drop(pipe);
        child.wait().unwrap();
    }

    /// A launcher that goes without a word (killed, as a crash would have it) takes its engine with it: the engine's
    /// standard input is the launcher's pipe, and OAIY's engine ends when that closes.
    #[cfg(unix)]
    #[test]
    fn an_engine_ends_when_its_launcher_is_killed() {
        let Some(launched) = Launched::new("killed", true) else { return };
        let mut child = launched.start(false, "oaiy-egpu: card at");
        let started = Instant::now();
        while std::fs::read_to_string(launched.dir.join("pids")).unwrap_or_default().trim().is_empty() {
            assert!(started.elapsed() < Duration::from_secs(20), "the engine was never started");
            std::thread::sleep(Duration::from_millis(50));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        std::thread::sleep(Duration::from_secs(1));
        assert_eq!(launched.engines_left(), 0, "the engine outlived its launcher");
    }
}
