//! oaiy-studio: a portable host for local AI on finite hardware.
//!
//!   oaiy-studio [--config FILE] [--open app|browser|none] [--ui-port N] [--port N]
//!
//! The library is the whole studio; `oaiy-studio` (a console program) and
//! `oaiy-studio-tray` (a Windows notification-area app) are two front ends.
//!
//! One folder holds everything: this program, `oaiy-llm-server` (language models),
//! `oaiy-media` (images and video) and `oaiy-studio.json`. The studio runs
//! the two engines as subprocesses, moves GPUs between them (a media job on the
//! LLM's GPU stops the LLM, which restarts after), serves a control UI, and
//! exposes a gateway whose routes -- paths, methods and API dialects -- come
//! from the configuration. See docs/STUDIO.md.

#![forbid(unsafe_code)]

mod config;
pub use config::AGENT_HTTP_ORIGIN;
mod system;
mod util;

// The control port (its page, JSON API and playground) and the public API: the gateway's routes, the discovery
// document, multipart bodies.
mod admin;
mod api;
// Getting models: what a file is, the catalog and its downloads, adding a file to the configuration.
mod models;
pub use models::downloads;
// The language model's server (and a Mac's eGPU through tinygrad).
mod llm;
/// What runs a Mac's eGPU when a plugin provides it ([`Studio::provide_egpu`]).
pub use llm::egpu::{available as egpu_available, Runtime as EgpuRuntime};
// The media worker and the jobs it takes: pictures, speech, music, sound effects, 3D models.
mod media;
// The models loaded on each GPU as one view, and what loaded and let them go.
mod resources;
use api::{discovery, gateway};
use llm::egpu;

use oaiy_engine::json::Json;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use util::{bool_or, int_or, str_or, LogRing};

pub const HELP: &str = "oaiy-studio: host language, image and video models behind configurable endpoints

  --config FILE      configuration (default: oaiy-studio.json beside this program;
                     created with defaults when missing)
  --open MODE        app (a browser window without tabs, Edge/Chrome), browser,
                     or none (default: the configuration's ui.open)
  --headless         same as --open none
  --ui-port N        control UI port (default: ui.port, 7860)
  --port N           gateway port (default: gateway.port, 8080)
  --start-llm        load the language model now instead of on first request

Console commands while running: status, start, stop, open, jobs, quit.
";

pub struct Studio {
    pub config_path: PathBuf,
    /// The configuration's folder: relative paths resolve against it.
    pub root: PathBuf,
    config: RwLock<Json>,
    pub llm: Arc<llm::Llm>,
    /// tinygrad's LLM server on a card the engine cannot reach itself (a Mac's eGPU); nothing elsewhere.
    pub egpu: Arc<egpu::Egpu>,
    pub media: Arc<media::Media>,
    pub system: system::System,
    pub log: Arc<LogRing>,
    /// Models being downloaded from the catalog.
    pub downloads: Arc<downloads::Downloads>,
    ui_url: RwLock<String>,
    gateway_url: RwLock<String>,
    /// Listener settings changed; they apply at the next start.
    restart_required: RwLock<bool>,
    /// One save at a time, so the file and memory never disagree.
    saving: std::sync::Mutex<()>,
    /// `--ui-port` / `--port`: in effect, never written to the file.
    port_overrides: (Option<u16>, Option<u16>),
}

impl Studio {
    pub fn config(&self) -> Json {
        self.config.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The eGPU's Python, tinygrad and launcher from a provider (OAIY Desktop's eGPU plugin), or the settings' again
    /// (None). A server that was idle on the other is started again on this, with its model; one that is answering
    /// changes at its next start.
    pub fn provide_egpu(self: &Arc<Self>, runtime: Option<EgpuRuntime>) {
        if let Some(model) = self.egpu.provide(runtime) {
            let studio = Arc::clone(self);
            std::thread::spawn(move || {
                if let Err(e) = studio.egpu.start(&studio.config(), &studio.root, &model) {
                    studio.egpu.log.push(format!("studio: the eGPU was not started again with {model}: {e}"));
                }
            });
        }
    }

    pub fn output_root(&self) -> PathBuf {
        let cfg = self.config();
        config::resolve(&self.root, cfg.get("media").map_or("outputs", |m| str_or(m, "output_dir", "outputs")))
    }

    /// Validate, save and apply a new configuration. A running LLM restarts when
    /// its settings changed; listener changes wait for the next start.
    pub fn set_config(&self, mut next: Json) -> Result<Json, String> {
        let _one = self.saving.lock().unwrap_or_else(|p| p.into_inner());
        config::merge_defaults(&mut next, &config::default_json());
        config::validate(&next)?;
        let before = self.config();
        // Command-line ports stay command-line ports: the file keeps its own.
        let mut on_disk = next.clone();
        if let Ok(file) = config::load(&self.config_path) {
            for (section, over) in [("ui", self.port_overrides.0), ("gateway", self.port_overrides.1)] {
                let same = next.get(section).and_then(|s| s.get("port")).and_then(Json::as_i64) == over.map(i64::from);
                if let (true, Some(Json::Obj(fields)), Some(p)) = (same && over.is_some(), on_disk.get(section).cloned(), file.get(section).and_then(|s| s.get("port")).cloned()) {
                    let mut s = Json::Obj(fields);
                    util::set(&mut s, "port", p);
                    util::set(&mut on_disk, section, s);
                }
            }
        }
        config::save(&self.config_path, &on_disk)?;
        *self.config.write().unwrap_or_else(|p| p.into_inner()) = next.clone();
        let listeners = |v: &Json| {
            ["ui", "gateway"].map(|k| v.get(k).map(|s| (str_or(s, "host", "").to_string(), int_or(s, "port", 0))))
        };
        if listeners(&before) != listeners(&next) {
            *self.restart_required.write().unwrap_or_else(|p| p.into_inner()) = true;
        }
        // Only a different launch restarts it: renaming the idle timeout must not
        // unload a model mid-conversation.
        let launch = |v: &Json| {
            let llm = v.get("llm").cloned().unwrap_or(Json::Null);
            let args = llm::arguments(&llm, &self.root, 0, "", true).map(|(a, _)| a).unwrap_or_default();
            let backend = ["server", "server_webgpu", "backend"].map(|k| str_or(&llm, k, "").to_string());
            let incognito = v.get("privacy").is_some_and(|p| bool_or(p, "incognito", false));
            (args, backend, llm.get("webgpu_gb").cloned(), bool_or(&llm, "enabled", true), incognito)
        };
        let llm_changed = launch(&before) != launch(&next);
        if llm_changed && self.llm.is_running() {
            self.log.push("LLM settings changed: restarting oaiy-llm-server");
            self.llm.stop();
            let llm = next.get("llm").unwrap_or(&Json::Null);
            if !llm::holds_a_model(llm) {
                self.log.push(llm::ALL_ON_EGPU);
            } else if bool_or(llm, "enabled", true) {
                // Saved either way; a failed restart is reported, not a failed save.
                if let Err(e) = self.llm.start(&next, &self.root) {
                    self.log.push(format!("the LLM did not restart: {e}"));
                }
            }
        }
        // tinygrad's server holds one model under one set of settings: with either changed it stops, and the next
        // request for a model of its own starts it again.
        let egpu_launch = |v: &Json| {
            let llm = v.get("llm").cloned().unwrap_or(Json::Null);
            let file = self.egpu.held().as_deref().filter(|m| egpu::assigned(&llm, m)).and_then(|m| egpu::arguments(&llm, &self.root, m, std::path::Path::new(""), 0).ok());
            (file, llm.get("egpu").cloned())
        };
        if self.egpu.state() != llm::State::Stopped && egpu_launch(&before) != egpu_launch(&next) {
            self.log.push("eGPU settings changed: stopping tinygrad's server");
            self.egpu.stop();
        }
        self.place_the_model(&before, &next);
        self.log.push("configuration saved");
        Ok(next)
    }

    /// One language model loaded at a time, where the default model runs: a default chosen on the eGPU (or moved
    /// there) has this computer's engine let go of the model it holds and the card load the default; one chosen
    /// here has the card let go of its model and this computer's engine load the default. Now, not at the next
    /// request (a model of 16 GB takes a minute to reach the card). Nothing changes while the default and where it
    /// runs stay as they were: a request for another model still loads it where it runs ([`llm::holds_a_model`]).
    fn place_the_model(&self, before: &Json, next: &Json) {
        let placed = |v: &Json| {
            let llm = v.get("llm").cloned().unwrap_or(Json::Null);
            let default = llm::default_model(&llm);
            (egpu::assigned(&llm, &default), default, bool_or(&llm, "enabled", true))
        };
        let (on_card, default, enabled) = placed(next);
        if placed(before) == (on_card, default.clone(), enabled) || !enabled || default.is_empty() {
            return;
        }
        if on_card && self.llm.is_running() {
            self.log.push(format!("{default} runs on the eGPU: this computer's engine lets go of its model"));
            self.llm.stop();
        }
        if !on_card && self.egpu.state() != llm::State::Stopped {
            self.log.push(format!("{default} runs on this computer: the eGPU lets go of its model"));
            self.egpu.stop();
        }
        // (the load begun beside the save, which it does not hold up: the card's waits for a server it replaces to
        // be gone, up to 25 s)
        let (egpu, mac, cfg, root, log) = (Arc::clone(&self.egpu), Arc::clone(&self.llm), next.clone(), self.root.clone(), Arc::clone(&self.log));
        let begun = std::thread::Builder::new().name("place-model".into()).spawn(move || {
            let (where_, started) = if on_card { ("on the eGPU", egpu.start(&cfg, &root, &default)) } else { ("on this computer", mac.start(&cfg, &root)) };
            match started {
                Ok(()) => log.push(format!("loading {default} {where_}")),
                Err(e) => log.push(format!("{default} did not start loading {where_}: {e}")),
            }
        });
        if let Err(e) = begun {
            self.log.push(format!("the language model was not loaded ahead of time: {e}"));
        }
    }

    /// Stop the language model and any media job, so no engine outlives the
    /// studio holding GPU memory. Waits briefly for a media worker to be killed.
    pub fn shutdown(&self) {
        self.llm.stop();
        self.egpu.stop();
        let mut cancelled = false;
        for job in self.media.list() {
            cancelled |= self.media.cancel(&job.id);
        }
        if cancelled {
            // The runner polls the cancel flag every 200 ms and kills the worker.
            for _ in 0..25 {
                if !self.media.busy() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        // The worker kept for pictures, waiting for the next: no job, so nothing above ends it, and an update would
        // find its program still running (on Windows the installer cannot replace it).
        self.media.end_serving(false, "the studio is stopping");
        self.log.push("studio stopped");
    }

    pub fn state(&self) -> Json {
        let root = self.output_root();
        Json::obj([
            ("version", Json::str(env!("CARGO_PKG_VERSION"))),
            ("config_path", Json::str(self.config_path.to_string_lossy())),
            ("root", Json::str(self.root.to_string_lossy())),
            ("output_root", Json::str(root.to_string_lossy())),
            ("ui_url", Json::str(self.ui_url.read().unwrap_or_else(|p| p.into_inner()).as_str())),
            ("gateway_url", Json::str(self.gateway_url.read().unwrap_or_else(|p| p.into_inner()).as_str())),
            ("restart_required", Json::Bool(*self.restart_required.read().unwrap_or_else(|p| p.into_inner()))),
            ("media_pauses_llm", Json::Bool(media::pauses_llm(&self.config()))),
            ("llm", self.llm.status()),
            ("egpu", self.egpu.status(self.config().get("llm").unwrap_or(&Json::Null))),
            ("media", Json::obj([
                ("busy", Json::Bool(self.media.busy())),
                ("jobs", Json::Arr(self.media.list().iter().take(60).map(|j| j.to_json(&root)).collect())),
                ("private", self.media.private_activity()),
            ])),
        ])
    }
}

/// The language model's supervisor and media's runner: a picture worker media keeps on this computer's GPU makes way for
/// the language model whenever it starts.
fn llm_and_media() -> (Arc<llm::Llm>, Arc<media::Media>) {
    let (llm, media) = (Arc::new(llm::Llm::new()), Arc::new(media::Media::new()));
    let kept = Arc::clone(&media);
    llm.before_start(move || kept.let_go_local());
    (llm, media)
}

#[cfg(test)]
impl Studio {
    /// A Studio with `cfg` (nothing started, nothing saved), for the tests of what a request comes to.
    pub(crate) fn for_test(root: &std::path::Path, cfg: Json) -> Studio {
        let (llm, media) = llm_and_media();
        Studio {
            config_path: root.join("oaiy-studio.json"),
            root: root.to_path_buf(),
            config: RwLock::new(cfg),
            llm,
            egpu: Arc::new(egpu::Egpu::new()),
            media,
            system: system::System::new(),
            log: Arc::new(LogRing::new(50)),
            downloads: Arc::new(downloads::Downloads::new()),
            ui_url: RwLock::new(String::new()),
            gateway_url: RwLock::new(String::new()),
            restart_required: RwLock::new(false),
            saving: std::sync::Mutex::new(()),
            port_overrides: (None, None),
        }
    }
}

/// Open a URL or folder with the desktop's handler.
pub fn open_path(target: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("explorer").arg(target).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(target).spawn();
    #[cfg(not(any(windows, target_os = "macos")))]
    let r = oaiy_host_env::command("xdg-open").arg(target).spawn();
    let _ = r;
}

/// A window without browser chrome when Edge or Chrome is installed, so the UI
/// feels like a desktop app; the default browser otherwise.
pub fn open_ui(url: &str, mode: &str) {
    if mode == "none" {
        return;
    }
    if mode == "app" {
        let mut candidates: Vec<PathBuf> = Vec::new();
        #[cfg(windows)]
        for base in ["ProgramFiles(x86)", "ProgramFiles", "LocalAppData"] {
            if let Some(dir) = std::env::var_os(base) {
                let dir = PathBuf::from(dir);
                candidates.push(dir.join("Microsoft/Edge/Application/msedge.exe"));
                candidates.push(dir.join("Google/Chrome/Application/chrome.exe"));
            }
        }
        #[cfg(not(windows))]
        for p in ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"] {
            candidates.push(PathBuf::from(p));
        }
        for exe in candidates.into_iter().filter(|p| p.is_file()) {
            if oaiy_host_env::command(exe).arg(format!("--app={url}")).arg("--window-size=1400,900").spawn().is_ok() {
                return;
            }
        }
    }
    #[cfg(windows)]
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(not(windows))]
    open_path(url);
}

/// Command-line options (both front ends take the same ones).
pub struct Args {
    pub config: PathBuf,
    pub open: Option<String>,
    pub ui_port: Option<u16>,
    pub port: Option<u16>,
    pub start_llm: bool,
}

pub fn parse_args_from(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args { config: config::default_path(), open: None, ui_port: None, port: None, start_llm: false };
    let mut it = args;
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--config" => a.config = val()?.into(),
            "--open" => {
                let v = val()?;
                if !["app", "browser", "none"].contains(&v.as_str()) {
                    return Err("--open wants app, browser or none".into());
                }
                a.open = Some(v);
            }
            "--headless" => a.open = Some("none".into()),
            "--ui-port" => a.ui_port = Some(val()?.parse().map_err(|_| "--ui-port: not a port")?),
            "--port" => a.port = Some(val()?.parse().map_err(|_| "--port: not a port")?),
            "--start-llm" => a.start_llm = true,
            other => return Err(format!("unknown option {other} (try --help)")),
        }
    }
    Ok(a)
}

/// How many ports past the configured one are tried before any free one the system gives.
const PORTS_AFTER: u16 = 20;

/// How long a language model that starts with OAIY on a Mac's eGPU waits for OAIY Desktop's eGPU plugin to say what
/// runs the card (it says a few seconds after the studio starts) before the settings' Python is used.
const EGPU_RUNTIME_WAIT: Duration = Duration::from_secs(30);

/// A listener on `host:port`, or, where that port is not to be had, on the next free one of the [`PORTS_AFTER`] after
/// it, else on any the system gives: with what was taken instead, for the log. Another program may hold the port
/// (8080 is many a program's), Windows may keep it reserved (Hyper-V's and Docker's excluded ranges refuse a bind
/// as access denied), or hold it a minute or two after an OAIY that ended suddenly (a desktop could not start its
/// engines at all: "Only one usage of each socket address ... is normally permitted"). What the studio says of its
/// addresses (`ui_url`, `gateway_url`, the discovery document's) is the port it has, and OAIY Desktop asks it.
fn listen(host: &str, port: u16, what: &str) -> Result<(TcpListener, Option<String>), String> {
    let taken = |e: &std::io::Error| matches!(e.kind(), std::io::ErrorKind::AddrInUse | std::io::ErrorKind::PermissionDenied);
    let first = match TcpListener::bind((host, port)) {
        Ok(l) => return Ok((l, None)),
        Err(e) if port != 0 && taken(&e) => e,
        Err(e) => return Err(format!("{what}: cannot listen on {host}:{port}: {e} (change the port with --{}port or in the configuration)", if what == "UI" { "ui-" } else { "" })),
    };
    let next = (1..=PORTS_AFTER).filter_map(|d| port.checked_add(d)).chain([0]);
    for other in next {
        if let Ok(l) = TcpListener::bind((host, other)) {
            let got = l.local_addr().map(|a| a.port()).unwrap_or(other);
            return Ok((l, Some(format!("{what}: {host}:{port} is not to be had ({first}): listening on {host}:{got} instead"))));
        }
    }
    Err(format!("{what}: cannot listen on {host}:{port} ({first}), nor on any other port"))
}

fn serve(listener: TcpListener, studio: Arc<Studio>, admin: bool) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || {
                oaiy_engine::http::serve(stream, |req, w| {
                    if admin {
                        return admin::handle(&studio, req, w, port);
                    }
                    public(&studio, req, w)
                });
            });
        }
    });
}

/// The gateway listener: key check, then the route table.
fn public(studio: &Arc<Studio>, req: &oaiy_engine::http::Request, w: &mut std::net::TcpStream) -> std::io::Result<bool> {
    if req.method == "OPTIONS" {
        oaiy_engine::http::respond(w, 204, "text/plain", b"", true)?;
        return Ok(true);
    }
    let cfg = studio.config();
    let gateway = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let routes = gateway.get("routes").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default();
    let key = str_or(&gateway, "api_key", "").to_string();
    let keyed = |req: &oaiy_engine::http::Request| key.is_empty() || req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")).map(str::trim) == Some(key.as_str());
    // A web page the user happens to visit can reach 127.0.0.1 too. Without a
    // key, browser requests (they carry Origin) are served only to the origins
    // listed in gateway.cors_origins; apps and scripts send no Origin.
    if key.is_empty() {
        if let Some(origin) = req.header("origin") {
            let allowed = gateway.get("cors_origins").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str)
                .any(|o| o == "*" || o.trim_end_matches('/').eq_ignore_ascii_case(origin));
            if !allowed {
                return gateway::json_reply(w, 403, &util::error_json(&format!("requests from {origin} are not allowed; add it to gateway.cors_origins, or set an API key"), "permission_error", "origin_not_allowed"));
            }
        }
    }
    // Discovery answers without the key too, saying only that one is needed.
    if req.method == "GET" && req.route() == "/.well-known/oaiy.json" {
        return gateway::json_reply(w, 200, &discovery::document(studio, &gateway::public_base(studio, req), keyed(req)));
    }
    let Some(m) = gateway::route(&routes, &req.method, req.route()) else {
        if req.route() == "/" {
            return gateway::json_reply(w, 200, &gateway::health(studio));
        }
        return gateway::json_reply(w, 404, &util::error_json(&format!("no route {} {}", req.method, req.route()), "invalid_request_error", "not_found"));
    };
    if m.target == "discovery" {
        return gateway::json_reply(w, 200, &discovery::document(studio, &gateway::public_base(studio, req), keyed(req)));
    }
    let key = key.as_str();
    if !key.is_empty() && m.target != "health" {
        let given = req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
        // Browsers cannot set headers on <img>/<video>: file links may carry ?key=.
        let query_key = (m.target == "files").then(|| req.query("key")).flatten();
        if given != Some(key) && query_key.as_deref() != Some(key) {
            return gateway::json_reply(w, 401, &util::error_json("missing or wrong API key", "authentication_error", "invalid_api_key"));
        }
    }
    gateway::handle(studio, req, w, m, false)
}

/// A started studio: its state and where it listens.
pub struct Running {
    pub studio: Arc<Studio>,
    pub ui_url: String,
    pub gateway_url: String,
    /// How to show the UI: `app`, `browser` or `none`.
    pub open: String,
}

/// The configuration with command-line overrides applied.
fn effective_config(args: &Args) -> Result<Json, String> {
    let mut cfg = config::load(&args.config)?;
    if let Some(p) = args.ui_port {
        let mut ui = cfg.get("ui").cloned().unwrap_or(Json::Null);
        util::set(&mut ui, "port", Json::Int(p as i64));
        util::set(&mut cfg, "ui", ui);
    }
    if let Some(p) = args.port {
        let mut g = cfg.get("gateway").cloned().unwrap_or(Json::Null);
        util::set(&mut g, "port", Json::Int(p as i64));
        util::set(&mut cfg, "gateway", g);
    }
    Ok(cfg)
}

/// The UI address of a studio already serving this configuration, if one is:
/// a second launch then shows that one instead of failing on busy ports.
pub fn running_instance(args: &Args) -> Option<String> {
    running_at(args)
}

/// The language-model server the configuration at `config_path` names (`llm.server`): a bare program name
/// (found beside the studio, else on the PATH) or a path. Made with defaults when there is none.
pub fn llm_server(config_path: &std::path::Path) -> Result<String, String> {
    let cfg = config::load(config_path)?;
    Ok(cfg.get("llm").map_or("oaiy-llm-server", |l| str_or(l, "server", "oaiy-llm-server")).to_string())
}

/// Set `llm.server` in the configuration file: a path, or the bare default again (OAIY Desktop takes a studio off
/// the CUDA engine an older version of it downloaded). For a studio that is not running; a running one is told
/// through `PUT /api/config`, which restarts its language model on the new server.
pub fn set_llm_server(config_path: &std::path::Path, value: &str) -> Result<(), String> {
    let mut cfg = config::load(config_path)?;
    let mut llm = cfg.get("llm").cloned().unwrap_or(Json::Obj(Vec::new()));
    util::set(&mut llm, "server", Json::str(value));
    util::set(&mut cfg, "llm", llm);
    config::save(config_path, &cfg)
}

/// For a host that runs the studio but is not its executable (OAIY Desktop):
/// the configuration at `config_path` (made with defaults when there is none),
/// with each engine program it names bare (`oaiy-llm-server`, `oaiy-media`)
/// pointed at `dir` when it is there and not beside the configuration.
/// Returns whether anything changed.
///
/// A program named by a path is the owner's choice and is left alone, but for
/// one case: the path leads to no program any more and `dir` has a program of
/// that name. That is the path an earlier start wrote, to a folder the host
/// was in then: an AppImage is mounted somewhere new at every start, a Mac's
/// app runs from a holding folder until it is moved to Applications, and an
/// install can be moved. Left alone, the model would not start from the second
/// launch on.
pub fn use_programs_from(config_path: &std::path::Path, dir: &std::path::Path) -> Result<bool, String> {
    let mut cfg = config::load(config_path)?;
    let root = config_path.parent().unwrap_or(std::path::Path::new("."));
    let mut changed = false;
    for (section, key) in [("llm", "server"), ("llm", "server_webgpu"), ("media", "worker")] {
        let Some(mut part) = cfg.get(section).cloned() else { continue };
        let Some(name) = part.get(key).and_then(Json::as_str).map(str::to_string) else { continue };
        let file = if name.contains(['/', '\\']) {
            let path = std::path::Path::new(&name);
            match path.file_name().and_then(|f| f.to_str()) {
                Some(file) if !root.join(path).is_file() => file.to_string(),
                _ => continue,
            }
        } else if cfg!(windows) && !name.to_ascii_lowercase().ends_with(".exe") {
            format!("{name}.exe")
        } else {
            name.clone()
        };
        // (the media worker the host carries is taken over a copy beside the configuration: that is where a person
        // was told to put one of their own while no installer carried it, and left there it would outlive every
        // update of the host's; a path named in the configuration is still theirs)
        let carried_first = section == "media" && !name.contains(['/', '\\']);
        if (root.join(&file).is_file() && !carried_first) || !dir.join(&file).is_file() {
            continue;
        }
        util::set(&mut part, key, Json::str(dir.join(&file).to_string_lossy()));
        util::set(&mut cfg, section, part);
        changed = true;
    }
    if changed {
        config::save(config_path, &cfg)?;
    }
    Ok(changed)
}

/// For a host with a models folder of its own (OAIY Desktop: the folder its window lists and opens): `downloads.dir`
/// in the configuration at `config_path` set to `dir`, where the configuration names none and nothing has been
/// downloaded into the default folder yet (`models` beside the configuration). A folder the person chose stays
/// theirs, and an install that already has downloads keeps finding them where they are: the catalog tells what is
/// installed by looking in the downloads folder. Returns whether anything changed.
pub fn use_downloads_dir(config_path: &std::path::Path, dir: &std::path::Path) -> Result<bool, String> {
    let mut cfg = config::load(config_path)?;
    let mut downloads = cfg.get("downloads").cloned().unwrap_or(Json::Obj(Vec::new()));
    if !str_or(&downloads, "dir", "").trim().is_empty() {
        return Ok(false);
    }
    let root = config_path.parent().unwrap_or(std::path::Path::new("."));
    if std::fs::read_dir(root.join("models")).is_ok_and(|mut entries| entries.next().is_some()) {
        return Ok(false);
    }
    util::set(&mut downloads, "dir", Json::str(dir.to_string_lossy()));
    util::set(&mut cfg, "downloads", downloads);
    config::save(config_path, &cfg)?;
    Ok(true)
}

fn running_at(args: &Args) -> Option<String> {
    let cfg = effective_config(args).ok()?;
    let ui = cfg.get("ui")?;
    let host = match str_or(ui, "host", "127.0.0.1") {
        "0.0.0.0" | "::" => "127.0.0.1",
        h => h,
    };
    let addr = format!("{host}:{}", int_or(ui, "port", 7860));
    let reply = oaiy_engine::http::fetch(&addr, "GET", "/api/state", &[], b"", Duration::from_secs(2)).ok()?;
    let body = reply.body(4 << 20).ok()?;
    let state = Json::parse(&body).ok()?;
    (state.get("config_path").is_some()).then(|| format!("http://{addr}"))
}

/// Load the configuration, bind both listeners and start the supervisors.
/// Returns once the UI and API serve; `quiet` keeps it off stdout (the tray).
pub fn launch(args: &Args, quiet: bool) -> Result<Running, String> {
    let cfg = effective_config(args)?;
    let root = args.config.parent().filter(|p| !p.as_os_str().is_empty()).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    // Absolute but not canonical: on Windows canonical paths are `\\?\`-prefixed,
    // which leaks into every output path shown to users.
    let root = std::path::absolute(&root).unwrap_or(root);
    let (llm, media) = llm_and_media();
    let studio = Arc::new(Studio {
        config_path: args.config.clone(),
        root,
        config: RwLock::new(cfg.clone()),
        llm,
        egpu: Arc::new(egpu::Egpu::new()),
        media,
        system: system::System::new(),
        log: Arc::new(LogRing::new(1000)),
        downloads: Arc::new(downloads::Downloads::new()),
        ui_url: RwLock::new(String::new()),
        gateway_url: RwLock::new(String::new()),
        restart_required: RwLock::new(false),
        saving: std::sync::Mutex::new(()),
        port_overrides: (args.ui_port, args.port),
    });
    // a media job's stages each read its device's memory: the card's (its launcher's word each second), or this
    // computer's GPU's
    let weak = Arc::downgrade(&studio);
    studio.media.set_memory_probe(Arc::new(move |device: &str| {
        let studio = weak.upgrade()?;
        if device == "egpu" {
            return studio.egpu.card_usage().get("memory_used_mb").and_then(Json::as_i64);
        }
        let index: i64 = device.strip_prefix("gpu")?.parse().ok()?;
        let gpus = studio.system.gpus();
        gpus.as_array()?.iter().find(|g| g.get("index").and_then(Json::as_i64) == Some(index))?.get("memory_used_mb").and_then(Json::as_i64)
    }));
    // The language models' engines watched each second, each change of theirs an event of `crate::resources`: loading,
    // loaded, let go, the card lent to media and given back, whoever caused it (a request, a flow, a setting, Stop).
    let watched = Arc::downgrade(&studio);
    let _ = std::thread::Builder::new().name("resources-watch".into()).spawn(move || {
        let mut last: [(String, String, bool); 2] = Default::default();
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let Some(studio) = watched.upgrade() else { return };
            let cfg = studio.config();
            let card = studio.egpu.status(cfg.get("llm").unwrap_or(&Json::Null));
            let local = studio.llm.status();
            let now = [
                (str_or(&card, "state", "stopped").to_string(), str_or(&card, "model", "").to_string(), bool_or(&card, "lent", false) || bool_or(&card, "paused", false)),
                (str_or(&local, "state", "stopped").to_string(), local.get("resident").and_then(Json::as_str).or_else(|| local.get("models").and_then(|m| m.at(0)).and_then(Json::as_str)).unwrap_or("").to_string(), bool_or(&local, "paused_for_media", false)),
            ];
            for (i, ((state, model, lent), (was, was_model, was_lent))) in now.iter().zip(&last).enumerate() {
                let device = if i == 0 { "egpu".to_string() } else { local.get("runs_on").and_then(Json::as_str).unwrap_or("this computer").to_string() };
                let model = if model.is_empty() { was_model } else { model };
                let seconds = |s: &Json| s.get("load_seconds").and_then(Json::as_f64).map_or(String::new(), |t| format!(" after {t:.1} s"));
                // (the card's server started paused, to lend the card to pictures, loads nothing)
                let paused_start = i == 0 && *lent;
                match (was.as_str(), state.as_str()) {
                    (a, b) if a == b => {}
                    (_, "starting") if !paused_start => resources::event("load", model, &device, "loading"),
                    (_, "ready") if !paused_start => resources::event("loaded", model, &device, &format!("ready{}", seconds(if i == 0 { &card } else { &local }))),
                    ("starting" | "ready", "stopped") => resources::event("unload", model, &device, "let go"),
                    (_, "failed") => resources::event("unload", model, &device, "it failed to load or stopped answering"),
                    _ => {}
                }
                if lent != was_lent && !was.is_empty() && !(i == 0 && *lent && was != "ready") {
                    if *lent {
                        resources::event(if i == 0 { "lend" } else { "evict" }, model, &device, if i == 0 { "the card lent to a media job: its language model let go" } else { "paused for a media job on its GPU" });
                    } else {
                        resources::event("give_back", model, &device, "back to its language model after media");
                    }
                }
            }
            last = now;
        }
    });
    let ui = cfg.get("ui").cloned().unwrap_or(Json::Null);
    let gw = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let (ui_host, gw_host) = (str_or(&ui, "host", "127.0.0.1").to_string(), str_or(&gw, "host", "127.0.0.1").to_string());
    let (ui_listener, ui_moved) = listen(&ui_host, int_or(&ui, "port", 7860) as u16, "UI")?;
    let (gw_listener, gw_moved) = listen(&gw_host, int_or(&gw, "port", 8080) as u16, "gateway")?;
    for moved in [ui_moved, gw_moved].into_iter().flatten() {
        if !quiet {
            println!("{moved}");
        }
        studio.log.push(moved);
    }
    let shown = |host: &str| if host == "0.0.0.0" || host == "::" { "127.0.0.1".to_string() } else { host.to_string() };
    let ui_url = format!("http://{}:{}", shown(&ui_host), ui_listener.local_addr().map_err(|e| e.to_string())?.port());
    let gw_url = format!("http://{}:{}", shown(&gw_host), gw_listener.local_addr().map_err(|e| e.to_string())?.port());
    *studio.ui_url.write().unwrap_or_else(|p| p.into_inner()) = ui_url.clone();
    *studio.gateway_url.write().unwrap_or_else(|p| p.into_inner()) = gw_url.clone();

    let runner = Arc::clone(&studio);
    std::thread::Builder::new().name("media".into()).spawn(move || {
        let media = Arc::clone(&runner.media);
        media.run(&runner);
    }).map_err(|e| e.to_string())?;
    // Stop an idle LLM so its memory returns to the machine; the next request loads it again.
    // Incognito folders left by a crash are gone before anything serves.
    let _ = std::fs::remove_dir_all(studio.output_root().join(".incognito"));
    let idle = Arc::clone(&studio);
    std::thread::spawn(move || {
        let mut ticks = 0u64;
        loop {
            std::thread::sleep(Duration::from_secs(30));
            ticks += 1;
            idle.media.sweep();
            // (a pinned language model is never stopped for being idle)
            let minutes = idle.config().get("llm").filter(|l| !bool_or(l, "pinned", false)).map_or(0, |l| int_or(l, "idle_stop_minutes", 0));
            // Never mid-reply: a long stream updates nothing until it ends.
            if minutes > 0 && idle.media.chats_active() == 0 && idle.llm.state() == llm::State::Ready && idle.llm.idle_for() > Duration::from_secs(minutes as u64 * 60) {
                idle.log.push(format!("LLM idle for {minutes} min: stopping it to free memory"));
                idle.llm.stop();
            }
            // tinygrad's server by the same rule: the card's memory back, the next request loads it again.
            if minutes > 0 && idle.egpu.stop_if_idle(Duration::from_secs(minutes as u64 * 60)) {
                idle.log.push(format!("eGPU idle for {minutes} min: tinygrad's server stopped to free the card"));
            }
            // Hourly: uploaded reference images older than a day.
            if ticks % 120 == 1 {
                let inputs = idle.output_root().join("inputs");
                for entry in std::fs::read_dir(&inputs).into_iter().flatten().flatten() {
                    let old = entry.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > Duration::from_secs(86_400));
                    if old {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
    });
    serve(ui_listener, Arc::clone(&studio), true);
    serve(gw_listener, Arc::clone(&studio), false);

    if !quiet {
        println!("oaiy-studio {}", env!("CARGO_PKG_VERSION"));
        println!("  control UI  {ui_url}");
        println!("  API         {gw_url}/v1   (OpenAI-compatible; routes are configurable in the UI)");
        println!("  config      {}", args.config.display());
        println!("  type 'help' for console commands");
    }
    let llm_cfg = cfg.get("llm").cloned().unwrap_or(Json::Null);
    let forced = args.start_llm;
    if (forced || bool_or(&llm_cfg, "autostart", false)) && !llm::holds_a_model(&llm_cfg) {
        // The default model runs on the eGPU: it starts there, not here, once the card's runtime is known.
        let s = Arc::clone(&studio);
        let begun = std::thread::Builder::new().name("autostart-egpu".into()).spawn(move || {
            let still = || {
                let cfg = s.config();
                let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
                let name = llm::default_model(&llm);
                let wanted = (forced || bool_or(&llm, "autostart", false)) && bool_or(&llm, "enabled", true) && !llm::holds_a_model(&llm) && !name.is_empty();
                wanted.then(|| (cfg, s.root.clone(), name))
            };
            match s.egpu.start_when_provided(EGPU_RUNTIME_WAIT, still) {
                Ok(Some(name)) => s.log.push(format!("loading {name} on the eGPU: the language model starts with OAIY")),
                Ok(None) => {}
                Err(e) => s.log.push(format!("the language model did not start on the eGPU with OAIY: {e}")),
            }
        });
        if let Err(e) = begun {
            studio.log.push(format!("the language model did not start on the eGPU with OAIY: {e}"));
        }
    } else if forced || bool_or(&llm_cfg, "autostart", false) {
        if let Err(e) = studio.llm.start(&cfg, &studio.root) {
            studio.log.push(format!("LLM not started: {e}"));
            if !quiet {
                println!("  LLM not started: {e}");
            }
        }
    }
    let open = args.open.clone().unwrap_or_else(|| str_or(&ui, "open", "app").to_string());
    Ok(Running { studio, ui_url, gateway_url: gw_url, open })
}

/// Line commands on stdin; without a console (stdin closed) just keep serving.
pub fn console(studio: &Arc<Studio>, ui_url: &str) {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        match line.trim() {
            "" => {}
            "help" => println!("status | start | stop | open | jobs | quit"),
            "status" => println!("{}", config::pretty(&studio.llm.status(), 0)),
            "start" => match studio.llm.start(&studio.config(), &studio.root) {
                Ok(()) => println!("starting the LLM..."),
                Err(e) => println!("{e}"),
            },
            "stop" => {
                studio.llm.stop();
                println!("LLM stopped");
            }
            "open" => open_ui(ui_url, "app"),
            "jobs" => {
                for j in studio.media.list().iter().take(10) {
                    println!("{} {:<11} {:>5.1}% {} {}", j.id, j.status, j.progress, j.model, j.prompt.chars().take(50).collect::<String>());
                }
            }
            "quit" | "exit" => {
                studio.shutdown();
                std::process::exit(0);
            }
            other => println!("unknown command {other:?} (help)"),
        }
    }
    loop {
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_port_another_program_holds_is_listened_beside_and_said() {
        // a free port, then held by another listener, as another program holds 8080
        let held = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = held.local_addr().unwrap().port();
        let (l, moved) = listen("127.0.0.1", port, "gateway").unwrap();
        let got = l.local_addr().unwrap().port();
        assert_ne!(got, port);
        let said = moved.expect("the move is said");
        assert!(said.contains(&format!("127.0.0.1:{port} is not to be had")) && said.contains(&format!("listening on 127.0.0.1:{got} instead")), "{said}");
        // the port free: it is the one listened on, and nothing is said (on another free port where a test beside this
        // one was handed the port just let go: seen on Linux, "127.0.0.1:46647 is not to be had (Address already in
        // use)" the moment after it was)
        drop((held, l));
        let mut port = port;
        let (got, moved) = (0..5)
            .map(|_| {
                let (l, moved) = listen("127.0.0.1", port, "gateway").unwrap();
                let got = (l.local_addr().unwrap().port(), moved);
                if got.1.is_some() {
                    drop(l);
                    port = TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port();
                }
                got
            })
            .find(|(_, moved)| moved.is_none())
            .expect("a free port listened on as it is");
        assert_eq!((got, moved), (port, None));
        // a host that is not this computer's is not worked around
        assert!(listen("203.0.113.9", port, "UI").is_err());
    }

    /// "Start when OAIY opens" is saved from the Overview as it is ticked: the model loaded meanwhile stays loaded.
    #[test]
    fn starting_with_oaiy_is_saved_without_stopping_the_model_that_is_loaded() {
        let dir = std::env::temp_dir().join(format!("oaiy-studio-autostart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("big.gguf"), b"GGUF").unwrap();
        let mut cfg = config::default_json();
        let mut llm = cfg.get("llm").cloned().unwrap();
        util::set(&mut llm, "egpu", Json::parse(br#"{"enabled": true}"#).unwrap());
        util::set(&mut llm, "models", Json::parse(br#"[{"name": "big", "path": "big.gguf", "egpu": true}]"#).unwrap());
        util::set(&mut cfg, "llm", llm);
        config::validate(&cfg).unwrap();
        let studio = Studio::for_test(&dir, cfg.clone());
        studio.egpu.adopt("127.0.0.1:1", "k", "big", None);
        for on in [true, false] {
            let mut next = studio.config();
            let mut llm = next.get("llm").cloned().unwrap();
            util::set(&mut llm, "autostart", Json::Bool(on));
            util::set(&mut next, "llm", llm);
            studio.set_config(next).unwrap();
            assert_eq!(studio.config().get("llm").map(|l| bool_or(l, "autostart", !on)), Some(on));
            assert_eq!((studio.egpu.state(), studio.egpu.held().as_deref()), (llm::State::Ready, Some("big")), "autostart {on}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_host_reads_and_sets_the_server_and_keeps_the_rest_of_the_configuration() {
        let dir = std::env::temp_dir().join(format!("oaiy-llm-server-setting-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(config::FILE_NAME);
        // No file yet: the default, and the file is made.
        assert_eq!(llm_server(&path).unwrap(), "oaiy-llm-server");
        let downloaded = dir.join("engines").join("cuda").join("0.2.0").join("oaiy-llm-server.exe");
        set_llm_server(&path, &downloaded.to_string_lossy()).unwrap();
        assert_eq!(llm_server(&path).unwrap(), downloaded.to_string_lossy());
        let cfg = config::load(&path).unwrap();
        let llm = cfg.get("llm").unwrap();
        assert_eq!(str_or(llm, "server_webgpu", ""), "oaiy-llm-server-webgpu", "the server's other name is kept");
        assert!(cfg.get("gateway").is_some(), "the other sections are kept");
        set_llm_server(&path, "oaiy-llm-server").unwrap();
        assert_eq!(llm_server(&path).unwrap(), "oaiy-llm-server");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_host_points_bare_engine_programs_at_their_folder() {
        let base = std::env::temp_dir().join(format!("oaiy-programs-{}", std::process::id()));
        let (conf_dir, bin) = (base.join("conf"), base.join("bin"));
        std::fs::create_dir_all(&conf_dir).unwrap();
        std::fs::create_dir_all(&bin).unwrap();
        let exe = |n: &str| if cfg!(windows) { format!("{n}.exe") } else { n.to_string() };
        std::fs::write(bin.join(exe("oaiy-llm-server")), b"").unwrap();
        std::fs::write(bin.join(exe("oaiy-media")), b"").unwrap();
        let path = conf_dir.join(config::FILE_NAME);
        // No configuration yet: made with defaults, then pointed at the folder.
        assert!(use_programs_from(&path, &bin).unwrap());
        let cfg = config::load(&path).unwrap();
        let server = cfg.get("llm").and_then(|l| l.get("server")).and_then(Json::as_str).unwrap().to_string();
        assert_eq!(std::path::Path::new(&server), bin.join(exe("oaiy-llm-server")));
        let worker = cfg.get("media").and_then(|m| m.get("worker")).and_then(Json::as_str).unwrap().to_string();
        assert_eq!(std::path::Path::new(&worker), bin.join(exe("oaiy-media")));
        // A program that is not in the folder (the WebGPU server) keeps its name; a second call changes nothing.
        assert_eq!(cfg.get("llm").and_then(|l| l.get("server_webgpu")).and_then(Json::as_str), Some("oaiy-llm-server-webgpu"));
        assert!(!use_programs_from(&path, &bin).unwrap());
        // Copies beside the configuration (a person's own, from before installers carried the media worker): the
        // language-model server's stays theirs, the media worker is the host's.
        let own = base.join("own");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(own.join(exe("oaiy-llm-server")), b"").unwrap();
        std::fs::write(own.join(exe("oaiy-media")), b"").unwrap();
        let own_path = own.join(config::FILE_NAME);
        assert!(use_programs_from(&own_path, &bin).unwrap());
        let cfg = config::load(&own_path).unwrap();
        assert_eq!(cfg.get("llm").and_then(|l| l.get("server")).and_then(Json::as_str), Some("oaiy-llm-server"));
        let worker = cfg.get("media").and_then(|m| m.get("worker")).and_then(Json::as_str).unwrap().to_string();
        assert_eq!(std::path::Path::new(&worker), bin.join(exe("oaiy-media")));
        // A worker the configuration names by its path stays, the host's or not.
        let mut cfg = config::load(&own_path).unwrap();
        let mut media = cfg.get("media").cloned().unwrap();
        util::set(&mut media, "worker", Json::str(own.join(exe("oaiy-media")).to_string_lossy()));
        util::set(&mut cfg, "media", media);
        config::save(&own_path, &cfg).unwrap();
        assert!(!use_programs_from(&own_path, &bin).unwrap());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_hosts_models_folder_takes_the_downloads_of_an_install_that_has_none() {
        let base = std::env::temp_dir().join(format!("oaiy-downloads-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (conf_dir, models) = (base.join("engines"), base.join("models"));
        std::fs::create_dir_all(&conf_dir).unwrap();
        let path = conf_dir.join(config::FILE_NAME);
        let named = |path: &std::path::Path| str_or(config::load(path).unwrap().get("downloads").unwrap(), "dir", "").to_string();
        // A new install: the host's folder is where downloads go, and a second call changes nothing.
        assert!(use_downloads_dir(&path, &models).unwrap());
        assert_eq!(std::path::PathBuf::from(named(&path)), models);
        assert!(!use_downloads_dir(&path, &base.join("other")).unwrap(), "a folder that is named stays");
        assert_eq!(std::path::PathBuf::from(named(&path)), models);
        // An install with something in the engines' own folder keeps it there: the catalog looks for it in that folder.
        let old = base.join("old");
        std::fs::create_dir_all(old.join("models").join("Qwen3.5-9B-GGUF")).unwrap();
        let old_path = old.join(config::FILE_NAME);
        assert!(!use_downloads_dir(&old_path, &models).unwrap());
        assert_eq!(named(&old_path), "");
        // An empty folder of its own is not a download.
        let empty = base.join("empty");
        std::fs::create_dir_all(empty.join("models")).unwrap();
        assert!(use_downloads_dir(&empty.join(config::FILE_NAME), &models).unwrap());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_host_that_moved_takes_its_programs_with_it() {
        let base = std::env::temp_dir().join(format!("oaiy-programs-moved-{}", std::process::id()));
        let (conf_dir, first, second, own) = (base.join("conf"), base.join("mount-1"), base.join("mount-2"), base.join("own"));
        for dir in [&conf_dir, &first, &second, &own] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let exe = |n: &str| if cfg!(windows) { format!("{n}.exe") } else { n.to_string() };
        let named = |path: &std::path::Path, section: &str, key: &str| {
            let cfg = config::load(path).unwrap();
            std::path::PathBuf::from(cfg.get(section).and_then(|s| s.get(key)).and_then(Json::as_str).unwrap())
        };
        std::fs::write(first.join(exe("oaiy-llm-server-webgpu")), b"").unwrap();
        std::fs::write(second.join(exe("oaiy-llm-server-webgpu")), b"").unwrap();
        let path = conf_dir.join(config::FILE_NAME);
        // The first start (an AppImage's mount, say) writes where the program was then.
        assert!(use_programs_from(&path, &first).unwrap());
        assert_eq!(named(&path, "llm", "server_webgpu"), first.join(exe("oaiy-llm-server-webgpu")));
        // The next start is somewhere else, and the first place is gone: the program is taken from the new one.
        std::fs::remove_dir_all(&first).unwrap();
        assert!(use_programs_from(&path, &second).unwrap());
        assert_eq!(named(&path, "llm", "server_webgpu"), second.join(exe("oaiy-llm-server-webgpu")));
        assert!(!use_programs_from(&path, &second).unwrap(), "a second call changes nothing");
        // A program of the owner's own, which is there, stays theirs wherever the host is.
        let theirs = own.join(exe("oaiy-llm-server-webgpu"));
        std::fs::write(&theirs, b"").unwrap();
        let mut cfg = config::load(&path).unwrap();
        let mut llm = cfg.get("llm").cloned().unwrap();
        util::set(&mut llm, "server_webgpu", Json::str(theirs.to_string_lossy()));
        util::set(&mut cfg, "llm", llm);
        config::save(&path, &cfg).unwrap();
        assert!(!use_programs_from(&path, &second).unwrap());
        assert_eq!(named(&path, "llm", "server_webgpu"), theirs);
        // A path to a program this folder has none of is left as it is: there is nothing better to say.
        std::fs::remove_file(&theirs).unwrap();
        std::fs::remove_file(second.join(exe("oaiy-llm-server-webgpu"))).unwrap();
        assert!(!use_programs_from(&path, &second).unwrap());
        assert_eq!(named(&path, "llm", "server_webgpu"), theirs);
        let _ = std::fs::remove_dir_all(&base);
    }
}
