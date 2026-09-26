//! nrob-studio: a portable host for local AI on finite hardware.
//!
//!   nrob-studio [--config FILE] [--open app|browser|none] [--ui-port N] [--port N]
//!
//! The library is the whole studio; `nrob-studio` (a console program) and
//! `nrob-studio-tray` (a Windows notification-area app) are two front ends.
//!
//! One folder holds everything: this program, `nrob-server` (language models),
//! `nrob-diffusion` (images and video) and `nrob-studio.json`. The studio runs
//! the two engines as subprocesses, moves GPUs between them (a media job on the
//! LLM's GPU stops the LLM, which restarts after), serves a control UI, and
//! exposes a gateway whose routes -- paths, methods and API dialects -- come
//! from the configuration. See docs/STUDIO.md.

#![forbid(unsafe_code)]

mod admin;
mod config;
mod detect;
mod discovery;
mod gateway;
mod llm;
mod media;
mod multipart;
mod registry;
mod system;
mod util;

use nrob::json::Json;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use util::{bool_or, int_or, str_or, LogRing};

pub const HELP: &str = "nrob-studio: host language, image and video models behind configurable endpoints

  --config FILE      configuration (default: nrob-studio.json beside this program;
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
    pub media: Arc<media::Media>,
    pub system: system::System,
    pub log: Arc<LogRing>,
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
            self.log.push("LLM settings changed: restarting nrob-server");
            self.llm.stop();
            if bool_or(next.get("llm").unwrap_or(&Json::Null), "enabled", true) {
                // Saved either way; a failed restart is reported, not a failed save.
                if let Err(e) = self.llm.start(&next, &self.root) {
                    self.log.push(format!("the LLM did not restart: {e}"));
                }
            }
        }
        self.log.push("configuration saved");
        Ok(next)
    }

    /// Stop the language model and any media job, so no engine outlives the
    /// studio holding GPU memory. Waits briefly for a media worker to be killed.
    pub fn shutdown(&self) {
        self.llm.stop();
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
            ("media", Json::obj([
                ("busy", Json::Bool(self.media.busy())),
                ("jobs", Json::Arr(self.media.list().iter().take(60).map(|j| j.to_json(&root)).collect())),
                ("private", self.media.private_activity()),
            ])),
        ])
    }
}

/// Open a URL or folder with the desktop's handler.
pub fn open_path(target: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("explorer").arg(target).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(target).spawn();
    #[cfg(not(any(windows, target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(target).spawn();
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
            if std::process::Command::new(exe).arg(format!("--app={url}")).arg("--window-size=1400,900").spawn().is_ok() {
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

fn listen(host: &str, port: u16, what: &str) -> Result<TcpListener, String> {
    TcpListener::bind((host, port)).map_err(|e| format!("{what}: cannot listen on {host}:{port}: {e} (change the port with --{}port or in the configuration)", if what == "UI" { "ui-" } else { "" }))
}

fn serve(listener: TcpListener, studio: Arc<Studio>, admin: bool) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let studio = Arc::clone(&studio);
            std::thread::spawn(move || {
                nrob::http::serve(stream, |req, w| {
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
fn public(studio: &Arc<Studio>, req: &nrob::http::Request, w: &mut std::net::TcpStream) -> std::io::Result<bool> {
    if req.method == "OPTIONS" {
        nrob::http::respond(w, 204, "text/plain", b"", true)?;
        return Ok(true);
    }
    let cfg = studio.config();
    let gateway = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let routes = gateway.get("routes").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default();
    let key = str_or(&gateway, "api_key", "").to_string();
    let keyed = |req: &nrob::http::Request| key.is_empty() || req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")).map(str::trim) == Some(key.as_str());
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
    if req.method == "GET" && req.route() == "/.well-known/nrob.json" {
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
    let cfg = effective_config(args).ok()?;
    let ui = cfg.get("ui")?;
    let host = match str_or(ui, "host", "127.0.0.1") {
        "0.0.0.0" | "::" => "127.0.0.1",
        h => h,
    };
    let addr = format!("{host}:{}", int_or(ui, "port", 7860));
    let reply = nrob::http::fetch(&addr, "GET", "/api/state", &[], b"", Duration::from_secs(2)).ok()?;
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
    let studio = Arc::new(Studio {
        config_path: args.config.clone(),
        root,
        config: RwLock::new(cfg.clone()),
        llm: Arc::new(llm::Llm::new()),
        media: Arc::new(media::Media::new()),
        system: system::System::new(),
        log: Arc::new(LogRing::new(1000)),
        ui_url: RwLock::new(String::new()),
        gateway_url: RwLock::new(String::new()),
        restart_required: RwLock::new(false),
        saving: std::sync::Mutex::new(()),
        port_overrides: (args.ui_port, args.port),
    });
    let ui = cfg.get("ui").cloned().unwrap_or(Json::Null);
    let gw = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let (ui_host, gw_host) = (str_or(&ui, "host", "127.0.0.1").to_string(), str_or(&gw, "host", "127.0.0.1").to_string());
    let ui_listener = listen(&ui_host, int_or(&ui, "port", 7860) as u16, "UI")?;
    let gw_listener = listen(&gw_host, int_or(&gw, "port", 8080) as u16, "gateway")?;
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
            let minutes = idle.config().get("llm").map_or(0, |l| int_or(l, "idle_stop_minutes", 0));
            // Never mid-reply: a long stream updates nothing until it ends.
            if minutes > 0 && idle.media.chats_active() == 0 && idle.llm.state() == llm::State::Ready && idle.llm.idle_for() > Duration::from_secs(minutes as u64 * 60) {
                idle.log.push(format!("LLM idle for {minutes} min: stopping it to free memory"));
                idle.llm.stop();
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
        println!("nrob-studio {}", env!("CARGO_PKG_VERSION"));
        println!("  control UI  {ui_url}");
        println!("  API         {gw_url}/v1   (OpenAI-compatible; routes are configurable in the UI)");
        println!("  config      {}", args.config.display());
        println!("  type 'help' for console commands");
    }
    let llm_cfg = cfg.get("llm").cloned().unwrap_or(Json::Null);
    if args.start_llm || bool_or(&llm_cfg, "autostart", false) {
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
