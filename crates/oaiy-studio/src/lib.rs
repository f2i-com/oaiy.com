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
    /// Counted up at each save: a page that saves settings it read before another save is refused
    /// ([`CONFIG_VERSION`]), rather than putting back what that save changed.
    version: std::sync::atomic::AtomicU64,
    /// `--ui-port` / `--port`: in effect, never written to the file.
    port_overrides: (Option<u16>, Option<u16>),
    /// Stopped for good ([`Studio::shutdown`]): its listeners closed, a request still on a connection it had refused,
    /// and nothing it supervises started again. A host that starts its engines again (OAIY Desktop, when an update is
    /// put back) starts a new studio, which reads the settings from their file, rather than finding this one: it kept
    /// answering, and was taken for one running, with its downloads refused for good.
    closed: std::sync::atomic::AtomicBool,
    /// Where each of its listeners listens, and the thread that takes its connections ([`serve`]): ended by
    /// [`Studio::shutdown`], so the ports are free for the next studio.
    listeners: std::sync::Mutex<Vec<(std::net::SocketAddr, std::thread::JoinHandle<()>)>>,
}

/// How long a studio that stops waits for the eGPU's server to let the card go ([`Studio::shutdown`]): an ordinary quit
/// does not wait for the whole of it.
const QUIT_WAIT: Duration = Duration::from_secs(3);

/// Wait (25 s at most) for the eGPU's servers this process stopped to have let the card go, whichever studio stopped
/// them: for OAIY Desktop's update, after its engines' stop and before it hands off to the installer
/// ([`Studio::shutdown`] waits a few seconds alone). Whether they had.
pub fn await_egpu_let_go() -> bool {
    egpu::await_let_go(egpu::GONE_WAIT)
}

/// The key `GET /api/config` adds to the settings it answers: their version, sent back with a save of them.
pub const CONFIG_VERSION: &str = "config_version";

/// The settings' first version at a start: the time it started, in ms, so a page left open across a restart does
/// not hold a version the new start counts to by chance (each start counted from 1 before).
fn first_version() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_millis() as u64)
}

/// What a secret of the settings reads as where they are answered (the Hugging Face token, the gateway's key): the
/// page shows it is set, and a save that sends it back keeps the secret as it was.
pub const SECRET_KEPT: &str = "(saved)";

/// The secrets of the settings, by section and key: never answered, kept by a save that does not change them.
const SECRETS: [(&str, &str); 2] = [("downloads", "hf_token"), ("gateway", "api_key")];

impl Studio {
    pub fn config(&self) -> Json {
        self.config.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Whether the studio has stopped for good ([`Studio::shutdown`]).
    pub fn closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
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

    /// The settings as the control page reads them: their secrets said to be set ([`SECRET_KEPT`]), not said, and their
    /// version ([`CONFIG_VERSION`]), which a save of them sends back.
    pub fn config_for_page(&self) -> Json {
        // (the settings and their version read together: a save writes both under the same lock, so a page never has
        // new settings with the version before them, or the other way round)
        let (mut cfg, version) = {
            let held = self.config.read().unwrap_or_else(|p| p.into_inner());
            (held.clone(), self.version.load(std::sync::atomic::Ordering::SeqCst))
        };
        for (section, key) in SECRETS {
            if let Some(sec) = cfg.get(section).cloned() {
                if !str_or(&sec, key, "").is_empty() {
                    let mut sec = sec;
                    util::set(&mut sec, key, Json::str(SECRET_KEPT));
                    util::set(&mut cfg, section, sec);
                }
            }
        }
        util::set(&mut cfg, CONFIG_VERSION, Json::Int(version as i64));
        cfg
    }

    /// Save settings the control page (or another client) read and changed: refused where they were read before
    /// another save (their [`CONFIG_VERSION`] not the settings' now: read again, the change made on those), so one
    /// does not put back what another changed; settings sent without a version are saved as they are (an older
    /// client).
    pub fn save_config_from(&self, next: Json) -> Result<Json, (u16, String)> {
        let _one = self.saving.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(read) = next.get(CONFIG_VERSION).and_then(Json::as_i64) {
            let now = self.version.load(std::sync::atomic::Ordering::SeqCst) as i64;
            if read != now {
                return Err((409, "the settings were changed since this page read them (a download, another window, or the desktop): they are read again; make the change once more".into()));
            }
        }
        self.save_locked(next).map_err(|e| (400, e))
    }

    /// Change the settings as they are now, under the save's lock: `change` sees them as saved (no other change
    /// between its reading and this save), and what it answers is answered with the saved settings.
    /// Errors: `change`'s own (with its status), else the save's (400).
    pub fn update_config<T>(&self, change: impl FnOnce(&mut Json) -> Result<T, (u16, String)>) -> Result<(T, Json), (u16, String)> {
        let _one = self.saving.lock().unwrap_or_else(|p| p.into_inner());
        let before = self.config();
        let mut next = before.clone();
        let answer = change(&mut next)?;
        // (nothing changed, a model unpinned that was not pinned: not saved, so the version stays, and a page that
        // read the settings is not refused for a save that changed nothing)
        if next == before {
            return Ok((answer, before));
        }
        let saved = self.save_locked(next).map_err(|e| (400, e))?;
        Ok((answer, saved))
    }

    /// Validate, save and apply a new configuration. A running LLM restarts when
    /// its settings changed; listener changes wait for the next start.
    pub fn set_config(&self, next: Json) -> Result<Json, String> {
        let _one = self.saving.lock().unwrap_or_else(|p| p.into_inner());
        self.save_locked(next)
    }

    /// [`Studio::set_config`], the save's lock held.
    fn save_locked(&self, mut next: Json) -> Result<Json, String> {
        // (the version and the secrets as the page reads them are not settings: a secret sent back as it was read
        // is the secret as it is)
        if let Json::Obj(fields) = &mut next {
            fields.retain(|(k, _)| k != CONFIG_VERSION);
        }
        let current = self.config();
        for (section, key) in SECRETS {
            if next.get(section).is_some_and(|s| str_or(s, key, "") == SECRET_KEPT) {
                let kept = current.get(section).map_or("", |s| str_or(s, key, "")).to_string();
                if let Some(mut sec) = next.get(section).cloned() {
                    util::set(&mut sec, key, Json::str(kept));
                    util::set(&mut next, section, sec);
                }
            }
        }
        // (entries set aside as the settings were read stay set aside, and in the file: a page that does not send them
        // back does not drop them)
        if next.get(config::SET_ASIDE).is_none() {
            if let Some(aside) = current.get(config::SET_ASIDE) {
                util::set(&mut next, config::SET_ASIDE, aside.clone());
            }
        }
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
        {
            // (the version moves with the settings, under the same lock: see `config_for_page`)
            let mut held = self.config.write().unwrap_or_else(|p| p.into_inner());
            *held = next.clone();
            self.version.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
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
            let args = llm::arguments(&llm, &self.root, 0, true).map(|(a, _)| a).unwrap_or_default();
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
                // Saved either way; a failed restart is reported, not a failed save. Not beside a media job on the GPU
                // the new settings load it on (another default model, on more GPUs): then once media is done.
                match start_llm_beside_media(&self.llm, &self.media, &next, &self.root) {
                    Ok(true) => {}
                    Ok(false) => self.log.push("the LLM restarts once the media jobs on its GPU are done"),
                    Err(e) => self.log.push(format!("the LLM did not restart: {e}")),
                }
            }
        }
        // tinygrad's server holds one model under one set of settings: with either changed it stops, and the next
        // request for a model of its own starts it again.
        let egpu_launch = |v: &Json| {
            let llm = v.get("llm").cloned().unwrap_or(Json::Null);
            let file = self.egpu.held().as_deref().filter(|m| egpu::assigned(&llm, m)).and_then(|m| egpu::arguments(&llm, &self.root, m, std::path::Path::new(""), 0).ok());
            // (incognito too, where the card's engine is started for it: `--incognito`, as this computer's is)
            (file, llm.get("egpu").cloned(), egpu::private(v))
        };
        if self.egpu.state() != llm::State::Stopped && egpu_launch(&before) != egpu_launch(&next) {
            // (not under a picture worker on the lent card: then as it is given back, whatever is asked after)
            if self.egpu.stop_unless_lent(egpu::Deferred::Changed) {
                self.log.push("eGPU settings changed: stopping tinygrad's server");
            } else {
                self.log.push("eGPU settings changed: tinygrad's server stops once the card is given back (it is lent to a picture worker)");
            }
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
    ///
    /// This computer's is not loaded beside a media job, though: while media has paused the language model, or a job
    /// holds the GPU the default would load on, it is left for media's end, which brings the language model back with
    /// the settings as they are then, this default ([`start_llm_beside_media`]). Picking another model while a clip was
    /// made on GPU 1 loaded a model of GPUs 0 and 1 beside it.
    fn place_the_model(&self, before: &Json, next: &Json) {
        let placed = |v: &Json| {
            let llm = v.get("llm").cloned().unwrap_or(Json::Null);
            let default = llm::default_model(&llm);
            (egpu::assigned(&llm, &default), default, bool_or(&llm, "enabled", true))
        };
        let (on_card, default, enabled) = placed(next);
        if placed(before) == (on_card, default.clone(), enabled) || !enabled || default.is_empty() || self.closed() {
            return;
        }
        if on_card && self.llm.is_running() {
            self.log.push(format!("{default} runs on the eGPU: this computer's engine lets go of its model"));
            self.llm.stop();
        }
        if !on_card && self.egpu.state() != llm::State::Stopped {
            // (not while the card is lent, a picture worker on the socket its server holds, a pinned model's or one kept
            // for the next picture: its engine is paused then, its model not on the card, and the server stops as the
            // card is given back, unless the card's model is chosen again meanwhile)
            if self.egpu.stop_unless_lent(egpu::Deferred::Unwanted) {
                self.log.push(format!("{default} runs on this computer: the eGPU lets go of its model"));
            } else {
                self.log.push(format!("{default} runs on this computer: the eGPU lets go of its model once the card is given back (it is lent to a picture worker)"));
            }
        }
        // (the load begun beside the save, which it does not hold up: the card's waits for a server it replaces to
        // be gone, up to 25 s)
        let (egpu, mac, media, cfg, root, log) = (Arc::clone(&self.egpu), Arc::clone(&self.llm), Arc::clone(&self.media), next.clone(), self.root.clone(), Arc::clone(&self.log));
        let begun = std::thread::Builder::new().name("place-model".into()).spawn(move || {
            let (where_, started) = if on_card { ("on the eGPU", egpu.start(&cfg, &root, &default).map(|()| true)) } else { ("on this computer", start_llm_beside_media(&mac, &media, &cfg, &root)) };
            match started {
                Ok(true) => log.push(format!("loading {default} {where_}")),
                Ok(false) => log.push(format!("{default} loads {where_} once the media jobs on its GPU are done")),
                Err(e) => log.push(format!("{default} did not start loading {where_}: {e}")),
            }
        });
        if let Err(e) = begun {
            self.log.push(format!("the language model was not loaded ahead of time: {e}"));
        }
    }

    /// Stop the language model and any media job, so no engine outlives the
    /// studio holding GPU memory. Waits briefly for a media worker to be killed.
    ///
    /// For good: its listeners are closed (the ports free for a studio started after it), and nothing is started again,
    /// the language model a media job paused as the job ends, a pinned picture model as the pins are looked after, a
    /// job that comes. Once: asked again (the tray asks as the session ends, then as it quits), it does nothing.
    pub fn shutdown(&self) {
        if self.closed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // The listeners first: a host that starts its engines again after this (OAIY Desktop, an update put back) finds
        // none answering and starts a new studio on the same ports, rather than taking this one for a running one.
        self.stop_listening();
        // (media's runner and the pins' keeper end, and the language model is started by nothing after: a job that
        // paused it ended after the stop and its runner started oaiy-llm-server again, a pin's keeper started oaiy-media
        // again within 3 s, and an update then found the engines' programs running. Before the downloads' stop, which
        // can take a few seconds, so no job starts meanwhile)
        self.media.close();
        self.llm.close();
        // Downloads next: one being added to the models saves the settings, and none may after this (a restore puts
        // its settings back once the studio stops), nor may its curl outlive it (the next start's resume would run a
        // second curl on the same partial file).
        self.downloads.shutdown();
        // (for good: no server is started on the card after this, and its last is waited for below)
        self.egpu.close();
        // (every job, incognito ones too: a pin's loading is one, and the list the page shows has none of them)
        let cancelled = self.media.cancel_all();
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
        // The card's server let go of the card, a few seconds of it: an ordinary quit waited up to 25 s for it, and the
        // server lets the card go by itself as OAIY ends (its lifeline closes). An update waits for the rest before it
        // hands off ([`await_egpu_let_go`]): the OAIY it started found tinygrad's lock still taken.
        if !self.egpu.await_gone(QUIT_WAIT) {
            self.log.push("studio: the eGPU's server was still letting the card go as the studio stopped");
        }
        self.log.push("studio stopped");
    }

    /// Its listeners closed ([`serve`]), each accept loop woken by a connection of this studio's own to find it stopped,
    /// and waited for (a few seconds at most) as it lets go of its port: a studio started just after this one listens on
    /// the same ports, not on others beside them (the gateway's port is what the Agent and Aokie are told).
    fn stop_listening(&self) {
        let loops = std::mem::take(&mut *self.listeners.lock().unwrap_or_else(|p| p.into_inner()));
        for (addr, _) in &loops {
            // (a listener on every address is reached on loopback; dropped at once, so the closing that waits is this
            // side's, not the port's)
            let mut to = *addr;
            if to.ip().is_unspecified() {
                to.set_ip(if to.is_ipv4() { std::net::Ipv4Addr::LOCALHOST.into() } else { std::net::Ipv6Addr::LOCALHOST.into() });
            }
            let _ = std::net::TcpStream::connect_timeout(&to, Duration::from_secs(1));
        }
        let until = std::time::Instant::now() + Duration::from_secs(3);
        for (addr, accepting) in loops {
            while !accepting.is_finished() && std::time::Instant::now() < until {
                std::thread::sleep(Duration::from_millis(10));
            }
            if accepting.is_finished() {
                let _ = accepting.join();
            } else {
                self.log.push(format!("the listener on {addr} did not stop: its port is let go as OAIY ends"));
            }
        }
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
            // the paired computers' models (OAIY Network), for the Playground and for what is OAIY's default
            ("peers", api::peers::models_json(downloads::peer_models().as_deref())),
            // the picture models kept loaded, and where each stands (`media::pins`)
            ("pins", media::pins::snapshot(&self.config(), &self.media)),
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

/// This computer's language model started with `cfg` now (a change of settings, another default model), or left for
/// media's end where it would load beside a media job: media has paused it (a job on its GPU, or one after another), or
/// a job holds the GPU it would load on (`media::needs_media_gpu`). Media's runner brings back the language model it
/// paused once its queue is empty, with the settings as they are then, so what was chosen meanwhile is what loads
/// ([`media::Media::run`]); one that was not running is marked paused for it here. Whether it was started.
///
/// A job is held off while it starts, as by a chat that shares its GPU ([`media::Media::chat_lease`]): one that comes
/// meanwhile finds it starting and stops it, rather than the two loading on the GPU together.
fn start_llm_beside_media(llm: &Arc<llm::Llm>, media: &media::Media, cfg: &Json, root: &std::path::Path) -> Result<bool, String> {
    if llm.paused() {
        return Ok(false);
    }
    let Ok(_held) = media.chat_lease(media::needs_media_gpu(cfg, None), Duration::ZERO) else {
        // (the runner is in that job meanwhile, and looks at the queue, and so at this, once it ends)
        llm.set_paused(true);
        return Ok(false);
    };
    llm.start(cfg, root).map(|()| true)
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
            version: std::sync::atomic::AtomicU64::new(first_version()),
            port_overrides: (None, None),
            closed: std::sync::atomic::AtomicBool::new(false),
            listeners: std::sync::Mutex::new(Vec::new()),
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

/// Connections served at once on each listener (each is a thread); one more is answered 503 and closed.
const MAX_CONNECTIONS: usize = 128;

/// A connection counted in a total until it is dropped (its thread ended, or never started).
struct Counted(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Serve a listener, each connection on a thread of its own, at most [`MAX_CONNECTIONS`] at once. Each request's head
/// is gated before its body is read ([`admin::gate`], [`gateway_gate`]): a request refused is refused without the
/// 64 MiB it may say it brings.
///
/// Until the studio stops ([`Studio::shutdown`], which wakes it): then the listener is let go, and a request that comes
/// on a connection it had is refused (`503`), so a client's next is made to whatever listens on the port then.
fn serve(listener: TcpListener, studio: Arc<Studio>, admin: bool) {
    let addr = listener.local_addr();
    let port = addr.as_ref().map(|a| a.port()).unwrap_or(0);
    let open = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let listening = Arc::clone(&studio);
    let accepting = std::thread::spawn(move || {
        let mut pause = Duration::from_millis(5);
        for conn in listener.incoming() {
            if studio.closed() {
                break;
            }
            let mut stream = match conn {
                Ok(stream) => stream,
                // Out of file descriptors (EMFILE) fails every accept at once until one closes: a pause, longer each
                // time, rather than a core spun at 100%.
                Err(e) if !matches!(e.kind(), std::io::ErrorKind::Interrupted | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::ConnectionReset) => {
                    std::thread::sleep(pause);
                    pause = (pause * 2).min(Duration::from_secs(1));
                    continue;
                }
                Err(_) => continue,
            };
            pause = Duration::from_millis(5);
            let counted = Counted(Arc::clone(&open));
            if open.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= MAX_CONNECTIONS {
                drop(counted);
                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                let body = util::error_json("too many connections: try again shortly", "server_error", "busy").to_json();
                let _ = oaiy_engine::http::respond(&mut stream, 503, "application/json", body.as_bytes(), false);
                continue;
            }
            // (who is at the other end, for the control port: this computer's own clients connect over loopback)
            let peer_loopback = stream.peer_addr().is_ok_and(|a| a.ip().to_canonical().is_loopback());
            let studio = Arc::clone(&studio);
            let log = Arc::clone(&studio.log);
            // (a thread that cannot start drops its connection with it, rather than taking the accept loop down)
            let started = std::thread::Builder::new().name(if admin { "ui-request" } else { "gateway-request" }.into()).spawn(move || {
                let _counted = counted;
                oaiy_engine::http::serve_gated(
                    stream,
                    |req| {
                        if studio.closed() {
                            return Some(oaiy_engine::http::Refusal { status: 503, body: util::error_json("the studio has stopped", "server_error", "stopped").to_json() });
                        }
                        if admin { admin::gate(&studio, req, port, peer_loopback) } else { gateway_gate(&studio, req) }
                    },
                    |req, w| {
                        if admin {
                            return admin::handle(&studio, req, w);
                        }
                        public(&studio, req, w)
                    },
                );
            });
            if let Err(e) = started {
                log.push(format!("a connection was dropped: its thread would not start ({e})"));
            }
        }
    });
    if let Ok(addr) = addr {
        listening.listeners.lock().unwrap_or_else(|p| p.into_inner()).push((addr, accepting));
    }
}

/// The names a gateway that listens on loopback answers to besides loopback's own: Docker's for this computer (a
/// container's app reaching it at host.docker.internal), and the host of `gateway.public_url` (a proxy in front of it
/// on this computer). A web page can be served from none of them by a site that is not this computer's.
fn gateway_host_ok(host: &str, gateway: &Json) -> bool {
    if oaiy_engine::http::loopback_name(host) {
        return true;
    }
    // (a host's name, its port aside: `[v6]:port`, `name:port`, or a bare name or address)
    let name = |h: &str| {
        let h = h.trim().to_ascii_lowercase();
        match h.strip_prefix('[') {
            Some(rest) => rest.split(']').next().unwrap_or("").to_string(),
            None if h.matches(':').count() == 1 => h.split(':').next().unwrap_or("").to_string(),
            None => h,
        }
    };
    let host = name(host);
    let public = str_or(gateway, "public_url", "").trim();
    let public = public.split_once("://").map_or(public, |(_, rest)| rest).split('/').next().unwrap_or("");
    host == "host.docker.internal" || host == "gateway.docker.internal" || (!public.is_empty() && host == name(public))
}

/// What the gateway refuses on a request's head, before its body is read (`None`: served): on a gateway that listens
/// on loopback only, a `Host` naming another computer without the key (DNS rebinding: a site's name pointed at
/// 127.0.0.1, whose page then reads the pictures, videos and songs made here as its own -- a page's request to its
/// own site carries no `Origin` to refuse); with no key, a web page of an origin not in `gateway.cors_origins`; with a
/// key, a request without it (but for the routes that answer without one: health, discovery, and a file link's
/// `?key=`).
fn gateway_gate(studio: &Studio, req: &oaiy_engine::http::Request) -> Option<oaiy_engine::http::Refusal> {
    use oaiy_engine::http::{key_matches, same_secret, Refusal};
    let refuse = |status: u16, message: &str, kind: &str, code: &str| Some(Refusal { status, body: util::error_json(message, kind, code).to_json() });
    let cfg = studio.config();
    let gateway = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let key = str_or(&gateway, "api_key", "").to_string();
    let target = gateway::route(gateway.get("routes").and_then(Json::as_array).unwrap_or(&[]), &req.method, req.route()).map(|m| m.target);
    let keyed = !key.is_empty()
        && (key_matches(req.header("authorization"), &key)
            || (target.as_deref() == Some("files") && req.query("key").is_some_and(|k| same_secret(k.as_bytes(), key.as_bytes()))));
    if !keyed && oaiy_engine::http::listens_on_loopback(str_or(&gateway, "host", "127.0.0.1")) && req.header("host").is_some_and(|h| !gateway_host_ok(h, &gateway)) {
        return refuse(403, "this gateway answers to its loopback address only (localhost or 127.0.0.1): set gateway.public_url for a proxy in front of it, or send the API key", "permission_error", "forbidden_host");
    }
    if req.method == "OPTIONS" {
        return None;
    }
    // A web page the user happens to visit can reach 127.0.0.1 too. Without a key, browser requests (they carry
    // Origin) are served only to the origins listed in gateway.cors_origins; apps and scripts send no Origin.
    if key.is_empty() {
        if let Some(origin) = req.header("origin") {
            let allowed = gateway.get("cors_origins").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str)
                .any(|o| o == "*" || o.trim_end_matches('/').eq_ignore_ascii_case(origin));
            if !allowed {
                return refuse(403, &format!("requests from {origin} are not allowed; add it to gateway.cors_origins, or set an API key"), "permission_error", "origin_not_allowed");
            }
        }
        return None;
    }
    // (no route, discovery and health answer without the key: discovery says only that one is needed)
    let open = req.method == "GET" && req.route() == "/.well-known/oaiy.json";
    match target.as_deref() {
        Some("health" | "discovery") | None => None,
        Some(_) if open || keyed => None,
        Some(_) => refuse(401, "missing or wrong API key", "authentication_error", "invalid_api_key"),
    }
}

/// The gateway listener: the route table (its key and origins checked on the head first, by [`gateway_gate`]).
fn public(studio: &Arc<Studio>, req: &oaiy_engine::http::Request, w: &mut std::net::TcpStream) -> std::io::Result<bool> {
    if req.method == "OPTIONS" {
        oaiy_engine::http::respond(w, 204, "text/plain", b"", true)?;
        return Ok(true);
    }
    let cfg = studio.config();
    let gateway = cfg.get("gateway").cloned().unwrap_or(Json::Null);
    let routes = gateway.get("routes").and_then(Json::as_array).map(<[Json]>::to_vec).unwrap_or_default();
    let key = str_or(&gateway, "api_key", "").to_string();
    let keyed = |req: &oaiy_engine::http::Request| key.is_empty() || oaiy_engine::http::key_matches(req.header("authorization"), &key);
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
        version: std::sync::atomic::AtomicU64::new(first_version()),
        port_overrides: (args.ui_port, args.port),
        closed: std::sync::atomic::AtomicBool::new(false),
        listeners: std::sync::Mutex::new(Vec::new()),
    });
    // (a model entry the check refused was set aside as the settings were read, the rest used: said where the Engines
    // page shows the log)
    for item in cfg.get(config::SET_ASIDE).and_then(Json::as_array).unwrap_or(&[]) {
        studio.log.push(format!("set aside (kept in the settings file, not used until it checks): {}", str_or(item, "why", "")));
    }
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
    // (the pinned picture models loaded, and loaded again whenever their worker ends: `media::pins`)
    let pinner = Arc::clone(&studio);
    std::thread::Builder::new().name("pins".into()).spawn(move || media::pins::keep(pinner)).map_err(|e| e.to_string())?;
    // Stop an idle LLM so its memory returns to the machine; the next request loads it again.
    // Incognito folders left by a crash are gone before anything serves.
    let _ = std::fs::remove_dir_all(studio.output_root().join(".incognito"));
    let idle = Arc::clone(&studio);
    std::thread::spawn(move || {
        let mut ticks = 0u64;
        loop {
            std::thread::sleep(Duration::from_secs(30));
            // (not past the studio's stop: this holds it, and with it what it held)
            if idle.closed() {
                return;
            }
            ticks += 1;
            idle.media.sweep();
            // (a pinned language model is never stopped for being idle)
            let minutes = idle.config().get("llm").filter(|l| !bool_or(l, "pinned", false)).map_or(0, |l| int_or(l, "idle_stop_minutes", 0));
            // Never mid-reply: a long stream updates nothing until it ends, and holds its chat's lease (asked with the
            // stop, under the engine's lock: a request that comes meanwhile finds it stopped and starts it again).
            if minutes > 0 && idle.llm.stop_if_idle(Duration::from_secs(minutes as u64 * 60), || idle.media.chats_active() > 0) {
                idle.log.push(format!("LLM idle for {minutes} min: stopped it to free memory"));
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

    /// The control page reads the settings with their version and without their secrets; a save of settings read
    /// before another save is refused, rather than putting back what that save changed; a secret sent back as it was
    /// read is kept.
    #[test]
    fn settings_read_before_another_save_are_refused_and_secrets_are_never_answered() {
        let root = std::env::temp_dir().join(format!("oaiy-studio-version-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut cfg = config::default_json();
        let mut d = Json::obj(Vec::<(&str, Json)>::new());
        util::set(&mut d, "hf_token", Json::str("hf_secret"));
        util::set(&mut cfg, "downloads", d);
        let studio = Studio::for_test(&root, cfg);
        let page = studio.config_for_page();
        assert!(!page.to_json().contains("hf_secret"), "the token is not answered");
        assert_eq!(page.get("downloads").map(|d| str_or(d, "hf_token", "").to_string()).as_deref(), Some(SECRET_KEPT));
        // another save meanwhile (a download added a model): the page's save of what it read is refused
        studio.update_config(|next| {
            util::set(next, "note", Json::str("a download's"));
            Ok(())
        })
        .unwrap();
        let mut stale = page.clone();
        util::set(&mut stale, "privacy", Json::obj([("incognito", Json::Bool(true))]));
        assert_eq!(studio.save_config_from(stale).map(|_| ()).unwrap_err().0, 409);
        assert_eq!(studio.config().get("note").and_then(Json::as_str), Some("a download's"), "nothing put back");
        // read again, saved: the token sent back as it was read is the token as it is
        let mut fresh = studio.config_for_page();
        util::set(&mut fresh, "privacy", Json::obj([("incognito", Json::Bool(true))]));
        studio.save_config_from(fresh).unwrap();
        assert_eq!(studio.config().get("downloads").map(|d| str_or(d, "hf_token", "").to_string()).as_deref(), Some("hf_secret"));
        assert!(studio.config().get(CONFIG_VERSION).is_none(), "the version is not a setting");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The settings' version starts from when the studio started (a page from before a restart does not match it by
    /// chance), moves by one with each save, and not with a change that changed nothing.
    #[test]
    fn the_version_starts_from_the_start_and_moves_with_each_real_save_alone() {
        let root = std::env::temp_dir().join(format!("oaiy-studio-version-moves-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let studio = Studio::for_test(&root, config::default_json());
        let version = |s: &Studio| s.config_for_page().get(CONFIG_VERSION).and_then(Json::as_i64).unwrap();
        let first = version(&studio);
        assert!(first > 1_700_000_000_000, "from the time it started, in ms: {first}");
        // a change that changes nothing is not saved: the version stays, a page that read it is not refused
        let page = studio.config_for_page();
        studio.update_config(|next| {
            let same = next.clone();
            *next = same;
            Ok(())
        })
        .unwrap();
        assert_eq!(version(&studio), first);
        let mut saved = page.clone();
        util::set(&mut saved, "privacy", Json::obj([("incognito", Json::Bool(true))]));
        studio.save_config_from(saved).unwrap();
        assert_eq!(version(&studio), first + 1, "a real save moves it by one");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The gateway's head checks: a site's own name for this computer (DNS rebinding) is refused where it listens on
    /// loopback, unless the key comes with it; another site's page without a key; a missing key on a keyed route.
    #[test]
    fn the_gateway_refuses_other_names_other_sites_and_a_missing_key_on_the_head() {
        let root = std::env::temp_dir().join(format!("oaiy-studio-gateway-gate-{}", std::process::id()));
        let studio = Studio::for_test(&root, config::default_json());
        let req = |method: &str, path: &str, h: &[(&str, &str)]| oaiy_engine::http::Request::new(method, path, h.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), Vec::new());
        let status = |s: &Studio, r: &oaiy_engine::http::Request| gateway_gate(s, r).map(|r| r.status);
        assert_eq!(status(&studio, &req("GET", "/v1/videos", &[("Host", "evil.example:8080")])), Some(403), "rebinding");
        assert_eq!(status(&studio, &req("OPTIONS", "/v1/videos", &[("Host", "evil.example:8080")])), Some(403));
        for host in ["127.0.0.1:8080", "localhost:8080", "[::1]:8080", "host.docker.internal:8080"] {
            assert_eq!(status(&studio, &req("GET", "/v1/videos", &[("Host", host)])), None, "{host}");
        }
        assert_eq!(status(&studio, &req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8080"), ("Origin", "https://evil.example")])), Some(403));
        assert_eq!(status(&studio, &req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8080"), ("Origin", "https://bot.computer")])), None, "a listed origin");
        // listening beyond loopback: any name (the key, required there for the controls, guards it)
        let mut cfg = config::default_json();
        let mut g = cfg.get("gateway").unwrap().clone();
        util::set(&mut g, "host", Json::str("0.0.0.0"));
        util::set(&mut cfg, "gateway", g.clone());
        assert_eq!(status(&Studio::for_test(&root, cfg.clone()), &req("GET", "/v1/videos", &[("Host", "studio.lan:8080")])), None);
        // a public_url's host is its proxy's
        util::set(&mut g, "host", Json::str("127.0.0.1"));
        util::set(&mut g, "public_url", Json::str("https://ai.example.org/"));
        util::set(&mut cfg, "gateway", g.clone());
        assert_eq!(status(&Studio::for_test(&root, cfg.clone()), &req("GET", "/v1/videos", &[("Host", "ai.example.org")])), None);
        // a key: wanted on the head, a rebinding page has none; health and discovery answer without it
        util::set(&mut g, "public_url", Json::str(""));
        util::set(&mut g, "api_key", Json::str("sk-test"));
        util::set(&mut cfg, "gateway", g);
        let keyed = Studio::for_test(&root, cfg);
        assert_eq!(status(&keyed, &req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8080")])), Some(401));
        assert_eq!(status(&keyed, &req("POST", "/v1/chat/completions", &[("Host", "127.0.0.1:8080"), ("Authorization", "Bearer sk-test")])), None);
        assert_eq!(status(&keyed, &req("POST", "/v1/chat/completions", &[("Host", "evil.example"), ("Authorization", "Bearer sk-test")])), None, "the key is enough");
        assert_eq!(status(&keyed, &req("GET", "/v1/models", &[("Host", "evil.example")])), Some(403));
        assert_eq!(status(&keyed, &req("GET", "/.well-known/oaiy.json", &[("Host", "127.0.0.1:8080")])), None);
        assert_eq!(status(&keyed, &req("GET", "/", &[("Host", "127.0.0.1:8080")])), None);
    }

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

    /// A studio stopped lets go of its ports, and a request on a connection it had is refused: a host that starts its
    /// engines again after it (OAIY Desktop, an update put back) finds none running and starts a new one, on the same
    /// ports. The stopped one kept answering, was taken for one running, and refused every download for good.
    #[test]
    fn a_stopped_studio_lets_go_of_its_ports_and_a_new_one_takes_them() {
        use std::io::{Read, Write};
        let dir = std::env::temp_dir().join(format!("oaiy-studio-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let free = || TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port();
        let (ui, gw) = (free(), free());
        let mut cfg = config::default_json();
        for (section, port) in [("ui", ui), ("gateway", gw)] {
            let mut s = cfg.get(section).cloned().unwrap();
            util::set(&mut s, "port", Json::Int(port as i64));
            util::set(&mut cfg, section, s);
        }
        // (nothing real is started: names no program has)
        let mut media = cfg.get("media").cloned().unwrap();
        util::set(&mut media, "worker", Json::str("no-such-worker-here"));
        util::set(&mut cfg, "media", media);
        let mut llm = cfg.get("llm").cloned().unwrap();
        util::set(&mut llm, "server", Json::str(dir.join("no-such-llm-server").to_string_lossy()));
        util::set(&mut llm, "server_webgpu", Json::str(dir.join("no-such-llm-server").to_string_lossy()));
        util::set(&mut cfg, "llm", llm);
        let path = dir.join(config::FILE_NAME);
        config::save(&path, &cfg).unwrap();
        let args = Args { config: path, open: Some("none".into()), ui_port: None, port: None, start_llm: false };
        let first = launch(&args, true).unwrap();
        assert_eq!((first.ui_url.clone(), first.gateway_url.clone()), (format!("http://127.0.0.1:{ui}"), format!("http://127.0.0.1:{gw}")));
        assert!(running_instance(&args).is_some());
        // A connection kept open, as a client keeps one for its next request.
        let mut kept = std::net::TcpStream::connect(("127.0.0.1", ui)).unwrap();
        kept.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let ask = |s: &mut std::net::TcpStream| -> String {
            write!(s, "GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:{ui}\r\nConnection: keep-alive\r\n\r\n").unwrap();
            let (mut got, mut buf) = (Vec::new(), [0u8; 8192]);
            // (the head, then the body to its length, so the next answer on the connection starts where this ends)
            loop {
                let n = s.read(&mut buf).unwrap_or(0);
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).to_string();
                if let Some((head, rest)) = text.split_once("\r\n\r\n") {
                    let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                    if rest.len() >= len || n == 0 {
                        return head.lines().next().unwrap_or("").to_string();
                    }
                }
                if n == 0 {
                    return text;
                }
            }
        };
        assert!(ask(&mut kept).contains(" 200 "), "served while it runs");
        // ...and one left open and idle across the next start, as a page's or a client's pool leaves one.
        let mut idle = std::net::TcpStream::connect(("127.0.0.1", ui)).unwrap();
        idle.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        assert!(ask(&mut idle).contains(" 200 "));
        first.studio.shutdown();
        assert!(running_instance(&args).is_none(), "nothing answers for the stopped studio");
        assert!(ask(&mut kept).contains(" 503 "), "a request on its connection is refused");
        let second = launch(&args, true).unwrap();
        let moved = second.studio.log.tail(20);
        assert_eq!((second.ui_url.clone(), second.gateway_url.clone()), (first.ui_url.clone(), first.gateway_url.clone()), "the same ports, a connection of the old one's still open: {moved}");
        assert!(running_instance(&args).is_some());
        assert!(ask(&mut idle).contains(" 503 "), "the old one's connection is still the old one's, and refused");
        assert!(!second.studio.closed() && !second.studio.media.closed());
        second.studio.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
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
