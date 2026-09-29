//! `oaiy-server` — OAIY Desktop's HTTP API + service supervisor, headless.
//!
//! Same axum API as the tray app (services / models / python on
//! `127.0.0.1:17972`), but with no window, tray, or webview — for running on a
//! Linux box (or any server) driven by a CO-LOCATED Node CLI / oaiy-web. The API
//! binds 127.0.0.1 by default; OAIY_SERVER_BIND=lan binds every interface for the
//! case this exists to serve — editing from a phone on the same network. Prefer an
//! SSH tunnel or an authenticating reverse proxy for anything beyond a trusted LAN.
//! OAIY_SERVER_BIND=lan refuses to start without OAIY_SERVER_TOKEN.
//!
//! Configuration is by environment variable instead of the GUI's pointer file:
//!   OAIY_DATA_DIR        data root (databases, venvs, templates)  [~/.oaiy-server]
//!   OAIY_MODELS_DIR      where downloads land                     [<data>/models]
//!   OAIY_EXTRA_MODEL_DIRS extra read-only model roots (`:`/`;`-separated)
//!   OAIY_SERVER_PORT     listen port                              [17972]
//!   OAIY_SERVER_BIND     `lan` binds 0.0.0.0 instead of loopback  [loopback]
//!   OAIY_SERVER_TOKEN    the bearer every request but health, capability
//!                       discovery and pairing must present       [none]
//!   OAIY_HF_TOKEN        HuggingFace token for this server's own gated
//!                       downloads (not passed on to services)    [none]
//!   OAIY_ENGINES_UI      the engines' control pages, when oaiy-studio runs beside
//!                       this server (e.g. http://127.0.0.1:7860): /api/engines*,
//!                       the AI gateway's engine provider and the control API's
//!                       engine tools read and relay them          [none]
//!   OAIY_VOICE_GATEWAY   `off`: do not serve the voice gateway (17872), for a server
//!                       tried beside the desktop, whose calls use it [on]
//!   OAIY_PLUGIN_SOURCES  a folder of plugin folders the setup wizard's catalog
//!                       offers to install from (<dir>/<plugin id>) [none]
//!
//! Every request but a few needs the bearer token: a headless server has no real
//! webview origin, and any local process can forge the `Origin` header, so the
//! token is the only credential trusted off the GUI. Public without one are only
//! the health route (`/api/health`), capability discovery
//! (`/api/bridge/capabilities`) and the pairing bootstrap (`/api/bridge/pairing`).
//! With no OAIY_SERVER_TOKEN set that is ALL the server answers: everything else,
//! reads included, gets 403. Set a token to administer the server — the CLI sends
//! `Authorization: Bearer <token>`. (A token that pairing minted, or the one this
//! process hands the flow runs it starts, is accepted too; approving a pairing
//! needs a bearer to begin with.)
//!
//! SIGTERM / Ctrl-C stops every managed service before exiting, and on unix the plugins
//! first (on Windows a plugin ends with the server's job object).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use oaiy_desktop_lib::http::{self, DesktopConfig, ConfigProvider};
use oaiy_desktop_lib::plugins::PluginHost;
use oaiy_desktop_lib::services::catalog::CatalogHandle;
use oaiy_desktop_lib::services::downloads::{Downloads, DownloadsHandle};
use oaiy_desktop_lib::services::python::{Python, PythonHandle};
use oaiy_desktop_lib::services::registry::{Registry, RegistryHandle};
use oaiy_desktop_lib::DESKTOP_PORT;

/// Env-var-backed config provider (the headless analogue of the GUI's
/// AppHandle-backed one). No pointer file, no restart-required concept.
struct EnvConfig {
    data_dir: PathBuf,
    models_dir: PathBuf,
}

impl ConfigProvider for EnvConfig {
    fn snapshot(&self, _registry: &RegistryHandle) -> DesktopConfig {
        let d = self.data_dir.display().to_string();
        let m = self.models_dir.display().to_string();
        DesktopConfig {
            active_dir: d.clone(),
            default_dir: d,
            configured_dir: None,
            is_custom: false,
            restart_required: false,
            models_active_dir: m.clone(),
            models_default_dir: m,
            models_configured_dir: None,
            models_is_custom: false,
            models_restart_required: false,
        }
    }
}

// --- minimal stderr logger (no extra deps; mirrors the macros lib.rs uses) ---
struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        eprintln!("[{}] {}", record.level(), record.args());
    }
    fn flush(&self) {}
}
static LOGGER: StderrLogger = StderrLogger;

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Block until Ctrl-C or (on unix) SIGTERM — so `systemctl stop` shuts us down
/// cleanly instead of leaking orphaned service processes.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}

fn auth_token_is_empty() -> bool {
    std::env::var("OAIY_SERVER_TOKEN").map(|s| s.trim().is_empty()).unwrap_or(true)
}

#[tokio::main]
async fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let data_dir = match std::env::var_os("OAIY_DATA_DIR") {
        Some(d) => PathBuf::from(d),
        None => match home_dir() {
            Some(h) => h.join(".oaiy-server"),
            None => {
                eprintln!(
                    "oaiy-server: set OAIY_DATA_DIR — no HOME/USERPROFILE to derive a default data dir"
                );
                std::process::exit(1);
            }
        },
    };
    let models_dir = std::env::var_os("OAIY_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("models"));
    // Use the platform path separator (`;` on Windows, `:` on Unix) so a
    // Windows drive path like `D:\ckpts` isn't split on its colon.
    let extra_model_dirs: Vec<PathBuf> = std::env::var_os("OAIY_EXTRA_MODEL_DIRS")
        .map(|s| {
            std::env::split_paths(&s)
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        })
        .unwrap_or_default();
    // Default only when UNSET/blank; fail loud on a typo / out-of-range / 0 rather
    // than silently binding the default (which leaves the server reachable on a
    // port the operator didn't choose, with no error).
    let port: u16 = match std::env::var("OAIY_SERVER_PORT") {
        Err(_) => DESKTOP_PORT,
        Ok(s) if s.trim().is_empty() => DESKTOP_PORT,
        Ok(s) => s.trim().parse::<u16>().ok().filter(|p| *p != 0).unwrap_or_else(|| {
            eprintln!("oaiy-server: invalid OAIY_SERVER_PORT {s:?} (want 1-65535)");
            std::process::exit(1);
        }),
    };
    // Opt-in only, and only on an exact value: anything else (including a typo
    // like `LAN ` or `true`) keeps loopback, because the failure mode of guessing
    // wrong here is a server on the network that nobody meant to expose.
    let bind_all = std::env::var("OAIY_SERVER_BIND")
        .map(|s| s.trim().eq_ignore_ascii_case("lan"))
        .unwrap_or(false);
    if bind_all && auth_token_is_empty() {
        eprintln!(
            "oaiy-server: OAIY_SERVER_BIND=lan requires a non-empty OAIY_SERVER_TOKEN"
        );
        std::process::exit(1);
    }

    // Trim symmetrically with the client (bearer_token trims), so surrounding
    // whitespace in the env var can't silently reject a valid token.
    let auth_token = std::env::var("OAIY_SERVER_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());

    // Owner-only when this is what makes it (unix mode 0700): the provider keys,
    // the account link, paired-app tokens and identity keys all live in here. A
    // folder that already exists, such as systemd's StateDirectory, is left as it is.
    if let Err(e) = oaiy_desktop_lib::secret_file::create_private_dir(&data_dir) {
        eprintln!(
            "oaiy-server: cannot create data dir {}: {e}",
            data_dir.display()
        );
        std::process::exit(1);
    }
    let _ = std::fs::create_dir_all(&models_dir);

    let registry: RegistryHandle =
        match Registry::init(data_dir.clone(), models_dir.clone(), extra_model_dirs) {
            Ok(r) => Arc::new(Mutex::new(r)),
            Err(e) => {
                eprintln!("oaiy-server: registry init failed: {e}");
                std::process::exit(1);
            }
        };
    // One-time migration of install-completion markers for venv services installed before the
    // marker existed — mirrors the GUI (lib.rs) so headless + GUI hosts report installed-ness
    // identically. Idempotent + sentinel-gated, so safe to call every boot.
    if let Ok(r) = registry.lock() {
        r.backfill_install_markers();
    }
    let downloads: DownloadsHandle = Downloads::new(models_dir.clone()).into_handle();
    if let Ok(tok) = std::env::var("OAIY_HF_TOKEN") {
        if !tok.is_empty() {
            downloads.set_token(Some(tok));
        }
    }
    let python: PythonHandle = Python::new(data_dir.clone()).into_handle();
    let catalog = CatalogHandle::new(data_dir.clone());

    let config: Arc<dyn ConfigProvider> = Arc::new(EnvConfig {
        data_dir: data_dir.clone(),
        models_dir: models_dir.clone(),
    });

    // Reap exited child processes so service status flips promptly.
    {
        let registry = registry.clone();
        let python = python.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                // Recover from poison so reaping survives a panic that poisoned the
                // mutex (otherwise exited children stop being reaped for the process
                // lifetime).
                {
                    let mut reg = registry.lock().unwrap_or_else(|e| e.into_inner());
                    reg.reap_exited();
                    // Retry anything whose crash backoff has elapsed.
                    reg.run_scheduled_restarts();
                }
                python.reap_exited();
            }
        });
    }

    // Health-probe ticker: catch services that spawned but didn't bind a port.
    {
        let registry = registry.clone();
        tokio::spawn(async move {
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("health: client build failed: {e}");
                    return;
                }
            };
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                // Recover from poison (consistent with the reaper + shutdown
                // handler) so a single panic that poisons the mutex doesn't silently
                // stop health-probing for the rest of the process lifetime.
                let targets: Vec<(String, String, u64)> = registry
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .health_targets();
                if targets.is_empty() {
                    continue;
                }
                let mut results = Vec::with_capacity(targets.len());
                for (id, url, timeout) in targets {
                    let req = client
                        .get(&url)
                        .timeout(std::time::Duration::from_secs(timeout.min(10)));
                    let ok = matches!(req.send().await, Ok(r) if r.status().is_success());
                    results.push((id, ok));
                }
                registry
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .apply_health_results(&results);
            }
        });
    }

    // Clean shutdown: stop the plugins (on unix) and every running service on SIGTERM / Ctrl-C.
    //
    // The plugin host does not exist yet (it is built with the bridge state, below),
    // and the signal is worth hearing from the start, so the task reads it from this
    // slot when the signal comes. Empty means nothing was started to stop.
    let plugin_host: Arc<OnceLock<Arc<PluginHost>>> = Arc::default();
    {
        let registry = registry.clone();
        let plugin_host = plugin_host.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            log::info!("oaiy-server: shutting down — stopping plugins and all services");
            stop_children(registry, plugin_host.get().cloned()).await;
            std::process::exit(0);
        });
    }

    // "starting", not "listening" — the socket binds later inside http::serve();
    // serve() logs the authoritative post-bind "listening" line, so a port-in-use
    // failure here no longer prints a contradictory success message first.
    log::info!(
        "oaiy-server starting on http://127.0.0.1:{port}  (data={}, models={}, auth={})",
        data_dir.display(),
        models_dir.display(),
        if auth_token.is_some() {
            "token"
        } else {
            "none — privileged/admin routes DISABLED; set OAIY_SERVER_TOKEN to administer"
        },
    );

    // Plugins live under the data dir so relocating it takes them along; they
    // hold their own state and leaving it behind reads as data loss.
    let node_runtime = oaiy_desktop_lib::services::node_runtime::new_handle(data_dir.clone());
    let bridge = oaiy_desktop_lib::build_bridge_state(
        data_dir.join("plugins"),
        data_dir.clone(),
        oaiy_desktop_lib::stable_device_id(&data_dir),
        Some(node_runtime.clone()),
    );
    // Now the shutdown above has plugins to stop (they start at boot, on the host's own thread).
    let _ = plugin_host.set(bridge.host.clone());
    // AI provider store under <data>/ai (holds provider API keys plaintext,
    // guarded by the full/public split — never over the wire).
    let ai_providers = oaiy_desktop_lib::ai::open_handle(data_dir.join("ai").join("providers.json"));
    let ai_codex = oaiy_desktop_lib::ai::codex::new_handle(&data_dir);

    // Start what was ticked "start with the app", plus whatever was running
    // before the last shutdown. A headless box that reboots should come up
    // serving, not waiting for someone to SSH in and start each service by hand
    // — and the ticked set is honoured here too, so a preference set in the GUI
    // means the same thing on a machine that only ever runs the server.
    {
        let started = registry
            .lock()
            .map(|mut r| r.autostart_on_boot())
            .unwrap_or_default();
        if !started.is_empty() {
            log::info!("autostarted {} service(s): {}", started.len(), started.join(", "));
        }
    }

    // Companion trust and its relay, shared with the plugin host so
    // `companion.admission` is answered from the same roster these routes
    // administer.
    let companion = oaiy_desktop_lib::companion::new_handle(data_dir.clone(), bridge.plugins.clone());
    let companion_upstream = oaiy_desktop_lib::companion::upstream::UpstreamStore::open(
        data_dir.join("companion").join("relay.json"),
    );
    let link = oaiy_desktop_lib::link::open_handle(data_dir.clone());
    // The guarded dispatcher: a command for a plugin runs only if the relay
    // policy lets the website run it (link/policy.rs).
    oaiy_desktop_lib::link::relay::spawn(
        link.clone(),
        oaiy_desktop_lib::link::ops::relay_dispatcher(
            registry.clone(),
            bridge.plugins.clone(),
            bridge.host.clone(),
            oaiy_desktop_lib::link::ops::RelayGuard::open(&data_dir),
        ),
    );
    // Execute the flow runs the account has queued. The other half of the flows
    // lane: without it this box reserves runs and then waits for a runtime that
    // may not exist, and every one of them sits queued.
    oaiy_desktop_lib::link::flow_runner::spawn(
        link.clone(),
        Some(node_runtime.clone()),
        // A binding's post-run actions may call a connector, through the
        // plugin's own gate as the relay's commands meet it, but NOT the relay
        // policy: the provider queues these flows and serves the bindings, so
        // this is a way in the policy does not cover yet (see
        // `link::ops::dispatcher`).
        Some(oaiy_desktop_lib::link::ops::dispatcher(
            registry.clone(),
            bridge.plugins.clone(),
            bridge.host.clone(),
        )),
    );
    // Answer the sealed chat turns the provider's web app relays. Without it
    // the website cannot encrypt to this machine at all and says the desktop
    // has published no key.
    oaiy_desktop_lib::ai::tunnel::spawn(
        link.clone(),
        oaiy_desktop_lib::ai::tunnel::AiSources {
            providers: ai_providers.clone(),
            codex: ai_codex.clone(),
        },
    );
    oaiy_desktop_lib::link::sealed_flows::spawn(link.clone(), Some(node_runtime.clone()));
    // Let plugin events reach the flows the user built on the provider's site.
    // Without it an Aokie call fires only this desktop's local bindings, and
    // the flow they actually wrote never runs.
    bridge.host.set_link(link.clone());
    bridge.host.set_companion_broker(oaiy_desktop_lib::plugins::CompanionBroker {
        companion: companion.clone(),
        upstream: companion_upstream.clone(),
    });

    // The engines, when a studio runs beside this server: the GUI finds or
    // starts them itself (engines.rs); a headless server is told where they are.
    if let Some(ui) = std::env::var("OAIY_ENGINES_UI").ok().map(|u| u.trim().trim_end_matches('/').to_string()).filter(|u| !u.is_empty()) {
        http::set_engines_ui(&ui);
    }

    // gui_mode = false: headless server is token-strict (no webview origin).
    let registry_for_exit = registry.clone();
    if let Err(e) = http::serve(
        port,
        bind_all, config, auth_token, false, registry, downloads, python, catalog, bridge,
        companion, companion_upstream, link, ai_providers, ai_codex, node_runtime,
    )
    .await
    {
        eprintln!("oaiy-server: HTTP server error: {e}");
        // A port that is in use is found only here, when serve binds it, after the plugins
        // and the services ticked "start with the app" are already running.
        stop_children(registry_for_exit, plugin_host.get().cloned()).await;
        std::process::exit(1);
    }
}

/// Stop what this process started, in the order the app does on quit: the warm script host
/// first (one Node child, which exits on a line), then the plugins (lighter to stop than
/// model servers, and one holding hardware should get its graceful shutdown before anything
/// slow runs), then the services.
///
/// Every exit this server makes on purpose comes through here. On unix a plugin left running
/// after its server has gone keeps its hardware and its port, and nothing else stops it (a
/// plugin no longer shares this process's group, so a Ctrl-C at the terminal does not reach it).
/// Windows is left as it was: a plugin is in the server's job object, which ends it with this
/// process, so it never outlives the server there.
async fn stop_children(registry: RegistryHandle, plugins: Option<Arc<PluginHost>>) {
    // On a blocking thread: each of these waits on a child, and a plugin gets up to
    // its grace period to answer the request to stop.
    let _ = tokio::task::spawn_blocking(move || {
        oaiy_desktop_lib::bridge::ScriptHost::global().shutdown();
        #[cfg(unix)]
        if let Some(host) = plugins {
            host.stop_all();
        }
        #[cfg(not(unix))]
        let _ = plugins;
    })
    .await;
    // Recover from a poisoned mutex: stopping services on exit matters more
    // than poison-safety — `if let Ok` would silently skip it and orphan every
    // running service (venv-python / OAIY Voice / multi-GB loaders).
    let mut r = registry.lock().unwrap_or_else(|e| e.into_inner());
    r.stop_all();
}
