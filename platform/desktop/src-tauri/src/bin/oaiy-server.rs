//! `oaiy-server` — OAIY Desktop's HTTP API + service supervisor, headless.
//!
//! Same axum API as the tray app (services / models / python on
//! `127.0.0.1:17972`), but with no window, tray, or webview — for running on a
//! Linux box (or any server) driven by a CO-LOCATED Node CLI / oaiy-web. The API
//! binds 127.0.0.1 only (never a network interface); remote access must go through
//! an SSH tunnel or an authenticating reverse proxy.
//!
//! Configuration is by environment variable instead of the GUI's pointer file:
//!   OAIY_DATA_DIR        data root (databases, venvs, templates)  [~/.oaiy-server]
//!   OAIY_MODELS_DIR      where downloads land                     [<data>/models]
//!   OAIY_EXTRA_MODEL_DIRS extra read-only model roots (`:`/`;`-separated)
//!   OAIY_SERVER_PORT     listen port                              [17972]
//!   OAIY_SERVER_TOKEN    bearer token gating privileged routes    [none]
//!   OAIY_HF_TOKEN        HuggingFace token for gated downloads    [none]
//!   OAIY_LLAMACPP_MODEL  GGUF the llama-cpp service loads         [none]
//!                       (the headless equivalent of the Model selector)
//!   OAIY_OLLAMA_MODEL    model the ollama service uses (${ollamaModel}) [qwen2.5:0.5b]
//!
//! With no OAIY_SERVER_TOKEN set, privileged routes (define service / install
//! python / create venv / delete) are CLOSED: a headless server has no real
//! webview origin, and any local process can forge the `Origin` header, so the
//! bearer token is the only credential trusted off the GUI. Set a token to
//! administer the server — the CLI sends `Authorization: Bearer <token>`.
//! (Non-privileged ops, e.g. starting an already-defined service, stay reachable
//! on loopback.)

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use oaiy_desktop_lib::http::{self, DesktopConfig, ConfigProvider};
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
            // Headless: reflect the OAIY_LLAMACPP_MODEL env (applied to the live
            // registry at startup) so /api/config reports the active model.
            llama_model: std::env::var("OAIY_LLAMACPP_MODEL")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            // Reflect OAIY_OLLAMA_MODEL (applied to the registry at startup) so
            // /api/config reports the active model; falls back to the service's
            // built-in default (qwen2.5:0.5b) when unset.
            ollama_model: std::env::var("OAIY_OLLAMA_MODEL")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
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
    // Trim symmetrically with the client (bearer_token trims), so surrounding
    // whitespace in the env var can't silently reject a valid token.
    let auth_token = std::env::var("OAIY_SERVER_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());

    if let Err(e) = std::fs::create_dir_all(&data_dir) {
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
    // Headless has no Model-selector UI; honor OAIY_LLAMACPP_MODEL so the built-in
    // llama-cpp service (which refuses to start with no model) is usable on a
    // server. Absent it, llama-cpp stays unstartable — define a custom template
    // with the gguf baked into run.args instead.
    if let Ok(m) = std::env::var("OAIY_LLAMACPP_MODEL") {
        let m = m.trim().to_string();
        if !m.is_empty() {
            if let Ok(mut r) = registry.lock() {
                r.set_llama_model(Some(m));
            }
        }
    }
    // Same for Ollama — OAIY_OLLAMA_MODEL picks the model nodes use via ${ollamaModel}
    // (the GUI has a selector; headless reads the env). Absent it, the service's
    // built-in default (qwen2.5:0.5b) applies.
    if let Ok(m) = std::env::var("OAIY_OLLAMA_MODEL") {
        let m = m.trim().to_string();
        if !m.is_empty() {
            if let Ok(mut r) = registry.lock() {
                r.set_ollama_model(Some(m));
            }
        }
    }
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

    // Clean shutdown: stop every running service on SIGTERM / Ctrl-C.
    {
        let registry = registry.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            log::info!("oaiy-server: shutting down — stopping all services");
            // Recover from a poisoned mutex: stopping services on exit matters more
            // than poison-safety — `if let Ok` would silently skip it and orphan every
            // running service (venv-python / llama.cpp / multi-GB loaders).
            let mut r = registry.lock().unwrap_or_else(|e| e.into_inner());
            r.stop_all();
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
    let bridge = oaiy_desktop_lib::build_bridge_state(
        data_dir.join("plugins"),
        data_dir.clone(),
        oaiy_desktop_lib::stable_device_id(&data_dir),
    );
    // AI provider store under <data>/ai (holds provider API keys plaintext,
    // guarded by the full/public split — never over the wire).
    let ai_providers = oaiy_desktop_lib::ai::open_handle(data_dir.join("ai").join("providers.json"));
    let ai_codex = oaiy_desktop_lib::ai::codex::new_handle(&data_dir);

    // Bring back whatever was running before the last shutdown. A headless box
    // that reboots should come up serving, not waiting for someone to SSH in and
    // start each service by hand.
    {
        let started = registry
            .lock()
            .map(|mut r| r.autostart_remembered())
            .unwrap_or_default();
        if !started.is_empty() {
            log::info!("autostarted {} service(s): {}", started.len(), started.join(", "));
        }
    }

    // gui_mode = false: headless server is token-strict (no webview origin).
    if let Err(e) = http::serve(
        port, config, auth_token, false, registry, downloads, python, catalog, bridge, ai_providers, ai_codex,
    )
    .await
    {
        eprintln!("oaiy-server: HTTP server error: {e}");
        std::process::exit(1);
    }
}
