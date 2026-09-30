//! `oaiy-server` — OAIY Desktop's HTTP API + service supervisor, headless.
//!
//! Same axum API as the tray app (services / models / python on
//! `127.0.0.1:17972`), but with no window, tray, or webview — for running on a
//! Linux box (or any server) driven by a CO-LOCATED Node CLI / oaiy-web. The API
//! binds 127.0.0.1 by default; OAIY_SERVER_BIND=lan binds every interface for the
//! case this exists to serve — editing from a phone on the same network — but only
//! with an owner login already made on the console (`oaiy-server auth init`), and
//! then it answers bearer tokens only: no cookies, no sign-in, no UI over plain
//! HTTP. Prefer an SSH tunnel or an authenticating reverse proxy (OAIY_PUBLIC_URL)
//! for anything beyond a trusted LAN.
//!
//! Configuration is by environment variable instead of the GUI's pointer file:
//!   OAIY_DATA_DIR        data root (databases, venvs, templates)  [~/.oaiy-server]
//!   OAIY_MODELS_DIR      where downloads land                     [<data>/models]
//!   OAIY_EXTRA_MODEL_DIRS extra read-only model roots (`:`/`;`-separated)
//!   OAIY_SERVER_PORT     listen port                              [17972]
//!   OAIY_SERVER_BIND     `loopback`, `lan` (0.0.0.0, bearer only, needs an
//!                       owner) or an IP address to bind           [loopback]
//!   OAIY_PUBLIC_URL      https://<host>[:port] of the dashboard: the install is
//!                       behind a reverse proxy (OAIY_AGENT_URL and OAIY_FLOWS_URL
//!                       name the other two app hosts)             [none]
//!   OAIY_TRUSTED_PROXIES the peers whose X-Forwarded-For is believed; a bind
//!                       beyond loopback behind a public URL must name them, and
//!                       then answers nothing else                 [loopback when
//!                                                                 OAIY_PUBLIC_URL is set]
//!   OAIY_ALLOWED_HOSTS   extra Host names, for every install       [none]
//!   OAIY_SERVER_TOKEN    a bearer of the `cli` preset: 32 to 256 printable
//!                       characters with no common pattern, a guard against
//!                       the obvious and not a strength meter (no word like
//!                       `test`, no run like `1234`, no phrase of common words:
//!                       `openssl rand -base64 32` is one; else exit 78) [none]
//!
//! A configuration that breaks a rule of the design (an unreadable bind, a lan bind with no owner,
//! a public URL with a path, a proxy not named, a weak token, a mode that is not allowed there)
//! stops the server with exit 78, and one line saying what to change. The shipped unit does not restart
//! that (`RestartPreventExitStatus=78`, and it runs `oaiy-server check` first, which lists every
//! violation at once; `scripts/check-release.mjs` holds the unit to both).
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
//!   OAIY_ACCESS_MODE     `legacy` (every route that existed before the access model is judged
//!                       as it always was), `scoped` (the new guard everywhere) or `shadow`
//!                       (scoped, but a scope a token lacks is logged, not refused, unless it is
//!                       a dangerous one) [legacy]
//!
//! Every request but a few needs a credential: a headless server has no real
//! webview origin, and any local process can forge the `Origin` header, so a
//! token is the only credential trusted off the GUI. Public without one are only
//! the health route (`/api/health`), capability discovery
//! (`/api/bridge/capabilities`) and the pairing bootstrap (`/api/bridge/pairing`):
//! on a build with the web login (the release's) the last two are open to a caller
//! with no credential once an owner exists, and before that, in setup-only mode, to
//! a caller WITH one (the token) only. With no OAIY_SERVER_TOKEN set that is ALL the
//! server answers: everything else, reads included, is refused. Set a token to
//! administer the server — the CLI sends `Authorization: Bearer <token>`. (A token
//! that pairing minted, or the one this process hands the flow runs it starts, is
//! accepted too; approving a pairing needs a bearer to begin with.)
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

/// The environment as text. A value that is not UTF-8 is read lossily, so that it fails whatever rule reads it
/// rather than passing for unset.
fn env_text(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// `oaiy-server auth ...`, `oaiy-server check` and `oaiy-server flows ...`: the console (design 4.7.9). They run
/// before the server's runtime exists, and never start the server.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(first) = args.first().map(String::as_str) {
        if matches!(first, "auth" | "check" | "flows") {
            std::process::exit(console(&args));
        }
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("the async runtime starts")
        .block_on(server_main());
}

#[cfg(feature = "web")]
fn console(args: &[String]) -> i32 {
    oaiy_desktop_lib::auth::console_cli::run_process(args)
}

#[cfg(not(feature = "web"))]
fn console(args: &[String]) -> i32 {
    // `check` needs no console: the rules of the design over the environment.
    if args.first().map(String::as_str) == Some("check") {
        return oaiy_desktop_lib::auth::exposure::check_headless(
            &env_text,
            &mut std::io::stdout(),
            &mut std::io::stderr(),
        );
    }
    eprintln!(
        "oaiy-server {}: this build has no web login (it was built without the `web` feature), so it has no console",
        args.first().map_or("", String::as_str)
    );
    2
}

async fn server_main() {
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
    // The configuration is judged whole, before anything is made or opened (design 4.5.5): the port, the bind and
    // what it needs (a lan bind needs an owner login, a bind beyond loopback behind a public URL names its
    // proxy), the public URLs, the token's shape, the access mode. The first violation is the one line printed and
    // the exit is 78, which the shipped unit does not restart; `oaiy-server check` lists them all. Nothing here
    // guesses: a bind that is not loopback, lan or an address used to keep loopback quietly, and does not now.
    // The owner file is read as the store reads it (`inspect_owner_file`): one that is there and that this server
    // cannot use is a refusal named by the file, first, as `oaiy-server check` lists it, and not a file that
    // satisfies rule 2 and stops the server one step later.
    let owner = oaiy_desktop_lib::auth::exposure::inspect_owner_file(&data_dir.join("auth"));
    if let Some(line) = owner.refusal() {
        eprintln!("oaiy-server: {line}");
        std::process::exit(oaiy_desktop_lib::auth::mode::EX_CONFIG);
    }
    let facts = oaiy_desktop_lib::auth::exposure::Facts {
        owner_exists: owner.exists(),
        web_login: cfg!(feature = "web"),
    };
    let config = match oaiy_desktop_lib::auth::exposure::validate_config(&env_text, &facts) {
        Ok(config) => config,
        Err(line) => {
            eprintln!("oaiy-server: {line}");
            std::process::exit(oaiy_desktop_lib::auth::mode::EX_CONFIG);
        }
    };
    for warning in &config.warnings {
        log::warn!("{warning}");
    }
    // (What this install is, in words, with no secret in it, is printed by `http::serve` once the credential store is
    // open and the listener is bound: a banner that says "listening" is not printed by a server that then stops.)
    let port: u16 = config.port;
    let bind = config.bind.addr();
    // Trimmed and of the shape of design 4.1, or the server would not be here.
    let auth_token = config.static_token.clone();

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

    let env_config: Arc<dyn ConfigProvider> = Arc::new(EnvConfig {
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
        #[cfg(feature = "web")]
        let auth_dir = data_dir.join("auth");
        tokio::spawn(async move {
            shutdown_signal().await;
            log::info!("oaiy-server: shutting down — stopping plugins and all services");
            stop_children(registry, plugin_host.get().cloned()).await;
            // The credential store's last-used times and the noise counted so far, before the process ends.
            oaiy_desktop_lib::auth::flush_installed();
            // The console's credential dies with the server: its files are removed (what a crash leaves holds a
            // credential that is already dead).
            #[cfg(feature = "web")]
            oaiy_desktop_lib::auth::console::remove_files(&auth_dir);
            std::process::exit(0);
        });
    }

    // "starting", not "listening" — the socket binds later inside http::serve();
    // serve() logs the authoritative post-bind "listening" line, so a port-in-use
    // failure here no longer prints a contradictory success message first.
    log::info!(
        "oaiy-server starting on {}:{port}  (data={}, models={}, auth={})",
        if bind.is_ipv6() { format!("http://[{bind}]") } else { format!("http://{bind}") },
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

    // Newer releases: this server only REPORTS them (GET /api/update/status, and POST /api/update/check,
    // which reads the release feed when asked). It never downloads or replaces itself: see docs/UPDATES.md
    // for how to upgrade it by hand.
    let updater = oaiy_desktop_lib::update::Updater::new(
        env!("CARGO_PKG_VERSION"),
        oaiy_desktop_lib::update::FeedSource::from_env(),
        std::time::Instant::now(),
    );

    // gui_mode = false: headless server is token-strict (no webview origin).
    let registry_for_exit = registry.clone();
    if let Err(e) = http::serve(
        port,
        bind, env_config, auth_token, false, registry, downloads, python, catalog, bridge,
        companion, companion_upstream, link, ai_providers, ai_codex, node_runtime, updater,
        oaiy_desktop_lib::auth::AccessSettings::for_server(config),
    )
    .await
    {
        eprintln!("oaiy-server: HTTP server error: {e}");
        // A refusal of the configuration or of the data folder (a mode that is not allowed here, a second
        // process on the folder, a credential file from a newer OAIY) is exit 78: not restarted by the unit.
        let refused = e.downcast_ref::<oaiy_desktop_lib::auth::ConfigRefusal>().is_some()
            || e.downcast_ref::<oaiy_desktop_lib::auth::store::StoreError>().is_some();
        // A port that is in use is found only here, when serve binds it, after the plugins
        // and the services ticked "start with the app" are already running.
        stop_children(registry_for_exit, plugin_host.get().cloned()).await;
        std::process::exit(if refused {
            oaiy_desktop_lib::auth::store::EX_CONFIG
        } else {
            1
        });
    }
}

/// Stop what this process started, in the order the app does on quit: the warm script host
/// first (one Node child, which exits on a line), then the plugins (lighter to stop than
/// model servers, and one holding hardware should get its graceful shutdown before anything
/// slow runs), then the services.
///
/// Every exit this server makes on purpose comes through here. On unix a plugin that ignores the
/// close of its stdin (which is what ends one that follows the plugin contract when its server
/// goes) outlives the server, keeps its hardware and its port, and nothing else stops it: a plugin
/// no longer shares this process's group, so a Ctrl-C or hangup at the terminal does not reach it.
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
