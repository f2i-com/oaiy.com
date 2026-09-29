//! The desktop's side of updates: the updater plugin, the download, the install, and the commands.
//!
//! Everything that changes what runs is a Tauri command of the dashboard's own window, never a route
//! (`update::routes` can only look): [`update_check`], [`update_download`] and [`update_install`].
//! Each answers ONLY the webview labelled `main`. The Agent, the flow editor and the engines' page
//! are webviews of the same window (`embed-agent`, `embed-flows`, `embed-engines`), so a check on the
//! window's label would let them through, and a page the Agent shows must not be able to restart OAIY
//! under a phone call. None of the three takes an address, a path or a version: the only update there
//! is is the one the release feed named, which the plugin fetched, and this module checked, already.
//! No `updater:*` permission is granted to any webview in `capabilities/`, so the plugin's own commands,
//! which the plugin registers, answer nobody: only the Rust API of the plugin is used, from here.
//!
//! The order of an install is `update::install`; this file supplies the real parts it stops and starts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::future::{AbortHandle, Abortable};
use tauri::{AppHandle, Manager, Webview};
use tauri_plugin_updater::{Update, UpdaterExt};

use super::blockers::Probes;
use super::feed::{check_asset_url, Release};
use super::install::{self, Outcome, Part, Steps};
use super::kind;
use super::updater::{AutoUpdate, BoxFuture, CheckRefusal, Platform, Status, UpdaterHandle};
use super::verify::{verify_package, VerifiedPackage, VerifyError};
use crate::plugins::PluginHost;
use crate::services::registry::RegistryHandle;

/// The dashboard's own webview.
pub const DASHBOARD_LABEL: &str = "main";

/// The most an installer may be. The real one is about a hundred megabytes; NSIS itself stops at 2 GB.
pub const MAX_INSTALLER_BYTES: u64 = 1 << 30;

/// A whole download, however slow: a stalled connection ends rather than waits for ever.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// How long the Agent page is given to save its work before an install goes on without it.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// A short while after start, then daily: when OAIY looks for a newer release by itself.
const FIRST_CHECK_DELAY: Duration = Duration::from_secs(90);
const CHECK_EVERY: Duration = Duration::from_secs(24 * 60 * 60);

/// Whether a webview may check for, download and install an update: only the dashboard's own.
pub fn is_dashboard(label: &str) -> bool {
    label == DASHBOARD_LABEL
}

fn dashboard_only(webview: &Webview) -> Result<(), String> {
    if is_dashboard(webview.label()) {
        Ok(())
    } else {
        log::warn!("update: refused a request from the webview \"{}\"", webview.label());
        Err("Only OAIY's own window can update OAIY.".to_string())
    }
}

/// What the plugin found: its handle on the release, kept from the check to the download and the install.
#[derive(Default)]
pub struct Store {
    update: Mutex<Option<Update>>,
}

impl Store {
    fn get(&self) -> Option<Update> {
        self.update.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set(&self, update: Option<Update>) {
        *self.update.lock().unwrap_or_else(|e| e.into_inner()) = update;
    }
}

/// An error of the updater in words for a person.
fn explain(error: &tauri_plugin_updater::Error) -> String {
    use tauri_plugin_updater::Error as E;
    match error {
        E::Minisign(_) | E::Base64(_) | E::SignatureUtf8(_) => VerifyError::DoesNotMatch.to_string(),
        E::SignedVersionMismatch { signed, announced } => VerifyError::SignedForOtherVersion { signed: signed.clone(), announced: announced.clone() }.to_string(),
        E::MissingSignedVersion => VerifyError::NoSignedVersion.to_string(),
        E::Reqwest(_) | E::Network(_) => "The download failed. Check the internet connection and try again.".to_string(),
        E::TargetNotFound(_) | E::TargetsNotFound(_) => "The update information has no release for this kind of install, so nothing was fetched.".to_string(),
        E::InvalidUpdaterFormat | E::BinaryNotFoundInArchive => "The downloaded file is not an installer, so it was refused.".to_string(),
        other => format!("The update failed ({other})."),
    }
}

/// The public key and `requireSignedVersion` the plugin is configured with (tauri.conf.json): the key an update is checked against.
fn updater_config(app: &AppHandle) -> (String, bool) {
    let config = app.config().plugins.0.get("updater").cloned().unwrap_or_default();
    let pubkey = config.get("pubkey").and_then(|k| k.as_str()).unwrap_or_default().to_string();
    let require = config.get("requireSignedVersion").and_then(|r| r.as_bool()).unwrap_or(false);
    (pubkey, require)
}

/// What this build can do about an update, and how it gets ready for one.
pub struct GuiPlatform {
    app: AppHandle,
    store: Arc<Store>,
    /// Installers must come from this project's releases on GitHub. Off only for a debug build reading a stub feed.
    strict_assets: bool,
    /// The feed a debug build was told to read (OAIY_UPDATE_FEED), so that the plugin reads the same one as the core.
    feed_override: Option<String>,
}

impl GuiPlatform {
    pub fn new(app: AppHandle, store: Arc<Store>, feed: &super::FeedSource) -> GuiPlatform {
        GuiPlatform { app, store, strict_assets: !feed.insecure, feed_override: feed.insecure.then(|| feed.url.clone()) }
    }
}

impl Platform for GuiPlatform {
    fn auto_update(&self) -> AutoUpdate {
        kind::auto_update_for(std::env::consts::OS, std::env::consts::ARCH, kind::bundle_name(tauri::utils::platform::bundle_type()))
    }

    /// The feed named a newer release for this platform: get the plugin's handle on it, and make sure it is the
    /// release the feed read here named and that its installer is one of ours to fetch.
    fn prepare<'a>(&'a self, release: &'a Release) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            let changed = || "The update information changed while it was being read. Check again.".to_string();
            let mut builder = self.app.updater_builder().timeout(Duration::from_secs(20));
            if let Some(url) = &self.feed_override {
                let url = url.parse().map_err(|_| "The update address is not a web address.".to_string())?;
                builder = builder.endpoints(vec![url]).map_err(|e| explain(&e))?;
            }
            let updater = builder.build().map_err(|e| explain(&e))?;
            let update = updater.check().await.map_err(|e| explain(&e))?.ok_or_else(changed)?;
            if update.version != release.version {
                return Err(changed());
            }
            if self.strict_assets {
                check_asset_url(update.download_url.as_str(), &update.version)?;
            }
            self.store.set(Some(update));
            Ok(())
        })
    }
}

/// Download and verify the installer: the plugin fetches it and checks its signature, then it is checked again here.
/// The bytes stay in memory. Whatever goes wrong, nothing is kept.
async fn download(update: Update, updater: UpdaterHandle, pubkey: String, require_signed_version: bool) -> Result<VerifiedPackage, String> {
    let mut update = update;
    update.timeout = Some(DOWNLOAD_TIMEOUT);
    let (version, signature) = (update.version.clone(), update.signature.clone());
    let (abort, registration) = AbortHandle::new_pair();
    let seen = Arc::new(AtomicU64::new(0));
    let progress = {
        let (updater, abort, seen) = (updater.clone(), abort.clone(), seen.clone());
        move |chunk: usize, total: Option<u64>| {
            let done = seen.fetch_add(chunk as u64, Ordering::SeqCst) + chunk as u64;
            updater.set_progress(done, total);
            // Stopped as soon as it is over the limit, by what the server says or by what has come.
            if done > MAX_INSTALLER_BYTES || total.is_some_and(|t| t > MAX_INSTALLER_BYTES) {
                abort.abort();
            }
        }
    };
    match Abortable::new(update.download(progress, || {}), registration).await {
        Err(_) => Err("The update is larger than an installer should be (over 1 GiB), so it was not used.".to_string()),
        Ok(Err(e)) => Err(explain(&e)),
        Ok(Ok(bytes)) => verify_package(bytes, &signature, &pubkey, &version, require_signed_version).map_err(|e| e.to_string()),
    }
}

/// The status, worked out off the async threads (what is asked of the engines may wait a moment).
async fn status_of(updater: &UpdaterHandle) -> Result<Status, String> {
    let updater = updater.clone();
    tauri::async_runtime::spawn_blocking(move || updater.status()).await.map_err(|_| "The update status could not be read.".to_string())
}

/// Look for a newer release now (at most once every 30 seconds). Answers the status.
#[tauri::command]
pub async fn update_check(webview: Webview, updater: tauri::State<'_, UpdaterHandle>) -> Result<Status, String> {
    dashboard_only(&webview)?;
    match updater.check().await {
        Ok(()) | Err(CheckRefusal::TooSoon { .. } | CheckRefusal::AlreadyChecking) => status_of(&updater).await,
        Err(refusal) => Err(refusal.to_string()),
    }
}

/// Download the release the last check found, and verify its signature. Answers the status (`downloading`); the
/// window follows the progress in `GET /api/update/status`.
#[tauri::command]
pub async fn update_download(webview: Webview, app: AppHandle, updater: tauri::State<'_, UpdaterHandle>, store: tauri::State<'_, Arc<Store>>) -> Result<Status, String> {
    dashboard_only(&webview)?;
    let release = updater.begin_download().map_err(|e| e.to_string())?;
    let Some(update) = store.get().filter(|u| u.version == release.version) else {
        let message = "The update information changed since the last check. Check again.".to_string();
        updater.finish_download(Err(message.clone()));
        return Err(message);
    };
    let (pubkey, require_signed_version) = updater_config(&app);
    let task = updater.inner().clone();
    tauri::async_runtime::spawn(async move {
        let outcome = download(update, task.clone(), pubkey, require_signed_version).await;
        if let Err(message) = &outcome {
            log::warn!("update: the download failed: {message}");
        }
        task.finish_download(outcome);
    });
    status_of(&updater).await
}

/// Restart to update: what `update::install` describes. It takes no argument: the update is the one downloaded and verified.
/// Refused (with the reasons, in words) while anything is in the way; a failure after OAIY began to stop starts it all again.
#[tauri::command]
pub async fn update_install(webview: Webview, app: AppHandle, updater: tauri::State<'_, UpdaterHandle>, store: tauri::State<'_, Arc<Store>>) -> Result<(), String> {
    dashboard_only(&webview)?;
    let Some(update) = store.get() else {
        return Err("There is no downloaded update to install. Download it first.".to_string());
    };
    let (updater, app_for_steps) = (updater.inner().clone(), app.clone());
    // Off the async and main threads: the stops wait on children, and the Agent page answers by way of the local API.
    let outcome = tauri::async_runtime::spawn_blocking(move || run_install(&app_for_steps, &updater, update))
        .await
        .map_err(|e| format!("The update stopped unexpectedly ({e})."))?;
    match outcome {
        Outcome::HandedOff => {
            // Windows never gets here: the plugin started the installer and ended this process. On Linux the AppImage
            // was replaced, so start it. Everything was stopped before the hand-off, and the same is done as the
            // plugin does on Windows (clean up, then relaunch and exit) rather than the normal exit, which would stop it
            // all a second time and empty the note of what was running that the hand-off wrote for the next start.
            app.cleanup_before_exit();
            tauri::process::restart(&app.env())
        }
        Outcome::Refused(refusal) => Err(refusal.to_string()),
        Outcome::Failed { message, .. } => Err(message),
    }
}

fn run_install(app: &AppHandle, updater: &UpdaterHandle, update: Update) -> Outcome {
    let parts = install_parts(app);
    let refs: Vec<&dyn Part> = parts.iter().map(|p| p.as_ref() as &dyn Part).collect();
    let flush = || flush_agent(app, updater);
    let hand_off = |package: &VerifiedPackage| {
        // The bytes that were verified are for the version the update handle names: never hand over any others.
        if package.version() != update.version {
            return Err(format!("the downloaded update is for version {}, not {}", package.version(), update.version));
        }
        update.install(package.bytes()).map_err(|e| explain(&e))
    };
    let clock = std::time::Instant::now;
    install::perform(updater, &Steps { flush: &flush, parts: &refs, hand_off: &hand_off, clock: &clock })
}

// ---- the agent page's flush ---------------------------------------------------------------------

/// What the Agent page is told to run: its own before-quit hook (which stops its turn and saves its projects and chat), then
/// a word to the local API that it is done, carrying the nonce it was given. The hook is the Agent's; without it the page
/// still answers, at once.
fn flush_script(nonce: &str) -> String {
    let nonce = serde_json::to_string(nonce).unwrap_or_else(|_| "\"\"".to_string());
    format!(
        "(async () => {{ try {{ if (typeof window.__botComputerBeforeQuit === 'function') await window.__botComputerBeforeQuit(); }} catch (e) {{}} \
         try {{ const d = window.__OAIY_DESKTOP__; if (d && d.origin && d.token) await fetch(d.origin + '/api/update/agent-flushed', {{ method: 'POST', \
         headers: {{ 'content-type': 'application/json', authorization: 'Bearer ' + d.token }}, body: JSON.stringify({{ nonce: {nonce} }}) }}); }} catch (e) {{}} }})();"
    )
}

/// Ask the Agent page to save its work, and wait a little for its word. No page, no answer or an error: the install goes on.
fn flush_agent(app: &AppHandle, updater: &UpdaterHandle) {
    let Some(webview) = crate::embed::agent_webview(app) else {
        log::info!("update: the Agent page is not open; nothing to save");
        return;
    };
    let (nonce, answer) = updater.arm_flush();
    if let Err(e) = webview.eval(flush_script(&nonce)) {
        log::warn!("update: the Agent page could not be asked to save: {e}");
        updater.disarm_flush();
        return;
    }
    match answer.recv_timeout(FLUSH_TIMEOUT) {
        Ok(()) => log::info!("update: the Agent page saved its work"),
        Err(_) => log::warn!("update: the Agent page did not answer within {} seconds; the update goes on", FLUSH_TIMEOUT.as_secs()),
    }
    updater.disarm_flush();
}

// ---- what is stopped and started --------------------------------------------------------------

/// The engines this desktop started (a studio that was already running is left alone).
struct EnginesPart {
    app: AppHandle,
    started_here: Mutex<bool>,
}

impl Part for EnginesPart {
    fn name(&self) -> &'static str {
        "the engines"
    }

    fn stop(&self) -> Result<(), String> {
        *self.started_here.lock().unwrap_or_else(|e| e.into_inner()) = crate::engines::started_here();
        crate::engines::stop();
        Ok(())
    }

    fn start(&self) -> Result<(), String> {
        if !*self.started_here.lock().unwrap_or_else(|e| e.into_inner()) {
            return Ok(());
        }
        let registry = self.app.try_state::<RegistryHandle>().ok_or("the services are not there")?;
        let data_dir = registry.lock().map_err(|_| "the services are busy")?.data_dir().to_path_buf();
        crate::engines::start(&data_dir).map(|_| ())
    }
}

/// The warm script host: one Node child that exits on a line, and starts again on the next request.
struct ScriptHostPart;

impl Part for ScriptHostPart {
    fn name(&self) -> &'static str {
        "the script host"
    }

    fn stop(&self) -> Result<(), String> {
        crate::bridge::ScriptHost::global().shutdown();
        Ok(())
    }

    fn start(&self) -> Result<(), String> {
        Ok(())
    }
}

/// The plugins (Aokie holds the phone dongle: it gets its graceful shutdown before anything slow runs).
struct PluginsPart {
    host: Option<Arc<PluginHost>>,
    was_running: Mutex<Vec<String>>,
}

impl Part for PluginsPart {
    fn name(&self) -> &'static str {
        "the plugins"
    }

    fn stop(&self) -> Result<(), String> {
        if let Some(host) = &self.host {
            *self.was_running.lock().unwrap_or_else(|e| e.into_inner()) = host.running_ids();
            log::info!("stopping all plugins on exit");
            host.stop_all();
        }
        Ok(())
    }

    fn start(&self) -> Result<(), String> {
        let Some(host) = &self.host else { return Ok(()) };
        host.resume();
        let ids = self.was_running.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let failed: Vec<String> = ids.iter().filter_map(|id| host.start(id).err().map(|e| format!("{id}: {e}"))).collect();
        if failed.is_empty() { Ok(()) } else { Err(failed.join("; ")) }
    }
}

/// The services (model servers, the speech server for calls).
struct ServicesPart {
    registry: Option<RegistryHandle>,
    was_running: Mutex<Vec<String>>,
    /// An update writes what was running as the note the next launch brings back. Quitting does not: a quit forgets.
    remember_for_relaunch: bool,
}

impl Part for ServicesPart {
    fn name(&self) -> &'static str {
        "the services"
    }

    fn stop(&self) -> Result<(), String> {
        if let Some(registry) = &self.registry {
            // Recover from a poisoned mutex: stopping services matters more than poison-safety (else they are orphaned).
            let mut r = registry.lock().unwrap_or_else(|e| e.into_inner());
            let running = r.running_ids();
            log::info!("stopping all services on exit");
            r.stop_all();
            if self.remember_for_relaunch {
                r.write_running_note(&running);
            }
            *self.was_running.lock().unwrap_or_else(|e| e.into_inner()) = running;
        }
        Ok(())
    }

    fn start(&self) -> Result<(), String> {
        let Some(registry) = &self.registry else { return Ok(()) };
        let ids = self.was_running.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut r = registry.lock().unwrap_or_else(|e| e.into_inner());
        let failed: Vec<String> = ids.iter().filter_map(|id| r.start(id).err().map(|e| format!("{id}: {e}"))).collect();
        if failed.is_empty() { Ok(()) } else { Err(failed.join("; ")) }
    }
}

fn parts_for(app: &AppHandle, remember_for_relaunch: bool) -> Vec<Box<dyn Part>> {
    let registry = app.try_state::<RegistryHandle>().map(|r| r.inner().clone());
    vec![
        Box::new(EnginesPart { app: app.clone(), started_here: Mutex::new(false) }),
        Box::new(ScriptHostPart),
        Box::new(PluginsPart { host: app.try_state::<Arc<PluginHost>>().map(|h| h.inner().clone()), was_running: Mutex::new(Vec::new()) }),
        Box::new(ServicesPart { registry, was_running: Mutex::new(Vec::new()), remember_for_relaunch }),
    ]
}

/// What an install stops, in the order quitting stops it.
fn install_parts(app: &AppHandle) -> Vec<Box<dyn Part>> {
    parts_for(app, true)
}

/// Quitting: the same parts, the same order, best effort. (Called on `RunEvent::Exit`.)
pub fn stop_on_exit(app: &AppHandle) {
    let parts = parts_for(app, false);
    let refs: Vec<&dyn Part> = parts.iter().map(|p| p.as_ref() as &dyn Part).collect();
    install::stop_best_effort(&refs);
}

// ---- wiring -----------------------------------------------------------------------------------

/// The updater plugin, judging what is newer by the same rule as everything else here ([`super::version`]).
pub fn plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R, tauri_plugin_updater::Config> {
    tauri_plugin_updater::Builder::new().default_version_comparator(|current, release| super::version::is_newer_version(&current, &release.version)).build()
}

/// Tell the updater what this desktop is and what it is doing: it can install, and asks the services, the downloads,
/// the engines and the rest before it does. Call once, where the handles exist (`node` is made with the local API).
pub fn attach(updater: &UpdaterHandle, app: &AppHandle, store: Arc<Store>, feed: &super::FeedSource, probes: Probes) {
    updater.set_platform(Arc::new(GuiPlatform::new(app.clone(), store, feed)));
    updater.set_activity(Arc::new(probes));
}

/// Looks for a newer release a little after start and then every day, unless it is switched off in Settings.
/// Never installs anything.
pub fn spawn_scheduler(updater: UpdaterHandle) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(FIRST_CHECK_DELAY).await;
        loop {
            if updater.auto_check() {
                match updater.check().await {
                    Ok(()) => {}
                    Err(refusal) => log::info!("update: the scheduled check did not run: {refusal}"),
                }
            }
            tokio::time::sleep(CHECK_EVERY).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_dashboards_own_webview_may_update_oaiy() {
        assert!(is_dashboard("main"));
        // The pages OAIY shows in its own window are webviews of that window, with labels of their own.
        for label in ["embed-agent", "embed-flows", "embed-engines", "", "Main", "main ", "main-2", "oaiy"] {
            assert!(!is_dashboard(label), "{label:?}");
        }
        // Every page OAIY embeds in the window is a webview whose label the gate keeps out.
        for page in crate::embed::Page::ALL {
            assert!(!is_dashboard(page.label()), "{}", page.label());
        }
    }

    /// This file and lib.rs as text, whatever line endings a checkout gave them.
    fn this_file() -> String {
        include_str!("gui.rs").replace("\r\n", "\n")
    }

    fn lib_file() -> String {
        include_str!("../lib.rs").replace("\r\n", "\n")
    }

    /// The text of a command in this file: its signature and body, up to the next item.
    fn command_source(name: &str) -> String {
        let src = this_file();
        let start = src.find(&format!("pub async fn {name}(")).unwrap_or_else(|| panic!("no command {name}"));
        let rest = &src[start..];
        let end = rest.find("\n}\n").expect("the command ends");
        rest[..end].to_string()
    }

    #[test]
    fn every_command_checks_the_webview_first_and_takes_no_address_path_or_version() {
        for name in ["update_check", "update_download", "update_install"] {
            let source = command_source(name);
            let (signature, body) = source.split_once("{\n").expect("a body");
            assert!(signature.contains("webview: Webview"), "{name} takes the calling webview");
            assert!(body.trim_start().starts_with("dashboard_only(&webview)?;"), "{name}: the label check is its first statement");
            // Only what the app injects: the calling webview, the app, the updater's state. No input of the caller's own.
            let parameters = &signature[signature.find('(').unwrap() + 1..signature.rfind(')').unwrap()];
            for argument in ["String", "PathBuf", "Path", "Url", "url", "path", "version", "Vec<", "&str", "Value"] {
                assert!(!parameters.contains(argument), "{name} takes {argument}: an update is never named by its caller");
            }
        }
        // The one setting a webview can change about updates is gated the same way.
        let lib = lib_file();
        let setter = lib.find("fn set_update_auto_check(").expect("the setting's command");
        assert!(lib[setter..setter + 600].contains("is_dashboard(webview.label())"));
    }

    #[test]
    fn quitting_and_updating_stop_the_same_parts_in_the_same_order() {
        let src = this_file();
        let start = src.find("fn parts_for(").expect("the one list of parts");
        let list = &src[start..start + src[start..].find("\n}\n").unwrap()];
        let order: Vec<usize> = ["EnginesPart {", "ScriptHostPart)", "PluginsPart {", "ServicesPart {"].iter().map(|part| list.find(&format!("Box::new({part}")).unwrap_or_else(|| panic!("{part} is in the list"))).collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "engines, script host, plugins, services: {order:?}");
        // Both callers use that list, and the exit body stops nothing of its own any more.
        assert!(src.contains("fn install_parts(app: &AppHandle) -> Vec<Box<dyn Part>> {
    parts_for(app, true)
}"));
        assert!(src.contains("let parts = parts_for(app, false);
    let refs: Vec<&dyn Part> = parts.iter().map(|p| p.as_ref() as &dyn Part).collect();
    install::stop_best_effort(&refs);"));
        let lib = lib_file();
        let exit = lib.find("RunEvent::Exit => {").expect("the exit arm");
        let arm = &lib[exit..exit + lib[exit..].find("\n                }\n").unwrap()];
        assert!(arm.contains("crate::update::gui::stop_on_exit(app_handle);"), "{arm}");
        for inline in ["stop_all()", "engines::stop()", "shutdown()"] {
            assert!(!arm.contains(inline), "the exit arm does not stop things itself: {inline}");
        }
    }

    #[test]
    fn the_install_restarts_by_hand_after_the_stop_and_never_by_the_apps_restart_command_or_a_second_stop() {
        let source = command_source("update_install");
        assert!(source.contains("tauri::process::restart(&app.env())"));
        assert!(source.contains("app.cleanup_before_exit();"));
        // restart_app skips the stop; request_restart runs the exit body again, which empties the running note
        // the install wrote for the next start.
        assert!(!source.contains("restart_app") && !source.contains("request_restart(") && !source.contains(".restart()"), "{source}");
    }

    #[test]
    fn the_updater_plugins_configuration_in_tauri_conf_json_is_one_it_starts_with() {
        // A configuration the plugin cannot read makes the app panic at start-up ("error while setting up plugin
        // updater"): read it here as the plugin does.
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        let config: tauri_plugin_updater::Config = serde_json::from_value(conf["plugins"]["updater"].clone()).expect("the plugin reads it");
        assert_eq!(config.endpoints.iter().map(|u| u.as_str()).collect::<Vec<_>>(), [crate::update::FEED_URL]);
        assert!(!config.pubkey.is_empty() && !config.dangerous_insecure_transport_protocol && !config.dangerous_accept_invalid_certs && !config.dangerous_accept_invalid_hostnames);
        assert!(!config.allow_downgrades, "an older release is never an update");
        assert_eq!(config.windows.expect("windows settings").install_mode.to_string(), "passive");
    }

    #[test]
    fn no_webview_is_granted_the_updater_plugins_own_commands() {
        // The plugin registers check, download, install and download_and_install for JavaScript; only a capability
        // that names an updater permission lets a webview call them. None does: OAIY uses the plugin from Rust.
        let capabilities = include_str!("../../capabilities/default.json");
        assert!(!capabilities.contains("updater"), "{capabilities}");
        let all = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/capabilities")).unwrap();
        for entry in all.flatten() {
            let text = std::fs::read_to_string(entry.path()).unwrap();
            assert!(!text.contains("updater"), "{}", entry.path().display());
        }
    }

    #[test]
    fn the_agent_is_asked_to_save_by_its_own_hook_and_answers_with_the_nonce_it_was_given() {
        let script = flush_script("a1b2c3d4");
        assert!(script.contains("window.__botComputerBeforeQuit"));
        assert!(script.contains("/api/update/agent-flushed"));
        assert!(script.contains("nonce: \"a1b2c3d4\""));
        // Whatever the hook does, the answer is still sent; and an error in either never escapes the page.
        assert_eq!(script.matches("catch (e) {}").count(), 2);
        assert!(script.contains("d.token"));
        // A nonce is only ever quoted data: a hostile one cannot break out of the string.
        let hostile = flush_script("\"); alert(1); (\"");
        assert!(hostile.contains(r#"nonce: "\"); alert(1); (\"""#), "{hostile}");
    }

    #[test]
    fn updater_errors_are_put_in_words() {
        use tauri_plugin_updater::Error as E;
        assert!(explain(&E::Network("connection reset".into())).contains("internet connection"));
        assert!(explain(&E::TargetsNotFound(vec!["windows-x86_64".into()])).contains("no release for this kind of install"));
        assert!(explain(&E::InvalidUpdaterFormat).contains("not an installer"));
        assert!(explain(&E::MissingSignedVersion).contains("does not say which version"));
        assert!(explain(&E::SignedVersionMismatch { signed: "0.1.5".into(), announced: "9.9.9".into() }).contains("0.1.5"));
        assert!(explain(&E::ReleaseNotFound).starts_with("The update failed"));
    }
}
