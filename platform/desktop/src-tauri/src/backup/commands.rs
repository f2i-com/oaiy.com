//! The dashboard's backup commands (Tauri, so only in the GUI build).
//!
//! These are the only way to make a backup, stage a restore, undo one, cancel one or restart to
//! apply one: no HTTP route does any of it. Each command checks that it is called from the
//! dashboard's own window (the webview labelled `main`), never from the Agent's or the flow
//! editor's page, and opens the native save or open dialog itself, so no page ever hands it a
//! path. The passphrase comes only as an argument, is used and dropped, and is never logged.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::Serialize;
use tauri::{AppHandle, Manager, Runtime, Webview};

use super::agent::AgentExport;
use super::busy::{local_signals, BusySignals};
use super::create::{create, CreateOptions, CreateResult};
use super::restore::{self, Preview, RestoreOptions, Staged};
use super::review::Ticks;
use super::EXTENSION;
use crate::services::downloads::{DownloadStatus, DownloadsHandle};
use crate::services::python::PythonHandle;
use crate::services::registry::{RegistryHandle, ServiceStatus};

/// The label of the dashboard's own webview.
const DASHBOARD_LABEL: &str = "main";

/// Only the dashboard's own window may call these.
pub(crate) fn check_label(label: &str) -> Result<(), String> {
    if label == DASHBOARD_LABEL {
        Ok(())
    } else {
        Err("Backups can only be started from OAIY's own dashboard window.".to_string())
    }
}

fn dashboard<R: Runtime>(webview: &Webview<R>) -> Result<(), String> {
    check_label(webview.label())
}

fn data_dir_of<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let registry = app.try_state::<RegistryHandle>().ok_or("OAIY has not finished starting.")?;
    let dir = registry.lock().map(|r| r.data_dir().to_path_buf()).map_err(|_| "OAIY is busy starting.".to_string())?;
    Ok(dir)
}

/// The call the desktop makes in the Agent's page to ask it for its storage.
pub(crate) fn export_script(id: &str, token: &str, include_keys: bool) -> String {
    format!("window.__oaiyBackup && window.__oaiyBackup.export({{ id: {id:?}, token: {token:?}, includeKeys: {include_keys} }});")
}

/// Asks the Agent's page (a webview of the window) by evaluating one call in it.
struct AgentPage<R: Runtime> {
    app: AppHandle<R>,
}

impl<R: Runtime> AgentExport for AgentPage<R> {
    fn request(&self, id: &str, token: &str, include_keys: bool) -> Result<(), String> {
        let webview = self.app.get_webview(crate::embed::Page::Agent.label()).ok_or("the Agent's page is not open")?;
        webview.eval(export_script(id, token, include_keys)).map_err(|e| e.to_string())
    }
}

/// What is going on in the app, as far as this window can see.
async fn gather_busy<R: Runtime>(app: &AppHandle<R>) -> BusySignals {
    let mut signals = local_signals();
    if let Some(downloads) = app.try_state::<DownloadsHandle>() {
        signals.downloads += downloads.snapshot().iter().filter(|d| matches!(d.status, DownloadStatus::Queued | DownloadStatus::Active)).count();
    }
    if let Some(python) = app.try_state::<PythonHandle>() {
        if python.snapshot().current_job.is_some_and(|job| job.finished_at.is_none()) {
            signals.installs += 1;
        }
    }
    if let Some(registry) = app.try_state::<RegistryHandle>() {
        if let Ok(r) = registry.lock() {
            signals.installs += r.snapshot().services.iter().filter(|s| s.status == ServiceStatus::Installing).count();
        }
    }
    // The engines are a program of their own: ask them what they are working on.
    if let Some(ui) = crate::http::engines_ui() {
        let status_of = |v: &serde_json::Value| v.get("status").and_then(|s| s.as_str()).map(str::to_string).unwrap_or_default();
        if let Ok(jobs) = crate::http::studio_json(&ui, reqwest::Method::GET, "/api/jobs", None).await {
            signals.engine_jobs += jobs.get("jobs").and_then(|j| j.as_array()).map_or(0, |j| j.iter().filter(|job| matches!(status_of(job).as_str(), "queued" | "in_progress")).count());
        }
        if let Ok(state) = crate::http::studio_json(&ui, reqwest::Method::GET, "/api/downloads", None).await {
            signals.downloads += state
                .get("models")
                .and_then(|m| m.as_array())
                .map_or(0, |models| models.iter().filter_map(|m| m.get("download")).filter(|d| !d.is_null()).filter(|d| matches!(status_of(d).as_str(), "queued" | "downloading" | "adding")).count());
        }
    }
    signals
}

async fn pick_save<R: Runtime>(app: &AppHandle<R>, name: &str) -> Option<PathBuf> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().set_title("Save your OAIY backup").set_file_name(name).add_filter("OAIY backup", &[EXTENSION]).save_file(move |path| {
        let _ = tx.send(path);
    });
    rx.await.ok().flatten().and_then(|p| p.as_path().map(Path::to_path_buf))
}

async fn pick_open<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().set_title("Choose an OAIY backup").add_filter("OAIY backup", &[EXTENSION]).pick_file(move |path| {
        let _ = tx.send(path);
    });
    rx.await.ok().flatten().and_then(|p| p.as_path().map(Path::to_path_buf))
}

/// Wait for a check or a staging that runs on its own thread, for a little longer than it is itself
/// allowed to run: it stops itself at its deadline, and this is the backstop that lets the panel go
/// on if a thread is stuck somewhere it cannot look at the clock.
async fn with_time_limit<T: Send + 'static>(work: tokio::task::JoinHandle<super::Result<T>>) -> super::Result<T> {
    match tokio::time::timeout(super::RESTORE_TIME_LIMIT + std::time::Duration::from_secs(30), work).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(super::BackupError::new(super::ErrorKind::Io, "That stopped unexpectedly.")),
        Err(_) => Err(super::BackupError::new(super::ErrorKind::Timeout, "That took too long, so it was stopped. Try again.")),
    }
}

/// Make a backup: asks where to save it, then does it. `null` when the person closes the dialog.
#[tauri::command]
pub async fn backup_create<R: Runtime>(app: AppHandle<R>, webview: Webview<R>, passphrase: String, include_keys: bool) -> Result<Option<CreateResult>, String> {
    dashboard(&webview)?;
    super::check_passphrase(&passphrase).map_err(|e| e.message)?;
    let data_dir = data_dir_of(&app)?;
    let busy = gather_busy(&app).await;
    busy.refuse_if_busy("making a backup").map_err(|e| e.message)?;
    let name = format!("oaiy-backup-{}.{EXTENSION}", chrono::Local::now().format("%Y-%m-%d"));
    let Some(dest) = pick_save(&app, &name).await else { return Ok(None) };
    let page = AgentPage { app: app.clone() };
    let made = tokio::task::spawn_blocking(move || {
        let mut options = CreateOptions::new(&data_dir, dest, &passphrase);
        options.include_keys = include_keys;
        options.busy = busy;
        options.agent = Some(&page);
        create(&options)
    })
    .await
    .map_err(|_| "The backup stopped unexpectedly.".to_string())?;
    made.map(Some).map_err(|e| e.message)
}

/// The file chosen in the last "check this backup", so staging can use it without any page giving a path.
struct Inspected {
    id: String,
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

static INSPECTED: Mutex<Option<Inspected>> = Mutex::new(None);

/// What the dashboard is told after it chose a file to restore from.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectOut {
    inspect_id: String,
    #[serde(flatten)]
    preview: Preview,
}

/// Restore, step 1: asks for the backup file, decrypts and checks it, and says what restoring would do.
/// Changes nothing.
#[tauri::command]
pub async fn backup_restore_inspect<R: Runtime>(app: AppHandle<R>, webview: Webview<R>, passphrase: String) -> Result<Option<InspectOut>, String> {
    dashboard(&webview)?;
    if passphrase.is_empty() {
        return Err("Type the passphrase the backup was made with.".to_string());
    }
    let data_dir = data_dir_of(&app)?;
    // Checking a backup takes a second of computing and up to a gigabyte of memory: not while a call is live.
    let options = RestoreOptions { busy: gather_busy(&app).await, ..RestoreOptions::default() };
    options.busy.refuse_if_busy("checking a backup").map_err(|e| e.message)?;
    let Some(path) = pick_open(&app).await else { return Ok(None) };
    let meta = std::fs::metadata(&path).map_err(|_| "That file could not be read.".to_string())?;
    let (len, modified) = (meta.len(), meta.modified().ok());
    let file = path.clone();
    let options = RestoreOptions { busy: gather_busy(&app).await, ..options };
    let preview = with_time_limit(tokio::task::spawn_blocking(move || restore::inspect(&data_dir, &file, &passphrase, &options)))
        .await
        .map_err(|e| e.message)?;
    let id = super::random_id();
    *INSPECTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Inspected { id: id.clone(), path, len, modified });
    Ok(Some(InspectOut { inspect_id: id, preview }))
}

/// Restore, step 2: unpack the backup that was just checked into the staging folder and note that it
/// is to be applied at the next start. Nothing live changes.
#[tauri::command]
pub async fn backup_restore_stage<R: Runtime>(app: AppHandle<R>, webview: Webview<R>, inspect_id: String, passphrase: String, classes: Vec<String>, keys: bool) -> Result<Staged, String> {
    dashboard(&webview)?;
    // What the person ticked: nothing that can run or reconfigure comes back without it.
    let ticks = Ticks::from_ids(&classes, keys).map_err(|e| e.message)?;
    let data_dir = data_dir_of(&app)?;
    let path = {
        let guard = INSPECTED.lock().unwrap_or_else(|e| e.into_inner());
        let inspected = guard.as_ref().filter(|i| i.id == inspect_id).ok_or("Choose the backup file again.".to_string())?;
        let now = std::fs::metadata(&inspected.path).map_err(|_| "The backup file is no longer there.".to_string())?;
        if now.len() != inspected.len || now.modified().ok() != inspected.modified {
            return Err("The backup file has changed since it was checked. Choose it again.".to_string());
        }
        inspected.path.clone()
    };
    let options = RestoreOptions { busy: gather_busy(&app).await, ..RestoreOptions::default() };
    let staged = with_time_limit(tokio::task::spawn_blocking(move || restore::stage(&data_dir, &path, &passphrase, &ticks, &options)))
        .await
        .map_err(|e| e.message)?;
    *INSPECTED.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(staged)
}

/// Stage "Undo the last restore": what the last restore replaced is put back at the next start.
#[tauri::command]
pub async fn backup_undo_stage<R: Runtime>(app: AppHandle<R>, webview: Webview<R>) -> Result<Staged, String> {
    dashboard(&webview)?;
    let data_dir = data_dir_of(&app)?;
    tokio::task::spawn_blocking(move || restore::stage_undo(&data_dir, &RestoreOptions::default()))
        .await
        .map_err(|_| "Preparing the undo stopped unexpectedly.".to_string())?
        .map_err(|e| e.message)
}

/// Cancel a staged restore before the restart.
#[tauri::command]
pub async fn backup_discard_pending<R: Runtime>(app: AppHandle<R>, webview: Webview<R>) -> Result<(), String> {
    dashboard(&webview)?;
    let data_dir = data_dir_of(&app)?;
    tokio::task::spawn_blocking(move || restore::discard_pending(&data_dir))
        .await
        .map_err(|_| "Cancelling stopped unexpectedly.".to_string())?
        .map_err(|e| e.message)
}

/// Restart OAIY so the staged restore is applied, unless the app is busy.
#[tauri::command]
pub async fn backup_restart_to_apply(app: AppHandle, webview: Webview) -> Result<(), String> {
    dashboard(&webview)?;
    let data_dir = data_dir_of(&app)?;
    if restore::pending_info(&data_dir).is_none() {
        return Err("No restore is waiting.".to_string());
    }
    gather_busy(&app).await.refuse_if_busy("restarting to finish the restore").map_err(|e| e.message)?;
    crate::gui::restart_app(app);
    Ok(())
}
