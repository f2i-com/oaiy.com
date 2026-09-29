//! The dashboard's backup commands (Tauri, so only in the GUI build).
//!
//! These are the only way to make a backup, stage a restore, undo one, cancel one or restart to
//! apply one: no HTTP route does any of it. Each command checks that it is called from the
//! dashboard's own window (the webview labelled `main`), never from the Agent's or the flow
//! editor's page, and opens the native save or open dialog itself, so no page ever hands it a
//! path. The passphrase comes only as an argument, is used and dropped, and is never logged.

use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager, Runtime, Webview};
use zeroize::Zeroizing;

use super::agent::AgentExport;
use super::busy::Busy;
use super::create::{create, CreateOptions, CreateResult};
use super::desk::{self, Desk, Host, InspectOut};
use super::restore::{self, RestoreOptions, Staged};
use super::EXTENSION;
use crate::services::registry::RegistryHandle;
use crate::update::UpdaterHandle;

/// Only the dashboard's own window may call these. Which window that is has one answer, the updater's
/// (`update::gui::is_dashboard`): a window that may not restart OAIY to update it may not back it up or restore it either.
pub(crate) fn check_label(label: &str) -> Result<(), String> {
    if crate::update::gui::is_dashboard(label) {
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
        let webview = crate::embed::agent_webview(&self.app).ok_or("the Agent's page is not open")?;
        webview.eval(export_script(id, token, include_keys)).map_err(|e| e.to_string())
    }
}

/// What stands in the way of a backup or a restore now: the updater's own decision (the same source and the
/// same function, `update::blockers`) and a backup already being made. It can ask a phone plugin and so wait a few
/// seconds, so it is asked off the async runtime's own thread; a look that fails is not taken for quiet.
async fn look_busy<R: Runtime>(app: &AppHandle<R>) -> Busy {
    let activity = app.try_state::<UpdaterHandle>().and_then(|updater| updater.activity());
    tokio::task::spawn_blocking(move || Busy::look(activity.as_deref())).await.unwrap_or_else(|_| Busy::cannot_tell())
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

/// The dashboard's own window, as the restore flow (`desk`) sees it: the updater's answer to "what is in the way", the
/// native open dialog, and where the data folder is.
struct DashboardHost {
    app: AppHandle,
}

/// The backup that was looked at last (one, for the one dashboard).
static DESK: Desk = Desk::new();

impl Host for DashboardHost {
    fn busy(&self) -> impl std::future::Future<Output = Busy> + Send {
        look_busy(&self.app)
    }

    fn pick_open(&self) -> impl std::future::Future<Output = Option<PathBuf>> + Send {
        pick_open(&self.app)
    }

    fn data_dir(&self) -> Result<PathBuf, String> {
        data_dir_of(&self.app)
    }

    fn desk(&self) -> &Desk {
        &DESK
    }

    fn restart(&self) {
        crate::gui::restart_app(self.app.clone());
    }
}

/// Make a backup: asks where to save it, then does it. `null` when the person closes the dialog.
#[tauri::command]
pub async fn backup_create<R: Runtime>(app: AppHandle<R>, webview: Webview<R>, passphrase: String, include_keys: bool) -> Result<Option<CreateResult>, String> {
    dashboard(&webview)?;
    // Wiped from memory when this is done with it, wherever it returns.
    let passphrase = Zeroizing::new(passphrase);
    super::check_passphrase(&passphrase).map_err(|e| e.message)?;
    let data_dir = data_dir_of(&app)?;
    look_busy(&app).await.refuse_if_busy("making a backup").map_err(|e| e.message)?;
    let name = format!("oaiy-backup-{}.{EXTENSION}", chrono::Local::now().format("%Y-%m-%d"));
    let Some(dest) = pick_save(&app, &name).await else { return Ok(None) };
    // The save dialog can stay open for minutes: a call may have started, or a download, since it was asked.
    let busy = look_busy(&app).await;
    busy.refuse_if_busy("making a backup").map_err(|e| e.message)?;
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

/// Restore, step 1: asks for the backup file, decrypts and checks it, and says what restoring would do.
/// Changes nothing. (The order of it, and what it remembers for step 2, is `desk::inspect`.)
#[tauri::command]
pub async fn backup_restore_inspect(app: AppHandle, webview: Webview, passphrase: String) -> Result<Option<InspectOut>, String> {
    dashboard(&webview)?;
    desk::inspect(&DashboardHost { app }, passphrase).await
}

/// Restore, step 2: unpack the backup that was just checked into the staging folder and note that it
/// is to be applied at the next start. Nothing live changes. (Held to what was checked: `desk::stage`.)
#[tauri::command]
pub async fn backup_restore_stage(app: AppHandle, webview: Webview, inspect_id: String, passphrase: String, classes: Vec<String>, keys: bool) -> Result<Staged, String> {
    dashboard(&webview)?;
    desk::stage(&DashboardHost { app }, inspect_id, passphrase, classes, keys).await
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

/// Restart OAIY so the staged restore is applied, unless the app is busy. (The two looks at what is in the way, and what
/// is checked first, are `desk::restart_to_apply`.)
#[tauri::command]
pub async fn backup_restart_to_apply(app: AppHandle, webview: Webview) -> Result<(), String> {
    dashboard(&webview)?;
    desk::restart_to_apply(&DashboardHost { app }).await
}
