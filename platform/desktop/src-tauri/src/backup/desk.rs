//! The order of what the dashboard's restore commands do, apart from the window, so that it can be tested.
//!
//! A restore is looked at first and prepared second, and the second is held to the first:
//!
//! - **Look** ([`inspect`]): ask what stands in the way (the updater's own decision), let the person choose the
//!   file, ask again (the dialog can stay open for minutes: a call may have started since), decrypt and check it,
//!   and remember what was found: which file, the SHA-256 of its decrypted contents, and the items the look listed.
//! - **Prepare** ([`stage`]): only for a backup that was looked at. The file is decrypted again and must have the
//!   same contents, byte for byte (its size and date prove nothing: a file of the same size and date can hold
//!   other things), and nothing is brought back that the look did not list.
//!
//! The window is a [`Host`]: it answers "what is in the way now" and "which file did the person choose". The
//! commands in `commands.rs` are this with the real window; the tests are this with a scripted one.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;
use zeroize::Zeroizing;

use super::busy::Busy;
use super::restore::{self, Inspection, Preview, RestoreOptions, Staged};
use super::review::Ticks;

/// What the flow needs of the window and of the app.
pub trait Host: Send + Sync {
    /// What stands in the way of a backup or a restore now, asked of every source afresh each time.
    fn busy(&self) -> impl Future<Output = Busy> + Send;
    /// The file the person chooses, in the native dialog (`None`: they closed it).
    fn pick_open(&self) -> impl Future<Output = Option<PathBuf>> + Send;
    fn data_dir(&self) -> Result<PathBuf, String>;
    /// Where the look is remembered for the prepare that follows.
    fn desk(&self) -> &Desk;
    /// Ask the Agent's page to save its work, and wait a little for its word (the updater's own handshake: no page, no answer or an
    /// error is not a reason to stop). OAIY is about to restart, and what the page has not saved is lost.
    fn save_agent(&self) -> impl Future<Output = ()> + Send;
    /// Restart OAIY so that what waits is applied at the start.
    fn restart(&self);
}

/// The backup that was looked at last, kept so that preparing can use it without any page giving a path.
struct Inspected {
    id: String,
    path: PathBuf,
    inspection: Inspection,
}

pub struct Desk {
    inspected: Mutex<Option<Inspected>>,
}

impl Desk {
    pub const fn new() -> Self {
        Self { inspected: Mutex::new(None) }
    }

    fn remember(&self, inspected: Inspected) {
        *self.inspected.lock().unwrap_or_else(|e| e.into_inner()) = Some(inspected);
    }

    fn recall(&self, id: &str) -> Result<(PathBuf, Inspection), String> {
        let guard = self.inspected.lock().unwrap_or_else(|e| e.into_inner());
        let seen = guard.as_ref().filter(|i| i.id == id).ok_or("Choose the backup file again.".to_string())?;
        Ok((seen.path.clone(), seen.inspection.clone()))
    }

    fn forget(&self) {
        *self.inspected.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Whether a look is being remembered (for the tests).
    #[cfg(test)]
    pub(crate) fn remembers(&self) -> bool {
        self.inspected.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

impl Default for Desk {
    fn default() -> Self {
        Self::new()
    }
}

/// What the dashboard is told after it chose a file to restore from.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectOut {
    pub inspect_id: String,
    #[serde(flatten)]
    pub preview: Preview,
}

/// Wait for a check or a staging that runs on its own thread, for a little longer than it is itself
/// allowed to run: it stops itself at its deadline, and this is the backstop that lets the panel go
/// on if a thread is stuck somewhere it cannot look at the clock.
pub(crate) async fn with_time_limit<T: Send + 'static>(work: tokio::task::JoinHandle<super::Result<T>>) -> super::Result<T> {
    match tokio::time::timeout(super::RESTORE_TIME_LIMIT + std::time::Duration::from_secs(30), work).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(super::BackupError::new(super::ErrorKind::Io, "That stopped unexpectedly.")),
        Err(_) => Err(super::BackupError::new(super::ErrorKind::Timeout, "That took too long, so it was stopped. Try again.")),
    }
}

/// Restore, step 1: asks for the backup file, decrypts and checks it, and says what restoring would do.
/// Changes nothing. `None` when the person closes the dialog.
pub async fn inspect<H: Host>(host: &H, passphrase: String) -> Result<Option<InspectOut>, String> {
    let passphrase = Zeroizing::new(passphrase);
    if passphrase.is_empty() {
        return Err("Type the passphrase the backup was made with.".to_string());
    }
    let data_dir = host.data_dir()?;
    // Checking a backup takes a second of computing and up to a gigabyte of memory: not while a call is live.
    host.busy().await.refuse_if_busy("checking a backup").map_err(|e| e.message)?;
    let Some(path) = host.pick_open().await else { return Ok(None) };
    // The dialog can stay open for minutes: what was quiet when it opened may not be now, so it is asked again.
    let options = RestoreOptions { busy: host.busy().await, ..RestoreOptions::default() };
    let file = path.clone();
    let (preview, inspection) = with_time_limit(tokio::task::spawn_blocking(move || restore::inspect_bound(&data_dir, &file, &passphrase, &options)))
        .await
        .map_err(|e| e.message)?;
    let id = super::random_id();
    host.desk().remember(Inspected { id: id.clone(), path, inspection });
    Ok(Some(InspectOut { inspect_id: id, preview }))
}

/// Restore, step 2: unpack the backup that was just checked into the staging folder and note that it is to be
/// applied at the next start. Nothing live changes. It is held to what was checked (see the module note).
pub async fn stage<H: Host>(host: &H, inspect_id: String, passphrase: String, classes: Vec<String>, keys: bool) -> Result<Staged, String> {
    let passphrase = Zeroizing::new(passphrase);
    // What the person ticked: nothing that can run or reconfigure comes back without it.
    let ticks = Ticks::from_ids(&classes, keys).map_err(|e| e.message)?;
    let data_dir = host.data_dir()?;
    let (path, looked_at) = host.desk().recall(&inspect_id)?;
    // (No look at what is in the way: preparing changes nothing that is live. See `restore::stage_checked`.)
    let options = RestoreOptions::default();
    let staged = with_time_limit(tokio::task::spawn_blocking(move || restore::stage_checked(&data_dir, &path, &passphrase, &ticks, &options, &looked_at))).await;
    match staged {
        Ok(staged) => {
            host.desk().forget();
            Ok(staged)
        }
        Err(e) => {
            // A file that is not the one that was looked at is not one to try again: it has to be looked at.
            if e.kind == super::ErrorKind::Conflict {
                host.desk().forget();
            }
            Err(e.message)
        }
    }
}

/// Restart OAIY so the staged restore is applied, unless the app is busy. The order is: a look at what is in the way, the Agent's
/// page is asked to save its work (as the updater asks before it installs), a last look, and the restart. The look is made twice
/// because the first can take seconds (it asks a phone plugin whether a call is live) and the save up to five more, and a call
/// can have started in them. A restart ends a call.
pub async fn restart_to_apply<H: Host>(host: &H) -> Result<(), String> {
    let data_dir = host.data_dir()?;
    match restore::pending_info(&data_dir) {
        None => return Err("No restore is waiting.".to_string()),
        Some(waiting) if waiting.expired => {
            let _ = restore::discard_pending(&data_dir);
            return Err("That restore was prepared more than a day ago, so it will not be applied. Prepare it again.".to_string());
        }
        Some(_) => {}
    }
    host.busy().await.refuse_if_busy("restarting to finish the restore").map_err(|e| e.message)?;
    // What the Agent's page has not saved is lost when OAIY stops: it is asked first, as the updater asks.
    host.save_agent().await;
    // The last look, with nothing between it and the restart.
    host.busy().await.refuse_if_busy("restarting to finish the restore").map_err(|e| e.message)?;
    host.restart();
    Ok(())
}
