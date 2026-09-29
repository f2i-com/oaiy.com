//! The small records the backup keeps in the data folder, and the phase a running backup is in.
//!
//! - `<data>/backup/status.json`: when the last backup was made, whether the most recent attempt
//!   worked, and how big the last good file was. Nothing secret (no path, no passphrase).
//! - `<data>/restore/pending.json`: the marker of a restore that is staged and waits for the next
//!   start (see [`super::restore`]).
//! - `<data>/restore/last-result.json`: how the last restore or undo went.
//!
//! The dashboard reads all of it through `GET /api/backup/status`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::restore;

/// Where a running backup is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Collecting,
    Agent,
    Packing,
    Encrypting,
    Verifying,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Running {
    pub phase: Phase,
    pub label: String,
}

static RUNNING: Mutex<Option<Running>> = Mutex::new(None);
static IN_USE: AtomicBool = AtomicBool::new(false);

/// Take the one place a backup runs in (`None` when one is running already). Dropped, it lets go.
pub(crate) fn begin_run() -> Option<RunGuard> {
    if IN_USE.swap(true, Ordering::SeqCst) {
        return None;
    }
    Some(RunGuard)
}

pub(crate) struct RunGuard;

impl Drop for RunGuard {
    fn drop(&mut self) {
        *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = None;
        IN_USE.store(false, Ordering::SeqCst);
    }
}

pub(crate) fn set_phase(phase: Phase, label: &str) {
    *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Running { phase, label: label.to_string() });
}

pub fn running() -> Option<Running> {
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

// ---- the last backup ------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StatusFile {
    /// When the last GOOD backup was made.
    last_backup_at: Option<String>,
    /// Whether the most recent attempt worked.
    last_backup_ok: Option<bool>,
    /// The size of the last good file.
    last_backup_size: Option<u64>,
}

fn status_path(data_dir: &Path) -> PathBuf {
    data_dir.join("backup").join("status.json")
}

fn read_status(data_dir: &Path) -> StatusFile {
    std::fs::read_to_string(status_path(data_dir)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn write_status(data_dir: &Path, status: &StatusFile) {
    let body = serde_json::to_string_pretty(status).unwrap_or_default();
    if let Err(e) = crate::secret_file::write(&status_path(data_dir), body) {
        log::warn!("backup: could not record the backup status: {e}");
    }
}

/// A backup was made and checked.
pub(crate) fn record_success(data_dir: &Path, created_at: &str, size: u64) {
    write_status(
        data_dir,
        &StatusFile { last_backup_at: Some(created_at.to_string()), last_backup_ok: Some(true), last_backup_size: Some(size) },
    );
}

/// The most recent attempt failed: the last good backup's date and size stay.
pub(crate) fn record_failure(data_dir: &Path) {
    let mut status = read_status(data_dir);
    status.last_backup_ok = Some(false);
    write_status(data_dir, &status);
}

// ---- what the status route says -------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub last_backup_at: Option<String>,
    pub last_backup_ok: Option<bool>,
    pub last_backup_size: Option<u64>,
    pub pending_restore: Option<restore::PendingInfo>,
    pub last_restore: Option<restore::LastRestore>,
    pub undo_available: bool,
    /// `restore` or `undo`: what made the newest snapshot (the button undoes a restore, and redoes an undo).
    pub undo_kind: Option<String>,
    pub running: Option<Running>,
}

/// Everything the dashboard shows about backups.
pub fn status(data_dir: &Path) -> Status {
    let file = read_status(data_dir);
    Status {
        last_backup_at: file.last_backup_at,
        last_backup_ok: file.last_backup_ok,
        last_backup_size: file.last_backup_size,
        pending_restore: restore::pending_info(data_dir),
        last_restore: restore::last_restore(data_dir),
        undo_available: restore::undo_available(data_dir),
        undo_kind: restore::undo_kind(data_dir),
        running: running(),
    }
}
