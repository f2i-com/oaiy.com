//! The Agent page's own storage, moved in and out of a backup.
//!
//! The Agent's conversations, projects and settings live in the WebView profile (the browser's
//! private file system and IndexedDB), which only the Agent page can read, and which is locked
//! while the app runs. So the page does the reading and the writing, and this module is the
//! desktop's half:
//!
//! - **Export.** A backup opens a session (an id and a secret token, made per backup) and asks the
//!   page to export through [`AgentExport`] (the GUI evaluates one call in the page). The page
//!   builds a ZIP with its own code and posts it here in parts of at most [`PART_SIZE`] bytes, then
//!   says it is done. If the page is not there, does not answer in time, or fails, the backup
//!   carries on without it and is marked partial. A part is accepted only with the session's token,
//!   in order, and only into the file this session owns.
//! - **Import.** A restore that carries Agent storage leaves it in `<data>/restore/agent-import/`.
//!   The page asks for it when it starts (before it opens any project), takes a snapshot of its
//!   current storage for the undo (posted here, into the restore's undo folder), writes what the
//!   backup holds, and says it is done.
//!
//! No route here starts anything: sessions are opened only by the code that runs a backup.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{random_id, random_token, restore_dir, sha256_file};
use crate::secret_file;

/// The largest part the page sends or is sent.
pub const PART_SIZE: usize = 4 * 1024 * 1024;

/// What the backup says when the Agent's storage could not be included.
pub const MISSING_WARNING: &str = "Agent conversations and projects were not included: open the Agent and try again";

/// The most an export may add up to (compressed), whatever the page says.
pub const MAX_EXPORT_BYTES: u64 = 1 << 30;

/// Asks the Agent page to export. The GUI implements it by evaluating a call in the page's webview.
pub trait AgentExport: Send + Sync {
    /// Ask the page to export into session `id`, proving it with `token`. `Err` when the page
    /// cannot be asked (it is not open); the reason is for the log, not for the person.
    fn request(&self, id: &str, token: &str, include_keys: bool) -> Result<(), String>;
}

/// How long to wait for the page.
#[derive(Clone, Copy, Debug)]
pub struct AgentWait {
    /// For the first sign of life (a part or the end).
    pub first_activity: Duration,
    /// Between one sign of life and the next.
    pub idle: Duration,
    /// For the whole export.
    pub total: Duration,
}

impl Default for AgentWait {
    fn default() -> Self {
        Self { first_activity: Duration::from_secs(45), idle: Duration::from_secs(60), total: Duration::from_secs(600) }
    }
}

/// What the page says when it is done (camelCase, as the page sends it).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DonePayload {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub parts: u32,
    #[serde(default)]
    pub counts: ExportCounts,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportCounts {
    #[serde(default)]
    pub projects: u64,
    #[serde(default)]
    pub incognito_skipped: u64,
    #[serde(default)]
    pub conversations: u64,
    #[serde(default)]
    pub files: u64,
    #[serde(default)]
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartError {
    /// No such session (never opened, or over).
    Unknown,
    /// The token does not match.
    Denied,
    /// A part out of order.
    Sequence,
    /// Past a limit.
    TooLarge,
    /// The session is closed to more.
    Closed,
    /// The disk.
    Io,
}

struct Session {
    token: String,
    path: PathBuf,
    file: Option<File>,
    next_seq: u32,
    bytes: u64,
    max_bytes: u64,
    done: Option<DonePayload>,
    last_activity: Instant,
    activity: bool,
}

type Shared = Arc<(Mutex<Session>, Condvar)>;

fn sessions() -> &'static Mutex<HashMap<String, Shared>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Shared>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn token_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// Open a session that receives parts into `path` (a new private file). Returns its id and token.
/// Only backup code calls this: no route can.
pub(crate) fn open_session(path: &Path, max_bytes: u64) -> std::io::Result<(String, String)> {
    let file = secret_file::create_new_owner_only(path)?;
    let (id, token) = (random_id(), random_token());
    let session = Session {
        token: token.clone(),
        path: path.to_path_buf(),
        file: Some(file),
        next_seq: 0,
        bytes: 0,
        max_bytes,
        done: None,
        last_activity: Instant::now(),
        activity: false,
    };
    sessions().lock().unwrap_or_else(|e| e.into_inner()).insert(id.clone(), Arc::new((Mutex::new(session), Condvar::new())));
    Ok((id, token))
}

fn find(id: &str) -> Option<Shared> {
    sessions().lock().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
}

/// Forget a session (its file stays where the caller put it).
pub(crate) fn close_session(id: &str) {
    sessions().lock().unwrap_or_else(|e| e.into_inner()).remove(id);
}

/// Take one part of the export: the session's token, in order, within the limits.
pub fn receive_part(id: &str, token: &str, seq: u32, bytes: &[u8]) -> Result<(), PartError> {
    let shared = find(id).ok_or(PartError::Unknown)?;
    let (lock, cv) = &*shared;
    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
    if !token_eq(&s.token, token) {
        return Err(PartError::Denied);
    }
    if s.done.is_some() || s.file.is_none() {
        return Err(PartError::Closed);
    }
    if seq != s.next_seq {
        return Err(PartError::Sequence);
    }
    if bytes.len() > PART_SIZE || s.bytes + bytes.len() as u64 > s.max_bytes {
        return Err(PartError::TooLarge);
    }
    let file = s.file.as_mut().expect("checked above");
    file.write_all(bytes).map_err(|_| PartError::Io)?;
    s.bytes += bytes.len() as u64;
    s.next_seq += 1;
    s.last_activity = Instant::now();
    s.activity = true;
    cv.notify_all();
    Ok(())
}

/// The page says it is done.
pub fn finish(id: &str, token: &str, done: DonePayload) -> Result<(), PartError> {
    let shared = find(id).ok_or(PartError::Unknown)?;
    let (lock, cv) = &*shared;
    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
    if !token_eq(&s.token, token) {
        return Err(PartError::Denied);
    }
    if s.done.is_some() {
        return Err(PartError::Closed);
    }
    if let Some(file) = s.file.take() {
        file.sync_all().map_err(|_| PartError::Io)?;
    }
    s.done = Some(done);
    s.activity = true;
    s.last_activity = Instant::now();
    cv.notify_all();
    Ok(())
}

#[derive(Debug)]
pub enum WaitError {
    /// Nothing was heard from the page.
    Silent,
    /// The page stopped answering part-way.
    Stalled,
    /// It took too long altogether.
    TooLong,
}

/// Wait for the page to finish session `id`.
fn wait_done(id: &str, wait: &AgentWait) -> Result<(DonePayload, u64, u32, PathBuf), WaitError> {
    let shared = find(id).ok_or(WaitError::Silent)?;
    let (lock, cv) = &*shared;
    let start = Instant::now();
    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(done) = s.done.clone() {
            return Ok((done, s.bytes, s.next_seq, s.path.clone()));
        }
        let allowed = if s.activity { wait.idle } else { wait.first_activity };
        let since = if s.activity { s.last_activity.elapsed() } else { start.elapsed() };
        if start.elapsed() >= wait.total {
            return Err(WaitError::TooLong);
        }
        if since >= allowed {
            return Err(if s.activity { WaitError::Stalled } else { WaitError::Silent });
        }
        let remaining = (allowed - since).min(wait.total - start.elapsed()).max(Duration::from_millis(1));
        let (guard, _) = cv.wait_timeout(s, remaining.min(Duration::from_millis(250))).unwrap_or_else(|e| e.into_inner());
        s = guard;
    }
}

/// What an export gave back.
pub struct Collected {
    pub done: DonePayload,
    pub bytes: u64,
}

/// Ask the page for its storage and wait for it, into `dest` (a new private file). Any failure is a
/// plain string for the log: the backup carries on without the Agent's storage.
pub fn collect(exporter: &dyn AgentExport, dest: &Path, include_keys: bool, wait: &AgentWait) -> Result<Collected, String> {
    let (id, token) = open_session(dest, MAX_EXPORT_BYTES).map_err(|e| format!("could not open a place for the Agent's storage: {e}"))?;
    let fail = |why: String| {
        close_session(&id);
        let _ = std::fs::remove_file(dest);
        Err(why)
    };
    if let Err(why) = exporter.request(&id, &token, include_keys) {
        return fail(format!("the Agent could not be asked: {why}"));
    }
    match wait_done(&id, wait) {
        Ok((done, bytes, parts, _)) => {
            close_session(&id);
            if !done.ok {
                let _ = std::fs::remove_file(dest);
                return Err(format!("the Agent could not export: {}", done.error.as_deref().unwrap_or("no reason given")));
            }
            if done.parts != parts {
                let _ = std::fs::remove_file(dest);
                return Err("the Agent's export did not arrive whole".to_string());
            }
            if bytes == 0 {
                let _ = std::fs::remove_file(dest);
                return Err("the Agent sent nothing".to_string());
            }
            Ok(Collected { done, bytes })
        }
        Err(e) => fail(match e {
            WaitError::Silent => "the Agent did not answer (is its page open?)".to_string(),
            WaitError::Stalled => "the Agent stopped part-way".to_string(),
            WaitError::TooLong => "the Agent took too long".to_string(),
        }),
    }
}

// ---- the import side -------------------------------------------------------------------------------

fn import_dir(data_dir: &Path) -> PathBuf {
    restore_dir(data_dir).join("agent-import")
}

/// What waits for the Agent page after a restore (or an undo) that carried its storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingImport {
    pub id: String,
    /// `restore` or `undo`.
    pub kind: String,
    pub size: u64,
    pub sha256: String,
}

/// What the page is told.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportMeta {
    pub pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_size: Option<u64>,
}

/// The token the page uses for the import in this run of the app: a new one each start, so a
/// token from an earlier run opens nothing.
fn import_token() -> &'static str {
    static TOKEN: OnceLock<String> = OnceLock::new();
    TOKEN.get_or_init(random_token)
}

pub(crate) fn read_pending_import(data_dir: &Path) -> Option<PendingImport> {
    let dir = import_dir(data_dir);
    let pending: PendingImport = serde_json::from_str(&std::fs::read_to_string(dir.join("current.json")).ok()?).ok()?;
    let zip = dir.join("current.zip");
    let meta = std::fs::symlink_metadata(&zip).ok()?;
    (meta.is_file() && meta.len() == pending.size).then_some(pending)
}

/// Leave `zip` (which is moved) for the page to import, replacing anything left from before.
pub(crate) fn leave_for_page(data_dir: &Path, id: &str, kind: &str, zip: &Path) -> std::io::Result<()> {
    let dir = import_dir(data_dir);
    secret_file::create_private_dir(&dir)?;
    let (sha256, size) = sha256_file(zip)?;
    let _ = std::fs::remove_file(dir.join("current.json"));
    secret_file::rename_over(zip, &dir.join("current.zip"))?;
    let meta = PendingImport { id: id.to_string(), kind: kind.to_string(), size, sha256 };
    secret_file::write(&dir.join("current.json"), serde_json::to_string_pretty(&meta).unwrap_or_default())
}

/// Forget an import nobody asked for (a restore that was cancelled or rolled back).
pub(crate) fn drop_pending_import(data_dir: &Path) {
    let dir = import_dir(data_dir);
    let _ = std::fs::remove_file(dir.join("current.json"));
    let _ = std::fs::remove_file(dir.join("current.zip"));
}

/// `GET /api/backup/agent-import`.
pub fn import_meta(data_dir: &Path) -> ImportMeta {
    match read_pending_import(data_dir) {
        None => ImportMeta { pending: false, id: None, token: None, kind: None, size: None, sha256: None, parts: None, part_size: None },
        Some(p) => ImportMeta {
            pending: true,
            token: Some(import_token().to_string()),
            parts: Some(p.size.div_ceil(PART_SIZE as u64)),
            part_size: Some(PART_SIZE as u64),
            kind: Some(p.kind),
            size: Some(p.size),
            sha256: Some(p.sha256),
            id: Some(p.id),
        },
    }
}

fn check_import(data_dir: &Path, id: &str, token: &str) -> Result<PendingImport, PartError> {
    if !token_eq(import_token(), token) {
        return Err(PartError::Denied);
    }
    let pending = read_pending_import(data_dir).ok_or(PartError::Unknown)?;
    if pending.id != id {
        return Err(PartError::Unknown);
    }
    Ok(pending)
}

/// `GET /api/backup/agent-import/{id}/part/{i}`.
pub fn import_part(data_dir: &Path, id: &str, token: &str, index: u64) -> Result<Vec<u8>, PartError> {
    let pending = check_import(data_dir, id, token)?;
    let start = index.checked_mul(PART_SIZE as u64).ok_or(PartError::Sequence)?;
    if start >= pending.size {
        return Err(PartError::Sequence);
    }
    let len = (pending.size - start).min(PART_SIZE as u64) as usize;
    let mut file = File::open(import_dir(data_dir).join("current.zip")).map_err(|_| PartError::Io)?;
    file.seek(SeekFrom::Start(start)).map_err(|_| PartError::Io)?;
    let mut out = vec![0u8; len];
    file.read_exact(&mut out).map_err(|_| PartError::Io)?;
    Ok(out)
}

/// The page's own storage as it was just before a restore replaced it, kept for the undo.
pub(crate) fn undo_agent_path(data_dir: &Path, id: &str) -> PathBuf {
    restore_dir(data_dir).join(format!("undo-{id}")).join("agent-storage.zip")
}

fn undo_partial_path(data_dir: &Path, id: &str) -> PathBuf {
    restore_dir(data_dir).join(format!("undo-{id}")).join("agent-storage.zip.part")
}

/// Where the page's snapshot has got to: the restore it belongs to, and the next part expected.
static UNDO_PROGRESS: Mutex<Option<HashMap<String, u32>>> = Mutex::new(None);

/// `POST .../undo-part?seq=n`: one part of the page's snapshot of its storage before it imports.
pub fn undo_part(data_dir: &Path, id: &str, token: &str, seq: u32, bytes: &[u8]) -> Result<(), PartError> {
    let pending = check_import(data_dir, id, token)?;
    if pending.kind != "restore" {
        return Err(PartError::Closed);
    }
    if bytes.len() > PART_SIZE {
        return Err(PartError::TooLarge);
    }
    let path = undo_partial_path(data_dir, id);
    let mut progress = UNDO_PROGRESS.lock().unwrap_or_else(|e| e.into_inner());
    if seq == 0 {
        if let Some(parent) = path.parent() {
            secret_file::create_private_dir(parent).map_err(|_| PartError::Io)?;
        }
        let _ = std::fs::remove_file(&path);
        secret_file::create_new_owner_only(&path).map_err(|_| PartError::Io)?;
        progress.get_or_insert_with(HashMap::new).insert(id.to_string(), 0);
    }
    match progress.as_mut().and_then(|p| p.get_mut(id)) {
        Some(next) if *next == seq => *next += 1,
        _ => return Err(PartError::Sequence),
    }
    let len = std::fs::metadata(&path).map_err(|_| PartError::Sequence)?.len();
    if len + bytes.len() as u64 > MAX_EXPORT_BYTES {
        return Err(PartError::TooLarge);
    }
    let mut file = OpenOptions::new().append(true).open(&path).map_err(|_| PartError::Io)?;
    file.write_all(bytes).map_err(|_| PartError::Io)?;
    Ok(())
}

/// `POST .../undo-done`: the snapshot is whole (or the page says it failed, and it is thrown away).
pub fn undo_done(data_dir: &Path, id: &str, token: &str, done: &DonePayload) -> Result<(), PartError> {
    let pending = check_import(data_dir, id, token)?;
    if pending.kind != "restore" {
        return Err(PartError::Closed);
    }
    let partial = undo_partial_path(data_dir, id);
    if !done.ok {
        let _ = std::fs::remove_file(&partial);
        return Ok(());
    }
    let parts = match UNDO_PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|p| p.remove(id)) {
        Some(next) => next,
        None => return Err(PartError::Sequence),
    };
    if parts != done.parts || parts == 0 {
        let _ = std::fs::remove_file(&partial);
        return Err(PartError::Sequence);
    }
    secret_file::rename_over(&partial, &undo_agent_path(data_dir, id)).map_err(|_| PartError::Io)
}

/// `POST .../done`: the page has imported (or could not).
pub fn import_done(data_dir: &Path, id: &str, token: &str, ok: bool, error: Option<&str>) -> Result<(), PartError> {
    check_import(data_dir, id, token)?;
    drop_pending_import(data_dir);
    super::restore::record_agent_result(data_dir, id, ok, error);
    Ok(())
}

