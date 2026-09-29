//! Backing up OAIY's own data and restoring it, on this computer or another.
//!
//! One file, `<name>.oaiybackup`: a ZIP (deflate) whose first entry is `manifest.json`, encrypted
//! as a whole in the standard age file format (passphrase mode, scrypt, no armor:
//! <https://age-encryption.org/v1>), so `age --decrypt` on any machine gives the ZIP back and
//! nothing here is home-made cryptography. The passphrase is typed by the person, is at least
//! [`MIN_PASSPHRASE_CHARS`] characters, and is never stored or logged: a lost passphrase is a lost
//! backup.
//!
//! This is the core, with no GUI types in it, so a command line tool and the headless server can
//! use it later. The parts:
//!
//! - [`table`]: THE classification table (`table.json`): every path under the data folder, every name
//!   in the Agent's storage and every key of a settings file, each excluded, data or runs-things. A
//!   restore is default-deny: what the table does not list is not restored.
//! - [`rules`]: what a backup holds, what it leaves out and why, from that table (every credential
//!   and key is left out; the API provider keys are added only when asked).
//! - [`manifest`]: the record at the front of the ZIP.
//! - [`container`]: the ZIP and age plumbing, and the checks every name and size goes through.
//! - [`create`]: making the file (staged, encrypted to a `.tmp`, checked by decrypting it, then
//!   renamed into place).
//! - [`restore`]: a dry run, staging, applying at the next start (before any store loads), and
//!   undoing the last restore.
//! - [`agent`]: the Agent page's own storage (conversations and projects in the WebView profile,
//!   which only that page can read), moved in parts through internal routes.
//! - [`busy`]: whether the app is in the middle of something (a call, a task, a download, a job).
//! - [`state`]: the small record of the last backup, the running phase and the last restore.
//! - [`routes`]: `GET /api/backup/status` and the internal routes the Agent page uses. No route
//!   creates or restores a backup: that is done by commands only the dashboard's own window can call.
//!
//! Every file this writes (the staged copies, the output, the restored files) is created private
//! from its first byte through [`crate::secret_file`], and no error or log line carries a passphrase
//! or the contents of a file.

pub mod agent;
pub mod agentzip;
pub mod busy;
#[cfg(feature = "gui")]
pub mod commands;
pub mod container;
pub mod create;
pub mod desk;
pub mod manifest;
pub mod restore;
pub mod review;
pub mod routes;
pub mod rules;
pub mod sanitize;
pub mod state;
pub mod table;

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The shortest passphrase a backup is made with, in characters.
pub const MIN_PASSPHRASE_CHARS: usize = 12;

/// The extension of a backup file.
pub const EXTENSION: &str = "oaiybackup";

/// The name of the manifest, the first entry of every backup.
pub const MANIFEST_NAME: &str = "manifest.json";

/// Where the Agent page's storage sits inside a backup.
pub const AGENT_ENTRY: &str = "agent/agent-storage.zip";

/// What a backup may hold and what a restore accepts, so a damaged or hostile file cannot make
/// this read or write without bound.
///
/// The numbers are what a real backup never reaches, with room to spare, not what a computer could
/// hold: a hostile file that asks for more is refused from its record, before any of it is read.
/// A real data folder (leaving out models, programs and logs) is a few hundred files: contacts, the
/// calendar, flows, templates, a few voices, a plugin's settings and the Agent's storage in one
/// archive. Its largest single things are the Agent's storage (the Agent's own export stops at
/// 512 MiB and never adds a file over 64 MiB), a voice sample (tens of megabytes) and the calendar
/// or the contacts (megabytes of JSON).
#[derive(Clone, Debug)]
pub struct Limits {
    /// Entries in the ZIP, the manifest included: 20,000, about 40 times what a real backup holds.
    pub max_entries: usize,
    /// One entry's size, uncompressed: 1 GiB, more than the largest thing a backup holds (below).
    pub max_entry_bytes: u64,
    /// All entries together, uncompressed: 4 GiB.
    pub max_total_bytes: u64,
    /// The length of one entry name.
    pub max_name_len: usize,
    /// The manifest's size: 16 MiB, about 600 bytes for each of 20,000 entries and more.
    pub max_manifest_bytes: u64,
    /// One JSON or text item (contacts, the calendar, a flow, the settings): 16 MiB. The calendar of a
    /// business over many years is a few megabytes; a file that is larger is not a calendar.
    pub max_json_bytes: u64,
    /// One voice item: 128 MiB.
    pub max_voice_bytes: u64,
    /// The Agent's storage archive: 640 MiB (its own export stops at 512 MiB and never adds a file over
    /// 64 MiB, so 576 MiB at most, and a margin for the archive's own records).
    pub max_agent_bytes: u64,
    /// Files in the Agent's archive: 50,000. Its export adds files until 512 MiB are in, so a very large
    /// number would be tens of thousands of small ones: a dependency tree in a project, not what the Agent wrote.
    pub max_agent_entries: usize,
    /// One file in the Agent's archive, as its own directory declares it: 64 MiB, the most its export puts in.
    pub max_agent_file_bytes: u64,
    /// All the files of the Agent's archive together, as its own directory declares them (what they unpack to,
    /// which a hostile archive can make far more than its size): 640 MiB.
    pub max_agent_total_bytes: u64,
    /// The most of one small structured file of the Agent's (a campaign, its settings, what it remembers) that
    /// is read whole to be described or rebuilt: 8 MiB.
    pub max_agent_read_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_entries: 20_000,
            max_entry_bytes: 1 << 30,
            max_total_bytes: 4 << 30,
            max_name_len: 512,
            max_manifest_bytes: 16 << 20,
            max_json_bytes: 16 << 20,
            max_voice_bytes: 128 << 20,
            max_agent_bytes: 640 << 20,
            max_agent_entries: 50_000,
            max_agent_file_bytes: 64 << 20,
            max_agent_total_bytes: 640 << 20,
            max_agent_read_bytes: 8 << 20,
        }
    }
}

impl Limits {
    /// The most one item of this name may be: what kind of thing it is decides, and no item is more than
    /// [`Limits::max_entry_bytes`]. (A name the table does not know is held to the least: it is only ever
    /// read to be checked, never brought back.)
    pub fn entry_cap(&self, name: &str) -> u64 {
        let by_kind = if name == AGENT_ENTRY {
            self.max_agent_bytes
        } else if name.to_lowercase().starts_with("voices/") {
            self.max_voice_bytes
        } else {
            self.max_json_bytes
        };
        by_kind.min(self.max_entry_bytes)
    }
}

/// How long a check or a staging may run: a hostile file must not be able to keep it busy for ever.
/// Checked between the pieces of every loop that reads or writes a backup.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    deadline: Option<std::time::Instant>,
}

impl Budget {
    /// No limit (a backup being made, which has its own bounds).
    pub const fn unlimited() -> Self {
        Self { deadline: None }
    }

    /// Stop after `limit` from now.
    pub fn within(limit: std::time::Duration) -> Self {
        Self { deadline: Some(std::time::Instant::now() + limit) }
    }

    pub(crate) fn check(&self) -> Result<()> {
        match self.deadline {
            Some(t) if std::time::Instant::now() >= t => Err(BackupError::new(ErrorKind::Timeout, "That took too long, so it was stopped. Try again.")),
            _ => Ok(()),
        }
    }
}

/// The longest an inspect or a staging is let run.
pub const RESTORE_TIME_LIMIT: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// Why something failed, so a caller can tell a wrong passphrase from a busy app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The app is in the middle of something.
    Busy,
    /// The passphrase is too short or missing.
    Passphrase,
    /// The passphrase did not open the file.
    WrongPassphrase,
    /// The file is not a backup, or it is damaged or cut short.
    Damaged,
    /// A backup this version does not understand.
    Unsupported,
    /// A backup that asks for something a backup must not (a name that leaves its folder, a
    /// credential, a symbolic link).
    Unsafe,
    /// A limit was passed.
    TooLarge,
    /// Not enough room on a disk.
    NoSpace,
    /// A disk or file problem.
    Io,
    /// Something else is already going on (a backup running, a restore waiting).
    Conflict,
    /// The check made after writing failed.
    Verify,
    /// It took longer than it is allowed to, and was stopped.
    Timeout,
}

/// A failure with a message a person can read: never a passphrase, never a file's contents.
#[derive(Clone, Debug)]
pub struct BackupError {
    pub kind: ErrorKind,
    pub message: String,
}

impl BackupError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    pub(crate) fn io(what: &str, e: &io::Error) -> Self {
        Self::new(ErrorKind::Io, format!("{what}: {}", plain_io(e)))
    }
}

impl fmt::Display for BackupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BackupError {}

pub type Result<T> = std::result::Result<T, BackupError>;

/// An `io::Error` in a few plain words (the operating system's text, without a path).
fn plain_io(e: &io::Error) -> String {
    match e.kind() {
        io::ErrorKind::NotFound => "not found".to_string(),
        io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        io::ErrorKind::AlreadyExists => "already exists".to_string(),
        io::ErrorKind::UnexpectedEof => "the file ends too soon".to_string(),
        _ => e.to_string(),
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// A random id of 16 hexadecimal characters.
pub(crate) fn random_id() -> String {
    let mut bytes = [0u8; 8];
    getrandom::getrandom(&mut bytes).expect("the operating system's randomness");
    hex(&bytes)
}

/// A random secret of 64 hexadecimal characters.
pub(crate) fn random_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("the operating system's randomness");
    hex(&bytes)
}

/// A private folder that is removed, with everything in it, when this goes out of scope.
pub(crate) struct TempFolder(pub(crate) PathBuf);

impl TempFolder {
    /// A new folder with a random name inside `parent` (which is made if it is missing).
    pub(crate) fn new(parent: &Path) -> Result<Self> {
        let path = parent.join(random_id());
        crate::secret_file::create_private_dir(&path).map_err(|e| BackupError::io("Could not make a working folder", &e))?;
        Ok(Self(path))
    }
}

impl Drop for TempFolder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The SHA-256 of a file, streamed.
pub(crate) fn sha256_file(path: &Path) -> io::Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

/// Where the backup keeps what it makes for a while (staged copies of a run, a restore being
/// looked at): private, under the data folder, never in the folder the person chose for the
/// backup (that may be synchronised or on a drive that is taken away).
pub(crate) fn scratch_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("backup").join("scratch")
}

/// Where staged restores and undo snapshots are kept.
pub(crate) fn restore_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("restore")
}

/// Free bytes on the disk that holds `path` (or the nearest folder that exists).
pub fn free_space(path: &Path) -> u64 {
    let mut at = Some(path);
    while let Some(p) = at {
        if p.exists() {
            return fs2::available_space(p).unwrap_or(u64::MAX);
        }
        at = p.parent();
    }
    u64::MAX
}

/// Check a passphrase is long enough (counting characters, not bytes) without ever putting it in a message.
pub fn check_passphrase(passphrase: &str) -> Result<()> {
    if passphrase.chars().count() < MIN_PASSPHRASE_CHARS {
        return Err(BackupError::new(
            ErrorKind::Passphrase,
            format!("The passphrase must be at least {MIN_PASSPHRASE_CHARS} characters."),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod peak;
#[cfg(test)]
mod table_tests;
#[cfg(test)]
mod tests;
