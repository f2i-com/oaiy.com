//! Making a backup.
//!
//! In order: refuse if the app is busy; walk the data folder by the rules and copy each file into a
//! private staging folder under the data folder, hashing as it copies (so the manifest describes
//! exactly what was copied, even if the live file changes a moment later); ask the Agent page for
//! its storage; write the ZIP with the manifest first; encrypt it with age to a `.tmp` beside the
//! final name; decrypt that `.tmp` again with the same passphrase and check every item against the
//! manifest; and only then rename it into place. A file that does not check out is deleted and the
//! backup fails, and a backup that already has the final name is left as it was.
//!
//! The staged plaintext lives under the data folder (never in the folder the person chose, which may
//! be synchronised or on a drive that is taken away) and is removed when the run ends, however it ends.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::agent::{self, AgentExport, AgentWait, MISSING_WARNING};
use super::busy::BusySignals;
use super::container::{self, Cost};
use super::manifest::{AppInfo, Counts, Entry, Manifest, VERSION};
use super::rules::{self, Excluded};
use super::state::{self, Phase};
use super::{check_passphrase, free_space, Budget, hex, random_id, scratch_dir, BackupError, ErrorKind, Limits, Result, TempFolder, AGENT_ENTRY, EXTENSION};
use crate::secret_file;

/// Room to leave on a disk beyond what the backup needs.
const MARGIN: u64 = 64 << 20;

/// What to back up, and where.
pub struct CreateOptions<'a> {
    pub data_dir: &'a Path,
    /// The file to make (`.oaiybackup` is added when the name has no extension of that kind).
    pub dest: PathBuf,
    pub passphrase: &'a str,
    pub include_keys: bool,
    pub limits: Limits,
    pub app_version: String,
    /// What is going on in the app now (the caller gathers it).
    pub busy: BusySignals,
    /// How to ask the Agent page for its storage; `None` when there is no page to ask.
    pub agent: Option<&'a dyn AgentExport>,
    pub agent_wait: AgentWait,
    pub(crate) cost: Cost,
    pub(crate) free_space: fn(&Path) -> u64,
}

impl<'a> CreateOptions<'a> {
    pub fn new(data_dir: &'a Path, dest: impl Into<PathBuf>, passphrase: &'a str) -> Self {
        Self {
            data_dir,
            dest: dest.into(),
            passphrase,
            include_keys: false,
            limits: Limits::default(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            busy: BusySignals::default(),
            agent: None,
            agent_wait: AgentWait::default(),
            cost: Cost::Default,
            free_space,
        }
    }
}

/// What a finished backup tells the dashboard (camelCase).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateResult {
    pub path: String,
    pub file_name: String,
    pub size: u64,
    pub created_at: String,
    pub counts: Counts,
    pub includes_keys: bool,
    pub partial: Vec<String>,
    pub excluded: Vec<Excluded>,
    /// The file was decrypted again and every item checked after it was written.
    pub verified: bool,
}

fn need(free: u64, needed: u64, what: &str) -> Result<()> {
    if free < needed {
        return Err(BackupError::new(
            ErrorKind::NoSpace,
            format!("There is not enough free space {what}: about {} MB is needed and {} MB is free.", needed >> 20, free >> 20),
        ));
    }
    Ok(())
}

/// Make a backup. Blocking: the caller runs it off the UI thread.
pub fn create(opts: &CreateOptions<'_>) -> Result<CreateResult> {
    opts.busy.refuse_if_busy("making a backup")?;
    check_passphrase(opts.passphrase)?;
    let Some(_running) = state::begin_run() else {
        return Err(BackupError::new(ErrorKind::Conflict, "A backup is already being made."));
    };
    let result = run(opts);
    if result.is_err() {
        state::record_failure(opts.data_dir);
    }
    result
}

fn final_path(dest: &Path) -> PathBuf {
    match dest.extension().and_then(|e| e.to_str()) {
        Some(e) if e.eq_ignore_ascii_case(EXTENSION) => dest.to_path_buf(),
        _ => {
            let mut name = dest.as_os_str().to_owned();
            name.push(format!(".{EXTENSION}"));
            PathBuf::from(name)
        }
    }
}

fn run(opts: &CreateOptions<'_>) -> Result<CreateResult> {
    let dest = final_path(&opts.dest);
    let dest_dir = dest.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    if dest.is_dir() {
        return Err(BackupError::new(ErrorKind::Io, "The place chosen for the backup is a folder."));
    }
    std::fs::create_dir_all(&dest_dir).map_err(|e| BackupError::io("Could not make the folder for the backup", &e))?;

    // ---- collect ----
    state::set_phase(Phase::Collecting, "Copying your files");
    let plan = rules::plan(opts.data_dir, opts.include_keys);
    let planned: u64 = plan.items.iter().map(|i| i.size).sum();
    need((opts.free_space)(opts.data_dir), planned.saturating_mul(2) + MARGIN, "on the drive OAIY keeps its data on")?;
    need((opts.free_space)(&dest_dir), planned + MARGIN, "in the folder chosen for the backup")?;

    // The staged plaintext lives in here and is removed when this function ends, however it ends.
    let cleanup = TempFolder::new(&scratch_dir(opts.data_dir))?;
    let run_dir = cleanup.0.clone();
    let tree = run_dir.join("tree");
    secret_file::create_private_dir(&tree).map_err(|e| BackupError::io("Could not make a working folder", &e))?;

    let mut partial: Vec<String> = Vec::new();
    let mut entries: Vec<Entry> = Vec::new();
    let mut sources: Vec<(String, PathBuf)> = Vec::new();
    let mut total_bytes = 0u64;
    let mut skipped_unreadable = 0usize;
    let mut skipped_large = 0usize;
    for (n, item) in plan.items.iter().enumerate() {
        if item.size > opts.limits.max_entry_bytes {
            skipped_large += 1;
            continue;
        }
        if total_bytes.saturating_add(item.size) > opts.limits.max_total_bytes || entries.len() + 3 > opts.limits.max_entries {
            return Err(BackupError::new(ErrorKind::TooLarge, "The data to back up is larger than a backup can hold."));
        }
        let staged = tree.join(format!("f{n:06}"));
        match copy_hashing(&item.abs, &staged) {
            Ok((sha256, size)) => {
                total_bytes += size;
                entries.push(Entry { name: item.rel.clone(), size, sha256 });
                sources.push((item.rel.clone(), staged));
            }
            // A file that went away or is locked: the backup goes on without it.
            Err(_) => skipped_unreadable += 1,
        }
    }
    if skipped_unreadable > 0 {
        partial.push(format!("{skipped_unreadable} file{} could not be read and {} left out.", plural(skipped_unreadable), were(skipped_unreadable)));
    }
    if skipped_large > 0 {
        partial.push(format!("{skipped_large} file{} too large for a backup and {} left out.", plural(skipped_large), were(skipped_large)));
    }
    if !plan.unreadable.is_empty() {
        partial.push(format!("{} folder{} could not be read.", plan.unreadable.len(), plural(plan.unreadable.len())));
    }

    // ---- the Agent's storage ----
    let mut counts = Counts::default();
    state::set_phase(Phase::Agent, "Asking the Agent for its conversations and projects");
    match opts.agent {
        Some(exporter) => {
            let zip = run_dir.join("agent-storage.zip");
            match agent::collect(exporter, &zip, opts.include_keys, &opts.agent_wait) {
                Ok(got) => {
                    let (sha256, size) = super::sha256_file(&zip).map_err(|e| BackupError::io("Could not read the Agent's storage", &e))?;
                    if size > opts.limits.max_entry_bytes || total_bytes.saturating_add(size) > opts.limits.max_total_bytes {
                        partial.push(MISSING_WARNING.to_string());
                    } else {
                        total_bytes += size;
                        entries.push(Entry { name: AGENT_ENTRY.to_string(), size, sha256 });
                        sources.push((AGENT_ENTRY.to_string(), zip));
                        counts.agent_projects = got.done.counts.projects;
                        counts.agent_conversations = got.done.counts.conversations;
                        counts.agent_files = got.done.counts.files;
                        partial.extend(got.done.warnings.iter().map(|w| format!("Agent: {w}")));
                    }
                }
                Err(why) => {
                    log::info!("backup: the Agent's storage was not included: {why}");
                    partial.push(MISSING_WARNING.to_string());
                }
            }
        }
        None => partial.push("The Agent's conversations and projects are not part of this backup: there was no Agent page to ask.".to_string()),
    }

    // ---- the record ----
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    sources.sort_by(|a, b| a.0.cmp(&b.0));
    counts.files = entries.len() as u64;
    counts.bytes = total_bytes;
    let mut excluded = plan.excluded.clone();
    for link in &plan.links {
        excluded.push(Excluded {
            pattern: link.clone(),
            reason: "A symbolic link or junction: it is never followed, so what it points at is not backed up.".to_string(),
            redo: None,
        });
    }
    let created_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let manifest = Manifest {
        v: VERSION,
        created_at: created_at.clone(),
        app: AppInfo { name: super::manifest::APP_NAME.to_string(), version: opts.app_version.clone() },
        platform: std::env::consts::OS.to_string(),
        entries,
        excluded,
        counts: counts.clone(),
        includes_keys: opts.include_keys,
        partial: partial.clone(),
    };
    // A manifest this code wrote must pass the checks a restore makes on it.
    manifest.validate(&opts.limits)?;

    // ---- pack, encrypt, check ----
    state::set_phase(Phase::Packing, "Packing the backup");
    let zip_path = run_dir.join("backup.zip");
    container::write_zip(&zip_path, &manifest.to_json(), &sources)?;
    let staged_len = std::fs::metadata(&zip_path).map(|m| m.len()).unwrap_or(0);
    need((opts.free_space)(&dest_dir), staged_len + MARGIN, "in the folder chosen for the backup")?;

    state::set_phase(Phase::Encrypting, "Encrypting the backup with your passphrase");
    let tmp = dest_dir.join(format!(".{}.{}.tmp", dest.file_name().and_then(|n| n.to_str()).unwrap_or("backup"), random_id()));
    let finished = (|| -> Result<u64> {
        container::encrypt_file(&zip_path, &tmp, opts.passphrase, opts.cost)?;
        #[cfg(test)]
        if CORRUPT_OUTPUT.with(|c| c.get()) {
            // A test's stand-in for a disk that wrote something other than it was given.
            let mut bytes = std::fs::read(&tmp).expect("the output");
            let at = bytes.len() * 3 / 4;
            bytes[at] ^= 0x40;
            std::fs::write(&tmp, bytes).expect("the output");
        }
        state::set_phase(Phase::Verifying, "Checking the backup: opening it again and testing every item");
        let scratch = run_dir.join("verify");
        secret_file::create_private_dir(&scratch).map_err(|e| BackupError::io("Could not make a working folder", &e))?;
        let verified = container::open_backup(&tmp, opts.passphrase, &scratch, &opts.limits, &Budget::unlimited())
            .map_err(|e| BackupError::new(ErrorKind::Verify, format!("The backup did not check out after it was written, so it was deleted: {e}")))?;
        if verified.manifest != manifest {
            return Err(BackupError::new(ErrorKind::Verify, "The backup did not check out after it was written, so it was deleted: it does not match what was meant to be in it."));
        }
        let _ = std::fs::remove_dir_all(&scratch);
        secret_file::rename_over(&tmp, &dest).map_err(|e| BackupError::io("Could not put the backup in place", &e))?;
        Ok(std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0))
    })();
    let size = match finished {
        Ok(size) => size,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    state::record_success(opts.data_dir, &created_at, size);
    Ok(CreateResult {
        path: dest.display().to_string(),
        file_name: dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        size,
        created_at,
        counts,
        includes_keys: opts.include_keys,
        partial,
        excluded: manifest.excluded,
        verified: true,
    })
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

fn were(n: usize) -> &'static str {
    if n == 1 {
        "was"
    } else {
        "were"
    }
}

/// Copy `from` to `to` (a new private file), hashing as it goes. Returns the SHA-256 and the length copied.
fn copy_hashing(from: &Path, to: &Path) -> std::io::Result<(String, u64)> {
    let mut input = BufReader::with_capacity(64 * 1024, File::open(from)?);
    let mut out = BufWriter::with_capacity(64 * 1024, secret_file::create_new_owner_only(to)?);
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
        out.write_all(&buf[..n])?;
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok((hex(&hasher.finalize()), total))
}

#[cfg(test)]
thread_local! {
    /// Make the file that was just written differ from what was encrypted, on this thread only.
    pub(crate) static CORRUPT_OUTPUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How long an Agent export may take in a test that wants it to fail quickly.
#[cfg(test)]
pub(crate) fn short_wait() -> AgentWait {
    use std::time::Duration;
    AgentWait { first_activity: Duration::from_millis(300), idle: Duration::from_millis(300), total: Duration::from_secs(3) }
}
