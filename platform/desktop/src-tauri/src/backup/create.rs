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
use super::busy::Busy;
use super::container::{self, Cost};
use super::manifest::{AppInfo, Counts, Entry, Manifest, VERSION};
use super::rules::{self, Excluded, Sanitize};
use super::sanitize;
use super::state::{self, Phase};
use super::table::{filter_json, table, Why};
use super::{check_passphrase, free_space, Budget, hex, random_id, scratch_dir, BackupError, ErrorKind, Limits, Result, TempFolder, AGENT_ENTRY, EXTENSION};
use crate::secret_file;

/// Room to leave on a disk beyond what the backup needs.
pub(crate) const MARGIN: u64 = 64 << 20;

/// What the drive that holds OAIY's data needs before a backup starts: the working copy of the files,
/// the ZIP made from them and the copy that is opened again to check it are each about the size of the
/// data, and the ZIP has some overhead of its own. (The Agent's storage is added when its size is known.)
pub(crate) fn data_drive_needed(planned: u64) -> u64 {
    planned.saturating_mul(32) / 10 + MARGIN
}

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
    pub busy: Busy,
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
            busy: Busy::none(),
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

/// Where a backup being written is (the one file that lives outside the data folder while it is made):
/// noted before it is written and forgotten when it is renamed into place or removed, so that if OAIY
/// is killed in between, the next start can find the half-written (encrypted) file and remove it.
fn output_record(data_dir: &Path) -> PathBuf {
    scratch_dir(data_dir).parent().map(|p| p.join("output.json")).unwrap_or_else(|| data_dir.join("backup").join("output.json"))
}

fn note_output(data_dir: &Path, tmp: &Path) {
    let record = serde_json::json!({ "path": tmp.display().to_string() });
    if let Some(parent) = output_record(data_dir).parent() {
        let _ = secret_file::create_private_dir(parent);
    }
    if let Err(e) = secret_file::write(&output_record(data_dir), record.to_string()) {
        log::warn!("backup: could not note where the backup is being written: {e}");
    }
}

fn forget_output(data_dir: &Path) {
    let _ = std::fs::remove_file(output_record(data_dir));
}

/// A name this code gives the file it writes first: `.<name>.<16 hex>.tmp`.
fn is_partial_name(name: &str) -> bool {
    let Some(stem) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp")) else { return false };
    let Some((_, id)) = stem.rsplit_once('.') else { return false };
    id.len() == 16 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// At the start of the app: remove the half-written (encrypted) file a backup that was killed left
/// beside where it was going, if it can still be found. Only a file of exactly the name this code gives
/// it, and never a link. True when a file was removed.
pub(crate) fn sweep_output(data_dir: &Path) -> bool {
    let record = output_record(data_dir);
    let Some(path) = std::fs::read_to_string(&record).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).and_then(|v| v.get("path").and_then(|p| p.as_str().map(PathBuf::from))) else {
        let _ = std::fs::remove_file(&record);
        return false;
    };
    let mut removed = false;
    if path.file_name().and_then(|n| n.to_str()).is_some_and(is_partial_name) {
        if let Ok(meta) = std::fs::symlink_metadata(&path) {
            if meta.is_file() && !super::rules::is_link(&meta) {
                removed = std::fs::remove_file(&path).is_ok();
            }
        }
    }
    let _ = std::fs::remove_file(&record);
    removed
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
    need((opts.free_space)(opts.data_dir), data_drive_needed(planned), "on the drive OAIY keeps its data on")?;
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
    let mut skipped_named = 0usize;
    let mut extra_excluded: Vec<Excluded> = Vec::new();
    for (n, item) in plan.items.iter().enumerate() {
        // A file whose name a restore would refuse (a short-name alias such as `REPORT~1.JSON`, a reserved device name) is left
        // out and named, not made into a backup that is refused whole.
        if let Some(why) = container::name_problem(&item.rel, &opts.limits) {
            skipped_named += 1;
            extra_excluded.push(Excluded {
                pattern: item.rel.clone(),
                reason: format!("Left out: its name is one a restore refuses ({why}), and a backup that held it could not be restored."),
                redo: Some("Rename the file if you want it in a backup.".to_string()),
            });
            continue;
        }
        // A file too large for a restore to take is left out and said so, never made into a backup that is refused.
        if item.size > opts.limits.entry_cap(&item.rel) {
            skipped_large += 1;
            continue;
        }
        if total_bytes.saturating_add(item.size) > opts.limits.max_total_bytes || entries.len() + 3 > opts.limits.max_entries {
            return Err(BackupError::new(ErrorKind::TooLarge, "The data to back up is larger than a backup can hold."));
        }
        let staged = tree.join(format!("f{n:06}"));
        let copied = match item.sanitize {
            Sanitize::None => copy_hashing(&item.abs, &staged).map_err(|_| None),
            kind => copy_cleaned(&item.abs, &staged, &item.rel, kind, &mut extra_excluded).map_err(Some),
        };
        match copied {
            Ok((sha256, size)) => {
                total_bytes += size;
                entries.push(Entry { name: item.rel.clone(), size, sha256 });
                sources.push((item.rel.clone(), staged));
            }
            // A file that could not be cleaned is left out, and listed with why.
            Err(Some(why)) => extra_excluded.push(Excluded { pattern: item.rel.clone(), reason: format!("Left out: {why}."), redo: None }),
            // A file that went away or is locked: the backup goes on without it.
            Err(None) => skipped_unreadable += 1,
        }
    }
    if skipped_unreadable > 0 {
        partial.push(format!("{skipped_unreadable} file{} could not be read and {} left out.", plural(skipped_unreadable), were(skipped_unreadable)));
    }
    if skipped_named > 0 {
        partial.push(format!("{skipped_named} file{} {} left out because {} name is one a restore refuses (they are named in what was left out).", plural(skipped_named), were(skipped_named), if skipped_named == 1 { "its" } else { "their" }));
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
                    if size > opts.limits.entry_cap(AGENT_ENTRY) || total_bytes.saturating_add(size) > opts.limits.max_total_bytes {
                        partial.push(MISSING_WARNING.to_string());
                    } else {
                        total_bytes += size;
                        entries.push(Entry { name: AGENT_ENTRY.to_string(), size, sha256 });
                        sources.push((AGENT_ENTRY.to_string(), zip));
                        counts.agent_projects = got.done.counts.projects;
                        counts.agent_conversations = got.done.counts.conversations;
                        counts.agent_files = got.done.counts.files;
                        // What the page says is cut to what a panel shows, and there is only so much of it.
                        partial.extend(super::parts::lines_of(&got.done.warnings, super::parts::MOST_PAGE_WARNINGS_NAMED, 300).into_iter().map(|w| format!("Agent: {w}")));
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

    // The Agent's storage is on the disk now, and in the count: the ZIP made of everything, and the
    // copy that is opened again to check it, still have to fit beside it.
    need((opts.free_space)(opts.data_dir), total_bytes.saturating_mul(2).saturating_add(MARGIN), "on the drive OAIY keeps its data on")?;

    // ---- the record ----
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    sources.sort_by(|a, b| a.0.cmp(&b.0));
    counts.files = entries.len() as u64;
    counts.bytes = total_bytes;
    let mut excluded = plan.excluded.clone();
    excluded.extend(extra_excluded);
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
    // The copy that is decrypted again to check the file is as big as the ZIP.
    need((opts.free_space)(opts.data_dir), staged_len.saturating_add(MARGIN), "on the drive OAIY keeps its data on")?;

    state::set_phase(Phase::Encrypting, "Encrypting the backup with your passphrase");
    let tmp = dest_dir.join(format!(".{}.{}.tmp", dest.file_name().and_then(|n| n.to_str()).unwrap_or("backup"), random_id()));
    note_output(opts.data_dir, &tmp);
    let finished = (|| -> Result<u64> {
        container::encrypt_file(&zip_path, &tmp, opts.passphrase, opts.cost)?;
        #[cfg(test)]
        if DIE_AFTER_WRITING.with(|c| c.get()) {
            // A test's stand-in for the process being killed here: nothing is cleaned up.
            return Err(BackupError::new(ErrorKind::Io, DIED));
        }
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
    #[cfg(test)]
    if finished.as_ref().err().is_some_and(|e| e.message == DIED) {
        return Err(finished.unwrap_err());
    }
    forget_output(opts.data_dir);
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

/// Copy a JSON file that is cleaned on the way (see [`sanitize`]) into `to` (a new private file).
/// What was taken out is added to `excluded`, one record per value. `Err` says why the file is left out.
fn copy_cleaned(from: &Path, to: &Path, rel: &str, kind: Sanitize, excluded: &mut Vec<Excluded>) -> std::result::Result<(String, u64), String> {
    let mut bytes = Vec::new();
    File::open(from).and_then(|f| f.take(sanitize::MAX_JSON_BYTES as u64 + 1).read_to_end(&mut bytes)).map_err(|_| "it could not be read".to_string())?;
    let cleaned = match kind {
        Sanitize::Keys(name) => {
            let keys = table().key_table(name).ok_or_else(|| "OAIY does not know how to read this file".to_string())?;
            let value: serde_json::Value = serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(&bytes)).map_err(|_| "it is not valid JSON".to_string())?;
            // Only the keys the table lets come back are copied: a PIN, an address that audio goes to, a
            // switch that grants access, or anything the table does not know stays out of the file.
            let found = filter_json(keys, &value, &|_| true);
            // (A list of records leaves the same key out of each of them: it is said once.)
            let mut said = std::collections::HashSet::new();
            for left in found.left.iter().filter(|l| said.insert(l.path.clone())).take(50) {
                let (reason, redo) = match (&left.why, left.row) {
                    (Why::Excluded, Some(row)) => (row.reason.clone(), row.redo.clone()),
                    (Why::BadValue(why), _) => (format!("Left out: its value is not one a restore accepts ({why})."), None),
                    _ => ("Not recognised, so it is not backed up.".to_string(), None),
                };
                excluded.push(Excluded { pattern: format!("{rel}: {}", left.path), reason, redo });
            }
            let distinct = found.left.iter().map(|l| l.path.as_str()).collect::<std::collections::HashSet<_>>().len();
            if distinct > 50 || found.left_more > 0 {
                excluded.push(Excluded { pattern: format!("{rel}: and more"), reason: format!("{} more keys of the file were left out in the same way.", distinct.saturating_sub(50) + found.left_more), redo: None });
            }
            serde_json::to_vec_pretty(&found.value).map_err(|_| "it could not be written".to_string())?
        }
        Sanitize::None => bytes,
    };
    let mut out = secret_file::create_new_owner_only(to).map_err(|_| "it could not be staged".to_string())?;
    out.write_all(&cleaned).and_then(|_| out.sync_all()).map_err(|_| "it could not be staged".to_string())?;
    Ok((hex(&Sha256::digest(&cleaned)), cleaned.len() as u64))
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

#[cfg(test)]
thread_local! {
    /// A test makes the backup "die" after its output is written, before it is checked or renamed, on this thread only.
    pub(crate) static DIE_AFTER_WRITING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
const DIED: &str = "__died__";
