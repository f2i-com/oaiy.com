//! Restoring a backup: look, stage, apply at the next start, and undo.
//!
//! A restore is never done in place. It goes in three steps, and the first two change nothing that
//! is live:
//!
//! 1. **Look** ([`inspect`]): decrypt the file, check every item against its manifest, and say per
//!    category what restoring would add, replace and leave alone, what the backup lacks, and what
//!    the person will have to do again because credentials are never in a backup.
//! 2. **Stage** ([`stage`]): unpack the items into `<data>/restore/pending-<id>/` (every name is
//!    checked first: nothing that leaves its folder, nothing on a drive, no duplicates, nothing a
//!    backup never holds, within the size and count limits and the free space), then write the
//!    marker `<data>/restore/pending.json`. Writing the marker is what commits the step.
//! 3. **Apply** ([`apply_pending`]): at the next start, before any store is opened, each staged file
//!    is put in place with an atomic rename, the file it replaces having been moved (not copied)
//!    into `<data>/restore/undo-<id>/` first. A journal is written before every step. Any failure
//!    puts everything back and is reported on the next start. The marker is removed last.
//!
//! The undo snapshot is taken at apply time by moving, not at staging time by copying: what is
//! saved is exactly what the restore replaced, even if the app changed a file between the person
//! asking and the restart, and nothing is copied twice. The last two snapshots are kept.
//! [`stage_undo`] stages putting the newest one back (and removing what the restore added), by the
//! same three steps. A restore never touches anything a backup does not hold, and never a
//! credential: those are not in a backup, and a file that says otherwise is refused.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::manifest::Manifest;
use super::rules::{self, Category, Excluded};
use super::{agent, container, free_space, restore_dir, scratch_dir, sha256_file, BackupError, ErrorKind, Limits, Result, TempFolder, AGENT_ENTRY};
use crate::secret_file;

pub const KIND_RESTORE: &str = "restore";
pub const KIND_UNDO: &str = "undo";

/// How many undo snapshots are kept.
const KEEP_UNDO: usize = 2;
/// Room to leave on the disk beyond what a restore needs.
const MARGIN: u64 = 32 << 20;

/// What restoring does, and what the caller chooses about how it is checked.
#[derive(Clone)]
pub struct RestoreOptions {
    pub limits: Limits,
    pub free_space: fn(&Path) -> u64,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self { limits: Limits::default(), free_space }
    }
}

// ---- what the dashboard is told -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryPlan {
    pub id: String,
    pub label: String,
    pub added: u64,
    pub replaced: u64,
    pub unchanged: u64,
    pub left_alone: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    pub file_name: String,
    pub created_at: String,
    pub app_version: String,
    pub platform: String,
    pub includes_keys: bool,
    pub categories: Vec<CategoryPlan>,
    /// Categories the backup does not have.
    pub lacks: Vec<String>,
    /// What went less than fully to plan when the backup was made.
    pub partial: Vec<String>,
    /// What the backup left out on purpose.
    pub excluded: Vec<Excluded>,
    /// What to do again after restoring.
    pub redo: Vec<String>,
    pub total_files: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Staged {
    pub id: String,
    pub kind: String,
    pub files: u64,
    pub bytes: u64,
    pub agent_storage: bool,
    pub redo: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingInfo {
    pub id: String,
    pub kind: String,
    pub staged_at: String,
    pub files: u64,
    pub agent_storage: bool,
}

/// How the last restore or undo went (camelCase).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastRestore {
    pub id: String,
    pub kind: String,
    pub at: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub redo: Vec<String>,
    /// `applied`, `pending` (waiting for the Agent page), `failed` or `none`.
    pub agent_storage: String,
}

// ---- files kept in <data>/restore ------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MarkerFile {
    name: String,
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MarkerAgent {
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Marker {
    v: u32,
    id: String,
    kind: String,
    staged_at: String,
    backup_created_at: String,
    includes_keys: bool,
    /// The folder in `<data>/restore` the staged files are in: `pending-<id>` or `undo-<id>`.
    source: String,
    files: Vec<MarkerFile>,
    /// Undo: what the restore added, to be taken away again.
    #[serde(default)]
    removals: Vec<String>,
    #[serde(default)]
    agent: Option<MarkerAgent>,
    #[serde(default)]
    redo: Vec<String>,
    bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UndoRecord {
    id: String,
    applied_at: String,
    backup_created_at: String,
    replaced: Vec<String>,
    added: Vec<String>,
}

fn marker_path(data_dir: &Path) -> PathBuf {
    restore_dir(data_dir).join("pending.json")
}

fn journal_path(data_dir: &Path) -> PathBuf {
    restore_dir(data_dir).join("apply-journal.jsonl")
}

fn last_result_path(data_dir: &Path) -> PathBuf {
    restore_dir(data_dir).join("last-result.json")
}

fn is_id(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    serde_json::from_str(std::fs::read_to_string(path).ok()?.trim_start_matches('\u{feff}')).ok()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    secret_file::write(path, serde_json::to_string_pretty(value).unwrap_or_default())
}

/// The time, to the millisecond: two restores in one second still tell which is newer.
fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

impl Marker {
    /// A marker read from disk is checked before anything is done with it: the folder names are
    /// ours, and every name passes the checks a backup's names pass.
    fn check(&self, limits: &Limits) -> std::result::Result<(), String> {
        if self.v != 1 || !is_id(&self.id) {
            return Err("the record of the restore is damaged".into());
        }
        let expected_source = match self.kind.as_str() {
            KIND_RESTORE => format!("pending-{}", self.id),
            KIND_UNDO => match self.source.strip_prefix("undo-") {
                Some(uid) if is_id(uid) => self.source.clone(),
                _ => return Err("the record of the restore is damaged".into()),
            },
            _ => return Err("the record of the restore is damaged".into()),
        };
        if self.source != expected_source {
            return Err("the record of the restore is damaged".into());
        }
        for f in &self.files {
            self.check_name(&f.name, limits)?;
        }
        for r in &self.removals {
            self.check_name(r, limits)?;
        }
        Ok(())
    }

    fn check_name(&self, name: &str, limits: &Limits) -> std::result::Result<(), String> {
        container::check_entry_name(name, limits).map_err(|e| e.message)?;
        if name == AGENT_ENTRY {
            return Err("the record of the restore is damaged".into());
        }
        rules::category_of_backup_entry(name, self.includes_keys).map(|_| ())
    }
}

// ---- looking ---------------------------------------------------------------------------------------

/// The state of one place in the data folder that a restore would write to.
enum Target {
    Absent,
    File,
}

/// Look at where `rel` would go, refusing when something other than a plain file is there or on the
/// way: a link is never followed, and a folder is never replaced by a file.
fn target_state(data_dir: &Path, rel: &str) -> Result<Target> {
    let mut cur = data_dir.to_path_buf();
    let parts: Vec<&str> = rel.split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        cur.push(part);
        let meta = match std::fs::symlink_metadata(&cur) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Target::Absent),
            Err(e) => return Err(BackupError::io("Could not look at a place to restore into", &e)),
        };
        if rules::is_link(&meta) {
            return Err(BackupError::new(ErrorKind::Unsafe, "A restore will not go through a symbolic link or junction, and one is in the way."));
        }
        let last = i + 1 == parts.len();
        if last && !meta.is_file() {
            return Err(BackupError::new(ErrorKind::Conflict, format!("Something other than a file is where \"{rel}\" would be restored.")));
        }
        if !last && !meta.is_dir() {
            return Err(BackupError::new(ErrorKind::Conflict, format!("A file is in the way of where \"{rel}\" would be restored.")));
        }
        if last {
            return Ok(Target::File);
        }
    }
    Ok(Target::Absent)
}

fn redo_of(manifest: &Manifest) -> Vec<String> {
    let mut seen = HashSet::new();
    manifest.excluded.iter().filter_map(|e| e.redo.clone()).filter(|r| seen.insert(r.clone())).collect()
}

fn preview_of(data_dir: &Path, manifest: &Manifest, file_name: &str) -> Result<Preview> {
    let local = rules::plan(data_dir, manifest.includes_keys);
    let in_backup: HashSet<String> = manifest.entries.iter().map(|e| e.name.to_lowercase()).collect();
    let mut plans: HashMap<Category, CategoryPlan> = HashMap::new();
    let mut bump = |c: Category, f: &dyn Fn(&mut CategoryPlan)| {
        let p = plans.entry(c).or_insert_with(|| CategoryPlan { id: c.id().to_string(), label: c.label().to_string(), added: 0, replaced: 0, unchanged: 0, left_alone: 0 });
        f(p);
    };
    for entry in &manifest.entries {
        let (category, _) = rules::category_of_backup_entry(&entry.name, manifest.includes_keys).map_err(|why| BackupError::new(ErrorKind::Unsafe, why))?;
        if category == Category::Agent {
            // Merged by the Agent page: files in the backup overwrite files of the same name, nothing is deleted.
            let files = manifest.counts.agent_files;
            bump(category, &|p| p.added += files);
            continue;
        }
        match target_state(data_dir, &entry.name)? {
            Target::Absent => bump(category, &|p| p.added += 1),
            Target::File => {
                let target = data_dir.join(entry.name.replace('/', std::path::MAIN_SEPARATOR_STR));
                let same = std::fs::metadata(&target).map(|m| m.len() == entry.size).unwrap_or(false) && sha256_file(&target).map(|(h, _)| h == entry.sha256).unwrap_or(false);
                if same {
                    bump(category, &|p| p.unchanged += 1)
                } else {
                    bump(category, &|p| p.replaced += 1)
                }
            }
        }
    }
    for item in &local.items {
        if !in_backup.contains(&item.rel.to_lowercase()) {
            bump(item.category, &|p| p.left_alone += 1);
        }
    }
    let mut categories: Vec<CategoryPlan> = Category::ALL.iter().filter_map(|c| plans.remove(c)).collect();
    categories.retain(|c| c.added + c.replaced + c.unchanged + c.left_alone > 0);
    let has: HashSet<&str> = categories.iter().filter(|c| c.added + c.replaced + c.unchanged > 0).map(|c| c.id.as_str()).collect();
    let lacks: Vec<String> = Category::ALL.iter().filter(|c| !has.contains(c.id())).map(|c| c.label().to_string()).collect();
    Ok(Preview {
        file_name: file_name.to_string(),
        created_at: manifest.created_at.clone(),
        app_version: manifest.app.version.clone(),
        platform: manifest.platform.clone(),
        includes_keys: manifest.includes_keys,
        categories,
        lacks,
        partial: manifest.partial.clone(),
        excluded: manifest.excluded.clone(),
        redo: redo_of(manifest),
        total_files: manifest.entries.len() as u64,
        total_bytes: manifest.entries.iter().map(|e| e.size).sum(),
    })
}

/// Step 1: decrypt and check the backup, and say what restoring would do. Changes nothing.
pub fn inspect(data_dir: &Path, file: &Path, passphrase: &str, opts: &RestoreOptions) -> Result<Preview> {
    let scratch = TempFolder::new(&scratch_dir(data_dir))?;
    let verified = container::open_backup(file, passphrase, &scratch.0, &opts.limits)?;
    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    preview_of(data_dir, &verified.manifest, &name)
}

// ---- staging ---------------------------------------------------------------------------------------

/// Step 2: unpack the backup into `<data>/restore/pending-<id>/` and write the marker. Nothing
/// live is changed.
pub fn stage(data_dir: &Path, file: &Path, passphrase: &str, opts: &RestoreOptions) -> Result<Staged> {
    let scratch = TempFolder::new(&scratch_dir(data_dir))?;
    let verified = container::open_backup(file, passphrase, &scratch.0, &opts.limits)?;
    let manifest = &verified.manifest;
    let total: u64 = manifest.entries.iter().map(|e| e.size).sum();
    let plain_len = std::fs::metadata(&verified.plain).map(|m| m.len()).unwrap_or(0);
    let needed = total.saturating_add(plain_len).saturating_add(MARGIN);
    let free = (opts.free_space)(data_dir);
    if free < needed {
        return Err(BackupError::new(
            ErrorKind::NoSpace,
            format!("There is not enough free space to prepare this restore: about {} MB is needed and {} MB is free.", needed >> 20, free >> 20),
        ));
    }
    // Everything it would replace must be a plain file, and nothing may be behind a link.
    for entry in &manifest.entries {
        if entry.name != AGENT_ENTRY {
            target_state(data_dir, &entry.name)?;
        }
    }

    discard_pending(data_dir)?;
    let id = super::random_id();
    let source = format!("pending-{id}");
    let root = restore_dir(data_dir).join(&source);
    let staged = (|| -> Result<Marker> {
        secret_file::create_private_dir(&root.join("files")).map_err(|e| BackupError::io("Could not make a folder for the restore", &e))?;
        let limits = &opts.limits;
        container::extract_all(&verified, |entry| {
            if entry.name == AGENT_ENTRY {
                Ok(root.join("agent-storage.zip"))
            } else {
                container::safe_join(&root.join("files"), &entry.name, limits)
            }
        })?;
        let agent = manifest.entries.iter().find(|e| e.name == AGENT_ENTRY).map(|e| MarkerAgent { size: e.size, sha256: e.sha256.clone() });
        Ok(Marker {
            v: 1,
            id: id.clone(),
            kind: KIND_RESTORE.to_string(),
            staged_at: now(),
            backup_created_at: manifest.created_at.clone(),
            includes_keys: manifest.includes_keys,
            source: source.clone(),
            files: manifest.entries.iter().filter(|e| e.name != AGENT_ENTRY).map(|e| MarkerFile { name: e.name.clone(), size: e.size, sha256: e.sha256.clone() }).collect(),
            removals: Vec::new(),
            agent,
            redo: redo_of(manifest),
            bytes: total,
        })
    })();
    let marker = match staged {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&root);
            return Err(e);
        }
    };
    // The marker is what commits the step.
    if let Err(e) = write_json(&marker_path(data_dir), &marker) {
        let _ = std::fs::remove_dir_all(&root);
        return Err(BackupError::io("Could not record the restore", &e));
    }
    log::info!("backup: a restore ({} files) is staged and will be applied at the next start", marker.files.len());
    Ok(Staged {
        id,
        kind: KIND_RESTORE.to_string(),
        files: marker.files.len() as u64,
        bytes: total,
        agent_storage: marker.agent.is_some(),
        redo: marker.redo,
    })
}

/// Cancel a staged restore: the marker and, for a restore, its staged files. An undo snapshot is not touched.
pub fn discard_pending(data_dir: &Path) -> Result<()> {
    let path = marker_path(data_dir);
    if let Some(marker) = read_json::<Marker>(&path) {
        if marker.kind == KIND_RESTORE && marker.source == format!("pending-{}", marker.id) && is_id(&marker.id) {
            let _ = std::fs::remove_dir_all(restore_dir(data_dir).join(&marker.source));
        }
    }
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(BackupError::io("Could not cancel the restore", &e)),
    }
}

// ---- undo ------------------------------------------------------------------------------------------

fn undo_dirs(data_dir: &Path) -> Vec<(String, UndoRecord)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(restore_dir(data_dir)) else { return out };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_prefix("undo-") else { continue };
        if !is_id(id) {
            continue;
        }
        if let Some(record) = read_json::<UndoRecord>(&entry.path().join("undo.json")) {
            if record.id == id {
                out.push((id.to_string(), record));
            }
        }
    }
    out.sort_by(|a, b| b.1.applied_at.cmp(&a.1.applied_at).then(b.0.cmp(&a.0)));
    out
}

pub fn undo_available(data_dir: &Path) -> bool {
    !undo_dirs(data_dir).is_empty()
}

/// Stage putting back what the last restore replaced (and taking away what it added), to be applied at the next start.
pub fn stage_undo(data_dir: &Path, opts: &RestoreOptions) -> Result<Staged> {
    let Some((uid, record)) = undo_dirs(data_dir).into_iter().next() else {
        return Err(BackupError::new(ErrorKind::Conflict, "There is no restore to undo."));
    };
    discard_pending(data_dir)?;
    let root = restore_dir(data_dir).join(format!("undo-{uid}"));
    let mut files = Vec::new();
    let mut bytes = 0u64;
    for rel in &record.replaced {
        container::check_entry_name(rel, &opts.limits)?;
        let path = container::safe_join(&root.join("files"), rel, &opts.limits)?;
        let meta = std::fs::symlink_metadata(&path).map_err(|_| BackupError::new(ErrorKind::Damaged, "The saved copy of what the restore replaced is incomplete, so it cannot be put back."))?;
        if !meta.is_file() {
            return Err(BackupError::new(ErrorKind::Damaged, "The saved copy of what the restore replaced is damaged, so it cannot be put back."));
        }
        let (sha256, size) = sha256_file(&path).map_err(|e| BackupError::io("Could not read the saved copy", &e))?;
        bytes += size;
        files.push(MarkerFile { name: rel.clone(), size, sha256 });
        target_state(data_dir, rel)?;
    }
    for rel in &record.added {
        container::check_entry_name(rel, &opts.limits)?;
        target_state(data_dir, rel)?;
    }
    let agent_zip = root.join("agent-storage.zip");
    let agent = match std::fs::metadata(&agent_zip) {
        Ok(m) if m.is_file() => {
            let (sha256, size) = sha256_file(&agent_zip).map_err(|e| BackupError::io("Could not read the saved copy", &e))?;
            Some(MarkerAgent { size, sha256 })
        }
        _ => None,
    };
    let marker = Marker {
        v: 1,
        id: super::random_id(),
        kind: KIND_UNDO.to_string(),
        staged_at: now(),
        backup_created_at: record.backup_created_at.clone(),
        // Names put back were accepted the first time; a provider key file is among them only if it was in the backup.
        includes_keys: record.replaced.iter().chain(record.added.iter()).any(|r| r.eq_ignore_ascii_case("ai/providers.json")),
        source: format!("undo-{uid}"),
        files,
        removals: record.added.clone(),
        agent,
        redo: Vec::new(),
        bytes,
    };
    write_json(&marker_path(data_dir), &marker).map_err(|e| BackupError::io("Could not record the undo", &e))?;
    Ok(Staged {
        id: marker.id.clone(),
        kind: KIND_UNDO.to_string(),
        files: (marker.files.len() + marker.removals.len()) as u64,
        bytes,
        agent_storage: marker.agent.is_some(),
        redo: Vec::new(),
    })
}

// ---- applying --------------------------------------------------------------------------------------

/// What happened when the app started with a restore waiting.
#[derive(Debug)]
pub enum ApplyOutcome {
    /// Nothing was waiting.
    None,
    Applied(LastRestore),
    /// It failed and everything was put back.
    Failed(LastRestore),
}

/// One line of the journal, written before its step.
#[derive(Serialize, Deserialize)]
struct JournalLine {
    op: String,
    #[serde(default)]
    rel: String,
    #[serde(default)]
    had: bool,
}

struct Journal {
    file: std::fs::File,
}

impl Journal {
    fn create(path: &Path) -> std::io::Result<Self> {
        let _ = std::fs::remove_file(path);
        Ok(Self { file: secret_file::create_new_owner_only(path)? })
    }

    fn line(&mut self, op: &str, rel: &str, had: bool) -> std::io::Result<()> {
        let mut text = serde_json::to_string(&JournalLine { op: op.to_string(), rel: rel.to_string(), had }).unwrap_or_default();
        text.push('\n');
        self.file.write_all(text.as_bytes())?;
        self.file.sync_data()
    }
}

fn read_journal(path: &Path) -> Vec<JournalLine> {
    std::fs::read_to_string(path).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
}

fn native(rel: &str) -> PathBuf {
    rel.split('/').collect()
}

/// Put back what the journal says was begun, newest first. Safe to run more than once, and after a crash at any point.
fn roll_back(data_dir: &Path, staged_root: &Path, holding: &Path, lines: &[JournalLine]) {
    for line in lines.iter().rev().filter(|l| l.op == "begin") {
        let target = data_dir.join(native(&line.rel));
        let kept = holding.join(native(&line.rel));
        if line.had {
            // The original is in the holding folder if it was moved: it goes back over whatever is there now.
            if kept.exists() {
                let _ = secret_file::rename_over(&kept, &target);
            }
        } else if !staged_root.join(native(&line.rel)).exists() && target.is_file() {
            // Installed (the staged file is gone) but there was nothing before: take it away again.
            let _ = std::fs::remove_file(&target);
        }
    }
}

fn holding_of(data_dir: &Path, marker: &Marker) -> PathBuf {
    match marker.kind.as_str() {
        KIND_RESTORE => restore_dir(data_dir).join(format!("undo-{}", marker.id)).join("files"),
        _ => restore_dir(data_dir).join(format!("undone-{}", marker.id)).join("files"),
    }
}

fn record_last(data_dir: &Path, last: &LastRestore) {
    if let Err(e) = write_json(&last_result_path(data_dir), last) {
        log::warn!("backup: could not record how the restore went: {e}");
    }
}

fn failed(marker_id: &str, kind: &str, why: &str) -> LastRestore {
    LastRestore { id: marker_id.to_string(), kind: kind.to_string(), at: now(), ok: false, error: Some(why.to_string()), redo: Vec::new(), agent_storage: "none".into() }
}

/// Step 3: called at the very start of the app, before any store is opened. If a restore is
/// staged, put it in place; otherwise do nothing.
pub fn apply_pending(data_dir: &Path) -> ApplyOutcome {
    let dir = restore_dir(data_dir);
    let marker_file = marker_path(data_dir);
    if std::fs::symlink_metadata(&marker_file).is_err() {
        return ApplyOutcome::None;
    }
    let limits = Limits::default();
    let Some(marker) = read_json::<Marker>(&marker_file) else {
        let _ = std::fs::rename(&marker_file, dir.join(format!("bad-marker-{}.json", super::random_id())));
        let last = failed("unknown", KIND_RESTORE, "The record of the restore was damaged, so nothing was changed.");
        record_last(data_dir, &last);
        return ApplyOutcome::Failed(last);
    };
    if let Err(why) = marker.check(&limits) {
        // What was staged is personal data: it goes with the record that named it.
        if is_id(&marker.id) && marker.kind == KIND_RESTORE {
            let _ = std::fs::remove_dir_all(dir.join(format!("pending-{}", marker.id)));
        }
        let _ = std::fs::rename(&marker_file, dir.join(format!("bad-marker-{}.json", super::random_id())));
        let last = failed("unknown", &marker.kind, &format!("The restore was refused, so nothing was changed: {why}."));
        record_last(data_dir, &last);
        return ApplyOutcome::Failed(last);
    }
    let staged_root = dir.join(&marker.source).join("files");
    let holding = holding_of(data_dir, &marker);
    let journal_file = journal_path(data_dir);

    // An earlier start began this and did not finish.
    if std::fs::symlink_metadata(&journal_file).is_ok() {
        let lines = read_journal(&journal_file);
        if lines.last().map(|l| l.op == "done").unwrap_or(false) {
            let applied: Vec<(String, bool)> = lines.iter().filter(|l| l.op == "begin").map(|l| (l.rel.clone(), l.had)).collect();
            return finalize(data_dir, &marker, &applied);
        }
        roll_back(data_dir, &staged_root, &holding, &lines);
        let last = failed(&marker.id, &marker.kind, "The restore was interrupted part-way, so everything it had changed was put back.");
        return conclude_failed(data_dir, &marker, last);
    }

    match apply_files(data_dir, &marker, &staged_root, &holding, &journal_file, &limits) {
        Ok(applied) => finalize(data_dir, &marker, &applied),
        // A test's stand-in for the process dying: nothing is rolled back and nothing is cleaned up.
        #[cfg(test)]
        Err(why) if why == CRASH => ApplyOutcome::None,
        Err(why) => {
            let lines = read_journal(&journal_file);
            roll_back(data_dir, &staged_root, &holding, &lines);
            let last = failed(&marker.id, &marker.kind, &format!("{why} Everything it had changed was put back."));
            conclude_failed(data_dir, &marker, last)
        }
    }
}

fn conclude_failed(data_dir: &Path, marker: &Marker, last: LastRestore) -> ApplyOutcome {
    let dir = restore_dir(data_dir);
    record_last(data_dir, &last);
    // What was staged is personal data: it does not stay behind after a failure. The folder that
    // held what was set aside goes only if the rollback emptied it: a file still in it is the only
    // copy of something, and stays.
    let holding_root = if marker.kind == KIND_RESTORE { dir.join(format!("undo-{}", marker.id)) } else { dir.join(format!("undone-{}", marker.id)) };
    if marker.kind == KIND_RESTORE {
        let _ = std::fs::remove_dir_all(dir.join(&marker.source));
    }
    if !remove_empty_tree(&holding_root) {
        log::warn!("backup: something set aside by the failed restore could not be put back and was kept in {}", holding_root.display());
    }
    let _ = std::fs::remove_file(journal_path(data_dir));
    let _ = std::fs::remove_file(marker_path(data_dir));
    log::warn!("backup: a restore did not finish and was rolled back");
    ApplyOutcome::Failed(last)
}

/// Remove `path`, and the folders under it, if no file is left in them; anything that holds a file stays.
/// True when `path` is gone.
fn remove_empty_tree(path: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(path) else { return true };
    let mut empty = true;
    for entry in entries.flatten() {
        match entry.file_type() {
            Ok(t) if t.is_dir() => empty &= remove_empty_tree(&entry.path()),
            _ => empty = false,
        }
    }
    empty && std::fs::remove_dir(path).is_ok()
}

/// Verify the staged files, then put each in place: the file it replaces (if any) is moved into the
/// holding folder first, and a line is journalled before each step.
fn apply_files(data_dir: &Path, marker: &Marker, staged_root: &Path, holding: &Path, journal_file: &Path, limits: &Limits) -> std::result::Result<Vec<(String, bool)>, String> {
    // What is staged is what was staged: every file is there, plain, and has its hash.
    for f in &marker.files {
        let path = container::safe_join(staged_root, &f.name, limits).map_err(|e| e.message)?;
        let meta = std::fs::symlink_metadata(&path).map_err(|_| "A staged file is missing, so nothing was changed.".to_string())?;
        if !meta.is_file() || rules::is_link(&meta) || meta.len() != f.size {
            return Err("A staged file is not what was staged, so nothing was changed.".into());
        }
        let (sha, _) = sha256_file(&path).map_err(|_| "A staged file could not be read, so nothing was changed.".to_string())?;
        if sha != f.sha256 {
            return Err("A staged file does not check out, so nothing was changed.".into());
        }
    }
    let mut journal = Journal::create(journal_file).map_err(|e| format!("Could not start the restore's journal ({e})."))?;
    let mut applied = Vec::new();
    for (index, f) in marker.files.iter().enumerate() {
        let staged = container::safe_join(staged_root, &f.name, limits).map_err(|e| e.message)?;
        let target = data_dir.join(native(&f.name));
        let had = match target_state(data_dir, &f.name).map_err(|e| e.message)? {
            Target::Absent => false,
            Target::File => true,
        };
        journal.line("begin", &f.name, had).map_err(|e| format!("Could not write the restore's journal ({e})."))?;
        if had {
            let kept = holding.join(native(&f.name));
            if let Some(parent) = kept.parent() {
                secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder for the saved copy ({e})."))?;
            }
            secret_file::rename_over(&target, &kept).map_err(|e| format!("Could not set aside a file it replaces ({e})."))?;
        }
        #[cfg(test)]
        inject_at(index)?;
        if let Some(parent) = target.parent() {
            secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder ({e})."))?;
        }
        secret_file::rename_over(&staged, &target).map_err(|e| format!("Could not put a restored file in place ({e})."))?;
        applied.push((f.name.clone(), had));
    }
    for rel in &marker.removals {
        let target = data_dir.join(native(rel));
        if !matches!(target_state(data_dir, rel).map_err(|e| e.message)?, Target::File) {
            continue;
        }
        journal.line("begin", rel, true).map_err(|e| format!("Could not write the restore's journal ({e})."))?;
        let kept = holding.join(native(rel));
        if let Some(parent) = kept.parent() {
            secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder for the saved copy ({e})."))?;
        }
        secret_file::rename_over(&target, &kept).map_err(|e| format!("Could not take away a file the restore had added ({e})."))?;
        applied.push((rel.clone(), true));
    }
    journal.line("done", "", false).map_err(|e| format!("Could not finish the restore's journal ({e})."))?;
    #[cfg(test)]
    if INJECT.with(|c| c.get()) == Some(Inject::CrashAfterDone) {
        return Err(CRASH.to_string());
    }
    Ok(applied)
}

/// Where a test makes an apply fail or the process "die", on this thread only.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Inject {
    /// Fail after the file at this index was set aside and before its replacement goes in.
    FailBeforeInstall(usize),
    /// Die after the file at this index was set aside (nothing rolled back, nothing cleaned up).
    CrashBeforeInstall(usize),
    /// Die after the journal is complete, before the marker is removed.
    CrashAfterDone,
}

#[cfg(test)]
thread_local! {
    pub(crate) static INJECT: std::cell::Cell<Option<Inject>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
const CRASH: &str = "__crash__";

#[cfg(test)]
fn inject_at(index: usize) -> std::result::Result<(), String> {
    match INJECT.with(|c| c.get()) {
        Some(Inject::FailBeforeInstall(k)) if k == index => Err("An injected failure.".to_string()),
        Some(Inject::CrashBeforeInstall(k)) if k == index => Err(CRASH.to_string()),
        _ => Ok(()),
    }
}

/// The files are in place: keep the undo record, hand the Agent's storage to its page, record how it
/// went, clean up, and remove the marker last.
fn finalize(data_dir: &Path, marker: &Marker, applied: &[(String, bool)]) -> ApplyOutcome {
    let dir = restore_dir(data_dir);
    let source = dir.join(&marker.source);
    if marker.kind == KIND_RESTORE {
        let record = UndoRecord {
            id: marker.id.clone(),
            applied_at: now(),
            backup_created_at: marker.backup_created_at.clone(),
            replaced: applied.iter().filter(|(_, had)| *had).map(|(r, _)| r.clone()).collect(),
            added: applied.iter().filter(|(_, had)| !*had).map(|(r, _)| r.clone()).collect(),
        };
        let undo_root = dir.join(format!("undo-{}", marker.id));
        if secret_file::create_private_dir(&undo_root).is_ok() {
            if let Err(e) = write_json(&undo_root.join("undo.json"), &record) {
                log::warn!("backup: could not record what the restore replaced: {e}");
            }
        }
    }
    let mut agent_storage = "none";
    if marker.agent.is_some() {
        let zip = source.join("agent-storage.zip");
        match agent::leave_for_page(data_dir, &marker.id, &marker.kind, &zip) {
            Ok(()) => agent_storage = "pending",
            Err(e) => {
                log::warn!("backup: the Agent's storage could not be handed over: {e}");
                agent_storage = "failed";
            }
        }
    }
    let last = LastRestore {
        id: marker.id.clone(),
        kind: marker.kind.clone(),
        at: now(),
        ok: true,
        error: None,
        redo: marker.redo.clone(),
        agent_storage: agent_storage.to_string(),
    };
    record_last(data_dir, &last);
    if marker.kind == KIND_RESTORE {
        let _ = std::fs::remove_dir_all(&source);
        prune_undo(data_dir);
    } else {
        // An undo uses up the snapshot it put back.
        let _ = std::fs::remove_dir_all(&source);
        let _ = std::fs::remove_dir_all(dir.join(format!("undone-{}", marker.id)));
    }
    let _ = std::fs::remove_file(journal_path(data_dir));
    let _ = std::fs::remove_file(marker_path(data_dir));
    log::info!("backup: a {} was applied ({} files)", marker.kind, applied.len());
    ApplyOutcome::Applied(last)
}

/// Keep the newest [`KEEP_UNDO`] undo snapshots and delete the rest.
fn prune_undo(data_dir: &Path) {
    for (id, _) in undo_dirs(data_dir).into_iter().skip(KEEP_UNDO) {
        let _ = std::fs::remove_dir_all(restore_dir(data_dir).join(format!("undo-{id}")));
    }
}

// ---- what the status route reads -------------------------------------------------------------------

pub fn pending_info(data_dir: &Path) -> Option<PendingInfo> {
    let marker = read_json::<Marker>(&marker_path(data_dir))?;
    Some(PendingInfo {
        id: marker.id,
        kind: marker.kind,
        staged_at: marker.staged_at,
        files: (marker.files.len() + marker.removals.len()) as u64,
        agent_storage: marker.agent.is_some(),
    })
}

pub fn last_restore(data_dir: &Path) -> Option<LastRestore> {
    let mut last = read_json::<LastRestore>(&last_result_path(data_dir))?;
    // The Agent's storage waits for its page: say so only while it does.
    if last.agent_storage == "pending" && agent::read_pending_import(data_dir).map(|p| p.id) != Some(last.id.clone()) {
        last.agent_storage = "applied".to_string();
    }
    Some(last)
}

/// The Agent page said how its part of a restore went.
pub(crate) fn record_agent_result(data_dir: &Path, id: &str, ok: bool, error: Option<&str>) {
    let Some(mut last) = read_json::<LastRestore>(&last_result_path(data_dir)) else { return };
    if last.id != id {
        return;
    }
    if ok {
        last.agent_storage = "applied".into();
    } else {
        last.agent_storage = "failed".into();
        // The reason goes in the plain: the person may need to try again.
        last.redo.push(format!("Restore again once the Agent is open: its conversations and projects could not be brought back ({}).", error.unwrap_or("no reason given")));
    }
    record_last(data_dir, &last);
}
