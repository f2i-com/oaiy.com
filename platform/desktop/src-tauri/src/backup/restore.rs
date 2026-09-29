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

use super::manifest::{Entry, Manifest};
use super::review::{self, ClassInfo, Local, RestoreClass, ReviewItem, Ticks};
use super::rules::{self, Category, Excluded};
use super::{agent, container, free_space, restore_dir, Budget, scratch_dir, sha256_file, BackupError, ErrorKind, Limits, Result, TempFolder, AGENT_ENTRY};
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
    /// How long looking at or staging a backup may take.
    pub time_limit: std::time::Duration,
    /// What is going on in the app now: looking at a backup asks for a second of computing and up to
    /// a gigabyte of memory, and is not done while a call is live.
    pub busy: super::busy::BusySignals,
    /// The largest Agent storage that is left for its page (which takes no more than this).
    pub agent_import_max: u64,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self { limits: Limits::default(), free_space, time_limit: super::RESTORE_TIME_LIMIT, busy: super::busy::BusySignals::default(), agent_import_max: agent::IMPORT_MAX }
    }
}

impl RestoreOptions {
    fn budget(&self) -> Budget {
        Budget::within(self.time_limit)
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
    /// The classes of things that can act, each to be ticked on its own (only those the backup has).
    pub classes: Vec<ClassInfo>,
    /// Every item of those classes, by name and by what it does.
    pub items: Vec<ReviewItem>,
    pub keys: KeysInfo,
    /// What is said about what will be left out or cleaned on the way in.
    pub notes: Vec<String>,
}

/// Whether the backup holds API keys (what its record says; the person decides whether they come back).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeysInfo {
    pub in_backup: bool,
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
    /// What was left out, cleaned or dropped, in plain words.
    pub skipped: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingInfo {
    pub id: String,
    pub kind: String,
    pub staged_at: String,
    pub files: u64,
    pub agent_storage: bool,
    /// The classes that were ticked.
    pub classes: Vec<String>,
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
    /// What was left out, cleaned or dropped on the way in.
    #[serde(default)]
    pub notes: Vec<String>,
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
    /// The Agent's own settings were ticked (or, in an undo, they are the person's own).
    #[serde(default)]
    apply_settings: bool,
    /// The API keys were ticked.
    #[serde(default)]
    apply_keys: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Marker {
    v: u32,
    id: String,
    kind: String,
    staged_at: String,
    backup_created_at: String,
    /// The classes that were ticked, and whether the keys were.
    #[serde(default)]
    ticked: Vec<String>,
    #[serde(default)]
    keys: bool,
    /// What was left out, cleaned or dropped on the way in.
    #[serde(default)]
    notes: Vec<String>,
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
    /// What made this snapshot: a `restore` (it holds what the restore replaced) or an `undo` (it holds
    /// what the undo replaced and took away: the redo).
    #[serde(default = "restore_kind")]
    kind: String,
    replaced: Vec<String>,
    added: Vec<String>,
}

fn restore_kind() -> String {
    KIND_RESTORE.to_string()
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
        rules::category_of_backup_entry(name).map(|_| ())
    }
}

// ---- looking ---------------------------------------------------------------------------------------

/// The state of one place in the data folder that a restore would write to.
enum Target {
    Absent,
    File,
}

/// Looks at where files would be restored to, remembering each folder's listing.
struct Targets {
    listed: HashMap<PathBuf, Option<HashSet<String>>>,
    /// Whether a folder's listing is kept: not while files are being put in place, which changes them.
    keep: bool,
}

impl Targets {
    /// For looking, when nothing changes.
    fn new() -> Self {
        Self { listed: HashMap::new(), keep: true }
    }

    /// For use while files are being moved: every look lists the folder again.
    fn fresh() -> Self {
        Self { listed: HashMap::new(), keep: false }
    }

    fn names_in(&mut self, dir: &Path) -> Option<&HashSet<String>> {
        if !self.keep {
            self.listed.remove(dir);
        }
        self.listed
            .entry(dir.to_path_buf())
            .or_insert_with(|| std::fs::read_dir(dir).ok().map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().to_lowercase()).collect()))
            .as_ref()
    }

    /// Look at where `rel` would go, refusing when something other than a plain file is there or on
    /// the way: a link is never followed, a folder is never replaced by a file, and a name that
    /// resolves but is not the name of anything in its folder (an NTFS short name such as
    /// `PAIRIN~1.JSO`) is an alias for a file the checks on names never saw.
    fn state(&mut self, data_dir: &Path, rel: &str) -> Result<Target> {
        let mut cur = data_dir.to_path_buf();
        let parts: Vec<&str> = rel.split('/').collect();
        for (i, part) in parts.iter().enumerate() {
            let parent = cur.clone();
            cur.push(part);
            let meta = match std::fs::symlink_metadata(&cur) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Target::Absent),
                Err(e) => return Err(BackupError::io("Could not look at a place to restore into", &e)),
            };
            if self.names_in(&parent).is_some_and(|names| !names.contains(&part.to_lowercase())) {
                return Err(BackupError::new(ErrorKind::Unsafe, format!("\"{part}\" is a short-name alias for another file, and a restore will not write through one.")));
            }
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
}

/// A test's way to ask whether a place could be restored to.
#[cfg(test)]
pub(crate) fn check_target(data_dir: &Path, rel: &str) -> Result<()> {
    Targets::new().state(data_dir, rel).map(|_| ())
}

fn redo_of(manifest: &Manifest) -> Vec<String> {
    let mut seen = HashSet::new();
    manifest.excluded.iter().filter_map(|e| e.redo.clone()).filter(|r| seen.insert(r.clone())).collect()
}

/// The most of an item's own file that is read to describe it.
const MAX_REVIEW_BYTES: u64 = 2 << 20;

/// Cut a hostile string to what a panel can show, and a list to a length.
fn clipped(lines: &[String], count: usize, each: usize) -> Vec<String> {
    lines.iter().take(count).map(|l| review::clip(l, each)).collect()
}

fn preview_of(data_dir: &Path, verified: &container::Verified, scratch: &Path, file_name: &str, budget: &Budget) -> Result<Preview> {
    let manifest = &verified.manifest;
    let local = rules::plan(data_dir, true);
    let in_backup: HashSet<String> = manifest.entries.iter().map(|e| e.name.to_lowercase()).collect();
    let mut targets = Targets::new();
    let mut plans: HashMap<Category, CategoryPlan> = HashMap::new();
    let mut bump = |c: Category, f: &dyn Fn(&mut CategoryPlan)| {
        let p = plans.entry(c).or_insert_with(|| CategoryPlan { id: c.id().to_string(), label: c.label().to_string(), added: 0, replaced: 0, unchanged: 0, left_alone: 0 });
        f(p);
    };
    for entry in &manifest.entries {
        let (category, _) = rules::category_of_backup_entry(&entry.name).map_err(|why| BackupError::new(ErrorKind::Unsafe, why))?;
        if category == Category::Agent {
            // Merged by the Agent page: files in the backup overwrite files of the same name, nothing is deleted.
            let files = manifest.counts.agent_files;
            bump(category, &|p| p.added += files);
            continue;
        }
        match targets.state(data_dir, &entry.name)? {
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

    // ---- everything that can act, by name and by what it does ----
    let mut archive = container::open_archive(&verified.plain)?;
    let here = Local::read(data_dir);
    let mut backup_templates: HashSet<String> = HashSet::new();
    let mut read: Vec<(Category, &str, Option<Vec<u8>>)> = Vec::new();
    for entry in &manifest.entries {
        budget.check()?;
        let Ok((category, _)) = rules::category_of_backup_entry(&entry.name) else { continue };
        if review::class_of(category).is_none() {
            continue;
        }
        let bytes = container::read_entry(&mut archive, &entry.name, MAX_REVIEW_BYTES)?;
        if entry.name.starts_with("templates/") {
            if let Some(id) = bytes.as_ref().and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok()).and_then(|v| v.get("id").and_then(|i| i.as_str().map(str::to_string))) {
                backup_templates.insert(id);
            }
        }
        read.push((category, entry.name.as_str(), bytes));
    }
    let mut items: Vec<ReviewItem> = Vec::new();
    for (category, name, bytes) in &read {
        match bytes {
            Some(bytes) => items.extend(review::describe(*category, name, bytes, &here, &backup_templates)),
            None => {
                if let Some(class) = review::class_of(*category) {
                    items.push(ReviewItem { class, name: (*name).to_string(), title: review::clip(name.rsplit('/').next().unwrap_or(name), 120), what: "Too large to look at: it is not brought back.".to_string() });
                }
            }
        }
        if items.len() > review::MAX_REVIEW_ITEMS {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup holds more things that can run or reconfigure OAIY than can be looked through, so it is refused."));
        }
    }
    if manifest.entries.iter().any(|e| e.name == AGENT_ENTRY) {
        let settings = container::read_nested_entry(&mut archive, AGENT_ENTRY, "idb/settings.json", scratch, MAX_REVIEW_BYTES, budget)?;
        if let Some(settings) = settings {
            items.extend(review::describe_agent_settings(&settings));
        }
    }
    if items.len() > review::MAX_REVIEW_ITEMS {
        return Err(BackupError::new(ErrorKind::TooLarge, "This backup holds more things that can run or reconfigure OAIY than can be looked through, so it is refused."));
    }
    let classes: Vec<ClassInfo> = RestoreClass::ALL
        .iter()
        .filter_map(|c| {
            let count = items.iter().filter(|i| i.class == *c).count();
            (count > 0).then(|| ClassInfo { id: c.id().to_string(), label: c.label().to_string(), description: c.description().to_string(), count })
        })
        .collect();
    let mut notes = Vec::new();
    if in_backup.contains("calendar/calendar.json") {
        notes.push("The calendar comes back without its FormLogic sync state: it pairs and syncs again when you link FormLogic.".to_string());
    }
    if items.iter().any(|i| i.class == RestoreClass::Plugins) {
        notes.push("Plugin settings come back without PINs, keys or values sealed to another computer; what this computer already has of those stays.".to_string());
    }
    if items.iter().any(|i| i.class == RestoreClass::Flows && i.name == "bridge/ledger.jsonl") {
        notes.push("Runs that were waiting or running when the backup was made are never brought back.".to_string());
    }
    Ok(Preview {
        file_name: review::clip(file_name, 200),
        created_at: review::clip(&manifest.created_at, 40),
        app_version: review::clip(&manifest.app.version, 64),
        platform: review::clip(&manifest.platform, 32),
        includes_keys: manifest.includes_keys,
        categories,
        lacks,
        partial: clipped(&manifest.partial, 50, 400),
        excluded: manifest.excluded.iter().take(300).map(|e| Excluded { pattern: review::clip(&e.pattern, 200), reason: review::clip(&e.reason, 400), redo: e.redo.as_ref().map(|r| review::clip(r, 400)) }).collect(),
        redo: clipped(&redo_of(manifest), 50, 400),
        total_files: manifest.entries.len() as u64,
        total_bytes: manifest.entries.iter().map(|e| e.size).sum(),
        classes,
        items,
        keys: KeysInfo { in_backup: manifest.includes_keys },
        notes,
    })
}

/// Step 1: decrypt and check the backup, and say what restoring would do. Changes nothing.
pub fn inspect(data_dir: &Path, file: &Path, passphrase: &str, opts: &RestoreOptions) -> Result<Preview> {
    opts.busy.refuse_if_busy("checking a backup")?;
    let scratch = TempFolder::new(&scratch_dir(data_dir))?;
    let budget = opts.budget();
    let verified = container::open_backup(file, passphrase, &scratch.0, &opts.limits, &budget)?;
    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    preview_of(data_dir, &verified, &scratch.0, &name, &budget)
}

// ---- staging ---------------------------------------------------------------------------------------

/// Step 2: unpack the backup into `<data>/restore/pending-<id>/` and write the marker. Nothing live
/// is changed. Data comes back as it is; a class that can run things or change settings comes back
/// only if `ticks` has it (see [`review`]).
pub fn stage(data_dir: &Path, file: &Path, passphrase: &str, ticks: &Ticks, opts: &RestoreOptions) -> Result<Staged> {
    opts.busy.refuse_if_busy("preparing a restore")?;
    let budget = opts.budget();
    let scratch = TempFolder::new(&scratch_dir(data_dir))?;
    let verified = container::open_backup(file, passphrase, &scratch.0, &opts.limits, &budget)?;
    let manifest = &verified.manifest;
    let mut skipped: Vec<String> = Vec::new();
    // What is brought back: data, and the classes that were ticked (and the Agent's storage, if it is small enough for its page).
    let mut selected: Vec<&Entry> = Vec::new();
    let mut left_out: HashMap<RestoreClass, usize> = HashMap::new();
    for entry in &manifest.entries {
        if entry.name == AGENT_ENTRY {
            if entry.size > opts.agent_import_max {
                skipped.push(format!("The Agent's conversations and projects were not brought back: they are {} MB, more than the {} MB its page takes back.", entry.size >> 20, opts.agent_import_max >> 20));
            } else {
                selected.push(entry);
            }
            continue;
        }
        let Ok((category, _)) = rules::category_of_backup_entry(&entry.name) else { continue };
        match review::class_of(category) {
            Some(class) if !ticks.has(class) => *left_out.entry(class).or_default() += 1,
            _ => selected.push(entry),
        }
    }
    for class in RestoreClass::ALL {
        if let Some(n) = left_out.get(&class) {
            skipped.push(format!("Not brought back (not ticked): {} ({n} file{}).", class.label(), if *n == 1 { "" } else { "s" }));
        }
    }
    let total: u64 = selected.iter().map(|e| e.size).sum();
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
    let mut targets = Targets::new();
    for entry in &selected {
        if entry.name != AGENT_ENTRY {
            targets.state(data_dir, &entry.name)?;
        }
    }

    discard_pending(data_dir)?;
    let id = super::random_id();
    let source = format!("pending-{id}");
    let root = restore_dir(data_dir).join(&source);
    let wanted: HashSet<&str> = selected.iter().map(|e| e.name.as_str()).collect();
    let staged = (|| -> Result<(Marker, Vec<String>)> {
        secret_file::create_private_dir(&root.join("files")).map_err(|e| BackupError::io("Could not make a folder for the restore", &e))?;
        let limits = &opts.limits;
        container::extract_all(&verified, |entry| {
            if !wanted.contains(entry.name.as_str()) {
                Ok(None)
            } else if entry.name == AGENT_ENTRY {
                Ok(Some(root.join("agent-storage.zip")))
            } else {
                container::safe_join(&root.join("files"), &entry.name, limits).map(Some)
            }
        }, &budget)?;
        let agent = selected.iter().find(|e| e.name == AGENT_ENTRY).map(|e| MarkerAgent { size: e.size, sha256: e.sha256.clone(), apply_settings: ticks.has(RestoreClass::AgentSettings), apply_keys: ticks.keys });
        // What can carry more than data is cleaned as it comes in, and the marker records what is there now.
        let names: Vec<String> = selected.iter().filter(|e| e.name != AGENT_ENTRY).map(|e| e.name.clone()).collect();
        let (kept, notes) = clean_staged(data_dir, &root.join("files"), &names, ticks, limits)?;
        let mut files = Vec::new();
        for name in &kept {
            let path = container::safe_join(&root.join("files"), name, limits)?;
            let (sha256, size) = sha256_file(&path).map_err(|e| BackupError::io("Could not read a staged file", &e))?;
            files.push(MarkerFile { name: name.clone(), size, sha256 });
        }
        let bytes: u64 = files.iter().map(|f| f.size).sum();
        Ok((
            Marker {
                v: 1,
                id: id.clone(),
                kind: KIND_RESTORE.to_string(),
                staged_at: now(),
                backup_created_at: review::clip(&manifest.created_at, 40),
                ticked: ticks.ids(),
                keys: ticks.keys,
                notes: notes.iter().map(|n| review::clip(n, 400)).take(50).collect(),
                source: source.clone(),
                files,
                removals: Vec::new(),
                agent,
                redo: clipped(&redo_of(manifest), 50, 400),
                bytes,
            },
            notes,
        ))
    })();
    let (marker, notes) = match staged {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&root);
            return Err(e);
        }
    };
    skipped.extend(notes);
    // The marker is what commits the step.
    if let Err(e) = write_json(&marker_path(data_dir), &marker) {
        let _ = std::fs::remove_dir_all(&root);
        return Err(BackupError::io("Could not record the restore", &e));
    }
    log::info!("backup: a restore ({} files, {} classes ticked) is staged and will be applied at the next start", marker.files.len(), marker.ticked.len());
    Ok(Staged {
        id,
        kind: KIND_RESTORE.to_string(),
        files: marker.files.len() as u64,
        bytes: marker.bytes,
        agent_storage: marker.agent.is_some(),
        redo: marker.redo,
        skipped: clipped(&skipped, 100, 400),
    })
}

/// Clean the staged files that carry more than data: the calendar loses its FormLogic sync state, a
/// plugin's settings lose every PIN, key and sealed value the backup holds while keeping the ones this
/// computer already has, the provider list loses its keys unless they were ticked, the run journal
/// keeps only runs that finished, and the autostart list keeps only services that have a template
/// here or in this restore. A file that cannot be cleaned is not brought back. Returns the names that
/// remain and what was said about the rest.
fn clean_staged(data_dir: &Path, files_root: &Path, names: &[String], ticks: &Ticks, limits: &Limits) -> Result<(Vec<String>, Vec<String>)> {
    let mut kept = Vec::new();
    let mut notes = Vec::new();
    let read = |path: &Path| std::fs::read(path).map_err(|e| e.to_string());
    for name in names {
        let path = container::safe_join(files_root, name, limits)?;
        let category = rules::category_of_backup_entry(name).map(|(c, _)| c).ok();
        let cleaned: Option<std::result::Result<Vec<u8>, String>> = match (category, name.as_str()) {
            (Some(Category::Calendar), _) => Some(read(&path).and_then(|b| super::sanitize::calendar_json(&b))),
            (Some(Category::PluginData), _) => {
                let local = std::fs::read(data_dir.join(native(name))).ok();
                Some(read(&path).and_then(|b| super::sanitize::plugin_json_with_local(&b, local.as_deref()).map(|(c, _)| c)))
            }
            (Some(Category::Providers), _) if !ticks.keys => Some(read(&path).and_then(|b| {
                super::sanitize::providers_without_keys(&b).map(|(c, removed)| {
                    if removed > 0 {
                        notes.push(format!("{removed} API key(s) in the provider list were left out: you did not tick the keys."));
                    }
                    c
                })
            })),
            (Some(Category::Flows), "bridge/ledger.jsonl") => Some(read(&path).map(|b| {
                let (c, left_out) = super::sanitize::ledger_finished_only(&b);
                if left_out > 0 {
                    notes.push(format!("{left_out} run record(s) of runs that were waiting or running were left out: nothing starts by itself."));
                }
                c
            })),
            _ => None,
        };
        match cleaned {
            None => kept.push(name.clone()),
            Some(Ok(bytes)) => {
                secret_file::write(&path, bytes).map_err(|e| BackupError::io("Could not clean a staged file", &e))?;
                kept.push(name.clone());
            }
            Some(Err(why)) => {
                let _ = std::fs::remove_file(&path);
                notes.push(format!("{name} was not brought back: {why}."));
            }
        }
    }
    // A service starts with OAIY only if OAIY has a template for it: one here already, or one in this restore.
    if kept.iter().any(|n| n == "services-autostart.json") {
        let mut known = Local::read(data_dir).template_ids;
        for name in kept.iter().filter(|n| n.starts_with("templates/")) {
            let staged = container::safe_join(files_root, name, limits)?;
            if let Some(id) = std::fs::read(&staged).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()).and_then(|v| v.get("id").and_then(|i| i.as_str().map(str::to_string))) {
                known.insert(id);
            }
        }
        let path = container::safe_join(files_root, "services-autostart.json", limits)?;
        match read(&path).and_then(|b| super::sanitize::autostart_known_only(&b, &known)) {
            Ok((bytes, dropped)) => {
                secret_file::write(&path, bytes).map_err(|e| BackupError::io("Could not clean a staged file", &e))?;
                if !dropped.is_empty() {
                    notes.push(format!("These services were not set to start with OAIY, because OAIY has no template for them: {}.", dropped.iter().take(20).map(|d| review::clip(d, 60)).collect::<Vec<_>>().join(", ")));
                }
            }
            Err(why) => {
                let _ = std::fs::remove_file(&path);
                kept.retain(|n| n != "services-autostart.json");
                notes.push(format!("services-autostart.json was not brought back: {why}."));
            }
        }
    }
    Ok((kept, notes))
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

/// What the newest snapshot was made by: `restore` (the button undoes it) or `undo` (the button redoes
/// what the undo took away).
pub fn undo_kind(data_dir: &Path) -> Option<String> {
    undo_dirs(data_dir).into_iter().next().map(|(_, record)| record.kind)
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
    let mut targets = Targets::new();
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
        targets.state(data_dir, rel)?;
    }
    for rel in &record.added {
        container::check_entry_name(rel, &opts.limits)?;
        targets.state(data_dir, rel)?;
    }
    let agent_zip = root.join("agent-storage.zip");
    let agent = match std::fs::metadata(&agent_zip) {
        Ok(m) if m.is_file() => {
            let (sha256, size) = sha256_file(&agent_zip).map_err(|e| BackupError::io("Could not read the saved copy", &e))?;
            // What the page saved is the person's own state: its settings go back, its keys were never in it.
            Some(MarkerAgent { size, sha256, apply_settings: true, apply_keys: false })
        }
        _ => None,
    };
    let marker = Marker {
        v: 1,
        id: super::random_id(),
        kind: KIND_UNDO.to_string(),
        staged_at: now(),
        backup_created_at: record.backup_created_at.clone(),
        ticked: Vec::new(),
        keys: false,
        notes: Vec::new(),
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
        skipped: Vec::new(),
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
    /// A staged file goes in at this step (as opposed to a file being taken away, in an undo).
    #[serde(default)]
    install: bool,
}

struct Journal {
    file: std::fs::File,
}

impl Journal {
    fn create(path: &Path) -> std::io::Result<Self> {
        let _ = std::fs::remove_file(path);
        Ok(Self { file: secret_file::create_new_owner_only(path)? })
    }

    fn line(&mut self, op: &str, rel: &str, had: bool, install: bool) -> std::io::Result<()> {
        let mut text = serde_json::to_string(&JournalLine { op: op.to_string(), rel: rel.to_string(), had, install }).unwrap_or_default();
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

/// One rename of a rollback. A test can make the next few fail, as a file that is locked at that
/// moment (a virus scanner, an indexer, the program itself) would.
fn put_back(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(test)]
    if ROLLBACK_FAILS.with(|c| {
        let left = c.get();
        c.set(left.saturating_sub(1));
        left > 0
    }) {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "the file is locked"));
    }
    secret_file::rename_over(from, to)
}

/// Put back what the journal says was begun, newest first. Safe to run more than once, and after a crash at any point.
///
/// Nothing is ever deleted here. What a step put in place goes back to where it came from first (for
/// an undo, the staged file is the only copy of what the restore replaced), and then the file that
/// was set aside goes back to its place. Returns the files that could not be put back, each with
/// whether a file of the person's had been set aside for it (its original is then in the holding
/// folder): the step is left as it is rather than risk overwriting the one copy there is.
fn roll_back(data_dir: &Path, staged_root: &Path, holding: &Path, lines: &[JournalLine]) -> Vec<(String, bool)> {
    let mut failed = Vec::new();
    for line in lines.iter().rev().filter(|l| l.op == "begin") {
        let target = data_dir.join(native(&line.rel));
        let kept = holding.join(native(&line.rel));
        let staged = staged_root.join(native(&line.rel));
        // Installed: the staged file is gone from its place and a plain file stands at the target.
        if line.install && !staged.exists() && target.is_file() && put_back(&target, &staged).is_err() {
            failed.push((line.rel.clone(), line.had));
            continue;
        }
        // The original is in the holding folder if it was set aside: it goes back over whatever is there now.
        if line.had && kept.exists() && put_back(&kept, &target).is_err() {
            failed.push((line.rel.clone(), true));
        }
    }
    failed.reverse();
    failed
}

/// Where the files an apply replaces or takes away are kept: for a restore, and for an undo too, so
/// an undo that overwrites or removes work done since the restore keeps a copy of it (the redo).
fn holding_of(data_dir: &Path, marker: &Marker) -> PathBuf {
    restore_dir(data_dir).join(format!("undo-{}", marker.id)).join("files")
}

fn record_last(data_dir: &Path, last: &LastRestore) {
    if let Err(e) = write_json(&last_result_path(data_dir), last) {
        log::warn!("backup: could not record how the restore went: {e}");
    }
}

fn failed(marker_id: &str, kind: &str, why: &str) -> LastRestore {
    LastRestore { id: marker_id.to_string(), kind: kind.to_string(), at: now(), ok: false, error: Some(why.to_string()), redo: Vec::new(), agent_storage: "none".into(), notes: Vec::new() }
}

/// Remove what a backup or a restore that was killed part-way leaves behind, at the start of the app
/// and before a staged restore is applied: the working copies under `backup/scratch` (plaintext, and
/// with the keys ticked, the provider keys), a `pending-<id>` folder that no marker names (a staging
/// that never got as far as its marker) and, when no marker waits, a set-aside folder of an undo.
/// What a waiting marker names, the undo snapshots and the Agent's import are left alone.
/// Returns how many folders were removed.
pub fn sweep_leftovers(data_dir: &Path) -> usize {
    let mut removed = 0;
    if let Ok(entries) = std::fs::read_dir(super::scratch_dir(data_dir)) {
        for entry in entries.flatten() {
            let path = entry.path();
            let done = match entry.file_type() {
                Ok(t) if t.is_dir() => std::fs::remove_dir_all(&path).is_ok(),
                _ => std::fs::remove_file(&path).is_ok(),
            };
            removed += usize::from(done);
        }
    }
    let waiting = read_json::<Marker>(&marker_path(data_dir));
    let marker_present = std::fs::symlink_metadata(marker_path(data_dir)).is_ok();
    let named = waiting.as_ref().map(|m| m.source.clone());
    if let Ok(entries) = std::fs::read_dir(restore_dir(data_dir)) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let orphan_pending = name.strip_prefix("pending-").is_some_and(is_id) && named.as_deref() != Some(name.as_str());
            // A set-aside folder that no snapshot record names and no waiting restore may need, and that
            // holds no file (a file in it is the only copy of something, and stays).
            let orphan_holding = name.strip_prefix("undo-").is_some_and(is_id) && !marker_present && !entry.path().join("undo.json").exists();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if orphan_pending && is_dir && std::fs::remove_dir_all(entry.path()).is_ok() {
                removed += 1;
            } else if orphan_holding && is_dir && remove_empty_tree(&entry.path()) {
                removed += 1;
            }
        }
    }
    if removed > 0 {
        log::info!("backup: removed {removed} leftover working folder(s) of a backup or restore that did not finish");
    }
    removed
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
        let stuck = roll_back(data_dir, &staged_root, &holding, &lines);
        if !stuck.is_empty() {
            return conclude_stuck(data_dir, &marker, "The restore was interrupted part-way.", &stuck, &holding);
        }
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
            let stuck = roll_back(data_dir, &staged_root, &holding, &lines);
            if !stuck.is_empty() {
                return conclude_stuck(data_dir, &marker, &why, &stuck, &holding);
            }
            let last = failed(&marker.id, &marker.kind, &format!("{why} Everything it had changed was put back."));
            conclude_failed(data_dir, &marker, last)
        }
    }
}

/// A rollback that could not put everything back: say which files, where their originals are, and
/// keep the evidence (the record of the restore and its journal are kept under other names, and
/// nothing that was set aside or staged is deleted), and never claim that all was put back.
fn conclude_stuck(data_dir: &Path, marker: &Marker, why: &str, stuck: &[(String, bool)], holding: &Path) -> ApplyOutcome {
    let dir = restore_dir(data_dir);
    let list = stuck.iter().take(10).map(|(rel, _)| review::clip(rel, 120)).collect::<Vec<_>>().join(", ");
    let more = if stuck.len() > 10 { format!(" and {} more", stuck.len() - 10) } else { String::new() };
    let set_aside = stuck.iter().filter(|(_, had)| *had).count();
    let mut message = format!("{why} Rolling it back did not finish: these files could not be put back: {list}{more}.");
    if set_aside > 0 {
        message.push_str(&format!(" Nothing was deleted: the original of {} of them is in {}.", if set_aside == stuck.len() { "each" } else { "some" }, holding.display()));
    }
    if set_aside < stuck.len() {
        message.push_str(" Some of them were not there before, and the file from the backup is still in place.");
    }
    message.push_str(" Close whatever may be holding them (another program, a virus scanner) and copy them back by hand, or ask for help.");
    let last = failed(&marker.id, &marker.kind, &message);
    record_last(data_dir, &last);
    let _ = std::fs::rename(marker_path(data_dir), dir.join(format!("failed-{}.json", marker.id)));
    let _ = std::fs::rename(journal_path(data_dir), dir.join(format!("apply-journal-{}.jsonl", marker.id)));
    log::error!("backup: a restore could not be rolled back completely: {} file(s), {} of them set aside in {}", stuck.len(), set_aside, holding.display());
    ApplyOutcome::Failed(last)
}

fn conclude_failed(data_dir: &Path, marker: &Marker, last: LastRestore) -> ApplyOutcome {
    let dir = restore_dir(data_dir);
    record_last(data_dir, &last);
    // What was staged is personal data: it does not stay behind after a failure. The folder that
    // held what was set aside goes only if the rollback emptied it: a file still in it is the only
    // copy of something, and stays.
    let holding_root = dir.join(format!("undo-{}", marker.id));
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
    let mut targets = Targets::fresh();
    let mut applied = Vec::new();
    for (_index, f) in marker.files.iter().enumerate() {
        let staged = container::safe_join(staged_root, &f.name, limits).map_err(|e| e.message)?;
        let target = data_dir.join(native(&f.name));
        let had = match targets.state(data_dir, &f.name).map_err(|e| e.message)? {
            Target::Absent => false,
            Target::File => true,
        };
        journal.line("begin", &f.name, had, true).map_err(|e| format!("Could not write the restore's journal ({e})."))?;
        if had {
            let kept = holding.join(native(&f.name));
            if let Some(parent) = kept.parent() {
                secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder for the saved copy ({e})."))?;
            }
            secret_file::rename_over(&target, &kept).map_err(|e| format!("Could not set aside a file it replaces ({e})."))?;
        }
        #[cfg(test)]
        inject_at(_index)?;
        if let Some(parent) = target.parent() {
            secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder ({e})."))?;
        }
        secret_file::rename_over(&staged, &target).map_err(|e| format!("Could not put a restored file in place ({e})."))?;
        applied.push((f.name.clone(), had));
    }
    for (_position, rel) in marker.removals.iter().enumerate() {
        let target = data_dir.join(native(rel));
        if !matches!(targets.state(data_dir, rel).map_err(|e| e.message)?, Target::File) {
            continue;
        }
        journal.line("begin", rel, true, false).map_err(|e| format!("Could not write the restore's journal ({e})."))?;
        let kept = holding.join(native(rel));
        if let Some(parent) = kept.parent() {
            secret_file::create_private_dir(parent).map_err(|e| format!("Could not make a folder for the saved copy ({e})."))?;
        }
        secret_file::rename_over(&target, &kept).map_err(|e| format!("Could not take away a file the restore had added ({e})."))?;
        #[cfg(test)]
        inject_at(marker.files.len() + _position)?;
        applied.push((rel.clone(), true));
    }
    journal.line("done", "", false, false).map_err(|e| format!("Could not finish the restore's journal ({e})."))?;
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
    // What was replaced or taken away is kept, for a restore and for an undo alike: an undo also
    // overwrites and removes files, and what a person did in them since is nowhere else.
    let record = UndoRecord {
        id: marker.id.clone(),
        applied_at: now(),
        backup_created_at: marker.backup_created_at.clone(),
        kind: marker.kind.clone(),
        replaced: applied.iter().filter(|(_, had)| *had).map(|(r, _)| r.clone()).collect(),
        added: applied.iter().filter(|(_, had)| !*had).map(|(r, _)| r.clone()).collect(),
    };
    let undo_root = dir.join(format!("undo-{}", marker.id));
    if secret_file::create_private_dir(&undo_root).is_ok() {
        if let Err(e) = write_json(&undo_root.join("undo.json"), &record) {
            log::warn!("backup: could not record what the {} replaced: {e}", marker.kind);
        }
    }
    let mut agent_storage = "none";
    if let Some(agent_marker) = &marker.agent {
        let zip = source.join("agent-storage.zip");
        match agent::leave_for_page(data_dir, &marker.id, &marker.kind, &zip, agent_marker.apply_settings, agent_marker.apply_keys) {
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
        notes: marker.notes.clone(),
    };
    record_last(data_dir, &last);
    // What was staged, or (for an undo) the snapshot that was just put back, is used up; the new
    // snapshot, of what this apply replaced, stays.
    let _ = std::fs::remove_dir_all(&source);
    prune_undo(data_dir);
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
        classes: marker.ticked,
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

#[cfg(test)]
thread_local! {
    /// How many of the next renames of a rollback fail, on this thread only.
    pub(crate) static ROLLBACK_FAILS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
