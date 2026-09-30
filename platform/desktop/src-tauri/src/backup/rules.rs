//! What a backup holds and what it leaves out, and why.
//!
//! The classification itself is `table.json` (see [`super::table`]): one table for every path under
//! the data folder, for the Agent's storage and for the keys inside a settings file. This module applies
//! it to the data folder: the walk that decides what a backup copies, and the answer to "what is this
//! name in a backup?" that a restore asks.
//!
//! The rule is default-deny. A file is backed up only if a row of the table says it comes back, and never
//! if an excluded row matches it first: every credential, key and machine-bound secret is denied by name,
//! so a new secret file added by later code is not swept in by accident (it is not in the table) and an
//! old one cannot be added by a badly named store (the excluded rows come first). Anything the table does
//! not know is left out and listed, so the person can see what was not saved.
//!
//! The same table decides what a restore accepts. A name in a backup that the table excludes (a credential,
//! a path into `plugins/`) makes the whole backup refused, so a restore never touches an excluded item; a
//! name the table does not know is not restored and is listed as "not restored: unknown item <name>".
//!
//! Paths are relative to the data folder, with forward slashes, and matched without regard to case.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::review::RestoreClass;
use super::table::{table, Class, Row};

/// What a backed-up file belongs to, for the summaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Category {
    Contacts,
    Calendar,
    Messages,
    Transfers,
    Flows,
    Settings,
    History,
    Voices,
    Templates,
    PluginData,
    Providers,
    Connectors,
    Agent,
}

impl Category {
    pub const ALL: [Category; 13] = [
        Category::Contacts,
        Category::Calendar,
        Category::Messages,
        Category::Transfers,
        Category::Flows,
        Category::Settings,
        Category::History,
        Category::Voices,
        Category::Templates,
        Category::PluginData,
        Category::Providers,
        Category::Connectors,
        Category::Agent,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Category::Contacts => "contacts",
            Category::Calendar => "calendar",
            Category::Messages => "messages",
            Category::Transfers => "transfers",
            Category::Flows => "flows",
            Category::Settings => "settings",
            Category::History => "history",
            Category::Voices => "voices",
            Category::Templates => "templates",
            Category::PluginData => "pluginData",
            Category::Providers => "providers",
            Category::Connectors => "connectors",
            Category::Agent => "agent",
        }
    }

    /// How the person reads it.
    pub fn label(self) -> &'static str {
        match self {
            Category::Contacts => "Contacts and what is remembered about callers",
            Category::Calendar => "Calendar",
            Category::Messages => "Messages callers left",
            Category::Transfers => "Transfer settings",
            Category::Flows => "Flows and triggers",
            Category::Settings => "Settings and setup",
            Category::History => "Run history and the Agent's change log",
            Category::Voices => "Voices",
            Category::Templates => "Service templates you edited or added",
            Category::PluginData => "Plugin data",
            Category::Providers => "AI providers and their API keys",
            Category::Connectors => "Connector descriptors",
            Category::Agent => "Agent conversations and projects",
        }
    }
}

/// One thing a backup leaves out, as the manifest records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Excluded {
    pub pattern: String,
    pub reason: String,
    /// What the person does again after a restore because of it (a credential that was left out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redo: Option<String>,
}

/// Every row of the table that leaves something out, for the tests.
pub fn all_rules() -> impl Iterator<Item = &'static Row> {
    table().desktop.iter().filter(|r| r.class == Class::Excluded)
}

/// The row for an unedited built-in template, which a restore has no need to bring back.
fn template_seed_row() -> &'static Row {
    table().desktop.iter().find(|r| r.id == "built-in-template").expect("the table has a row for built-in templates")
}

/// What to do with a path.
#[derive(Clone, Copy)]
pub enum Decision {
    Include { category: Category, secret: bool },
    Exclude(&'static Row),
    /// Not in the table and not denied by name: left out, and listed as not recognised.
    Unknown,
}

/// Decide about one relative path (`include_keys`: the person ticked "Include my API provider keys").
pub fn classify(rel: &str, include_keys: bool) -> Decision {
    match table().desktop_row(rel, include_keys) {
        None => Decision::Unknown,
        Some(row) if row.class == Class::Excluded => Decision::Exclude(row),
        Some(row) => Decision::Include { category: row.category.expect("a row that comes back has a category"), secret: row.secret },
    }
}

/// Whether the walk should go into this folder, or leave it out (and why). A folder is walked if
/// something in the table that comes back could be inside it, even when a row leaves out most of it
/// (the rest of a plugin's data folder, with its settings file).
pub fn classify_dir(rel: &str, include_keys: bool) -> Decision {
    let t = table();
    let holds = t.desktop_holds_under(rel);
    let excluded = t.desktop.iter().find(|r| r.class == Class::Excluded && !(r.skip_when_keys && include_keys) && r.matches(rel, true));
    match (excluded, holds) {
        (_, true) => Decision::Include { category: Category::Settings, secret: false },
        (Some(row), false) => Decision::Exclude(row),
        (None, false) => Decision::Unknown,
    }
}

/// What a name in a backup is, for a restore.
#[derive(Clone, Copy, Debug)]
pub struct Standing {
    pub category: Category,
    pub secret: bool,
    /// `data` (comes back without a tick) or `runs` (only with its tick).
    pub class: Class,
    pub tick: Option<RestoreClass>,
    /// The row of the table it is (none for the Agent's storage archive, which has rows of its own inside).
    pub row: Option<&'static Row>,
}

/// What a name in a backup is: `Ok(Some(_))` for a thing the table knows and lets come back, `Ok(None)`
/// for one it does not know (it is not restored, and it is listed), and `Err` for one a backup must
/// never hold (the whole backup is refused). What a backup says about itself (that it holds keys, say)
/// plays no part: whether the provider list comes back, and with or without its keys, is what the person
/// ticks when restoring.
pub fn standing_of_backup_entry(name: &str) -> Result<Option<Standing>, String> {
    if name == super::AGENT_ENTRY {
        return Ok(Some(Standing { category: Category::Agent, secret: false, class: Class::Data, tick: None, row: None }));
    }
    match table().desktop_row(name, true) {
        None => Ok(None),
        Some(row) if row.class == Class::Excluded => Err(format!("it holds an item OAIY never backs up ({})", row.pattern)),
        Some(row) => Ok(Some(Standing { category: row.category.expect("a row that comes back has a category"), secret: row.secret, class: row.class, tick: row.tick, row: Some(row) })),
    }
}

/// The category a name in a backup belongs to (`None`: the table does not know it), or why the backup
/// must be refused for holding it.
pub fn category_of_backup_entry(name: &str) -> Result<Option<(Category, bool)>, String> {
    standing_of_backup_entry(name).map(|s| s.map(|s| (s.category, s.secret)))
}

impl Standing {
    /// The tick the whole file needs to come back, when it needs one. A settings file that has a key table
    /// of its own needs none: each key in it has its own class, and the keys that cannot act come back
    /// without a tick.
    pub fn file_tick(&self) -> Option<RestoreClass> {
        match self.row {
            Some(row) => row.file_tick(),
            None => self.tick,
        }
    }

    /// The key table the file is filtered through, when it has one.
    pub fn keys(&self) -> Option<&'static str> {
        self.row.and_then(|r| r.keys.as_deref())
    }
}

/// The tick a name in a backup needs, when it needs one (see [`Standing::file_tick`]).
pub fn tick_of(name: &str) -> Option<RestoreClass> {
    standing_of_backup_entry(name).ok().flatten().and_then(|s| s.file_tick())
}

/// The key table a name in a backup is filtered through, when it has one.
pub fn keys_of(name: &str) -> Option<&'static str> {
    standing_of_backup_entry(name).ok().flatten().and_then(|s| s.keys())
}

/// How a file is cleaned as it is copied into a backup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sanitize {
    None,
    /// A file passed through the key table of that name: only the keys it lists come through (a plugin's or the Agent's
    /// settings, the calendar).
    Keys(&'static str),
}

/// One file the plan will copy.
#[derive(Clone, Debug)]
pub struct PlanItem {
    pub rel: String,
    pub abs: PathBuf,
    pub size: u64,
    pub category: Category,
    pub secret: bool,
    pub sanitize: Sanitize,
}

/// What a walk of the data folder found.
#[derive(Debug, Default)]
pub struct Plan {
    pub items: Vec<PlanItem>,
    /// What was left out, one record per rule (or per unrecognised top-level name).
    pub excluded: Vec<Excluded>,
    /// Symbolic links and junctions, which are never followed.
    pub links: Vec<String>,
    /// Entries that could not be read.
    pub unreadable: Vec<String>,
}

/// Whether a directory entry is a link that must not be followed: a symbolic link, a junction, or
/// any other reparse point.
pub(crate) fn is_link(entry_meta: &std::fs::Metadata) -> bool {
    if entry_meta.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if entry_meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return true;
        }
    }
    false
}

/// Walk `data_dir` and decide about everything in it, without following a link and without going
/// into a folder that is left out (a models folder can be tens of gigabytes).
pub fn plan(data_dir: &Path, include_keys: bool) -> Plan {
    let mut plan = Plan::default();
    let mut excluded: BTreeMap<String, Excluded> = BTreeMap::new();
    let mut stack: Vec<(PathBuf, String)> = vec![(data_dir.to_path_buf(), String::new())];
    while let Some((dir, rel_dir)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => {
                if !rel_dir.is_empty() {
                    plan.unreadable.push(rel_dir.clone());
                }
                continue;
            }
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel = if rel_dir.is_empty() { name.clone() } else { format!("{rel_dir}/{name}") };
            let abs = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&abs) else {
                plan.unreadable.push(rel);
                continue;
            };
            if is_link(&meta) {
                plan.links.push(rel);
                continue;
            }
            if meta.is_dir() {
                if let Some(record) = plugin_folder_exclusion(&rel) {
                    excluded.entry(record.pattern.clone()).or_insert(record);
                    continue;
                }
                match classify_dir(&rel, include_keys) {
                    Decision::Include { .. } => stack.push((abs, rel)),
                    Decision::Exclude(row) if row.descend => stack.push((abs, rel)),
                    Decision::Exclude(row) => {
                        excluded.entry(row.pattern.clone()).or_insert_with(|| row.record());
                    }
                    Decision::Unknown => {
                        note_unknown(&mut excluded, &rel, true);
                    }
                }
                continue;
            }
            if !meta.is_file() {
                continue;
            }
            match classify(&rel, include_keys) {
                Decision::Include { category, secret } => {
                    if category == Category::Templates && template_is_unedited(&abs) {
                        let seed = template_seed_row();
                        excluded.entry(seed.pattern.clone()).or_insert_with(|| seed.record());
                        continue;
                    }
                    let row = table().desktop_row(&rel, include_keys);
                    let sanitize = match row.and_then(|r| r.keys.as_deref()) {
                        Some(keys) => Sanitize::Keys(keys),
                        None => Sanitize::None,
                    };
                    plan.items.push(PlanItem { rel, abs, size: meta.len(), category, secret, sanitize });
                }
                Decision::Exclude(row) => {
                    excluded.entry(row.pattern.clone()).or_insert_with(|| row.record());
                }
                Decision::Unknown => note_unknown(&mut excluded, &rel, false),
            }
        }
    }
    plan.items.sort_by(|a, b| a.rel.cmp(&b.rel));
    plan.links.sort();
    plan.unreadable.sort();
    plan.excluded = excluded.into_values().collect();
    plan
}

/// `plugin-data/<id>` for a plugin the table has no key table for: its folder is left out whole, and
/// listed by its own name.
fn plugin_folder_exclusion(rel: &str) -> Option<Excluded> {
    let lower = rel.to_lowercase();
    let id = lower.strip_prefix("plugin-data/")?;
    if id.contains('/') || table().plugin_ids().iter().any(|known| known == id) {
        return None;
    }
    let row = table().desktop.iter().find(|r| r.id == "plugin-data-other")?;
    Some(Excluded { pattern: format!("plugin-data/{id}/"), reason: row.reason.clone(), redo: row.redo.clone() })
}

fn note_unknown(excluded: &mut BTreeMap<String, Excluded>, rel: &str, dir: bool) {
    let top = rel.split('/').next().unwrap_or(rel);
    let pattern = if dir || rel.contains('/') { format!("{top}/") } else { top.to_string() };
    excluded.entry(pattern.clone()).or_insert_with(|| Excluded {
        pattern,
        reason: "Not recognised as personal data, so it is not backed up.".to_string(),
        redo: None,
    });
}

/// A template file that is the copy OAIY seeded: its `.<name>.seed` snapshot is the same bytes.
fn template_is_unedited(abs: &Path) -> bool {
    let Some(name) = abs.file_name().and_then(|n| n.to_str()) else { return false };
    let Some(dir) = abs.parent() else { return false };
    match (std::fs::read(abs), std::fs::read(dir.join(format!(".{name}.seed")))) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
