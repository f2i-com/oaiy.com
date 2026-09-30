//! The classification table: everything a restore can write, and what each thing is.
//!
//! `table.json` is the one place that says it. A restore is DEFAULT-DENY: a path or a key that is not
//! in the table is excluded (and the dry run says "not restored: unknown item <name>"). Each entry
//! has exactly one class:
//!
//! - **excluded**: never restored, with the reason and what to do again;
//! - **data**: a value that no code path turns into behaviour (typed, carrying no words), which comes back without a tick;
//! - **runs**: something a model reads as instructions, that plays audio to callers, sends messages or
//!   calls, or changes trust or network destinations. It is listed by name, unticked by default, and
//!   applied only when its tick is ticked.
//!
//! Three kinds of thing are classified: the paths under the desktop's data folder (`desktop`), the
//! names inside the Agent's storage archive (`agent`: OPFS files and IndexedDB), and the keys inside a
//! JSON file that holds settings (`keyTables`: a plugin's settings, the Agent's settings, a campaign).
//! The docs (docs/BACKUP.md) are generated from this file, and tests read the real writers and fail
//! when a store is not classified here.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::review::RestoreClass;
use super::rules::{Category, Excluded};

const TABLE_JSON: &str = include_str!("table.json");

/// What a row says about what it matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Never restored.
    Excluded,
    /// Comes back without a tick.
    Data,
    /// Comes back only with its tick.
    Runs,
}

impl Class {
    fn parse(text: &str) -> Result<Class, String> {
        match text {
            "excluded" => Ok(Class::Excluded),
            "data" => Ok(Class::Data),
            "runs" => Ok(Class::Runs),
            other => Err(format!("unknown class \"{other}\"")),
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Class::Excluded => "excluded",
            Class::Data => "data",
            Class::Runs => "runs",
        }
    }
}

/// How an item that is applied is combined with what is there already.
pub const MERGES: [&str; 5] = ["union", "campaign", "campaign-index", "union-lines", "replace"];

// ---- globs -------------------------------------------------------------------------------------------

/// A path pattern of `/`-separated parts: `*` inside a part stands for any run of characters, and a part
/// that is `**` stands for any number of parts (none included), so `dir/**` is `dir` and everything in it.
#[derive(Clone, Debug)]
pub struct Glob {
    raw: String,
    parts: Vec<String>,
}

fn part_matches(pattern: &str, part: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = part.chars().collect();
    // Iterative wildcard match with backtracking to the last `*`.
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn parts_match(pattern: &[String], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((first, rest)) if first == "**" => (0..=path.len()).any(|skip| parts_match(rest, &path[skip..])),
        Some((first, rest)) => match path.split_first() {
            Some((head, tail)) => part_matches(first, head) && parts_match(rest, tail),
            None => false,
        },
    }
}

/// Whether some path inside the folder `dir` could match `pattern`.
fn parts_could_match_under(pattern: &[String], dir: &[&str]) -> bool {
    match (pattern.split_first(), dir.split_first()) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some((first, _)), Some(_)) if first == "**" => true,
        (Some((first, rest)), Some((head, tail))) => part_matches(first, head) && parts_could_match_under(rest, tail),
    }
}

fn split(path: &str, fold: bool) -> Vec<String> {
    path.split('/').map(|p| if fold { p.to_lowercase() } else { p.to_string() }).collect()
}

impl Glob {
    fn new(raw: &str) -> Glob {
        Glob { raw: raw.to_string(), parts: raw.split('/').map(str::to_string).collect() }
    }

    /// The pattern as written.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Whether `path` is matched. `fold`: without regard to case (the desktop's names are; the Agent's are not).
    pub fn matches(&self, path: &str, fold: bool) -> bool {
        let parts = split(path, fold);
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        if fold {
            let pattern: Vec<String> = self.parts.iter().map(|p| p.to_lowercase()).collect();
            parts_match(&pattern, &refs)
        } else {
            parts_match(&self.parts, &refs)
        }
    }

    /// Whether something inside the folder `dir` could be matched.
    pub fn could_match_under(&self, dir: &str, fold: bool) -> bool {
        let parts = split(dir, fold);
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        if fold {
            let pattern: Vec<String> = self.parts.iter().map(|p| p.to_lowercase()).collect();
            parts_could_match_under(&pattern, &refs)
        } else {
            parts_could_match_under(&self.parts, &refs)
        }
    }
}

// ---- rows --------------------------------------------------------------------------------------------

/// One row of the `desktop` or `agent` list: a set of paths and what they are.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub class: Class,
    pub globs: Vec<Glob>,
    /// How the manifest and the docs show it.
    pub pattern: String,
    pub category: Option<Category>,
    pub tick: Option<RestoreClass>,
    pub secret: bool,
    /// An excluded folder that is still walked, because something inside it has a row of its own.
    pub descend: bool,
    /// An excluded row that stops applying when the provider keys are included.
    pub skip_when_keys: bool,
    /// Not mentioned in the dry run (the archive's own record).
    pub quiet: bool,
    pub merge: Option<String>,
    /// The key table that filters the inside of the file.
    pub keys: Option<String>,
    pub what: String,
    pub reason: String,
    pub redo: Option<String>,
    /// What reads the value, and why that is (or is not) behaviour: the audit's answer for this row.
    pub reads: Option<String>,
    /// Where it is read: `<path from the repository root>#<a name that is in that file>`.
    pub readers: Vec<String>,
    /// For a data row that has no key table: why the value can be trusted without one.
    pub keyless_because: Option<String>,
    words_under: Option<String>,
    words: Vec<String>,
}

impl Row {
    /// Whether the row matches `path` (`fold`: without regard to case).
    pub fn matches(&self, path: &str, fold: bool) -> bool {
        if self.globs.iter().any(|g| g.matches(path, fold)) {
            return true;
        }
        let Some(under) = &self.words_under else { return false };
        let lower = path.to_lowercase();
        let parts: Vec<&str> = lower.split('/').collect();
        parts.first() == Some(&under.as_str()) && parts.iter().any(|part| self.words.iter().any(|w| part.contains(w.as_str())))
    }

    /// The tick the whole file needs to come back, when it needs one. A file with a key table that holds keys
    /// that cannot act (data) has none: each key has its own class, and those come back without a tick. A file
    /// whose every key acts (a campaign) needs its tick as a file.
    pub fn file_tick(&self) -> Option<RestoreClass> {
        match &self.keys {
            Some(name) if table().key_table(name).is_some_and(|t| t.keys.iter().any(|k| k.class == Class::Data)) => None,
            _ => self.tick,
        }
    }

    /// Whether the row could match something inside the folder `dir`.
    pub fn could_match_under(&self, dir: &str, fold: bool) -> bool {
        self.globs.iter().any(|g| g.could_match_under(dir, fold))
    }

    /// The manifest's record of what this row leaves out.
    pub fn record(&self) -> Excluded {
        Excluded { pattern: self.pattern.clone(), reason: self.reason.clone(), redo: self.redo.clone() }
    }
}

// ---- key tables --------------------------------------------------------------------------------------

/// What a value at a key may be. A value of another kind is not restored.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueType {
    Bool,
    Int { min: i64, max: i64 },
    Number { min: f64, max: f64 },
    Str { max_chars: usize },
    Enum(Vec<String>),
    Url { max_chars: usize },
    /// A list of short strings.
    Strings { max_items: usize, max_chars: usize },
    /// An object whose keys are the person's (a campaign's fields): every value a short string.
    StringsMap { max_items: usize, max_chars: usize },
    /// The same, but a value may be a string, a number or a yes/no.
    ScalarsMap { max_items: usize, max_chars: usize },
    /// An object whose keys have rows of their own below it.
    Object,
    /// A list of objects whose keys have rows of their own below `path[]`.
    Objects { max_items: usize },
    /// A time of day, `HH:MM`.
    Time,
    /// A local date and time, `YYYY-MM-DDTHH:MM` (seconds allowed).
    DateTime,
    /// A moment as RFC 3339 writes it.
    Stamp,
    /// An identifier: letters, digits, `_`, `-` and `.`, at most 100 characters.
    Slug,
    /// The id the calendar gives an appointment: `appt_` and 32 lowercase hexadecimal digits (and no other id).
    CalendarId,
    /// Seven days, Monday first, each a list of opening spans `{open, close}` in `HH:MM`.
    WeekHours,
}

impl ValueType {
    /// Whether a value of this type can carry words: text a person typed, a list or map of it, an address. The audit does not let
    /// such a value be data (a model can read words as instructions, and a person can be told them).
    pub fn carries_words(&self) -> bool {
        matches!(self, ValueType::Str { .. } | ValueType::Url { .. } | ValueType::Strings { .. } | ValueType::StringsMap { .. } | ValueType::ScalarsMap { .. })
    }
}

#[derive(Clone, Debug)]
pub struct KeyRow {
    /// `a.b` for a key inside an object, `a[]` for the elements of a list, `a[].b` for a key inside them.
    pub path: String,
    pub class: Class,
    pub tick: Option<RestoreClass>,
    pub ty: Option<ValueType>,
    pub secret: bool,
    /// The address (a sibling key, of the type `url`) that a secret is for: a key comes back only with the address it was kept for,
    /// so when that address is not brought back (it holds a credential, is not a web address, or was left out) the key is not either.
    /// Without it a key for a gateway would arrive in a record that has no address, and the Agent takes that for the vendor's own.
    pub goes_with: Option<String>,
    pub merge: Option<String>,
    pub what: String,
    pub reason: String,
    pub redo: Option<String>,
    /// What reads the value, and why that is (or is not) behaviour: the audit's answer for this key.
    pub reads: Option<String>,
    /// Where it is read: `<path from the repository root>#<a name that is in that file>`.
    pub readers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct KeyTable {
    pub name: String,
    pub file: String,
    pub about: String,
    pub keys: Vec<KeyRow>,
}

impl KeyTable {
    pub fn row(&self, path: &str) -> Option<&KeyRow> {
        self.keys.iter().find(|k| k.path == path)
    }
}

/// What the scanner tests read.
#[derive(Clone, Debug, Default)]
pub struct Scan {
    pub desktop_names: BTreeMap<String, String>,
    pub agent_names: BTreeMap<String, String>,
    /// The modules of the Agent's page that touch browser storage, and what each does with it.
    pub agent_modules: BTreeMap<String, String>,
}

pub struct Table {
    pub desktop: Vec<Row>,
    pub agent: Vec<Row>,
    pub key_tables: BTreeMap<String, KeyTable>,
    /// What the audit decided about things that are not rows of the table.
    pub audit: Vec<AuditNote>,
    pub scan: Scan,
}

// ---- loading -----------------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawTable {
    #[serde(rename = "_about")]
    _about: String,
    version: u32,
    desktop: Vec<RawRow>,
    agent: Vec<RawRow>,
    key_tables: BTreeMap<String, RawKeyTable>,
    #[serde(default)]
    audit: Vec<RawAudit>,
    scan: RawScan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawRow {
    id: String,
    class: String,
    paths: Vec<String>,
    #[serde(default)]
    pattern: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    tick: Option<String>,
    #[serde(default)]
    secret: bool,
    #[serde(default)]
    descend: bool,
    #[serde(default)]
    skip_when_keys: bool,
    #[serde(default)]
    quiet: bool,
    #[serde(default)]
    merge: Option<String>,
    #[serde(default)]
    keys: Option<String>,
    #[serde(default)]
    words_under: Option<String>,
    #[serde(default)]
    words: Vec<String>,
    what: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    redo: Option<String>,
    #[serde(default)]
    reads: Option<String>,
    #[serde(default)]
    readers: Vec<String>,
    #[serde(default)]
    keyless_because: Option<String>,
}

/// A judgment of the audit about something that is not a row of the table (a place nothing is stored in).
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawAudit {
    item: String,
    class: String,
    reads: String,
    why: String,
    #[serde(default)]
    readers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AuditNote {
    pub item: String,
    pub class: String,
    pub reads: String,
    pub why: String,
    pub readers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawKeyTable {
    file: String,
    about: String,
    keys: Vec<RawKey>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawKey {
    path: String,
    class: String,
    #[serde(default)]
    tick: Option<String>,
    #[serde(default, rename = "type")]
    ty: Option<String>,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default)]
    max_items: Option<usize>,
    #[serde(default)]
    options: Vec<String>,
    #[serde(default)]
    secret: bool,
    #[serde(default)]
    goes_with: Option<String>,
    #[serde(default)]
    merge: Option<String>,
    what: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    redo: Option<String>,
    #[serde(default)]
    reads: Option<String>,
    #[serde(default)]
    readers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawScan {
    #[serde(rename = "_about")]
    _about: String,
    desktop_names: BTreeMap<String, String>,
    agent_names: BTreeMap<String, String>,
    agent_modules: BTreeMap<String, String>,
}

fn tick_of(raw: &Option<String>) -> Result<Option<RestoreClass>, String> {
    match raw {
        None => Ok(None),
        Some(id) => RestoreClass::from_id(id).map(Some).ok_or_else(|| format!("unknown tick \"{id}\"")),
    }
}

fn row_from(raw: RawRow) -> Result<Row, String> {
    let fail = |why: &str| format!("row {}: {why}", raw.id);
    let class = Class::parse(&raw.class).map_err(|e| fail(&e))?;
    let tick = tick_of(&raw.tick).map_err(|e| fail(&e))?;
    let category = match &raw.category {
        None => None,
        Some(id) => Some(Category::ALL.into_iter().find(|c| c.id() == id).ok_or_else(|| fail(&format!("unknown category \"{id}\"")))?),
    };
    if class == Class::Runs && tick.is_none() {
        return Err(fail("a row that runs things needs a tick"));
    }
    if class != Class::Runs && tick.is_some() {
        return Err(fail("only a row that runs things has a tick"));
    }
    if class == Class::Excluded && raw.reason.trim().is_empty() {
        return Err(fail("an excluded row says why"));
    }
    if class != Class::Excluded && raw.redo.is_some() {
        return Err(fail("only an excluded row says what to do again"));
    }
    if class != Class::Excluded && (raw.descend || raw.skip_when_keys) {
        return Err(fail("only an excluded row is walked or skipped"));
    }
    if let Some(merge) = &raw.merge {
        if !MERGES.contains(&merge.as_str()) {
            return Err(fail(&format!("unknown merge \"{merge}\"")));
        }
    }
    let pattern = raw.pattern.clone().unwrap_or_else(|| raw.paths.join(", "));
    Ok(Row {
        id: raw.id.clone(),
        class,
        globs: raw.paths.iter().map(|p| Glob::new(p)).collect(),
        pattern,
        category,
        tick,
        secret: raw.secret,
        descend: raw.descend,
        skip_when_keys: raw.skip_when_keys,
        quiet: raw.quiet,
        merge: raw.merge,
        keys: raw.keys,
        what: raw.what,
        reason: raw.reason,
        redo: raw.redo,
        reads: raw.reads,
        readers: raw.readers,
        keyless_because: raw.keyless_because,
        words_under: raw.words_under,
        words: raw.words,
    })
}

fn key_from(raw: RawKey) -> Result<KeyRow, String> {
    let fail = |why: &str| format!("key {}: {why}", raw.path);
    let class = Class::parse(&raw.class).map_err(|e| fail(&e))?;
    let tick = tick_of(&raw.tick).map_err(|e| fail(&e))?;
    if class == Class::Runs && tick.is_none() {
        return Err(fail("a key that runs things needs a tick"));
    }
    if class != Class::Runs && tick.is_some() {
        return Err(fail("only a key that runs things has a tick"));
    }
    if class == Class::Excluded && raw.reason.trim().is_empty() {
        return Err(fail("an excluded key says why"));
    }
    if let Some(merge) = &raw.merge {
        if !MERGES.contains(&merge.as_str()) {
            return Err(fail(&format!("unknown merge \"{merge}\"")));
        }
    }
    let ty = match (class, raw.ty.as_deref()) {
        (Class::Excluded, _) => None,
        (_, None) => return Err(fail("a key that comes back has a type")),
        (_, Some("bool")) => Some(ValueType::Bool),
        (_, Some("int")) => Some(ValueType::Int { min: raw.min.ok_or_else(|| fail("an int has a min"))? as i64, max: raw.max.ok_or_else(|| fail("an int has a max"))? as i64 }),
        (_, Some("number")) => Some(ValueType::Number { min: raw.min.ok_or_else(|| fail("a number has a min"))?, max: raw.max.ok_or_else(|| fail("a number has a max"))? }),
        (_, Some("string")) => Some(ValueType::Str { max_chars: raw.max_chars.ok_or_else(|| fail("a string has maxChars"))? }),
        (_, Some("enum")) => Some(ValueType::Enum(raw.options.clone())),
        (_, Some("url")) => Some(ValueType::Url { max_chars: raw.max_chars.ok_or_else(|| fail("a url has maxChars"))? }),
        (_, Some("strings")) => Some(ValueType::Strings { max_items: raw.max_items.ok_or_else(|| fail("strings has maxItems"))?, max_chars: raw.max_chars.ok_or_else(|| fail("strings has maxChars"))? }),
        (_, Some("strings-map")) => Some(ValueType::StringsMap { max_items: raw.max_items.ok_or_else(|| fail("a map has maxItems"))?, max_chars: raw.max_chars.ok_or_else(|| fail("a map has maxChars"))? }),
        (_, Some("scalars-map")) => Some(ValueType::ScalarsMap { max_items: raw.max_items.ok_or_else(|| fail("a map has maxItems"))?, max_chars: raw.max_chars.ok_or_else(|| fail("a map has maxChars"))? }),
        (_, Some("time")) => Some(ValueType::Time),
        (_, Some("datetime")) => Some(ValueType::DateTime),
        (_, Some("stamp")) => Some(ValueType::Stamp),
        (_, Some("slug")) => Some(ValueType::Slug),
        (_, Some("calendarId")) => Some(ValueType::CalendarId),
        (_, Some("weekHours")) => Some(ValueType::WeekHours),
        (_, Some("object")) => Some(ValueType::Object),
        (_, Some("objects")) => Some(ValueType::Objects { max_items: raw.max_items.ok_or_else(|| fail("objects has maxItems"))? }),
        (_, Some(other)) => return Err(fail(&format!("unknown type \"{other}\""))),
    };
    if raw.goes_with.is_some() && !raw.secret {
        return Err(fail("only a secret goes with an address"));
    }
    Ok(KeyRow { path: raw.path.clone(), class, tick, ty, secret: raw.secret, goes_with: raw.goes_with, merge: raw.merge, what: raw.what, reason: raw.reason, redo: raw.redo, reads: raw.reads, readers: raw.readers })
}

impl Table {
    /// Read a table. Every row is checked as it is read: a table that does not hold together does not load.
    pub fn parse(text: &str) -> Result<Table, String> {
        let raw: RawTable = serde_json::from_str(text).map_err(|e| format!("table.json: {e}"))?;
        if raw.version != 1 {
            return Err("table.json: unknown version".to_string());
        }
        let desktop = raw.desktop.into_iter().map(row_from).collect::<Result<Vec<_>, _>>()?;
        let agent = raw.agent.into_iter().map(row_from).collect::<Result<Vec<_>, _>>()?;
        let mut key_tables = BTreeMap::new();
        for (name, kt) in raw.key_tables {
            let keys = kt.keys.into_iter().map(key_from).collect::<Result<Vec<_>, _>>()?;
            key_tables.insert(name.clone(), KeyTable { name, file: kt.file, about: kt.about, keys });
        }
        let audit = raw.audit.into_iter().map(|a| AuditNote { item: a.item, class: a.class, reads: a.reads, why: a.why, readers: a.readers }).collect();
        let table = Table { desktop, agent, key_tables, audit, scan: Scan { desktop_names: raw.scan.desktop_names, agent_names: raw.scan.agent_names, agent_modules: raw.scan.agent_modules } };
        let problems = table.problems();
        if problems.is_empty() {
            Ok(table)
        } else {
            Err(format!("table.json does not hold together: {}", problems.join("; ")))
        }
    }

    /// What is wrong with the table as a whole (empty when it is sound).
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (row, is_desktop) in self.desktop.iter().map(|r| (r, true)).chain(self.agent.iter().map(|r| (r, false))) {
            if !seen.insert(row.id.as_str()) {
                out.push(format!("the id \"{}\" is used twice", row.id));
            }
            if is_desktop && row.class != Class::Excluded && row.category.is_none() {
                out.push(format!("row {} comes back but has no category", row.id));
            }
            if row.pattern.trim().is_empty() {
                out.push(format!("row {} has no pattern", row.id));
            }
            if let Some(keys) = &row.keys {
                if !self.key_tables.contains_key(keys) {
                    out.push(format!("row {} names the key table \"{keys}\", which is not there", row.id));
                }
            }
            if row.globs.is_empty() && row.words.is_empty() && row.id != "built-in-template" {
                out.push(format!("row {} matches nothing", row.id));
            }
        }
        // The audit's rule, held by the loader: a value is data only if it says what reads it (and where), it feeds no
        // behaviour, and it carries no words; a data row that has no key table says why it needs none.
        let audited = |what: &str, reads: &Option<String>, readers: &[String], out: &mut Vec<String>| {
            match reads {
                Some(text) if text.to_lowercase().contains(FEEDS_NOTHING) => {}
                _ => out.push(format!("{what} is data and does not say what reads it and that it \"{FEEDS_NOTHING}\" (reads)")),
            }
            if readers.is_empty() {
                out.push(format!("{what} is data and does not name the code that reads it (readers)"));
            }
        };
        for row in self.desktop.iter().chain(&self.agent).filter(|r| r.class == Class::Data) {
            audited(&format!("row {}", row.id), &row.reads, &row.readers, &mut out);
            if row.keys.is_none() && row.keyless_because.as_deref().is_none_or(|w| w.trim().is_empty()) {
                out.push(format!("row {} is data and has no key table, and does not say why it needs none (keylessBecause)", row.id));
            }
        }
        for (name, kt) in &self.key_tables {
            for key in kt.keys.iter().filter(|k| k.class == Class::Data && !matches!(k.ty, Some(ValueType::Object | ValueType::Objects { .. }))) {
                audited(&format!("key {name}: \"{}\"", key.path), &key.reads, &key.readers, &mut out);
                if key.ty.as_ref().is_some_and(ValueType::carries_words) {
                    out.push(format!("key table {name}: \"{}\" is data and can carry words: words are read", key.path));
                }
            }
        }
        for (name, kt) in &self.key_tables {
            let mut paths = HashSet::new();
            for key in &kt.keys {
                if !paths.insert(key.path.as_str()) {
                    out.push(format!("key table {name}: \"{}\" is listed twice", key.path));
                }
                if key.path.is_empty() || key.path.starts_with('.') || key.path.ends_with('.') || key.path.contains("..") {
                    out.push(format!("key table {name}: \"{}\" is not a path", key.path));
                }
                // A secret that goes with an address names one that is in the table, beside it, and is one.
                if let Some(partner) = &key.goes_with {
                    let sibling = sibling_path(&key.path, partner);
                    if !matches!(kt.row(&sibling), Some(KeyRow { class: Class::Runs | Class::Data, ty: Some(ValueType::Url { .. }), .. })) {
                        out.push(format!("key table {name}: \"{}\" goes with \"{sibling}\", which is not an address that comes back", key.path));
                    }
                }
                // Every key that comes back sits under a container that is listed and comes back.
                if let Some(parent) = parent_of(&key.path) {
                    match kt.row(&parent) {
                        Some(p) if p.class != Class::Excluded && matches!(p.ty, Some(ValueType::Object | ValueType::Objects { .. })) => {}
                        _ => {
                            if key.class != Class::Excluded {
                                out.push(format!("key table {name}: \"{}\" has no container \"{parent}\" that comes back", key.path));
                            }
                        }
                    }
                }
            }
        }
        for kt in self.key_tables.values() {
            if !self.desktop.iter().chain(&self.agent).any(|r| r.keys.as_deref() == Some(kt.name.as_str())) {
                out.push(format!("key table {} is used by no row", kt.name));
            }
        }
        out
    }

    /// The row for a path under the data folder (case is ignored), or `None`: not in the table.
    pub fn desktop_row(&self, rel: &str, include_keys: bool) -> Option<&Row> {
        self.desktop.iter().find(|r| !(r.skip_when_keys && include_keys) && r.matches(rel, true))
    }

    /// The row for a name inside the Agent's storage archive (case counts), or `None`: not in the table.
    pub fn agent_row(&self, name: &str) -> Option<&Row> {
        self.agent.iter().find(|r| r.matches(name, false))
    }

    /// Whether some path that is not excluded could sit inside the folder `dir` of the data folder.
    pub fn desktop_holds_under(&self, dir: &str) -> bool {
        self.desktop.iter().any(|r| r.class != Class::Excluded && r.could_match_under(dir, true))
    }

    /// The ids of the plugins whose settings are backed up (`plugin.<id>` key tables).
    pub fn plugin_ids(&self) -> Vec<String> {
        self.key_tables.keys().filter_map(|k| k.strip_prefix("plugin.").map(str::to_string)).collect()
    }

    pub fn key_table(&self, name: &str) -> Option<&KeyTable> {
        self.key_tables.get(name)
    }
}

/// What a data row or key must say of what reads it.
pub const FEEDS_NOTHING: &str = "feeds no behaviour";

/// The key a key sits inside (`gate` for `gate.mode`, `providers` for `providers[].id`), if it sits in one.
/// The path of a key beside `path` (in the same object): `providers[].apiKey` and `baseUrl` make `providers[].baseUrl`.
fn sibling_path(path: &str, name: &str) -> String {
    match path.rfind('.') {
        Some(cut) => format!("{}{name}", &path[..=cut]),
        None => name.to_string(),
    }
}

fn parent_of(path: &str) -> Option<String> {
    let cut = path.rfind('.')?;
    Some(path[..cut].trim_end_matches("[]").to_string())
}

/// The table, read once. A table that does not load stops everything that needs it: nothing is
/// restored by guesswork.
pub fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| Table::parse(TABLE_JSON).unwrap_or_else(|e| panic!("{e}")))
}

// ---- filtering a JSON document by its key table -----------------------------------------------------

/// Why a key was not brought back.
#[derive(Clone, Debug, PartialEq)]
pub enum Why {
    /// Not in the table: never restored.
    Unknown,
    /// In the table, as excluded.
    Excluded,
    /// It runs things and its tick was not ticked.
    NotTicked(RestoreClass),
    /// The value is not what the table allows there.
    BadValue(String),
    /// A secret whose address (see `KeyRow::goes_with`) is not brought back: it would arrive at no address, or at another one.
    KeyWithoutAddress,
}

/// What is said of a key that is left out because the address it was for is (see [`Why::KeyWithoutAddress`]).
pub const KEY_WITHOUT_ADDRESS: &str = "its address is not one that comes back (a name and password or a key in it, or it is not a web address), and a key goes only with the address it was kept for";

#[derive(Clone, Debug)]
pub struct Left {
    pub path: String,
    pub why: Why,
    pub row: Option<&'static KeyRow>,
}

#[derive(Clone, Debug)]
pub struct Kept {
    pub path: String,
    pub row: &'static KeyRow,
    pub value: Value,
    /// Which element of each list it sits in, outermost first (`providers[].baseUrl` of the third
    /// provider that comes back: `[2]`).
    pub at: Vec<usize>,
}

#[derive(Debug)]
pub struct Filtered {
    /// What is left of the document: only keys the table lets through.
    pub value: Value,
    /// The keys that were kept, with a value (a container is not listed, its keys are).
    pub kept: Vec<Kept>,
    /// The first keys that were not brought back.
    pub left: Vec<Left>,
    /// How many more were left that are not in `left`.
    pub left_more: usize,
}

impl Filtered {
    /// What to say when nothing of a document may come back: that a tick would bring something, that the document holds nothing the
    /// table brings back (only what it excludes, does not know or has a bad value for), or that it holds nothing at all.
    pub fn nothing_comes_back(&self) -> String {
        let left = self.left.iter().map(|l| l.path.as_str()).collect::<HashSet<_>>().len() + self.left_more;
        let s = if left == 1 { "" } else { "s" };
        if left == 0 {
            "it holds nothing that OAIY brings back".to_string()
        } else if self.left.iter().any(|l| matches!(l.why, Why::NotTicked(_))) {
            format!("nothing in it comes back without its tick ({left} setting{s} left out)")
        } else {
            format!("nothing in it is something OAIY brings back ({left} setting{s} left out)")
        }
    }
}

const MAX_LEFT: usize = 500;
const MAX_DEPTH: usize = 8;
const MAX_NODES: usize = 200_000;
const MAX_KEY_CHARS: usize = 128;

/// Text a person could have typed: no control characters but line breaks and tabs.
fn plain_text(text: &str) -> bool {
    !text.chars().any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
}

fn plain_url(text: &str, max_chars: usize) -> bool {
    let lower = text.get(..8).unwrap_or(text).to_ascii_lowercase();
    text.chars().count() <= max_chars && !text.chars().any(|c| c.is_control() || c.is_whitespace()) && (lower.starts_with("http://") || lower.starts_with("https://")) && text.len() > 8 && holds_no_credential(text)
}

/// The only parameters an address may carry in its query: the version of an API (an Azure address names one). Any other could be a key
/// (`api_key`, `key`, `token`, `sig`, `code`, or a name nobody has thought of): a credential belongs in the key of the provider, where
/// the keys box decides whether it travels.
pub const SAFE_QUERY_NAMES: [&str; 2] = ["api-version", "api_version"];

/// Whether an address holds no credential: no name or password before the host (`https://alice:hunter2@gw.example`), no fragment (an
/// empty one, a bare `#`, holds nothing: the page's rule is the same), and in its query nothing but the version of an API.
fn holds_no_credential(text: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(text) else { return false };
    url.username().is_empty() && url.password().is_none() && url.fragment().is_none_or(str::is_empty) && url.query_pairs().all(|(name, value)| safe_query_pair(&name, &value))
}

/// A parameter of a query that may stay: the version of an API, written as a version is (see [`is_api_version`]).
fn safe_query_pair(name: &str, value: &str) -> bool {
    SAFE_QUERY_NAMES.contains(&name.to_lowercase().as_str()) && is_api_version(value)
}

/// What the value of the version of an API may be, and nothing that has room for a key: a date (`2024-02-15`, with `-preview` after
/// it if it is one), or up to four numbers of up to four digits joined by dots (`1`, `2.1`, `1.0.3`). The Agent's page holds the same
/// rule (`SAFE_QUERY_VALUE` in `app/src/desktop/backup.ts`), and the two are tested against one list of addresses.
fn is_api_version(value: &str) -> bool {
    let digits = |part: &str, most: usize| (1..=most).contains(&part.len()) && part.bytes().all(|b| b.is_ascii_digit());
    let date = value.strip_suffix("-preview").unwrap_or(value).split('-').collect::<Vec<_>>();
    if date.len() == 3 && date[0].len() == 4 && date[1].len() == 2 && date[2].len() == 2 && date.iter().all(|part| digits(part, 4)) {
        return true;
    }
    let numbers = value.split('.').collect::<Vec<_>>();
    numbers.len() <= 4 && numbers.iter().all(|part| digits(part, 4))
}

/// An address without what an address must not hold (see [`holds_no_credential`]): the name and password, the fragment and every
/// parameter of the query but the version of an API are taken out, and the rest stays. `None` when there is nothing to take out (or
/// it is not an address).
pub fn address_without_credentials(text: &str) -> Option<String> {
    if holds_no_credential(text) {
        return None;
    }
    let mut url = reqwest::Url::parse(text).ok()?;
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_fragment(None);
    let kept: Vec<(String, String)> = url.query_pairs().filter(|(name, value)| safe_query_pair(name, value)).map(|(name, value)| (name.into_owned(), value.into_owned())).collect();
    if kept.is_empty() {
        url.set_query(None);
    } else {
        url.query_pairs_mut().clear().extend_pairs(kept);
    }
    Some(url.to_string())
}

/// Whether a string is something that must never travel: a key or a value sealed to a computer.
fn looks_secret(text: &str) -> bool {
    let lower = text.trim_start().to_lowercase();
    ["dpapi", "sealed:", "flk_", "sk-", "hf_", "bearer ", "age-secret-key", "-----begin"].iter().any(|p| lower.starts_with(p)) || lower.contains("dpapi:")
}

/// Check a scalar against its type. `Err` says what is wrong, in a few words. `exact`: the value is one the person's own page
/// held (an undo), where an address is put back as it was: an empty one means "none", and one that holds a name and password or a
/// parameter of its own is the person's own and comes back with them.
fn check_value(ty: &ValueType, value: &Value, is_a_key: bool, exact: bool) -> Result<(), String> {
    match (ty, value) {
        (ValueType::Bool, Value::Bool(_)) => Ok(()),
        (ValueType::Int { min, max }, Value::Number(n)) => match n.as_i64() {
            Some(i) if i >= *min && i <= *max => Ok(()),
            _ => Err(format!("a whole number from {min} to {max} is expected")),
        },
        (ValueType::Number { min, max }, Value::Number(n)) => match n.as_f64() {
            Some(f) if f.is_finite() && f >= *min && f <= *max => Ok(()),
            _ => Err(format!("a number from {min} to {max} is expected")),
        },
        (ValueType::Str { max_chars }, Value::String(s)) => {
            if s.chars().count() > *max_chars {
                Err(format!("longer than {max_chars} characters"))
            } else if !plain_text(s) {
                Err("it has control characters".to_string())
            } else if looks_secret(s) && !is_a_key {
                Err("it looks like a key or a sealed value".to_string())
            } else {
                Ok(())
            }
        }
        (ValueType::Enum(options), Value::String(s)) => if options.iter().any(|o| o == s) { Ok(()) } else { Err("it is not one of the choices".to_string()) },
        // An undo puts back the address the person's own page held, as it was: the empty one ("none"), one with a name and password or
        // a parameter of its own (`?tenant=acme`), one written without its scheme. What a backup may hold or a restore may write is
        // another matter (see `holds_no_credential`): here nothing is cleaned, and nothing the person had is refused for its shape.
        (ValueType::Url { max_chars }, Value::String(s)) if exact => {
            if s.chars().count() > *max_chars {
                Err(format!("longer than {max_chars} characters"))
            } else if s.chars().any(char::is_control) {
                Err("it has control characters".to_string())
            } else if looks_secret(s) {
                Err("it looks like a key or a sealed value".to_string())
            } else {
                Ok(())
            }
        }
        (ValueType::Url { max_chars }, Value::String(s)) => {
            if plain_url(s, *max_chars) && !looks_secret(s) {
                Ok(())
            } else {
                Err("it is not a plain web address (no name or password in it, and nothing in its query but the version of an API: a key belongs in the key)".to_string())
            }
        }
        (ValueType::Time, Value::String(s)) => if time_of_day(s).is_some() { Ok(()) } else { Err("a time such as 09:30 is expected".to_string()) },
        (ValueType::DateTime, Value::String(s)) => if chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M").is_ok() || chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").is_ok() { Ok(()) } else { Err("a date and time such as 2026-10-01T09:30 is expected".to_string()) },
        (ValueType::Stamp, Value::String(s)) => if s.len() <= 40 && chrono::DateTime::parse_from_rfc3339(s).is_ok() { Ok(()) } else { Err("a moment such as 2026-10-01T09:30:00Z is expected".to_string()) },
        (ValueType::Slug, Value::String(s)) => {
            if !s.is_empty() && s.len() <= 100 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')) {
                Ok(())
            } else {
                Err("an identifier of letters, digits, _, - and . is expected".to_string())
            }
        }
        (ValueType::CalendarId, Value::String(s)) => if crate::calendar::is_appointment_id(s) { Ok(()) } else { Err("an appointment id such as appt_ and 32 hexadecimal digits is expected".to_string()) },
        (ValueType::WeekHours, Value::Array(days)) => {
            let fine = days.len() == 7
                && days.iter().all(|day| {
                    day.as_array().is_some_and(|spans| {
                        spans.len() <= 8
                            && spans.iter().all(|span| match (span.get("open").and_then(Value::as_str).and_then(time_of_day), span.get("close").and_then(Value::as_str).and_then(time_of_day)) {
                                (Some(open), Some(close)) => open < close && span.as_object().is_some_and(|o| o.len() == 2),
                                _ => false,
                            })
                    })
                });
            if fine { Ok(()) } else { Err("seven days of opening times such as 09:00 to 17:00 are expected".to_string()) }
        }
        _ => Err("it is not the kind of value expected".to_string()),
    }
}

/// A time of day written `HH:MM` (two digits each), as minutes since midnight.
fn time_of_day(text: &str) -> Option<u32> {
    let bytes = text.as_bytes();
    if bytes.len() != 5 || bytes[2] != b':' || !bytes.iter().enumerate().all(|(i, b)| i == 2 || b.is_ascii_digit()) {
        return None;
    }
    let (h, m) = (text[..2].parse::<u32>().ok()?, text[3..].parse::<u32>().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

struct Walk<'a> {
    table: &'static KeyTable,
    /// The document is the person's own state, put back by an undo: an empty address is "none", not a bad address.
    exact: bool,
    at: Vec<usize>,
    keep: &'a dyn Fn(&KeyRow) -> bool,
    kept: Vec<Kept>,
    left: Vec<Left>,
    left_more: usize,
    nodes: usize,
}

impl Walk<'_> {
    fn leave(&mut self, path: &str, why: Why, row: Option<&'static KeyRow>) {
        if self.left.len() < MAX_LEFT {
            self.left.push(Left { path: path.to_string(), why, row });
        } else {
            self.left_more += 1;
        }
    }

    /// Whether the address (`partner`, beside a key in `node`) is there and is not one that comes back. No address, or an empty one, is
    /// not a refusal: the record is then for the vendor's own address, which is where its key belongs.
    fn address_refused(&self, node: &Map<String, Value>, path: &str, partner: &str) -> bool {
        let Some(address) = node.get(partner) else { return false };
        match address {
            Value::Null => return false,
            Value::String(s) if s.trim().is_empty() => return false,
            _ => {}
        }
        let sibling = if path.is_empty() { partner.to_string() } else { format!("{path}.{partner}") };
        match self.table.row(&sibling) {
            Some(row) if row.class != Class::Excluded => match &row.ty {
                Some(ty) => !(self.keep)(row) || check_value(ty, address, false, self.exact).is_err(),
                None => true,
            },
            _ => true,
        }
    }

    fn object(&mut self, node: &Map<String, Value>, path: &str, depth: usize) -> Map<String, Value> {
        let mut out = Map::new();
        for (key, value) in node {
            self.nodes += 1;
            // A key with nothing in it (a setting that was never set) says nothing.
            if value.is_null() {
                continue;
            }
            let child = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
            // A key that could pass for a path of its own is never looked up.
            if key.is_empty() || key.chars().count() > MAX_KEY_CHARS || key.contains(['.', '[', ']']) || key.chars().any(char::is_control) || self.nodes > MAX_NODES || depth > MAX_DEPTH {
                self.leave(&child, Why::Unknown, None);
                continue;
            }
            let table: &'static KeyTable = self.table;
            let Some(row) = table.row(&child) else {
                self.leave(&child, Why::Unknown, None);
                continue;
            };
            if row.class == Class::Excluded {
                self.leave(&child, Why::Excluded, Some(row));
                continue;
            }
            // A container is opened whatever its class says: the keys inside it decide what comes back.
            let container = matches!(row.ty, Some(ValueType::Object | ValueType::Objects { .. }));
            if !container && !(self.keep)(row) {
                self.leave(&child, Why::NotTicked(row.tick.unwrap_or(RestoreClass::Settings)), Some(row));
                continue;
            }
            let Some(ty) = &row.ty else {
                self.leave(&child, Why::Excluded, Some(row));
                continue;
            };
            // A key goes with the address it was kept for: where that address is not brought back, the key is not either (the record
            // would arrive without an address, and a provider without one is the vendor's own).
            if let Some(partner) = &row.goes_with {
                if self.address_refused(node, path, partner) {
                    self.leave(&child, Why::KeyWithoutAddress, Some(row));
                    continue;
                }
            }
            match ty {
                ValueType::Object => match value {
                    Value::Object(inner) => {
                        let filtered = self.object(inner, &child, depth + 1);
                        // A container of which nothing comes back is not there at all.
                        if !filtered.is_empty() {
                            out.insert(key.clone(), Value::Object(filtered));
                        }
                    }
                    _ => self.leave(&child, Why::BadValue("an object is expected".to_string()), Some(row)),
                },
                ValueType::Objects { max_items } => match value {
                    Value::Array(items) => {
                        let element = format!("{child}[]");
                        let mut list = Vec::new();
                        for (index, item) in items.iter().enumerate() {
                            if index >= *max_items {
                                self.left_more += items.len() - index;
                                self.leave(&child, Why::BadValue(format!("more than {max_items} entries: the rest are not brought back")), Some(row));
                                break;
                            }
                            match item {
                                Value::Object(inner) => {
                                    self.at.push(list.len());
                                    let filtered = self.object(inner, &element, depth + 1);
                                    self.at.pop();
                                    // An entry of which nothing comes back is not a record.
                                    if !filtered.is_empty() {
                                        list.push(Value::Object(filtered));
                                    }
                                }
                                _ => self.leave(&element, Why::BadValue("an object is expected".to_string()), Some(row)),
                            }
                        }
                        if !list.is_empty() {
                            out.insert(key.clone(), Value::Array(list));
                        }
                    }
                    _ => self.leave(&child, Why::BadValue("a list is expected".to_string()), Some(row)),
                },
                ValueType::Strings { max_items, max_chars } => match value {
                    Value::Array(items) => {
                        let mut list = Vec::new();
                        for item in items.iter().take(*max_items) {
                            match item {
                                Value::String(s) if s.chars().count() <= *max_chars && plain_text(s) && !looks_secret(s) => list.push(item.clone()),
                                _ => self.leave(&child, Why::BadValue("an entry is not a short plain string".to_string()), Some(row)),
                            }
                        }
                        if items.len() > *max_items {
                            self.leave(&child, Why::BadValue(format!("more than {max_items} entries: the rest are not brought back")), Some(row));
                        }
                        let value = Value::Array(list);
                        self.kept.push(Kept { path: child.clone(), row, value: value.clone(), at: self.at.clone() });
                        out.insert(key.clone(), value);
                    }
                    _ => self.leave(&child, Why::BadValue("a list is expected".to_string()), Some(row)),
                },
                ValueType::StringsMap { max_items, max_chars } | ValueType::ScalarsMap { max_items, max_chars } => match value {
                    Value::Object(entries) => {
                        let scalars = matches!(ty, ValueType::ScalarsMap { .. });
                        let mut map = Map::new();
                        for (k, v) in entries.iter().take(*max_items) {
                            let fine = k.chars().count() <= MAX_KEY_CHARS && !k.chars().any(char::is_control);
                            let ok = match v {
                                Value::String(s) => s.chars().count() <= *max_chars && plain_text(s) && !looks_secret(s),
                                Value::Number(_) | Value::Bool(_) => scalars,
                                _ => false,
                            };
                            if fine && ok {
                                map.insert(k.clone(), v.clone());
                            } else {
                                self.leave(&format!("{child}.{k}"), Why::BadValue("not a short plain value".to_string()), Some(row));
                            }
                        }
                        if entries.len() > *max_items {
                            self.leave(&child, Why::BadValue(format!("more than {max_items} entries: the rest are not brought back")), Some(row));
                        }
                        let value = Value::Object(map);
                        self.kept.push(Kept { path: child.clone(), row, value: value.clone(), at: self.at.clone() });
                        out.insert(key.clone(), value);
                    }
                    _ => self.leave(&child, Why::BadValue("an object is expected".to_string()), Some(row)),
                },
                scalar => match check_value(scalar, value, row.secret, self.exact) {
                    Ok(()) => {
                        self.kept.push(Kept { path: child.clone(), row, value: value.clone(), at: self.at.clone() });
                        out.insert(key.clone(), value.clone());
                    }
                    Err(why) => self.leave(&child, Why::BadValue(why), Some(row)),
                },
            }
        }
        out
    }
}

/// Bring a JSON document through its key table: only keys that have a row that comes back survive, and
/// only with a value of the kind the row allows. `keep` says which rows that come back are wanted (all of
/// them when a backup is made; the ticked ones when one is restored).
pub fn filter_json(table: &'static KeyTable, input: &Value, keep: &dyn Fn(&KeyRow) -> bool) -> Filtered {
    filter_document(table, input, keep, false)
}

/// The same for the person's own state, as an undo puts it back: a value that is empty ("" for an address that was not set) is
/// carried like any other, so that what an undo restores is what there was and not what is left when the empty ones are dropped.
pub fn filter_json_exact(table: &'static KeyTable, input: &Value, keep: &dyn Fn(&KeyRow) -> bool) -> Filtered {
    filter_document(table, input, keep, true)
}

fn filter_document(table: &'static KeyTable, input: &Value, keep: &dyn Fn(&KeyRow) -> bool, exact: bool) -> Filtered {
    let mut walk = Walk { table, exact, at: Vec::new(), keep, kept: Vec::new(), left: Vec::new(), left_more: 0, nodes: 0 };
    let Value::Object(root) = input else {
        walk.leave("", Why::BadValue("it is not a JSON object".to_string()), None);
        return Filtered { value: Value::Object(Map::new()), kept: walk.kept, left: walk.left, left_more: walk.left_more };
    };
    let value = Value::Object(walk.object(root, "", 0));
    Filtered { value, kept: walk.kept, left: walk.left, left_more: walk.left_more }
}

/// The lines of a list kept one a line (numbers and the like), in the way two lists are joined:
/// `local` first, then what `backup` has that `local` has not (numbers are the same when their last nine
/// digits are).
pub fn union_lines(local: &str, backup: &str, max_chars: usize) -> String {
    fn key(line: &str) -> String {
        let digits: String = line.chars().filter(char::is_ascii_digit).collect();
        if digits.len() >= 6 {
            digits[digits.len().saturating_sub(9)..].to_string()
        } else {
            line.trim().to_lowercase()
        }
    }
    let lines = |text: &str| -> Vec<String> { text.split(['\n', '\r', ',']).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect() };
    let mut out: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    for line in lines(local).into_iter().chain(lines(backup)) {
        if seen.insert(key(&line)) {
            out.push(line);
        }
    }
    let mut joined = String::new();
    // What is already there is kept whole; what is added stops where the limit is.
    let local_count = lines(local).iter().map(|l| key(l)).collect::<BTreeSet<_>>().len();
    for (i, line) in out.iter().enumerate() {
        let next = if joined.is_empty() { line.clone() } else { format!("{joined}\n{line}") };
        if next.chars().count() > max_chars && i >= local_count {
            break;
        }
        joined = next;
    }
    joined
}

/// A document with what a backup brings put into what is here: for each kept key the backup's value
/// replaces the local one (a list kept one a line is joined instead), and everything the table did not
/// let through stays exactly as it is here.
pub fn merge_into_local(local: Option<&Value>, filtered: &Filtered) -> Value {
    let mut result = match local {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        _ => Value::Object(Map::new()),
    };
    for kept in &filtered.kept {
        if kept.path.contains("[]") {
            continue;
        }
        let parts: Vec<&str> = kept.path.split('.').collect();
        let mut node = &mut result;
        for part in &parts[..parts.len() - 1] {
            if !node.is_object() {
                *node = Value::Object(Map::new());
            }
            node = node.as_object_mut().expect("an object").entry((*part).to_string()).or_insert_with(|| Value::Object(Map::new()));
        }
        if !node.is_object() {
            *node = Value::Object(Map::new());
        }
        let last = parts[parts.len() - 1];
        let map = node.as_object_mut().expect("an object");
        let value = match (kept.row.merge.as_deref(), map.get(last), &kept.value) {
            (Some("union-lines"), Some(Value::String(here)), Value::String(theirs)) => {
                let max = match &kept.row.ty {
                    Some(ValueType::Str { max_chars }) => *max_chars,
                    _ => 4000,
                };
                Value::String(union_lines(here, theirs, max))
            }
            _ => kept.value.clone(),
        };
        map.insert(last.to_string(), value);
    }
    result
}

// ---- the documentation, generated from the table ---------------------------------------------------

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}

fn comes_back(class: Class, tick: Option<RestoreClass>, merge: Option<&str>, keyed: bool) -> String {
    let base = match class {
        Class::Excluded => "Never".to_string(),
        Class::Data => "Yes, without a tick".to_string(),
        Class::Runs if keyed => "Key by key (see the table of its keys)".to_string(),
        Class::Runs => format!("Only with the tick \"{}\"", tick.map(|t| t.label()).unwrap_or("?")),
    };
    let how = match merge {
        Some("union") => "; added to yours, none of yours is ever taken away",
        Some("union-lines") => "; joined to yours, none of yours is ever taken away",
        Some("campaign") => "; as a paused campaign, never running",
        Some("campaign-index") => "; the list of campaigns, without one that is running",
        _ => "",
    };
    format!("{base}{how}")
}

/// What a thing is and why it is classed so, with what to do again: the reason alone when it already starts with the name.
fn why_cell(what: &str, reason: &str, redo: Option<&str>) -> String {
    let mut text = if reason.is_empty() {
        cell(what)
    } else if reason.to_lowercase().starts_with(&what.to_lowercase().chars().take(24).collect::<String>()) {
        cell(reason)
    } else {
        format!("{}: {}", cell(what), cell(reason))
    };
    if let Some(redo) = redo {
        text.push_str(&format!(" To do again: {}", cell(redo)));
    }
    text
}

/// A markdown block found between `<!-- BEGIN GENERATED: name ... -->` and `<!-- END GENERATED: name -->`
/// in `doc`, replaced by `block` (the markers stay). `None` when the markers are not there.
pub fn splice_generated(doc: &str, name: &str, block: &str) -> Option<String> {
    let begin = format!("<!-- BEGIN GENERATED: {name}");
    let end = format!("<!-- END GENERATED: {name} -->");
    let start = doc.find(&begin)?;
    let start_line_end = start + doc[start..].find("-->")? + 3;
    let stop = doc.find(&end)?;
    if stop < start_line_end {
        return None;
    }
    Some(format!("{}\n\n{}\n\n{}", &doc[..start_line_end], block.trim_end(), &doc[stop..]))
}

impl Table {
    /// The kinds a restore asks you to tick, from the classes and the rows that use them.
    pub fn render_kinds(&self) -> String {
        let mut out = String::from("| Kind (tick) | What it holds | Why it needs a tick |\n|---|---|---|\n");
        for class in RestoreClass::ALL {
            let mut holds: Vec<String> = Vec::new();
            for row in self.desktop.iter().chain(&self.agent) {
                if row.class == Class::Runs && row.tick == Some(class) && row.keys.is_none() && !holds.contains(&row.what) {
                    holds.push(row.what.clone());
                }
            }
            for table in self.key_tables.values() {
                if table.keys.iter().any(|k| k.class == Class::Runs && k.tick == Some(class) && !matches!(k.ty, Some(ValueType::Object | ValueType::Objects { .. }))) {
                    let what = format!("Some keys of {}", table.file);
                    if !holds.contains(&what) {
                        holds.push(what);
                    }
                }
            }
            out.push_str(&format!("| {} | {} | {} |\n", cell(class.label()), cell(&holds.join("; ")), cell(class.description())));
        }
        out
    }

    /// The audit: for everything that says what reads it, what reads it, where, the class and why, as markdown.
    pub fn render_audit(&self) -> String {
        let mut out = String::from("| Item | Class | What reads it | Where | Why it is classed so |\n|---|---|---|---|---|\n");
        let readers = |list: &[String]| if list.is_empty() { "-".to_string() } else { list.iter().map(|r| format!("`{}`", cell(r))).collect::<Vec<_>>().join(", ") };
        for row in self.desktop.iter().chain(&self.agent).filter(|r| r.reads.is_some()) {
            out.push_str(&format!("| `{}` | {} | {} | {} | {} |\n", cell(&row.pattern), row.class.id(), cell(row.reads.as_deref().unwrap_or("")), readers(&row.readers), why_cell(&row.what, &row.reason, None)));
        }
        for table in self.key_tables.values() {
            for key in table.keys.iter().filter(|k| k.reads.is_some()) {
                out.push_str(&format!("| `{}`: `{}` | {} | {} | {} | {} |\n", cell(&table.file), cell(&key.path), key.class.id(), cell(key.reads.as_deref().unwrap_or("")), readers(&key.readers), why_cell(&key.what, &key.reason, None)));
            }
        }
        for note in &self.audit {
            out.push_str(&format!("| {} | {} | {} | {} | {} |\n", cell(&note.item), cell(&note.class), cell(&note.reads), readers(&note.readers), cell(&note.why)));
        }
        out
    }

    /// Every path, name and key the table classifies, as markdown.
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str("#### Files in OAIY's data folder\n\nA path is matched without regard to case. The first row that matches counts; a path that no row matches is **not restored** (the dry run says \"not restored: unknown item\").\n\n");
        out.push_str("| Path | Class | Comes back | What it is, why, and what to do again |\n|---|---|---|---|\n");
        for row in &self.desktop {
            let what = why_cell(&row.what, &row.reason, row.redo.as_deref());
            let comes = if row.skip_when_keys { "Never, unless the keys box is ticked when the backup is made".to_string() } else { comes_back(row.class, row.tick, row.merge.as_deref(), row.keys.is_some()) };
            out.push_str(&format!("| `{}` | {} | {} | {} |\n", cell(&row.pattern), row.class.id(), cell(&comes), what));
        }
        out.push_str("\n#### The Agent's storage\n\nThese are names inside the archive the Agent's page makes of its browser storage (the private file system, and IndexedDB as `idb/settings.json`). Names are matched exactly. A name that no row matches is **not restored**.\n\n");
        out.push_str("| Name | Class | Comes back | What it is, why, and what to do again |\n|---|---|---|---|\n");
        for row in self.agent.iter().filter(|r| !r.quiet) {
            let what = why_cell(&row.what, &row.reason, row.redo.as_deref());
            out.push_str(&format!("| `{}` | {} | {} | {} |\n", cell(&row.pattern), row.class.id(), cell(&comes_back(row.class, row.tick, row.merge.as_deref(), row.keys.is_some())), what));
        }
        for table in self.key_tables.values() {
            out.push_str(&format!("\n#### Keys of `{}` ({})\n\n{}\n\n", table.file, table.name, cell(&table.about)));
            out.push_str("| Key | Class | Comes back | What it is, why, and what to do again |\n|---|---|---|---|\n");
            for key in &table.keys {
                let what = why_cell(&key.what, &key.reason, key.redo.as_deref());
                out.push_str(&format!("| `{}` | {} | {} | {} |\n", cell(&key.path), key.class.id(), cell(&comes_back(key.class, key.tick, key.merge.as_deref(), false)), what));
            }
        }
        out
    }
}
