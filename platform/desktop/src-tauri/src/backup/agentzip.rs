//! The Agent's storage archive: what a dry run says it holds, and the archive the page is given.
//!
//! The Agent's page keeps its conversations, projects, brief, knowledge files, campaigns and settings in its
//! own browser storage, and hands them to a backup as one ZIP (an entry of the backup, `agent/agent-storage.zip`).
//! A restore does not hand that ZIP back. It goes through the classification table (see [`super::table`]) here:
//!
//! - **Looking** ([`describe`]) reads the ZIP's own directory, not what the backup says of it, and lists what
//!   would come back by name and size: each project, the front desk's brief, each knowledge file, what is
//!   remembered about people, each outreach campaign (by name, with how many people it would contact), and the
//!   Agent's settings by key and value; and what is not restored, and why.
//! - **Preparing** ([`filter`]) writes a new ZIP of only what is to come back: what is data, and what runs
//!   things if its kind was ticked, and nothing the table does not know. An outreach campaign is rebuilt from
//!   the keys the table lets through and comes back paused, with anyone who was in the middle of being reached
//!   set aside; the list of numbers not to be contacted is checked and marked to be added to the one that is
//!   there, never to replace it; the settings are cut to the keys that may come back. The archive's own record,
//!   `agent-manifest.json`, is written again and names every item and how the page is to bring it back. The
//!   page brings back nothing that record does not name.
//!
//! Everything is read a piece at a time and within limits, so a hostile archive costs a bounded amount of
//! memory and time: a count of entries, a size for each and for all, and a small size for the files that are
//! read to be described or rebuilt.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::container::{self, Archive};
use super::review::{clip, key_item, show_value, NotRestored, RestoreClass, ReviewItem, Ticks};
use super::table::{filter_json, filter_json_exact, table, Class, KeyRow, Row, ValueType, Why};
use super::{BackupError, Budget, ErrorKind, Limits, Result};
use crate::secret_file;

/// The most of the brief or a project's record that is read to be quoted: 64 KiB.
const MAX_QUOTE_BYTES: u64 = 64 << 10;
/// The most items listed by name for one kind of file before the rest are counted.
const MAX_NAMED: usize = 300;
/// The most numbers not to be contacted that one restore brings, each once (the page adds them to the list it has, and holds to the
/// same bound on the numbers it ADDS). A person's list of opt-outs can run to tens of thousands: a restore that could not take it all
/// would leave someone who opted out to be contacted, so the bound is a defence against a hostile file, not a limit a real list meets.
pub const MAX_DO_NOT_CONTACT: usize = 50_000;

/// A name inside the Agent's archive that is a plain relative path.
pub fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 1024
        && !name.starts_with('/')
        && !name.contains('\\')
        && !name.contains(':')
        && !name.chars().any(char::is_control)
        && !name.split('/').any(|s| s.is_empty() || s == "." || s == ".." || s.len() > 255)
}

/// One file of the archive, from its directory.
#[derive(Clone, Debug)]
pub struct Entry {
    pub index: usize,
    pub name: String,
    pub size: u64,
    /// What the table says of it: `None` is not in the table (never restored).
    pub row: Option<&'static Row>,
    /// Why it cannot be looked at or brought back whatever the table says (an unsafe name, too large).
    pub unfit: Option<&'static str>,
}

/// What the archive's directory says it holds.
#[derive(Debug, Default)]
pub struct Listing {
    pub entries: Vec<Entry>,
}

impl Listing {
    pub fn total(&self) -> u64 {
        self.entries.iter().map(|e| e.size).sum()
    }
}

/// Copy the archive out of the backup into `dest` (a new private file), streaming.
pub fn extract(outer: &mut Archive, dest: &Path, budget: &Budget) -> Result<()> {
    let mut entry = outer.by_name(super::AGENT_ENTRY).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage is not in this backup."))?;
    let out = secret_file::create_new_owner_only(dest).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
    let mut out = BufWriter::with_capacity(64 << 10, out);
    let mut buf = vec![0u8; 64 << 10];
    loop {
        budget.check()?;
        let n = entry.read(&mut buf).map_err(|_| BackupError::new(ErrorKind::Damaged, "This file is not an OAIY backup, or it is damaged or incomplete."))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
    }
    out.flush().map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))
}

fn open(path: &Path, limits: &Limits) -> Result<Archive> {
    // How many entries it claims is read before its directory is parsed.
    let directory = container::peek_zip_directory(path)?;
    if directory.entries > limits.max_agent_entries as u64 + 1 || directory.bytes > (limits.max_agent_entries as u64 + 1) * container::ENTRY_RECORD_MAX {
        return Err(BackupError::new(ErrorKind::TooLarge, "The Agent's storage in this backup holds more files than OAIY will bring back."));
    }
    container::open_archive(path)
}

/// The names of the files an Agent archive holds, in the order of its directory (its directory only is read), or none when it cannot
/// be read.
pub(crate) fn file_names(path: &Path) -> Option<Vec<String>> {
    let mut archive = open(path, &Limits::default()).ok()?;
    let mut names = Vec::new();
    for index in 0..archive.len() {
        let file = archive.by_index_raw(index).ok()?;
        if !(file.is_dir() || file.name().ends_with('/')) {
            names.push(file.name().to_string());
        }
    }
    Some(names)
}

/// Read the archive's directory (never its contents): each file's name and its declared size, and the table's
/// answer for it. A backup whose archive lists a name twice, or more than the limits allow, is refused.
pub fn read_listing(path: &Path, limits: &Limits) -> Result<Listing> {
    let mut archive = open(path, limits)?;
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut total = 0u64;
    for index in 0..archive.len() {
        let file = archive.by_index_raw(index).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage in this backup is damaged."))?;
        if file.is_dir() || file.name().ends_with('/') {
            continue;
        }
        let name = file.name().to_string();
        let size = file.size();
        if !seen.insert(name.clone()) {
            return Err(BackupError::new(ErrorKind::Unsafe, "This backup is refused: the Agent's storage in it lists the same file twice."));
        }
        total = total.saturating_add(size);
        if total > limits.max_agent_total_bytes {
            return Err(BackupError::new(ErrorKind::TooLarge, "The Agent's storage in this backup is larger than OAIY will bring back."));
        }
        let unfit = if !safe_name(&name) {
            Some("its name is not a plain path")
        } else if size > limits.max_agent_file_bytes {
            Some("it is larger than the Agent's own export ever makes a file")
        } else {
            None
        };
        let row = if unfit.is_some() { None } else { table().agent_row(&name) };
        entries.push(Entry { index, name, size, row, unfit });
    }
    Ok(Listing { entries })
}

/// One file's bytes when it is no larger than `max`, checked against what its directory says (`None`: too large).
fn read_small(archive: &mut Archive, entry: &Entry, max: u64) -> Result<Option<Vec<u8>>> {
    if entry.size > max {
        return Ok(None);
    }
    let file = archive.by_index(entry.index).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage in this backup is damaged."))?;
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage in this backup is damaged."))?;
    // The directory said how large it was: a file that is another size is not what it says.
    if bytes.len() as u64 != entry.size {
        return Err(BackupError::new(ErrorKind::Damaged, "A file in the Agent's storage is not the size its record says."));
    }
    Ok(Some(bytes))
}

fn kb(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} bytes")
    } else if bytes < 10 << 20 {
        format!("{} KB", bytes.div_ceil(1024))
    } else {
        format!("{} MB", bytes >> 20)
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)
}

// ---- outreach campaigns ------------------------------------------------------------------------------

/// A campaign as it comes back, and what was changed or left out of it on the way.
pub struct Rebuilt {
    pub campaign: Value,
    /// People whose number is not a full phone number (E.164, `+` and 7 to 15 digits), which the Agent never writes: left out.
    pub bad_numbers: usize,
    /// People set aside at planning whose reason was not one the Agent gives: it comes back as "other" (a reason is said to the Agent as it is).
    pub other_reasons: usize,
    /// Who the backup said started it (the project's id and name): a restored campaign is started by the front desk, whoever it says.
    pub started_by: Option<(String, String)>,
}

/// A full phone number as the Agent writes it into a campaign: `+`, then 7 to 15 digits, not starting with 0.
fn full_number(number: &str) -> bool {
    number.strip_prefix('+').is_some_and(|d| (7..=15).contains(&d.len()) && d.starts_with(|c: char| ('1'..='9').contains(&c)) && d.chars().all(|c| c.is_ascii_digit()))
}

/// A campaign as it comes back, rebuilt from what the table lets through: paused, nothing scheduled, and nobody
/// who was in the middle of being reached is contacted again. A campaign that had finished (`was` says `done` or
/// `stopped`) and has nobody left to reach stays finished, since that is the truth and nothing can run from it.
/// It is started by the front desk, whoever the backup says started it (where a campaign reports to, and what it is
/// then told to do, follows from who started it), and only what the Agent itself writes into a campaign is kept: a
/// person whose number is not a full phone number is left out, and the number as it was given is not carried.
/// `Err` says why it cannot come back.
pub fn rebuild_campaign(found: &Value, was: Option<&str>) -> std::result::Result<Rebuilt, String> {
    let Value::Object(c) = found else { return Err("it is not a campaign".to_string()) };
    let text = |key: &str, default: &str| c.get(key).and_then(Value::as_str).unwrap_or(default).to_string();
    let id = text("id", "");
    if id.is_empty() || id.len() > 128 || !id.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_') {
        return Err("it has no usable id".to_string());
    }
    let kind = text("kind", "");
    if kind != "call" && kind != "text" {
        return Err("it is neither a text campaign nor a call campaign".to_string());
    }
    // Where a campaign writes its results is part of the front desk's files: a name for its own folder, and a path inside
    // `/outreach/`, so that a campaign can never be pointed at the brief or at a knowledge file.
    let plain = |s: &str| !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    let slug = {
        let given = text("slug", "");
        if plain(&given) {
            given
        } else {
            id.chars().take(64).collect::<String>()
        }
    };
    let results_path = {
        let given = text("resultsPath", "");
        let inside = given.starts_with("/outreach/") && given.ends_with(".md") && given.len() <= 512 && !given.split('/').any(|s| s == ".." || s == ".") && !given.contains('\\') && !given.chars().any(char::is_control);
        if inside {
            given
        } else {
            format!("/outreach/{slug}/results.md")
        }
    };
    let mut people = Vec::new();
    let mut bad_numbers = 0usize;
    for p in c.get("people").and_then(Value::as_array).into_iter().flatten() {
        let Value::Object(p) = p else { continue };
        let number = p.get("number").and_then(Value::as_str).unwrap_or("").to_string();
        if number.is_empty() {
            continue;
        }
        if !full_number(&number) {
            bad_numbers += 1;
            continue;
        }
        let get = |key: &str| p.get(key).cloned();
        let state = p.get("state").and_then(Value::as_str).unwrap_or("queued");
        let mut person = serde_json::Map::new();
        person.insert("id".into(), Value::String(p.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("p{}", people.len() + 1))));
        person.insert("name".into(), Value::String(p.get("name").and_then(Value::as_str).unwrap_or("").to_string()));
        person.insert("number".into(), Value::String(number.clone()));
        // (The number as it was given is not carried: the number is what is called.)
        person.insert("raw".into(), Value::String(number.clone()));
        if let Some(notes) = get("notes") {
            person.insert("notes".into(), notes);
        }
        // Details are named as the Agent names them: a letter or underscore, then letters, digits or underscores.
        let fields: serde_json::Map<String, Value> = get("fields")
            .and_then(|f| f.as_object().cloned())
            .map(|f| f.into_iter().filter(|(k, _)| k.len() <= 32 && k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).collect())
            .unwrap_or_default();
        person.insert("fields".into(), Value::Object(fields));
        person.insert("tries".into(), json!(0));
        person.insert("nextAt".into(), json!(0));
        person.insert("history".into(), json!([]));
        match state {
            // Finished people keep what they said.
            "done" | "skipped" => {
                person.insert("state".into(), Value::String(state.to_string()));
                for key in ["outcome", "summary", "doneAt", "why"] {
                    if let Some(v) = get(key) {
                        person.insert(key.to_string(), v);
                    }
                }
                person.insert("answers".into(), get("answers").filter(Value::is_object).unwrap_or_else(|| json!({})));
            }
            // Not yet reached: waits, and is reached only when the campaign is started.
            "queued" => {
                person.insert("state".into(), json!("queued"));
                person.insert("answers".into(), json!({}));
            }
            // In the middle of being reached, or waiting for a reply: OAIY cannot tell whether they were reached, so
            // a restore does not do it again.
            _ => {
                person.insert("state".into(), json!("skipped"));
                person.insert("outcome".into(), json!("skipped"));
                person.insert("why".into(), json!("They were being reached when the backup was made. A restore does not call or text anyone again, so they were set aside."));
                person.insert("answers".into(), json!({}));
            }
        }
        people.push(Value::Object(person));
    }
    let mut out = serde_json::Map::new();
    out.insert("id".into(), json!(id));
    out.insert("kind".into(), json!(kind));
    out.insert("name".into(), json!(text("name", &id)));
    out.insert("slug".into(), json!(slug));
    out.insert("objective".into(), json!(text("objective", "")));
    out.insert("collect".into(), c.get("collect").filter(|v| v.is_array()).cloned().unwrap_or_else(|| json!([])));
    out.insert("openingLine".into(), json!(text("openingLine", "")));
    out.insert("textTemplate".into(), json!(text("textTemplate", "")));
    out.insert("voicemail".into(), json!(if text("voicemail", "") == "leave_message" { "leave_message" } else { "no_message" }));
    out.insert("voicemailMessage".into(), json!(text("voicemailMessage", "")));
    out.insert("retries".into(), c.get("retries").filter(|v| v.is_object()).cloned().unwrap_or_else(|| json!({ "times": 2, "gapMinutes": 60 })));
    out.insert("replyDeadlineHours".into(), c.get("replyDeadlineHours").cloned().unwrap_or_else(|| json!(48)));
    out.insert("window".into(), c.get("window").filter(|v| v.is_object()).cloned().unwrap_or_else(|| json!({ "from": "09:00", "to": "18:00" })));
    out.insert("afterwards".into(), json!(text("afterwards", "")));
    let started_by = c.get("origin").and_then(|o| Some((o.get("projectId")?.as_str()?.to_string(), o.get("projectName").and_then(Value::as_str).unwrap_or("").to_string()))).filter(|(id, _)| id != "front-desk");
    out.insert("origin".into(), json!({ "kind": "runner", "projectId": "front-desk", "projectName": "Front desk" }));
    out.insert("resultsPath".into(), json!(results_path));
    if let Some(identity) = c.get("identity").filter(|v| v.is_object()) {
        out.insert("identity".into(), identity.clone());
    }
    out.insert("createdAt".into(), c.get("createdAt").cloned().unwrap_or_else(|| json!(0)));
    // Never running, and nothing scheduled: it is paused, and the person starts it (unless it had finished and has no one left to reach).
    let people_left = people.iter().any(|x| x.get("state").and_then(Value::as_str) == Some("queued"));
    let finished = matches!(was, Some("done" | "stopped")) && !people_left;
    out.insert("state".into(), json!(if finished { was.unwrap_or("done") } else { "paused" }));
    out.insert("pausedWhy".into(), json!(if finished { "Restored from a backup." } else { "Restored from a backup. Nothing is sent or called until you start it." }));
    out.insert("waitingFor".into(), json!(""));
    out.insert("faults".into(), json!(0));
    out.insert("approvedAt".into(), json!(0));
    let ended = people.iter().filter_map(|x| x.get("doneAt").and_then(Value::as_f64)).fold(0.0f64, f64::max);
    out.insert("endedAt".into(), if finished { json!(if ended > 0.0 { ended } else { c.get("createdAt").and_then(Value::as_f64).unwrap_or(0.0) }) } else { Value::Null });
    out.insert("report".into(), json!({ "text": "", "pending": false, "delivered": true }));
    out.insert("lines".into(), json!([]));
    out.insert("people".into(), Value::Array(people));
    // The people set aside at planning come back with one of the reasons the Agent gives (the table's list) and never other words: the
    // Agent's report of a campaign says these reasons to it as they are. Any other text, or none, is "other", and is counted.
    let reasons = planner_reasons();
    let mut other_reasons = 0usize;
    let skipped: Vec<Value> = c
        .get("skipped")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|s| {
            let mut s = s.clone();
            if let Value::Object(entry) = &mut s {
                if !entry.get("why").and_then(Value::as_str).is_some_and(|w| reasons.iter().any(|r| r == w)) {
                    entry.insert("why".to_string(), json!(REASON_OTHER));
                    other_reasons += 1;
                }
            }
            s
        })
        .collect();
    out.insert("skipped".into(), Value::Array(skipped));
    Ok(Rebuilt { campaign: Value::Object(out), bad_numbers, other_reasons, started_by })
}

/// What a reason that is not one the Agent gives comes back as (it is one of the table's reasons, so a restored campaign restores again).
const REASON_OTHER: &str = "other";

/// What is said of the reasons that came back as "other" (at a restore, and in the dry run).
fn other_reasons_said(n: usize) -> String {
    format!("{} for setting {} aside at planning {} not {} the Agent gives, so {} back as \"{REASON_OTHER}\".", plural(n, "reason", "reasons"), if n == 1 { "someone" } else { "people" }, if n == 1 { "was" } else { "were" }, if n == 1 { "one" } else { "ones" }, if n == 1 { "it comes" } else { "they come" })
}

/// The reasons the Agent gives for setting a person aside when it plans a campaign, as the table lists them (`skipped[].why`).
fn planner_reasons() -> &'static [String] {
    match table().key_table("agent.campaign").and_then(|kt| kt.row("skipped[].why")).and_then(|row| row.ty.as_ref()) {
        Some(ValueType::Enum(options)) => options,
        _ => &[],
    }
}

/// (For the tests: how a campaign is described.)
#[cfg(test)]
pub fn describe_campaign_for_test(kept: &super::table::Filtered, rebuilt: &Rebuilt, was: Option<&str>) -> String {
    describe_campaign(kept, rebuilt, was)
}

/// The most people of a campaign the dry run names one by one, with what a model reads of each (the rest are counted), and the most
/// people skipped at planning it does the same for.
const MAX_PEOPLE_LISTED: usize = 10;

/// The most that is said of one person or of one person who was skipped (each value is cut, with how long it is; this is the whole).
const MAX_PERSON_TEXT: usize = 700;

/// The most a campaign's description says (a campaign's own words come to a few thousand characters, and ten people to seven thousand).
pub(crate) const MAX_CAMPAIGN_TEXT: usize = 12_000;

/// What a rebuilt person holds that is not something a model reads or that is not said of them another way (their number is what
/// names them): not listed as their words.
const PERSON_KEYS_NOT_LISTED: [&str; 8] = ["id", "raw", "number", "state", "tries", "nextAt", "history", "doneAt"];

/// The keys of a campaign that are only names for it, or that the restore makes again: not listed as something it says.
const CAMPAIGN_KEYS_NOT_LISTED: [&str; 5] = ["id", "slug", "createdAt", "resultsPath", "name"];

/// A key of a campaign as a person reads it: `collect[2].question` for the second question.
fn campaign_key(kept: &super::table::Kept) -> String {
    let mut at = kept.at.iter();
    kept.path.split("[]").enumerate().map(|(i, part)| if i == 0 { part.to_string() } else { format!("[{}]{part}", at.next().map(|n| n + 1).unwrap_or(0)) }).collect()
}

/// What a model reads of one person (or one person skipped at planning), by value: their name, notes and details, and for one who
/// was done how it ended, what they said and what they answered. Every key of the person that is not an identifier or run state is
/// said, so a key the rebuild carries is listed by construction (the test builds a person from the table and looks for each).
fn person_words(person: &Value) -> String {
    let Some(map) = person.as_object() else { return String::new() };
    let mut parts = Vec::new();
    for (key, value) in map {
        if PERSON_KEYS_NOT_LISTED.contains(&key.as_str()) {
            continue;
        }
        let shown = match value {
            Value::String(s) if s.is_empty() => continue,
            Value::Object(m) if m.is_empty() => continue,
            Value::Object(m) => {
                let more = m.len().saturating_sub(5);
                format!("{}{}", m.iter().take(5).map(|(k, v)| format!("{} = {}", clip(k, 40), show_value(v))).collect::<Vec<_>>().join(", "), if more > 0 { format!(" and {more} more") } else { String::new() })
            }
            other => show_value(other),
        };
        let label = match key.as_str() {
            "fields" => "details",
            other => other,
        };
        parts.push(format!("{label} {shown}"));
    }
    parts.join("; ")
}

/// How a rebuilt campaign is described: by name, with how many people it would contact, and then EVERY key of it that acts, by
/// its value (cut, with how long it is): what it says to them (the objective, the text, the opening line, the questions, the
/// voicemail), what it does afterwards, who it speaks as, who started it, when it tries and how often. It is made from the key
/// table, so a key added to the table is listed by construction. The people follow: for the first few, everything a model reads of
/// them (their name, notes and details, and for one who was done how it ended, what they said and answered), and for the people
/// skipped at planning their name and why; the rest are counted.
fn describe_campaign(kept: &super::table::Filtered, rebuilt: &Rebuilt, was: Option<&str>) -> String {
    let campaign = &rebuilt.campaign;
    let people = campaign.get("people").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let count = |state: &str| people.iter().filter(|p| p.get("state").and_then(Value::as_str) == Some(state)).count();
    let (queued, done) = (count("queued"), count("done"));
    let set_aside = count("skipped");
    let kind = if campaign.get("kind").and_then(Value::as_str) == Some("text") { "text messages" } else { "phone calls" };
    let mut what = format!(
        "{kind} to {} ({queued} not yet contacted, {done} finished, {set_aside} set aside).",
        plural(people.len(), "person", "people")
    );
    // What comes of the campaign, before anything long: the description is cut at the end.
    match was {
        Some("running") => what.push_str(" It was RUNNING when the backup was made; it comes back PAUSED, and nothing is sent or called until you start it."),
        _ if campaign.get("state").and_then(Value::as_str) != Some("paused") => what.push_str(" It had finished, and comes back as it was: nothing more is sent or called."),
        _ => what.push_str(" It comes back PAUSED: nothing is sent or called until you start it."),
    }
    if rebuilt.bad_numbers > 0 {
        what.push_str(&format!(" {} without a full phone number {} left out.", plural(rebuilt.bad_numbers, "person", "people"), if rebuilt.bad_numbers == 1 { "was" } else { "were" }));
    }
    if rebuilt.other_reasons > 0 {
        what.push(' ');
        what.push_str(&other_reasons_said(rebuilt.other_reasons));
    }
    // Everything it says or does, by value: each key that the table lets through and that is not a person (listed below).
    let mut listed = 0usize;
    for k in kept.kept.iter().filter(|k| k.row.class == Class::Runs && !k.path.starts_with("people[]") && !k.path.starts_with("skipped[]") && !CAMPAIGN_KEYS_NOT_LISTED.contains(&k.path.as_str())) {
        if matches!(&k.value, Value::String(s) if s.is_empty()) || matches!(&k.value, Value::Array(a) if a.is_empty()) {
            continue;
        }
        if listed >= 60 {
            what.push_str(" (More of its settings are not listed here.)");
            break;
        }
        listed += 1;
        let note = if k.path.starts_with("origin.") { " (it comes back started by the front desk)" } else { "" };
        what.push_str(&format!(" {}, {}: {}{note}.", campaign_key(k), k.row.what.to_lowercase(), show_value(&k.value)));
    }
    // The people: what a model reads of each of the first few, and the rest counted.
    for (at, p) in people.iter().enumerate().take(MAX_PEOPLE_LISTED) {
        what.push_str(&format!(" Person {} ({}): {}.", at + 1, p.get("number").and_then(Value::as_str).unwrap_or("?"), clip(&person_words(p), MAX_PERSON_TEXT)));
    }
    if people.len() > MAX_PEOPLE_LISTED {
        what.push_str(&format!(" {} more people are not listed here; what a model reads of them (their names, notes, details and results) is read in the same way.", people.len() - MAX_PEOPLE_LISTED));
    }
    // The people skipped at planning: their names and why, which the report says to the Agent.
    let skipped = campaign.get("skipped").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    for (at, s) in skipped.iter().enumerate().take(MAX_PEOPLE_LISTED) {
        what.push_str(&format!(" Skipped at planning {} ({}): {}.", at + 1, s.get("number").and_then(Value::as_str).unwrap_or("?"), clip(&person_words(s), MAX_PERSON_TEXT)));
    }
    if skipped.len() > MAX_PEOPLE_LISTED {
        what.push_str(&format!(" {} more people skipped at planning are not listed here.", skipped.len() - MAX_PEOPLE_LISTED));
    }
    what
}

/// The numbers not to be contacted, cleaned.
pub struct CleanedList {
    /// The entries that come back: of the shape the Agent writes, each number once (the same digits written another way
    /// count once: the page adds the ones that are not yet here, person by person), and no more than [`MAX_DO_NOT_CONTACT`].
    pub entries: Vec<Value>,
    /// Entries that repeated a number already in the list.
    pub repeated: usize,
    /// Entries left out because the list was already as long as one restore takes.
    pub over: usize,
}

/// The numbers not to be contacted, checked: only entries of the shape the Agent writes, each number once, and no more than
/// [`MAX_DO_NOT_CONTACT`]. `None`: it is not a list. (A file of a hundred thousand entries fits the size a file may be, and
/// the page compares each with the list it has: it is cut here, to a list a person could have made, before it is handed over.)
pub fn clean_do_not_contact(value: &Value) -> Option<CleanedList> {
    let list = value.as_array()?;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cleaned = CleanedList { entries: Vec::new(), repeated: 0, over: 0 };
    for entry in list {
        let Some(number) = entry.get("number").and_then(Value::as_str).filter(|n| !n.is_empty() && n.len() <= 40 && !n.chars().any(char::is_control)) else { continue };
        // The same digits, however they are written, are one number here; a number of letters is compared as it is.
        let digits: String = number.chars().filter(char::is_ascii_digit).collect();
        let key = if digits.is_empty() { number.trim().to_lowercase() } else { digits };
        if !seen.insert(key) {
            cleaned.repeated += 1;
            continue;
        }
        if cleaned.entries.len() >= MAX_DO_NOT_CONTACT {
            cleaned.over += 1;
            continue;
        }
        let at = entry.get("at").filter(|a| a.as_f64().is_some_and(|f| f.is_finite() && f >= 0.0)).cloned().unwrap_or(json!(0));
        let why = entry.get("why").and_then(Value::as_str).map(|w| w.chars().take(300).collect::<String>()).unwrap_or_default();
        cleaned.entries.push(json!({ "number": number, "at": at, "why": why }));
    }
    Some(cleaned)
}

// ---- looking ------------------------------------------------------------------------------------------

/// What the dry run says of the Agent's archive: the items that can come back, and what is never restored.
pub struct Described {
    pub items: Vec<ReviewItem>,
    pub not_restored: Vec<NotRestored>,
    /// How many files would come back without a tick (data) and how many need one, for the summary.
    pub restorable: usize,
    /// The names, as the backup calls them, of everything the look lists as able to come back.
    pub names: Vec<String>,
}

fn display(name: &str) -> String {
    format!("agent/{}", name.strip_prefix("opfs/").unwrap_or(name))
}

/// Describe what is in the Agent's archive at `path` (a file made from the backup's own copy). Reads its
/// directory, and the few small files that say what a project, a campaign or a setting is; nothing else.
pub fn describe(path: &Path, listing: &Listing, limits: &Limits, budget: &Budget) -> Result<Described> {
    let mut archive = open(path, limits)?;
    let mut items: Vec<ReviewItem> = Vec::new();
    let mut not_restored: Vec<NotRestored> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut restorable = 0usize;

    #[derive(Default)]
    struct Project {
        files: usize,
        bytes: u64,
        conversation: bool,
        sessions: usize,
        record: Option<usize>,
    }
    let mut projects: BTreeMap<String, Project> = BTreeMap::new();
    let mut knowledge: Vec<&Entry> = Vec::new();
    let (mut other_desk, mut other_desk_bytes) = (0usize, 0u64);
    let (mut desk_sessions, mut desk_session_bytes) = (0usize, 0u64);
    let mut excluded: BTreeMap<&str, (usize, String)> = BTreeMap::new();
    let mut unknown = 0usize;

    for (position, entry) in listing.entries.iter().enumerate() {
        budget.check()?;
        let shown = display(&entry.name);
        if let Some(why) = entry.unfit {
            not_restored.push(NotRestored { name: clip(&shown, 200), why: format!("not restored: {why}") });
            continue;
        }
        let Some(row) = entry.row else {
            unknown += 1;
            if unknown <= MAX_NAMED {
                not_restored.push(NotRestored { name: clip(&shown, 200), why: "not restored: unknown item".to_string() });
            }
            continue;
        };
        if row.class == Class::Excluded {
            if !row.quiet {
                let slot = excluded.entry(row.id.as_str()).or_insert_with(|| (0, shown.clone()));
                slot.0 += 1;
            }
            continue;
        }
        restorable += 1;
        names.push(entry.name.clone());
        let Some(class) = row.tick else { continue };
        let inner = entry.name.strip_prefix("opfs/").unwrap_or(&entry.name);
        let parts: Vec<&str> = inner.split('/').collect();
        match row.id.as_str() {
            "agent-project-meta" | "agent-project-files" | "agent-project-sessions" => {
                let p = projects.entry(parts[1].to_string()).or_default();
                p.files += 1;
                p.bytes += entry.size;
                if parts.get(2) == Some(&"chat.json") {
                    p.conversation = true;
                }
                if parts.get(2) == Some(&"sessions") {
                    p.sessions += 1;
                }
                if parts.get(2) == Some(&"project.json") {
                    p.record = Some(position);
                }
            }
            "agent-desk-brief" => {
                let quoted = read_small(&mut archive, entry, MAX_QUOTE_BYTES)?.map(|b| String::from_utf8_lossy(&b).into_owned());
                let says = quoted.map(|t| format!(" Says: {}", show_value(&Value::String(t)))).unwrap_or_else(|| " (too large to quote)".to_string());
                items.push(ReviewItem {
                    class,
                    name: shown,
                    title: "The front desk's brief".to_string(),
                    what: clip(&format!("{}. Every call, text and task reads it before each reply, and it wins over what the phone's agents would otherwise say.{says}", kb(entry.size)), 700),
                });
            }
            "agent-desk-knowledge" => knowledge.push(entry),
            "agent-desk-files" => {
                other_desk += 1;
                other_desk_bytes += entry.size;
            }
            "agent-desk-sessions" => {
                desk_sessions += 1;
                desk_session_bytes += entry.size;
            }
            "agent-desk-meta" => {}
            "agent-desk-chat" => {
                items.push(ReviewItem { class, name: shown, title: "The front desk's own conversation".to_string(), what: format!("{}: loaded as what was said before.", kb(entry.size)) });
            }
            "agent-desk-callers" => {
                let count = read_small(&mut archive, entry, limits.max_agent_read_bytes)?
                    .and_then(|b| serde_json::from_slice::<Value>(strip_bom(&b)).ok())
                    .map(|v| v.as_array().map(Vec::len).or_else(|| v.as_object().map(|o| o.len())).unwrap_or(0));
                let what = match count {
                    Some(n) => format!("{}: the facts and notes that the phone's agents read about a person before they answer them.", plural(n, "entry", "entries")),
                    None => format!("{} that OAIY could not read as a list, or that is too large to look at.", kb(entry.size)),
                };
                items.push(ReviewItem { class, name: shown, title: "What the phone's agents remember about people".to_string(), what });
            }
            "agent-outreach-campaign" => {
                let bytes = read_small(&mut archive, entry, limits.max_agent_read_bytes)?;
                let Some(bytes) = bytes else {
                    items.push(ReviewItem { class, name: shown, title: clip(parts.last().copied().unwrap_or("campaign"), 120), what: super::review::TOO_LARGE.to_string() });
                    continue;
                };
                let outcome = serde_json::from_slice::<Value>(strip_bom(&bytes)).map_err(|_| "it is not valid JSON".to_string()).and_then(|v| {
                    let kt = table().key_table("agent.campaign").ok_or_else(|| "no key table".to_string())?;
                    let kept = filter_json(kt, &v, &|_| true);
                    let was = v.get("state").and_then(Value::as_str).map(str::to_string);
                    rebuild_campaign(&kept.value, was.as_deref()).map(|c| (kept, c, was))
                });
                match outcome {
                    Ok((kept, rebuilt, was)) => items.push(ReviewItem {
                        class,
                        name: shown,
                        title: clip(&format!("Campaign \"{}\"", rebuilt.campaign.get("name").and_then(Value::as_str).unwrap_or("?")), 120),
                        what: clip(&describe_campaign(&kept, &rebuilt, was.as_deref()), MAX_CAMPAIGN_TEXT),
                    }),
                    Err(why) => items.push(ReviewItem { class, name: shown, title: clip(parts.last().copied().unwrap_or("campaign"), 120), what: format!("Could not be read ({why}): OAIY would not load it, so it is not brought back.") }),
                }
            }
            "agent-outreach-index" => {}
            "agent-settings" => {
                if let Some(bytes) = read_small(&mut archive, entry, limits.max_agent_read_bytes)? {
                    items.extend(describe_settings(&bytes, &mut not_restored));
                } else {
                    items.push(ReviewItem { class, name: shown, title: "The Agent's settings".to_string(), what: super::review::TOO_LARGE.to_string() });
                }
            }
            _ => {}
        }
    }

    // The projects, by name (the first few hundred: a person has a few dozen, and a hostile archive can name 50,000).
    let shown_projects = projects.len().min(MAX_NAMED);
    for (id, p) in projects.iter().take(MAX_NAMED) {
        let title = p
            .record
            .and_then(|position| listing.entries.get(position))
            .map(|e| read_small(&mut archive, e, MAX_QUOTE_BYTES))
            .transpose()?
            .flatten()
            .and_then(|b| serde_json::from_slice::<Value>(strip_bom(&b)).ok())
            .and_then(|v| v.get("name").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| id.clone());
        let mut what = format!("A project: {} ({})", plural(p.files, "file", "files"), kb(p.bytes));
        if p.conversation {
            what.push_str(", with its conversation");
        }
        if p.sessions > 0 {
            what.push_str(&format!(", {}", plural(p.sessions, "phone conversation file", "phone conversation files")));
        }
        what.push_str(". The Agent reads a project's files and conversation as context when it is opened.");
        items.push(ReviewItem { class: RestoreClass::AgentData, name: format!("agent/projects/{id}"), title: clip(&title, 120), what });
    }
    if projects.len() > shown_projects {
        let (files, bytes) = projects.values().skip(shown_projects).fold((0usize, 0u64), |(f, b), p| (f + p.files, b + p.bytes));
        items.push(ReviewItem {
            class: RestoreClass::AgentData,
            name: "agent/projects".to_string(),
            title: "More projects".to_string(),
            what: format!(
                "{} more {}, with {} ({}).",
                projects.len() - shown_projects,
                if projects.len() - shown_projects == 1 { "project" } else { "projects" },
                plural(files, "file", "files"),
                kb(bytes)
            ),
        });
    }
    // The knowledge files, each by name.
    for entry in knowledge.iter().take(MAX_NAMED) {
        items.push(ReviewItem {
            class: RestoreClass::AgentData,
            name: display(&entry.name),
            title: clip(entry.name.rsplit('/').next().unwrap_or(&entry.name), 120),
            what: format!("{}. The phone's agents read it to answer callers.", kb(entry.size)),
        });
    }
    if knowledge.len() > MAX_NAMED {
        let rest: u64 = knowledge[MAX_NAMED..].iter().map(|e| e.size).sum();
        items.push(ReviewItem { class: RestoreClass::AgentData, name: "agent/front-desk/files/knowledge".to_string(), title: "More knowledge files".to_string(), what: format!("{} more ({}).", plural(knowledge.len() - MAX_NAMED, "file", "files"), kb(rest)) });
    }
    if other_desk > 0 {
        items.push(ReviewItem {
            class: RestoreClass::AgentData,
            name: "agent/front-desk/files".to_string(),
            title: "Other files at the front desk".to_string(),
            what: format!("{} ({}): outreach results and other files the Agent made.", plural(other_desk, "file", "files"), kb(other_desk_bytes)),
        });
    }
    if desk_sessions > 0 {
        items.push(ReviewItem {
            class: RestoreClass::Conversations,
            name: "agent/front-desk/sessions".to_string(),
            title: "The phone's conversations".to_string(),
            what: format!("{} ({}): each call and text thread, loaded as what was said before.", plural(desk_sessions, "file", "files"), kb(desk_session_bytes)),
        });
    }
    if unknown > MAX_NAMED {
        not_restored.push(NotRestored { name: format!("and {} more", unknown - MAX_NAMED), why: "not restored: unknown item".to_string() });
    }
    for (id, (count, first)) in excluded {
        let row = table().agent.iter().find(|r| r.id == id);
        let why = row.map(|r| r.reason.clone()).unwrap_or_default();
        let name = if count > 1 { format!("{first} and {} more like it", count - 1) } else { first };
        not_restored.push(NotRestored { name: clip(&name, 200), why: clip(&format!("not restored: {why}"), 400) });
    }
    Ok(Described { items, not_restored, restorable, names })
}

/// The Agent's settings, by key, by name and by value: what comes back with a tick, and what is never restored.
pub fn describe_settings(bytes: &[u8], not_restored: &mut Vec<NotRestored>) -> Vec<ReviewItem> {
    const FILE: &str = "agent/idb/settings.json";
    let Some(kt) = table().key_table("agent.settings") else { return Vec::new() };
    let Ok(value) = serde_json::from_slice::<Value>(strip_bom(bytes)) else {
        return vec![ReviewItem {
            class: RestoreClass::AgentSettings,
            name: FILE.to_string(),
            title: "The Agent's settings".to_string(),
            what: "Could not be read (it is not valid JSON): OAIY would not load it, so it is not brought back.".to_string(),
        }];
    };
    let found = filter_json(kt, &value, &|_| true);
    let mut items: Vec<ReviewItem> = Vec::new();
    // A provider is one thing: its kind, its address, its model and whether it has a key.
    let mut providers: BTreeMap<usize, Vec<&super::table::Kept>> = BTreeMap::new();
    for k in &found.kept {
        if k.path.starts_with("providers[].") {
            providers.entry(k.at.first().copied().unwrap_or(0)).or_default().push(k);
        }
    }
    for (_, keys) in providers {
        let get = |path: &str| keys.iter().find(|k| k.path == format!("providers[].{path}")).and_then(|k| k.value.as_str());
        let id = get("id").unwrap_or("(no id)");
        let key = if get("apiKey").is_some_and(|k| !k.trim().is_empty()) { "; has an API key (brought back only with the keys box, and only where yours has none)" } else { "" };
        items.push(ReviewItem {
            class: RestoreClass::AgentSettings,
            name: format!("{FILE}#providers"),
            title: clip(&format!("{} ({id})", get("name").unwrap_or(id)), 120),
            what: clip(
                &format!(
                    "Agent provider of type {} at {}{}{key}. If yours of the same id is at another address, this one arrives beside it, without a key.",
                    get("type").unwrap_or("?"),
                    get("baseUrl").unwrap_or("(default address)"),
                    get("modelId").map(|m| format!(", model {m}")).unwrap_or_default()
                ),
                500,
            ),
        });
    }
    for k in found.kept.iter().filter(|k| k.row.class == Class::Runs && !k.path.starts_with("providers[]") && !k.row.secret && !matches!(k.value, Value::Object(_))) {
        items.push(key_item(FILE, k));
    }
    // What is left out, by key.
    for l in &found.left {
        let why = match (&l.why, l.row) {
            (Why::Unknown, _) => "not restored: unknown item".to_string(),
            (Why::Excluded, Some(row)) => format!("not restored: {}{}", row.reason, row.redo.as_ref().map(|r| format!(" To do again: {r}")).unwrap_or_default()),
            (Why::BadValue(why), _) => format!("not restored: its value is not one this version accepts ({why})"),
            (Why::KeyWithoutAddress, _) => format!("not restored: {}", super::table::KEY_WITHOUT_ADDRESS),
            _ => continue,
        };
        not_restored.push(NotRestored { name: clip(&format!("{FILE}#{}", l.path), 200), why: clip(&why, 400) });
    }
    if found.left_more > 0 {
        not_restored.push(NotRestored { name: FILE.to_string(), why: format!("and {} more keys that are not restored", found.left_more) });
    }
    items
}

// ---- preparing ----------------------------------------------------------------------------------------

/// Whether the archive is being restored from a backup, or an undo is putting back what a restore replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Data comes back, and what runs things comes back if its kind was ticked.
    Restore,
    /// What a restore replaced comes back: it is the person's own, so no tick is asked for, and every rule of
    /// the table that keeps a campaign paused and a list of numbers whole still holds.
    Undo,
}

/// One item of the archive the page is given, and how it is to be brought back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub name: String,
    pub mode: String,
}

/// What preparing the Agent's archive left, and what was said about the rest.
#[derive(Debug, Default)]
pub struct Prepared {
    pub items: Vec<Applied>,
    /// What was left out, in plain words.
    pub notes: Vec<String>,
    /// Whether the settings came back in any part.
    pub settings: bool,
}

enum Source {
    Nested(usize, u64),
    Temp(PathBuf),
}

/// Write `out` (a new private file): the archive of only what is to come back. See the module note.
pub fn filter(nested: &Path, out: &Path, scratch: &Path, ticks: &Ticks, mode: Mode, limits: &Limits, budget: &Budget) -> Result<Prepared> {
    let listing = read_listing(nested, limits)?;
    let mut archive = open(nested, limits)?;
    let mut prepared = Prepared::default();
    let mut plan: Vec<(String, String, Source)> = Vec::new();
    let mut left_out: BTreeMap<RestoreClass, usize> = BTreeMap::new();
    let mut unknown = 0usize;
    let mut excluded: BTreeMap<&str, usize> = BTreeMap::new();
    let mut campaigns: Vec<String> = Vec::new();
    let wanted = |class: RestoreClass| mode == Mode::Undo || ticks.has(class);
    let keep_key = |row: &KeyRow| -> bool {
        if row.secret && (!ticks.keys || mode == Mode::Undo) {
            return false;
        }
        row.class == Class::Data || mode == Mode::Undo || row.tick.is_some_and(|t| ticks.has(t))
    };
    let mut temp_count = 0usize;
    let mut temp_file = |bytes: &[u8]| -> Result<PathBuf> {
        temp_count += 1;
        let path = scratch.join(format!("agent-part-{temp_count:06}"));
        secret_file::write(&path, bytes).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
        Ok(path)
    };

    for entry in &listing.entries {
        budget.check()?;
        if entry.unfit.is_some() {
            unknown += 1;
            continue;
        }
        let Some(row) = entry.row else {
            unknown += 1;
            continue;
        };
        if row.class == Class::Excluded {
            if !row.quiet {
                *excluded.entry(row.id.as_str()).or_default() += 1;
            }
            continue;
        }
        if let Some(tick) = row.file_tick() {
            if !wanted(tick) {
                *left_out.entry(tick).or_default() += 1;
                continue;
            }
        }
        match (row.keys.as_deref(), row.merge.as_deref()) {
            (Some("agent.settings"), _) => {
                let Some(bytes) = read_small(&mut archive, entry, limits.max_agent_read_bytes)? else {
                    prepared.notes.push("The Agent's settings were not brought back: the file is too large to be read.".to_string());
                    continue;
                };
                let Ok(value) = serde_json::from_slice::<Value>(strip_bom(&bytes)) else {
                    prepared.notes.push("The Agent's settings were not brought back: the file is not valid JSON.".to_string());
                    continue;
                };
                let Some(kt) = table().key_table("agent.settings") else { continue };
                // An undo puts back the person's own state, empty addresses included; a restore takes what is plain.
                let found = if mode == Mode::Undo { filter_json_exact(kt, &value, &keep_key) } else { filter_json(kt, &value, &keep_key) };
                // A key whose address does not come back is left out with it (it would arrive at no address, and a provider with none is the
                // vendor's own): said on its own, since the person has to enter it again.
                let keys_without_address = found.left.iter().filter(|l| l.why == Why::KeyWithoutAddress).count();
                if keys_without_address > 0 {
                    let one = keys_without_address == 1;
                    prepared.notes.push(format!(
                        "{} of the Agent's {} left out: {} come back (a name and password or a key in it, or it is not a web address), and a key goes only with the address it was kept for. Enter {} again as the key of {}.",
                        plural(keys_without_address, "API key", "API keys"),
                        if one { "was" } else { "were" },
                        if one { "its address does not" } else { "their addresses do not" },
                        if one { "it" } else { "them" },
                        if one { "its provider" } else { "their providers" },
                    ));
                }
                let left = found.left.len() + found.left_more - keys_without_address;
                if found.kept.is_empty() {
                    if left > 0 {
                        prepared.notes.push(format!("Nothing in the Agent's settings comes back without its tick ({left} setting{} left out).", if left == 1 { "" } else { "s" }));
                    }
                    continue;
                }
                if left > 0 {
                    prepared.notes.push(format!("{left} setting{} of the Agent's not brought back (addresses that were read from a service, a project that was open on another computer, and anything not in the table are never restored; the rest need their tick).", if left == 1 { " was" } else { "s were" }));
                }
                let bytes = serde_json::to_vec(&found.value).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's settings could not be written."))?;
                plan.push((entry.name.clone(), "settings".to_string(), Source::Temp(temp_file(&bytes)?)));
                prepared.settings = true;
            }
            (_, Some("campaign")) => {
                let Some(bytes) = read_small(&mut archive, entry, limits.max_agent_read_bytes)? else {
                    prepared.notes.push(format!("{} was not brought back: it is too large to be read.", clip(&display(&entry.name), 120)));
                    continue;
                };
                let rebuilt = serde_json::from_slice::<Value>(strip_bom(&bytes)).map_err(|_| "it is not valid JSON".to_string()).and_then(|v| {
                    let kt = table().key_table("agent.campaign").ok_or_else(|| "no key table".to_string())?;
                    rebuild_campaign(&filter_json(kt, &v, &|_| true).value, v.get("state").and_then(Value::as_str))
                });
                match rebuilt {
                    Ok(Rebuilt { campaign, bad_numbers, other_reasons, started_by }) => {
                        let id = campaign.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                        // The file is named for its campaign: a campaign cannot be written over another's file.
                        if entry.name != format!("opfs/front-desk/outreach/{id}.json") {
                            prepared.notes.push(format!("{} was not brought back: its name is not its campaign's.", clip(&display(&entry.name), 120)));
                            continue;
                        }
                        if bad_numbers > 0 {
                            prepared.notes.push(format!("{}: {} without a full phone number {} left out.", clip(&display(&entry.name), 120), plural(bad_numbers, "person", "people"), if bad_numbers == 1 { "was" } else { "were" }));
                        }
                        if other_reasons > 0 {
                            prepared.notes.push(format!("{}: {}", clip(&display(&entry.name), 120), other_reasons_said(other_reasons)));
                        }
                        if let Some((project, _)) = &started_by {
                            prepared.notes.push(format!("{}: it said it was started by the project \"{}\"; it comes back started by the front desk.", clip(&display(&entry.name), 120), clip(project, 60)));
                        }
                        campaigns.push(id);
                        let bytes = serde_json::to_vec(&campaign).map_err(|_| BackupError::new(ErrorKind::Damaged, "A campaign could not be written."))?;
                        plan.push((entry.name.clone(), "campaign".to_string(), Source::Temp(temp_file(&bytes)?)));
                    }
                    Err(why) => prepared.notes.push(format!("{} was not brought back: {why}.", clip(&display(&entry.name), 120))),
                }
            }
            (_, Some("campaign-index")) => {
                // Rebuilt below, from the campaigns that came back.
            }
            (_, Some("union")) => {
                let Some(bytes) = read_small(&mut archive, entry, limits.max_agent_read_bytes)? else {
                    prepared.notes.push("The list of numbers not to be contacted was not brought back: it is too large to be read.".to_string());
                    continue;
                };
                let Some(list) = serde_json::from_slice::<Value>(strip_bom(&bytes)).ok().as_ref().and_then(clean_do_not_contact) else {
                    prepared.notes.push("The list of numbers not to be contacted was not brought back: it is not a list of numbers.".to_string());
                    continue;
                };
                if list.repeated > 0 {
                    prepared.notes.push(format!("{} in the list of numbers not to be contacted repeated a number and {} counted once.", plural(list.repeated, "entry", "entries"), if list.repeated == 1 { "was" } else { "were" }));
                }
                if list.over > 0 {
                    prepared.notes.push(format!("{} more of the numbers not to be contacted were left out: at most {MAX_DO_NOT_CONTACT} come back in one restore.", list.over));
                }
                if list.entries.is_empty() {
                    continue;
                }
                let bytes = serde_json::to_vec(&list.entries).map_err(|_| BackupError::new(ErrorKind::Damaged, "The list of numbers could not be written."))?;
                plan.push((entry.name.clone(), "union".to_string(), Source::Temp(temp_file(&bytes)?)));
            }
            _ => plan.push((entry.name.clone(), "replace".to_string(), Source::Nested(entry.index, entry.size))),
        }
    }
    // The list of campaigns is the ones that came back.
    if !campaigns.is_empty() && listing.entries.iter().any(|e| e.row.is_some_and(|r| r.merge.as_deref() == Some("campaign-index"))) {
        let bytes = serde_json::to_vec(&campaigns).map_err(|_| BackupError::new(ErrorKind::Damaged, "The list of campaigns could not be written."))?;
        plan.push(("opfs/front-desk/outreach/index.json".to_string(), "campaign-index".to_string(), Source::Temp(temp_file(&bytes)?)));
    }

    for (class, n) in &left_out {
        prepared.notes.push(format!("Not brought back (not ticked): the Agent's {} ({}).", class.label().to_lowercase(), plural(*n, "file", "files")));
    }
    if unknown > 0 {
        prepared.notes.push(format!("Not restored: {} in the Agent's storage that this version of OAIY does not know (unknown items are never restored).", plural(unknown, "item", "items")));
    }
    for (id, n) in &excluded {
        let reason = table().agent.iter().find(|r| r.id == *id).map(|r| r.reason.clone()).unwrap_or_default();
        prepared.notes.push(clip(&format!("Not restored ({}): {reason}", plural(*n, "file", "files")), 400));
    }

    // Written: the record first, then each item.
    let file = secret_file::create_new_owner_only(out).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
    let mut writer = ZipWriter::new(BufWriter::with_capacity(64 << 10, file));
    let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(6));
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let zip_err = |_| BackupError::new(ErrorKind::Io, "Could not stage the Agent's storage.");
    let record = json!({
        "v": 2,
        "kind": "oaiy-agent-storage",
        "createdAt": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "items": plan.iter().map(|(name, mode, _)| json!({ "name": name, "mode": mode })).collect::<Vec<_>>(),
    });
    writer.start_file("agent-manifest.json", deflated).map_err(zip_err)?;
    writer.write_all(&serde_json::to_vec(&record).unwrap_or_default()).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
    let mut buf = vec![0u8; 64 << 10];
    for (name, mode, source) in &plan {
        let already = name.ends_with(".zip") || name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".jpeg") || name.ends_with(".webp") || name.ends_with(".mp3") || name.ends_with(".ogg") || name.ends_with(".mp4") || name.ends_with(".glb");
        match source {
            Source::Nested(index, size) => {
                let file = archive.by_index(*index).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage in this backup is damaged."))?;
                let mut reader = file.take(size + 1);
                writer.start_file(name.as_str(), if already { stored } else { deflated }).map_err(zip_err)?;
                let mut total = 0u64;
                loop {
                    budget.check()?;
                    let n = reader.read(&mut buf).map_err(|_| BackupError::new(ErrorKind::Damaged, "The Agent's storage in this backup is damaged."))?;
                    if n == 0 {
                        break;
                    }
                    total += n as u64;
                    writer.write_all(&buf[..n]).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
                }
                if total != *size {
                    return Err(BackupError::new(ErrorKind::Damaged, "A file in the Agent's storage is not the size its record says."));
                }
            }
            Source::Temp(path) => {
                writer.start_file(name.as_str(), deflated).map_err(zip_err)?;
                let mut input = BufReader::new(File::open(path).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?);
                std::io::copy(&mut input, &mut writer).map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
                let _ = std::fs::remove_file(path);
            }
        }
        prepared.items.push(Applied { name: name.clone(), mode: mode.clone() });
    }
    let buffered = writer.finish().map_err(zip_err)?;
    let file = buffered.into_inner().map_err(|e| BackupError::io("Could not stage the Agent's storage", &e.into_error()))?;
    file.sync_all().map_err(|e| BackupError::io("Could not stage the Agent's storage", &e))?;
    Ok(prepared)
}
