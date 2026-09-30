//! Making what a backup carries safe to bring back, and safe to make.
//!
//! A backup is a file anyone could have made, and the data in it is read by programs that act on it.
//! So the files that carry more than data are cleaned, on the way in to a backup (so a secret is never
//! written) and again on the way out (so a hostile backup cannot plant one):
//!
//! - **Settings files** (a plugin's settings, the Agent's settings) pass through the key table of
//!   [`super::table`]: only the keys it lists come through, each with a value of the kind it allows. A PIN, an
//!   address that audio goes to, a switch that grants access and anything the table does not know never
//!   travel, in either direction.
//! - **The calendar** is brought back key by key ([`calendar_merge`]): its words and its appointments (the phone
//!   sends every appointment it has no copy of at FormLogic to the linked account) need a tick, the typed values
//!   that carry none (the opening hours, the steps between the times offered) do not, and its FormLogic sync state (the form it is
//!   paired with, the cursor, the list of records to delete, each appointment's remote copy, the tombstones
//!   of deleted ones) never comes back: restored on a computer that is linked to another account, or to none,
//!   it would delete or duplicate records.
//! - **The provider list** loses its API keys unless the person asked for them.
//! - **The run journal** keeps only the runs that finished: a run that was waiting would start at the
//!   next start.
//! - **The autostart list** keeps only services whose template exists.
//! - **The messages callers left** are cleaned as the store cleans one it takes, and held to the store's own limits ([`messages_for_restore`]).

use std::collections::HashSet;

use serde_json::Value;

/// The largest JSON file that is read to be cleaned.
pub const MAX_JSON_BYTES: usize = 16 << 20;

fn parse(bytes: &[u8]) -> Result<Value, String> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err("it is too large to be read".to_string());
    }
    serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)).map_err(|_| "it is not valid JSON".to_string())
}

/// The calendar a restore leaves, and what it says of it.
pub struct CalendarMerge {
    pub bytes: Vec<u8>,
    pub notes: Vec<String>,
}

/// The most appointments a calendar may hold once a restore has added to it.
const MAX_APPOINTMENTS: usize = 10_000;

/// The calendar in a backup put into the one that is here, key by key (see the key table `calendar`).
///
/// What the table lets through for this choice is brought in: the typed values that carry no words always (the opening hours
/// and the steps between the times offered), and the rest (the business's and the receptionist's names, the services, and the
/// appointments) only when `ticks` has the calendar. An appointment is not data: the phone sends every appointment it has no copy
/// of at FormLogic to the linked account, so a restore that was not ticked for the calendar brings back none, and an appointment
/// that is here is never touched by it. With the calendar ticked, an appointment of the same id takes the backup's and keeps the
/// record of FormLogic's copy that is here, a new one is added (the calendar holds no more than [`MAX_APPOINTMENTS`]), and
/// appointments that are only here stay. The FormLogic sync state (deleted records, where the sync got to) is never taken from a backup.
///
/// The result is a calendar the calendar module reads (a file it cannot read is read as an empty calendar and then
/// saved over): when it would not be one, `Err` says so and the file that is here is left alone.
pub fn calendar_merge(local: Option<&Value>, staged: &Value, ticks: &super::review::Ticks) -> Result<CalendarMerge, String> {
    use super::table::{filter_json, table, Class};
    use serde_json::{json, Map};

    let keys = table().key_table("calendar").ok_or_else(|| "OAIY does not know how to read it".to_string())?;
    let found = filter_json(keys, staged, &|row| row.class == Class::Data || row.tick.is_some_and(|t| ticks.has(t)));
    let left = found.left.iter().map(|l| l.path.as_str()).collect::<HashSet<_>>().len() + found.left_more;
    if found.kept.is_empty() {
        return Err(found.nothing_comes_back());
    }
    let theirs = found.value.as_object().cloned().unwrap_or_default();
    let mut notes = Vec::new();
    if left > 0 && !ticks.has(super::review::RestoreClass::Calendar) {
        notes.push(format!(
            "{left} setting{} of the calendar {} not brought back: its words, its services and its appointments need the tick \"{}\".",
            if left == 1 { "" } else { "s" },
            if left == 1 { "was" } else { "were" },
            super::review::RestoreClass::Calendar.label()
        ));
    }

    // What is here, if it is a calendar the module reads; otherwise a calendar with nothing in it.
    let mut book: Map<String, Value> = match local {
        Some(Value::Object(m)) if crate::calendar::is_readable(&Value::Object(m.clone()).to_string()) => m.clone(),
        _ => {
            let mut fresh = Map::new();
            fresh.insert("settings".into(), crate::calendar::default_settings_json());
            fresh.insert("appointments".into(), json!([]));
            fresh
        }
    };

    // The settings: each key that came through replaces the one that is here.
    if let Some(Value::Object(their_settings)) = theirs.get("settings") {
        let settings = book.entry("settings").or_insert_with(|| json!({}));
        if !settings.is_object() {
            *settings = json!({});
        }
        let settings = settings.as_object_mut().expect("an object");
        for (key, value) in their_settings {
            if key == "services" {
                // A service is its words: a list of them is taken only whole, and only when the words were ticked.
                let usable: Vec<Value> = value
                    .as_array()
                    .map(|list| list.iter().filter(|s| s.get("name").and_then(Value::as_str).is_some_and(|n| !n.trim().is_empty()) && s.get("minutes").is_some_and(Value::is_number)).cloned().collect())
                    .unwrap_or_default();
                if !usable.is_empty() {
                    settings.insert(key.clone(), Value::Array(usable));
                }
            } else {
                settings.insert(key.clone(), value.clone());
            }
        }
    }

    // The appointments (there are some only when they were ticked): by id, and never one of the person's own lost.
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut list: Vec<Value> = book.get("appointments").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut at: std::collections::HashMap<String, usize> = list.iter().enumerate().filter_map(|(i, a)| a.get("id").and_then(Value::as_str).map(|id| (id.to_string(), i))).collect();
    let (mut added, mut replaced, mut skipped, mut over) = (0usize, 0usize, 0usize, 0usize);
    for appointment in theirs.get("appointments").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
        let text = |key: &str| appointment.get(key).and_then(Value::as_str).map(str::to_string);
        let (Some(id), Some(start), Some(status)) = (text("id"), text("start"), text("status")) else {
            skipped += 1;
            continue;
        };
        let Some(minutes) = appointment.get("minutes").and_then(Value::as_u64) else {
            skipped += 1;
            continue;
        };
        let mut composed = json!({
            "id": id, "service": text("service").unwrap_or_default(), "start": start, "minutes": minutes, "status": status,
            "name": text("name").unwrap_or_default(), "phone": text("phone").unwrap_or_default(), "notes": text("notes").unwrap_or_default(),
            "source": text("source").unwrap_or_else(|| "manual".to_string()),
            "createdAt": text("createdAt").unwrap_or_else(|| now.clone()), "updatedAt": text("updatedAt").unwrap_or_else(|| now.clone()),
        });
        match at.get(&id).copied() {
            Some(i) => {
                // The record of FormLogic's copy, and of the request and the call, belong to this computer.
                for keep in ["formlogic", "requestId", "callId"] {
                    if let Some(v) = list[i].get(keep).cloned() {
                        composed[keep] = v;
                    }
                }
                list[i] = composed;
                replaced += 1;
            }
            None if list.len() >= MAX_APPOINTMENTS => over += 1,
            None => {
                at.insert(id, list.len());
                list.push(composed);
                added += 1;
            }
        }
    }
    book.insert("appointments".into(), Value::Array(list));

    if skipped > 0 {
        notes.push(format!("{skipped} appointment{} without a valid id, time, length or state {} left out.", if skipped == 1 { "" } else { "s" }, if skipped == 1 { "was" } else { "were" }));
    }
    if over > 0 {
        notes.push(format!("{over} more appointment{} would make more than {MAX_APPOINTMENTS} in the calendar, so {} not added.", if over == 1 { "" } else { "s" }, if over == 1 { "it was" } else { "they were" }));
    }
    if added + replaced > 0 {
        notes.push(format!("{} appointment{} came back: the phone sends {} to your linked FormLogic account at its next sync, as it does any appointment it has no copy of there.", added + replaced, if added + replaced == 1 { "" } else { "s" }, if added + replaced == 1 { "it" } else { "them" }));
    }
    if replaced > 0 {
        notes.push(format!("{replaced} appointment{} here {} replaced by the backup's.", if replaced == 1 { "" } else { "s" }, if replaced == 1 { "was" } else { "were" }));
    }

    let bytes = serde_json::to_vec_pretty(&Value::Object(book)).map_err(|_| "it could not be written".to_string())?;
    if !crate::calendar::is_readable(&String::from_utf8_lossy(&bytes)) {
        return Err("it would not make a calendar OAIY can read".to_string());
    }
    Ok(CalendarMerge { bytes, notes })
}

/// The provider list as it may come back: without its API keys unless `keep_keys` (the keys box), and, keys or not, with no
/// credential in an address (a name and password before the host, a fragment, a key in the query: see
/// `table::address_without_credentials`; the key of a provider is where a key belongs). Returns the bytes, how many keys were taken
/// out and how many addresses were cleaned.
pub fn providers_for_restore(bytes: &[u8], keep_keys: bool) -> Result<(Vec<u8>, usize, usize), String> {
    let mut value = parse(bytes)?;
    let (mut removed, mut cleaned) = (0, 0);
    if let Some(Value::Array(providers)) = value.get_mut("providers") {
        for p in providers {
            if let Value::Object(o) = p {
                if !keep_keys {
                    for key in ["apiKey", "api_key"] {
                        if o.remove(key).is_some_and(|v| v.as_str().is_some_and(|s| !s.trim().is_empty())) {
                            removed += 1;
                        }
                    }
                }
                if let Some(address) = o.get("baseUrl").and_then(Value::as_str).and_then(super::table::address_without_credentials) {
                    o.insert("baseUrl".to_string(), Value::String(address));
                    cleaned += 1;
                }
            }
        }
    }
    Ok((serde_json::to_vec_pretty(&value).map_err(|_| "it could not be written".to_string())?, removed, cleaned))
}

/// The run journal with only the runs that finished. Returns the bytes and how many lines were left out.
pub fn ledger_finished_only(bytes: &[u8]) -> (Vec<u8>, usize) {
    const FINISHED: [&str; 4] = ["succeeded", "failed", "timed_out", "cancelled"];
    let mut out = Vec::new();
    let mut left_out = 0;
    for line in String::from_utf8_lossy(bytes).lines().filter(|l| !l.trim().is_empty()) {
        let finished = serde_json::from_str::<Value>(line).ok().and_then(|v| v.get("status").and_then(Value::as_str).map(|s| FINISHED.contains(&s))).unwrap_or(false);
        if finished {
            out.extend_from_slice(line.as_bytes());
            out.push(b'\n');
        } else {
            left_out += 1;
        }
    }
    (out, left_out)
}

/// The autostart list with only the ids in `known`. Returns the bytes and the ids left out.
pub fn autostart_known_only(bytes: &[u8], known: &HashSet<String>) -> Result<(Vec<u8>, Vec<String>), String> {
    let ids: Vec<String> = serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)).map_err(|_| "it is not a list of services".to_string())?;
    let (kept, dropped): (Vec<String>, Vec<String>) = ids.into_iter().partition(|id| known.contains(id));
    Ok((serde_json::to_vec_pretty(&kept).map_err(|_| "it could not be written".to_string())?, dropped))
}

/// The messages a restore brings back: each is cleaned as the store cleans a message it takes (its words, its name, its numbers and its ids), what is not
/// a message is left out, an id that is there twice is taken once, and there are at most as many as the store holds, the newest. The store enforces
/// its limits when a message is taken and not when its file is read, so a file that is put in place is held to them here, and what comes out is what
/// the store would have written (it reads it without complaint: it never puts a restored file aside). Returns the file, and what was said of it.
pub fn messages_for_restore(bytes: &[u8]) -> Result<(Vec<u8>, Vec<String>), String> {
    use crate::messages::{clean, MAX_MESSAGE, MAX_NAME, MAX_STORED};
    use serde_json::json;
    let value = parse(bytes)?;
    let Some(list) = value.get("messages").and_then(Value::as_array) else { return Err("it is not a list of messages".to_string()) };
    // What a message says in `key`, cleaned as the store cleans it, and whether cleaning changed it.
    let text = |m: &Value, key: &str, most: usize| -> (String, bool) {
        let said = m.get(key).and_then(Value::as_str).unwrap_or("");
        let cleaned = clean(said, most);
        let changed = cleaned != said;
        (cleaned, changed)
    };
    let stamp = |m: &Value, key: &str| m.get(key).and_then(Value::as_str).filter(|s| chrono::DateTime::parse_from_rfc3339(s).is_ok()).map(str::to_string);
    let mut kept: Vec<Value> = Vec::new();
    let mut ids: HashSet<String> = HashSet::new();
    let (mut malformed, mut repeated, mut changed) = (0usize, 0usize, 0usize);
    for m in list {
        let ((id, _), (message, message_changed), at) = (text(m, "id", 100), text(m, "message", MAX_MESSAGE), stamp(m, "at"));
        if id.is_empty() || message.is_empty() || at.is_none() {
            malformed += 1;
            continue;
        }
        if !ids.insert(id.clone()) {
            repeated += 1;
            continue;
        }
        // (The store keeps the number a caller rang from as the phone said it, and the number to ring back on as digits, or else the number they rang from:
        // so both are kept as words, at the most the store keeps of either.)
        let ((call_id, a), (from, b), (name, c), (callback, d)) = (text(m, "callId", 200), text(m, "from", 40), text(m, "name", MAX_NAME), text(m, "callback", 40));
        if message_changed || a || b || c || d {
            changed += 1;
        }
        let state = match m.get("state").and_then(Value::as_str) {
            Some(s @ ("seen" | "handled")) => s,
            _ => "new",
        };
        kept.push(json!({
            "id": id, "at": at, "callId": call_id, "from": from, "name": name, "callback": callback, "message": message,
            "urgency": if m.get("urgency").and_then(Value::as_str) == Some("urgent") { "urgent" } else { "normal" },
            "wantsCallback": m.get("wantsCallback").and_then(Value::as_bool).unwrap_or(false), "state": state,
            "seenAt": stamp(m, "seenAt"), "handledAt": stamp(m, "handledAt"),
            "handledBy": m.get("handledBy").and_then(Value::as_str).map(|s| clean(s, 40)).filter(|s| !s.is_empty()),
        }));
    }
    // The newest are kept (by the time each says, not by where it is in the file), and what is kept stays in the order the file had.
    let over = kept.len().saturating_sub(MAX_STORED);
    if over > 0 {
        let when = |m: &Value| m["at"].as_str().and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()).map(|t| t.timestamp_millis()).unwrap_or(i64::MIN);
        let mut order: Vec<usize> = (0..kept.len()).collect();
        order.sort_by_key(|&i| (when(&kept[i]), i));
        let oldest: HashSet<usize> = order.into_iter().take(over).collect();
        let mut i = 0;
        kept.retain(|_| {
            i += 1;
            !oldest.contains(&(i - 1))
        });
    }
    let messages = |n: usize| if n == 1 { "1 message".to_string() } else { format!("{n} messages") };
    let were = |n: usize| if n == 1 { "was" } else { "were" };
    let mut notes = Vec::new();
    if malformed > 0 {
        notes.push(format!("messages/messages.json: {} that {} not messages (no id, no words or no time) {} left out.", messages(malformed), were(malformed), were(malformed)));
    }
    if repeated > 0 {
        notes.push(format!("messages/messages.json: {} with an id that was already in the file {} left out.", messages(repeated), were(repeated)));
    }
    if over > 0 {
        notes.push(format!("messages/messages.json: the {} {} left out: OAIY keeps {MAX_STORED}.", if over == 1 { "oldest message".to_string() } else { format!("{over} oldest messages") }, were(over)));
    }
    if changed > 0 {
        notes.push(format!("messages/messages.json: {} had characters a message never holds taken out, or {} cut to the length the store keeps (a message {MAX_MESSAGE} characters, a name {MAX_NAME}).", messages(changed), were(changed)));
    }
    Ok((serde_json::to_vec_pretty(&json!({ "version": 1, "messages": kept })).map_err(|_| "it could not be written".to_string())?, notes))
}
