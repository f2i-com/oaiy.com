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
//! - **The calendar** loses its FormLogic sync state (the form it is paired with, the cursor, the list
//!   of records to delete, each appointment's remote copy, the tombstones of deleted ones): restored on
//!   a computer that is linked to another account, or to none, it would delete or duplicate records.
//! - **The provider list** loses its API keys unless the person asked for them.
//! - **The run journal** keeps only the runs that finished: a run that was waiting would start at the
//!   next start.
//! - **The autostart list** keeps only services whose template exists.

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

/// The calendar without its FormLogic sync state.
pub fn calendar_json(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut value = parse(bytes)?;
    if let Value::Object(book) = &mut value {
        book.remove("sync");
        book.remove("deleted");
        if let Some(Value::Array(appointments)) = book.get_mut("appointments") {
            for appointment in appointments {
                if let Value::Object(a) = appointment {
                    a.remove("formlogic");
                }
            }
        }
    }
    serde_json::to_vec_pretty(&value).map_err(|_| "it could not be written".to_string())
}

/// The provider list without its API keys. Returns the bytes and how many keys were taken out.
pub fn providers_without_keys(bytes: &[u8]) -> Result<(Vec<u8>, usize), String> {
    let mut value = parse(bytes)?;
    let mut removed = 0;
    if let Some(Value::Array(providers)) = value.get_mut("providers") {
        for p in providers {
            if let Value::Object(o) = p {
                for key in ["apiKey", "api_key"] {
                    if o.remove(key).is_some_and(|v| v.as_str().is_some_and(|s| !s.trim().is_empty())) {
                        removed += 1;
                    }
                }
            }
        }
    }
    Ok((serde_json::to_vec_pretty(&value).map_err(|_| "it could not be written".to_string())?, removed))
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
