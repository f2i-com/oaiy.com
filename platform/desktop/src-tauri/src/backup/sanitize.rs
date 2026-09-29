//! Making what a backup carries safe to bring back, and safe to make.
//!
//! A backup is a file anyone could have made, and the data in it is read by programs that act on it.
//! So the files that carry more than data are cleaned, on the way in to a backup (so a secret is never
//! written) and again on the way out (so a hostile backup cannot plant one):
//!
//! - **Plugin settings** lose every key that names a PIN, a key, a token, a pairing or a sealed value,
//!   and every value that is sealed to a computer (DPAPI) or looks like a key. What a restore brings
//!   back of such a file is the backup's other settings plus the values this computer already holds.
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

/// Words that mark a key as holding something secret (matched on whole words of the key, split at
/// capitals, underscores and dashes: `managerPin` is `manager`, `pin`).
const SENSITIVE_WORDS: &[&str] = &[
    "pin", "secret", "secrets", "token", "tokens", "key", "keys", "password", "passwd", "credential", "credentials", "pair", "paired", "pairing", "manager", "auth",
    "seal", "sealed", "dpapi", "cookie", "cookies", "session", "private", "cert", "certificate", "bearer", "signature", "salt",
];

/// Pieces that mark a key wherever they sit in it (`apikey`, `clientsecret`).
const SENSITIVE_PIECES: &[&str] = &["apikey", "secret", "password", "passwd", "credential", "token", "dpapi", "cookie", "privatekey", "pairing"];

/// Value starts that mark a string as a secret or as sealed to a computer.
const SECRET_PREFIXES: &[&str] = &["dpapi", "sealed:", "flk_", "sk-", "hf_", "bearer ", "age-secret-key", "-----begin"];

fn key_words(key: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for c in key.chars() {
        if !c.is_alphanumeric() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            previous_lower = false;
            continue;
        }
        if c.is_uppercase() && previous_lower && !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
        previous_lower = c.is_lowercase() || c.is_ascii_digit();
        current.extend(c.to_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Whether a key names something that must not travel in a backup.
pub fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    SENSITIVE_PIECES.iter().any(|p| lower.contains(p)) || key_words(key).iter().any(|w| SENSITIVE_WORDS.contains(&w.as_str()))
}

/// Whether a string value is a secret, or sealed to a computer.
fn is_secret_value(value: &str) -> bool {
    let lower = value.trim_start().to_lowercase();
    SECRET_PREFIXES.iter().any(|p| lower.starts_with(p)) || lower.contains("dpapi:")
}

/// Take out of `value` every key that names a secret and every secret value, recording where each was
/// (as a dotted path).
pub fn strip_sensitive(value: &mut Value, path: &str, stripped: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            let doomed: Vec<String> = map
                .iter()
                .filter(|(k, v)| is_sensitive_key(k) || matches!(v, Value::String(s) if is_secret_value(s)))
                .map(|(k, _)| k.clone())
                .collect();
            for key in doomed {
                map.remove(&key);
                stripped.push(if path.is_empty() { key } else { format!("{path}.{key}") });
            }
            for (k, v) in map.iter_mut() {
                let child = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                strip_sensitive(v, &child, stripped);
            }
        }
        Value::Array(items) => {
            let before = items.len();
            items.retain(|v| !matches!(v, Value::String(s) if is_secret_value(s)));
            if items.len() != before {
                stripped.push(format!("{path}[]"));
            }
            for (i, v) in items.iter_mut().enumerate() {
                strip_sensitive(v, &format!("{path}[{i}]"), stripped);
            }
        }
        _ => {}
    }
}

/// Put back into `staged` the sensitive values `local` holds (the same keys at the same places), so
/// restoring a settings file does not take away this computer's own PIN or sign-in.
pub fn merge_sensitive(local: &Value, staged: &mut Value) {
    if let (Value::Object(local), Value::Object(staged)) = (local, staged) {
        for (k, lv) in local {
            if is_sensitive_key(k) || matches!(lv, Value::String(s) if is_secret_value(s)) {
                staged.insert(k.clone(), lv.clone());
            } else if let Some(sv) = staged.get_mut(k) {
                merge_sensitive(lv, sv);
            }
        }
    }
}

fn parse(bytes: &[u8]) -> Result<Value, String> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err("it is too large to be read".to_string());
    }
    serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)).map_err(|_| "it is not valid JSON".to_string())
}

/// A plugin's settings file, cleaned. Returns the bytes and what was taken out (dotted paths).
pub fn plugin_json(bytes: &[u8]) -> Result<(Vec<u8>, Vec<String>), String> {
    let mut value = parse(bytes)?;
    let mut stripped = Vec::new();
    strip_sensitive(&mut value, "", &mut stripped);
    Ok((serde_json::to_vec_pretty(&value).map_err(|_| "it could not be written".to_string())?, stripped))
}

/// A plugin's settings file, cleaned, with what `local` (this computer's own copy) holds of the same
/// sensitive keys put back.
pub fn plugin_json_with_local(bytes: &[u8], local: Option<&[u8]>) -> Result<(Vec<u8>, Vec<String>), String> {
    let mut value = parse(bytes)?;
    let mut stripped = Vec::new();
    strip_sensitive(&mut value, "", &mut stripped);
    if let Some(local) = local.and_then(|l| parse(l).ok()) {
        merge_sensitive(&local, &mut value);
    }
    Ok((serde_json::to_vec_pretty(&value).map_err(|_| "it could not be written".to_string())?, stripped))
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
