//! Leases: one holder at a time for a job several pages could each do.
//!
//! Every OAIY page follows the same event ring, so without this two of them
//! (OAIY's own window and the app open in a browser tab) would both answer the
//! same text message. A page that answers holds the `answer-texts` lease and
//! renews it while it runs; a page that does not hold it leaves the job alone.
//! OAIY's own window asks with `prefer`, and takes the lease over.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

/// The longest a lease is granted for: a holder that stops renewing loses it this soon.
const MAX_TTL_MS: u64 = 120_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Lease {
    pub holder: String,
    pub expires_at_ms: u64,
}

static LEASES: Mutex<Option<HashMap<String, Lease>>> = Mutex::new(None);

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Take or renew `name` for `holder`: granted when it is free, expired, already
/// the holder's, or `prefer` is set (taken over). Returns the lease as it now stands.
pub fn take(leases: &mut HashMap<String, Lease>, name: &str, holder: &str, ttl_ms: u64, prefer: bool, now: u64) -> (bool, Lease) {
    let ttl = ttl_ms.clamp(1_000, MAX_TTL_MS);
    let free = match leases.get(name) {
        None => true,
        Some(l) => l.holder == holder || l.expires_at_ms <= now,
    };
    if free || prefer {
        let lease = Lease { holder: holder.to_string(), expires_at_ms: now + ttl };
        leases.insert(name.to_string(), lease.clone());
        (true, lease)
    } else {
        (false, leases[name].clone())
    }
}

/// Who holds `name` now (None when no one does, or their lease lapsed).
pub fn holder(name: &str) -> Option<String> {
    let guard = LEASES.lock().unwrap_or_else(|e| e.into_inner());
    let lease = guard.as_ref()?.get(name)?;
    (lease.expires_at_ms > now_ms()).then(|| lease.holder.clone())
}

/// Give `name` up, if `holder` has it.
pub fn release(leases: &mut HashMap<String, Lease>, name: &str, holder: &str) -> bool {
    if leases.get(name).is_some_and(|l| l.holder == holder) {
        leases.remove(name);
        true
    } else {
        false
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TakeLease {
    holder: String,
    #[serde(default)]
    ttl_ms: Option<u64>,
    #[serde(default)]
    prefer: bool,
    /// Give it up rather than take it.
    #[serde(default)]
    release: bool,
}

fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

/// `POST /api/bridge/leases/:name {holder, ttlMs?, prefer?, release?}` →
/// `{name, granted, holder, expiresAtMs}`.
pub async fn take_lease(Path(name): Path<String>, Json(body): Json<TakeLease>) -> axum::response::Response {
    if !valid_name(&name) || body.holder.is_empty() || body.holder.len() > 200 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": {"code": "invalid_request", "message": "a lease name (letters, digits, - _ .) and a holder"}}))).into_response();
    }
    let mut guard = LEASES.lock().unwrap_or_else(|e| e.into_inner());
    let leases = guard.get_or_insert_with(HashMap::new);
    if body.release {
        let released = release(leases, &name, &body.holder);
        return (StatusCode::OK, Json(json!({"name": name, "released": released}))).into_response();
    }
    let (granted, lease) = take(leases, &name, &body.holder, body.ttl_ms.unwrap_or(30_000), body.prefer, now_ms());
    (StatusCode::OK, Json(json!({"name": name, "granted": granted, "holder": lease.holder, "expiresAtMs": lease.expires_at_ms}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_holder_at_a_time_until_it_lapses_or_is_taken_over() {
        let mut l = HashMap::new();
        assert!(take(&mut l, "answer-texts", "tab", 30_000, false, 1_000).0);
        // Another page is refused while the first holds it…
        let (granted, lease) = take(&mut l, "answer-texts", "window", 30_000, false, 2_000);
        assert!(!granted);
        assert_eq!(lease.holder, "tab");
        // …the holder renews it…
        assert!(take(&mut l, "answer-texts", "tab", 30_000, false, 20_000).0);
        // …it lapses when not renewed…
        assert!(take(&mut l, "answer-texts", "window", 30_000, false, 60_000).0);
        // …and OAIY's own window takes it over when it prefers to.
        assert!(take(&mut l, "answer-texts", "tab", 30_000, false, 61_000).0 == false);
        assert!(take(&mut l, "answer-texts", "tab", 30_000, true, 62_000).0);
        assert!(release(&mut l, "answer-texts", "tab"));
        assert!(!release(&mut l, "answer-texts", "tab"));
    }

    #[test]
    fn a_lease_lasts_at_most_two_minutes() {
        let mut l = HashMap::new();
        let (_, lease) = take(&mut l, "x", "a", 10_000_000, false, 0);
        assert_eq!(lease.expires_at_ms, MAX_TTL_MS);
        assert!(!valid_name("a/b"));
        assert!(valid_name("answer-texts"));
    }
}
