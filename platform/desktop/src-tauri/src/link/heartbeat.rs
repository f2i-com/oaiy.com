//! Telling the provider this desktop is still here.
//!
//! A link creates a connection record, but records go stale. Providers decide
//! whether a desktop is reachable from how recently it last spoke — FormLogic's
//! window is 90 seconds — so without a periodic ping the app links successfully
//! and then shows as offline a minute later, which is exactly what "No Desktop"
//! means.
//!
//! Descriptor-driven like everything else here: the path, the interval and the
//! field names come from the connector, so a provider with a different presence
//! endpoint needs no code. A connector that declares no heartbeat simply never
//! gets one.
//!
//! One long-lived thread that asks the store what to do on each tick, rather
//! than a thread started and stopped around each link. Linking and unlinking are
//! then ordinary state changes instead of lifecycle events something has to
//! subscribe to — and there is no way to leak a thread per link attempt.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::descriptor::{self, HeartbeatSpec};
use super::{LinkHandle, LinkedAccount};

/// How often the worker wakes to see whether a beat is due.
///
/// Independent of any connector's interval: it only has to be fine-grained
/// enough that the first beat after linking is prompt.
const TICK: Duration = Duration::from_secs(5);

/// Send the first beat this soon after a link, rather than waiting a full
/// interval — the provider's presence window starts counting immediately, and a
/// user watching the UI wants to see it come online now.
const FIRST_BEAT_DELAY: Duration = Duration::from_secs(2);

pub fn spawn(store: LinkHandle) {
    std::thread::spawn(move || {
        let mut last_beat: Option<(String, Instant)> = None;
        loop {
            std::thread::sleep(TICK);
            let Some(account) = store.account() else {
                // Unlinked: forget the schedule so a future link beats promptly
                // instead of inheriting an old timer.
                last_beat = None;
                continue;
            };
            let Some(spec) = heartbeat_spec(&store, &account) else {
                continue;
            };

            let key = account.account_id.clone().unwrap_or_else(|| account.base_url.clone());
            let due = match &last_beat {
                Some((k, at)) if *k == key => {
                    at.elapsed() >= Duration::from_secs(spec.interval_seconds)
                }
                // A different account (or none yet) — beat almost at once.
                _ => {
                    last_beat = Some((key.clone(), Instant::now() - FIRST_BEAT_DELAY));
                    false
                }
            };
            let first = matches!(&last_beat, Some((k, at))
                if *k == key && at.elapsed() >= FIRST_BEAT_DELAY && at.elapsed() < Duration::from_secs(spec.interval_seconds));
            if !due && !first {
                continue;
            }

            match send(&account, &spec, &store.instance_id()) {
                Ok(()) => store.note_heartbeat(None),
                // Recorded, not retried harder: a provider that is down will be
                // up again on the next tick, and a burst of retries would only
                // make a bad moment worse. The status carries the reason so the
                // panel can say why it looks offline.
                Err(e) => store.note_heartbeat(Some(e)),
            }
            last_beat = Some((key, Instant::now()));
        }
    });
}

fn heartbeat_spec(store: &LinkHandle, account: &LinkedAccount) -> Option<HeartbeatSpec> {
    descriptor::find(store.data_dir(), &account.connector_id)?.heartbeat
}

/// One beat. Blocking; the worker thread exists for this.
fn send(account: &LinkedAccount, spec: &HeartbeatSpec, instance_id: &str) -> Result<(), String> {
    let url = super::oauth::join(&account.base_url, &spec.path);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("could not build the heartbeat client: {e}"))?;

    let mut body = serde_json::Map::new();
    body.insert(
        spec.instance_id_field.clone(),
        serde_json::Value::String(instance_id.to_string()),
    );
    if let (Some(field), Some(name)) = (spec.device_name_field.as_ref(), account.account_name.as_ref())
    {
        body.insert(field.clone(), serde_json::Value::String(name.clone()));
    }

    let resp = client
        .post(&url)
        .bearer_auth(&account.credential)
        .json(&serde_json::Value::Object(body))
        .send()
        .map_err(|e| format!("could not reach {}: {e}", account.base_url))?;

    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    // 401/403 is the one worth naming: it means the key was revoked at the
    // provider, so the link is dead and re-linking is the fix — not something a
    // user would guess from "offline".
    let detail: serde_json::Value = resp.json().unwrap_or(serde_json::Value::Null);
    let message = detail
        .get("message")
        .or_else(|| detail.get("error"))
        .and_then(|v| v.as_str())
        .unwrap_or("the provider refused the heartbeat");
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(format!(
            "the provider no longer accepts this desktop's key ({message}) — link again"
        ));
    }
    Err(format!("HTTP {}: {message}", status.as_u16()))
}

/// A stable per-install id, in the shape providers accept.
///
/// Letters, digits, `.`, `_`, `-` only — FormLogic rejects anything else, and
/// that character set is the common denominator across the providers seen so
/// far. Generated once and persisted: a changing id would look like a new
/// desktop on every launch and accumulate ghost rows.
pub fn new_instance_id() -> String {
    format!("oaiy-{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_instance_id_is_accepted_by_the_providers_character_rule() {
        // FormLogic: ^[A-Za-z0-9._-]+$, max 128. A uuid with braces or a
        // hostname with a space would be refused and the desktop would never
        // appear online — with a 400 nobody reads.
        for _ in 0..20 {
            let id = new_instance_id();
            assert!(!id.is_empty() && id.len() <= 128, "{id}");
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')),
                "{id}"
            );
        }
    }

    #[test]
    fn instance_ids_are_unique_per_install() {
        assert_ne!(new_instance_id(), new_instance_id());
    }
}
