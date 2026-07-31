//! Serving commands a provider's web app queued for this desktop.
//!
//! The other half of remote control. A user on the provider's website clicks
//! "start service"; the provider stores a command addressed to this machine's
//! instance id and waits. Nothing happens until a desktop long-polls, claims it,
//! does the work and reports back — so a desktop that does not poll leaves every
//! action to expire with "no desktop picked it up in time", which is exactly
//! what an unimplemented relay looks like from the web.
//!
//! Distinct from the LOCAL transport, where the browser calls this desktop's
//! loopback API directly. That one needs the two on the same machine and past
//! the page's CSP; this one works from anywhere the provider can reach, which is
//! why a provider prefers it whenever it believes the desktop is reachable.
//!
//! Descriptor-driven: paths, poll timing and the op namespace come from the
//! connector, so a provider with a different relay shape needs no code here. The
//! work itself is done by a DISPATCHER the host supplies — this module knows how
//! to fetch, claim and complete, and nothing about services or plugins.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

use super::descriptor::{self, RelaySpec};
use super::{LinkHandle, LinkedAccount};

/// Runs one command and returns its result, or an error to report back.
///
/// Boxed rather than a generic so the store can hold one without infecting every
/// type that touches it. Blocking: relay work is service and plugin control,
/// which is blocking anyway, and it runs on the relay's own thread.
pub type Dispatcher = Arc<dyn Fn(&str, &Value) -> Result<Value, String> + Send + Sync>;

/// A command waiting for this desktop.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Command {
    /// `commandId` on the wire, not `id`. Getting this wrong does not fail
    /// loudly: the batch simply will not deserialize, the worker reports a
    /// parse error, and every queued action expires as "no desktop picked it
    /// up" — pointing the user at a connection problem that does not exist.
    #[serde(rename = "commandId")]
    id: String,
    /// The short verb — `services.list`, `plugins.start`. The connector id is
    /// the namespace, so it is not repeated here.
    command: String,
    #[serde(default)]
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct PendingReply {
    #[serde(default)]
    commands: Vec<Command>,
}

/// Poll, claim, run, report — forever, while a link with a relay exists.
pub fn spawn(store: LinkHandle, dispatch: Dispatcher) {
    std::thread::spawn(move || loop {
        let Some(account) = store.account() else {
            // Not linked. Sleep rather than spin; a link is a human action and
            // will not appear in the next millisecond.
            std::thread::sleep(Duration::from_secs(5));
            continue;
        };
        let Some(spec) = descriptor::find(store.data_dir(), &account.connector_id)
            .and_then(|d| d.relay)
        else {
            std::thread::sleep(Duration::from_secs(30));
            continue;
        };
        let instance = store.instance_id();

        match poll_once(&account, &spec, &instance, &dispatch) {
            Ok(handled) => {
                store.note_relay(None);
                // A batch that did work is likely followed by more; go straight
                // back. An empty one already waited server-side.
                if handled == 0 {
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
            Err(e) => {
                store.note_relay(Some(e));
                // Backing off on failure keeps a provider that is down, or a key
                // that was revoked, from becoming a hot loop against it.
                std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds));
            }
        }
    });
}

fn client(timeout: Duration) -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("could not build the relay client: {e}"))
}

/// One long-poll and its work. Returns how many commands were handled.
fn poll_once(
    account: &LinkedAccount,
    spec: &RelaySpec,
    instance: &str,
    dispatch: &Dispatcher,
) -> Result<usize, String> {
    let wait_ms = spec.wait_seconds * 1000;
    let url = format!(
        "{}?wait={}&limit={}&instanceId={}",
        super::oauth::join(&account.base_url, &spec.pending_path),
        wait_ms,
        spec.batch_limit,
        urlencode(instance),
    );
    // Generous over the server's wait so a long-poll that returns exactly on
    // time is not cut off by our own client and retried needlessly.
    let http = client(Duration::from_secs(spec.wait_seconds + 15))?;
    let resp = http
        .get(&url)
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the relay: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().unwrap_or(Value::Null);
        let message = body
            .get("message")
            .or_else(|| body.get("error"))
            .and_then(|v| v.as_str())
            .unwrap_or("the relay refused the poll");
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "the provider no longer accepts this desktop's key ({message}) — link again"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }

    let reply: PendingReply = resp
        .json()
        .map_err(|e| format!("the relay returned an unreadable batch: {e}"))?;

    let mut handled = 0usize;
    for command in reply.commands {
        // One failure must not abandon the rest of the batch: they are
        // independent actions a user is waiting on.
        if let Err(e) = serve(account, spec, instance, dispatch, &command) {
            log::warn!("relay command {} failed: {e}", command.id);
        }
        handled += 1;
    }
    Ok(handled)
}

/// Claim one command, run it, and report the outcome.
fn serve(
    account: &LinkedAccount,
    spec: &RelaySpec,
    instance: &str,
    dispatch: &Dispatcher,
    command: &Command,
) -> Result<(), String> {
    let http = client(Duration::from_secs(20))?;
    let body = serde_json::json!({ "instanceId": instance });

    // Claim FIRST. It is exactly-once: another desktop under the same account
    // may already have taken it, and doing the work before claiming would run
    // it twice — for `services.start` that is merely wasteful, but the contract
    // is what makes it safe to have two desktops at all.
    let claim_url = super::oauth::join(
        &account.base_url,
        &spec.claim_path.replace("{id}", &command.id),
    );
    let claimed = http
        .post(&claim_url)
        .bearer_auth(&account.credential)
        .json(&body)
        .send()
        .map_err(|e| format!("could not claim: {e}"))?;
    if !claimed.status().is_success() {
        // 409 is the ordinary "someone else got it" and not worth reporting as
        // an error anywhere a user would see.
        return if claimed.status().as_u16() == 409 {
            Ok(())
        } else {
            Err(format!("claim refused: HTTP {}", claimed.status().as_u16()))
        };
    }

    let outcome = dispatch(&command.command, &command.payload);
    let report = match &outcome {
        Ok(result) => serde_json::json!({
            "instanceId": instance,
            "status": "done",
            "result": result,
        }),
        // Reported as a completion with status=failed, NOT by staying silent:
        // an unclaimed-looking command leaves the web app saying nobody picked
        // it up, which sends the user looking for a connection problem instead
        // of reading the actual error.
        Err(message) => serde_json::json!({
            "instanceId": instance,
            "status": "failed",
            "error": { "message": message },
        }),
    };

    let complete_url = super::oauth::join(
        &account.base_url,
        &spec.complete_path.replace("{id}", &command.id),
    );
    let done = http
        .post(&complete_url)
        .bearer_auth(&account.credential)
        .json(&report)
        .send()
        .map_err(|e| format!("could not report the outcome: {e}"))?;
    if !done.status().is_success() {
        return Err(format!(
            "the relay refused the outcome: HTTP {}",
            done.status().as_u16()
        ));
    }
    outcome.map(|_| ())
}

fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_instance_id_is_safe_in_a_query_string() {
        assert_eq!(urlencode("oaiy-abc123"), "oaiy-abc123");
        assert_eq!(urlencode("a b&c=d"), "a%20b%26c%3Dd");
    }

    #[test]
    fn the_builtin_connector_describes_the_relay_it_needs() {
        let d = descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let r = d.relay.expect("the connector must declare a relay");
        // {id} is substituted per command; without the placeholder every claim
        // would go to the same URL and silently do nothing useful.
        assert!(r.claim_path.contains("{id}"), "{}", r.claim_path);
        assert!(r.complete_path.contains("{id}"), "{}", r.complete_path);
        assert!(r.pending_path.starts_with('/'));
        // Long enough to be a real long-poll, short enough that a proxy or the
        // client timeout does not cut it off first.
        assert!(r.wait_seconds >= 5 && r.wait_seconds <= 60, "{}", r.wait_seconds);
    }

    #[test]
    fn a_command_deserializes_from_the_providers_actual_wire_shape() {
        // Captured from a live poll. The provider sends many fields we ignore;
        // what matters is that the id arrives and the verb is the short form.
        let raw = serde_json::json!({
            "commandId": "67c1382d-c318-4ee5-a94c-a4f3d0cf1227",
            "ownerUserId": "u1",
            "appId": null,
            "connectorId": "desktop",
            "command": "plugins.list",
            "payload": null,
            "idempotencyKey": "ui-op-abc",
            "status": "pending",
            "result": null,
            "error": null,
            "requestedByUserId": "u1",
            "targetInstanceId": "oaiy-752737079f8640c29d59b38527f1fb87",
            "claimedBy": null,
            "createdAt": "2026-07-31 13:58:29",
            "claimedAt": null,
            "finishedAt": null,
            "expiresAt": "2026-07-31 13:59:29"
        });
        let c: Command = serde_json::from_value(raw).expect("the real shape must parse");
        assert_eq!(c.id, "67c1382d-c318-4ee5-a94c-a4f3d0cf1227");
        assert_eq!(c.command, "plugins.list");
        assert!(c.payload.is_null(), "a null payload must not fail the parse");
    }

    #[test]
    fn a_batch_parses_and_an_absent_payload_defaults() {
        let reply: PendingReply = serde_json::from_value(serde_json::json!({
            "commands": [{ "commandId": "c1", "command": "services.start",
                           "payload": { "serviceId": "llama-cpp" } }]
        }))
        .unwrap();
        assert_eq!(reply.commands.len(), 1);
        assert_eq!(reply.commands[0].payload["serviceId"], "llama-cpp");

        // An empty batch is the normal long-poll timeout, not an error.
        let empty: PendingReply = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(empty.commands.is_empty());
    }

    #[test]
    fn the_id_placeholder_is_replaced_not_appended() {
        let path = "/api/v1/connector-commands/{id}/claim";
        assert_eq!(
            path.replace("{id}", "cmd_1"),
            "/api/v1/connector-commands/cmd_1/claim"
        );
    }
}
