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
pub type Dispatcher = Arc<dyn Fn(&str, &str, &Value) -> Result<Value, String> + Send + Sync>;

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
    /// The short verb — `services.list`, `plugins.start`, `call.current`. The
    /// connector id is the namespace, so it is not repeated here.
    command: String,
    /// WHOSE verb it is. `desktop` means this app itself; anything else names a
    /// plugin's connector. Ignoring this and matching the verb alone was a real
    /// bug: every `aokie` command was measured against the desktop's own list
    /// and refused, so the provider's call console got nothing back.
    #[serde(default)]
    connector_id: Option<String>,
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
            // `trouble` is a poll that WORKED carrying commands that did not.
            // Recorded as the lane's state because from the provider's side it
            // is indistinguishable from a desktop that never polled at all.
            Ok((handled, trouble)) => {
                store.note_relay(trouble);
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

/// One long-poll and its work.
///
/// Returns how many commands were handled, and the first that could not be
/// served at all — the claim or the report failing, not the work failing, which
/// is answered rather than swallowed.
fn poll_once(
    account: &LinkedAccount,
    spec: &RelaySpec,
    instance: &str,
    dispatch: &Dispatcher,
) -> Result<(usize, Option<String>), String> {
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
    let mut trouble: Option<String> = None;
    for command in reply.commands {
        // One failure must not abandon the rest of the batch: they are
        // independent actions a user is waiting on.
        if let Err(e) = serve(account, spec, instance, dispatch, &command) {
            log::warn!("relay command {} could not be served: {e}", command.id);
            // Kept, not just logged. This is the failure that looks like
            // nothing: the poll succeeded so the lane appears healthy, while
            // every command the user issues expires unanswered.
            trouble.get_or_insert(e);
        }
        handled += 1;
    }
    Ok((handled, trouble))
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

    let outcome = dispatch(
        command.connector_id.as_deref().unwrap_or("desktop"),
        &command.command,
        &command.payload,
    );
    if let Err(message) = &outcome {
        // Logged so the same failure is diagnosable from this side too, but see
        // the return below: it is not the lane's failure.
        log::info!("relay command {} ({}) failed: {message}", command.id, command.command);
    }
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
    // Ok even when the WORK failed. The command was answered, so the user reads
    // "no plugin named ghost" on the provider's page; calling that a lane
    // failure would put a red "not receiving commands" on this panel every time
    // somebody asks for something that does not exist.
    Ok(())
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
    fn a_command_carries_whose_verb_it_is() {
        // The bug this pins, seen live: the provider's call console polls
        // `call.current` on the `aokie` connector three times a second. Reading
        // the verb without the connector measured every one against the
        // desktop's own op list and refused it — so the console got nothing
        // back for an entire call, while the call itself connected fine.
        let reply: PendingReply = serde_json::from_value(serde_json::json!({
            "commands": [
                { "commandId": "c1", "connectorId": "aokie", "command": "call.current" },
                { "commandId": "c2", "connectorId": "desktop", "command": "services.list" },
                { "commandId": "c3", "command": "plugins.list" }
            ]
        }))
        .unwrap();
        assert_eq!(reply.commands[0].connector_id.as_deref(), Some("aokie"));
        assert_eq!(reply.commands[1].connector_id.as_deref(), Some("desktop"));
        // Absent means this app itself — the shape older rows have.
        assert_eq!(reply.commands[2].connector_id, None);
        let effective = |c: &Command| c.connector_id.clone().unwrap_or_else(|| "desktop".into());
        assert_eq!(effective(&reply.commands[2]), crate::link::ops::DESKTOP_CONNECTOR);
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

    /// A stub relay: one batch of pending work, then claim and complete.
    ///
    /// Hand-rolled HTTP like the companion tests. These assertions are about the
    /// claim/complete sequence and what it reports, not about a server.
    ///
    /// Returns the base URL and a channel of `(request line, whole request)` in
    /// the order they arrived, so a test can assert on what was NOT sent too.
    fn stub_relay(
        batch: &'static str,
        claim_status: &'static str,
    ) -> (String, std::sync::mpsc::Receiver<(String, String)>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut served_batch = false;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { return };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(head_end) = text.find("\r\n\r\n") {
                        let len: usize = text
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length: ")
                                    .or_else(|| l.strip_prefix("Content-Length: "))
                            })
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        if raw.len() >= head_end + 4 + len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&raw).to_string();
                let line = text.lines().next().unwrap_or_default().to_string();
                let (status, reply) = if line.contains("/pending") {
                    // Once. A second poll must not re-serve work already done.
                    if std::mem::replace(&mut served_batch, true) {
                        ("200 OK", r#"{"commands":[]}"#.to_string())
                    } else {
                        ("200 OK", batch.to_string())
                    }
                } else if line.contains("/claim") {
                    (claim_status, "{}".to_string())
                } else {
                    ("200 OK", "{}".to_string())
                };
                let _ = tx.send((line, text));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.flush();
            }
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    fn spec() -> RelaySpec {
        RelaySpec {
            pending_path: "/pending".into(),
            claim_path: "/commands/{id}/claim".into(),
            complete_path: "/commands/{id}/complete".into(),
            wait_seconds: 1,
            batch_limit: 5,
            error_backoff_seconds: 1,
        }
    }

    fn account(base: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: base,
            credential: "flk_secret".into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("oaiy-test".into()),
        }
    }

    const ONE_COMMAND: &str =
        r#"{"commands":[{"commandId":"c1","command":"plugins.start","payload":{"pluginId":"ghost"}}]}"#;

    #[test]
    fn work_that_fails_is_answered_and_is_not_blamed_on_the_lane() {
        // Two things at once, because they are the same mistake in both
        // directions: the provider must be TOLD the command failed (silence
        // makes it expire as "no desktop picked it up"), and this desktop must
        // NOT then report its own lane as broken — asking to start a plugin
        // that does not exist is a normal answer, not a connection fault.
        let (base, rx) = stub_relay(ONE_COMMAND, "200 OK");
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value| -> Result<Value, String> {
            Err("no plugin named \"ghost\"".to_string())
        });

        let (handled, trouble) =
            poll_once(&account(base), &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!(handled, 1);
        assert_eq!(trouble, None, "a failed command is not a failing lane");

        let (poll, _) = rx.recv().unwrap();
        assert!(poll.starts_with("GET /pending?"), "{poll}");
        let (claim, _) = rx.recv().unwrap();
        assert!(claim.starts_with("POST /commands/c1/claim "), "{claim}");
        // The id is substituted, and the outcome carries the real reason.
        let (complete, raw) = rx.recv().unwrap();
        assert!(complete.starts_with("POST /commands/c1/complete "), "{complete}");
        assert!(raw.contains("\"status\":\"failed\""), "{raw}");
        assert!(raw.contains("no plugin named"), "the reason must survive: {raw}");
    }

    #[test]
    fn a_claim_that_is_refused_becomes_something_the_panel_can_show() {
        // The gap this closes. The poll succeeds, so nothing here looks wrong,
        // while on the provider's website every action expires unanswered and
        // the user is sent hunting for a connection problem that is not there.
        let (base, _rx) = stub_relay(ONE_COMMAND, "500 Internal Server Error");
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value| -> Result<Value, String> {
            panic!("work must not run without a claim")
        });

        let (handled, trouble) =
            poll_once(&account(base), &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!(handled, 1);
        let reported = trouble.expect("a refused claim must reach the status");
        assert!(reported.contains("claim refused"), "{reported}");
        assert!(reported.contains("500"), "the code is the diagnosis: {reported}");
    }

    #[test]
    fn losing_a_race_for_a_command_is_not_reported_as_trouble() {
        // 409 is the ordinary outcome when the same account has two desktops:
        // the other one got there first. Surfacing that as a lane failure would
        // put a red banner on a machine that is working perfectly.
        let (base, rx) = stub_relay(ONE_COMMAND, "409 Conflict");
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value| -> Result<Value, String> {
            panic!("a lost claim must not run the work")
        });

        let (_, trouble) = poll_once(&account(base), &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!(trouble, None);

        let _ = rx.recv().unwrap(); // the poll
        let (claim, _) = rx.recv().unwrap();
        assert!(claim.contains("/claim"), "{claim}");
        // …and nothing was reported, because there was nothing to report on.
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "a command we never claimed must not be completed"
        );
    }

    #[test]
    fn the_credential_is_presented_on_every_leg_and_never_in_the_url() {
        // Claim and complete are separate requests from the poll; a bearer
        // missing on either one turns into a 401 that reads as a dead link.
        let (base, rx) = stub_relay(ONE_COMMAND, "200 OK");
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value| -> Result<Value, String> {
            Ok(serde_json::json!({ "ok": true }))
        });
        poll_once(&account(base), &spec(), "oaiy-test", &dispatch).unwrap();

        for leg in ["poll", "claim", "complete"] {
            let (line, raw) = rx.recv().unwrap();
            assert!(raw.contains("flk_secret"), "the {leg} leg carried no bearer");
            // URLs end up in access logs and error messages; headers do not.
            assert!(!line.contains("flk_secret"), "the {leg} leg put it in the URL: {line}");
        }
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
