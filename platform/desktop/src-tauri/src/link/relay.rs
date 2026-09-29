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
/// Called with `(connector, command, payload, idempotency key)`.
///
/// Boxed rather than a generic so the store can hold one without infecting every
/// type that touches it. Blocking: relay work is service and plugin control,
/// which is blocking anyway, and it runs on the relay's own thread.
///
/// The key is the relayed command's own id. It matters: a plugin refuses any
/// side-effecting command that arrives without one, precisely so that a
/// redelivered command cannot answer the call or send the message twice.
///
/// What a relayed command may reach is the dispatcher's to decide, so the relay
/// is always given [`super::ops::relay_dispatcher`], which asks the relay policy
/// first.
pub type Dispatcher =
    Arc<dyn Fn(&str, &str, &Value, &str) -> Result<Value, String> + Send + Sync>;

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

        let polled = std::time::Instant::now();
        match poll_once(&account, &spec, &instance, &dispatch) {
            // `trouble` is a poll that WORKED carrying commands that did not.
            // Recorded as the lane's state because from the provider's side it
            // is indistinguishable from a desktop that never polled at all.
            Ok((handled, trouble)) => {
                // A poll that works while its commands cannot be claimed or
                // answered (the provider half down) backs off like a failed
                // poll, rather than going straight round again.
                let backoff = trouble.is_some();
                store.note_relay(trouble);
                // A batch that did work is likely followed by more; go straight
                // back. An empty one already waited server-side.
                if backoff {
                    HTTP.start_afresh();
                    std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds));
                } else if handled == 0 {
                    std::thread::sleep(spec.idle_pause(polled.elapsed()));
                }
            }
            Err(e) => {
                store.note_relay(Some(e));
                // The try after a failure is on a new client, not one that has
                // seen the trouble.
                HTTP.start_afresh();
                // Backing off on failure keeps a provider that is down, or a key
                // that was revoked, from becoming a hot loop against it.
                std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds));
            }
        }
    });
}

/// The client this lane keeps from poll to poll, so that a poll, the claim it
/// leads to and the report share a connection instead of each making one.
static HTTP: super::net::LaneClient<reqwest::blocking::Client> =
    super::net::LaneClient::new(build_client);

/// How long a claim or a report may take. The poll's is longer, by the wait.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(20);

fn build_client() -> Result<reqwest::blocking::Client, String> {
    super::net::blocking_builder()
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
    let http = HTTP.get()?;
    let resp = http
        .get(&url)
        // Generous over the server's wait so a long-poll that returns exactly on
        // time is not cut off by our own client and retried needlessly.
        .timeout(Duration::from_secs(spec.wait_seconds + 15))
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the relay: {}", super::net::unreachable(&e)))?;

    let status = resp.status();
    if !status.is_success() {
        // Asked to slow down: wait as long as the provider said before the lane's own back-off.
        if status.as_u16() == 429 {
            if let Some(wait) = super::net::retry_after(resp.headers()) {
                std::thread::sleep(wait);
            }
        }
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
    let http = HTTP.get()?;
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
        .timeout(ANSWER_TIMEOUT)
        .bearer_auth(&account.credential)
        .json(&body)
        .send()
        .map_err(|e| format!("could not claim: {e}"))?;
    // Only the status is wanted; the rest is read out so the connection can
    // carry the report.
    let claim_status = claimed.status();
    super::net::drain(claimed);
    if !claim_status.is_success() {
        // 409 is the ordinary "someone else got it" and not worth reporting as
        // an error anywhere a user would see.
        return if claim_status.as_u16() == 409 {
            Ok(())
        } else {
            Err(format!("claim refused: HTTP {}", claim_status.as_u16()))
        };
    }

    let outcome = dispatch(
        command.connector_id.as_deref().unwrap_or("desktop"),
        &command.command,
        &command.payload,
        // The command's own id, which is stable across redelivery — the one
        // property an idempotency key has to have.
        &command.id,
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
    // The work is done (a text may have gone), so its outcome is worth a few
    // tries: lost, the provider shows the command as never picked up, and a
    // person asking again would do the work twice. Within the command's short
    // life, so no longer than a few seconds.
    let mut last = String::new();
    for wait in [0u64, 1, 3] {
        std::thread::sleep(Duration::from_secs(wait));
        match http
            .post(&complete_url)
            .timeout(ANSWER_TIMEOUT)
            .bearer_auth(&account.credential)
            .json(&report)
            .send()
        {
            Ok(done) => {
                let status = done.status();
                super::net::drain(done);
                // Ok even when the WORK failed. The command was answered, so the
                // user reads "no plugin named ghost" on the provider's page; calling
                // that a lane failure would put a red "not receiving commands" on
                // this panel every time somebody asks for something that does not exist.
                if status.is_success() {
                    return Ok(());
                }
                // Refused outright: another try says the same.
                if status.is_client_error() {
                    return Err(format!("the relay refused the outcome: HTTP {}", status.as_u16()));
                }
                last = format!("the relay refused the outcome: HTTP {}", status.as_u16());
            }
            Err(e) => last = format!("could not report the outcome: {}", super::net::unreachable(&e)),
        }
    }
    Err(last)
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
                           "payload": { "serviceId": "oaiy-voice" } }]
        }))
        .unwrap();
        assert_eq!(reply.commands.len(), 1);
        assert_eq!(reply.commands[0].payload["serviceId"], "oaiy-voice");

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
            idle_pause_ms: 500,
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
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value, _: &str| -> Result<Value, String> {
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
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value, _: &str| -> Result<Value, String> {
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
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value, _: &str| -> Result<Value, String> {
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
        let dispatch: Dispatcher = Arc::new(|_: &str, _: &str, _: &Value, _: &str| -> Result<Value, String> {
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

    #[test]
    fn a_command_the_policy_refuses_reaches_the_provider_as_a_failure_in_its_own_words() {
        // The whole path, from the provider's queue to the provider's page: a
        // command that installs a driver is claimed, refused by the relay policy
        // and reported as a completion with status "failed", carrying the sentence
        // the person is to read. It is answered, so the lane is not blamed, and the
        // plugin is never contacted (it is installed and not running: only its manifest
        // is read, to see that it does declare the command it is being refused).
        let dir = std::env::temp_dir().join(format!("oaiy-relay-refusal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plugin_dir = dir.join("plugins").join("aokie");
        std::fs::create_dir_all(plugin_dir.join("definitions")).unwrap();
        std::fs::write(
            plugin_dir.join("definitions/phone.json"),
            include_str!("../plugins/fixtures/aokie-phone.definition.json"),
        )
        .unwrap();
        std::fs::write(
            plugin_dir.join("manifest.json"),
            include_str!("../plugins/fixtures/aokie-v4.manifest.json"),
        )
        .unwrap();
        // Turned off, so the host's own autostart leaves it alone (there is no executable).
        std::fs::write(dir.join("plugins").join("disabled.json"), r#"["aokie"]"#).unwrap();
        let plugins = crate::plugins::registry::new_handle(dir.join("plugins"));
        plugins.lock().unwrap().scan();
        let triggers = Arc::new(std::sync::Mutex::new(crate::plugins::TriggerStore::load(
            dir.join("triggers.json"),
        )));
        let host = crate::plugins::PluginHost::new(
            plugins.clone(),
            crate::bridge::ledger::new_handle(),
            triggers,
            crate::bridge::deadletters::open_handle(dir.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );
        let services = Arc::new(std::sync::Mutex::new(crate::services::registry::Registry::empty(
            dir.join("data"),
            dir.join("models"),
        )));
        let dispatch = crate::link::ops::relay_dispatcher(
            services,
            plugins,
            host,
            crate::link::ops::RelayGuard::new(
                crate::link::policy::RelayPolicy::shipped(),
                dir.join("relay-log.jsonl"),
            ),
        );

        let (base, rx) = stub_relay(
            r#"{"commands":[{"commandId":"c9","connectorId":"aokie","command":"dongle.installDriver","payload":{"vid":4660,"pid":22136}}]}"#,
            "200 OK",
        );
        let (handled, trouble) = poll_once(&account(base), &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!(handled, 1);
        assert_eq!(trouble, None, "a refused command is an answer, not a failing lane");

        let _ = rx.recv().unwrap(); // the poll
        let (claim, _) = rx.recv().unwrap();
        assert!(claim.starts_with("POST /commands/c9/claim "), "{claim}");
        let (complete, raw) = rx.recv().unwrap();
        assert!(complete.starts_with("POST /commands/c9/complete "), "{complete}");
        assert!(raw.contains("\"status\":\"failed\""), "{raw}");
        assert!(raw.contains("dongle.installDriver"), "the command is named: {raw}");
        assert!(raw.contains("can only be run from OAIY on this computer"), "{raw}");

        let logged = std::fs::read_to_string(dir.join("relay-log.jsonl")).unwrap();
        assert!(logged.contains("c9") && logged.contains("refused"), "{logged}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- the connections it makes ---------------------------------------------------

    use crate::link::testkit::{Provider, Reply};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A relay that keeps its connections: one batch of work, once, then nothing.
    /// Its answer to a claim is padded, as a command's can be (a payload up to 16 KiB).
    fn keeping_relay(claim_padding: usize) -> Provider {
        let served = AtomicBool::new(false);
        Provider::start(move |req| {
            if req.target.starts_with("/pending") {
                if served.swap(true, Ordering::SeqCst) {
                    Reply::ok(r#"{"commands":[]}"#)
                } else {
                    Reply::ok(ONE_COMMAND)
                }
            } else if req.target.ends_with("/claim") {
                Reply::ok(&format!(r#"{{"claimed":true,"pad":"{}"}}"#, "x".repeat(claim_padding)))
            } else {
                Reply::ok("{}")
            }
        })
    }

    fn working_dispatcher() -> Dispatcher {
        Arc::new(|_: &str, _: &str, _: &Value, _: &str| -> Result<Value, String> {
            Ok(serde_json::json!({ "ok": true }))
        })
    }

    const POLL: &str = "GET /pending?wait=1000&limit=5&instanceId=oaiy-test";

    #[test]
    fn a_poll_the_claim_it_leads_to_the_report_and_the_next_poll_share_one_connection() {
        // A connection to the provider used to be made, and a TLS handshake done, for
        // each of these four requests.
        let server = keeping_relay(200_000);
        let linked = account(server.base.clone());
        let dispatch = working_dispatcher();

        let (handled, trouble) = poll_once(&linked, &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!((handled, trouble), (1, None));
        let (handled, trouble) = poll_once(&linked, &spec(), "oaiy-test", &dispatch).unwrap();
        assert_eq!((handled, trouble), (0, None));

        assert_eq!(
            server.lines(),
            [POLL, "POST /commands/c1/claim", "POST /commands/c1/complete", POLL]
        );
        assert_eq!(server.connections(), 1, "{:?}", server.lines());
    }

    #[test]
    fn a_lane_that_keeps_its_client_still_gives_each_account_only_its_own_credential() {
        // The credential is on the request, not on the client. So the connection
        // one account's poll leaves behind carries the next request, from another
        // account or to another provider, with that one's credential and no other.
        let (a, b) = (
            Provider::start(|_| Reply::ok(r#"{"commands":[]}"#)),
            Provider::start(|_| Reply::ok(r#"{"commands":[]}"#)),
        );
        let with = |server: &Provider, credential: &str| {
            let mut linked = account(server.base.clone());
            linked.credential = credential.into();
            poll_once(&linked, &spec(), "oaiy-test", &working_dispatcher()).unwrap();
        };
        with(&a, "flk_alpha");
        with(&b, "flk_beta");
        with(&a, "flk_gamma");
        with(&a, "flk_alpha");

        let bearers = |server: &Provider| -> Vec<String> {
            server.requests().iter().map(|r| r.header("authorization").unwrap_or("none").to_string()).collect()
        };
        assert_eq!(bearers(&a), ["Bearer flk_alpha", "Bearer flk_gamma", "Bearer flk_alpha"]);
        assert_eq!(bearers(&b), ["Bearer flk_beta"]);
        assert_eq!((a.connections(), b.connections()), (1, 1), "each provider's requests shared its connection");
    }

    // --- the pause after an empty poll -----------------------------------------------

    const PENDING: &str = "/api/v1/connector-commands/pending";

    /// Run the lane's own loop against a provider that has nothing for it, with the
    /// connector as `edit` leaves it, and say how far apart its first polls were.
    fn gaps_between_empty_polls(tag: &str, edit: impl FnOnce(&mut RelaySpec)) -> Vec<Duration> {
        let server = Provider::start(|_| Reply::ok(r#"{"commands":[]}"#));
        let (store, dir) = crate::link::testkit::linked_to(&server.base, tag, |d| edit(d.relay.as_mut().unwrap()));
        spawn(store, working_dispatcher());
        let polls = server.wait_for(PENDING, 3, Duration::from_secs(30));
        let _ = std::fs::remove_dir_all(dir);
        polls.windows(2).map(|pair| pair[1].at - pair[0].at).collect()
    }

    #[test]
    fn the_lane_waits_after_an_empty_poll_as_long_as_its_descriptor_says() {
        // The lane's own loop, not a function beside it: a provider that asks for a
        // three second pause is not polled again for three seconds. (Every lane
        // used to wait half a second, or the rest of two seconds if the provider
        // cut its poll short, and there was nowhere to say otherwise.)
        let gaps = gaps_between_empty_polls("relay-slow", |relay| {
            relay.wait_seconds = 1;
            relay.idle_pause_ms = 3_000;
        });
        for gap in gaps {
            assert!(gap >= Duration::from_millis(2_800) && gap < Duration::from_secs(6), "{gap:?}");
        }
    }

    #[test]
    fn a_descriptor_that_says_nothing_about_the_pause_polls_as_often_as_the_lane_always_did() {
        // A provider that answers at once, and a descriptor with no pause in it: two
        // seconds between polls, as before.
        let gaps = gaps_between_empty_polls("relay-default", |_| {});
        for gap in gaps {
            assert!(gap >= Duration::from_millis(1_800) && gap < Duration::from_millis(2_800), "{gap:?}");
        }
    }

    #[test]
    fn a_redirect_is_followed_as_it_always_was_but_the_credential_does_not_go_with_it() {
        // The redirect policy of this lane is the client's default, unchanged: a
        // provider that moves the queue is followed. What is not followed is the
        // bearer, which reqwest drops on the way to another origin.
        let elsewhere = Provider::start(|_| Reply::ok(r#"{"commands":[]}"#));
        let moved_to = format!("{}/moved", elsewhere.base);
        let origin = Provider::start(move |_| Reply::redirect(307, &moved_to));

        let (handled, trouble) =
            poll_once(&account(origin.base.clone()), &spec(), "oaiy-test", &working_dispatcher()).unwrap();
        assert_eq!((handled, trouble), (0, None), "the answer from where it was sent");

        assert_eq!(origin.requests()[0].header("authorization"), Some("Bearer flk_secret"));
        let followed = elsewhere.requests();
        assert_eq!(followed.len(), 1);
        assert!(followed[0].target.starts_with("/moved"), "{}", followed[0].target);
        assert_eq!(followed[0].header("authorization"), None, "the credential stayed with the provider");
    }
}
