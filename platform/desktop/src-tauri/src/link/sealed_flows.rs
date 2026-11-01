//! Outbound-only, account-linked flow relay. The provider routes ciphertext;
//! only this machine opens inputs and only the requesting browser opens results.
//! A successful claim precedes execution. Lost completions retry the same sealed
//! result, never the flow (which may already have performed side effects).

use super::{
    descriptor::{self, RelaySpec},
    LinkHandle, LinkedAccount,
};
use crate::{
    ai::e2e::{E2eIdentity, E2eSessions},
    services::node_runtime::NodeHandle,
};
use reqwest::blocking::{Client, Response};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{io::Read, time::Duration};

const MAX_WIRE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_RESULT_BYTES: usize = 192 * 1024;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    request_id: String,
    flow_id: String,
    target_instance_id: Option<String>,
    eph_pub: String,
    envelope: String,
}

#[derive(Deserialize)]
struct Pending {
    requests: Vec<Request>,
}

#[derive(Deserialize)]
struct Claim {
    request: Request,
    claimed: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Inputs {
    v: u8,
    flow_id: String,
    inputs: Value,
}

fn read_json<T: serde::de::DeserializeOwned>(response: Response) -> Result<T, String> {
    let mut bytes = Vec::new();
    response
        .take(MAX_WIRE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "could not read the sealed flow reply")?;
    if bytes.len() as u64 > MAX_WIRE_BYTES {
        return Err("sealed flow reply is too large".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid sealed flow reply".into())
}

fn open_inputs(
    identity: &E2eIdentity,
    sessions: &E2eSessions,
    request: &Request,
) -> Result<Value, String> {
    let bytes = sessions
        .open_inbound(
            identity,
            &request.request_id,
            &request.eph_pub,
            &request.envelope,
        )
        .map_err(|_| "The encrypted flow request could not be authenticated.".to_string())?;
    let body: Inputs = serde_json::from_slice(&bytes)
        .map_err(|_| "The encrypted flow inputs are invalid.".to_string())?;
    if body.v != 1 || body.flow_id != request.flow_id || !body.inputs.is_object() {
        return Err("The encrypted flow identity or inputs do not match this request.".into());
    }
    Ok(body.inputs)
}

fn terminal(outcome: Result<Value, String>) -> (bool, Value) {
    match outcome {
        Ok(result) => {
            let body = json!({"v": 1, "type": "flow_result", "result": result});
            if body.to_string().len() <= MAX_RESULT_BYTES {
                return (true, body);
            }
            terminal(Err(
                "The flow result exceeds the relay's 192 KiB limit.".into()
            ))
        }
        Err(message) => (
            false,
            json!({"v": 1, "type": "error", "code": "node_failed",
            "message": message.chars().take(2000).collect::<String>()}),
        ),
    }
}

// The execute callback is also the integration-test seam: tests exercise the
// real HTTP/claim/crypto contract without invoking a user's automation.
fn poll_once(
    http: &Client,
    account: &LinkedAccount,
    spec: &RelaySpec,
    instance: &str,
    identity: &E2eIdentity,
    still_linked: &impl Fn() -> bool,
    execute: &impl Fn(&str, Value) -> Result<Value, String>,
) -> Result<bool, String> {
    // A process that cannot hold the spent list cannot open what it claims, so it claims nothing: the row stays for
    // the process that can. (An error, for the loop's one line and its back-off.)
    identity.may_open_requests()?;
    // The same for all three requests of a run, as the one timeout of the
    // client this lane used to build for each poll was: over the server's wait,
    // so a poll that comes back on time is not cut off by this side.
    let timeout = Duration::from_secs(spec.wait_seconds + 15);
    let pending = http
        .get(super::oauth::join(&account.base_url, &spec.pending_path))
        .timeout(timeout)
        .bearer_auth(&account.credential)
        .query(&[
            ("instanceId", instance.to_owned()),
            ("wait", (spec.wait_seconds * 1000).to_string()),
            ("limit", "1".to_string()),
        ])
        .send()
        .map_err(|_| "sealed flow poll could not reach the provider")?;
    if !pending.status().is_success() {
        let status = pending.status();
        // Asked to slow down: wait as long as the provider said before the lane's own back-off.
        let asked = if status.as_u16() == 429 { super::net::retry_after(pending.headers()) } else { None };
        super::net::drain(pending);
        if let Some(wait) = asked {
            std::thread::sleep(wait);
        }
        return Err(format!("sealed flow poll: HTTP {status}"));
    }
    let pending: Pending = read_json(pending)?;
    let Some(request) = pending.requests.into_iter().next() else {
        return Ok(false);
    };
    if !still_linked() {
        return Ok(false);
    }
    if uuid::Uuid::parse_str(&request.request_id).is_err()
        || request
            .target_instance_id
            .as_deref()
            .is_some_and(|target| target != instance)
    {
        return Err("sealed flow has an invalid request id or target".into());
    }
    let id = request.request_id;
    let claimed = http
        .post(super::oauth::join(
            &account.base_url,
            &spec.claim_path.replace("{id}", &id),
        ))
        .timeout(timeout)
        .bearer_auth(&account.credential)
        .json(&json!({"instanceId": instance}))
        .send()
        .map_err(|_| "sealed flow claim could not be confirmed; execution skipped")?;
    if claimed.status().as_u16() == 409 {
        super::net::drain(claimed);
        return Ok(false);
    }
    if !claimed.status().is_success() {
        let status = claimed.status();
        super::net::drain(claimed);
        return Err(format!("sealed flow claim: HTTP {status}"));
    }
    let claim: Claim = read_json(claimed)?;
    if !claim.claimed
        || claim.request.request_id != id
        || claim
            .request
            .target_instance_id
            .as_deref()
            .is_some_and(|target| target != instance)
    {
        return Err("sealed flow claim did not confirm this request and target".into());
    }
    let sessions = E2eSessions::new();
    let outcome = open_inputs(identity, &sessions, &claim.request).and_then(|inputs| {
        if !still_linked() {
            return Err("The account was disconnected before this flow started.".into());
        }
        execute(&claim.request.flow_id, inputs)
    });
    let (success, payload) = terminal(outcome);
    let sealed = sessions
        .seal_outbound(&id, payload.to_string().as_bytes())
        .ok();
    sessions.drop_thread(&id);
    // Authentication failure may leave no session. Report failed without details
    // rather than returning any plaintext input, output, or runner error.
    let body = json!({"instanceId": instance, "status": if success && sealed.is_some() {"done"} else {"failed"},
        "resultEnvelope": sealed});
    let url = super::oauth::join(&account.base_url, &spec.complete_path.replace("{id}", &id));
    // Sent again as the provider is away, longer and longer apart (a minute in all), then kept on the disk and sent at
    // each later poll until it is taken: a flow that ran, its effects made, is not reported as never done because the
    // provider was away for half a minute.
    for attempt in 0..6u32 {
        let mut asked = Duration::ZERO;
        if let Ok(response) = http
            .post(&url)
            .timeout(timeout)
            .bearer_auth(&account.credential)
            .json(&body)
            .send()
        {
            let status = response.status();
            asked = super::net::retry_after(response.headers()).unwrap_or_default();
            super::net::drain(response);
            if status.is_success() {
                return Ok(true);
            }
            if refused_for_good(status) {
                return Err(format!(
                    "sealed flow completion: HTTP {status}; flow will not be repeated"
                ));
            }
        }
        // Not taken, and not refused either: the same sealed result is sent again, and no
        // sooner than the provider asked.
        std::thread::sleep(if attempt < 5 { asked.max(Duration::from_secs(1 << attempt)) } else { asked });
    }
    keep_completion(&id, &account.base_url, &url, &body);
    Err("sealed flow completion could not be confirmed yet: it is kept and sent again; the flow will not be repeated".into())
}

/// A completion the provider refuses for good, which nothing is gained by sending again. A 408, a 425 or a 429 says not
/// now, as a 5xx does: taken for a refusal, the result of a run that had done its work was gone on that one answer, at
/// the moment every lane comes back at once.
fn refused_for_good(status: reqwest::StatusCode) -> bool {
    status.is_client_error() && !matches!(status.as_u16(), 408 | 425 | 429)
}

/// Where completions not taken yet are kept (`<data>/link/sealed-pending`), set when the lane starts.
static KEPT_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// How long a kept completion holds the next claim back. A day of one result not landing is no longer the provider being
/// away for a while: it is that file, or that one route, and this desktop is not kept from its work for it.
const KEPT_HOLDS_CLAIMS_FOR: Duration = Duration::from_secs(24 * 3600);
/// How long a completion is kept before it is let go.
const KEPT_FOR: Duration = Duration::from_secs(7 * 24 * 3600);
/// How far apart a result past [`KEPT_HOLDS_CLAIMS_FOR`] is sent again, beside the lane's claims: it is not worth a request
/// at every poll, on a route that has been failing for a day.
const LATE_EVERY: Duration = Duration::from_secs(60);
/// What the log says while a result is past [`KEPT_HOLDS_CLAIMS_FOR`]: the same words each time, so it is one line. Not the
/// lane's status, which the panel shows as "Encrypted flows unavailable": the lane is working.
const LATE_NOTE: &str = "a finished flow's result has not been taken for over a day: it is sent again now and then, and new flows are claimed beside it";

/// Keep a sealed completion the provider has not taken, to be sent again ([`resend_kept`]). The body is sealed: no
/// input, output or error is in it in the clear.
fn keep_completion(id: &str, base_url: &str, url: &str, body: &Value) {
    let Some(dir) = KEPT_DIR.get() else { return };
    let safe: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')).take(80).collect();
    if safe.is_empty() || std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let kept = json!({ "baseUrl": base_url, "url": url, "body": body });
    let _ = crate::secret_file::write(&dir.join(format!("{safe}.json")), kept.to_string());
}

/// What the kept completions say about the next claim.
#[derive(Debug, Default)]
struct Kept {
    /// Why it has to wait: a completion not taken yet, and not a day old.
    held: Option<String>,
    /// A completion is past [`KEPT_HOLDS_CLAIMS_FOR`] and still not taken.
    late: bool,
    /// How long the provider asked to be left, where it said.
    asked: Option<Duration>,
}

/// How long a kept file has waited. A date in the future (a clock that was wrong, a copy from elsewhere) counts from now,
/// and the file is dated so: read as young each time, it would hold the lane for good. One that cannot be dated does not
/// hold it either.
fn kept_age(path: &std::path::Path, modified: std::time::SystemTime) -> Duration {
    modified.elapsed().unwrap_or_else(|_| {
        let now = std::time::SystemTime::now();
        match std::fs::File::options().write(true).open(path).and_then(|file| file.set_modified(now)) {
            Ok(()) => Duration::ZERO,
            Err(_) => KEPT_HOLDS_CLAIMS_FOR,
        }
    })
}

/// Whether `url` is an address at the provider: the account's bearer goes on what is sent to it.
fn at_provider(url: &str, base_url: &str) -> bool {
    reqwest::Url::parse(url).is_ok() && url.starts_with(&format!("{}/", base_url.trim_end_matches('/')))
}

/// Put aside a kept file that can never be sent: it would otherwise wait out its week.
fn set_aside(path: &std::path::Path) {
    if let Err(why) = super::put_aside(path, "a kept sealed flow result") {
        log::warn!("{why}");
    }
}

/// Send the kept completions again: each goes once it is taken or refused for good, and while the provider is away the
/// rest wait. One that is more than a day old waits beside the lane rather than ahead of it, and is sent only when
/// `try_late` says so. A file that cannot be sent goes aside, and a week unsent, one is let go.
fn resend_kept_in(dir: &std::path::Path, http: &Client, account: &LinkedAccount, timeout: Duration, try_late: bool) -> Kept {
    let mut kept = Kept::default();
    let Ok(entries) = std::fs::read_dir(dir) else { return kept };
    for entry in entries.flatten() {
        let path = entry.path();
        let age = entry.metadata().ok().and_then(|m| m.modified().ok()).map_or(Duration::ZERO, |modified| kept_age(&path, modified));
        if age > KEPT_FOR {
            log::warn!("a finished flow's result was not taken in a week: it is let go");
            let _ = std::fs::remove_file(&path);
            continue;
        }
        // (what was put aside stays aside until then)
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        // (a file another program holds for now is read the next time)
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(file) = serde_json::from_slice::<Value>(&bytes) else {
            set_aside(&path);
            continue;
        };
        let Some(base_url) = file["baseUrl"].as_str() else {
            set_aside(&path);
            continue;
        };
        if base_url != account.base_url {
            continue;
        }
        let Some(url) = file["url"].as_str().filter(|url| at_provider(url, base_url)) else {
            set_aside(&path);
            continue;
        };
        let late = age >= KEPT_HOLDS_CLAIMS_FOR;
        if late && !try_late {
            kept.late = true;
            continue;
        }
        let why = match http.post(url).timeout(timeout).bearer_auth(&account.credential).json(&file["body"]).send() {
            Ok(response) => {
                let status = response.status();
                let asked = super::net::retry_after(response.headers());
                super::net::drain(response);
                if status.is_success() || refused_for_good(status) {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                kept.asked = kept.asked.max(asked);
                format!("HTTP {status}")
            }
            Err(_) => "the provider could not be reached".to_string(),
        };
        if late {
            kept.late = true;
            continue;
        }
        kept.held = Some(format!("a finished flow's result has not been taken yet ({why}): it is sent again, and new flows wait until it lands"));
        return kept;
    }
    kept
}

pub fn spawn(store: LinkHandle, node: Option<NodeHandle>) {
    spawn_on(store, node, &HTTP);
}

/// [`spawn`] on the client `lane`. The lane's client is one for the process, and a
/// test that starts the loop gives it a client of its own, so that the connections
/// of the other tests that use the lane's are not the loop's to close.
fn spawn_on(store: LinkHandle, node: Option<NodeHandle>, lane: &'static Lane) {
    let kept_dir = store.data_dir().join("link").join("sealed-pending");
    let _ = KEPT_DIR.set(kept_dir.clone());
    std::thread::spawn(move || {
        let identity = match E2eIdentity::load_or_create(store.data_dir()) {
            Ok(identity) => identity,
            Err(error) => {
                store.note_sealed_flow(Some(error.clone()));
                log::error!("sealed flow identity unavailable: {error}");
                return;
            }
        };
        // Said once per reason, not every ten seconds while the provider is away.
        let mut last_warned: Option<String> = None;
        // The same for what the kept completions say, which the polls beside them do not change.
        let mut last_kept: Option<String> = None;
        let mut next_late = std::time::Instant::now();
        loop {
            let Some(account) = store.account() else {
                std::thread::sleep(Duration::from_secs(5));
                continue;
            };
            let Some(descriptor) = descriptor::find(store.data_dir(), &account.connector_id) else {
                std::thread::sleep(Duration::from_secs(30));
                continue;
            };
            let (Some(spec), Some(flows)) = (descriptor.desktop_flows, descriptor.flows) else {
                std::thread::sleep(Duration::from_secs(30));
                continue;
            };
            let http = match lane.get() {
                Ok(http) => http,
                Err(_) => {
                    std::thread::sleep(Duration::from_secs(10));
                    continue;
                }
            };
            // (completions kept from before, the provider away when they were made: told first)
            // A finished run is held at the provider until its completion lands. A claim now would
            // say this desktop had let go of it: the run would expire, and its result be refused.
            let try_late = std::time::Instant::now() >= next_late;
            let kept = resend_kept_in(&kept_dir, &http, &account, Duration::from_secs(15), try_late);
            let asked = kept.asked.unwrap_or_default();
            if try_late && kept.late {
                next_late = std::time::Instant::now() + LATE_EVERY.max(asked);
            }
            let waiting = kept.held.clone().or_else(|| kept.late.then(|| LATE_NOTE.to_string()));
            if waiting != last_kept {
                if let Some(why) = &waiting {
                    log::warn!("{why}");
                }
                last_kept = waiting;
            }
            if kept.held.is_some() {
                if store.account().as_ref() == Some(&account) {
                    store.note_sealed_flow(kept.held);
                }
                drop(http);
                lane.start_afresh();
                std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds).max(asked));
                continue;
            }
            let polled = std::time::Instant::now();
            let result = poll_once(
                &http,
                &account,
                &spec,
                &store.instance_id(),
                &identity,
                &|| {
                    store.account().is_some_and(|current| {
                        current.base_url == account.base_url
                            && current.credential == account.credential
                            && current.connector_id == account.connector_id
                    })
                },
                &|flow_id, inputs| {
                    super::flow_runner::execute_sealed(
                        &account,
                        &flows,
                        node.as_ref(),
                        flow_id,
                        inputs,
                        store.data_dir(),
                    )
                },
            );
            // This clone of the client shares its pool: kept through the pause below it
            // would hold the connection the lane lets go of there (and the one it
            // starts afresh after a failure) open until the next poll.
            drop(http);
            if store.account().as_ref() == Some(&account) {
                store.note_sealed_flow(result.as_ref().err().cloned());
            }
            match result {
                Ok(worked) => {
                    last_warned = None;
                    if !worked {
                        let pause = spec.idle_pause(polled.elapsed());
                        lane.rest(pause);
                        std::thread::sleep(pause);
                    }
                }
                Err(error) => {
                    if last_warned.as_deref() != Some(error.as_str()) {
                        log::warn!("{error}");
                        last_warned = Some(error);
                    }
                    // The try after a failure is on a new client, not one that
                    // has seen the trouble.
                    lane.start_afresh();
                    std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds));
                }
            }
        }
    });
}

/// A lane's client, and the connections it keeps.
type Lane = super::net::LaneClient<Client>;

/// The client this lane keeps from poll to poll, so that a poll, its claim and the
/// completion share a connection instead of each making one.
static HTTP: Lane = super::net::LaneClient::new(build_client);

fn build_client() -> Result<Client, String> {
    super::net::blocking_builder(super::net::Keep::Between)
        // A reply that sends this lane somewhere else is an error to report, not
        // a request to make again, elsewhere, with the credential on it.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("could not build the sealed flow client: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD as B64, Engine};
    use crypto_box::{aead::Aead, PublicKey, SalsaBox, SecretKey};
    use std::{cell::Cell, io::Write, net::TcpListener};

    const ID: &str = "d0f04426-d01b-4471-aeef-f65ecae1eb79";

    fn request(identity: &E2eIdentity, flow_id: &str, body: Value) -> Value {
        let secret = SecretKey::from([42; 32]);
        let key = SalsaBox::new(&PublicKey::from(identity.public_key_bytes()), &secret);
        let nonce = crate::ai::e2e::frame_nonce(0, 0);
        let ciphertext = key
            .encrypt((&nonce).into(), body.to_string().as_bytes())
            .unwrap();
        json!({"requestId": ID, "flowId": flow_id, "targetInstanceId": "test-computer",
            "ephPub": B64.encode(secret.public_key().to_bytes()),
            "envelope": B64.encode([nonce.as_slice(), ciphertext.as_slice()].concat())})
    }

    fn open_result(identity: &E2eIdentity, envelope: &str) -> Value {
        let bytes = B64.decode(envelope).unwrap();
        let opened = crate::ai::e2e::open_detached(
            &[42; 32],
            &identity.public_key_bytes(),
            1,
            &B64.encode(&bytes[..24]),
            &B64.encode(&bytes[24..]),
        )
        .unwrap();
        serde_json::from_slice(&opened).unwrap()
    }

    // Finite server: every expected request must arrive; no detached infinite
    // accept loop or background port remains after a test.
    fn server(replies: Vec<(u16, Value)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, body) in replies {
                let deadline = std::time::Instant::now() + Duration::from_secs(8);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < deadline,
                                "expected relay request did not arrive"
                            );
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("{error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buffer = [0; 4096];
                    let count = stream.read(&mut buffer).unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&buffer[..count]);
                    let text = String::from_utf8_lossy(&bytes);
                    if let Some(end) = text.find("\r\n\r\n") {
                        let length: usize = text[..end]
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length: ")
                                    .and_then(|n| n.parse().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8(bytes).unwrap());
                let body = body.to_string();
                write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            requests
        });
        (url, thread)
    }

    fn account(url: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: url,
            credential: "test-scoped-key".into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("test-computer".into()),
        }
    }

    fn spec() -> RelaySpec {
        descriptor::builtin().remove(0).desktop_flows.unwrap()
    }
    fn client() -> Client {
        Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    #[test]
    fn a_completion_still_kept_says_so_until_it_lands() {
        // What holds the next claim back: a finished run is the provider's to expire once this
        // desktop claims another, and its result would then be refused.
        let dir = crate::test_dirs::temp_path("oaiy-sealed-kept");
        std::fs::create_dir_all(&dir).unwrap();
        let keep = |url: &str| {
            let kept = json!({ "baseUrl": url, "url": format!("{url}/done"), "body": {"status": "done"} });
            std::fs::write(dir.join("run.json"), kept.to_string()).unwrap();
        };

        let (url, away) = server(vec![(503, json!({}))]);
        keep(&url);
        assert!(resend_kept_in(&dir, &client(), &account(url), Duration::from_secs(5), true).held.is_some_and(|why| why.contains("503")));
        assert!(dir.join("run.json").exists(), "kept while the provider is away");
        away.join().unwrap();

        let (url, back) = server(vec![(200, json!({}))]);
        keep(&url);
        assert!(resend_kept_in(&dir, &client(), &account(url), Duration::from_secs(5), true).held.is_none());
        assert!(!dir.join("run.json").exists(), "gone once taken");
        back.join().unwrap();
    }

    // --- what a kept completion does to the lane -------------------------------------

    fn kept_body(base: &str) -> String {
        json!({ "baseUrl": base, "url": format!("{base}/done"), "body": {"status": "done"} }).to_string()
    }

    /// A folder of its own holding `files`.
    fn kept_folder(files: &[(&str, &[u8])]) -> std::path::PathBuf {
        let dir = crate::test_dirs::temp_path("oaiy-sealed-kept");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        dir
    }

    fn dated(path: &std::path::Path, modified: std::time::SystemTime) {
        std::fs::File::options().write(true).open(path).unwrap().set_modified(modified).unwrap();
    }

    fn days_ago(days: u64) -> std::time::SystemTime {
        std::time::SystemTime::now() - Duration::from_secs(days * 24 * 3600)
    }

    #[test]
    fn a_completion_the_provider_cannot_take_for_now_is_kept_and_the_lane_waits_as_long_as_it_asked() {
        // On a 429 as every lane came back, the file went: the run's work done, and nobody ever told the result.
        for status in [408, 425, 429] {
            let provider = Provider::start(move |_| Reply { status, headers: vec![("Retry-After".into(), "7".into())], body: "{}".into() });
            let dir = kept_folder(&[("run.json", kept_body(&provider.base).as_bytes())]);
            let kept = resend_kept_in(&dir, &client(), &account(provider.base.clone()), Duration::from_secs(5), true);
            assert!(kept.held.is_some_and(|why| why.contains(&status.to_string())), "{status}");
            assert_eq!(kept.asked, Some(Duration::from_secs(7)), "{status}");
            assert!(dir.join("run.json").exists(), "{status}: the result is kept");
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_completion_the_provider_refuses_for_good_goes() {
        for status in [400, 401, 403, 404, 409, 410, 422] {
            let provider = Provider::start(move |_| Reply::status(status, "{}"));
            let dir = kept_folder(&[("run.json", kept_body(&provider.base).as_bytes())]);
            let kept = resend_kept_in(&dir, &client(), &account(provider.base.clone()), Duration::from_secs(5), true);
            assert!(kept.held.is_none() && !kept.late, "{status}");
            assert!(!dir.join("run.json").exists(), "{status}");
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn the_first_send_of_a_completion_is_not_refused_by_a_provider_that_says_not_now() {
        for status in [408, 425, 429] {
            let identity = E2eIdentity::from_secret_bytes([17; 32]);
            let sealed = request(&identity, "flow-one", json!({"v":1,"flowId":"flow-one","inputs":{}}));
            let asked = AtomicBool::new(false);
            let provider = Provider::start(move |req| {
                if req.target.starts_with("/api/v1/desktop-flows/pending") {
                    Reply::ok(&json!({"requests":[sealed.clone()]}).to_string())
                } else if req.target.ends_with("/claim") {
                    Reply::ok(&json!({"claimed":true,"request":sealed.clone()}).to_string())
                } else if asked.swap(true, Ordering::SeqCst) {
                    Reply::ok("{}")
                } else {
                    Reply { status, headers: vec![("Retry-After".into(), "2".into())], body: "{}".into() }
                }
            });
            let ran = Cell::new(0);
            let worked = poll_once(&client(), &account(provider.base.clone()), &spec(), "test-computer", &identity, &|| true, &|_, _| {
                ran.set(ran.get() + 1);
                Ok(json!("done"))
            });
            assert_eq!(worked, Ok(true), "{status}");
            assert_eq!(ran.get(), 1, "{status}: the flow is not repeated");
            let sent = provider.wait_for(&format!("/api/v1/desktop-flows/{ID}/complete"), 2, Duration::from_secs(5));
            let gap = sent[1].at - sent[0].at;
            assert!(gap >= Duration::from_millis(1_900), "{status}: sent again after {gap:?}, not the 2 s it asked for");
        }
    }

    #[test]
    fn a_kept_file_that_can_never_be_sent_goes_aside_with_the_lane_going_on_and_no_key_sent_anywhere() {
        let provider = Provider::start(|_| Reply::ok("{}"));
        let base = provider.base.clone();
        let with_url = |url: &str| json!({ "baseUrl": base, "url": url, "body": {"status": "done"} }).to_string();
        let bad = [
            ("not-json.json", b"not json at all".to_vec()),
            ("not-text.json", vec![0xff, 0xfe, 0x00, 0x7b]),
            ("a-list.json", b"[1, 2]".to_vec()),
            ("no-base.json", json!({ "url": format!("{base}/done"), "body": {} }).to_string().into_bytes()),
            ("no-url.json", json!({ "baseUrl": base, "body": {} }).to_string().into_bytes()),
            ("not-a-url.json", with_url("not a url at all").into_bytes()),
            ("elsewhere.json", with_url("http://elsewhere.invalid/done").into_bytes()),
            ("longer-host.json", with_url(&format!("{base}.elsewhere.invalid/done")).into_bytes()),
            ("the-base-itself.json", with_url(&base).into_bytes()),
        ];
        let good = kept_body(&base);
        let mut files: Vec<(&str, &[u8])> = bad.iter().map(|(name, bytes)| (*name, bytes.as_slice())).collect();
        files.push(("good.json", good.as_bytes()));
        let dir = kept_folder(&files);

        let kept = resend_kept_in(&dir, &client(), &account(base.clone()), Duration::from_secs(5), true);
        assert!(kept.held.is_none() && !kept.late, "{kept:?}");
        for (name, _) in &bad {
            assert!(!dir.join(name).exists(), "{name} is out of the way");
            assert!(dir.join(format!("{name}.corrupt")).exists(), "{name} is kept aside");
        }
        assert_eq!(provider.lines(), ["POST /done"], "the good file went, and nothing else was tried");

        // And what is aside is not read again.
        assert!(resend_kept_in(&dir, &client(), &account(base), Duration::from_secs(5), true).held.is_none());
        assert_eq!(provider.lines().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_kept_file_of_another_provider_is_left_where_it_is() {
        let provider = Provider::start(|_| Reply::status(500, "{}"));
        let dir = kept_folder(&[("theirs.json", kept_body("http://other.invalid").as_bytes())]);
        let kept = resend_kept_in(&dir, &client(), &account(provider.base.clone()), Duration::from_secs(5), true);
        assert!(kept.held.is_none() && !kept.late);
        assert!(dir.join("theirs.json").exists());
        assert!(provider.requests().is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_kept_file_dated_in_the_future_counts_from_now() {
        let provider = Provider::start(|_| Reply::status(500, "{}"));
        let dir = kept_folder(&[("run.json", kept_body(&provider.base).as_bytes())]);
        let file = dir.join("run.json");
        dated(&file, std::time::SystemTime::now() + Duration::from_secs(365 * 24 * 3600));

        let kept = resend_kept_in(&dir, &client(), &account(provider.base.clone()), Duration::from_secs(5), true);
        assert!(kept.held.is_some(), "as new as the day it came: it holds the lane, for now");
        let age = std::fs::metadata(&file).unwrap().modified().unwrap().elapsed();
        assert!(age.as_ref().is_ok_and(|age| *age < Duration::from_secs(600)), "and is dated now, so that it ages: {age:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn after_a_day_a_kept_file_stops_holding_the_lane_and_is_still_sent() {
        let answer = std::sync::Arc::new(std::sync::atomic::AtomicU16::new(500));
        let said = answer.clone();
        let provider = Provider::start(move |_| Reply::status(said.load(Ordering::SeqCst), "{}"));
        let dir = kept_folder(&[("run.json", kept_body(&provider.base).as_bytes())]);
        let file = dir.join("run.json");
        let linked = account(provider.base.clone());

        dated(&file, days_ago(2));
        let kept = resend_kept_in(&dir, &client(), &linked, Duration::from_secs(5), true);
        assert!(kept.held.is_none(), "the lane goes back to claiming");
        assert!(kept.late, "and it is said");
        assert!(file.exists(), "a late result is better than none");
        assert_eq!(provider.requests().len(), 1);

        // Between its turns it is not sent, and it is still said.
        assert!(resend_kept_in(&dir, &client(), &linked, Duration::from_secs(5), false).late);
        assert_eq!(provider.requests().len(), 1);

        // A file not a day old beside it still holds the lane.
        std::fs::write(dir.join("newer.json"), kept_body(&provider.base)).unwrap();
        assert!(resend_kept_in(&dir, &client(), &linked, Duration::from_secs(5), false).held.is_some());
        std::fs::remove_file(dir.join("newer.json")).unwrap();

        // And it lands the day the provider takes it.
        answer.store(200, Ordering::SeqCst);
        let kept = resend_kept_in(&dir, &client(), &linked, Duration::from_secs(5), true);
        assert!(kept.held.is_none() && !kept.late);
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_week_on_a_kept_file_is_let_go_unsent() {
        let provider = Provider::start(|_| Reply::status(500, "{}"));
        let dir = kept_folder(&[("run.json", kept_body(&provider.base).as_bytes()), ("aside.json.corrupt", b"x")]);
        dated(&dir.join("run.json"), days_ago(8));
        dated(&dir.join("aside.json.corrupt"), days_ago(8));
        let kept = resend_kept_in(&dir, &client(), &account(provider.base.clone()), Duration::from_secs(5), true);
        assert!(kept.held.is_none() && !kept.late);
        assert!(provider.requests().is_empty());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The lane's own loop on a folder with a kept file in it, until `done` has seen what it wants.
    fn run_lane_on_a_kept_file(tag: &str, days: u64, provider: &Provider, done: impl FnOnce(&LinkHandle)) {
        let (store, dir) = crate::link::testkit::linked_to(&provider.base, tag, |d| {
            let lane = d.desktop_flows.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.error_backoff_seconds = 1;
        });
        let kept = dir.join("link").join("sealed-pending");
        std::fs::create_dir_all(&kept).unwrap();
        std::fs::write(kept.join("run.json"), kept_body(&provider.base)).unwrap();
        dated(&kept.join("run.json"), days_ago(days));
        spawn_on(store.clone(), None, a_client_of_its_own());
        done(&store);
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_lane_held_back_by_a_kept_file_says_so() {
        let provider = Provider::start(|req| if req.target == "/done" { Reply::status(503, "{}") } else { Reply::ok(&json!({"requests":[]}).to_string()) });
        run_lane_on_a_kept_file("sealed-held", 0, &provider, |store| {
            provider.wait_for("/done", 2, Duration::from_secs(30));
            let why = store.status().sealed_flow_error;
            assert!(why.as_deref().is_some_and(|why| why.contains("503") && why.contains("new flows wait")), "{why:?}");
            assert!(provider.requests().iter().all(|r| r.target == "/done"), "no claim while it holds: {:?}", provider.lines());
        });
    }

    #[test]
    fn a_lane_goes_on_beside_a_kept_file_past_its_day() {
        let provider = Provider::start(|req| if req.target == "/done" { Reply::status(500, "{}") } else { Reply::ok(&json!({"requests":[]}).to_string()) });
        run_lane_on_a_kept_file("sealed-late", 2, &provider, |store| {
            provider.wait_for("/api/v1/desktop-flows/pending", 3, Duration::from_secs(30));
            assert_eq!(provider.wait_for("/done", 1, Duration::from_secs(1)).len(), 1, "sent again, but not at every poll: {:?}", provider.lines());
            assert_eq!(store.status().sealed_flow_error, None, "the lane is working, and the panel is not told otherwise");
        });
    }

    #[test]
    fn remote_claim_decrypt_execute_encrypt_and_retry_completion_once_without_repeating_work() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let request = request(
            &identity,
            "flow-one",
            json!({"v":1,"flowId":"flow-one","inputs":{"private":"hello"}}),
        );
        let (url, server) = server(vec![
            (200, json!({"requests":[request.clone()]})),
            (200, json!({"claimed":true,"request":request})),
            (503, json!({})),
            (200, json!({})),
        ]);
        let executions = Cell::new(0);
        assert!(poll_once(
            &client(),
            &account(url),
            &spec(),
            "test-computer",
            &identity,
            &|| true,
            &|id, inputs| {
                assert_eq!(id, "flow-one");
                assert_eq!(inputs["private"], "hello");
                executions.set(executions.get() + 1);
                Ok(json!({"answer":"private-result"}))
            }
        )
        .unwrap());
        assert_eq!(executions.get(), 1);
        let requests = server.join().unwrap();
        assert!(requests[0].contains("instanceId=test-computer"));
        assert!(requests[1].starts_with(&format!("POST /api/v1/desktop-flows/{ID}/claim ")));
        let completion = requests[2].split("\r\n\r\n").nth(1).unwrap();
        assert_eq!(completion, requests[3].split("\r\n\r\n").nth(1).unwrap());
        assert!(!completion.contains("private-result"));
        let completion: Value = serde_json::from_str(completion).unwrap();
        assert_eq!(completion["status"], "done");
        assert_eq!(
            open_result(&identity, completion["resultEnvelope"].as_str().unwrap())["result"]
                ["answer"],
            "private-result"
        );
    }

    /// Another process on the folder holds the spent list. This one used to claim the row and then fail it, a lost turn
    /// for the person, where the holder would have answered.
    #[test]
    fn a_lane_that_cannot_hold_the_spent_list_asks_the_provider_for_nothing() {
        let dir = crate::secret_file::testing::TempDir::new("sealed-flows-held");
        let identity = E2eIdentity::load_or_create(&dir.0).unwrap();
        let holder = std::fs::File::create(dir.0.join("desktop-e2e-spent.lock")).unwrap();
        fs2::FileExt::try_lock_exclusive(&holder).unwrap();
        let (url, silent) = server(vec![]);
        let refused = poll_once(&client(), &account(url), &spec(), "test-computer", &identity, &|| true, &|_, _| panic!("nothing was claimed"));
        assert!(refused.unwrap_err().contains("another OAIY process"));
        assert!(silent.join().unwrap().is_empty());

        // The holder gone, the lane polls again by itself.
        drop(holder);
        let (url, asked) = server(vec![(200, json!({"requests":[]}))]);
        assert!(!poll_once(&client(), &account(url), &spec(), "test-computer", &identity, &|| true, &|_, _| panic!("nothing is pending")).unwrap());
        assert_eq!(asked.join().unwrap().len(), 1);
        crate::ai::e2e::let_go_of_spent_list(&dir.0);
    }

    #[test]
    fn lost_claim_never_executes_or_completes() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let request = request(
            &identity,
            "flow-one",
            json!({"v":1,"flowId":"flow-one","inputs":{}}),
        );
        let (url, server) = server(vec![(200, json!({"requests":[request]})), (409, json!({}))]);
        assert!(!poll_once(
            &client(),
            &account(url),
            &spec(),
            "test-computer",
            &identity,
            &|| true,
            &|_, _| panic!("a lost claim must not execute")
        )
        .unwrap());
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn sealed_flow_identity_mismatch_reports_only_an_encrypted_failure() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let request = request(
            &identity,
            "flow-two",
            json!({"v":1,"flowId":"flow-one","inputs":{}}),
        );
        let (url, server) = server(vec![
            (200, json!({"requests":[request.clone()]})),
            (200, json!({"claimed":true,"request":request})),
            (200, json!({})),
        ]);
        poll_once(
            &client(),
            &account(url),
            &spec(),
            "test-computer",
            &identity,
            &|| true,
            &|_, _| panic!("mismatched flow must not execute"),
        )
        .unwrap();
        let requests = server.join().unwrap();
        let completion: Value =
            serde_json::from_str(requests[2].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(completion["status"], "failed");
        assert!(completion.get("message").is_none());
        assert_eq!(
            open_result(&identity, completion["resultEnvelope"].as_str().unwrap())["type"],
            "error"
        );
    }

    #[test]
    fn unlink_while_polling_leaves_work_unclaimed() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let request = request(
            &identity,
            "flow-one",
            json!({"v":1,"flowId":"flow-one","inputs":{}}),
        );
        let (url, server) = server(vec![(200, json!({"requests":[request]}))]);
        assert!(!poll_once(
            &client(),
            &account(url),
            &spec(),
            "test-computer",
            &identity,
            &|| false,
            &|_, _| panic!("disconnected account must not execute")
        )
        .unwrap());
        assert_eq!(server.join().unwrap().len(), 1);
    }

    #[test]
    fn oversized_output_is_a_failure_not_a_success_placeholder() {
        let (success, body) = terminal(Ok(Value::String("x".repeat(MAX_RESULT_BYTES))));
        assert!(!success);
        assert_eq!(body["type"], "error");
    }

    // --- the connections it makes, with the client it keeps -------------------------

    use crate::link::testkit::{Provider, Reply};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A client for a loop that a test starts, which no other test uses: the process's
    /// is used by the tests that count connections on it, and a loop closes them (it
    /// starts its client afresh after a failure, and lets go of it for a long pause).
    fn a_client_of_its_own() -> &'static Lane {
        Box::leak(Box::new(crate::link::net::LaneClient::new(build_client)))
    }

    #[test]
    fn a_poll_its_claim_the_completion_and_the_next_poll_share_one_connection() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let sealed = request(
            &identity,
            "flow-one",
            json!({"v":1,"flowId":"flow-one","inputs":{"n":1}}),
        );
        let served = AtomicBool::new(false);
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-flows/pending") {
                if served.swap(true, Ordering::SeqCst) {
                    Reply::ok(&json!({"requests":[]}).to_string())
                } else {
                    Reply::ok(&json!({"requests":[sealed.clone()]}).to_string())
                }
            } else if req.target.ends_with("/claim") {
                Reply::ok(&json!({"claimed":true,"request":sealed.clone()}).to_string())
            } else {
                Reply::ok("{}")
            }
        });
        let run = |worked_expected: bool| {
            // The client the lane keeps, taken for each poll as its loop takes it.
            let worked = poll_once(
                &HTTP.get().unwrap(),
                &account(server.base.clone()),
                &spec(),
                "test-computer",
                &identity,
                &|| true,
                &|_, _| Ok(json!("done")),
            )
            .unwrap();
            assert_eq!(worked, worked_expected);
        };
        run(true);
        run(false);

        let pending = "GET /api/v1/desktop-flows/pending?instanceId=test-computer&wait=25000&limit=1";
        assert_eq!(
            server.lines(),
            [
                pending.to_string(),
                format!("POST /api/v1/desktop-flows/{ID}/claim"),
                format!("POST /api/v1/desktop-flows/{ID}/complete"),
                pending.to_string(),
            ]
        );
        assert_eq!(server.connections(), 1, "{:?}", server.lines());
    }

    #[test]
    fn the_lane_waits_after_an_empty_poll_as_long_as_its_descriptor_says() {
        // The lane's own loop, not a function beside it: a provider that asks for a
        // three second pause is not polled again for three seconds. (Every lane
        // used to wait half a second, or the rest of two seconds if the provider
        // cut its poll short, and there was nowhere to say otherwise.)
        let server = Provider::start(|_| Reply::ok(&json!({"requests":[]}).to_string()));
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "sealed-slow", |d| {
            let lane = d.desktop_flows.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.idle_pause_ms = 3_000;
        });
        spawn_on(store.clone(), None, a_client_of_its_own());
        let polls = server.wait_for("/api/v1/desktop-flows/pending", 3, Duration::from_secs(30));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        for pair in polls.windows(2) {
            let gap = pair[1].at - pair[0].at;
            assert!(gap >= Duration::from_millis(2_800) && gap < Duration::from_secs(6), "{gap:?}");
        }
    }

    #[test]
    fn a_long_pause_after_an_empty_poll_is_not_spent_holding_a_connection_open() {
        // The pause of a provider that cannot hold polls (a hold of a second, a
        // pause of five) is time it means to be free of the lane, and a connection
        // held through it, for four seconds or more, would not be.
        let server = Provider::start(|_| Reply::ok(&json!({"requests":[]}).to_string()));
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "sealed-rest", |d| {
            let lane = d.desktop_flows.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.idle_pause_ms = 5_000;
        });
        spawn_on(store.clone(), None, a_client_of_its_own());
        server.wait_for("/api/v1/desktop-flows/pending", 1, Duration::from_secs(30));
        let held = server.try_closed_after_reply(0, Duration::from_secs(3));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        assert!(held.is_some_and(|held| held < Duration::from_secs(2)), "the connection was held open: {held:?}");
    }

    #[test]
    fn a_poll_that_failed_is_followed_by_one_on_a_new_connection() {
        // A lane starts its client afresh after a failed poll, as it was when every poll
        // made its own client. The second poll comes a second after the first, well
        // inside what a connection is kept for, so on a client that had been kept it
        // would arrive on the same connection.
        let failed = AtomicBool::new(false);
        let server = Provider::start(move |_| {
            if failed.swap(true, Ordering::SeqCst) {
                Reply::ok(&json!({"requests":[]}).to_string())
            } else {
                Reply::status(500, "{}")
            }
        });
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "sealed-afresh", |d| {
            d.desktop_flows.as_mut().unwrap().error_backoff_seconds = 1;
        });
        spawn_on(store.clone(), None, a_client_of_its_own());
        let polls = server.wait_for("/api/v1/desktop-flows/pending", 2, Duration::from_secs(30));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        assert_eq!(polls.iter().map(|p| p.conn).collect::<Vec<_>>(), [0, 1], "{:?}", server.lines());
    }

    #[test]
    fn the_sealed_lane_never_follows_a_redirect_and_sends_its_credential_nowhere_else() {
        // Set on the client this lane builds, and kept there: a provider that answers
        // with a redirect is reported, not followed with the bearer on the request.
        for status in [301, 302, 307, 308] {
            let elsewhere = Provider::start(|_| Reply::ok(&json!({"requests":[]}).to_string()));
            let moved_to = format!("{}/api/v1/desktop-flows/pending", elsewhere.base);
            let origin = Provider::start(move |_| Reply::redirect(status, &moved_to));
            let identity = E2eIdentity::from_secret_bytes([17; 32]);
            let error = poll_once(
                &HTTP.get().unwrap(),
                &account(origin.base.clone()),
                &spec(),
                "test-computer",
                &identity,
                &|| true,
                &|_, _| panic!("nothing was claimed"),
            )
            .unwrap_err();
            assert!(error.starts_with(&format!("sealed flow poll: HTTP {status}")), "{error}");
            assert_eq!(origin.requests().len(), 1);
            assert!(elsewhere.requests().is_empty(), "{status}: the redirect was followed");
        }
    }

    #[test]
    fn a_lane_that_keeps_its_client_gives_each_account_only_its_own_credential() {
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let empty = || Provider::start(|_| Reply::ok(&json!({"requests":[]}).to_string()));
        let (a, b) = (empty(), empty());
        for (server, credential) in [(&a, "flk_alpha"), (&b, "flk_beta"), (&a, "flk_gamma")] {
            let mut linked = account(server.base.clone());
            linked.credential = credential.into();
            assert!(!poll_once(
                &HTTP.get().unwrap(),
                &linked,
                &spec(),
                "test-computer",
                &identity,
                &|| true,
                &|_, _| panic!("nothing to run"),
            )
            .unwrap());
        }
        let bearers = |server: &Provider| -> Vec<String> {
            server.requests().iter().map(|r| r.header("authorization").unwrap_or("none").to_string()).collect()
        };
        assert_eq!(bearers(&a), ["Bearer flk_alpha", "Bearer flk_gamma"]);
        assert_eq!(bearers(&b), ["Bearer flk_beta"]);
        assert_eq!((a.connections(), b.connections()), (1, 1));
    }

    #[test]
    #[ignore = "requires a built CLI: set OAIY_CLI then run this test explicitly"]
    fn encrypted_relay_executes_the_real_zipp_cli() {
        assert!(
            std::env::var("OAIY_CLI").is_ok(),
            "set OAIY_CLI to the built CLI entry"
        );
        let identity = E2eIdentity::from_secret_bytes([17; 32]);
        let request = request(
            &identity,
            "flow-one",
            json!({"v":1,"flowId":"flow-one","inputs":{"message":"remote echo"}}),
        );
        // The completion carries the flow's OUTPUT, not the CLI's envelope, so
        // the graph itself asks the question: a flow on the ZIPP VM has no Node
        // `process`; one the host ran itself would.
        let graph = json!({"nodes":[
            {"id":"probe","type":"logic_block","position":{"x":0,"y":0},"data":{"code":"return typeof process;"}},
            {"id":"out","type":"output","position":{"x":200,"y":0},"data":{"value":"$nodes.probe"}}
        ],"edges":[{"id":"edge","source":"probe","target":"out"}]});
        let (url, server) = server(vec![
            (200, json!({"requests":[request.clone()]})),
            (200, json!({"claimed":true,"request":request})),
            (200, json!({"flows":[{"id":"flow-one","flowJson":graph}]})),
            (200, json!({})),
        ]);
        let account = account(url);
        let flows = descriptor::builtin().remove(0).flows.unwrap();
        poll_once(
            &client(),
            &account,
            &spec(),
            "test-computer",
            &identity,
            &|| true,
            &|id, inputs| {
                super::super::flow_runner::execute_sealed(
                    &account,
                    &flows,
                    None,
                    id,
                    inputs,
                    // No descriptor in this temp dir names a script
                    // profile, so the run carries none — the same
                    // state as a provider that publishes none.
                    &std::env::temp_dir(),
                )
            },
        )
        .unwrap();
        let requests = server.join().unwrap();
        assert!(requests[2].starts_with("GET /api/v1/flows "));
        let completion: Value =
            serde_json::from_str(requests[3].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        let opened = open_result(&identity, completion["resultEnvelope"].as_str().unwrap());
        assert_eq!(completion["status"], "done", "{opened}");
        assert_eq!(opened["result"], "undefined", "the flow must have run on ZIPP, where there is no process object: {opened}");
    }
}
