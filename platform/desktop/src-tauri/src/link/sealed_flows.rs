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
    let pending = http
        .get(super::oauth::join(&account.base_url, &spec.pending_path))
        .bearer_auth(&account.credential)
        .query(&[
            ("instanceId", instance.to_owned()),
            ("wait", (spec.wait_seconds * 1000).to_string()),
            ("limit", "1".to_string()),
        ])
        .send()
        .map_err(|_| "sealed flow poll could not reach the provider")?;
    if !pending.status().is_success() {
        return Err(format!("sealed flow poll: HTTP {}", pending.status()));
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
        .bearer_auth(&account.credential)
        .json(&json!({"instanceId": instance}))
        .send()
        .map_err(|_| "sealed flow claim could not be confirmed; execution skipped")?;
    if claimed.status().as_u16() == 409 {
        return Ok(false);
    }
    if !claimed.status().is_success() {
        return Err(format!("sealed flow claim: HTTP {}", claimed.status()));
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
    for attempt in 0..3 {
        match http
            .post(&url)
            .bearer_auth(&account.credential)
            .json(&body)
            .send()
        {
            Ok(response) if response.status().is_success() => return Ok(true),
            Ok(response) if response.status().is_client_error() => {
                return Err(format!(
                    "sealed flow completion: HTTP {}; flow will not be repeated",
                    response.status()
                ))
            }
            _ => {
                if attempt < 2 {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        }
    }
    Err("sealed flow completion could not be confirmed; flow will not be repeated".into())
}

pub fn spawn(store: LinkHandle, node: Option<NodeHandle>) {
    std::thread::spawn(move || {
        let identity = match E2eIdentity::load_or_create(store.data_dir()) {
            Ok(identity) => identity,
            Err(error) => {
                store.note_sealed_flow(Some(error.clone()));
                log::error!("sealed flow identity unavailable: {error}");
                return;
            }
        };
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
            let http = match Client::builder()
                .timeout(Duration::from_secs(spec.wait_seconds + 15))
                .redirect(reqwest::redirect::Policy::none())
                .build()
            {
                Ok(http) => http,
                Err(_) => {
                    std::thread::sleep(Duration::from_secs(10));
                    continue;
                }
            };
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
                    )
                },
            );
            if store.account().as_ref() == Some(&account) {
                store.note_sealed_flow(result.as_ref().err().cloned());
            }
            match result {
                Ok(true) => (),
                Ok(false) => std::thread::sleep(Duration::from_millis(500)),
                Err(error) => {
                    log::warn!("{error}");
                    std::thread::sleep(Duration::from_secs(spec.error_backoff_seconds));
                }
            }
        }
    });
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
        let graph = json!({"nodes":[
            {"id":"start","type":"input","position":{"x":0,"y":0},"data":{}},
            {"id":"out","type":"output","position":{"x":200,"y":0},"data":{"value":"$inputs.message"}}
        ],"edges":[{"id":"edge","source":"start","target":"out"}]});
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
                super::super::flow_runner::execute_sealed(&account, &flows, None, id, inputs)
            },
        )
        .unwrap();
        let requests = server.join().unwrap();
        assert!(requests[2].starts_with("GET /api/v1/flows "));
        let completion: Value =
            serde_json::from_str(requests[3].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        let opened = open_result(&identity, completion["resultEnvelope"].as_str().unwrap());
        assert_eq!(completion["status"], "done", "{opened}");
        assert_eq!(opened["result"], "remote echo");
    }
}
