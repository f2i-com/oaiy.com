//! Brokering a companion admission from an upstream issuer.
//!
//! The plugin hosts the WebRTC peer, but it must not be the thing that decides
//! which phones may reach it — so it asks the host, and the host asks whichever
//! issuer the user has connected. FormLogic is one such issuer; the self-host
//! relay's README documents the same contract for a custom one. Nothing here
//! names FormLogic, which is what makes the connection generic.
//!
//! OAIY deliberately does NOT mint admissions itself. Minting requires the
//! gateway's HMAC secret, and a desktop that holds that secret can admit any
//! device to any deployment. Forwarding keeps OAIY a requester: the worst a
//! compromised OAIY can do is ask for an admission it is already entitled to.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Where to obtain admissions, and the credential that proves we may.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UpstreamConfig {
    /// Issuer base, e.g. `https://formlogic.com/api/v1`. The admission path is
    /// appended, so one setting serves any deployment of the same contract.
    pub base_url: String,
    /// Bearer presented to the issuer.
    ///
    /// Never serialized outward: [`UpstreamStatus`] reports only whether one is
    /// present. It is the credential that makes this desktop *this* account's,
    /// and a status endpoint is read by the plugin screen.
    pub token: String,
    /// Used when the plugin names no app of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
}

/// What a UI may know about the upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpstreamStatus {
    pub configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    /// Whether a credential is held — never the credential.
    pub has_token: bool,
}

/// The fields an admission binds, echoed back by the issuer.
///
/// Split out because verifying the echo is the single most important check in
/// this file: see [`verify_echo`].
const BOUND_FIELDS: [&str; 5] = [
    "endpointPublicKey",
    "holderKeyThumbprint",
    "approvedPeerKeyThumbprints",
    "peerRosterRevision",
    "peerRosterHash",
];

/// Deliberately inside the plugin's own 10s admission timeout.
///
/// If we waited longer, the plugin would give up first and report "Desktop
/// admission refresh timed out" — hiding the real reason, which is the one
/// piece of information the user needs to fix their relay settings.
const BROKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);

pub struct UpstreamStore {
    path: PathBuf,
    config: Mutex<Option<UpstreamConfig>>,
}

pub type UpstreamHandle = Arc<UpstreamStore>;

impl UpstreamStore {
    pub fn open(path: PathBuf) -> UpstreamHandle {
        let config = std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<UpstreamConfig>(&raw).ok());
        Arc::new(Self {
            path,
            config: Mutex::new(config),
        })
    }

    pub fn status(&self) -> UpstreamStatus {
        let guard = self.config.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            None => UpstreamStatus {
                configured: false,
                base_url: None,
                app_id: None,
                has_token: false,
            },
            Some(c) => UpstreamStatus {
                configured: true,
                base_url: Some(c.base_url.clone()),
                app_id: c.app_id.clone(),
                has_token: !c.token.is_empty(),
            },
        }
    }

    pub fn get(&self) -> Option<UpstreamConfig> {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn set(&self, config: UpstreamConfig) -> Result<UpstreamStatus, String> {
        let config = normalize(config)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create the companion directory: {e}"))?;
        }
        let raw = serde_json::to_string_pretty(&config)
            .map_err(|e| format!("could not encode the upstream config: {e}"))?;
        std::fs::write(&self.path, raw)
            .map_err(|e| format!("could not save the upstream config: {e}"))?;
        restrict_to_owner(&self.path);
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = Some(config);
        Ok(self.status())
    }

    pub fn clear(&self) -> UpstreamStatus {
        let _ = std::fs::remove_file(&self.path);
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.status()
    }
}

/// Keep the bearer out of reach of other accounts on a shared machine.
///
/// Best-effort: on Windows the file already inherits a per-user profile ACL,
/// and a failure here must not stop the user configuring their relay.
fn restrict_to_owner(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Reject an issuer we would be unwise to send a credential to.
fn normalize(mut config: UpstreamConfig) -> Result<UpstreamConfig, String> {
    config.base_url = config.base_url.trim().trim_end_matches('/').to_string();
    config.token = config.token.trim().to_string();
    config.app_id = config
        .app_id
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty());

    if config.token.is_empty() {
        return Err("an upstream token is required".into());
    }
    // Same rule as the account link: https anywhere, plain http only for
    // loopback or a .local name. A relay is as likely to be a local deployment
    // as the provider is.
    if !crate::origin::may_carry_credential(&config.base_url) {
        return Err(crate::origin::INSECURE_ADDRESS_HELP.into());
    }
    if config.base_url.len() > 512 {
        return Err("the upstream base URL is too long".into());
    }
    Ok(config)
}

/// The request body the issuer contract expects.
///
/// Carries public proof-of-possession policy and nothing else — the desktop
/// seed never enters it.
pub fn admission_request_body(
    app_id: &str,
    plugin_id: &str,
    display_name: Option<&str>,
    binding: &Value,
) -> Result<Value, String> {
    let mut body = json!({ "appId": app_id, "pluginId": plugin_id });
    if let Some(name) = display_name.map(str::trim).filter(|n| !n.is_empty()) {
        body["displayName"] = json!(name);
    }
    for field in BOUND_FIELDS {
        let value = binding
            .get(field)
            .ok_or_else(|| format!("the endpoint binding is missing {field}"))?;
        body[field] = value.clone();
    }
    Ok(body)
}

/// Confirm the issuer bound the admission to the roster WE sent.
///
/// This is the check that makes forwarding safe. The issuer signs the peer
/// policy into the token, so if the value it echoes differs from what we asked
/// for, the token in our hand admits a set of devices the owner never approved
/// on this machine. Refusing is the only correct response: we cannot inspect
/// the signed claims, so the echo is the only evidence available.
pub fn verify_echo(sent: &Value, received: &Value) -> Result<(), String> {
    for field in BOUND_FIELDS {
        let ours = sent.get(field);
        let theirs = received.get(field);
        if ours != theirs {
            return Err(format!(
                "the issuer returned a different {field} than this desktop asked for; \
                 the admission would bind a peer policy the owner did not approve"
            ));
        }
    }
    Ok(())
}

/// Ask the issuer for an admission, and refuse anything that does not match.
///
/// Blocking on purpose. The only caller is the plugin RPC handler, which runs
/// on a thread of its own precisely so a network round trip cannot stall the
/// plugin's output stream — so a blocking client here is simpler than threading
/// a runtime handle through the plugin host for one call.
pub fn broker(
    config: &UpstreamConfig,
    app_id: &str,
    plugin_id: &str,
    display_name: Option<&str>,
    binding: &Value,
) -> Result<Value, String> {
    let body = admission_request_body(app_id, plugin_id, display_name, binding)?;
    let url = format!("{}/aokie-companion/admission", config.base_url);
    let client = reqwest::blocking::Client::builder()
        .timeout(BROKER_TIMEOUT)
        .build()
        .map_err(|e| format!("could not build the upstream client: {e}"))?;

    let resp = client
        .post(&url)
        .bearer_auth(&config.token)
        .json(&body)
        .send()
        .map_err(|e| format!("the admission issuer could not be reached: {e}"))?;

    let status = resp.status();
    let value: Value = resp
        .json()
        .map_err(|e| format!("the admission issuer returned a non-JSON response: {e}"))?;
    if !status.is_success() {
        let message = value
            .get("error")
            .and_then(Value::as_str)
            .or_else(|| value.get("message").and_then(Value::as_str))
            .unwrap_or("the admission issuer refused the request");
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }

    verify_echo(&body, &value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> Value {
        json!({
            "endpointPublicKey": { "algorithm": "ed25519", "publicKey": "pk", "thumbprint": "tp" },
            "holderKeyThumbprint": "tp",
            "approvedPeerKeyThumbprints": ["a", "b"],
            "peerRosterRevision": 7,
            "peerRosterHash": "hash"
        })
    }

    /// A one-shot HTTP issuer. Small enough to be obviously correct, which
    /// matters more here than generality: these tests are about what the broker
    /// does with a reply, not about HTTP.
    fn stub_issuer(reply: &'static str, status_line: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let Ok((mut stream, _)) = listener.accept() else { return };
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            // Read headers, then exactly the declared body length.
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
            let _ = tx.send(String::from_utf8_lossy(&raw).to_string());
            let _ = write!(
                stream,
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.flush();
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    fn config(base: String) -> UpstreamConfig {
        UpstreamConfig { base_url: base, token: "flk_secret".into(), app_id: None }
    }

    #[test]
    fn an_honest_issuer_round_trips_and_the_bearer_is_presented() {
        let reply = r#"{"accessToken":"aokie-adm-v2.aa.bb","gatewayUrl":"wss://relay.example/v2",
            "endpointPublicKey":{"algorithm":"ed25519","publicKey":"pk","thumbprint":"tp"},
            "holderKeyThumbprint":"tp","approvedPeerKeyThumbprints":["a","b"],
            "peerRosterRevision":7,"peerRosterHash":"hash"}"#;
        let (base, rx) = stub_issuer(reply, "200 OK");
        let out = broker(&config(base), "app_1", "aokie", Some("Aokie Desktop"), &binding()).unwrap();
        assert_eq!(out["accessToken"], "aokie-adm-v2.aa.bb");
        assert_eq!(out["gatewayUrl"], "wss://relay.example/v2");

        let request = rx.recv().unwrap();
        assert!(request.starts_with("POST /aokie-companion/admission "), "{}", &request[..60]);
        assert!(request.contains("flk_secret"), "the bearer must be presented");
        // …and the desktop's private seed must not be anywhere in the request.
        assert!(!request.contains("privateKey") && !request.contains("seed"), "{request}");
    }

    #[test]
    fn an_issuer_that_widens_the_roster_is_refused_over_the_wire() {
        // The end-to-end form of the check: even a 200 with a valid-looking
        // token is thrown away, because the token would be signed over a roster
        // containing a device this desktop never approved.
        let reply = r#"{"accessToken":"aokie-adm-v2.aa.bb",
            "endpointPublicKey":{"algorithm":"ed25519","publicKey":"pk","thumbprint":"tp"},
            "holderKeyThumbprint":"tp","approvedPeerKeyThumbprints":["a","b","attacker"],
            "peerRosterRevision":7,"peerRosterHash":"hash"}"#;
        let (base, _rx) = stub_issuer(reply, "200 OK");
        let err = broker(&config(base), "app_1", "aokie", None, &binding()).unwrap_err();
        assert!(err.contains("approvedPeerKeyThumbprints"), "{err}");
    }

    #[test]
    fn an_issuer_refusal_is_reported_with_its_own_reason() {
        // The user has to be able to tell "not linked" from "unreachable" —
        // they are different problems with different fixes.
        let (base, _rx) = stub_issuer(r#"{"error":"this desktop is not linked"}"#, "403 Forbidden");
        let err = broker(&config(base), "app_1", "aokie", None, &binding()).unwrap_err();
        assert!(err.contains("403"), "{err}");
        assert!(err.contains("not linked"), "the issuer's reason must survive: {err}");
    }

    #[test]
    fn an_unreachable_issuer_says_so_rather_than_looking_like_a_refusal() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener); // nothing is listening now
        let err = broker(
            &config(format!("http://127.0.0.1:{port}")),
            "app_1",
            "aokie",
            None,
            &binding(),
        )
        .unwrap_err();
        assert!(err.contains("could not be reached"), "{err}");
    }

    #[test]
    fn the_request_carries_public_policy_and_no_secret() {
        let body = admission_request_body("app_1", "aokie", Some(" Aokie Desktop "), &binding()).unwrap();
        assert_eq!(body["appId"], "app_1");
        assert_eq!(body["pluginId"], "aokie");
        assert_eq!(body["displayName"], "Aokie Desktop", "trimmed");
        assert_eq!(body["peerRosterRevision"], 7);
        // Whatever else changes, a private key must never appear here.
        let raw = body.to_string();
        assert!(!raw.contains("seed") && !raw.contains("privateKey"), "{raw}");
    }

    #[test]
    fn a_binding_missing_a_bound_field_is_refused_before_the_request_is_sent() {
        // Sending a partial binding would let the issuer choose the missing
        // half of the peer policy.
        let mut b = binding();
        b.as_object_mut().unwrap().remove("peerRosterHash");
        let err = admission_request_body("app_1", "aokie", None, &b).unwrap_err();
        assert!(err.contains("peerRosterHash"), "{err}");
    }

    #[test]
    fn an_echo_that_matches_is_accepted() {
        let sent = admission_request_body("app_1", "aokie", None, &binding()).unwrap();
        let mut received = sent.clone();
        // The issuer legitimately adds its own fields; only the bound ones matter.
        received["accessToken"] = json!("aokie-adm-v2.aa.bb");
        received["gatewayUrl"] = json!("wss://relay.example/v2");
        verify_echo(&sent, &received).unwrap();
    }

    #[test]
    fn an_issuer_that_widens_the_roster_is_refused() {
        // The exact attack this check exists for: the admission would be signed
        // over a roster containing a device this desktop never approved.
        let sent = admission_request_body("app_1", "aokie", None, &binding()).unwrap();
        let mut received = sent.clone();
        received["approvedPeerKeyThumbprints"] = json!(["a", "b", "attacker"]);
        let err = verify_echo(&sent, &received).unwrap_err();
        assert!(err.contains("approvedPeerKeyThumbprints"), "{err}");
        assert!(err.contains("did not approve"), "the message must say why it matters");
    }

    #[test]
    fn an_issuer_that_swaps_the_holder_key_is_refused() {
        // Rebinding the holder would hand the session to a different endpoint.
        let sent = admission_request_body("app_1", "aokie", None, &binding()).unwrap();
        let mut received = sent.clone();
        received["holderKeyThumbprint"] = json!("someone-else");
        assert!(verify_echo(&sent, &received).is_err());
    }

    #[test]
    fn a_missing_echo_is_refused_rather_than_treated_as_agreement() {
        // `None != Some(..)` must not quietly pass: an issuer that omits the
        // field has not agreed to anything.
        let sent = admission_request_body("app_1", "aokie", None, &binding()).unwrap();
        let mut received = sent.clone();
        received.as_object_mut().unwrap().remove("peerRosterRevision");
        assert!(verify_echo(&sent, &received).is_err());
    }

    #[test]
    fn plain_http_to_a_remote_issuer_is_refused() {
        // The bearer would cross the network in clear.
        let err = normalize(UpstreamConfig {
            base_url: "http://relay.example/api/v1".into(),
            token: "t".into(),
            app_id: None,
        })
        .unwrap_err();
        assert!(err.contains("https"), "{err}");
    }

    #[test]
    fn a_local_relay_may_be_reached_over_plain_http() {
        let c = normalize(UpstreamConfig {
            base_url: "http://relay.local:18788/api/v1".into(),
            token: "t".into(),
            app_id: None,
        })
        .unwrap();
        assert_eq!(c.base_url, "http://relay.local:18788/api/v1");
    }

    #[test]
    fn loopback_http_is_allowed_for_a_local_relay() {
        let c = normalize(UpstreamConfig {
            base_url: "http://127.0.0.1:18788/api/v1/".into(),
            token: " t ".into(),
            app_id: Some("  ".into()),
        })
        .unwrap();
        assert_eq!(c.base_url, "http://127.0.0.1:18788/api/v1", "trailing slash trimmed");
        assert_eq!(c.token, "t");
        assert_eq!(c.app_id, None, "a blank app id is absent, not empty");
    }

    #[test]
    fn a_config_without_a_token_is_refused() {
        let err = normalize(UpstreamConfig {
            base_url: "https://formlogic.com/api/v1".into(),
            token: "   ".into(),
            app_id: None,
        })
        .unwrap_err();
        assert!(err.contains("token"), "{err}");
    }

    #[test]
    fn the_status_never_carries_the_token() {
        let dir = std::env::temp_dir().join(format!("oaiy-upstream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = UpstreamStore::open(dir.join("upstream.json"));
        assert!(!store.status().configured);

        store
            .set(UpstreamConfig {
                base_url: "https://formlogic.com/api/v1".into(),
                token: "flk_supersecret".into(),
                app_id: Some("app_1".into()),
            })
            .unwrap();
        let status = store.status();
        assert!(status.configured && status.has_token);
        let raw = serde_json::to_string(&status).unwrap();
        assert!(!raw.contains("flk_supersecret"), "{raw}");

        // …but it survives a restart, or the plugin would lose its broker on
        // every launch.
        let reopened = UpstreamStore::open(dir.join("upstream.json"));
        assert_eq!(reopened.get().unwrap().token, "flk_supersecret");
        assert!(!reopened.clear().configured);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
