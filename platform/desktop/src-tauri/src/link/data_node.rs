//! Enrolling this desktop as a storage node the linked account can trust.
//!
//! A provider that lets a machine hold encrypted data has to know WHICH
//! machine, and the owner has to be able to say yes to it deliberately. So this
//! desktop mints one long-term Ed25519 signing identity, registers its public
//! half over the authenticated link, and then waits: the node starts with no
//! authority at all, and gains it only when the owner approves it in their
//! browser by signing a certificate over exactly this key.
//!
//! The security of that ceremony rests on ONE thing — the owner comparing a
//! fingerprint shown here against the one shown in the browser. So the
//! fingerprint is derived exactly as the provider derives it (SHA-256 over the
//! raw 32-byte public key, hex) and displayed in the same grouping. A different
//! derivation would show two different strings for the same key and train the
//! user to approve without looking.
//!
//! Registration is idempotent, and a CHANGED key is a rotation: the provider
//! drops the node back to pending and the owner must approve again. That is why
//! the identity is persisted and never silently re-minted — a new key each boot
//! would ask the owner to re-approve forever.
//!
//! Scope: this is the enrolment half. Actually holding datasets — snapshots,
//! account backups, the encrypted local store — is a separate surface this
//! module does not implement, and a node that is approved simply sits ready.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::descriptor::{self, DataNodeSpec};
use super::{LinkHandle, LinkedAccount};

const IDENTITY_FILE: &str = "data-node-signing.key";

/// How often the node record is read back.
///
/// Enrolment itself is hourly, but APPROVAL happens in a browser at a moment
/// nothing here can predict, and the owner is usually looking at both screens
/// when it does. A minute is the difference between "it worked" and "it didn't".
const REFRESH_INTERVAL: Duration = Duration::from_secs(45);

/// STANDARD base64 with padding — the provider decodes the published key
/// strictly, and this app's usual alphabet is URL-safe and unpadded.
const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

/// This desktop's node identity.
pub struct NodeIdentity {
    signing: SigningKey,
}

impl NodeIdentity {
    pub fn from_secret_bytes(bytes: [u8; 32]) -> Self {
        Self {
            signing: SigningKey::from_bytes(&bytes),
        }
    }

    pub fn public_key_b64(&self) -> String {
        B64.encode(self.signing.verifying_key().to_bytes())
    }

    /// SHA-256 over the raw public key, lowercase hex — the provider's own
    /// derivation. It computes this itself and never trusts a value sent to it,
    /// so a different rule here would simply show the owner a fingerprint that
    /// does not match the one they are asked to approve.
    pub fn fingerprint(&self) -> String {
        let digest = Sha256::digest(self.signing.verifying_key().to_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Load this machine's node identity, minting one on first use.
    ///
    /// An unreadable key is a HARD failure, not a re-mint: a new key is a
    /// rotation that revokes the node's authority and asks the owner to approve
    /// it again, which is not something to do by accident once per boot.
    pub fn load_or_create(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join(IDENTITY_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let bytes = B64
                    .decode(text.trim())
                    .ok()
                    .and_then(|b| <[u8; 32]>::try_from(b.as_slice()).ok());
                return bytes.map(Self::from_secret_bytes).ok_or_else(|| {
                    format!(
                        "the data-node identity at {} is unreadable — move it aside to mint a \
                         new one, but the owner will have to approve this desktop again",
                        path.display()
                    )
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("could not read the data-node identity: {e}")),
        }
        let mut secret = [0u8; 32];
        getrandom::getrandom(&mut secret)
            .map_err(|e| format!("could not read OS randomness for the node identity: {e}"))?;
        let identity = Self::from_secret_bytes(secret);
        let mut tmp = path.clone();
        tmp.set_extension("tmp");
        std::fs::create_dir_all(dir)
            .and_then(|()| std::fs::write(&tmp, B64.encode(secret).as_bytes()))
            .and_then(|()| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
                }
                std::fs::rename(&tmp, &path)
            })
            .map_err(|e| format!("could not persist the data-node identity: {e}"))?;
        Ok(identity)
    }
}

/// What a UI may show about this desktop's enrolment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataNodeStatus {
    /// Full hex; the UI groups it for reading.
    pub fingerprint: String,
    /// `pending`, `approved` or `revoked` — the provider's own vocabulary.
    pub status: String,
    /// Approved AND holding an unexpired owner certificate. A node can be
    /// `approved` with an expired certificate and have no authority, so the two
    /// are reported separately rather than collapsed into one word.
    pub approved: bool,
    pub key_generation: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

/// Group a fingerprint the way the provider's browser UI does, so the two
/// strings can be compared at a glance rather than character by character.
pub fn display_fingerprint(hex: &str) -> String {
    hex.chars()
        .take(24)
        .collect::<Vec<_>>()
        .chunks(4)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Register on link, then keep the enrolment fresh.
pub fn spawn(store: LinkHandle) {
    std::thread::spawn(move || {
        let identity = match NodeIdentity::load_or_create(store.data_dir()) {
            Ok(i) => Arc::new(i),
            Err(e) => {
                log::error!("data-node enrolment has no usable identity: {e}");
                return;
            }
        };
        let mut next_register = std::time::Instant::now();
        let mut next_refresh = std::time::Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(5));
            let Some(account) = store.account() else {
                // Unlinked: act promptly when a link next appears rather than
                // inheriting an old timer.
                next_register = std::time::Instant::now();
                next_refresh = std::time::Instant::now();
                continue;
            };
            let Some(spec) =
                descriptor::find(store.data_dir(), &account.connector_id).and_then(|d| d.data_node)
            else {
                continue;
            };
            let now = std::time::Instant::now();

            // Registering is a WRITE and belongs on the slow schedule.
            if now >= next_register {
                match register(&account, &spec, &identity) {
                    Ok(status) => store.note_data_node(Ok(status)),
                    Err(e) => store.note_data_node(Err(e)),
                }
                next_register = now + Duration::from_secs(spec.interval_seconds.max(60));
                // A register answers with the record too, so the read-back can
                // wait its full interval rather than firing straight after.
                next_refresh = now + REFRESH_INTERVAL;
                continue;
            }

            // Reading the record back is cheap, and it is the ONLY way this
            // desktop learns the owner approved it. Left to the hourly
            // registration, an approval sat invisible for up to an hour —
            // which reads as the approval not having worked.
            if now >= next_refresh {
                if let Some(status) = refresh(&account, &spec) {
                    store.note_data_node(Ok(status));
                }
                next_refresh = now + REFRESH_INTERVAL;
            }
        }
    });
}

/// One registration. Idempotent server-side.
fn register(
    account: &LinkedAccount,
    spec: &DataNodeSpec,
    identity: &NodeIdentity,
) -> Result<DataNodeStatus, String> {
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("could not build the enrolment client: {e}"))?;
    let body = json!({
        "signingPublicKey": identity.public_key_b64(),
        "displayName": account.account_name.clone().unwrap_or_else(|| "OAIY Desktop".into()),
        "capabilities": spec.capabilities,
        "protocolMin": spec.protocol_min,
        "protocolMax": spec.protocol_max,
    });
    let resp = http
        .post(super::oauth::join(&account.base_url, &spec.register_path))
        .bearer_auth(&account.credential)
        .json(&body)
        .send()
        .map_err(|e| format!("could not reach the data-node lane: {e}"))?;

    let status = resp.status();
    let payload: Value = resp.json().unwrap_or(Value::Null);
    if !status.is_success() {
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the enrolment");
        // 409 while the connection row is still being created is ordinary on a
        // fresh link, and the next tick fixes it.
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    node_status(&payload).ok_or_else(|| "the provider returned no node record".to_string())
}

/// Read this desktop's node record back.
///
/// Returns `None` rather than an error on any failure: a transient read that
/// could not be made is not news, and replacing a good record with an error
/// would blank the fingerprint the owner is mid-way through comparing.
fn refresh(account: &LinkedAccount, spec: &DataNodeSpec) -> Option<DataNodeStatus> {
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .ok()?;
    let resp = http
        .get(super::oauth::join(&account.base_url, &spec.self_path))
        .bearer_auth(&account.credential)
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    node_status(&resp.json::<Value>().ok()?)
}

/// Read the node record out of a register or self response.
fn node_status(payload: &Value) -> Option<DataNodeStatus> {
    let node = payload
        .pointer("/data/node")
        .or_else(|| payload.get("node"))
        .or(Some(payload))?;
    let fingerprint = node.get("fingerprint").and_then(Value::as_str)?;
    Some(DataNodeStatus {
        fingerprint: fingerprint.to_string(),
        status: node
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("pending")
            .to_string(),
        approved: node.get("approved").and_then(Value::as_bool).unwrap_or(false),
        key_generation: node
            .get("signingKeyGeneration")
            .and_then(Value::as_u64)
            .unwrap_or(1) as u32,
        display_name: node
            .get("displayName")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_matches_the_providers_own_derivation() {
        // The whole approval ceremony is the owner comparing this string with
        // the one their browser shows. A different rule here shows two
        // different fingerprints for one key and teaches them not to look.
        //
        // Provider rule: sha256 over the RAW 32-byte public key, lowercase hex.
        let identity = NodeIdentity::from_secret_bytes([7u8; 32]);
        let raw = B64.decode(identity.public_key_b64()).unwrap();
        assert_eq!(raw.len(), 32);
        let expected: String = Sha256::digest(&raw).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(identity.fingerprint(), expected);
        assert_eq!(identity.fingerprint().len(), 64);
        assert!(identity.fingerprint().chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn the_published_key_is_standard_base64_the_provider_will_accept() {
        // It is decoded strictly, and this app's usual alphabet is URL-safe and
        // unpadded — which would be refused outright.
        for seed in [1u8, 0x5a, 0xff] {
            let published = NodeIdentity::from_secret_bytes([seed; 32]).public_key_b64();
            assert_eq!(published.len(), 44, "{published}");
            assert!(!published.contains('-') && !published.contains('_'), "{published}");
            assert_eq!(B64.decode(&published).unwrap().len(), 32);
        }
    }

    #[test]
    fn the_fingerprint_is_grouped_the_way_the_browser_groups_it() {
        // Same 24 characters, same spacing — the two are meant to be compared
        // at a glance, and different formatting makes that a chore people skip.
        let fp = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(display_fingerprint(fp), "0123 4567 89ab cdef 0123 4567");
        assert_eq!(display_fingerprint("abcd").as_str(), "abcd");
        assert_eq!(display_fingerprint("").as_str(), "");
    }

    #[test]
    fn the_identity_survives_a_restart_and_is_never_silently_rotated() {
        // A new key revokes this node's authority and asks the owner to approve
        // it again. Doing that once per boot would make the feature unusable.
        let dir = std::env::temp_dir().join(format!("oaiy-dn-{}", uuid::Uuid::new_v4().simple()));
        let first = NodeIdentity::load_or_create(&dir).unwrap();
        let second = NodeIdentity::load_or_create(&dir).unwrap();
        assert_eq!(first.public_key_b64(), second.public_key_b64());
        assert_eq!(first.fingerprint(), second.fingerprint());

        std::fs::write(dir.join(IDENTITY_FILE), b"not a key").unwrap();
        match NodeIdentity::load_or_create(&dir) {
            Ok(_) => panic!("a broken identity must fail loudly, not rotate"),
            Err(e) => assert!(e.contains("approve this desktop again"), "{e}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_node_record_is_read_from_the_shape_the_provider_returns() {
        // The record is nested under data.node; reading the top level would
        // leave the panel with no fingerprint to compare and no status.
        let payload = json!({
            "data": { "node": {
                "id": "dn_0123",
                "displayName": "OAIY Desktop on DESKTOP-HESQH3A",
                "fingerprint": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
                "signingKeyGeneration": 2,
                "status": "pending",
                "approved": false,
                "capabilities": ["storage"],
            }}
        });
        let s = node_status(&payload).expect("the real shape must parse");
        assert_eq!(s.status, "pending");
        assert!(!s.approved);
        assert_eq!(s.key_generation, 2);
        assert_eq!(display_fingerprint(&s.fingerprint), "a1b2 c3d4 e5f6 0718 293a 4b5c");

        // Approved-with-certificate is the state that actually grants anything.
        let approved = json!({ "data": { "node": {
            "fingerprint": "ff", "status": "approved", "approved": true, "signingKeyGeneration": 1 }}});
        assert!(node_status(&approved).unwrap().approved);
        // A body with no node at all is None, not a panic.
        assert!(node_status(&json!({ "data": {} })).is_none());
    }
}
