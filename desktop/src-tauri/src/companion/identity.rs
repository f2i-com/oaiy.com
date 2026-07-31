//! Endpoint identity for companion-device trust.
//!
//! A companion app — the Aokie phone app today — takes a live call's audio on
//! its own microphone and speakers. The audio itself travels peer-to-peer over
//! encrypted WebRTC; the relay only carries signalling and never sees decoded
//! PCM. What makes that safe is this: both ends hold an Ed25519 identity, and
//! the desktop only ever accepts a phone whose signature it can verify AND
//! whose owner approved it here, on this machine.
//!
//! # Why the wire format is copied exactly, not designed
//!
//! The mobile app already exists and already verifies these signatures. Every
//! byte that goes into a signed payload — the canonical thumbprint string, the
//! base64 alphabet, the field names — is fixed by that peer. `URL_SAFE_NO_PAD`
//! rather than standard base64, and the thumbprint's canonical JWK ordering
//! (`crv`, `kty`, `x`), are not stylistic choices; change either and every
//! signature this produces is rejected by a phone we cannot patch.
//!
//! # Generic over the plugin, unlike the implementation it is ported from
//!
//! FormLogic Desktop hardcodes `aokie` as the owning plugin and leaves a TODO
//! for "capability-based gating when a second broker plugin exists". OAIY is a
//! generic plugin host, so the owner is a parameter and the gate is the
//! declared `companion.admission` capability. Nothing here knows what a phone
//! bridge is.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Longest acceptable protocol identifier. Bounded so a hostile relay cannot
/// push an unbounded string into a filename or a log line.
const MAX_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKeyAlgorithm {
    Ed25519,
}

/// A public OKP identity, in the shape the companion protocol defines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EndpointPublicKey {
    pub algorithm: EndpointKeyAlgorithm,
    pub public_key: String,
    pub thumbprint: String,
}

impl EndpointPublicKey {
    fn from_verifying_key(key: &VerifyingKey) -> Self {
        let public_key = URL_SAFE_NO_PAD.encode(key.to_bytes());
        let thumbprint = endpoint_thumbprint(&public_key);
        Self {
            algorithm: EndpointKeyAlgorithm::Ed25519,
            public_key,
            thumbprint,
        }
    }

    /// The key, but only once its self-description checks out.
    ///
    /// The thumbprint is re-derived rather than trusted: it arrives over an
    /// untrusted relay, and a peer that could name its own thumbprint could
    /// impersonate an already-approved device by presenting a different key
    /// under a trusted fingerprint.
    pub fn verifying_key(&self) -> Result<VerifyingKey, String> {
        let bytes = URL_SAFE_NO_PAD
            .decode(&self.public_key)
            .map_err(|_| "endpoint public key is not base64url".to_string())?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "endpoint public key must be 32 bytes".to_string())?;
        let key = VerifyingKey::from_bytes(&bytes)
            .map_err(|_| "endpoint public key is not valid Ed25519".to_string())?;
        safe_id("endpoint key thumbprint", &self.thumbprint)?;
        if self.thumbprint != endpoint_thumbprint(&self.public_key) {
            return Err("endpoint key thumbprint does not match its public key".into());
        }
        Ok(key)
    }

    pub fn verify(&self, message: &[u8], signature: &str) -> Result<(), String> {
        let key = self.verifying_key()?;
        let bytes = URL_SAFE_NO_PAD
            .decode(signature)
            .map_err(|_| "pairing signature is not base64url".to_string())?;
        let bytes: [u8; 64] = bytes
            .try_into()
            .map_err(|_| "pairing signature must be 64 bytes".to_string())?;
        key.verify(message, &Signature::from_bytes(&bytes))
            .map_err(|_| "mobile endpoint signature verification failed".to_string())
    }
}

/// Where the private key lives.
///
/// Recorded and shown to the user because it is the difference between a key
/// the OS protects and one sitting in a file this process can read — a
/// distinction they are entitled to see before trusting a remote microphone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyProtection {
    WindowsCredentialManager,
    SoftwareFile,
}

impl KeyProtection {
    pub fn label(self) -> &'static str {
        match self {
            Self::WindowsCredentialManager => "Windows Credential Manager",
            Self::SoftwareFile => "Software key file",
        }
    }
}

/// A companion endpoint the owner has approved.
///
/// Never the handset paired over Bluetooth — that is the call's other end. This
/// is a device that lends the call its microphone and speakers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovedMobile {
    pub device_id: String,
    pub display_name: String,
    pub endpoint_key: EndpointPublicKey,
    pub approved_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RevokedMobile {
    device_id: String,
    thumbprint: String,
    revoked_at: DateTime<Utc>,
}

/// What is persisted between runs.
///
/// `revision` increments on every change and feeds [`peer_roster_hash`], which
/// is how a peer detects that its view of who-is-trusted is stale.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PersistedRoster {
    #[serde(default)]
    schema_version: u16,
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    approved: Vec<ApprovedMobile>,
    #[serde(default)]
    revoked: Vec<RevokedMobile>,
}

/// What the pairing UI renders.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IdentityStatus {
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protection: Option<KeyProtection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protection_label: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_key: Option<EndpointPublicKey>,
    pub roster_revision: u64,
    pub roster_hash: String,
    pub approved_mobiles: Vec<ApprovedMobile>,
    pub pending_approvals: Vec<PendingMobileApproval>,
    pub remote_access_ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// A mobile that proved possession of its key and now awaits the owner's word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PendingMobileApproval {
    pub id: String,
    pub device_id: String,
    pub display_name: String,
    pub endpoint_key: EndpointPublicKey,
    pub received_at: DateTime<Utc>,
}

struct Inner {
    signing_key: Option<SigningKey>,
    protection: KeyProtection,
    roster: PersistedRoster,
    pending: Vec<PendingMobileApproval>,
    warning: Option<String>,
}

/// The desktop's half of companion trust.
pub struct EndpointIdentity {
    root: PathBuf,
    /// The plugin this identity belongs to. One identity per broker plugin, so
    /// revoking one plugin's companions cannot touch another's.
    owner_plugin: String,
    inner: Mutex<Inner>,
}

pub type EndpointIdentityHandle = Arc<EndpointIdentity>;

impl EndpointIdentity {
    /// Load (or mint) the identity for `owner_plugin` under `root`.
    pub fn open(root: PathBuf, owner_plugin: &str) -> EndpointIdentityHandle {
        let dir = root.join("companion").join(owner_plugin);
        let roster = std::fs::read_to_string(dir.join("roster.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<PersistedRoster>(&t).ok())
            .unwrap_or_default();

        let (signing_key, protection, warning) = match load_or_create_key(&dir) {
            Ok((k, p)) => (Some(k), p, None),
            Err(e) => (None, KeyProtection::SoftwareFile, Some(e)),
        };

        Arc::new(Self {
            root,
            owner_plugin: owner_plugin.to_string(),
            inner: Mutex::new(Inner {
                signing_key,
                protection,
                roster,
                pending: Vec::new(),
                warning,
            }),
        })
    }

    fn dir(&self) -> PathBuf {
        self.root.join("companion").join(&self.owner_plugin)
    }

    pub fn status(&self) -> IdentityStatus {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let endpoint_key = inner
            .signing_key
            .as_ref()
            .map(|k| EndpointPublicKey::from_verifying_key(&k.verifying_key()));
        let thumbprints: Vec<String> = inner
            .roster
            .approved
            .iter()
            .map(|m| m.endpoint_key.thumbprint.clone())
            .collect();
        IdentityStatus {
            available: endpoint_key.is_some(),
            protection: endpoint_key.as_ref().map(|_| inner.protection),
            protection_label: endpoint_key.as_ref().map(|_| inner.protection.label()),
            endpoint_key,
            roster_revision: inner.roster.revision,
            roster_hash: peer_roster_hash(inner.roster.revision, &thumbprints),
            approved_mobiles: inner.roster.approved.clone(),
            pending_approvals: inner.pending.clone(),
            // Ready only when there is a key AND something approved to use it.
            // Reporting ready with an empty roster would tell the user remote
            // audio works when no device could possibly answer.
            remote_access_ready: inner.signing_key.is_some() && !inner.roster.approved.is_empty(),
            warning: inner.warning.clone(),
        }
    }

    /// Approve a mobile that has already proved possession of its key.
    pub fn approve(&self, pending_id: &str) -> Result<ApprovedMobile, String> {
        safe_id("pending approval id", pending_id)?;
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let idx = inner
            .pending
            .iter()
            .position(|p| p.id == pending_id)
            .ok_or_else(|| "no pending approval with that id".to_string())?;
        let p = inner.pending.remove(idx);

        let approved = ApprovedMobile {
            device_id: p.device_id,
            display_name: p.display_name,
            endpoint_key: p.endpoint_key,
            approved_at: Utc::now(),
        };
        // Replace rather than append: re-pairing the same handset must not
        // leave a second entry the user has to revoke twice.
        inner
            .roster
            .approved
            .retain(|m| m.endpoint_key.thumbprint != approved.endpoint_key.thumbprint);
        inner.roster.approved.push(approved.clone());
        inner.roster.revision += 1;
        self.persist(&inner.roster);
        Ok(approved)
    }

    pub fn deny(&self, pending_id: &str) -> Result<(), String> {
        safe_id("pending approval id", pending_id)?;
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let before = inner.pending.len();
        inner.pending.retain(|p| p.id != pending_id);
        if inner.pending.len() == before {
            return Err("no pending approval with that id".into());
        }
        Ok(())
    }

    /// Withdraw trust from an approved device.
    ///
    /// The thumbprint is remembered, not merely forgotten: a device that comes
    /// back with the same key must be re-approved deliberately rather than
    /// slipping through as if it were new.
    pub fn revoke(&self, thumbprint: &str) -> Result<(), String> {
        safe_id("endpoint key thumbprint", thumbprint)?;
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let idx = inner
            .roster
            .approved
            .iter()
            .position(|m| m.endpoint_key.thumbprint == thumbprint)
            .ok_or_else(|| "no approved device with that thumbprint".to_string())?;
        let gone = inner.roster.approved.remove(idx);
        inner.roster.revoked.push(RevokedMobile {
            device_id: gone.device_id,
            thumbprint: gone.endpoint_key.thumbprint,
            revoked_at: Utc::now(),
        });
        inner.roster.revision += 1;
        self.persist(&inner.roster);
        Ok(())
    }

    /// Mint a new desktop key, invalidating every existing pairing.
    ///
    /// Clearing the roster is the point, not a side effect: the approvals were
    /// statements about a key that no longer exists, and keeping them would
    /// leave devices listed as trusted that can no longer prove anything.
    pub fn rotate(&self) -> Result<IdentityStatus, String> {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let dir = self.dir();
            let (key, protection) = mint_key(&dir)?;
            inner.signing_key = Some(key);
            inner.protection = protection;
            inner.warning = None;
            inner.pending.clear();
            inner.roster.approved.clear();
            inner.roster.revoked.clear();
            inner.roster.revision += 1;
            self.persist(&inner.roster);
        }
        Ok(self.status())
    }

    fn persist(&self, roster: &PersistedRoster) {
        let dir = self.dir();
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let path = dir.join("roster.json");
        let tmp = path.with_extension("json.tmp");
        let mut out = roster.clone();
        out.schema_version = 1;
        let write = || -> std::io::Result<()> {
            let body = serde_json::to_string_pretty(&out).map_err(std::io::Error::other)?;
            std::fs::write(&tmp, body)?;
            std::fs::rename(&tmp, &path)
        };
        if let Err(e) = write() {
            log::warn!("cannot persist companion roster to {}: {e}", path.display());
        }
    }
}

/// The canonical thumbprint of a public key.
///
/// The exact JWK spelling — members in `crv`, `kty`, `x` order, no whitespace —
/// is what the mobile app hashes. It is reproduced literally rather than built
/// with a JSON serialiser, because a serialiser is free to reorder keys and
/// would silently produce a fingerprint no peer agrees with.
fn endpoint_thumbprint(public_key: &str) -> String {
    let canonical = format!(
        "{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":{}}}",
        serde_json::to_string(public_key).expect("string serialization cannot fail")
    );
    URL_SAFE_NO_PAD.encode(Sha256::digest(canonical.as_bytes()))
}

/// A stable hash of who is trusted right now, so a peer can notice its view is
/// stale without being told the roster itself.
pub fn peer_roster_hash(revision: u64, thumbprints: &[String]) -> String {
    let mut sorted: Vec<&str> = thumbprints.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(revision.to_string().as_bytes());
    for t in sorted {
        hasher.update(b"\n");
        hasher.update(t.as_bytes());
    }
    URL_SAFE_NO_PAD.encode(hasher.finalize())
}

/// Identifiers that reach a filename, a log line, or a protocol frame.
fn safe_id(field: &str, value: &str) -> Result<(), String> {
    if !value.is_empty()
        && value.len() <= MAX_ID_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        Ok(())
    } else {
        Err(format!("{field} is not a safe protocol identifier"))
    }
}

fn load_or_create_key(dir: &std::path::Path) -> Result<(SigningKey, KeyProtection), String> {
    let path = dir.join("endpoint.key");
    if let Ok(raw) = std::fs::read_to_string(&path) {
        let bytes = URL_SAFE_NO_PAD
            .decode(raw.trim())
            .map_err(|_| "stored endpoint key is not base64url".to_string())?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "stored endpoint key must be 32 bytes".to_string())?;
        return Ok((SigningKey::from_bytes(&bytes), KeyProtection::SoftwareFile));
    }
    mint_key(dir)
}

fn mint_key(dir: &std::path::Path) -> Result<(SigningKey, KeyProtection), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed).map_err(|_| "OS randomness is unavailable".to_string())?;
    let key = SigningKey::from_bytes(&seed);
    let path = dir.join("endpoint.key");
    std::fs::write(&path, URL_SAFE_NO_PAD.encode(seed))
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok((key, KeyProtection::SoftwareFile))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(PathBuf);
    impl Dir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("oaiy-companion-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn pending(id: &str, key: &EndpointPublicKey) -> PendingMobileApproval {
        PendingMobileApproval {
            id: id.into(),
            device_id: format!("dev-{id}"),
            display_name: "Test phone".into(),
            endpoint_key: key.clone(),
            received_at: Utc::now(),
        }
    }

    fn a_key(seed: u8) -> EndpointPublicKey {
        let sk = SigningKey::from_bytes(&[seed; 32]);
        EndpointPublicKey::from_verifying_key(&sk.verifying_key())
    }

    #[test]
    fn the_thumbprint_matches_the_canonical_jwk_the_peer_hashes() {
        // The mobile app derives this fingerprint independently. If the
        // canonical form drifts — member order, whitespace, base64 alphabet —
        // every signature we make is rejected by a phone we cannot patch, so
        // this pins the exact bytes rather than merely "some hash".
        let key = a_key(7);
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(
            format!(
                "{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"{}\"}}",
                key.public_key
            )
            .as_bytes(),
        ));
        assert_eq!(key.thumbprint, expected);
        // base64url, so it is safe in a URL path segment and a filename.
        assert!(!key.thumbprint.contains('+') && !key.thumbprint.contains('/'));
        assert!(!key.thumbprint.ends_with('='));
    }

    #[test]
    fn a_key_that_lies_about_its_own_thumbprint_is_refused() {
        // The thumbprint arrives over an untrusted relay. A peer that could
        // name its own would impersonate an approved device by presenting a
        // different key under a trusted fingerprint.
        let mut key = a_key(9);
        key.thumbprint = a_key(10).thumbprint;
        let err = key.verifying_key().unwrap_err();
        assert!(err.contains("does not match"), "{err}");
    }

    #[test]
    fn a_signature_verifies_only_against_the_key_that_made_it() {
        let sk = SigningKey::from_bytes(&[3u8; 32]);
        let pk = EndpointPublicKey::from_verifying_key(&sk.verifying_key());
        let msg = b"pairing challenge";
        let sig = URL_SAFE_NO_PAD.encode(
            ed25519_dalek::Signer::sign(&sk, msg).to_bytes(),
        );
        assert!(pk.verify(msg, &sig).is_ok());
        assert!(pk.verify(b"a different message", &sig).is_err());
        assert!(a_key(4).verify(msg, &sig).is_err());
    }

    #[test]
    fn an_identity_survives_a_restart_and_keeps_its_roster() {
        let dir = Dir::new("persist");
        let (thumb, key_pub) = {
            let id = EndpointIdentity::open(dir.0.clone(), "aokie");
            let k = a_key(21);
            id.inner.lock().unwrap().pending.push(pending("p1", &k));
            id.approve("p1").unwrap();
            let s = id.status();
            (
                s.approved_mobiles[0].endpoint_key.thumbprint.clone(),
                s.endpoint_key.unwrap().public_key,
            )
        };

        // A new process over the same root: the desktop must present the SAME
        // identity, or every phone that trusted it would have to re-pair.
        let again = EndpointIdentity::open(dir.0.clone(), "aokie");
        let s = again.status();
        assert_eq!(s.endpoint_key.unwrap().public_key, key_pub, "key must be stable");
        assert_eq!(s.approved_mobiles.len(), 1);
        assert_eq!(s.approved_mobiles[0].endpoint_key.thumbprint, thumb);
        assert!(s.remote_access_ready);
    }

    #[test]
    fn identities_are_per_plugin() {
        // OAIY is a generic host: revoking one broker plugin's companions must
        // not disturb another's, so they cannot share a key or a roster.
        let dir = Dir::new("scoped");
        let a = EndpointIdentity::open(dir.0.clone(), "aokie");
        let b = EndpointIdentity::open(dir.0.clone(), "other");
        assert_ne!(
            a.status().endpoint_key.unwrap().public_key,
            b.status().endpoint_key.unwrap().public_key
        );
    }

    #[test]
    fn rotating_invalidates_every_existing_pairing() {
        let dir = Dir::new("rotate");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        id.inner.lock().unwrap().pending.push(pending("p1", &a_key(31)));
        id.approve("p1").unwrap();
        let before = id.status();
        assert_eq!(before.approved_mobiles.len(), 1);

        let after = id.rotate().unwrap();
        // The approvals were statements about a key that no longer exists.
        assert!(after.approved_mobiles.is_empty(), "rotation must clear the roster");
        assert!(!after.remote_access_ready);
        assert_ne!(
            after.endpoint_key.unwrap().public_key,
            before.endpoint_key.unwrap().public_key
        );
        assert!(after.roster_revision > before.roster_revision);
    }

    #[test]
    fn re_approving_the_same_device_does_not_duplicate_it() {
        let dir = Dir::new("dup");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let k = a_key(41);
        id.inner.lock().unwrap().pending.push(pending("p1", &k));
        id.approve("p1").unwrap();
        id.inner.lock().unwrap().pending.push(pending("p2", &k));
        id.approve("p2").unwrap();
        // Otherwise the user has to revoke the same handset twice to be rid of it.
        assert_eq!(id.status().approved_mobiles.len(), 1);
    }

    #[test]
    fn revoking_records_the_thumbprint_rather_than_forgetting_it() {
        let dir = Dir::new("revoke");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let k = a_key(51);
        id.inner.lock().unwrap().pending.push(pending("p1", &k));
        id.approve("p1").unwrap();

        id.revoke(&k.thumbprint).unwrap();
        assert!(id.status().approved_mobiles.is_empty());
        assert!(!id.status().remote_access_ready);
        // Remembered, so the same key returning is a deliberate re-approval
        // rather than something that slips through looking new.
        assert_eq!(id.inner.lock().unwrap().roster.revoked.len(), 1);
        assert!(id.revoke(&k.thumbprint).is_err(), "revoking twice is an error");
    }

    #[test]
    fn the_roster_hash_tracks_membership_not_ordering() {
        // A peer compares this to decide whether its view is stale, so it must
        // change when trust changes and NOT when the list is merely reordered.
        let a = "AAA".to_string();
        let b = "BBB".to_string();
        assert_eq!(
            peer_roster_hash(3, &[a.clone(), b.clone()]),
            peer_roster_hash(3, &[b.clone(), a.clone()])
        );
        assert_ne!(peer_roster_hash(3, &[a.clone()]), peer_roster_hash(4, &[a.clone()]));
        assert_ne!(peer_roster_hash(3, &[a.clone()]), peer_roster_hash(3, &[b]));
    }

    #[test]
    fn unsafe_identifiers_are_refused_before_they_reach_a_path() {
        let dir = Dir::new("ids");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        for bad in ["", "../escape", "has space", "semi;colon", &"x".repeat(129)] {
            assert!(id.revoke(bad).is_err(), "{bad:?} should be refused");
            assert!(id.approve(bad).is_err(), "{bad:?} should be refused");
        }
    }
}
