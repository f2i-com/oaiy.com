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
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::pairing::{
    domain_separated_canonical, random_id, MobilePairingResponse, PairingPayload, OFFER_KIND,
    PAIRING_RESPONSE_DOMAIN, PAIRING_TTL_SECONDS, RESPONSE_KIND, SCHEMA_VERSION,
};

/// Longest acceptable protocol identifier. Bounded so a hostile relay cannot
/// push an unbounded string into a filename or a log line.
const MAX_ID_BYTES: usize = 128;

/// Tolerance for the phone's clock disagreeing with ours.
///
/// Small on purpose. It exists so a handset a few seconds out does not fail to
/// pair; widening it widens the window in which a captured response is still
/// replayable.
const CLOCK_SKEW_SECONDS: u64 = 5;

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

/// A published pairing offer: the payload, the text to paste, and a QR of it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingOffer {
    pub request_id: String,
    pub payload: PairingPayload,
    /// The exact JSON the operator pastes into the phone. Kept alongside the QR
    /// because a camera is not always an option.
    pub encoded_payload: String,
    /// Raw SVG, not a data URI — the UI wraps it itself.
    pub qr_svg: String,
}

/// A mobile that proved possession of its key and now awaits the owner's word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PendingMobileApproval {
    pub id: String,
    pub device_id: String,
    pub display_name: String,
    pub endpoint_key: EndpointPublicKey,
    pub thumbprint: String,
    /// The colon-grouped uppercase hex digest the OPERATOR compares against the
    /// one shown on the phone before approving.
    ///
    /// A second rendering of the same key, and the whole security of the
    /// ceremony rests on a human reading it: the relay is untrusted, so the
    /// only thing distinguishing the intended phone from an attacker's is that
    /// these two strings match. Hence the grouping — `AB:CD:EF…` is checkable
    /// by eye in a way an unbroken 64-character run is not.
    pub fingerprint: String,
    pub received_at: DateTime<Utc>,
}

struct Inner {
    signing_key: Option<SigningKey>,
    protection: KeyProtection,
    roster: PersistedRoster,
    pending: Vec<PendingMobileApproval>,
    warning: Option<String>,
    /// Live offers, by nonce. In memory only: an offer that did not survive a
    /// restart SHOULD be dead, because its whole purpose is to be answered
    /// within ten minutes by someone standing at the machine.
    challenges: HashMap<String, PairingPayload>,
    /// `(nonce, jti)` pairs already answered. Kept after the challenge is
    /// removed so a replayed response is refused as "already used" rather than
    /// as "unknown", which is the honest reason and the one that tells an
    /// operator they are looking at a duplicate rather than a typo.
    consumed: HashSet<(String, String)>,
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
                challenges: HashMap::new(),
                consumed: HashSet::new(),
            }),
        })
    }

    /// The identity-only bootstrap handed to the broker plugin at init.
    ///
    /// This carries the desktop's PRIVATE seed. That is the protocol's design,
    /// not an oversight: the plugin hosts the WebRTC endpoint, so it is the
    /// process that has to sign as this desktop. What stays here is
    /// ADMINISTRATION — who is approved, and the ability to revoke them — which
    /// is why the pairing routes are served by the host and not by the plugin.
    ///
    /// `None` when there is nothing worth sending: no key, or no approved
    /// device. An identity-only bootstrap with an empty roster is rejected by
    /// the decoder (it requires 1..64 keys), so sending one would only turn a
    /// not-yet-paired desktop into a plugin that fails to start.
    ///
    /// Identity-only means NO admission fields — no gateway URL, token or ICE.
    /// The plugin asks for those over `companion.admission` when it needs them,
    /// so a short-lived credential is never baked into a handshake that happens
    /// once at launch.
    pub fn private_bootstrap(&self, plugin_api_version: u16) -> Option<serde_json::Value> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let key = inner.signing_key.as_ref()?;
        if inner.roster.approved.is_empty() || inner.roster.approved.len() > 64 {
            return None;
        }
        let public = EndpointPublicKey::from_verifying_key(&key.verifying_key());
        let thumbprints: Vec<String> = inner
            .roster
            .approved
            .iter()
            .map(|m| m.endpoint_key.thumbprint.clone())
            .collect();
        let keys: Vec<&EndpointPublicKey> = inner
            .roster
            .approved
            .iter()
            .map(|m| &m.endpoint_key)
            .collect();
        let _ = plugin_api_version;
        Some(serde_json::json!({
            "schemaVersion": SCHEMA_VERSION,
            "pluginId": self.owner_plugin,
            "endpointIdentity": {
                "algorithm": public.algorithm,
                "publicKey": public.public_key,
                "thumbprint": public.thumbprint,
                "privateKeySeed": URL_SAFE_NO_PAD.encode(key.to_bytes()),
            },
            "approvedMobileRoster": {
                "revision": inner.roster.revision,
                "rosterHash": peer_roster_hash(inner.roster.revision, &thumbprints),
                "keys": keys,
            },
        }))
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

    /// Publish an offer for a phone to answer.
    pub fn create_offer(
        &self,
        app_id: &str,
        workspace_id: Option<&str>,
        desktop_connection_id: &str,
    ) -> Result<PairingOffer, String> {
        self.create_offer_at(
            app_id,
            workspace_id,
            desktop_connection_id,
            Utc::now().timestamp().max(1) as u64,
        )
    }

    fn create_offer_at(
        &self,
        app_id: &str,
        workspace_id: Option<&str>,
        desktop_connection_id: &str,
        now: u64,
    ) -> Result<PairingOffer, String> {
        safe_id("app id", app_id)?;
        safe_id("desktop connection id", desktop_connection_id)?;
        if let Some(w) = workspace_id {
            safe_id("workspace id", w)?;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        purge_expired(&mut inner, now);
        let key = inner
            .signing_key
            .as_ref()
            .ok_or_else(|| "the desktop endpoint identity is unavailable".to_string())?;

        let nonce = random_id(32)?;
        let jti = format!("pair-{}", uuid::Uuid::new_v4().simple());
        let payload = PairingPayload {
            kind: OFFER_KIND.into(),
            schema_version: SCHEMA_VERSION,
            app_id: app_id.into(),
            workspace_id: workspace_id.map(str::to_string),
            desktop_connection_id: desktop_connection_id.into(),
            desktop_endpoint_key: EndpointPublicKey::from_verifying_key(&key.verifying_key()),
            nonce: nonce.clone(),
            jti: jti.clone(),
            issued_at: now,
            expires_at: now.saturating_add(PAIRING_TTL_SECONDS),
        };

        let encoded_payload = serde_json::to_string(&payload)
            .map_err(|e| format!("could not encode the pairing payload: {e}"))?;
        let qr_svg = qrcode::QrCode::new(encoded_payload.as_bytes())
            .map_err(|_| "the pairing payload is too large for a QR code".to_string())?
            .render::<qrcode::render::svg::Color>()
            .min_dimensions(280, 280)
            .dark_color(qrcode::render::svg::Color("#111827"))
            .light_color(qrcode::render::svg::Color("#ffffff"))
            .build();

        inner.challenges.insert(nonce, payload.clone());
        Ok(PairingOffer {
            request_id: jti,
            payload,
            encoded_payload,
            qr_svg,
        })
    }

    /// Accept a phone's answer to a live offer.
    ///
    /// Everything here establishes that the answer is cryptographically sound
    /// and bound to an offer WE issued. It deliberately stops short of trust:
    /// the result is a pending item the owner must approve after comparing a
    /// fingerprint, because nothing checked below can distinguish the intended
    /// phone from anyone else who saw the offer.
    pub fn receive_response(
        &self,
        response: &MobilePairingResponse,
    ) -> Result<PendingMobileApproval, String> {
        self.receive_response_at(response, Utc::now().timestamp().max(1) as u64)
    }

    fn receive_response_at(
        &self,
        response: &MobilePairingResponse,
        now: u64,
    ) -> Result<PendingMobileApproval, String> {
        if response.kind != RESPONSE_KIND || response.schema_version != SCHEMA_VERSION {
            return Err("unsupported mobile pairing response".into());
        }
        let claims = &response.claims;
        safe_id("app id", &claims.app_id)?;
        safe_id("desktop connection id", &claims.desktop_connection_id)?;
        safe_id("desktop key thumbprint", &claims.desktop_key_thumbprint)?;
        safe_id("mobile device id", &claims.device_id)?;
        safe_id("pairing nonce", &claims.pairing_nonce)?;
        safe_id("pairing jti", &claims.jti)?;
        if let Some(w) = claims.workspace_id.as_deref() {
            safe_id("workspace id", w)?;
        }

        // Purge BEFORE the lifetime check, not after it. An expired response
        // returns early, so leaving the sweep downstream meant a machine that
        // published one offer and never paired kept that challenge in memory
        // for the life of the process — the one case where nothing else would
        // come along to clear it.
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        purge_expired(&mut inner, now);

        // Lifetime next, so an expired response still costs no signature check.
        if claims.issued_at > now.saturating_add(CLOCK_SKEW_SECONDS)
            || claims.expires_at <= claims.issued_at
            || claims.expires_at.saturating_add(CLOCK_SKEW_SECONDS) < now
        {
            return Err("the mobile pairing response is expired or has an invalid lifetime".into());
        }

        if inner
            .consumed
            .contains(&(claims.pairing_nonce.clone(), claims.jti.clone()))
        {
            return Err("that pairing challenge was already used".into());
        }
        let expected = inner
            .challenges
            .get(&claims.pairing_nonce)
            .cloned()
            .ok_or_else(|| "no live pairing offer matches that response".to_string())?;

        if claims.jti != expected.jti {
            return Err("the pairing response JTI does not match the one-use request".into());
        }
        if claims.app_id != expected.app_id || claims.workspace_id != expected.workspace_id {
            return Err("the pairing response is for a different app or workspace".into());
        }
        // Binding to OUR key, not merely to a nonce: without it a response
        // could be replayed against another desktop that issued the same nonce.
        if claims.desktop_connection_id != expected.desktop_connection_id
            || claims.desktop_key_thumbprint != expected.desktop_endpoint_key.thumbprint
        {
            return Err("the pairing response is for a different desktop endpoint".into());
        }
        if claims.issued_at.saturating_add(CLOCK_SKEW_SECONDS) < expected.issued_at
            || claims.issued_at > expected.expires_at.saturating_add(CLOCK_SKEW_SECONDS)
        {
            return Err("the pairing response is outside the desktop challenge window".into());
        }

        // A key we previously revoked must not walk back in through a fresh
        // ceremony without the owner noticing it is the same device.
        if inner
            .roster
            .revoked
            .iter()
            .any(|r| r.thumbprint == claims.mobile_endpoint_key.thumbprint)
        {
            return Err("that device's key was revoked on this machine".into());
        }

        // Signature LAST: the cheap structural checks above shed hostile input
        // before spending a verification on it.
        let message = domain_separated_canonical(PAIRING_RESPONSE_DOMAIN, claims)?;
        claims
            .mobile_endpoint_key
            .verify(&message, &response.signature)?;

        let display_name = claims
            .display_name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| claims.device_id.clone());
        let pending = PendingMobileApproval {
            id: format!("approval-{}", uuid::Uuid::new_v4().simple()),
            device_id: claims.device_id.clone(),
            display_name,
            thumbprint: claims.mobile_endpoint_key.thumbprint.clone(),
            fingerprint: display_fingerprint(&claims.mobile_endpoint_key)?,
            endpoint_key: claims.mobile_endpoint_key.clone(),
            received_at: Utc::now(),
        };

        // One answer per offer: burn the challenge and remember the pair, so a
        // second copy of the same response is refused as a replay.
        inner.challenges.remove(&claims.pairing_nonce);
        inner
            .consumed
            .insert((claims.pairing_nonce.clone(), claims.jti.clone()));
        inner.pending.push(pending.clone());
        Ok(pending)
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

/// The human-comparable rendering of a key: uppercase hex, colon-grouped in
/// pairs of bytes.
///
/// Distinct from the thumbprint, which is a base64url hash used by machines.
/// This one exists to be read aloud or compared on sight against a phone
/// screen, which is the only step in the pairing ceremony an attacker on the
/// relay cannot forge.
pub fn display_fingerprint(key: &EndpointPublicKey) -> Result<String, String> {
    use std::fmt::Write as _;
    let bytes = key.verifying_key()?.to_bytes();
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64 + 15);
    for (index, byte) in digest.iter().enumerate() {
        if index > 0 && index % 2 == 0 {
            encoded.push(':');
        }
        let _ = write!(encoded, "{byte:02X}");
    }
    Ok(encoded)
}

/// The canonical thumbprint of a public key.
///
/// The exact JWK spelling — members in `crv`, `kty`, `x` order, no whitespace —
/// is what the mobile app hashes. It is reproduced literally rather than built
/// with a JSON serialiser, because a serialiser is free to reorder keys and
/// would silently produce a fingerprint no peer agrees with.
pub fn thumbprint_for(public_key: &str) -> String {
    endpoint_thumbprint(public_key)
}

fn endpoint_thumbprint(public_key: &str) -> String {
    let canonical = format!(
        "{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":{}}}",
        serde_json::to_string(public_key).expect("string serialization cannot fail")
    );
    URL_SAFE_NO_PAD.encode(Sha256::digest(canonical.as_bytes()))
}

/// A stable hash of who is trusted right now, so a peer can notice its view is
/// stale without being told the roster itself.
///
/// The derivation is fixed by the v2 protocol, NOT free for this host to
/// choose: the plugin recomputes it from the roster it was handed and refuses a
/// bootstrap whose hash disagrees, and the gateway signs the same value into an
/// admission. Domain tag, canonical JSON, sorted thumbprints — all load-bearing.
pub fn peer_roster_hash(revision: u64, thumbprints: &[String]) -> String {
    let mut sorted = thumbprints.to_vec();
    sorted.sort();
    let payload = serde_json::json!({
        "approvedPeerKeyThumbprints": sorted,
        "peerRosterRevision": revision,
    });
    let canonical = super::pairing::canonical_json_value(&payload)
        .expect("a roster of strings and one integer is always canonicalizable");
    let mut message = b"aokie/v2/peer-roster\0".to_vec();
    message.extend_from_slice(&canonical);
    URL_SAFE_NO_PAD.encode(Sha256::digest(message))
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

/// Drop offers whose window has closed.
fn purge_expired(inner: &mut Inner, now: u64) {
    inner
        .challenges
        .retain(|_, p| p.expires_at.saturating_add(CLOCK_SKEW_SECONDS) >= now);
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
    use super::super::pairing::{
        domain_separated_canonical, MobilePairingResponse, PAIRING_RESPONSE_DOMAIN, RESPONSE_KIND,
        SCHEMA_VERSION,
    };
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
            thumbprint: key.thumbprint.clone(),
            fingerprint: display_fingerprint(key).unwrap(),
            received_at: Utc::now(),
        }
    }

    fn a_key(seed: u8) -> EndpointPublicKey {
        let sk = SigningKey::from_bytes(&[seed; 32]);
        EndpointPublicKey::from_verifying_key(&sk.verifying_key())
    }

    /// Answer an offer the way the phone does, so the tests exercise the real
    /// verification path rather than a hand-built approximation.
    fn answer(offer: &PairingOffer, sk: &SigningKey) -> MobilePairingResponse {
        use ed25519_dalek::Signer;
        let pk = EndpointPublicKey::from_verifying_key(&sk.verifying_key());
        let claims = super::super::pairing::MobilePairingClaims {
            app_id: offer.payload.app_id.clone(),
            workspace_id: offer.payload.workspace_id.clone(),
            desktop_connection_id: offer.payload.desktop_connection_id.clone(),
            desktop_key_thumbprint: offer.payload.desktop_endpoint_key.thumbprint.clone(),
            device_id: "phone-1".into(),
            display_name: Some("Lance's phone".into()),
            mobile_endpoint_key: pk,
            pairing_nonce: offer.payload.nonce.clone(),
            jti: offer.payload.jti.clone(),
            issued_at: offer.payload.issued_at,
            expires_at: offer.payload.expires_at,
        };
        let msg = domain_separated_canonical(PAIRING_RESPONSE_DOMAIN, &claims).unwrap();
        MobilePairingResponse {
            kind: RESPONSE_KIND.into(),
            schema_version: SCHEMA_VERSION,
            claims,
            signature: URL_SAFE_NO_PAD.encode(sk.sign(&msg).to_bytes()),
        }
    }

    #[test]
    fn an_offer_carries_no_secret_and_expires() {
        // It is shown as a QR on screen, so everything in it is public by
        // construction — that is the property that makes displaying it safe.
        let dir = Dir::new("offer");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer("app_a", None, "conn-1").unwrap();

        assert_eq!(offer.payload.kind, "aokie_mobile_pairing");
        assert_eq!(offer.payload.schema_version, 2);
        assert_eq!(
            offer.payload.expires_at - offer.payload.issued_at,
            600,
            "ten minutes: long enough to reach the phone, short enough that an offer left on screen is not a standing invitation"
        );
        // The desktop's PUBLIC key, and nothing resembling the private one.
        let encoded = &offer.encoded_payload;
        assert!(encoded.contains(&offer.payload.desktop_endpoint_key.public_key));
        let seed = std::fs::read_to_string(dir.0.join("companion/aokie/endpoint.key")).unwrap();
        assert!(!encoded.contains(seed.trim()), "an offer must never carry the private seed");
        assert!(offer.qr_svg.starts_with("<?xml") || offer.qr_svg.contains("<svg"));
    }

    #[test]
    fn a_genuine_answer_becomes_a_pending_approval_not_a_trusted_device() {
        // The signature proves possession of a key. It does NOT prove the
        // answering device is the phone in your hand, so the ceremony must stop
        // at pending and wait for a human.
        let dir = Dir::new("answer");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer("app_a", None, "conn-1").unwrap();
        let phone = SigningKey::from_bytes(&[7u8; 32]);

        let pending = id.receive_response(&answer(&offer, &phone)).unwrap();
        assert_eq!(pending.device_id, "phone-1");
        assert_eq!(pending.display_name, "Lance's phone");
        // The operator compares THIS against the phone's screen.
        assert!(pending.fingerprint.contains(':'), "{}", pending.fingerprint);

        let status = id.status();
        assert_eq!(status.pending_approvals.len(), 1);
        assert!(status.approved_mobiles.is_empty(), "signing alone must not confer trust");
        assert!(!status.remote_access_ready);
    }

    #[test]
    fn the_same_answer_cannot_be_replayed() {
        // A relay that captured a valid response could otherwise enrol the same
        // device repeatedly, or race a second approval past the operator.
        let dir = Dir::new("replay");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer("app_a", None, "conn-1").unwrap();
        let phone = SigningKey::from_bytes(&[8u8; 32]);
        let response = answer(&offer, &phone);

        assert!(id.receive_response(&response).is_ok());
        let err = id.receive_response(&response).unwrap_err();
        // "already used" rather than "unknown": the honest reason, and the one
        // that tells an operator they are looking at a duplicate not a typo.
        assert!(err.contains("already used"), "{err}");
    }

    #[test]
    fn an_answer_to_another_desktops_offer_is_refused() {
        // The claims bind to OUR thumbprint. Without that a response could be
        // replayed against a different desktop that issued the same nonce.
        let dir = Dir::new("crossdesktop");
        let mine = EndpointIdentity::open(dir.0.join("a"), "aokie");
        let theirs = EndpointIdentity::open(dir.0.join("b"), "aokie");
        let phone = SigningKey::from_bytes(&[9u8; 32]);

        let their_offer = theirs.create_offer("app_a", None, "conn-1").unwrap();
        let err = mine.receive_response(&answer(&their_offer, &phone)).unwrap_err();
        assert!(err.contains("no live pairing offer"), "{err}");
    }

    #[test]
    fn an_expired_offer_is_refused_and_forgotten() {
        let dir = Dir::new("expiry");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer_at("app_a", None, "conn-1", 1_000).unwrap();
        let phone = SigningKey::from_bytes(&[10u8; 32]);
        let response = answer(&offer, &phone);

        // Comfortably past the ten-minute window.
        let err = id.receive_response_at(&response, 1_000 + 600 + 60).unwrap_err();
        assert!(err.contains("expired"), "{err}");
        // And the challenge is gone rather than lingering to be answered later.
        assert!(id.inner.lock().unwrap().challenges.is_empty());
    }

    #[test]
    fn a_revoked_key_cannot_quietly_re_enrol() {
        // Otherwise revoking a lost phone buys nothing: it re-pairs and the
        // owner sees an ordinary approval rather than the return of a device
        // they deliberately cut off.
        let dir = Dir::new("revoked-return");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let phone = SigningKey::from_bytes(&[11u8; 32]);

        let first = id.create_offer("app_a", None, "conn-1").unwrap();
        let pending = id.receive_response(&answer(&first, &phone)).unwrap();
        id.approve(&pending.id).unwrap();
        id.revoke(&pending.thumbprint).unwrap();

        let second = id.create_offer("app_a", None, "conn-1").unwrap();
        let err = id.receive_response(&answer(&second, &phone)).unwrap_err();
        assert!(err.contains("revoked"), "{err}");
    }

    #[test]
    fn a_tampered_answer_is_refused() {
        let dir = Dir::new("tamper");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer("app_a", None, "conn-1").unwrap();
        let phone = SigningKey::from_bytes(&[12u8; 32]);

        let mut r = answer(&offer, &phone);
        r.claims.device_id = "someone-elses-phone".into();
        assert!(id.receive_response(&r).is_err());

        // …and the failed attempt must not have burned the challenge, or a
        // hostile frame would deny the real phone its one chance to answer.
        assert!(id.receive_response(&answer(&offer, &phone)).is_ok());
    }

    /// Emit a real bootstrap for the PLUGIN's decoder to verify.
    ///
    /// Ignored by default because it writes a file and proves nothing on its
    /// own. Run it, then parse the output with
    /// `aokie_plugin::companion_gateway::CompanionBootstrap::parse` — that
    /// decoder recomputes the roster hash and refuses a mismatch, which is how
    /// a divergence in `peer_roster_hash` is caught end to end rather than only
    /// against the vectors above.
    ///
    ///   cargo test --lib dump_bootstrap_for_cross_check -- --ignored
    #[test]
    #[ignore]
    fn dump_bootstrap_for_cross_check() {
        let dir = Dir::new("bootstrap-dump");
        let id = EndpointIdentity::open(dir.0.clone(), "aokie");
        let offer = id.create_offer("app_a", None, "conn-1").unwrap();
        let phone = SigningKey::from_bytes(&[21u8; 32]);
        let pending = id.receive_response(&answer(&offer, &phone)).unwrap();
        id.approve(&pending.id).unwrap();
        let b = id.private_bootstrap(1).expect("a bootstrap once a device is approved");
        std::fs::write(
            std::env::temp_dir().join("oaiy-bootstrap.json"),
            serde_json::to_string_pretty(&b).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn the_roster_hash_matches_the_v2_protocol_byte_for_byte() {
        // Vectors produced by aokie-protocol's own peer_roster_hash. This is a
        // CROSS-IMPLEMENTATION check, not a change detector: the plugin refuses
        // a bootstrap whose hash it cannot reproduce, and the gateway signs the
        // same value into an admission, so an OAIY-only derivation would fail
        // at connect time with nothing pointing back here.
        //
        // An earlier version of this function hashed the revision and
        // thumbprints newline-separated, and would have done exactly that.
        for (revision, thumbprints, expected) in [
            (0u64, vec![], "fGb5DK3ZHkoj_GOqfYsq1yCFhJPoqcSpF1j7nKb5jT8"),
            (1, vec!["alpha"], "xyefRCrz0zy72_7NNrosaWgUxSvsHXaW8To0MF3L2LI"),
            (7, vec!["b", "a"], "aIhQyqG1JxQPkllVN85_Gsc2dcUS72i3ZcCxk4dttNg"),
            (
                42,
                vec!["zzz", "aaa", "mmm"],
                "NTCkF0ojDLAJR38yF9DSpGTE4DYpohvraTsEmKsl89Y",
            ),
        ] {
            let owned: Vec<String> = thumbprints.iter().map(|t| t.to_string()).collect();
            assert_eq!(peer_roster_hash(revision, &owned), expected, "revision {revision}");
        }
    }

    #[test]
    fn the_roster_hash_ignores_the_order_it_is_given() {
        // The roster is a SET. Two desktops holding the same devices must agree,
        // whatever order their storage happened to yield.
        let a: Vec<String> = ["m", "a", "z"].iter().map(|s| s.to_string()).collect();
        let b: Vec<String> = ["z", "m", "a"].iter().map(|s| s.to_string()).collect();
        assert_eq!(peer_roster_hash(3, &a), peer_roster_hash(3, &b));
        // …but the revision is part of it, or a stale view would look current.
        assert_ne!(peer_roster_hash(3, &a), peer_roster_hash(4, &a));
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
