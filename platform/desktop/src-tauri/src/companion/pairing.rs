//! The pairing ceremony: how a phone becomes a trusted companion endpoint.
//!
//! Deliberately a copy-paste ritual rather than an automatic discovery, because
//! the relay carrying these messages is untrusted. The desktop publishes an
//! offer (QR or JSON text) containing its public key and a one-use nonce; the
//! phone signs a response binding *its* key to *that* nonce and *this* desktop's
//! thumbprint; the desktop verifies the signature and then — crucially — asks a
//! human to compare a fingerprint shown on both screens before trusting it.
//!
//! That last step is what the cryptography cannot supply. Everything up to it
//! proves only that whoever answered holds the key they claim; it does not
//! prove they are the phone in your hand. An attacker who can see the offer can
//! answer it. The fingerprint comparison is the only part they cannot forge.
//!
//! # The wire format is dictated, not chosen
//!
//! `apps/aokie-mobile/src-tauri/src/desktop_pairing.rs` decodes these messages
//! with `deny_unknown_fields`, so an extra field is a hard rejection, and it
//! signs over a canonical JSON encoding with a domain-separation prefix. Both
//! are reproduced here exactly: object keys sorted, no whitespace, integers
//! only, `domain \0 canonical` as the signed message. A stray space or a
//! reordered key produces a signature the phone will not verify, and no amount
//! of correctness elsewhere recovers from that.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::identity::EndpointPublicKey;

/// Domain separation for the phone's signature.
///
/// Prefixing the signed bytes with a purpose string is what stops a signature
/// gathered for one ceremony being replayed as another message type. It must
/// match the phone byte for byte.
pub const PAIRING_RESPONSE_DOMAIN: &str = "aokie/v2/mobile-pairing-response";

/// How long an offer stays valid. Ten minutes: long enough to walk to the
/// phone, short enough that an offer left on screen is not a standing
/// invitation.
pub const PAIRING_TTL_SECONDS: u64 = 600;

pub const OFFER_KIND: &str = "aokie_mobile_pairing";
pub const RESPONSE_KIND: &str = "aokie_mobile_pairing_response";
pub const SCHEMA_VERSION: u16 = 2;

/// The offer the desktop publishes. Carries no bearer token — everything in it
/// is public, which is why it is safe to show as a QR code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairingPayload {
    pub kind: String,
    pub schema_version: u16,
    pub app_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    pub desktop_connection_id: String,
    pub desktop_endpoint_key: EndpointPublicKey,
    /// One-use. Binds a response to THIS offer; see [`super::identity`]'s
    /// challenge store, which forgets it once consumed.
    pub nonce: String,
    pub jti: String,
    pub issued_at: u64,
    pub expires_at: u64,
}

/// What the phone signs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MobilePairingClaims {
    pub app_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    pub desktop_connection_id: String,
    /// The desktop it is answering. Without this a response could be replayed
    /// against a DIFFERENT desktop that happened to issue the same nonce.
    pub desktop_key_thumbprint: String,
    pub device_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub mobile_endpoint_key: EndpointPublicKey,
    pub pairing_nonce: String,
    pub jti: String,
    pub issued_at: u64,
    pub expires_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MobilePairingResponse {
    pub kind: String,
    pub schema_version: u16,
    pub claims: MobilePairingClaims,
    pub signature: String,
}

impl MobilePairingResponse {
    /// Verify the signature over the claims, and nothing else.
    ///
    /// Deliberately narrow: this says "whoever sent this holds the key inside
    /// it". Binding to a live offer, expiry, and owner approval are separate
    /// checks precisely so none of them can be mistaken for one another.
    pub fn verify_signature(&self) -> Result<(), String> {
        if self.kind != RESPONSE_KIND || self.schema_version != SCHEMA_VERSION {
            return Err("not a v2 mobile pairing response".into());
        }
        let message = domain_separated_canonical(PAIRING_RESPONSE_DOMAIN, &self.claims)?;
        self.claims
            .mobile_endpoint_key
            .verify(&message, &self.signature)
    }
}

/// `domain \0 canonical(value)` — the exact bytes the phone signs.
pub fn domain_separated_canonical<T: Serialize>(domain: &str, value: &T) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value)
        .map_err(|_| "could not serialize mobile pairing claims".to_string())?;
    let mut bytes = domain.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&canonical_json_value(&value)?);
    Ok(bytes)
}

/// Deterministic JSON: keys sorted, no whitespace, integers only.
///
/// Floats are refused rather than encoded. Two implementations will not agree
/// on how to render `0.1`, and a signature over a number that round-trips
/// differently on the peer is a signature that fails for reasons nobody can
/// see — so the protocol simply forbids them.
pub fn canonical_json_value(value: &Value) -> Result<Vec<u8>, String> {
    fn write_value(value: &Value, output: &mut Vec<u8>) -> Result<(), String> {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(true) => output.extend_from_slice(b"true"),
            Value::Bool(false) => output.extend_from_slice(b"false"),
            Value::Number(number) => {
                if !number.is_u64() && !number.is_i64() {
                    return Err("signed pairing claims cannot contain floating-point numbers".into());
                }
                output.extend_from_slice(number.to_string().as_bytes());
            }
            Value::String(text) => output.extend_from_slice(
                serde_json::to_string(text)
                    .map_err(|_| "could not canonicalize pairing text".to_string())?
                    .as_bytes(),
            ),
            Value::Array(values) => {
                output.push(b'[');
                for (index, item) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    write_value(item, output)?;
                }
                output.push(b']');
            }
            Value::Object(map) => {
                output.push(b'{');
                let mut keys = map.keys().collect::<Vec<_>>();
                keys.sort();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        output.push(b',');
                    }
                    write_value(&Value::String(key.clone()), output)?;
                    output.push(b':');
                    write_value(&map[key], output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    write_value(value, &mut out)?;
    Ok(out)
}

/// A URL-safe random identifier of `bytes` entropy.
pub fn random_id(bytes: usize) -> Result<String, String> {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).map_err(|_| "OS randomness is unavailable".to_string())?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;

    fn key(seed: u8) -> (SigningKey, EndpointPublicKey) {
        let sk = SigningKey::from_bytes(&[seed; 32]);
        let pk: EndpointPublicKey =
            serde_json::from_value(json!({
                "algorithm": "ed25519",
                "publicKey": URL_SAFE_NO_PAD.encode(sk.verifying_key().to_bytes()),
                "thumbprint": super::super::identity::thumbprint_for(
                    &URL_SAFE_NO_PAD.encode(sk.verifying_key().to_bytes())
                ),
            }))
            .unwrap();
        (sk, pk)
    }

    fn claims(pk: &EndpointPublicKey, nonce: &str) -> MobilePairingClaims {
        MobilePairingClaims {
            app_id: "aokie".into(),
            workspace_id: None,
            desktop_connection_id: "conn-1".into(),
            desktop_key_thumbprint: "desktop-thumb".into(),
            device_id: "phone-1".into(),
            display_name: Some("Test phone".into()),
            mobile_endpoint_key: pk.clone(),
            pairing_nonce: nonce.into(),
            jti: "pair-1".into(),
            issued_at: 1_000,
            expires_at: 1_600,
        }
    }

    fn sign(sk: &SigningKey, c: &MobilePairingClaims) -> MobilePairingResponse {
        let msg = domain_separated_canonical(PAIRING_RESPONSE_DOMAIN, c).unwrap();
        MobilePairingResponse {
            kind: RESPONSE_KIND.into(),
            schema_version: SCHEMA_VERSION,
            claims: c.clone(),
            signature: URL_SAFE_NO_PAD.encode(sk.sign(&msg).to_bytes()),
        }
    }

    #[test]
    fn canonical_json_sorts_keys_and_omits_whitespace() {
        // The phone signs over these exact bytes. A reordered key or a stray
        // space is a signature it will not verify, with nothing on either side
        // able to say why.
        let v = json!({"b": 1, "a": {"z": [1, 2], "y": "x"}});
        let out = String::from_utf8(canonical_json_value(&v).unwrap()).unwrap();
        assert_eq!(out, r#"{"a":{"y":"x","z":[1,2]},"b":1}"#);
    }

    #[test]
    fn canonical_json_refuses_floats() {
        // Two implementations will not agree on how to render 0.1, so the
        // protocol forbids it rather than producing signatures that fail
        // mysteriously on one peer.
        let err = canonical_json_value(&json!({"n": 0.5})).unwrap_err();
        assert!(err.contains("floating-point"), "{err}");
    }

    #[test]
    fn the_signed_message_is_domain_prefixed_and_nul_separated() {
        // Domain separation is what stops a signature gathered for one purpose
        // being replayed as a different message type.
        let bytes = domain_separated_canonical("d/v2/x", &json!({"a": 1})).unwrap();
        assert!(bytes.starts_with(b"d/v2/x"));
        assert_eq!(bytes[6], 0, "a NUL must separate domain from payload");
        assert_eq!(&bytes[7..], br#"{"a":1}"#);
    }

    #[test]
    fn a_genuine_response_verifies() {
        let (sk, pk) = key(11);
        let r = sign(&sk, &claims(&pk, "nonce-1"));
        assert!(r.verify_signature().is_ok());
    }

    #[test]
    fn tampering_with_any_claim_breaks_the_signature() {
        // The whole point of signing the claims rather than just the key: a
        // relay that could edit device_id or the nonce could bind an approved
        // key to a different device or replay it against another offer.
        let (sk, pk) = key(12);
        let good = claims(&pk, "nonce-1");
        for mutate in [
            (|c: &mut MobilePairingClaims| c.device_id = "someone-else".into()) as fn(&mut MobilePairingClaims),
            |c: &mut MobilePairingClaims| c.pairing_nonce = "another-nonce".into(),
            |c: &mut MobilePairingClaims| c.desktop_key_thumbprint = "another-desktop".into(),
            |c: &mut MobilePairingClaims| c.expires_at = u64::MAX,
        ] {
            let mut r = sign(&sk, &good);
            mutate(&mut r.claims);
            assert!(r.verify_signature().is_err(), "a tampered claim must not verify");
        }
    }

    #[test]
    fn a_response_signed_by_a_different_key_is_refused() {
        // Substituting the key inside the claims does not help an attacker:
        // the signature is checked against the key the claims carry, so a
        // mismatch fails rather than authenticating the wrong device.
        let (sk_a, pk_a) = key(13);
        let (_, pk_b) = key(14);
        let mut r = sign(&sk_a, &claims(&pk_a, "n"));
        r.claims.mobile_endpoint_key = pk_b;
        assert!(r.verify_signature().is_err());
    }

    #[test]
    fn a_response_of_the_wrong_kind_or_version_is_refused_before_any_crypto() {
        let (sk, pk) = key(15);
        let mut r = sign(&sk, &claims(&pk, "n"));
        r.kind = "something_else".into();
        assert!(r.verify_signature().is_err());

        let mut r = sign(&sk, &claims(&pk, "n"));
        r.schema_version = 1;
        assert!(r.verify_signature().is_err());
    }

    #[test]
    fn the_signed_bytes_match_the_peers_fixture_exactly() {
        // A known-answer test built from the mobile app's OWN test fixture
        // (apps/aokie-mobile/src-tauri/src/desktop_pairing.rs: desktop key
        // [9;32], mobile key [7;32], now = 1_700_000_000). Both sides sign
        // these bytes, so pinning them literally is the only way a drift in
        // this encoder shows up here rather than as a phone that silently
        // refuses to pair.
        let (_, pk) = key(7);
        let c = MobilePairingClaims {
            app_id: "app_a".into(),
            workspace_id: Some("workspace_a".into()),
            desktop_connection_id: "desktop_connection_a".into(),
            desktop_key_thumbprint: "desktop_thumb_a".into(),
            device_id: "mobile_device_a".into(),
            display_name: None,
            mobile_endpoint_key: pk.clone(),
            pairing_nonce: "pairing_nonce_a".into(),
            jti: "pairing_jti_a".into(),
            issued_at: 1_700_000_000,
            expires_at: 1_700_000_300,
        };

        let bytes = domain_separated_canonical(PAIRING_RESPONSE_DOMAIN, &c).unwrap();
        let text = String::from_utf8(bytes[PAIRING_RESPONSE_DOMAIN.len() + 1..].to_vec()).unwrap();

        // Keys sorted; `displayName` absent entirely rather than null, because
        // the claims skip_serializing_if it — a null would change the bytes.
        let expected = format!(
            concat!(
                r#"{{"appId":"app_a","desktopConnectionId":"desktop_connection_a","#,
                r#""desktopKeyThumbprint":"desktop_thumb_a","deviceId":"mobile_device_a","#,
                r#""expiresAt":1700000300,"issuedAt":1700000000,"jti":"pairing_jti_a","#,
                r#""mobileEndpointKey":{{"algorithm":"ed25519","publicKey":"{pk}","thumbprint":"{tp}"}},"#,
                r#""pairingNonce":"pairing_nonce_a","workspaceId":"workspace_a"}}"#
            ),
            pk = pk.public_key,
            tp = pk.thumbprint,
        );
        assert_eq!(text, expected);
        assert!(!text.contains(' '), "canonical JSON carries no whitespace");
        assert!(!text.contains("displayName"), "an absent option must not appear at all");
    }

    #[test]
    fn random_ids_are_url_safe_and_do_not_repeat() {
        let a = random_id(32).unwrap();
        let b = random_id(32).unwrap();
        assert_ne!(a, b);
        // Goes into a QR payload and a JSON field; must not need escaping.
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }
}
