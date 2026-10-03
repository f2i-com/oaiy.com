//! The phone's pairing response (README 10.1, `pairing-response.schema.json`).
//!
//! `{"kind":"aokie_mobile_pairing_response","schemaVersion":3,"claims":{...},"signature":<b64u>,"mac":<b64u>}`. The claims are in canonical form for signing: the phone signs
//! `"oaiy/pairing/3/response" || 0x00 || canonical(claims)` with its endpoint key (proof it holds the key) and MACs
//! `"oaiy/pairing/3/response-mac" || 0x00 || canonical(claims)` under the pairing MAC key (proof it knows the secret). The desktop **re-canonicalises what it parsed**
//! (the one sanctioned re-serialisation of the protocol: the claims arrive as a member of a larger document) and checks, in the order the README gives: shape and
//! identifiers, the lifetime against relay time with 30 seconds of slack, the binding to the offer, the key of the phone (not of small order), the MAC in constant time,
//! and the signature last, strictly.

use oaiy_crypto::zeroize::Secret;

use crate::b64;
use crate::error::{Error, Result};
use crate::ids;
use crate::json::{self, quote, Json};
use crate::keys::{signature_from_b64u, SignDomain, Signer, VerifyKey, X25519Public};
use crate::pairing::math::{response_mac, verify_response_mac};
use crate::pairing::offer::{read_endpoint_key, Offer, SKEW_S};

/// A response's claims live at most 120 seconds (`expiresAt - issuedAt`).
pub const CLAIMS_LIFE_MAX_S: u64 = 120;

/// The claims a phone signs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claims {
    /// `appId`: the offer's.
    pub app_id: String,
    /// `desktopConnectionId`: the offer's.
    pub desktop_connection_id: String,
    /// `desktopKeyThumbprint`: the thumbprint of the offer's `desktopEndpointKey`.
    pub desktop_key_thumbprint: String,
    /// `deviceId`: an id the phone chooses for itself (`dev-` and 22 characters).
    pub device_id: String,
    /// `displayName`: at most 60 characters, optional.
    pub display_name: Option<String>,
    /// `mobileEndpointKey`: the phone's Ed25519 key.
    pub mobile_endpoint: VerifyKey,
    /// `mobileX25519`: the key the token is sealed to.
    pub mobile_x25519: X25519Public,
    /// `pairingNonce`: the offer's nonce, 32 raw bytes.
    pub pairing_nonce: [u8; 32],
    /// `jti`: the offer's.
    pub jti: String,
    /// `issuedAt`, relay-corrected time.
    pub issued_at: u64,
    /// `expiresAt`: at most `issuedAt + 120`.
    pub expires_at: u64,
}

impl Claims {
    fn to_json(&self) -> Json {
        let mut members = vec![
            ("appId", Json::str(self.app_id.clone())),
            ("desktopConnectionId", Json::str(self.desktop_connection_id.clone())),
            ("desktopKeyThumbprint", Json::str(self.desktop_key_thumbprint.clone())),
            ("deviceId", Json::str(self.device_id.clone())),
        ];
        if let Some(name) = &self.display_name {
            members.push(("displayName", Json::str(name.clone())));
        }
        members.extend([
            (
                "mobileEndpointKey",
                Json::obj([
                    ("algorithm", Json::str("ed25519")),
                    ("publicKey", Json::str(self.mobile_endpoint.to_b64u())),
                    ("thumbprint", Json::str(self.mobile_endpoint.thumbprint())),
                ]),
            ),
            ("mobileX25519", Json::str(self.mobile_x25519.to_b64u())),
            ("pairingNonce", Json::str(b64::encode(&self.pairing_nonce))),
            ("jti", Json::str(self.jti.clone())),
            ("issuedAt", Json::int(self.issued_at)),
            ("expiresAt", Json::int(self.expires_at)),
        ]);
        Json::Obj(members.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// The canonical text of the claims: what the signature and the MAC cover.
    pub fn canonical(&self) -> Result<String> {
        Ok(self.to_json().to_canonical()?)
    }

    /// Reads the claims object (every member of `pairing-claims.schema.json`, no other).
    fn from_json(v: &Json) -> Result<Claims> {
        let members = v.as_object().ok_or(Error::Invalid("response: claims"))?;
        const REQUIRED: [&str; 10] = [
            "appId",
            "desktopConnectionId",
            "desktopKeyThumbprint",
            "deviceId",
            "mobileEndpointKey",
            "mobileX25519",
            "pairingNonce",
            "jti",
            "issuedAt",
            "expiresAt",
        ];
        if members.iter().any(|(k, _)| !REQUIRED.contains(&k.as_str()) && k != "displayName") || !REQUIRED.iter().all(|k| v.get(k).is_some()) {
            return Err(Error::Invalid("response: claims members"));
        }
        let text = |k: &'static str| v.get_str(k).ok_or(Error::Invalid(k));
        let app_id = text("appId").and_then(|a| if ids::is_app_id(a) { Ok(a) } else { Err(Error::Invalid("response: appId")) })?;
        let desktop_connection_id =
            text("desktopConnectionId")
                .and_then(|a| if ids::is_device_id(a) { Ok(a) } else { Err(Error::Invalid("response: desktopConnectionId")) })?;
        let desktop_key_thumbprint = text("desktopKeyThumbprint").and_then(|a| {
            if ids::is_thumbprint(a) {
                Ok(a)
            } else {
                Err(Error::Invalid("response: desktopKeyThumbprint"))
            }
        })?;
        let device_id = text("deviceId").and_then(|a| if ids::is_device_id(a) { Ok(a) } else { Err(Error::Invalid("response: deviceId")) })?;
        let display_name = match v.get("displayName") {
            None => None,
            Some(n) => {
                let n = n
                    .as_str()
                    .filter(|n| n.chars().count() <= 60 && !n.chars().any(|c| c <= '\u{1f}' || c == '\u{7f}'))
                    .ok_or(Error::Invalid("response: displayName"))?;
                Some(n.to_string())
            }
        };
        let mobile_endpoint = read_endpoint_key(v.get("mobileEndpointKey").ok_or(Error::Invalid("response"))?, "response: mobileEndpointKey")?;
        let mobile_x25519 = X25519Public::from_b64u(text("mobileX25519")?)?;
        let pairing_nonce = b64::decode_exact::<32>(text("pairingNonce")?)?;
        let jti = text("jti").and_then(|a| if ids::is_pairing_jti(a) { Ok(a) } else { Err(Error::Invalid("response: jti")) })?;
        let issued_at = v.get_uint53("issuedAt").ok_or(Error::Invalid("response: issuedAt"))?;
        let expires_at = v.get_uint53("expiresAt").ok_or(Error::Invalid("response: expiresAt"))?;
        if expires_at < issued_at || expires_at - issued_at > CLAIMS_LIFE_MAX_S {
            return Err(Error::Invalid("response: expiresAt - issuedAt"));
        }
        Ok(Claims {
            app_id: app_id.to_string(),
            desktop_connection_id: desktop_connection_id.to_string(),
            desktop_key_thumbprint: desktop_key_thumbprint.to_string(),
            device_id: device_id.to_string(),
            display_name,
            mobile_endpoint,
            mobile_x25519,
            pairing_nonce,
            jti: jti.to_string(),
            issued_at,
            expires_at,
        })
    }
}

/// A response, as text and as parts.
#[derive(Debug, Clone)]
pub struct Response {
    /// The text posted as `{"response": ...}`.
    pub text: String,
    /// The claims.
    pub claims: Claims,
    /// `signature`, as written.
    pub signature: String,
    /// `mac`, as written.
    pub mac: String,
}

impl Response {
    /// Builds the response: the claims signed with the phone's endpoint key and MACed under the pairing MAC key, written as the fixture writes them (`kind`,
    /// `schemaVersion`, `claims` in canonical form, `signature`, `mac`). The key in the claims must be the signer's.
    pub fn build(phone_endpoint: &Signer, mac_key: &Secret<32>, claims: Claims) -> Result<Response> {
        if claims.mobile_endpoint != phone_endpoint.verify_key() {
            return Err(Error::Mismatch("response: the claims name another key than the signer's"));
        }
        let canonical = claims.canonical()?;
        let signature = phone_endpoint.sign_b64u(SignDomain::PairingResponse, &[canonical.as_bytes()]);
        let mac = response_mac(mac_key, &canonical)?;
        let text = format!(
            "{{\"kind\":{},\"schemaVersion\":3,\"claims\":{},\"signature\":{},\"mac\":{}}}",
            quote("aokie_mobile_pairing_response"),
            canonical,
            quote(&signature),
            quote(&mac)
        );
        Ok(Response { text, claims, signature, mac })
    }

    /// Parses the response text: shape and identifiers only (no clock, no MAC, no signature).
    pub fn parse(text: &str) -> Result<Response> {
        let doc = json::parse(text.as_bytes())?;
        let members = doc.as_object().ok_or(Error::Invalid("response: not an object"))?;
        if members.len() != 5
            || doc.get_str("kind") != Some("aokie_mobile_pairing_response")
            || doc.get("schemaVersion").and_then(Json::as_int) != Some(3)
        {
            return Err(Error::Invalid("response: members, kind or schemaVersion"));
        }
        let claims = Claims::from_json(doc.get("claims").ok_or(Error::Invalid("response: claims"))?)?;
        let signature = doc.get_str("signature").ok_or(Error::Invalid("response: signature"))?;
        let mac = doc.get_str("mac").ok_or(Error::Invalid("response: mac"))?;
        b64::decode_exact::<64>(signature)?;
        b64::decode_exact::<32>(mac)?;
        Ok(Response { text: text.to_string(), claims, signature: signature.to_string(), mac: mac.to_string() })
    }

    /// The desktop's checks of a parsed response against its own offer, in the README's order, **stopping at the first failure**: the lifetime against `relay_now` with 30
    /// seconds of slack; the binding to the offer (app, desktop, the desktop key's thumbprint, the nonce, the `jti`); the MAC, in constant time, over the canonical claims
    /// rebuilt from what was parsed; the phone's signature last, strictly, with its own key. (The phone's X25519 key was refused at parse if it is of small order. That the
    /// nonce and `jti` have not been used before, and that the phone key is not revoked, are the desktop's records, kept by [`crate::pairing::desktop`].)
    pub fn verify(&self, offer: &Offer, mac_key: &Secret<32>, relay_now: i64) -> Result<()> {
        let c = &self.claims;
        if (c.issued_at as i64) > relay_now + SKEW_S || (c.expires_at as i64) + SKEW_S <= relay_now {
            return Err(Error::OutsideWindow("pairing response"));
        }
        if c.app_id != offer.app_id
            || c.desktop_connection_id != offer.desktop_connection_id
            || c.desktop_key_thumbprint != offer.desktop_endpoint.thumbprint()
            || c.pairing_nonce != offer.nonce
            || c.jti != offer.jti
        {
            return Err(Error::Mismatch("pairing response: not bound to this offer"));
        }
        let canonical = c.canonical()?;
        verify_response_mac(mac_key, &canonical, &self.mac)?;
        let signature = signature_from_b64u(&self.signature)?;
        c.mobile_endpoint.verify(SignDomain::PairingResponse, &[canonical.as_bytes()], &signature)
    }
}
