//! The pairing offer (README 10.1, `pairing-offer.schema.json`): canonical JSON stored on the relay as the exact text, with a MAC over that text.
//!
//! The phone verifies the MAC over the text it received and **does not re-serialise it**; only then does it parse the text, and from then on the keys inside it are the
//! only source of its peer pin (`hostIdentity` and `desktopEndpointKey`), never the relay's answer to anything.

use oaiy_crypto::zeroize::Secret;

use crate::b64;
use crate::error::{Error, Result};
use crate::ids;
use crate::json::{self, Json};
use crate::keys::{thumbprint_of, VerifyKey, X25519Public};
use crate::pairing::math::{offer_mac, verify_offer_mac};
use crate::url::RelayUrl;

/// An offer lives 600 seconds (`expiresAt - issuedAt`).
pub const OFFER_LIFE_S: u64 = 600;
/// The slack either side of an offer's window, in relay-corrected time (README 9.4).
pub const SKEW_S: i64 = 30;

/// What a desktop puts into an offer.
pub struct OfferParams<'a> {
    /// The app the phone is paired for.
    pub app_id: &'a str,
    /// The desktop's relay device id.
    pub desktop_connection_id: &'a str,
    /// The desktop's name as the owner will see it (cleaned to 60 characters).
    pub desktop_name: &'a str,
    /// The desktop's endpoint key (the one that signs the approval receipt).
    pub desktop_endpoint: &'a VerifyKey,
    /// The desktop's endpoint X25519 key.
    pub desktop_x25519: &'a X25519Public,
    /// The host identity's Ed25519 key, which the phone pins.
    pub host_ed25519: &'a VerifyKey,
    /// The host identity's X25519 key, which the phone pins.
    pub host_x25519: &'a X25519Public,
    /// 32 random bytes.
    pub nonce: [u8; 32],
    /// `pair-` and an id.
    pub jti: &'a str,
    /// Relay time at creation.
    pub issued_at: u64,
    /// The relay the offer is for.
    pub relay: &'a RelayUrl,
    /// The thumbprint of that relay's key.
    pub relay_fingerprint: &'a str,
}

/// A parsed offer, with the exact text it was parsed from.
#[derive(Debug, Clone)]
pub struct Offer {
    /// The exact text (what the MAC covers and the relay stores).
    pub text: String,
    /// `appId`.
    pub app_id: String,
    /// `desktopConnectionId`.
    pub desktop_connection_id: String,
    /// `desktopName`.
    pub desktop_name: String,
    /// `desktopEndpointKey.publicKey`.
    pub desktop_endpoint: VerifyKey,
    /// `desktopX25519`.
    pub desktop_x25519: X25519Public,
    /// `hostIdentity.ed25519`.
    pub host_ed25519: VerifyKey,
    /// `hostIdentity.x25519`.
    pub host_x25519: X25519Public,
    /// `nonce`, 32 raw bytes.
    pub nonce: [u8; 32],
    /// `jti`.
    pub jti: String,
    /// `issuedAt`.
    pub issued_at: u64,
    /// `expiresAt`.
    pub expires_at: u64,
    /// `relay.url`.
    pub relay: RelayUrl,
    /// `relay.fingerprint`.
    pub relay_fingerprint: String,
}

fn endpoint_key_json(key: &VerifyKey) -> Json {
    Json::obj([("algorithm", Json::str("ed25519")), ("publicKey", Json::str(key.to_b64u())), ("thumbprint", Json::str(key.thumbprint()))])
}

/// Reads an `EndpointPublicKey` (`{"algorithm":"ed25519","publicKey","thumbprint"}`, exactly those members, the thumbprint recomputed).
pub(crate) fn read_endpoint_key(v: &Json, what: &'static str) -> Result<VerifyKey> {
    let members = v.as_object().ok_or(Error::Invalid(what))?;
    if members.len() != 3 || v.get_str("algorithm") != Some("ed25519") {
        return Err(Error::Invalid(what));
    }
    let key = VerifyKey::from_b64u(v.get_str("publicKey").ok_or(Error::Invalid(what))?)?;
    if v.get_str("thumbprint") != Some(key.thumbprint().as_str()) {
        return Err(Error::Mismatch(what));
    }
    Ok(key)
}

fn key32(v: &Json, key: &str, what: &'static str) -> Result<[u8; 32]> {
    Ok(b64::decode_exact::<32>(v.get_str(key).ok_or(Error::Invalid(what))?)?)
}

impl Offer {
    /// Builds the offer text in canonical form (`expiresAt = issuedAt + 600`) and parses it back, so that what a desktop stores is what a phone will read.
    pub fn build(p: &OfferParams<'_>) -> Result<Offer> {
        let name = ids::clean_name(p.desktop_name, 60);
        let doc = Json::obj([
            ("kind", Json::str("aokie_mobile_pairing")),
            ("schemaVersion", Json::int(3)),
            ("appId", Json::str(p.app_id)),
            ("desktopConnectionId", Json::str(p.desktop_connection_id)),
            ("desktopName", Json::str(name)),
            ("desktopEndpointKey", endpoint_key_json(p.desktop_endpoint)),
            ("desktopX25519", Json::str(p.desktop_x25519.to_b64u())),
            (
                "hostIdentity",
                Json::obj([
                    ("ed25519", Json::str(p.host_ed25519.to_b64u())),
                    ("thumbprint", Json::str(p.host_ed25519.thumbprint())),
                    ("x25519", Json::str(p.host_x25519.to_b64u())),
                ]),
            ),
            ("nonce", Json::str(b64::encode(&p.nonce))),
            ("jti", Json::str(p.jti)),
            ("issuedAt", Json::int(p.issued_at)),
            ("expiresAt", Json::int(p.issued_at + OFFER_LIFE_S)),
            ("relay", Json::obj([("url", Json::str(p.relay.origin())), ("fingerprint", Json::str(p.relay_fingerprint))])),
        ]);
        Offer::parse(&doc.to_canonical()?)
    }

    /// The offer's MAC for the relay's `POST /v1/pair` and the phone's check.
    pub fn mac(&self, mac_key: &Secret<32>) -> Result<String> {
        offer_mac(mac_key, &self.text)
    }

    /// The phone's reading: verify the MAC over `text` exactly as received (in constant time), and only then parse and validate. A text that does not verify is never
    /// parsed.
    pub fn verify(text: &str, mac: &str, mac_key: &Secret<32>) -> Result<Offer> {
        verify_offer_mac(mac_key, text, mac)?;
        Offer::parse(text)
    }

    /// Parses and validates the offer's shape (every member of `pairing-offer.schema.json`, no other; the keys good; the thumbprints recomputed; `expiresAt - issuedAt` is
    /// 600). It checks no MAC and no clock.
    pub fn parse(text: &str) -> Result<Offer> {
        let doc = json::parse(text.as_bytes())?;
        let members = doc.as_object().ok_or(Error::Invalid("offer: not an object"))?;
        const NAMES: [&str; 13] = [
            "kind",
            "schemaVersion",
            "appId",
            "desktopConnectionId",
            "desktopName",
            "desktopEndpointKey",
            "desktopX25519",
            "hostIdentity",
            "nonce",
            "jti",
            "issuedAt",
            "expiresAt",
            "relay",
        ];
        if members.len() != NAMES.len() || !NAMES.iter().all(|n| doc.get(n).is_some()) {
            return Err(Error::Invalid("offer: members"));
        }
        if doc.get_str("kind") != Some("aokie_mobile_pairing") || doc.get("schemaVersion").and_then(Json::as_int) != Some(3) {
            return Err(Error::Invalid("offer: kind or schemaVersion"));
        }
        let app_id = doc.get_str("appId").filter(|a| ids::is_app_id(a)).ok_or(Error::Invalid("offer: appId"))?;
        let desktop_connection_id =
            doc.get_str("desktopConnectionId").filter(|a| ids::is_device_id(a)).ok_or(Error::Invalid("offer: desktopConnectionId"))?;
        let desktop_name = doc
            .get_str("desktopName")
            .filter(|n| n.chars().count() <= 60 && !n.chars().any(|c| c <= '\u{1f}' || c == '\u{7f}'))
            .ok_or(Error::Invalid("offer: desktopName"))?;
        let desktop_endpoint = read_endpoint_key(doc.get("desktopEndpointKey").ok_or(Error::Invalid("offer"))?, "offer: desktopEndpointKey")?;
        let desktop_x25519 = X25519Public::from_bytes(&key32(&doc, "desktopX25519", "offer: desktopX25519")?)?;
        let host = doc.get("hostIdentity").ok_or(Error::Invalid("offer"))?;
        if host.as_object().map(<[_]>::len) != Some(3) {
            return Err(Error::Invalid("offer: hostIdentity"));
        }
        let host_ed_bytes = key32(host, "ed25519", "offer: hostIdentity.ed25519")?;
        let host_ed25519 = VerifyKey::from_bytes(&host_ed_bytes)?;
        if host.get_str("thumbprint") != Some(thumbprint_of(&host_ed_bytes).as_str()) {
            return Err(Error::Mismatch("offer: hostIdentity.thumbprint"));
        }
        let host_x25519 = X25519Public::from_bytes(&key32(host, "x25519", "offer: hostIdentity.x25519")?)?;
        let nonce = key32(&doc, "nonce", "offer: nonce")?;
        let jti = doc.get_str("jti").filter(|j| ids::is_pairing_jti(j)).ok_or(Error::Invalid("offer: jti"))?;
        let issued_at = doc.get_uint53("issuedAt").ok_or(Error::Invalid("offer: issuedAt"))?;
        let expires_at = doc.get_uint53("expiresAt").ok_or(Error::Invalid("offer: expiresAt"))?;
        if expires_at != issued_at + OFFER_LIFE_S {
            return Err(Error::Invalid("offer: expiresAt - issuedAt must be 600"));
        }
        let relay = doc.get("relay").ok_or(Error::Invalid("offer"))?;
        if relay.as_object().map(<[_]>::len) != Some(2) {
            return Err(Error::Invalid("offer: relay"));
        }
        let relay_url = RelayUrl::parse(relay.get_str("url").ok_or(Error::Invalid("offer: relay.url"))?)?;
        let relay_fingerprint = relay.get_str("fingerprint").filter(|f| ids::is_thumbprint(f)).ok_or(Error::Invalid("offer: relay.fingerprint"))?;
        Ok(Offer {
            text: text.to_string(),
            app_id: app_id.to_string(),
            desktop_connection_id: desktop_connection_id.to_string(),
            desktop_name: desktop_name.to_string(),
            desktop_endpoint,
            desktop_x25519,
            host_ed25519,
            host_x25519,
            nonce,
            jti: jti.to_string(),
            issued_at,
            expires_at,
            relay: relay_url,
            relay_fingerprint: relay_fingerprint.to_string(),
        })
    }

    /// Judges the offer's window in relay-corrected time with 30 seconds of slack: not issued in the future, not expired.
    pub fn check_window(&self, relay_now: i64) -> Result<()> {
        if (self.issued_at as i64) > relay_now + SKEW_S || (self.expires_at as i64) + SKEW_S <= relay_now {
            return Err(Error::OutsideWindow("pairing offer"));
        }
        Ok(())
    }
}
