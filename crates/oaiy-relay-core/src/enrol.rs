//! Enrolment (README section 10.2): an enrolment key (`oaiy://enroll?...`), the key and seed derived from its secret, the request that redeems it and the answer.
//!
//! The secret `s` (16 bytes) is never transmitted. With `salt = "oaiy/enroll/1"` and `IKM = s`: `kid = b64u(HKDF(info = "id", L = 8))` and `seed = HKDF(info = "sig", L = 32)`
//! is an Ed25519 seed; the relay stores only the derived public key, so a copy of its database yields nothing redeemable. The request body is signed as the exact bytes
//! sent (`X-OAIY-Proof = b64u(Ed25519(seed, "oaiy/relay/1/enroll" || 0x00 || body))`), so the member order [`EnrolRequest`] writes is part of what is signed. The client
//! proves the relay's identity (README 8) before it sends anything.

use oaiy_crypto::kdf::hkdf_sha256_secret;
use oaiy_crypto::zeroize::Secret;

use crate::b64;
use crate::error::{Error, Result};
use crate::ids::{self, Token};
use crate::json::{self, Json};
use crate::keys::{SignDomain, Signer, VerifyKey, X25519Public};
use crate::url::{percent_decode, percent_encode, RelayUrl};

/// The HKDF salt of enrolment.
pub const HKDF_SALT: &[u8] = b"oaiy/enroll/1";
/// A key URI is at most this long (README 10.1 states 512 for the pairing key; the enrolment key is held to the same).
pub const MAX_URI_LEN: usize = 512;

/// Who an enrolment key is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A desktop: it gets a `dev-` id.
    Desktop,
    /// A provider such as FormLogic: it gets a `prov-` id.
    Provider,
}

impl Role {
    /// `desktop` or `provider`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Role::Desktop => "desktop",
            Role::Provider => "provider",
        }
    }
}

/// A parsed enrolment key. The secret is wiped when it is dropped and printed nowhere.
pub struct EnrolmentKey {
    /// The relay's base URL (`u`).
    pub relay: RelayUrl,
    /// The thumbprint the relay's key must have (`f`): what the identity proof is checked against.
    pub relay_thumbprint: String,
    /// The key id (`k`), which the secret also derives.
    pub kid: String,
    /// The role the key redeems for (`r`).
    pub role: Role,
    /// When the key expires, Unix seconds (`x`), in the relay's time.
    pub expires_at: u64,
    secret: Secret<16>,
}

impl core::fmt::Debug for EnrolmentKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "EnrolmentKey({}, {}, redacted)", self.relay, self.kid)
    }
}

fn derive<const N: usize>(secret: &Secret<16>, info: &str) -> Result<Secret<N>> {
    Ok(hkdf_sha256_secret::<N>(secret.expose(), Some(HKDF_SALT), info.as_bytes())?)
}

impl EnrolmentKey {
    /// Parses `oaiy://enroll?v=1&u=...&f=...&k=...&s=...&r=...&x=...` (any order of the parameters, unknown ones ignored, a repeated one refused) and checks that
    /// `k` is the key id the secret derives.
    pub fn parse(uri: &str) -> Result<EnrolmentKey> {
        if uri.len() > MAX_URI_LEN {
            return Err(Error::Uri("enrolment key: too long"));
        }
        let query = uri.strip_prefix("oaiy://enroll?").ok_or(Error::Uri("enrolment key: scheme or host"))?;
        let (mut v, mut u, mut f, mut k, mut s, mut r, mut x) = (None, None, None, None, None, None, None);
        for pair in query.split('&') {
            let (name, value) = pair.split_once('=').ok_or(Error::Uri("enrolment key: a parameter has no value"))?;
            let slot = match name {
                "v" => &mut v,
                "u" => &mut u,
                "f" => &mut f,
                "k" => &mut k,
                "s" => &mut s,
                "r" => &mut r,
                "x" => &mut x,
                _ => continue,
            };
            if slot.replace(value).is_some() {
                return Err(Error::Uri("enrolment key: a parameter is repeated"));
            }
        }
        if v != Some("1") {
            return Err(Error::Uri("enrolment key: v must be 1"));
        }
        let relay = RelayUrl::parse(&percent_decode(u.ok_or(Error::Uri("enrolment key: u"))?)?)?;
        let relay_thumbprint = f.filter(|t| ids::is_thumbprint(t)).ok_or(Error::Uri("enrolment key: f"))?.to_string();
        let kid = k.filter(|t| ids::is_epoch(t)).ok_or(Error::Uri("enrolment key: k"))?.to_string();
        let secret =
            Secret::<16>::new(b64::decode_exact::<16>(s.ok_or(Error::Uri("enrolment key: s"))?).map_err(|_| Error::Uri("enrolment key: s"))?);
        let role = match r {
            Some("desktop") => Role::Desktop,
            Some("provider") => Role::Provider,
            _ => return Err(Error::Uri("enrolment key: r")),
        };
        let x = x.ok_or(Error::Uri("enrolment key: x"))?;
        if x.is_empty() || x.len() > 16 || !x.bytes().all(|b| b.is_ascii_digit()) || (x.len() > 1 && x.starts_with('0')) {
            return Err(Error::Uri("enrolment key: x"));
        }
        let expires_at: u64 = x.parse().map_err(|_| Error::Uri("enrolment key: x"))?;
        let id: Secret<8> = derive(&secret, "id")?;
        if b64::encode(id.expose()) != kid {
            return Err(Error::Mismatch("enrolment key: k is not the key id of s"));
        }
        Ok(EnrolmentKey { relay, relay_thumbprint, kid, role, expires_at, secret })
    }

    /// Writes a key in the canonical order (what the relay's CLI and a test make): `v, u, f, k, s, r, x`.
    pub fn to_uri(relay: &RelayUrl, relay_thumbprint: &str, secret: &[u8; 16], role: Role, expires_at: u64) -> Result<String> {
        let s = Secret::<16>::new(*secret);
        let id: Secret<8> = derive(&s, "id")?;
        Ok(format!(
            "oaiy://enroll?v=1&u={}&f={}&k={}&s={}&r={}&x={}",
            percent_encode(&relay.origin()),
            relay_thumbprint,
            b64::encode(id.expose()),
            b64::encode(secret),
            role.as_str(),
            expires_at
        ))
    }

    /// The signer derived from the secret (`HKDF(info = "sig", L = 32)` as an Ed25519 seed): it signs the one enrolment request.
    pub fn signer(&self) -> Result<Signer> {
        let seed: Secret<32> = derive(&self.secret, "sig")?;
        Ok(Signer::from_seed(&seed))
    }
}

/// The request that redeems a key: the body text (written in the order the proof covers) and the value of `X-OAIY-Proof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrolRequest {
    /// The exact body to send.
    pub body: String,
    /// The `X-OAIY-Proof` header.
    pub proof: String,
}

/// Builds the request: `{"kid","role","name","n","keys":{"ed25519","x25519"}}` in that order, with a name cleaned as the relay cleans it (at most 60 characters, control
/// characters removed), and the proof over those exact bytes.
pub fn build_request(key: &EnrolmentKey, name: &str, ed25519: &VerifyKey, x25519: &X25519Public, nonce: &[u8; 16]) -> Result<EnrolRequest> {
    let name = ids::clean_name(name, 60);
    if name.is_empty() {
        return Err(Error::Invalid("enrolment: a device name is required"));
    }
    let body = Json::obj([
        ("kid", Json::str(key.kid.clone())),
        ("role", Json::str(key.role.as_str())),
        ("name", Json::str(name)),
        ("n", Json::str(b64::encode(nonce))),
        ("keys", Json::obj([("ed25519", Json::str(ed25519.to_b64u())), ("x25519", Json::str(x25519.to_b64u()))])),
    ])
    .to_compact();
    let proof = key.signer()?.sign_b64u(SignDomain::Enroll, &[body.as_bytes()]);
    Ok(EnrolRequest { body, proof })
}

/// The `201` answer of `POST /v1/enroll`.
pub struct Enrolled {
    /// `dev-...` for a desktop, `prov-...` for a provider.
    pub device_id: String,
    /// The device token, a credential.
    pub token: Token,
    /// The relay's id, which must be the id of the relay whose proof was verified.
    pub relay_id: String,
    /// The relay's time.
    pub time: u64,
}

impl core::fmt::Debug for Enrolled {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Enrolled({}, {}, token redacted)", self.device_id, self.relay_id)
    }
}

impl Enrolled {
    /// Reads and checks the answer for `role`.
    pub fn parse(body: &[u8], role: Role) -> Result<Enrolled> {
        let doc = json::parse(body)?;
        let device_id = doc.get_str("deviceId").ok_or(Error::Invalid("enrol answer: deviceId"))?;
        let id_ok = match role {
            Role::Desktop => ids::is_device_id(device_id),
            Role::Provider => ids::is_provider_id(device_id),
        };
        if !id_ok {
            return Err(Error::Invalid("enrol answer: deviceId"));
        }
        let token = Token::parse(doc.get_str("token").ok_or(Error::Invalid("enrol answer: token"))?)?;
        let relay_id = doc.get_str("relayId").filter(|r| ids::is_relay_id(r)).ok_or(Error::Invalid("enrol answer: relayId"))?;
        let time = doc.get_uint53("time").ok_or(Error::Invalid("enrol answer: time"))?;
        Ok(Enrolled { device_id: device_id.to_string(), token, relay_id: relay_id.to_string(), time })
    }
}
