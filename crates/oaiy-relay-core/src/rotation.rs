//! A provider's key rotation statement (README section 10.2, vector A10).
//!
//! A later change of a provider's keys arrives as a `ctl` hint and is accepted without asking the owner only if it carries a statement signed by the Ed25519 key that is
//! pinned now. The statement is shipped bytes (`{"v":1,"prev","new":{"ed25519","thumbprint","x25519"},"serial","iat","exp"}`) and `signature = Ed25519(K_prev,
//! "oaiy/relay/1/provider-rotate" || 0x00 || bytes)`, delivered as `{"t":"provider.rotated","b":<b64u bytes>,"s":<b64u signature>}`. The check is made over the bytes
//! received, strictly, and only then are they parsed.

use crate::b64;
use crate::error::{Error, Result};
use crate::json;
use crate::keys::{SignDomain, VerifyKey, X25519Public};

/// A rotation statement that has been verified.
#[derive(Debug, Clone)]
pub struct Statement {
    /// The thumbprint of the key that was replaced.
    pub prev: String,
    /// The new signing key.
    pub new_ed25519: VerifyKey,
    /// The new sealing key.
    pub new_x25519: X25519Public,
    /// The statement's serial: greater than the last accepted one.
    pub serial: u64,
    /// When it was made, in provider time.
    pub iat: u64,
    /// When it lapses, in provider time.
    pub exp: u64,
}

/// The longest a statement may live: 24 hours.
pub const MAX_LIFE_S: u64 = 86_400;
/// The slack either side of `[iat, exp]`, in provider time.
pub const SKEW_S: i64 = 30;

/// Verifies a `provider.rotated` hint: `b` and `s` as the `ctl` item carries them. `pinned` is the key the owner pinned, `last_serial` the last serial accepted for this
/// provider, `provider_now` the provider clock (README 9.4: not the relay's).
pub fn verify(pinned: &VerifyKey, last_serial: u64, provider_now: i64, b: &str, s: &str) -> Result<Statement> {
    let bytes = b64::decode(b)?;
    pinned.verify_b64u(SignDomain::ProviderRotate, &[&bytes], s)?;
    let doc = json::parse(&bytes)?;
    let members = doc.as_object().ok_or(Error::Invalid("rotation: not an object"))?;
    if members.iter().any(|(k, _)| !matches!(k.as_str(), "v" | "prev" | "new" | "serial" | "iat" | "exp")) || members.len() != 6 {
        return Err(Error::Invalid("rotation: members"));
    }
    if doc.get("v").and_then(|v| v.as_int()) != Some(1) {
        return Err(Error::Invalid("rotation: v"));
    }
    let prev = doc.get_str("prev").ok_or(Error::Invalid("rotation: prev"))?;
    if prev != pinned.thumbprint() {
        return Err(Error::Mismatch("rotation: prev is not the pinned key"));
    }
    let new = doc.get("new").ok_or(Error::Invalid("rotation: new"))?;
    let new_members = new.as_object().ok_or(Error::Invalid("rotation: new"))?;
    if new_members.len() != 3 {
        return Err(Error::Invalid("rotation: new"));
    }
    let new_ed25519 = VerifyKey::from_b64u(new.get_str("ed25519").ok_or(Error::Invalid("rotation: new.ed25519"))?)?;
    let new_x25519 = X25519Public::from_b64u(new.get_str("x25519").ok_or(Error::Invalid("rotation: new.x25519"))?)?;
    if new.get_str("thumbprint") != Some(new_ed25519.thumbprint().as_str()) {
        return Err(Error::Mismatch("rotation: new.thumbprint"));
    }
    let serial = doc.get_uint53("serial").filter(|n| *n >= 1).ok_or(Error::Invalid("rotation: serial"))?;
    if serial <= last_serial {
        return Err(Error::Mismatch("rotation: the serial is not above the last accepted"));
    }
    let iat = doc.get_uint53("iat").ok_or(Error::Invalid("rotation: iat"))?;
    let exp = doc.get_uint53("exp").ok_or(Error::Invalid("rotation: exp"))?;
    if exp < iat || exp - iat > MAX_LIFE_S {
        return Err(Error::Invalid("rotation: exp - iat"));
    }
    let (lo, hi) = (iat as i64 - SKEW_S, exp as i64 + SKEW_S);
    if provider_now < lo || provider_now > hi {
        return Err(Error::OutsideWindow("rotation"));
    }
    Ok(Statement { prev: prev.to_string(), new_ed25519, new_x25519, serial, iat, exp })
}
