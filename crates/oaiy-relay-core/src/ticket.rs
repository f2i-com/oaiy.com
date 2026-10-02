//! Provider tickets: JWS Compact Serialization, `alg` `EdDSA` over Ed25519 (README section 9.2, vectors A1 and A9).
//!
//! A ticket lets a browser reach the relay after a provider has vouched for it, and the desktop verifies the same ticket independently. The signature is over the ASCII bytes
//! of `b64u(header) "." b64u(payload)` **exactly as transmitted** (nothing is canonicalised, and there is no domain string: this is RFC 7515 and RFC 8037). The header is
//! exactly `{"alg":"EdDSA","typ":"oaiy-ticket+jwt","kid":<thumbprint>}` and nothing else, so `none`, `HS256`, `crit`, `jku`, `jwk`, `x5u` and `x5c` are refused by the shape.
//! [`verify_signature`] makes the check that needs the provider's key; [`Claims::check`] makes the ones that need a clock and the desktop's own identity, in **provider
//! time** (README 9.4), never the relay's.

use crate::b64;
use crate::error::{Error, Result};
use crate::ids;
use crate::json::{self, Json};
use crate::keys::{signature_from_b64u, VerifyKey};

/// The claims of a ticket (`ticket-claims.schema.json`: exactly these members).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claims {
    /// `iss`: the provider id.
    pub iss: String,
    /// `aud`: the relay id the ticket is for.
    pub aud: String,
    /// `sub`: an opaque member reference of at most 64 characters.
    pub sub: String,
    /// `iat`.
    pub iat: u64,
    /// `exp` (`exp - iat <= 300`).
    pub exp: u64,
    /// `jti`: single use.
    pub jti: String,
    /// `lane`: `ai` in v1.
    pub lane: String,
    /// `dev`: the desktop's device id the browser may reach.
    pub dev: String,
    /// `org`: the browser origin.
    pub org: String,
    /// `eph`: b64u of SHA-256 of the browser's ephemeral X25519 public key.
    pub eph: String,
}

/// The kid of a ticket's header, to find the pinned provider key with, before anything is verified.
pub fn kid(ticket: &str) -> Result<String> {
    let (header, _, _) = split(ticket)?;
    parse_header(header)
}

fn split(ticket: &str) -> Result<(&str, &str, &str)> {
    let mut it = ticket.split('.');
    match (it.next(), it.next(), it.next(), it.next()) {
        (Some(h), Some(p), Some(s), None) if !h.is_empty() && !p.is_empty() && !s.is_empty() => Ok((h, p, s)),
        _ => Err(Error::Invalid("ticket: exactly two dots")),
    }
}

fn parse_header(header_b64: &str) -> Result<String> {
    let header = json::parse(&b64::decode(header_b64)?)?;
    let members = header.as_object().ok_or(Error::Invalid("ticket: header"))?;
    if members.len() != 3 || header.get_str("alg") != Some("EdDSA") || header.get_str("typ") != Some("oaiy-ticket+jwt") {
        return Err(Error::Invalid("ticket: header"));
    }
    let kid = header.get_str("kid").filter(|k| ids::is_thumbprint(k)).ok_or(Error::Invalid("ticket: kid"))?;
    Ok(kid.to_string())
}

/// Verifies the shape and the signature of `ticket` with the provider key whose thumbprint is the header's `kid`, strictly, and returns the claims. Nothing about time,
/// audience or origin is judged here.
pub fn verify_signature(ticket: &str, provider_key: &VerifyKey) -> Result<Claims> {
    let (header_b64, payload_b64, signature_b64) = split(ticket)?;
    let kid = parse_header(header_b64)?;
    if kid != provider_key.thumbprint() {
        return Err(Error::Mismatch("ticket: kid is not the key given"));
    }
    let signature = signature_from_b64u(signature_b64)?;
    let signing_input = &ticket[..header_b64.len() + 1 + payload_b64.len()];
    provider_key.verify_raw(signing_input.as_bytes(), &signature)?;
    parse_claims(&json::parse(&b64::decode(payload_b64)?)?)
}

fn parse_claims(doc: &Json) -> Result<Claims> {
    let members = doc.as_object().ok_or(Error::Invalid("ticket: claims"))?;
    if members.len() != 10 {
        return Err(Error::Invalid("ticket: claims members"));
    }
    let text = |k: &'static str| doc.get_str(k).map(str::to_string).ok_or(Error::Invalid(k));
    let claims = Claims {
        iss: text("iss")?,
        aud: text("aud")?,
        sub: text("sub")?,
        iat: doc.get_uint53("iat").ok_or(Error::Invalid("iat"))?,
        exp: doc.get_uint53("exp").ok_or(Error::Invalid("exp"))?,
        jti: text("jti")?,
        lane: text("lane")?,
        dev: text("dev")?,
        org: text("org")?,
        eph: text("eph")?,
    };
    let origin_ok = claims
        .org
        .strip_prefix("https://")
        .is_some_and(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':')));
    if !ids::is_provider_id(&claims.iss)
        || !ids::is_relay_id(&claims.aud)
        || !(1..=64).contains(&claims.sub.len())
        || !(1..=64).contains(&claims.jti.len())
        || claims.lane != "ai"
        || !ids::is_device_id(&claims.dev)
        || !origin_ok
        || !ids::is_key32(&claims.eph)
    {
        return Err(Error::Invalid("ticket: a claim"));
    }
    Ok(claims)
}

/// What the desktop knows when it judges a ticket.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    /// This relay's id: a ticket for another relay is refused.
    pub relay_id: &'a str,
    /// This desktop's device id: the ticket must name it.
    pub desktop: &'a str,
    /// The provider clock's `now`, Unix seconds.
    pub provider_now: i64,
}

/// The slack either side of `[iat, exp]`.
pub const SKEW_S: i64 = 30;
/// The longest a ticket may live.
pub const MAX_LIFE_S: u64 = 300;

impl Claims {
    /// The desktop's own checks: `aud` is this relay, `dev` is this desktop, `exp - iat <= 300` and `now` inside `[iat - 30, exp + 30]` in provider time.
    pub fn check(&self, ctx: &Context<'_>) -> Result<()> {
        if self.aud != ctx.relay_id {
            return Err(Error::Mismatch("ticket: aud"));
        }
        if self.dev != ctx.desktop {
            return Err(Error::Mismatch("ticket: dev"));
        }
        if self.exp < self.iat || self.exp - self.iat > MAX_LIFE_S {
            return Err(Error::Invalid("ticket: exp - iat"));
        }
        if ctx.provider_now < self.iat as i64 - SKEW_S || ctx.provider_now > self.exp as i64 + SKEW_S {
            return Err(Error::OutsideWindow("ticket"));
        }
        Ok(())
    }
}
