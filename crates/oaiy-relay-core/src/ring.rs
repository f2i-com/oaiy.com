//! Ring hints (README section 10.5): the body of a `ring` item, the signature in `hdr.sig`, and the window a phone judges it in.
//!
//! The body is the FCM `data` map: a JSON object whose members are exactly the class's members, every value a string (the shipped Android parser forbids extra members,
//! which is why the signature rides in the header). The signature is `b64u(Ed25519(host identity, "oaiy/relay/1/ring" || 0x00 || the body text exactly as posted))`; a
//! relay-profile phone drops a ring whose signature does not verify against the host identity it pinned at pairing. `expiresAt` is judged in **relay-corrected** time: later
//! than now and at most now + 300 (86,400 for an informational ring). A ring is a wake hint and never authority: no SDP, ICE, token, caption or caller number.

use crate::error::{Error, Result};
use crate::json;
use crate::keys::{SignDomain, Signer, VerifyKey};

/// The four classes of ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingClass {
    /// `voice_offer`
    VoiceOffer,
    /// `voice_offer_cancel`
    VoiceOfferCancel,
    /// `assistance_offer`
    AssistanceOffer,
    /// `informational`
    Informational,
}

/// A validated ring body: its class and every member (all strings), in the order written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingBody {
    /// The class.
    pub class: RingClass,
    /// The members, `aokieClass` and `schemaVersion` included.
    pub members: Vec<(String, String)>,
}

/// How a member's value is checked.
#[derive(Clone, Copy)]
enum Rule {
    /// `^[A-Za-z0-9_.:-]{1,200}$`
    Id,
    /// `^[1-9][0-9]{0,15}$`
    CallEpoch,
    /// `^(0|[1-9][0-9]{0,15})$`
    OwnerEpoch,
    /// `^[0-9]{1,16}$`
    Time,
    /// 1 to 120 characters, no control character.
    Reason,
    /// 1 to 80 characters.
    Title,
    /// 1 to 240 characters.
    Body,
}

fn check(rule: Rule, v: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.len() <= 16 && s.bytes().all(|b| b.is_ascii_digit());
    match rule {
        Rule::Id => (1..=200).contains(&v.len()) && v.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-')),
        Rule::CallEpoch => digits(v) && !v.starts_with('0'),
        Rule::OwnerEpoch => digits(v) && (v == "0" || !v.starts_with('0')),
        Rule::Time => digits(v),
        Rule::Reason => (1..=120).contains(&v.chars().count()) && !v.chars().any(|c| c <= '\u{1f}' || c == '\u{7f}'),
        Rule::Title => (1..=80).contains(&v.chars().count()),
        Rule::Body => (1..=240).contains(&v.chars().count()),
    }
}

/// `(member, rule, required)` of each member of a class, besides `aokieClass` and `schemaVersion`.
type Members = &'static [(&'static str, Rule, bool)];

fn spec(class: &str) -> Option<(RingClass, Members)> {
    use Rule::*;
    Some(match class {
        "voice_offer" => (
            RingClass::VoiceOffer,
            &[
                ("eventId", Id, true),
                ("offerId", Id, true),
                ("appId", Id, true),
                ("callId", Id, true),
                ("callEpoch", CallEpoch, true),
                ("ownerEpoch", OwnerEpoch, true),
                ("expiresAt", Time, true),
            ],
        ),
        "voice_offer_cancel" => (RingClass::VoiceOfferCancel, &[("eventId", Id, true), ("offerId", Id, true), ("reason", Reason, false)]),
        "assistance_offer" => (
            RingClass::AssistanceOffer,
            &[
                ("eventId", Id, true),
                ("appId", Id, true),
                ("requestId", Id, true),
                ("callId", Id, true),
                ("callEpoch", CallEpoch, true),
                ("ownerEpoch", OwnerEpoch, true),
                ("expiresAt", Time, true),
            ],
        ),
        "informational" => {
            (RingClass::Informational, &[("eventId", Id, true), ("title", Title, true), ("body", Body, true), ("expiresAt", Time, true)])
        }
        _ => return None,
    })
}

impl RingBody {
    /// Parses and validates a ring body text: an object, `schemaVersion` `"1"`, a known `aokieClass`, exactly the class's members (every one a string that matches its
    /// rule) and no other.
    pub fn parse(text: &str) -> Result<RingBody> {
        let doc = json::parse(text.as_bytes())?;
        let members = doc.as_object().ok_or(Error::Invalid("ring: not an object"))?;
        let mut strings = Vec::with_capacity(members.len());
        for (k, v) in members {
            strings.push((k.clone(), v.as_str().ok_or(Error::Invalid("ring: a value is not a string"))?.to_string()));
        }
        let class_name = doc.get_str("aokieClass").ok_or(Error::Invalid("ring: aokieClass"))?;
        let (class, rules) = spec(class_name).ok_or(Error::Invalid("ring: unknown aokieClass"))?;
        if doc.get_str("schemaVersion") != Some("1") {
            return Err(Error::Invalid("ring: schemaVersion"));
        }
        for (k, v) in &strings {
            if k == "aokieClass" || k == "schemaVersion" {
                continue;
            }
            let (_, rule, _) = rules.iter().find(|(name, _, _)| name == k).ok_or(Error::Invalid("ring: a member the class does not have"))?;
            if !check(*rule, v) {
                return Err(Error::Invalid("ring: a member's value"));
            }
        }
        for (name, _, required) in rules {
            if *required && !strings.iter().any(|(k, _)| k == name) {
                return Err(Error::Invalid("ring: a required member is missing"));
            }
        }
        Ok(RingBody { class, members: strings })
    }

    /// The value of a member.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.members.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// `expiresAt` as a number, when the class has one.
    pub fn expires_at(&self) -> Option<u64> {
        self.get("expiresAt").and_then(|t| t.parse().ok())
    }

    /// Judges `expiresAt` against `relay_now`, the phone's clock corrected by the relay offset: later than now and at most `now + 300` (`now + 86,400` for an informational
    /// ring). A cancel has no expiry and is always inside its window.
    pub fn check_window(&self, relay_now: i64) -> Result<()> {
        let Some(expires) = self.expires_at() else {
            return Ok(());
        };
        let horizon: i64 = if self.class == RingClass::Informational { 86_400 } else { 300 };
        let expires = i64::try_from(expires).map_err(|_| Error::OutsideWindow("ring"))?;
        if expires <= relay_now || expires > relay_now.saturating_add(horizon) {
            return Err(Error::OutsideWindow("ring"));
        }
        Ok(())
    }
}

/// `hdr.sig` for a ring body: the host identity's signature over `"oaiy/relay/1/ring" || 0x00 || body text exactly as posted`.
pub fn sign(host: &Signer, body_text: &str) -> String {
    host.sign_b64u(SignDomain::Ring, &[body_text.as_bytes()])
}

/// Verifies `hdr.sig` over the body text exactly as received with the host identity the phone pinned at pairing.
pub fn verify(host: &VerifyKey, body_text: &str, hdr_sig: &str) -> Result<()> {
    host.verify_b64u(SignDomain::Ring, &[body_text.as_bytes()], hdr_sig)
}
