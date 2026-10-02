//! The Aokie admission (README 10.6): the bearer's shape.
//!
//! `aokie-adm-v2.` + lowercase hex of the claims JSON + `.` + lowercase hex of the relay's HMAC-SHA-256 of those exact bytes. A client cannot verify the HMAC (the relay
//! keeps the secret) and treats the whole token as opaque: it sends it back as `Authorization: Bearer` on the compatibility routes and reads only its shape, so that a
//! malformed answer is refused before it is used. The sizes of vectors A4 and A4b (888 characters for a phone, 964 and 92 more per phone for the plugin) are what
//! [`Bearer::len`] reports.

use zeroize::Zeroizing;

use crate::error::{Error, Result};
use crate::json::{self, Json};

/// The bearer's prefix.
pub const PREFIX: &str = "aokie-adm-v2.";
/// The longest bearer `admission-mobile-response.schema.json` allows.
pub const MAX_LEN: usize = 8192;

/// An admission bearer: a credential for the compatibility routes, wiped when dropped and printed nowhere.
pub struct Bearer {
    text: Zeroizing<String>,
    claims: Json,
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    // (`len() % 2` and not `is_multiple_of`, which is newer than the toolchains this crate is built with elsewhere.)
    #[allow(clippy::manual_is_multiple_of)]
    let odd = text.len() % 2 != 0;
    if odd || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return None;
    }
    text.as_bytes().chunks(2).map(|p| u8::from_str_radix(core::str::from_utf8(p).ok()?, 16).ok()).collect()
}

impl Bearer {
    /// Reads the shape: the prefix, lowercase hex that decodes to a JSON object with an integer `exp`, a dot and exactly 64 lowercase hex characters.
    pub fn parse(text: &str) -> Result<Bearer> {
        if text.len() > MAX_LEN {
            return Err(Error::Invalid("bearer: too long"));
        }
        let rest = text.strip_prefix(PREFIX).ok_or(Error::Invalid("bearer: prefix"))?;
        let (claims_hex, mac_hex) = rest.split_once('.').ok_or(Error::Invalid("bearer: shape"))?;
        if mac_hex.len() != 64 || unhex(mac_hex).is_none() {
            return Err(Error::Invalid("bearer: mac"));
        }
        let claims_bytes = unhex(claims_hex).filter(|b| !b.is_empty()).ok_or(Error::Invalid("bearer: claims"))?;
        let claims = json::parse(&claims_bytes)?;
        if !claims.is_object() || claims.get("exp").and_then(Json::as_uint53).is_none() {
            return Err(Error::Invalid("bearer: claims"));
        }
        Ok(Bearer { text: Zeroizing::new(text.to_string()), claims })
    }

    /// The token, for `Authorization: Bearer`. Keep the borrow short.
    pub fn expose(&self) -> &str {
        &self.text
    }

    /// Its length in characters (888 for a phone in vector A4).
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Always false: a bearer has claims.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The claims, as the relay wrote them (advisory to a client: the relay is the party that reads them).
    pub fn claims(&self) -> &Json {
        &self.claims
    }

    /// `exp`, Unix seconds in relay time. The relay accepts the bearer up to and including `exp + 30` seconds.
    pub fn expires_at(&self) -> u64 {
        self.claims.get("exp").and_then(Json::as_uint53).unwrap_or(0)
    }
}

impl core::fmt::Debug for Bearer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Bearer({} characters, redacted)", self.text.len())
    }
}
