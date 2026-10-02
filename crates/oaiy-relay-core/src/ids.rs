//! The identifiers of README section 2 and the device token, as pure functions and a type that cannot be printed.
//!
//! Every function here is the pattern of `common.schema.json` and `Ids.php`, no more and no less: a device id is `dev-` and 22 characters of the base64url alphabet (the
//! relay's reader checks the pattern and not the canonical spelling of an id, and so does this); a thumbprint, a key, a nonce and a token are decoded canonically, because
//! their bytes are what is hashed, MACed or signed.

use core::fmt;

use zeroize::Zeroizing;

use crate::b64;
use crate::error::{Error, Result};

/// The token prefix.
pub const TOKEN_PREFIX: &str = "oaiyrt1.";
/// A token is always 63 characters: the prefix (8), the id (11), a dot and the secret (43).
pub const TOKEN_LEN: usize = 63;

fn all_b64u(text: &str) -> bool {
    text.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// `prefix` and then `n` characters of the base64url alphabet.
fn prefixed(text: &str, prefix: &str, n: usize) -> bool {
    text.strip_prefix(prefix).is_some_and(|rest| rest.len() == n && all_b64u(rest))
}

/// `dev-` and 22 characters.
pub fn is_device_id(text: &str) -> bool {
    prefixed(text, "dev-", 22)
}

/// `prov-` and 22 characters.
pub fn is_provider_id(text: &str) -> bool {
    prefixed(text, "prov-", 22)
}

/// `rly-` and 22 characters.
pub fn is_relay_id(text: &str) -> bool {
    prefixed(text, "rly-", 22)
}

/// A device id or a provider id.
pub fn is_principal_id(text: &str) -> bool {
    is_device_id(text) || is_provider_id(text)
}

/// 22 characters of the alphabet: a `pid`, a `rid`.
pub fn is_pid(text: &str) -> bool {
    text.len() == 22 && all_b64u(text)
}

/// 11 characters of the alphabet: an epoch, a token id, an enrolment key id. The epoch is echoed byte for byte as the relay gave it and is not decoded.
pub fn is_epoch(text: &str) -> bool {
    text.len() == 11 && all_b64u(text)
}

/// An item id: 1 to 128 characters of `A-Za-z0-9._-`, and neither `.` nor `..` (Interpretation 22).
pub fn is_item_id(text: &str) -> bool {
    (1..=128).contains(&text.len())
        && text != "."
        && text != ".."
        && text.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// An app or plugin id: 1 to 64 characters of `A-Za-z0-9_.:-`.
pub fn is_app_id(text: &str) -> bool {
    (1..=64).contains(&text.len()) && text.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
}

/// A key thumbprint, a SHA-256 or an HMAC-SHA-256 as 43 characters that are the canonical spelling of 32 bytes.
pub fn is_key32(text: &str) -> bool {
    b64::decode_exact::<32>(text).is_ok()
}

/// A key thumbprint: the same form as [`is_key32`].
pub fn is_thumbprint(text: &str) -> bool {
    is_key32(text)
}

/// A grant name: `[a-z][a-z0-9_]{0,31}`.
pub fn is_grant(text: &str) -> bool {
    let b = text.as_bytes();
    (1..=32).contains(&b.len()) && b[0].is_ascii_lowercase() && b.iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}

/// The fourteen grant names of the Aokie protocol (Interpretation 32). The relay refuses an approval that carries another.
pub const KNOWN_GRANTS: [&str; 14] = [
    "state_read",
    "caller_read",
    "captions_read",
    "assistance_read",
    "assistance_respond",
    "monitor",
    "consult",
    "takeover",
    "resume_aokie",
    "end_caller",
    "rtc_signal",
    "participants_read",
    "participant_identity_read",
    "audio_levels_read",
];

/// True for one of the fourteen.
pub fn is_known_grant(text: &str) -> bool {
    KNOWN_GRANTS.contains(&text)
}

/// A pairing `jti`: `pair-` and 1 to 64 characters of the alphabet.
pub fn is_pairing_jti(text: &str) -> bool {
    text.strip_prefix("pair-").is_some_and(|rest| (1..=64).contains(&rest.len()) && all_b64u(rest))
}

/// A display name as the relay stores it (`Ids::cleanName`): control characters removed, spaces trimmed, at most `max_chars` characters and at most 120 bytes, cut
/// between characters and never inside one. The byte cap is the shipped phone's (Interpretation 47).
pub fn clean_name(text: &str, max_chars: usize) -> String {
    let stripped: String = text.chars().filter(|c| !(*c <= '\u{1f}' || *c == '\u{7f}')).collect();
    let mut out: String = stripped.trim_matches(' ').chars().take(max_chars).collect();
    while out.len() > 120 {
        out.pop();
    }
    out.trim_matches(' ').to_string()
}

/// A device capability token: `oaiyrt1.` + b64u of 8 bytes + `.` + b64u of 32 bytes (63 characters). It is a credential: it has no `Display`, its `Debug` prints
/// nothing of it, and it is wiped when dropped.
pub struct Token(Zeroizing<String>);

impl Token {
    /// Parses a token as the relay does (README 9.1, rule 3a): the prefix, two parts of the right lengths, the alphabet and a canonical last character in each.
    pub fn parse(text: &str) -> Result<Token> {
        let rest = text.strip_prefix(TOKEN_PREFIX).ok_or(Error::Invalid("token prefix"))?;
        let (id, secret) = rest.split_once('.').ok_or(Error::Invalid("token shape"))?;
        if text.len() != TOKEN_LEN {
            return Err(Error::Invalid("token length"));
        }
        b64::decode_exact::<8>(id)?;
        b64::decode_exact::<32>(secret)?;
        Ok(Token(Zeroizing::new(text.to_string())))
    }

    /// The text, for the one place a bearer is applied. Keep the borrow short and never log it.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// The 11 characters of the token's id (not a secret: the relay's lookup key).
    pub fn id(&self) -> &str {
        &self.0[TOKEN_PREFIX.len()..TOKEN_PREFIX.len() + 11]
    }
}

impl Clone for Token {
    fn clone(&self) -> Self {
        Token(Zeroizing::new(self.0.to_string()))
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({}.., redacted)", &self.0[..TOKEN_PREFIX.len() + 11])
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        oaiy_crypto::zeroize::ct_eq(self.0.as_bytes(), other.0.as_bytes())
    }
}

impl Eq for Token {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_patterns() {
        assert!(is_device_id("dev-oKGio6SlpqeoqaqrrK2urw"));
        assert!(!is_device_id("dev-oKGio6SlpqeoqaqrrK2ur"));
        assert!(!is_device_id("dev-oKGio6SlpqeoqaqrrK2urw="));
        assert!(!is_device_id("prov-oKGio6SlpqeoqaqrrK2urw"));
        assert!(is_provider_id("prov-wMHCw8TFxsfIycrLzM3Ozw"));
        assert!(is_relay_id("rly-0NHS09TV1tfY2drb3N3e3w"));
        assert!(is_principal_id("prov-wMHCw8TFxsfIycrLzM3Ozw") && is_principal_id("dev-oKGio6SlpqeoqaqrrK2urw"));
        assert!(is_pid("b5YkfMcTvJb0g1GTv3kNNQ") && !is_pid("b5YkfMcTvJb0g1GTv3kNN") && !is_pid("b5YkfMcTvJb0g1GTv3kNN+"));
        assert!(is_epoch("eVp54C0-EJY") && !is_epoch("eVp54C0-EJ") && !is_epoch("eVp54C0-EJY1") && !is_epoch("eVp54C0 EJY"));
        assert!(
            is_item_id("cmd-0001")
                && is_item_id("a..b")
                && is_item_id("...")
                && !is_item_id(".")
                && !is_item_id("..")
                && !is_item_id("")
                && !is_item_id("a/b")
        );
        assert!(is_item_id(&"x".repeat(128)) && !is_item_id(&"x".repeat(129)));
        assert!(
            is_app_id("aokie") && is_app_id("a:b.c_d-e") && !is_app_id("") && !is_app_id(&"a".repeat(65)) && !is_app_id("a b") && !is_app_id("a@b")
        );
        assert!(is_thumbprint("atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR4"));
        assert!(!is_thumbprint("atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR5"), "a last character with unused bits set");
        assert!(is_grant("state_read") && !is_grant("State") && !is_grant("") && !is_grant("1a") && !is_grant(&"a".repeat(33)));
        assert!(is_known_grant("takeover") && !is_known_grant("admin"));
        assert!(
            is_pairing_jti("pair-0001")
                && !is_pairing_jti("pair-")
                && !is_pairing_jti("pairx0001")
                && !is_pairing_jti(&format!("pair-{}", "a".repeat(65)))
        );
    }

    #[test]
    fn the_vectors_valid_and_invalid_tokens() {
        for ok in [
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.__________8.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "oaiyrt1.AAECAwQFBgc.AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
        ] {
            assert!(Token::parse(ok).is_ok(), "{ok}");
        }
        for bad in [
            "oaiyrt2.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "OAIYRT1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBw.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBwgA.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8A",
            "oaiyrt1AQIDBAUGBwgICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8.AAAA",
            "oaiyrt1.AQIDBAUGBwg.+CEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8=",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj/",
            " oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8\n",
            "",
            "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj9",
            "oaiyrt1.AQIDBAUGBwh.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
        ] {
            assert!(Token::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_token_prints_nothing_of_its_secret() {
        let t = Token::parse("oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8").unwrap();
        assert_eq!(t.id(), "AQIDBAUGBwg");
        let shown = format!("{t:?}");
        assert!(!shown.contains("ICEiIyQl"), "{shown}");
        assert_eq!(t.clone(), t);
    }

    #[test]
    fn names_are_cleaned_as_the_relay_cleans_them() {
        assert_eq!(clean_name("  Front\u{0}desk\u{7f} PC\n ", 60), "Frontdesk PC");
        assert_eq!(clean_name(&"a".repeat(80), 60).len(), 60);
        // 50 CJK characters are 150 bytes: cut at 120 bytes between characters.
        let cjk = "\u{65e5}".repeat(50);
        let cut = clean_name(&cjk, 60);
        assert_eq!(cut.chars().count(), 40);
        assert_eq!(cut.len(), 120);
        assert_eq!(clean_name("\u{0}\u{1}  ", 60), "");
    }
}
