//! Tokens: the grammar, the hash at rest, the CSRF value, and the strict parsing of a bearer.
//!
//! ```text
//! token  = "oaiy" kind "_" id "_" secret
//! kind   = "pat" / "ses" / "dsk" / "run" / "con" / "dev"
//! id     = 16 lowercase hex characters   (8 random bytes; public; shown in lists; used to revoke)
//! secret = 43 characters of base64url, no padding   (32 random bytes)
//! ```
//!
//! Always 68 bytes. What is stored is `lowercase_hex(SHA-256(secret as its 43 ASCII characters))`: a
//! 256-bit secret needs no salt and no slow hash. Every comparison of a secret in this design uses
//! `subtle::ConstantTimeEq` on fixed-length values ([`hashes_equal`]); a test greps the auth code for
//! anything else.
//!
//! Randomness is the operating system's. If it gives none, minting fails: there is no fallback.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// The length of every token: `oaiy` + kind (3) + `_` + id (16) + `_` + secret (43).
pub const TOKEN_LEN: usize = 68;
/// A bearer longer than this is refused before any lookup.
pub const MAX_BEARER_LEN: usize = 128;
/// The length of a legacy pairing token, `oaiypat_` + 64 hex characters.
pub const LEGACY_TOKEN_LEN: usize = 72;

const ID_LEN: usize = 16;
const SECRET_LEN: usize = 43;

/// What a credential is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// A paired app or tool.
    Pat,
    /// A cookie session.
    Ses,
    /// A desktop webview's credential.
    Dsk,
    /// A per-run or derived credential.
    Run,
    /// The console.
    Con,
    /// The owner's known-device cookie (not a principal: it only routes a login).
    Dev,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Pat => "pat",
            Kind::Ses => "ses",
            Kind::Dsk => "dsk",
            Kind::Run => "run",
            Kind::Con => "con",
            Kind::Dev => "dev",
        }
    }

    pub fn from_name(name: &str) -> Option<Kind> {
        [
            Kind::Pat,
            Kind::Ses,
            Kind::Dsk,
            Kind::Run,
            Kind::Con,
            Kind::Dev,
        ]
        .into_iter()
        .find(|k| k.name() == name)
    }

    /// `oaiypat_`: the text a token of this kind starts with.
    pub fn prefix(self) -> String {
        format!("oaiy{}_", self.name())
    }

    /// `ses` and `dev` travel only in their cookies; the rest only in `Authorization: Bearer`.
    pub fn is_cookie_only(self) -> bool {
        matches!(self, Kind::Ses | Kind::Dev)
    }
}

/// A token taken apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parsed<'a> {
    pub kind: Kind,
    pub id: &'a str,
    pub secret: &'a str,
}

fn is_lower_hex(b: u8) -> bool {
    b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
}

fn is_base64url(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Take a token apart if it has the grammar, `^oaiy(pat|ses|dsk|run|con|dev)_[0-9a-f]{16}_[A-Za-z0-9_-]{43}$`
/// (exactly 68 bytes), and `None` otherwise.
pub fn parse(token: &str) -> Option<Parsed<'_>> {
    let b = token.as_bytes();
    if b.len() != TOKEN_LEN || !token.is_ascii() || !token.starts_with("oaiy") {
        return None;
    }
    let kind = Kind::from_name(&token[4..7])?;
    if b[7] != b'_' || b[8 + ID_LEN] != b'_' {
        return None;
    }
    let id = &token[8..8 + ID_LEN];
    let secret = &token[9 + ID_LEN..];
    debug_assert_eq!(secret.len(), SECRET_LEN);
    (id.bytes().all(is_lower_hex) && secret.bytes().all(is_base64url)).then_some(Parsed {
        kind,
        id,
        secret,
    })
}

/// A legacy pairing token, `oaiypat_` and 64 lowercase hex characters (72 bytes): accepted only for
/// records the migration imported, looked up by the hash of the whole token.
pub fn is_legacy_shape(token: &str) -> bool {
    token.len() == LEGACY_TOKEN_LEN
        && token.starts_with("oaiypat_")
        && token.as_bytes()[8..].iter().all(|b| is_lower_hex(*b))
}

pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

/// What is stored for a token: `lowercase_hex(SHA-256(secret))`, the secret as its 43 ASCII characters.
pub fn secret_hash(secret: &str) -> String {
    to_hex(&Sha256::digest(secret.as_bytes()))
}

/// What is stored for a legacy pairing token: the hash of the whole token.
pub fn legacy_hash(token: &str) -> String {
    to_hex(&Sha256::digest(token.as_bytes()))
}

/// Constant-time equality of two stored hashes. The only way a secret's hash is compared.
pub fn hashes_equal(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Constant-time equality of two secrets (a CSRF value, a code) of any origin: for the values that
/// are not hashed before they are compared.
pub fn secrets_equal(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// Why a token could not be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MintError {
    /// The operating system gave no randomness. Minting fails loudly: there is no fallback.
    NoRandomness,
    /// This kind is not made through the store (the device cookie lives in `owner.json`).
    KindNotMintable(Kind),
}

impl std::fmt::Display for MintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MintError::NoRandomness => write!(
                f,
                "the operating system gave no randomness, so no credential can be made"
            ),
            MintError::KindNotMintable(k) => {
                write!(f, "a {} credential is not made here", k.name())
            }
        }
    }
}

impl std::error::Error for MintError {}

/// A new token and what is stored for it. The `Debug` form never shows the token.
#[derive(Clone)]
pub struct NewToken {
    pub token: String,
    pub id: String,
    pub hash: String,
}

impl std::fmt::Debug for NewToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewToken")
            .field("id", &self.id)
            .field("token", &"[redacted]")
            .finish()
    }
}

/// A source of random bytes; the operating system's in production.
pub type Fill<'a> = &'a mut dyn FnMut(&mut [u8]) -> Result<(), MintError>;

/// The operating system's randomness, or [`MintError::NoRandomness`].
pub fn os_random(buf: &mut [u8]) -> Result<(), MintError> {
    getrandom::getrandom(buf).map_err(|_| MintError::NoRandomness)
}

/// Make a token of `kind` from the random source `fill`.
pub fn mint_with(kind: Kind, fill: Fill<'_>) -> Result<NewToken, MintError> {
    let mut id = [0u8; 8];
    let mut secret = [0u8; 32];
    fill(&mut id)?;
    fill(&mut secret)?;
    Ok(build(kind, &id, &secret))
}

/// Make a token of `kind` from the operating system's randomness.
pub fn mint(kind: Kind) -> Result<NewToken, MintError> {
    mint_with(kind, &mut os_random)
}

/// The token for these exact bytes. Deterministic; for the vectors and the tests.
pub fn build(kind: Kind, id: &[u8; 8], secret: &[u8; 32]) -> NewToken {
    let secret_text = URL_SAFE_NO_PAD.encode(secret);
    let id_text = to_hex(id);
    let token = format!("oaiy{}_{}_{}", kind.name(), id_text, secret_text);
    let hash = secret_hash(&secret_text);
    NewToken {
        token,
        id: id_text,
        hash,
    }
}

/// The CSRF value of a cookie session (design 4.5.2): `base64url(HMAC-SHA256(key = the session
/// token's 32 secret bytes, message = "oaiy-csrf-v1"))`, 43 characters. The server recomputes it from
/// the cookie it received; nothing is stored. `None` if `secret` is not 43 base64url characters that
/// decode to 32 bytes.
pub fn csrf_value(secret: &str) -> Option<String> {
    if secret.len() != SECRET_LEN || !secret.bytes().all(is_base64url) {
        return None;
    }
    let key = URL_SAFE_NO_PAD
        .decode(secret)
        .ok()
        .filter(|k| k.len() == 32)?;
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key).ok()?;
    mac.update(b"oaiy-csrf-v1");
    Some(URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes()))
}

/// Why a request's `Authorization` header was refused before any lookup (all `400 bad_request`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BearerError {
    /// More than one `Authorization` header.
    Multiple,
    /// Not `Bearer <token>`: another scheme, the scheme in another case, no space or two.
    NotBearer,
    /// Over 128 bytes.
    TooLong,
    /// A byte outside `[A-Za-z0-9._~+/=-]` (a control character, a space, non-ASCII), or no token.
    BadCharset,
    /// A session or device token in `Authorization`: a bearer skips the CSRF and Fetch Metadata
    /// rules, so a session must never travel as one.
    CookieOnlyKind,
}

impl BearerError {
    pub fn message(self) -> &'static str {
        match self {
            BearerError::Multiple => "more than one Authorization header",
            BearerError::NotBearer => "the Authorization header must be `Bearer <token>`",
            BearerError::TooLong => "the bearer token is too long",
            BearerError::BadCharset => "the bearer token has a character a token never has",
            BearerError::CookieOnlyKind => "a session token travels only in its cookie",
        }
    }
}

fn bearer_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'~' | b'+' | b'/' | b'=' | b'-')
}

/// The bearer token in a request's `Authorization` headers (`values`: the raw value of each one), if
/// any. Strict, and before any lookup: two headers, a scheme in another case, a token over 128 bytes
/// or with a character outside `[A-Za-z0-9._~+/=-]`, and an `oaiyses_` or `oaiydev_` token are each
/// refused.
pub fn bearer_from_headers<'a>(values: &[&'a [u8]]) -> Result<Option<&'a str>, BearerError> {
    let value = match values {
        [] => return Ok(None),
        [one] => *one,
        _ => return Err(BearerError::Multiple),
    };
    let rest = value
        .strip_prefix(b"Bearer ")
        .ok_or(BearerError::NotBearer)?;
    if rest.len() > MAX_BEARER_LEN {
        return Err(BearerError::TooLong);
    }
    if rest.is_empty() || !rest.iter().all(|b| bearer_char(*b)) {
        return Err(BearerError::BadCharset);
    }
    // Every byte is ASCII by the check above.
    let token = std::str::from_utf8(rest).map_err(|_| BearerError::BadCharset)?;
    if token.starts_with("oaiyses_") || token.starts_with("oaiydev_") {
        return Err(BearerError::CookieOnlyKind);
    }
    Ok(Some(token))
}

/// The shortest static token (`OAIY_SERVER_TOKEN`) the design allows.
pub const STATIC_TOKEN_MIN_LEN: usize = 32;
/// The longest.
pub const STATIC_TOKEN_MAX_LEN: usize = 256;
/// The fewest distinct characters it may be made of.
pub const STATIC_TOKEN_MIN_DISTINCT: usize = 8;
/// What it must be worth, in thousandths of a bit: its length times what one character of the alphabet it seems to be
/// drawn from carries (see [`millibits_per_character`]). 128 bits: 32 hex digits, 22 letters and digits, 44 base64.
pub const STATIC_TOKEN_MIN_MILLIBITS: usize = 128_000;
/// A run of this many characters, each the next or the previous one in the character set (`abcdefgh`, `87654321`).
const SEQUENTIAL_RUN: usize = 8;
/// A run of this many characters along one row of the keyboard (`qwerty`, `poiuyt`), whatever the case.
const KEYBOARD_RUN: usize = 6;
/// A piece of this many characters that occurs twice. Every token that repeats itself with a period up to its length
/// less this contains one, so the check for a token made of one short piece over and over is this one.
const REPEATED_PIECE: usize = 8;
/// What a token that a person made up, or copied from an example, is likely to contain, whatever the case.
const PLACEHOLDERS: [&str; 12] = [
    "change-me",
    "changeme",
    "password",
    "secret",
    "token",
    "example",
    "test",
    "admin",
    "default",
    "xxxx",
    "0000",
    "1234",
];
const KEYBOARD_ROWS: [&str; 3] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];

/// Why a value is not a static token the server takes (design 4.1 gave it a shape, `^[\x21-\x7e]{32,256}$` with 16
/// different characters, which refused what `openssl rand -hex 32` makes about one time in four and took
/// `abcdefghijklmnopqrstuvwxyz0123456789ABCD`: it is now worth 128 bits and no pattern).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticTokenShape {
    TooShort,
    TooLong,
    /// A space, a control character, a byte over 0x7e.
    BadCharacter,
    TooFewDistinct,
    /// A word that a placeholder is made of (`change-me`, `password`, `test`, `1234`).
    Placeholder,
    /// A run that counts up or down, or follows the keyboard.
    Sequential,
    /// A piece of it occurs twice.
    Repeats,
    /// Its length, for the alphabet it is written in, is worth less than 128 bits.
    TooLittleEntropy,
}

impl StaticTokenShape {
    pub fn message(self) -> &'static str {
        match self {
            StaticTokenShape::TooShort => "OAIY_SERVER_TOKEN is shorter than 32 characters",
            StaticTokenShape::TooLong => "OAIY_SERVER_TOKEN is longer than 256 characters",
            StaticTokenShape::BadCharacter => {
                "OAIY_SERVER_TOKEN has a space, a control character or a non-ASCII character"
            }
            StaticTokenShape::TooFewDistinct => {
                "OAIY_SERVER_TOKEN is made of fewer than 8 different characters"
            }
            StaticTokenShape::Placeholder => {
                "OAIY_SERVER_TOKEN has a word in it that a placeholder is made of (change-me, password, secret, token, example, test, admin, default, 1234)"
            }
            StaticTokenShape::Sequential => {
                "OAIY_SERVER_TOKEN has a run of characters that count up or down (abcdefgh, 12345678) or follow the keyboard (qwerty)"
            }
            StaticTokenShape::Repeats => {
                "OAIY_SERVER_TOKEN repeats itself (a piece of 8 characters occurs twice)"
            }
            StaticTokenShape::TooLittleEntropy => {
                "OAIY_SERVER_TOKEN is too short for what it is made of: it must be worth 128 bits (32 hex digits, 22 letters and digits, 20 printable characters)"
            }
        }
    }
}

/// What one character of `token` is worth, in thousandths of a bit: the log of the size of the smallest of the
/// alphabets people generate tokens in that holds all of it. Decimal digits 10, hex digits of one case 16, base32
/// 32, letters and digits 62, base64 and base64url 64, anything else the 94 printable characters. Trailing `=` is
/// the padding of base32 and base64.
fn millibits_per_character(token: &[u8]) -> usize {
    fn decimal(b: u8) -> bool {
        b.is_ascii_digit()
    }
    fn hex_lower(b: u8) -> bool {
        matches!(b, b'0'..=b'9' | b'a'..=b'f')
    }
    fn hex_upper(b: u8) -> bool {
        matches!(b, b'0'..=b'9' | b'A'..=b'F')
    }
    fn base32(b: u8) -> bool {
        b.is_ascii_uppercase() || matches!(b, b'2'..=b'7')
    }
    fn alphanumeric(b: u8) -> bool {
        b.is_ascii_alphanumeric()
    }
    fn base64(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'+' || b == b'/'
    }
    fn base64url(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
    }
    let body_len = token.iter().rposition(|b| *b != b'=').map_or(0, |i| i + 1);
    let padded = body_len < token.len();
    let body = &token[..body_len];
    let all = |member: fn(u8) -> bool| !body.is_empty() && body.iter().all(|b| member(*b));
    if !padded && all(decimal) {
        3321
    } else if !padded && (all(hex_lower) || all(hex_upper)) {
        4000
    } else if all(base32) {
        5000
    } else if !padded && all(alphanumeric) {
        5954
    } else if all(base64) || (!padded && all(base64url)) {
        6000
    } else {
        6554
    }
}

/// A run of [`SEQUENTIAL_RUN`] characters that each differ from the one before by one in the character set.
fn counts_in_the_character_set(token: &[u8]) -> bool {
    let mut run = 1;
    for pair in token.windows(2) {
        if pair[1].abs_diff(pair[0]) == 1 {
            run += 1;
            if run >= SEQUENTIAL_RUN {
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}

/// A piece of [`KEYBOARD_RUN`] characters that is along a row of the keyboard, either way.
fn follows_the_keyboard(token: &[u8]) -> bool {
    let lower: Vec<u8> = token.iter().map(u8::to_ascii_lowercase).collect();
    KEYBOARD_ROWS.iter().any(|row| {
        let forward = row.as_bytes();
        let backward: Vec<u8> = forward.iter().rev().copied().collect();
        lower.windows(KEYBOARD_RUN).any(|piece| {
            forward.windows(KEYBOARD_RUN).any(|p| p == piece)
                || backward.windows(KEYBOARD_RUN).any(|p| p == piece)
        })
    })
}

/// A piece of [`REPEATED_PIECE`] characters that occurs twice, the two allowed to overlap.
fn repeats_a_piece(token: &[u8]) -> bool {
    let mut seen = std::collections::HashSet::new();
    token
        .windows(REPEATED_PIECE)
        .any(|piece| !seen.insert(piece))
}

/// Whether `token` is one the server takes as `OAIY_SERVER_TOKEN`: 32 to 256 printable ASCII characters (0x21 to
/// 0x7e), at least 8 of them different, worth at least 128 bits for the alphabet it is written in, and no pattern: no
/// word a placeholder is made of, no run that counts or follows the keyboard, no piece of 8 characters twice.
/// Every generator of the tokens people use (`openssl rand -hex 32`, `-base64 32`, `uuidgen`, `secrets.token_urlsafe`)
/// passes about 999 times in a thousand or better, and what a person makes up does not. `oaiy-server` refuses to
/// start with a token that fails it (`auth::exposure`, rule 5), the guard ignores such a token in `scoped` and
/// `shadow` mode (the desktop's way), it decides which wide tokens [`bearer_or_static`] takes, and `oaiy-server
/// check` says so: this is the one function.
pub fn check_static_token_shape(token: &str) -> Result<(), StaticTokenShape> {
    let bytes = token.as_bytes();
    if !bytes.iter().all(|b| (0x21..=0x7e).contains(b)) {
        return Err(StaticTokenShape::BadCharacter);
    }
    if bytes.len() < STATIC_TOKEN_MIN_LEN {
        return Err(StaticTokenShape::TooShort);
    }
    if bytes.len() > STATIC_TOKEN_MAX_LEN {
        return Err(StaticTokenShape::TooLong);
    }
    let mut seen = [false; 128];
    for b in bytes {
        seen[*b as usize] = true;
    }
    if seen.iter().filter(|s| **s).count() < STATIC_TOKEN_MIN_DISTINCT {
        return Err(StaticTokenShape::TooFewDistinct);
    }
    let lower = token.to_ascii_lowercase();
    if PLACEHOLDERS.iter().any(|word| lower.contains(word)) {
        return Err(StaticTokenShape::Placeholder);
    }
    if counts_in_the_character_set(bytes) || follows_the_keyboard(bytes) {
        return Err(StaticTokenShape::Sequential);
    }
    if repeats_a_piece(bytes) {
        return Err(StaticTokenShape::Repeats);
    }
    if bytes.len() * millibits_per_character(bytes) < STATIC_TOKEN_MIN_MILLIBITS {
        return Err(StaticTokenShape::TooLittleEntropy);
    }
    Ok(())
}

/// [`bearer_from_headers`], except that the operator's own static token is a bearer whatever its shape
/// within design 4.1's rule for it: `[\x21-\x7e]{32,256}` is wider than the strict bearer rule
/// (`[A-Za-z0-9._~+/=-]` and 128 bytes), and a token with a `$` in it, or one of 200 characters, must not be
/// a `400` in front of a server that was configured with it. Only a bearer that is exactly the configured
/// token, compared in constant time, gets this; every other bearer is held to the strict rule, so nothing
/// hostile that is not the operator's own token gets past it.
pub fn bearer_or_static<'a>(
    values: &[&'a [u8]],
    static_token: Option<&str>,
) -> Result<Option<&'a str>, BearerError> {
    match bearer_from_headers(values) {
        Err(e @ (BearerError::TooLong | BearerError::BadCharset)) => {
            if let ([one], Some(want)) = (values, static_token) {
                if let Some(rest) = one.strip_prefix(b"Bearer ") {
                    if check_static_token_shape(want).is_ok()
                        && secrets_equal(rest, want.as_bytes())
                    {
                        // Printable ASCII, by the shape.
                        return std::str::from_utf8(rest)
                            .map(Some)
                            .map_err(|_| BearerError::BadCharset);
                    }
                }
            }
            Err(e)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The design's known-answer vectors (4.1), computed with Node's crypto.
    const SECRET_BYTES: [u8; 32] = {
        let mut b = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            b[i] = i as u8;
            i += 1;
        }
        b
    };
    const ID_BYTES: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    const SECRET: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const TOKEN: &str = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const HASH: &str = "ea866a757e4c38babfa8127cbe9a409d3e1f93a00ff1488ff735fcf917afffd0";
    const CSRF: &str = "A-wCyU91xe9jXHeH2QyQhZgv00DTWUwk1H9x_7RSRmg";

    #[test]
    fn the_known_answer_token_and_stored_hash() {
        let made = build(Kind::Pat, &ID_BYTES, &SECRET_BYTES);
        assert_eq!(made.token, TOKEN);
        assert_eq!(made.token.len(), TOKEN_LEN);
        assert_eq!(made.id, "0123456789abcdef");
        assert_eq!(made.hash, HASH);
        assert_eq!(secret_hash(SECRET), HASH);
        let parsed = parse(TOKEN).unwrap();
        assert_eq!(
            (parsed.kind, parsed.id, parsed.secret),
            (Kind::Pat, "0123456789abcdef", SECRET)
        );
    }

    #[test]
    fn the_known_answer_csrf_value_for_that_session_secret() {
        assert_eq!(csrf_value(SECRET).as_deref(), Some(CSRF));
        assert_eq!(csrf_value(SECRET).unwrap().len(), 43);
        // Not a secret: nothing to derive from.
        assert_eq!(csrf_value(""), None);
        assert_eq!(csrf_value(&SECRET[..42]), None);
        assert_eq!(csrf_value(&format!("{SECRET}A")), None);
        assert_eq!(
            csrf_value("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh!"),
            None
        );
        // 43 characters that decode to 32 bytes but are not canonical still decode: the last
        // character of a 32-byte value has two spare bits, and a different one is a different key.
        assert_ne!(
            csrf_value("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh4"),
            Some(CSRF.to_string())
        );
    }

    #[test]
    fn the_flow_hash_vectors_are_plain_sha256_of_the_bytes() {
        // FA1: hash = lowercase_hex(SHA-256(the bytes the store holds)). The vector for the 34 bytes.
        let flow = br#"{"name":"t","nodes":[],"edges":[]}"#;
        assert_eq!(flow.len(), 34);
        assert_eq!(
            to_hex(&Sha256::digest(flow)),
            "8704924f033b1c3bd1d185430e578e49e15b3ec2dc910466e13e4246066c5111"
        );
        // The design's second vector says "one space added before `]`"; it is a space before the
        // final `}` (found by trying every position).
        let edited = br#"{"name":"t","nodes":[],"edges":[] }"#;
        assert_eq!(
            to_hex(&Sha256::digest(edited)),
            "21a26da4586da22321d6dd31f970049ce8d7a69b597292608cb065757c2b95a3"
        );
    }

    #[test]
    fn the_legacy_token_is_72_bytes_and_hashed_whole() {
        let legacy = format!("oaiypat_{}", "0123456789abcdef".repeat(4));
        assert_eq!(legacy.len(), LEGACY_TOKEN_LEN);
        assert!(is_legacy_shape(&legacy));
        assert!(!is_legacy_shape(&legacy[..71]));
        assert!(!is_legacy_shape(&format!("{legacy}0")));
        assert!(!is_legacy_shape(&legacy.to_uppercase()));
        assert!(!is_legacy_shape(&format!("oaiyses_{}", "0".repeat(64))));
        assert_eq!(parse(&legacy), None, "a legacy token is not the grammar");
        assert_eq!(legacy_hash(&legacy).len(), 64);
    }

    #[test]
    fn a_token_is_exactly_68_bytes_of_the_grammar_and_nothing_else() {
        assert!(parse(TOKEN).is_some());
        for kind in ["pat", "ses", "dsk", "run", "con", "dev"] {
            let t = format!("oaiy{kind}_0123456789abcdef_{SECRET}");
            assert_eq!(parse(&t).map(|p| p.kind.name()), Some(kind), "{t}");
        }
        // One byte off the length, either way (the "regex length off by one" mutant).
        assert_eq!(parse(&format!("{TOKEN}A")), None, "69 bytes");
        assert_eq!(parse(&TOKEN[..67]), None, "67 bytes");
        assert_eq!(
            parse(&format!("{}{}", &TOKEN[..25], &SECRET[1..])),
            None,
            "a 42-character secret"
        );
        for bad in [
            TOKEN.replace("oaiypat_", "oaiypaT_"),
            TOKEN.replace("oaiypat_", "oaiyxyz_"),
            TOKEN.replace("oaiypat_", "OAIYPAT_"),
            TOKEN.replace("oaiypat_", "oaiypat-"),
            TOKEN.replace("0123456789abcdef", "0123456789ABCDEF"),
            TOKEN.replace("0123456789abcdef", "0123456789abcdeg"),
            TOKEN.replace("_AAEC", "-AAEC"),
            TOKEN.replace("AAEC", "AA+C"),
            TOKEN.replace("AAEC", "AA=C"),
            TOKEN.replace("AAEC", "AA C"),
            TOKEN.replace("AAEC", "AA\nC"),
            TOKEN.replace("AAEC", "AAE\u{e9}"),
            format!(" {}", &TOKEN[1..]),
        ] {
            assert_eq!(parse(&bad), None, "{bad:?}");
        }
        // A multi-byte character that keeps the byte length at 68 must not slice a boundary.
        let multibyte = format!("oaiypat_0123456789abcdef_{}\u{e9}", &SECRET[..41]);
        assert_eq!(multibyte.len(), TOKEN_LEN);
        assert_eq!(parse(&multibyte), None);
    }

    #[test]
    fn cookie_only_kinds_are_ses_and_dev() {
        assert!(Kind::Ses.is_cookie_only() && Kind::Dev.is_cookie_only());
        assert!(
            !Kind::Pat.is_cookie_only()
                && !Kind::Dsk.is_cookie_only()
                && !Kind::Run.is_cookie_only()
                && !Kind::Con.is_cookie_only()
        );
        assert_eq!(Kind::Pat.prefix(), "oaiypat_");
        assert_eq!(Kind::from_name("pat"), Some(Kind::Pat));
        assert_eq!(Kind::from_name("PAT"), None);
        assert_eq!(Kind::from_name("token"), None);
    }

    #[test]
    fn minting_uses_the_random_source_and_fails_loudly_without_one() {
        let a = mint(Kind::Pat).unwrap();
        let b = mint(Kind::Pat).unwrap();
        assert_ne!(a.token, b.token);
        assert_ne!(a.id, b.id);
        let parsed = parse(&a.token).expect("a minted token has the grammar");
        assert_eq!(parsed.id, a.id);
        assert_eq!(secret_hash(parsed.secret), a.hash);
        // No randomness: no token, and no partial one.
        let mut none = |_: &mut [u8]| Err(MintError::NoRandomness);
        assert_eq!(
            mint_with(Kind::Pat, &mut none).map(|t| t.token),
            Err(MintError::NoRandomness)
        );
        // A source that fails on the second draw (the secret) fails the whole mint.
        let mut calls = 0;
        let mut flaky = |buf: &mut [u8]| {
            calls += 1;
            if calls == 2 {
                Err(MintError::NoRandomness)
            } else {
                buf.fill(7);
                Ok(())
            }
        };
        assert!(mint_with(Kind::Run, &mut flaky).is_err());
        // The source is used for both the id and the secret, and they are what the token holds.
        let mut counter = 0u8;
        let mut counting = |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                *b = counter;
                counter = counter.wrapping_add(1);
            }
            Ok(())
        };
        let t = mint_with(Kind::Con, &mut counting).unwrap();
        assert_eq!(t.id, "0001020304050607");
        assert!(t.token.starts_with("oaiycon_0001020304050607_"));
    }

    #[test]
    fn a_new_token_never_shows_itself_in_debug_output() {
        let t = build(Kind::Pat, &ID_BYTES, &SECRET_BYTES);
        let shown = format!("{t:?}");
        assert!(
            !shown.contains(SECRET) && !shown.contains(&t.token),
            "{shown}"
        );
        assert!(shown.contains("0123456789abcdef"));
    }

    #[test]
    fn hashes_compare_equal_only_when_every_byte_is() {
        assert!(hashes_equal(HASH, HASH));
        assert!(!hashes_equal(HASH, &HASH.replace('e', "f")));
        assert!(!hashes_equal(HASH, &HASH[..63]));
        assert!(!hashes_equal(HASH, ""));
        assert!(hashes_equal("", ""));
        // The last byte, the first byte and a middle byte each decide.
        for i in [0, 31, 63] {
            let mut other = HASH.to_string().into_bytes();
            other[i] = if other[i] == b'0' { b'1' } else { b'0' };
            assert!(
                !hashes_equal(HASH, &String::from_utf8(other).unwrap()),
                "byte {i}"
            );
        }
        assert!(
            secrets_equal(b"abc", b"abc")
                && !secrets_equal(b"abc", b"abd")
                && !secrets_equal(b"abc", b"ab")
        );
    }

    fn one(v: &[u8]) -> Result<Option<String>, BearerError> {
        bearer_from_headers(&[v]).map(|o| o.map(str::to_string))
    }

    #[test]
    fn a_well_formed_bearer_is_read() {
        assert_eq!(bearer_from_headers(&[]), Ok(None));
        assert_eq!(
            one(format!("Bearer {TOKEN}").as_bytes()),
            Ok(Some(TOKEN.to_string()))
        );
        // A token of the static shape (any of the permitted characters) is a bearer too.
        assert_eq!(
            one(b"Bearer abc.DEF_123~+/=-"),
            Ok(Some("abc.DEF_123~+/=-".to_string()))
        );
        // 128 bytes is the longest.
        let longest = "a".repeat(MAX_BEARER_LEN);
        assert_eq!(
            one(format!("Bearer {longest}").as_bytes()),
            Ok(Some(longest))
        );
    }

    #[test]
    fn a_hostile_authorization_header_is_refused_before_any_lookup() {
        // T11: two headers, 129 bytes, control characters, wrong-case scheme, non-ASCII, `oaiyses_`.
        let good = format!("Bearer {TOKEN}");
        assert_eq!(
            bearer_from_headers(&[good.as_bytes(), good.as_bytes()]),
            Err(BearerError::Multiple)
        );
        assert_eq!(
            bearer_from_headers(&[good.as_bytes(), b"Bearer x"]),
            Err(BearerError::Multiple)
        );
        let too_long = "a".repeat(MAX_BEARER_LEN + 1);
        assert_eq!(
            one(format!("Bearer {too_long}").as_bytes()),
            Err(BearerError::TooLong)
        );
        assert_eq!(one(b"Bearer abc\x00def"), Err(BearerError::BadCharset));
        assert_eq!(one(b"Bearer abc\ndef"), Err(BearerError::BadCharset));
        assert_eq!(one(b"Bearer abc\r\ndef"), Err(BearerError::BadCharset));
        assert_eq!(one(b"Bearer abc\x7fdef"), Err(BearerError::BadCharset));
        assert_eq!(
            one(b"Bearer abc def"),
            Err(BearerError::BadCharset),
            "a space inside"
        );
        assert_eq!(
            one(b"Bearer abc "),
            Err(BearerError::BadCharset),
            "a trailing space"
        );
        assert_eq!(one(b"Bearer abc,def"), Err(BearerError::BadCharset));
        assert_eq!(one(b"Bearer abc\"def"), Err(BearerError::BadCharset));
        assert_eq!(
            one("Bearer caf\u{e9}".as_bytes()),
            Err(BearerError::BadCharset)
        );
        assert_eq!(one(b"Bearer \xff\xfe"), Err(BearerError::BadCharset));
        assert_eq!(one(b"Bearer "), Err(BearerError::BadCharset), "no token");
        for scheme in ["bearer", "BEARER", "Basic", "Token", "Bearer\t", ""] {
            assert_eq!(
                one(format!("{scheme} {TOKEN}").as_bytes()),
                Err(BearerError::NotBearer),
                "{scheme:?}"
            );
        }
        assert_eq!(one(b"Bearer"), Err(BearerError::NotBearer));
        assert_eq!(
            one(TOKEN.as_bytes()),
            Err(BearerError::NotBearer),
            "no scheme"
        );
        assert_eq!(
            one(format!("Bearer  {TOKEN}").as_bytes()),
            Err(BearerError::BadCharset),
            "two spaces"
        );
        let session = format!("Bearer oaiyses_0123456789abcdef_{SECRET}");
        assert_eq!(one(session.as_bytes()), Err(BearerError::CookieOnlyKind));
        let device = format!("Bearer oaiydev_0123456789abcdef_{SECRET}");
        assert_eq!(one(device.as_bytes()), Err(BearerError::CookieOnlyKind));
        // Even a short one that only starts like a session token.
        assert_eq!(one(b"Bearer oaiyses_x"), Err(BearerError::CookieOnlyKind));
    }

    #[test]
    fn a_length_of_128_is_the_line_and_129_is_over_it() {
        let a = "a".repeat(128);
        let b = "a".repeat(129);
        assert!(one(format!("Bearer {a}").as_bytes()).is_ok());
        assert_eq!(
            one(format!("Bearer {b}").as_bytes()),
            Err(BearerError::TooLong)
        );
    }

    #[test]
    fn the_error_messages_carry_no_secret() {
        for e in [
            BearerError::Multiple,
            BearerError::NotBearer,
            BearerError::TooLong,
            BearerError::BadCharset,
            BearerError::CookieOnlyKind,
        ] {
            assert!(!e.message().is_empty());
        }
    }

    /// A splitmix64 stream: seeded, so that a failure is the same failure the next time.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
        /// `len` characters drawn from `alphabet`.
        fn text(&mut self, len: usize, alphabet: &str) -> String {
            let a: Vec<char> = alphabet.chars().collect();
            (0..len).map(|_| a[self.below(a.len())]).collect()
        }
    }

    const PRINTABLE: &str = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    const ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    /// `len` printable characters that the rule takes (the first of a seeded series that it does), for the tests
    /// that need a token and are about something else. Not longer than 256.
    fn wide(len: usize) -> String {
        for seed in 0u64.. {
            let mut rng = Rng(seed * 1000 + len as u64);
            let token = rng.text(len, PRINTABLE);
            if check_static_token_shape(&token).is_ok() {
                return token;
            }
        }
        unreachable!()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn base32(bytes: &[u8]) -> String {
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
        let (mut out, mut bits, mut have) = (String::new(), 0u32, 0u32);
        for b in bytes {
            bits = (bits << 8) | *b as u32;
            have += 8;
            while have >= 5 {
                out.push(alphabet[((bits >> (have - 5)) & 31) as usize] as char);
                have -= 5;
            }
        }
        out
    }

    /// A version 4 UUID, in the form `uuidgen` prints it.
    fn uuid4(rng: &mut Rng) -> String {
        let mut b = rng.bytes(16);
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        let h = hex(&b);
        format!(
            "{}-{}-{}-{}-{}",
            &h[..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..]
        )
    }

    /// What the tools that people make tokens with make, each named for the command.
    fn generators() -> Vec<(&'static str, fn(&mut Rng) -> String)> {
        use base64::Engine;
        vec![
            ("openssl rand -hex 32 (64 hex)", |r| hex(&r.bytes(32))),
            (
                "openssl rand -hex 24 (48 hex, the length of a real token)",
                |r| hex(&r.bytes(24)),
            ),
            ("openssl rand -hex 16 (32 hex)", |r| hex(&r.bytes(16))),
            ("openssl rand -base64 32 (44, padded)", |r| {
                base64::engine::general_purpose::STANDARD.encode(r.bytes(32))
            }),
            ("python secrets.token_urlsafe(32) (43)", |r| {
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r.bytes(32))
            }),
            ("python secrets.token_urlsafe(24) (32)", |r| {
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r.bytes(24))
            }),
            ("uuidgen twice, joined (73)", |r| {
                let first = uuid4(r);
                format!("{first}{}", uuid4(r))
            }),
            ("uuidgen once (36)", |r| uuid4(r)),
            ("uuidgen on macOS, upper case (36)", |r| {
                uuid4(r).to_uppercase()
            }),
            ("base32 of 20 bytes (32)", |r| base32(&r.bytes(20))),
            ("tr -dc A-Za-z0-9 | head -c 32", |r| r.text(32, ALNUM)),
            ("a password manager's 40 printable characters", |r| {
                r.text(40, PRINTABLE)
            }),
        ]
    }

    #[test]
    fn a_static_token_is_of_the_length_of_the_design_or_is_named_for_why_not() {
        assert_eq!(check_static_token_shape(&wide(32)), Ok(()));
        // The two ends of the length, and one over each.
        for (len, want) in [
            (0, Err(StaticTokenShape::TooShort)),
            (31, Err(StaticTokenShape::TooShort)),
            (32, Ok(())),
            (100, Ok(())),
            (256, Ok(())),
        ] {
            let token = if len < 32 {
                "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1"[..len].to_string()
            } else {
                wide(len)
            };
            assert_eq!(check_static_token_shape(&token), want, "{len}");
        }
        for len in [257, 1000] {
            let token = Rng(len as u64).text(len, PRINTABLE);
            assert_eq!(
                check_static_token_shape(&token),
                Err(StaticTokenShape::TooLong),
                "{len}"
            );
        }
        // Every byte: printable ASCII (0x21 to 0x7e) is allowed, and nothing else is.
        for b in 0u32..=0x2FF {
            let Some(c) = char::from_u32(b) else { continue };
            let mut token = "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ".to_string();
            token.push(c);
            let printable_ascii = (0x21..=0x7e).contains(&b);
            // The rule that a character breaks may be another one of the pattern rules (a digit or a letter
            // next to its neighbour) but a byte outside printable ASCII is this one, and only this one.
            let verdict = check_static_token_shape(&token);
            if printable_ascii {
                assert!(
                    verdict != Err(StaticTokenShape::BadCharacter),
                    "U+{b:04X} is printable"
                );
            } else {
                assert_eq!(verdict, Err(StaticTokenShape::BadCharacter), "U+{b:04X}");
            }
        }
        for reason in [
            StaticTokenShape::TooShort,
            StaticTokenShape::TooLong,
            StaticTokenShape::BadCharacter,
            StaticTokenShape::TooFewDistinct,
            StaticTokenShape::Placeholder,
            StaticTokenShape::Sequential,
            StaticTokenShape::Repeats,
            StaticTokenShape::TooLittleEntropy,
        ] {
            assert!(reason.message().starts_with("OAIY_SERVER_TOKEN"));
            assert!(
                !reason.message().contains("random hex"),
                "{reason:?}: a lie now"
            );
        }
    }

    /// What a token that a person made up looks like, and the clause that refuses it. Each is refused by the rule
    /// that is named and by no other that comes before it.
    #[test]
    fn a_static_token_that_is_a_pattern_or_an_example_is_refused_and_named_for_why() {
        use StaticTokenShape::*;
        for (token, why) in [
            // The design's own example, and the owner's habits.
            ("change-me", TooShort),
            ("change-me-change-me-change-me-change-me", Placeholder),
            ("passwordpasswordpasswordpassword12", Placeholder),
            ("verysecrettokenverysecrettoken1234", Placeholder),
            ("integration-test-token-0123456789ABCDEF", Placeholder),
            // The token this test module used to hold up: 40 different characters that count.
            ("abcdefghijklmnopqrstuvwxyz0123456789ABCD", Placeholder),
            ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN", Sequential),
            ("ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqpon", Sequential),
            ("qwertyuiopasdfghjklzxcvbnm1234567890", Placeholder),
            ("qwertyuiopasdfghjklzxcvbnmqwertyui", Sequential),
            ("poiuytrewqlkjhgfdsamnbvcxzpoiuytrew", Sequential),
            ("0123456789abcdef0123456789abcdef", Placeholder),
            ("1234567890123456789012345678901234567890", Placeholder),
            // A few characters over and over, however they are arranged.
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", TooFewDistinct),
            ("abababababababababababababababababababab", TooFewDistinct),
            ("Tr0ub4dor&3Tr0ub4dor&3Tr0ub4dor&3xx", Repeats),
            ("correcthorsebatterystaplecorrecthorse", Repeats),
            ("k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1k7Qz!mV3", Repeats),
            // Too few bits: 32 decimal digits are worth 106, and the same 32 characters as hex are worth 128.
            ("67374834151859291927224391593075", TooLittleEntropy),
        ] {
            assert_eq!(check_static_token_shape(token), Err(why), "{token}");
        }
        // What the same clauses leave alone: no word, no run, no repeat, and the bits.
        for token in [
            "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1",
            // 32 hex digits are 128 bits, exactly enough; 39 decimal digits are 129.
            "f450474c461f635bacde4cc71a4d10a9",
            "402018455530338697140336938258712660836",
            // The base64 that `openssl rand -base64 32` prints, padding and all.
            "zkwdexEjP8A727+Q2moKCafq4F2IVlZbLJOseRU8Ric=",
        ] {
            assert_eq!(check_static_token_shape(token), Ok(()), "{token}");
        }
    }

    /// Each clause of the rule at its edge: one step inside it is a token, and the step over is named for the clause.
    /// Each word of the list, in each case; a run of 8 in the character set and 7; 6 along the keyboard and 5; a
    /// piece of 8 twice and of 7; 8 different characters and 7; 39 decimal digits and 38.
    #[test]
    fn every_clause_of_the_rule_has_its_edge() {
        use StaticTokenShape::*;
        // 32 characters that the rule takes, and a place in the middle to put a piece.
        let base = "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1";
        assert_eq!(check_static_token_shape(base), Ok(()));
        let with = |piece: &str| format!("{}{piece}{}", &base[..12], &base[12..]);
        for word in PLACEHOLDERS {
            let capital = format!("{}{}", word[..1].to_uppercase(), &word[1..]);
            for spelled in [word.to_string(), word.to_uppercase(), capital] {
                assert_eq!(
                    check_static_token_shape(&with(&spelled)),
                    Err(Placeholder),
                    "{spelled}"
                );
            }
            // Not the word: one letter of it changed, or a character in it.
            let broken = format!("{}~{}", &word[..word.len() / 2], &word[word.len() / 2..]);
            assert_eq!(check_static_token_shape(&with(&broken)), Ok(()), "{broken}");
        }
        for (piece, run) in [
            ("ABCDEFGH", true),
            ("ABCDEFG", false),
            ("HGFEDCBA", true),
            ("HGFEDCB", false),
            ("hijklmno", true),
            ("hijklmn", false),
            ("23456789", true),
            ("2345678", false),
            ("9876543", false),
        ] {
            assert_eq!(
                check_static_token_shape(&with(piece)),
                if run { Err(Sequential) } else { Ok(()) },
                "{piece}"
            );
        }
        for (piece, along) in [
            ("qwerty", true),
            ("QWERTY", true),
            ("poiuyt", true),
            ("asdfgh", true),
            ("lkjhgf", true),
            ("zxcvbn", true),
            ("mnbvcx", true),
            ("qwert", false),
            ("poiuy", false),
            ("asdfg", false),
            ("zxcvb", false),
            // Two rows are not a row.
            ("opasdf", false),
            ("plkjhg", false),
        ] {
            assert_eq!(
                check_static_token_shape(&with(piece)),
                if along { Err(Sequential) } else { Ok(()) },
                "{piece}"
            );
        }
        // A piece of 8 that occurs twice, and of 7 and then a different character.
        assert_eq!(
            check_static_token_shape(&format!("{base}{}", &base[..8])),
            Err(Repeats)
        );
        assert_eq!(
            check_static_token_shape(&format!("{base}{}~", &base[..7])),
            Ok(())
        );
        // The same piece twice with something between.
        assert_eq!(
            check_static_token_shape(&format!("{base}!!{}", &base[4..12])),
            Err(Repeats)
        );
        // 8 different characters in 40 that no piece repeats in, and 7.
        for (alphabet, want) in [("kQzmVpWx", Ok(())), ("kQzmVpW", Err(TooFewDistinct))] {
            // (Of 40 characters drawn from 8, one time in 25 misses one, and is 7: the first that is taken.)
            let token = (0u64..)
                .map(|seed| Rng(seed).text(40, alphabet))
                .find(|t| want.is_err() || check_static_token_shape(t).is_ok())
                .unwrap();
            assert_eq!(
                check_static_token_shape(&token),
                want,
                "{alphabet}: {token}"
            );
        }
        // 39 decimal digits are worth 129 bits and 38 are worth 126; 32 hex digits are worth 128 and 31 are short.
        for (digits, want) in [(38, Err(TooLittleEntropy)), (39, Ok(()))] {
            let token = (0u64..)
                .map(|seed| Rng(seed).text(digits, "0123456789"))
                .find(|t| {
                    let verdict = check_static_token_shape(t);
                    verdict == Ok(()) || verdict == Err(TooLittleEntropy)
                })
                .unwrap();
            assert_eq!(check_static_token_shape(&token), want, "{token}");
        }
        // The edge of every alphabet the estimate tells apart: what 32 characters of it are worth. Hex of one case
        // is the one whose 32 characters are exactly enough; two cases of it are letters and digits.
        let hex32 = "f450474c461f635bacde4cc71a4d10a9";
        assert_eq!(check_static_token_shape(hex32), Ok(()));
        assert_eq!(check_static_token_shape(&hex32.to_uppercase()), Ok(()));
        assert_eq!(check_static_token_shape(&hex32[..31]), Err(TooShort));
    }

    /// The owner's own token has 48 characters and 15 different ones, and the rule the design gave the static
    /// token refused it (the server would not start, and a desktop moved to `scoped` dropped it with a line in
    /// its log). It is not their token in this test, only one of its kind.
    #[test]
    fn a_hex_token_of_48_characters_that_misses_one_digit_is_a_token_and_was_refused() {
        // 48 hex digits made with no `e`: 15 different characters, 192 bits.
        let token = "d812d74ba4c163bbf58d6fa2605b596946c9d50880c35500";
        let mut seen: Vec<char> = token.chars().collect();
        seen.sort();
        seen.dedup();
        assert_eq!((token.len(), seen.len()), (48, 15));
        // The design's rule was 32 to 256 printable characters, at least 16 different: it refused this.
        assert!(seen.len() < 16);
        assert_eq!(check_static_token_shape(token), Ok(()));
        // ... and it took this, which is 40 different characters that count up and are made of the word for
        // one of the examples.
        assert!(check_static_token_shape("abcdefghijklmnopqrstuvwxyz0123456789ABCD").is_err());
    }

    /// How often each of the tools that people make tokens with makes one that the rule takes. The design's rule
    /// took about one `openssl rand -hex 24` in two, one `uuidgen` in three and one `-hex 16` in fourteen. The
    /// words and runs the new rule looks for turn away about one hex token in five hundred, by chance and no more.
    #[test]
    fn what_the_tools_that_make_tokens_make_is_taken_nearly_every_time() {
        const N: usize = 10_000;
        for (name, make) in generators() {
            let mut rng = Rng(0x0A1D_5EED ^ name.len() as u64);
            let mut taken = 0;
            let mut first_refused = None;
            for _ in 0..N {
                let token = make(&mut rng);
                match check_static_token_shape(&token) {
                    Ok(()) => taken += 1,
                    Err(why) => {
                        first_refused.get_or_insert((token, why));
                    }
                }
            }
            eprintln!("{name}: {:.2}% taken", 100.0 * taken as f64 / N as f64);
            assert!(
                taken * 1000 >= N * 995,
                "{name}: {taken} of {N} taken; the first one refused: {first_refused:?}"
            );
        }
    }

    /// A second implementation of the rule, written from its description and not from the code: characters and
    /// strings where the rule reads bytes, `f64` where it counts thousandths, a search for every piece where it
    /// keeps a set.
    fn reference_takes(token: &str) -> bool {
        let chars: Vec<char> = token.chars().collect();
        let n = chars.len();
        if n < 32 || n > 256 || chars.iter().any(|c| !('!'..='~').contains(c)) {
            return false;
        }
        let mut different = chars.clone();
        different.sort_unstable();
        different.dedup();
        if different.len() < 8 {
            return false;
        }
        let lower: String = token.to_lowercase();
        for word in [
            "change-me",
            "changeme",
            "password",
            "secret",
            "token",
            "example",
            "test",
            "admin",
            "default",
            "xxxx",
            "0000",
            "1234",
        ] {
            if lower.contains(word) {
                return false;
            }
        }
        // Each character one step from the one before, eight in a row.
        let mut run = 1;
        for i in 1..n {
            if (chars[i] as i32 - chars[i - 1] as i32).abs() == 1 {
                run += 1;
                if run >= 8 {
                    return false;
                }
            } else {
                run = 1;
            }
        }
        // Six in a row along a row of the keyboard, either way.
        let lower_chars: Vec<char> = lower.chars().collect();
        for row in ["qwertyuiop", "asdfghjkl", "zxcvbnm"] {
            let forward: Vec<char> = row.chars().collect();
            let backward: Vec<char> = row.chars().rev().collect();
            for start in 0..=(n - 6) {
                let piece = &lower_chars[start..start + 6];
                for line in [&forward, &backward] {
                    if (0..=(line.len() - 6)).any(|s| &line[s..s + 6] == piece) {
                        return false;
                    }
                }
            }
        }
        // A piece of eight that occurs at two places.
        for a in 0..=(n - 8) {
            for b in (a + 1)..=(n - 8) {
                if chars[a..a + 8] == chars[b..b + 8] {
                    return false;
                }
            }
        }
        // The alphabet: what is left when the `=` at the end is off.
        let mut end = n;
        while end > 0 && chars[end - 1] == '=' {
            end -= 1;
        }
        let body = &chars[..end];
        let padded = end < n;
        let all = |f: &dyn Fn(char) -> bool| !body.is_empty() && body.iter().all(|c| f(*c));
        let size: f64 = if !padded && all(&|c| c.is_ascii_digit()) {
            10.0
        } else if !padded
            && (all(&|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
                || all(&|c| c.is_ascii_digit() || ('A'..='F').contains(&c)))
        {
            16.0
        } else if all(&|c| c.is_ascii_uppercase() || ('2'..='7').contains(&c)) {
            32.0
        } else if !padded && all(&|c| c.is_ascii_alphanumeric()) {
            62.0
        } else if all(&|c| c.is_ascii_alphanumeric() || c == '+' || c == '/')
            || (!padded && all(&|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        {
            64.0
        } else {
            94.0
        };
        // Whole thousandths of a bit, as the rule counts them, so that the two agree at the edge.
        (n as f64 * (size.log2() * 1000.0).floor()) >= 128_000.0
    }

    /// A token that a generator made, and then what a person might do to it.
    fn tampered(rng: &mut Rng, gens: &[(&'static str, fn(&mut Rng) -> String)]) -> String {
        let mut t = (gens[rng.below(gens.len())].1)(rng);
        let pieces = [
            "change-me",
            "Password",
            "SECRET",
            "token",
            "Example",
            "test",
            "admin",
            "DEFAULT",
            "xxxx",
            "0000",
            "1234",
            "abcdefgh",
            "HGFEDCBA",
            "12345678",
            "87654321",
            "qwerty",
            "POIUYT",
            "asdfgh",
            "mnbvcx",
            "AbCdEfGhIj",
        ];
        for _ in 0..rng.below(3) {
            let step = rng.below(9);
            match step {
                0 => {
                    let cut = 8 + rng.below(t.len().max(9) - 8);
                    t.truncate(cut);
                }
                1 => {
                    let more = 1 + rng.below(6);
                    t.push_str(&rng.text(more, PRINTABLE));
                }
                2 => {
                    let at = rng.below(t.len() + 1);
                    t.insert_str(at, pieces[rng.below(pieces.len())]);
                }
                3 => {
                    if t.len() > 16 {
                        let a = rng.below(t.len() - 8);
                        let piece =
                            t[a..a + 8 + rng.below((t.len() - a - 8).min(8) + 1)].to_string();
                        t.push_str(&piece);
                    }
                }
                4 => t = t.to_uppercase(),
                5 => t = t.to_lowercase(),
                6 | 7 => {
                    let len = 30 + rng.below(30);
                    t = rng.text(len, if step == 6 { "0123456789" } else { "abcd" });
                }
                _ => t.push_str(&"=".repeat(rng.below(4))),
            }
        }
        t
    }

    /// The rule and a second implementation of it agree on 30,000 tokens made and mistreated as tokens are, and the
    /// corpus holds enough of both answers for that to mean something.
    #[test]
    fn the_rule_and_a_second_implementation_of_it_agree_on_30000_tokens() {
        let gens = generators();
        let mut rng = Rng(0x7047_0C0D_E5EE_D001);
        let (mut taken, mut refused) = (0, 0);
        let mut by_reason: std::collections::BTreeMap<String, usize> = Default::default();
        for i in 0..30_000 {
            let token = tampered(&mut rng, &gens);
            let verdict = check_static_token_shape(&token);
            assert_eq!(
                verdict.is_ok(),
                reference_takes(&token),
                "case {i}: {token:?}: the rule says {verdict:?}"
            );
            match verdict {
                Ok(()) => taken += 1,
                Err(why) => {
                    refused += 1;
                    *by_reason.entry(format!("{why:?}")).or_default() += 1;
                }
            }
        }
        eprintln!("30000 tokens: {taken} taken, {refused} refused: {by_reason:?}");
        assert!(
            taken > 5_000 && refused > 5_000,
            "{taken} taken, {refused} refused"
        );
        for reason in [
            "TooShort",
            "TooFewDistinct",
            "Placeholder",
            "Sequential",
            "Repeats",
            "TooLittleEntropy",
            "BadCharacter",
        ] {
            // (a token that a generator made has no bad character: the corpus does not test that one)
            if reason != "BadCharacter" {
                assert!(
                    by_reason.get(reason).copied().unwrap_or(0) > 50,
                    "{reason}: {by_reason:?}"
                );
            }
        }
    }

    #[test]
    fn the_static_token_is_a_bearer_in_the_shape_the_design_gives_it() {
        // Outside the strict bearer rule (`[A-Za-z0-9._~+/=-]`, 128 bytes) and inside the static token's
        // (`[\x21-\x7e]{32,256}`): a `$` and a `!`, 129 and 256 characters, every punctuation mark.
        let with_dollar = "Sup3r$ecret!Zq7kLm9VbNw2XyHdFg5!ab".to_string();
        let long_129: String = wide(129);
        let long_256: String = wide(256);
        let marks: String = format!("{}\"'(){{}}[]<>|^`,;:@#%&*?\\", wide(32));
        for token in [with_dollar, long_129, long_256, marks] {
            assert_eq!(check_static_token_shape(&token), Ok(()), "{token}");
            let header = format!("Bearer {token}");
            let values: &[&[u8]] = &[header.as_bytes()];
            assert!(
                bearer_from_headers(values).is_err(),
                "the strict rule alone refuses it: {token}"
            );
            assert_eq!(
                bearer_or_static(values, Some(&token)),
                Ok(Some(token.as_str())),
                "{token}"
            );
            // Configured with nothing, or another token of the same shape: the strict rule's refusal.
            let other = wide(40);
            for configured in [None, Some(other.as_str())] {
                assert!(
                    matches!(
                        bearer_or_static(values, configured),
                        Err(BearerError::TooLong | BearerError::BadCharset)
                    ),
                    "{token} against {configured:?}"
                );
            }
        }
    }

    #[test]
    fn nothing_but_the_exact_static_token_gets_past_the_strict_bearer_rule() {
        let token = "Sup3r$ecret!Zq7kLm9VbNw2XyHdFg5!ab";
        let one = |presented: &str| {
            let header = format!("Bearer {presented}");
            bearer_or_static(&[header.as_bytes()], Some(token)).map(|o| o.map(str::to_string))
        };
        assert_eq!(one(token), Ok(Some(token.to_string())));
        // One character more, one less, one changed, another case: not the token.
        for near in [
            format!("{token}x"),
            token[..token.len() - 1].to_string(),
            token.replace("Sup3r", "sup3r"),
            token.replace("ab", "ac"),
            token.replace('$', "%"),
        ] {
            assert_eq!(one(&near), Err(BearerError::BadCharset), "{near}");
        }
        // A wide token of 129 bytes that is not the static token.
        assert_eq!(one(&"$".repeat(129)), Err(BearerError::TooLong));
        // What the static token can never contain is refused, configured or not.
        for hostile in ["a b", "a\tb", "a\u{7f}b", "caf\u{e9}", "a\nb", "a\rb", ""] {
            assert_eq!(one(hostile), Err(BearerError::BadCharset), "{hostile:?}");
        }
        // The other rules stand: two headers, another scheme, the scheme in another case, a session token.
        let good = format!("Bearer {token}");
        assert_eq!(
            bearer_or_static(&[good.as_bytes(), good.as_bytes()], Some(token)),
            Err(BearerError::Multiple)
        );
        assert_eq!(
            bearer_or_static(&[format!("bearer {token}").as_bytes()], Some(token)),
            Err(BearerError::NotBearer)
        );
        assert_eq!(bearer_or_static(&[], Some(token)), Ok(None));
        // A session token never travels as a bearer, even if the operator configured one (32 characters).
        let session_like = "oaiyses_Zq7kLm9VbNw2XyHdFg5Sup3r";
        assert_eq!(check_static_token_shape(session_like), Ok(()));
        let header = format!("Bearer {session_like}");
        assert_eq!(
            bearer_or_static(&[header.as_bytes()], Some(session_like)),
            Err(BearerError::CookieOnlyKind)
        );
        // A token that fails the shape rule is not widened: it is what the strict rule says it is.
        let short = "ab$cd";
        assert_eq!(
            bearer_or_static(&[b"Bearer ab$cd"], Some(short)),
            Err(BearerError::BadCharset)
        );
        let too_long = format!("{}$", wide(256));
        let header = format!("Bearer {too_long}");
        assert_eq!(
            bearer_or_static(&[header.as_bytes()], Some(&too_long)),
            Err(BearerError::TooLong)
        );
        // A token inside the strict rule is a bearer as before, configured or not.
        assert_eq!(
            bearer_or_static(&[b"Bearer abc.DEF_123~+/=-"], None),
            Ok(Some("abc.DEF_123~+/=-"))
        );
    }
}
