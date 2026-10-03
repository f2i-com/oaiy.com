//! The arithmetic of pairing v3 (README section 10.1): the secret, the typed code, the derivations, the MACs, the short authentication string and the approval receipt.
//!
//! **Raw bytes, never their text** (README section 1 and Interpretation 21). A value that has a text spelling enters SHA-256, HMAC, a signature or HKDF as the bytes the
//! text decodes to: the `pid`, the nonce and the keys are bytes here and are written as base64url only where a URL or a JSON string needs them. The one test that matters
//! most is the SAS: `info = "oaiy/pairing/3/sas" || 0x00 || pid` has the 16 raw bytes of the `pid` (35 bytes), not the 22 characters of its text (a 41-byte `info` and the
//! wrong SAS `24b574fd2e0d1e24` of `extras.sasNegative`). Everything in this file is tested against vector A3 and against that negative.

use oaiy_crypto::kdf::{hkdf_sha256, hkdf_sha256_secret, hmac_sha256, hmac_sha256_verify, sha256};
use oaiy_crypto::zeroize::{ct_eq, Secret};
use zeroize::Zeroizing;

use crate::b64;
use crate::error::{Error, Result};
use crate::json::Json;
use crate::keys::{domain_message, signature_from_b64u, Signer, VerifyKey};
use crate::url::{percent_decode, percent_encode, RelayUrl};

/// The HKDF salt of pairing: the UTF-8 string `oaiy/pairing/3`.
pub const HKDF_SALT: &[u8] = b"oaiy/pairing/3";
/// The domain of the offer's MAC.
pub const OFFER_MAC_DOMAIN: &str = "oaiy/pairing/3/offer-mac";
/// The domain of the response's MAC.
pub const RESPONSE_MAC_DOMAIN: &str = "oaiy/pairing/3/response-mac";
/// The domain of the typed code's check characters.
pub const TYPED_DOMAIN: &str = "oaiy/pairing/3/typed";
/// The domain (and the HKDF `info` prefix) of the short authentication string.
pub const SAS_DOMAIN: &str = "oaiy/pairing/3/sas";
/// The domain of the SAS's check character.
pub const SAS_CHECK_DOMAIN: &str = "oaiy/pairing/3/sas-check";
/// A pairing key is at most this long (README 10.1).
pub const MAX_URI_LEN: usize = 512;

/// Crockford's base32 alphabet (no `I`, `L`, `O` or `U`).
pub const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// 16 random bytes used once and never sent to the relay: the whole of a pairing's secrecy. Wiped when dropped, printed nowhere.
pub struct PairingSecret(Secret<16>);

/// What the secret derives.
pub struct Derived {
    /// The rendezvous id: 16 raw bytes (`HKDF(info = "rendezvous", L = 16)`).
    pub pid: [u8; 16],
    /// The MAC key: `HKDF(info = "mac", L = 32)`, which never leaves the two endpoints.
    pub mac_key: Secret<32>,
}

impl PairingSecret {
    /// Wraps 16 bytes.
    pub fn new(bytes: [u8; 16]) -> PairingSecret {
        PairingSecret(Secret::new(bytes))
    }

    /// From the OS random generator.
    pub fn generate() -> Result<PairingSecret> {
        Ok(PairingSecret(Secret::<16>::random()?))
    }

    /// The 16 bytes (for the QR's `s` and the typed code only).
    pub fn expose(&self) -> &[u8; 16] {
        self.0.expose()
    }

    /// The `pid` and the MAC key.
    pub fn derive(&self) -> Result<Derived> {
        let mut pid = [0u8; 16];
        hkdf_sha256(self.0.expose(), Some(HKDF_SALT), b"rendezvous", &mut pid)?;
        let mac_key: Secret<32> = hkdf_sha256_secret(self.0.expose(), Some(HKDF_SALT), b"mac")?;
        Ok(Derived { pid, mac_key })
    }

    /// The pairing key's `s`: 22 characters.
    pub fn b64u(&self) -> String {
        b64::encode(self.0.expose())
    }

    /// The typed code of 28 characters in seven groups.
    pub fn typed_code(&self) -> String {
        typed_code(self.0.expose())
    }
}

impl core::fmt::Debug for PairingSecret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PairingSecret(redacted)")
    }
}

/// The `pid` as it is written in a URL and in JSON: 22 characters of base64url of the 16 raw bytes.
pub fn pid_text(pid: &[u8; 16]) -> String {
    b64::encode(pid)
}

/// Writes `bits` (most significant first) as Crockford characters, five bits each; the last group is padded with zero bits.
fn crockford_encode(bytes: &[u8], bit_count: usize) -> String {
    let mut out = String::with_capacity(bit_count.div_ceil(5));
    crockford_push(&mut out, bytes, bit_count);
    out
}

/// [`crockford_encode`] onto the end of `out` (a buffer the caller wipes, for a secret: no temporary holds a copy).
fn crockford_push(out: &mut String, bytes: &[u8], bit_count: usize) {
    let mut pos = 0;
    while pos < bit_count {
        let mut v = 0u8;
        for i in 0..5 {
            let bit = pos + i;
            let b = if bit < bit_count { (bytes[bit / 8] >> (7 - bit % 8)) & 1 } else { 0 };
            v = (v << 1) | b;
        }
        out.push(char::from(CROCKFORD[usize::from(v)]));
        pos += 5;
    }
}

/// The 5-bit value of a Crockford character (already normalised: upper case, `I`/`L`/`O` mapped), `None` outside the alphabet.
fn crockford_value(c: char) -> Option<u8> {
    CROCKFORD.iter().position(|&a| char::from(a) == c).map(|i| i as u8)
}

/// Upper-cases, reads `I` and `L` as `1` and `O` as `0`, drops dashes and spaces, and refuses `U` and anything else outside the alphabet. `None` is "not a code". The result
/// is wiped when it is dropped: it is a pairing secret or the SAS the owner types.
pub fn normalise(text: &str) -> Option<Zeroizing<String>> {
    let mut out = Zeroizing::new(String::with_capacity(text.len()));
    for c in text.chars() {
        let c = match c.to_ascii_uppercase() {
            '-' | ' ' => continue,
            'I' | 'L' => '1',
            'O' => '0',
            c => c,
        };
        crockford_value(c)?;
        out.push(c);
    }
    Some(out)
}

fn typed_check(secret: &[u8; 16]) -> Zeroizing<String> {
    // The message holds the secret and the digest is a function of it: both are wiped.
    let message = Zeroizing::new(domain_message(TYPED_DOMAIN, &[secret]));
    let digest = Zeroizing::new(sha256(&message));
    Zeroizing::new(crockford_encode(&*digest, 10))
}

/// The typed code of `secret`: 26 characters carrying its 128 bits (the last two bits zero) and 2 check characters (the first 10 bits of
/// `SHA-256("oaiy/pairing/3/typed" || 0x00 || s)`), in seven groups of four separated by dashes.
pub fn typed_code(secret: &[u8; 16]) -> String {
    let mut chars = Zeroizing::new(String::with_capacity(28));
    crockford_push(&mut chars, secret, 128);
    chars.push_str(&typed_check(secret));
    let mut out = String::with_capacity(34);
    for (i, c) in chars.chars().enumerate() {
        if i > 0 && i % 4 == 0 {
            out.push('-');
        }
        out.push(c);
    }
    out
}

/// Reads a typed code: normalised as [`normalise`] says, 28 characters, the last two bits of the 26th character zero, and the check characters verified. This is made
/// locally before any network call, so a typo costs nothing. `Err` is "not a pairing code" and says no more than that.
pub fn parse_typed_code(text: &str) -> Result<PairingSecret> {
    let chars = normalise(text).ok_or(Error::Invalid("typed code: characters"))?;
    // (`bits` and `secret` are stack arrays; every heap copy is wiped.)
    if chars.len() != 28 {
        return Err(Error::Invalid("typed code: length"));
    }
    let mut bits = [0u8; 17];
    for (i, c) in chars[..26].chars().enumerate() {
        let v = crockford_value(c).ok_or(Error::Invalid("typed code: characters"))?;
        for j in 0..5 {
            let bit = i * 5 + j;
            if (v >> (4 - j)) & 1 == 1 {
                bits[bit / 8] |= 1 << (7 - bit % 8);
            }
        }
    }
    // 26 characters carry 130 bits: the last two are zero in a code this crate writes.
    if bits[16] != 0 {
        return Err(Error::Invalid("typed code: trailing bits"));
    }
    let mut secret = [0u8; 16];
    secret.copy_from_slice(&bits[..16]);
    if !ct_eq(typed_check(&secret).as_bytes(), &chars.as_bytes()[26..]) {
        return Err(Error::Invalid("typed code: check"));
    }
    Ok(PairingSecret::new(secret))
}

/// The short authentication string (README 10.1): computed from the two Ed25519 endpoint keys, the offer's nonce and the `pid`, all as raw bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct Sas {
    /// `HKDF-SHA256(IKM = desktopEd25519Pub || phoneEd25519Pub, salt = nonce, info = "oaiy/pairing/3/sas" || 0x00 || pid, L = 8)`.
    raw: [u8; 8],
    /// The top 60 bits of `raw` as 12 Crockford characters.
    chars12: String,
    /// The check character: `Crockford32[SHA-256("oaiy/pairing/3/sas-check" || 0x00 || chars12)[0] >> 3]`.
    check: char,
}

/// The code the owner is shown and types is what makes a pairing the owner's: its `Debug` prints nothing of it.
impl core::fmt::Debug for Sas {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Sas(redacted)")
    }
}

impl Drop for Sas {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.raw.zeroize();
        self.chars12.zeroize();
    }
}

impl Sas {
    /// A value from its parts (a test, or a host that keeps the parts): the one [`sas`] makes is the only one that means anything for a pairing, and this does not check that the parts
    /// belong together. The fields are private, so that nothing can change a value that was made.
    pub fn from_parts(raw: [u8; 8], chars12: impl Into<String>, check: char) -> Sas {
        Sas { raw, chars12: chars12.into(), check }
    }

    /// The 8 bytes the SAS is made of.
    pub fn raw(&self) -> &[u8; 8] {
        &self.raw
    }

    /// The 12 characters (without the check character, without dashes).
    pub fn chars12(&self) -> &str {
        &self.chars12
    }

    /// The check character.
    pub fn check(&self) -> char {
        self.check
    }

    /// `XXXX-XXXX-XXXX-C`, as the phone shows it. A value that [`sas`] made has 12 characters; for any other ([`Sas::from_parts`] takes what it is given) the groups are whatever is there,
    /// and this never panics.
    pub fn display(&self) -> String {
        let part = |from: usize, to: usize| self.chars12.get(from..to).or_else(|| self.chars12.get(from..)).unwrap_or("");
        format!("{}-{}-{}-{}", part(0, 4), part(4, 8), part(8, 12), self.check)
    }
}

/// The check character of 12 SAS characters.
pub fn sas_check_char(chars12: &str) -> char {
    let digest = sha256(&domain_message(SAS_CHECK_DOMAIN, &[chars12.as_bytes()]));
    char::from(CROCKFORD[usize::from(digest[0] >> 3)])
}

/// Computes the SAS. `pid` is the **16 raw bytes**, `nonce` the 32 raw bytes of `offer.nonce`, the keys the 32 raw bytes of the two endpoint keys.
pub fn sas(desktop_endpoint: &[u8; 32], phone_endpoint: &[u8; 32], nonce: &[u8; 32], pid: &[u8; 16]) -> Result<Sas> {
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(desktop_endpoint);
    ikm[32..].copy_from_slice(phone_endpoint);
    let info = domain_message(SAS_DOMAIN, &[pid]);
    let mut raw = [0u8; 8];
    hkdf_sha256(&ikm, Some(nonce), &info, &mut raw)?;
    let chars12 = crockford_encode(&raw, 60);
    let check = sas_check_char(&chars12);
    Ok(Sas { raw, chars12, check })
}

/// What the owner's entry of the SAS on the desktop was, and whether it counts as an attempt (README 10.1, step 6: "three wrong 12-character entries deny the pending item
/// and burn the rendezvous; an incomplete entry or a wrong check character is not an attempt").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SasEntry {
    /// Fewer than 13 characters: still being typed. Not an attempt.
    Incomplete,
    /// Not made of the alphabet, or more than 13 characters. Not an attempt (it cannot be a code at all).
    Invalid,
    /// 13 characters whose check character does not verify: a mistyped character, caught locally. Not an attempt.
    BadCheck,
    /// 13 characters with a good check character that are not this SAS: **one of the three attempts**.
    Wrong,
    /// The SAS.
    Right,
}

impl SasEntry {
    /// True only for [`SasEntry::Wrong`].
    pub fn counts_as_attempt(self) -> bool {
        self == SasEntry::Wrong
    }
}

/// Judges what the owner typed against the SAS the desktop computed.
pub fn judge_sas_entry(expected: &Sas, typed: &str) -> SasEntry {
    let Some(chars) = normalise(typed) else {
        return SasEntry::Invalid;
    };
    match chars.len() {
        0..=12 => SasEntry::Incomplete,
        13 => {
            let (twelve, check) = chars.split_at(12);
            if !check.starts_with(sas_check_char(twelve)) {
                SasEntry::BadCheck
            } else if ct_eq(twelve.as_bytes(), expected.chars12.as_bytes()) {
                SasEntry::Right
            } else {
                SasEntry::Wrong
            }
        }
        _ => SasEntry::Invalid,
    }
}

/// `b64u(HMAC-SHA256(mac_key, "oaiy/pairing/3/offer-mac" || 0x00 || offer text bytes))`: the offer's MAC, over the text exactly as stored.
pub fn offer_mac(mac_key: &Secret<32>, offer_text: &str) -> Result<String> {
    Ok(b64::encode(&hmac_sha256(mac_key.expose(), &domain_message(OFFER_MAC_DOMAIN, &[offer_text.as_bytes()]))?))
}

/// Verifies an offer's MAC in constant time.
pub fn verify_offer_mac(mac_key: &Secret<32>, offer_text: &str, mac: &str) -> Result<()> {
    let tag = b64::decode_exact::<32>(mac)?;
    if hmac_sha256_verify(mac_key.expose(), &domain_message(OFFER_MAC_DOMAIN, &[offer_text.as_bytes()]), &tag) {
        Ok(())
    } else {
        Err(Error::BadMac("pairing offer"))
    }
}

/// `b64u(HMAC-SHA256(mac_key, "oaiy/pairing/3/response-mac" || 0x00 || canonical(claims)))`.
pub fn response_mac(mac_key: &Secret<32>, canonical_claims: &str) -> Result<String> {
    Ok(b64::encode(&hmac_sha256(mac_key.expose(), &domain_message(RESPONSE_MAC_DOMAIN, &[canonical_claims.as_bytes()]))?))
}

/// Verifies a response's MAC in constant time.
pub fn verify_response_mac(mac_key: &Secret<32>, canonical_claims: &str, mac: &str) -> Result<()> {
    let tag = b64::decode_exact::<32>(mac)?;
    if hmac_sha256_verify(mac_key.expose(), &domain_message(RESPONSE_MAC_DOMAIN, &[canonical_claims.as_bytes()]), &tag) {
        Ok(())
    } else {
        Err(Error::BadMac("pairing response"))
    }
}

/// The approval receipt's document, in canonical form: `{"appId","grants" (sorted),"issuedAt","phoneThumbprint","pid"}`, where `pid` is the 22-character text (a JSON
/// string holds text, so this is the one place the `pid` is not its raw bytes).
pub fn receipt_text(app_id: &str, grants: &[String], issued_at: u64, phone_thumbprint: &str, pid_text: &str) -> Result<String> {
    let mut sorted: Vec<&String> = grants.iter().collect();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    Ok(Json::obj([
        ("appId", Json::str(app_id)),
        ("grants", Json::Arr(sorted.into_iter().map(|g| Json::str(g.clone())).collect())),
        ("issuedAt", Json::int(issued_at)),
        ("phoneThumbprint", Json::str(phone_thumbprint)),
        ("pid", Json::str(pid_text)),
    ])
    .to_canonical()?)
}

/// Signs the approval receipt with the desktop's endpoint key; returns the signature as base64url.
pub fn sign_receipt(
    desktop_endpoint: &Signer,
    app_id: &str,
    grants: &[String],
    issued_at: u64,
    phone_thumbprint: &str,
    pid_text: &str,
) -> Result<String> {
    let text = receipt_text(app_id, grants, issued_at, phone_thumbprint, pid_text)?;
    Ok(desktop_endpoint.sign_b64u(crate::keys::SignDomain::PairingApproval, &[text.as_bytes()]))
}

/// Verifies an approval receipt with the desktop endpoint key pinned from the MAC-verified offer. The grants are an explicit input: the receipt signs them, and the
/// relay's `GET /v1/pair/{pid}` does not carry them (see the README of this crate: the gap this leaves in the contract).
pub fn verify_receipt(
    desktop_endpoint: &VerifyKey,
    app_id: &str,
    grants: &[String],
    issued_at: u64,
    phone_thumbprint: &str,
    pid_text: &str,
    signature: &str,
) -> Result<()> {
    let text = receipt_text(app_id, grants, issued_at, phone_thumbprint, pid_text)?;
    let sig = signature_from_b64u(signature)?;
    desktop_endpoint.verify(crate::keys::SignDomain::PairingApproval, &[text.as_bytes()], &sig)
}

/// A parsed pairing key (`oaiy://pair?v=3&u=...&f=...&s=...&x=...`).
pub struct PairingKey {
    /// The relay's base URL.
    pub relay: RelayUrl,
    /// The thumbprint the relay's key must have, when the key carries it (`f`; the typed route has none).
    pub relay_thumbprint: Option<String>,
    /// The pairing secret.
    pub secret: PairingSecret,
    /// When the key expires, Unix seconds (`x`), when it says.
    pub expires_at: Option<u64>,
}

impl core::fmt::Debug for PairingKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PairingKey({}, redacted)", self.relay)
    }
}

impl PairingKey {
    /// Parses a pairing key: the scheme and host exactly `oaiy://pair`, `v` exactly `3`, `u` a relay URL, `s` exactly 22 characters decoding to 16 bytes, `f` and `x`
    /// optional, unknown parameters ignored, a repeated parameter refused, at most 512 characters in all. Scanning or opening one starts nothing by itself: the caller shows
    /// a confirmation (README 10.1).
    pub fn parse(uri: &str) -> Result<PairingKey> {
        if uri.len() > MAX_URI_LEN {
            return Err(Error::Uri("pairing key: too long"));
        }
        let query = uri.strip_prefix("oaiy://pair?").ok_or(Error::Uri("pairing key: scheme or host"))?;
        let (mut v, mut u, mut f, mut s, mut x) = (None, None, None, None, None);
        for pair in query.split('&') {
            let (name, value) = pair.split_once('=').ok_or(Error::Uri("pairing key: a parameter has no value"))?;
            let slot = match name {
                "v" => &mut v,
                "u" => &mut u,
                "f" => &mut f,
                "s" => &mut s,
                "x" => &mut x,
                _ => continue,
            };
            if slot.replace(value).is_some() {
                return Err(Error::Uri("pairing key: a parameter is repeated"));
            }
        }
        if v != Some("3") {
            return Err(Error::Uri("pairing key: v must be 3"));
        }
        let relay = RelayUrl::parse(&percent_decode(u.ok_or(Error::Uri("pairing key: u"))?)?)?;
        let secret = PairingSecret::new(b64::decode_exact::<16>(s.ok_or(Error::Uri("pairing key: s"))?).map_err(|_| Error::Uri("pairing key: s"))?);
        let relay_thumbprint = match f {
            Some(t) if crate::ids::is_thumbprint(t) => Some(t.to_string()),
            Some(_) => return Err(Error::Uri("pairing key: f")),
            None => None,
        };
        let expires_at = match x {
            Some(t) if !t.is_empty() && t.len() <= 16 && t.bytes().all(|b| b.is_ascii_digit()) && (t.len() == 1 || !t.starts_with('0')) => {
                Some(t.parse().map_err(|_| Error::Uri("pairing key: x"))?)
            }
            Some(_) => return Err(Error::Uri("pairing key: x")),
            None => None,
        };
        Ok(PairingKey { relay, relay_thumbprint, secret, expires_at })
    }

    /// Writes a pairing key in the canonical order `v, u, f, s, x`.
    pub fn to_uri(relay: &RelayUrl, relay_thumbprint: &str, secret: &PairingSecret, expires_at: u64) -> String {
        use core::fmt::Write as _;
        // In a buffer sized for the longest key a reader takes (it is never copied by growing), and the text of the secret that goes into it is wiped.
        let secret_text = Zeroizing::new(secret.b64u());
        let mut out = String::with_capacity(MAX_URI_LEN);
        let _ =
            write!(out, "oaiy://pair?v=3&u={}&f={}&s={}&x={}", percent_encode(&relay.origin()), relay_thumbprint, secret_text.as_str(), expires_at);
        out
    }
}
