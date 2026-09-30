//! Key derivation: `crypto_kdf` (design 4.1.1 `kdf`), HKDF-SHA256 (RFC 5869), HMAC-SHA256 and SHA-256.
//!
//! `kdf(id, ctx8, key)` is libsodium's `crypto_kdf_derive_from_key`: BLAKE2b keyed with the 32-byte master key,
//! `salt = LE64(id) || 0x00 * 8`, `personal = ctx || 0x00 * 8`, an empty message, and an output of 16, 32 or 64 bytes. Each
//! purpose has its own eight-character context, listed in [`REGISTRY`] (design 4.1.2, append-only); a test rejects a
//! duplicate context and a context that is not eight characters of `[a-z0-9]`. The typed entry point is
//! [`derive`], which takes a [`Purpose`]; [`derive_subkey`] takes any id and any context and exists for the
//! vectors (FormLogic's second `flrecov1` vector uses subkey id 7) and for a consumer that owns a context of its own.
//!
//! HKDF-SHA256 is the derivation of the ceremony (`oaiy-kt:1`, 4.8) and of the browser device cache (4.4.3); HMAC-SHA256 and
//! SHA-256 are here so that the relay and access code use the same implementation and the same constant-time comparison.

use blake2::digest::consts::{U16, U32, U64};
use blake2::digest::Mac;
use blake2::Blake2bMac;
use hkdf::Hkdf;
use hmac::Hmac;
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use crate::error::Error;
use crate::zeroize::Secret;

/// The eight-character context of a `crypto_kdf` derivation: ASCII `[a-z0-9]`, exactly eight of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context([u8; 8]);

impl Context {
    /// Checks at run time.
    pub fn new(text: &str) -> Result<Context, Error> {
        let bytes = text.as_bytes();
        if bytes.len() != 8 || !bytes.iter().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()) {
            return Err(Error::KdfContext);
        }
        let mut out = [0u8; 8];
        out.copy_from_slice(bytes);
        Ok(Context(out))
    }

    /// Checks at compile time: a malformed constant is a compile error, not a run-time surprise.
    pub const fn from_static(text: &'static str) -> Context {
        let bytes = text.as_bytes();
        if bytes.len() != 8 {
            panic!("a KDF context is exactly eight characters");
        }
        let mut out = [0u8; 8];
        let mut i = 0;
        while i < 8 {
            let b = bytes[i];
            if !(b.is_ascii_lowercase() || b.is_ascii_digit()) {
                panic!("a KDF context is [a-z0-9]");
            }
            out[i] = b;
            i += 1;
        }
        Context(out)
    }

    /// The eight bytes.
    pub const fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// The context as text (it is ASCII by construction).
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.0).unwrap_or("")
    }
}

/// Where a registry row stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// In FormLogic's code today.
    Existing,
    /// Introduced by the vault design.
    New,
    /// Named so that nothing else takes it; not derived by anything yet.
    Reserved,
}

/// One row of the registry of design 4.1.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The context.
    pub context: Context,
    /// The subkey id (1 for every row so far).
    pub id: u64,
    /// What the master key is.
    pub master: &'static str,
    /// What the derived key is for.
    pub purpose: &'static str,
    /// Where the row stands.
    pub status: Status,
}

/// The kit wrap key: master is the FLRK1 kit's 256-bit key.
pub const FLRECOV1: Context = Context::from_static("flrecov1");
/// The phrase wrap key: master is `argon2id13(entropy16, salt16, 3, 64 MiB)`.
pub const FLPHRAS1: Context = Context::from_static("flphras1");
/// A WebAuthn PRF wrapper (reserved).
pub const FLPRF001: Context = Context::from_static("flprf001");
/// The backup recipient secret (an age identity): master is the UMK.
pub const FLBKRCP1: Context = Context::from_static("flbkrcp1");
/// The backup manifest signing seed: master is the UMK.
pub const FLBKSIG1: Context = Context::from_static("flbksig1");
/// The local data key of the deferred sealing wave (reserved): master is the UMK.
pub const FLLOCAL1: Context = Context::from_static("fllocal1");

/// The registry (design 4.1.2). Append-only: a row is never removed or changed, and no context appears twice.
pub const REGISTRY: [Entry; 6] = [
    Entry { context: FLRECOV1, id: 1, master: "FLRK1 kit", purpose: "kit wrap key", status: Status::Existing },
    Entry { context: FLPHRAS1, id: 1, master: "argon2id13(entropy16, salt16, 3, 64 MiB)", purpose: "phrase wrap key", status: Status::New },
    Entry { context: FLPRF001, id: 1, master: "WebAuthn PRF output", purpose: "vault-level PRF wrapper", status: Status::Reserved },
    Entry { context: FLBKRCP1, id: 1, master: "UMK", purpose: "backup recipient secret (age identity)", status: Status::New },
    Entry { context: FLBKSIG1, id: 1, master: "UMK", purpose: "backup manifest signing seed", status: Status::New },
    Entry { context: FLLOCAL1, id: 1, master: "UMK", purpose: "local data key (deferred)", status: Status::Reserved },
];

/// The purposes of [`REGISTRY`], as a type so that a context cannot be misspelt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// `flrecov1`.
    KitWrap,
    /// `flphras1`.
    PhraseWrap,
    /// `flprf001`.
    PrfWrap,
    /// `flbkrcp1`.
    BackupRecipient,
    /// `flbksig1`.
    BackupSigning,
    /// `fllocal1`.
    LocalData,
}

impl Purpose {
    /// The registry row.
    pub const fn entry(self) -> &'static Entry {
        match self {
            Purpose::KitWrap => &REGISTRY[0],
            Purpose::PhraseWrap => &REGISTRY[1],
            Purpose::PrfWrap => &REGISTRY[2],
            Purpose::BackupRecipient => &REGISTRY[3],
            Purpose::BackupSigning => &REGISTRY[4],
            Purpose::LocalData => &REGISTRY[5],
        }
    }
}

/// `kdf(id, ctx8, key)` for a registered purpose: 32 bytes.
pub fn derive(master: &Secret<32>, purpose: Purpose) -> Result<Secret<32>, Error> {
    let entry = purpose.entry();
    derive_subkey(master, entry.id, &entry.context)
}

/// `crypto_kdf_derive_from_key` with a 32-byte output, for any subkey id and any (valid) context.
pub fn derive_subkey(master: &Secret<32>, id: u64, context: &Context) -> Result<Secret<32>, Error> {
    let mut out = [0u8; 32];
    let result = derive_into(master, id, context, &mut out);
    let secret = Secret::new(out);
    out.zeroize();
    result.map(|()| secret)
}

/// `crypto_kdf_derive_from_key` for a 16, 32 or 64 byte output (libsodium allows 16 to 64; nothing here needs the others).
pub fn derive_subkey_into(master: &Secret<32>, id: u64, context: &Context, out: &mut [u8]) -> Result<(), Error> {
    if !matches!(out.len(), 16 | 32 | 64) {
        return Err(Error::KdfContext);
    }
    derive_into(master, id, context, out)
}

fn derive_into(master: &Secret<32>, id: u64, context: &Context, out: &mut [u8]) -> Result<(), Error> {
    let mut salt = [0u8; 16];
    salt[..8].copy_from_slice(&id.to_le_bytes());
    let mut personal = [0u8; 16];
    personal[..8].copy_from_slice(context.as_bytes());
    // The key is 32 bytes and the salt and personalisation 16 each, all within BLAKE2b's limits, so the constructor
    // does not fail; the error is mapped rather than unwrapped so that nothing here can panic.
    macro_rules! fill {
        ($size:ty) => {{
            let mac = Blake2bMac::<$size>::new_with_salt_and_personal(master.expose(), &salt, &personal).map_err(|_| Error::KdfContext)?;
            let mut tag = mac.finalize().into_bytes();
            out.copy_from_slice(&tag);
            tag.as_mut_slice().zeroize();
        }};
    }
    match out.len() {
        16 => fill!(U16),
        64 => fill!(U64),
        _ => fill!(U32),
    }
    Ok(())
}

/// The largest HKDF-SHA256 output: 255 blocks of 32 bytes (RFC 5869 section 2.3).
pub const HKDF_SHA256_MAX_OUTPUT: usize = 255 * 32;

/// HKDF-SHA256 (RFC 5869): extract with `salt` (`None` is a string of 32 zero bytes), expand with `info`, fill `out`.
pub fn hkdf_sha256(ikm: &[u8], salt: Option<&[u8]>, info: &[u8], out: &mut [u8]) -> Result<(), Error> {
    if out.len() > HKDF_SHA256_MAX_OUTPUT {
        return Err(Error::HkdfLength);
    }
    Hkdf::<Sha256>::new(salt, ikm).expand(info, out).map_err(|_| Error::HkdfLength)
}

/// HKDF-SHA256 with a fixed-size secret output.
pub fn hkdf_sha256_secret<const N: usize>(ikm: &[u8], salt: Option<&[u8]>, info: &[u8]) -> Result<Secret<N>, Error> {
    let mut out = [0u8; N];
    let result = hkdf_sha256(ikm, salt, info, &mut out);
    let secret = Secret::new(out);
    out.zeroize();
    result.map(|()| secret)
}

/// HMAC-SHA256. HMAC takes a key of any length, so the error is never returned; it is mapped, not unwrapped, so that
/// nothing here can panic.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Result<[u8; 32], Error> {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).map_err(|_| Error::InvalidLength("hmac key"))?;
    mac.update(message);
    Ok(mac.finalize().into_bytes().into())
}

/// Verifies an HMAC-SHA256 tag in constant time. A tag of the wrong length is a mismatch.
pub fn hmac_sha256_verify(key: &[u8], message: &[u8], tag: &[u8]) -> bool {
    match <Hmac<Sha256> as Mac>::new_from_slice(key) {
        Ok(mut mac) => {
            mac.update(message);
            mac.verify_slice(tag).is_ok()
        }
        Err(_) => false,
    }
}

/// SHA-256.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// Lowercase hex of SHA-256 (design 4.1.5): how a label, a body or a state enters a signed string.
pub fn sha256_hex(data: &[u8]) -> String {
    hex_lower(&sha256(data))
}

/// Lowercase hex.
pub fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 15)]));
    }
    out
}
