//! `xaead` and `wrap` (design 4.1.1): XChaCha20-Poly1305-IETF, the primitive of every wrapper, envelope and package.
//!
//! [`seal`] and [`open`] are the primitive as libsodium's `crypto_aead_xchacha20poly1305_ietf_*` and the XChaCha draft
//! define it: a 24-byte nonce, a 16-byte tag after the ciphertext, and associated data of any bytes (the ceremony's is the
//! 32-byte transcript hash). [`wrap`] is the format the vault uses for keys: `nonce(24) || ciphertext || tag(16)`, 72 bytes
//! for a 32-byte key, with a random nonce and an [`Aad`](crate::canon::Aad) that is a checked canonical string, so that the
//! associated data of a wrapper always names its domain, its user and its purpose (4.2.1).
//!
//! Failure is uniform: a wrong key, a wrong nonce, a wrong AAD, a flipped bit anywhere, a truncated or an extended
//! ciphertext are all `Error::DecryptFailed`, and the function has done the same work in each case up to the tag check.
//! A wrapped blob of 40 bytes or fewer (an empty plaintext) is refused, as `unwrapKey` in FormLogic's `vault.ts` does: a
//! wrapper holds a key, never nothing.

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Key, Tag, XChaCha20Poly1305, XNonce};
use zeroize::Zeroizing;

use crate::canon::Aad;
use crate::error::Error;
use crate::zeroize::{Secret, SecretVec};

/// XChaCha20-Poly1305 key length.
pub const KEY_LEN: usize = 32;
/// XChaCha20-Poly1305 nonce length.
pub const NONCE_LEN: usize = 24;
/// Poly1305 tag length.
pub const TAG_LEN: usize = 16;
/// What `wrap` adds to the plaintext: the nonce and the tag.
pub const WRAP_OVERHEAD: usize = NONCE_LEN + TAG_LEN;
/// A wrapped 32-byte key: 24 + 32 + 16.
pub const WRAPPED_KEY_LEN: usize = WRAP_OVERHEAD + KEY_LEN;

/// Encrypts `plaintext` and returns `ciphertext || tag`. The caller owns the uniqueness of `(key, nonce)`: use [`wrap`] unless the
/// nonce is fixed by a protocol or a known-answer test.
pub fn seal(key: &Secret<32>, nonce: &[u8; NONCE_LEN], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, Error> {
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key.expose()));
    let mut out = Vec::with_capacity(plaintext.len() + TAG_LEN);
    out.extend_from_slice(plaintext);
    let tag = cipher.encrypt_in_place_detached(XNonce::from_slice(nonce), aad, &mut out).map_err(|_| Error::InvalidLength("plaintext"))?;
    out.extend_from_slice(&tag);
    Ok(out)
}

/// Decrypts `ciphertext || tag`. Any failure is `Error::DecryptFailed`.
pub fn open(key: &Secret<32>, nonce: &[u8; NONCE_LEN], aad: &[u8], ciphertext: &[u8]) -> Result<SecretVec, Error> {
    if ciphertext.len() < TAG_LEN {
        return Err(Error::DecryptFailed);
    }
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key.expose()));
    let (body, tag) = ciphertext.split_at(ciphertext.len() - TAG_LEN);
    let mut buffer = Zeroizing::new(body.to_vec());
    cipher.decrypt_in_place_detached(XNonce::from_slice(nonce), aad, &mut buffer, Tag::from_slice(tag)).map_err(|_| Error::DecryptFailed)?;
    Ok(SecretVec::from_zeroizing(buffer))
}

/// `wrap`: a fresh random nonce, then `nonce || ciphertext || tag`. The plaintext must not be empty.
pub fn wrap(key: &Secret<32>, aad: &Aad, plaintext: &[u8]) -> Result<Vec<u8>, Error> {
    if plaintext.is_empty() {
        return Err(Error::InvalidLength("plaintext"));
    }
    let mut nonce = [0u8; NONCE_LEN];
    crate::random::fill(&mut nonce)?;
    let sealed = seal(key, &nonce, aad.as_bytes(), plaintext)?;
    let mut out = Vec::with_capacity(NONCE_LEN + sealed.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&sealed);
    Ok(out)
}

/// The inverse of [`wrap`]. A blob of `WRAP_OVERHEAD` bytes or fewer is refused, like every other failure, as `DecryptFailed`.
pub fn unwrap(key: &Secret<32>, aad: &Aad, blob: &[u8]) -> Result<SecretVec, Error> {
    if blob.len() <= WRAP_OVERHEAD {
        return Err(Error::DecryptFailed);
    }
    let (nonce, rest) = blob.split_at(NONCE_LEN);
    let nonce: &[u8; NONCE_LEN] = nonce.try_into().map_err(|_| Error::DecryptFailed)?;
    open(key, nonce, aad.as_bytes(), rest)
}

/// Wraps a 32-byte key: exactly [`WRAPPED_KEY_LEN`] (72) bytes.
pub fn wrap_key(wrapping: &Secret<32>, aad: &Aad, key: &Secret<32>) -> Result<[u8; WRAPPED_KEY_LEN], Error> {
    let blob = wrap(wrapping, aad, key.expose())?;
    blob.try_into().map_err(|_| Error::InvalidLength("wrapped key"))
}

/// Unwraps a 32-byte key from exactly 72 bytes; any other length is `DecryptFailed`.
pub fn unwrap_key(wrapping: &Secret<32>, aad: &Aad, blob: &[u8]) -> Result<Secret<32>, Error> {
    if blob.len() != WRAPPED_KEY_LEN {
        return Err(Error::DecryptFailed);
    }
    let plain = unwrap(wrapping, aad, blob)?;
    Secret::from_slice(plain.expose()).map_err(|_| Error::DecryptFailed)
}
