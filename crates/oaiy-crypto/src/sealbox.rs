//! `sealbox` (design 4.1.1): libsodium's `crypto_box_seal`, byte for byte, so that what FormLogic's JavaScript and PHP seal opens
//! here and what is sealed here opens in libsodium.
//!
//! The format is `ephemeral_pk(32) || tag(16) || ciphertext`, where the ciphertext is XSalsa20-Poly1305 under
//! `HSalsa20(X25519(eph_sk, recipient_pk), 0^16)` with the nonce `BLAKE2b-24(ephemeral_pk || recipient_pk)` (`crypto_box_beforenm`
//! and `crypto_box_easy`). The Diffie-Hellman is this crate's [`x25519`](crate::x25519) one, so an ephemeral key of small order
//! (an all-zero shared secret) is refused, as libsodium's `crypto_box_seal_open` refuses it, and it is refused with the same
//! `DecryptFailed` as a bad tag.
//!
//! The design names the `crypto_box` crate with its `seal` feature. It is not used, because it does no all-zero check and this
//! module needs one; the XSalsa20-Poly1305 it uses, `crypto_secretbox`, is what `crypto_box` is built on.

use blake2::digest::consts::U24;
use blake2::{Blake2b, Digest};
use crypto_secretbox::aead::generic_array::GenericArray;
use crypto_secretbox::aead::{AeadInPlace, KeyInit};
use crypto_secretbox::{Kdf, Key, Nonce, Tag, XSalsa20Poly1305};
use zeroize::{Zeroize, Zeroizing};

use crate::error::Error;
use crate::x25519::{PublicKey, SecretKey};
use crate::zeroize::{Secret, SecretVec};

/// What sealing adds to the plaintext: the ephemeral public key and the tag (`crypto_box_SEALBYTES`).
pub const SEAL_OVERHEAD: usize = 48;

/// `crypto_box_beforenm`: HSalsa20 of the shared secret with a zero nonce.
fn box_key(shared: &Secret<32>) -> XSalsa20Poly1305 {
    let mut key: Key = <XSalsa20Poly1305 as Kdf>::kdf(Key::from_slice(shared.expose()), &GenericArray::default());
    let cipher = XSalsa20Poly1305::new(&key);
    key.as_mut_slice().zeroize();
    cipher
}

/// The nonce of a sealed box: BLAKE2b with a 24-byte output over both public keys.
fn seal_nonce(ephemeral: &PublicKey, recipient: &PublicKey) -> Nonce {
    let mut hasher = Blake2b::<U24>::new();
    hasher.update(ephemeral.as_bytes());
    hasher.update(recipient.as_bytes());
    hasher.finalize()
}

/// Seals `plaintext` to `recipient` under a fresh random ephemeral key.
pub fn seal(recipient: &PublicKey, plaintext: &[u8]) -> Result<Vec<u8>, Error> {
    let ephemeral = SecretKey::generate()?;
    seal_with_ephemeral(recipient, plaintext, &ephemeral)
}

/// Seals under a given ephemeral key. Not public: the nonce is a function of the ephemeral key, so sealing two messages under one
/// ephemeral key reuses a (key, nonce) pair. It exists for the known-answer tests.
pub(crate) fn seal_with_ephemeral(recipient: &PublicKey, plaintext: &[u8], ephemeral: &SecretKey) -> Result<Vec<u8>, Error> {
    let ephemeral_pk = ephemeral.public_key();
    let shared = ephemeral.diffie_hellman(recipient)?;
    let cipher = box_key(&shared);
    let nonce = seal_nonce(&ephemeral_pk, recipient);
    let mut buffer = Zeroizing::new(plaintext.to_vec());
    let tag = cipher.encrypt_in_place_detached(&nonce, &[], &mut buffer).map_err(|_| Error::InvalidLength("plaintext"))?;
    let mut out = Vec::with_capacity(SEAL_OVERHEAD + buffer.len());
    out.extend_from_slice(ephemeral_pk.as_bytes());
    out.extend_from_slice(&tag);
    out.extend_from_slice(&buffer);
    Ok(out)
}

/// Opens a sealed box addressed to `recipient`. Any failure (too short, an ephemeral key of small order, a bad tag, a box for
/// another key) is `Error::DecryptFailed`.
pub fn open(recipient: &SecretKey, sealed: &[u8]) -> Result<SecretVec, Error> {
    if sealed.len() < SEAL_OVERHEAD {
        return Err(Error::DecryptFailed);
    }
    let (ephemeral_bytes, rest) = sealed.split_at(32);
    let (tag, body) = rest.split_at(16);
    let ephemeral_pk = PublicKey::from_slice(ephemeral_bytes).map_err(|_| Error::DecryptFailed)?;
    let shared = recipient.diffie_hellman(&ephemeral_pk).map_err(|_| Error::DecryptFailed)?;
    let cipher = box_key(&shared);
    let nonce = seal_nonce(&ephemeral_pk, &recipient.public_key());
    let mut buffer = Zeroizing::new(body.to_vec());
    cipher.decrypt_in_place_detached(&nonce, &[], &mut buffer, Tag::from_slice(tag)).map_err(|_| Error::DecryptFailed)?;
    Ok(SecretVec::from_zeroizing(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// Nine sealed boxes built here under the ephemeral keys the generator chose equal, byte for byte, what libsodium's construction gives
    /// (the generator opened each with libsodium's own `crypto_box_seal_open` before writing it, and Python's hand-written XSalsa20-Poly1305
    /// recomputed every one). So what `seal` writes is what `crypto_box_seal_open` reads.
    #[test]
    fn the_deterministic_seal_equals_libsodiums_construction() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!("../tests/vectors/libsodium-oracle.json")).unwrap();
        let cases = oracle["sealedbox_kat"].as_array().unwrap();
        assert_eq!(cases.len(), 9);
        for case in cases {
            let text = |k: &str| case[k].as_str().unwrap();
            let recipient = SecretKey::from_libsodium_seed(&Secret::from_slice(&unhex(text("recipient_seed"))).unwrap());
            let ephemeral = SecretKey::from_slice(&unhex(text("eph_sk"))).unwrap();
            assert_eq!(recipient.public_key().as_bytes().as_slice(), unhex(text("recipient_pk")).as_slice());
            assert_eq!(ephemeral.public_key().as_bytes().as_slice(), unhex(text("eph_pk")).as_slice());
            let sealed = seal_with_ephemeral(&recipient.public_key(), &unhex(text("msg")), &ephemeral).unwrap();
            assert_eq!(sealed, unhex(text("sealed")), "message of {} bytes", text("msg").len() / 2);
            assert_eq!(seal_nonce(&ephemeral.public_key(), &recipient.public_key()).as_slice(), unhex(text("nonce")).as_slice());
            assert_eq!(open(&recipient, &sealed).unwrap().expose(), unhex(text("msg")).as_slice());
        }
    }

    #[test]
    fn a_random_seal_differs_every_time_and_always_opens() {
        let recipient = SecretKey::generate().unwrap();
        let a = seal(&recipient.public_key(), b"same message").unwrap();
        let b = seal(&recipient.public_key(), b"same message").unwrap();
        assert_ne!(a, b, "a fresh ephemeral key each time");
        assert_ne!(a[..32], b[..32]);
        assert_eq!(open(&recipient, &a).unwrap().expose(), b"same message");
        assert_eq!(open(&recipient, &b).unwrap().expose(), b"same message");
    }
}
