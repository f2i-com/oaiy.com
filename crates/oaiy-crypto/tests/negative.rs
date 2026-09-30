//! The negative corpus: bit flips in every field, truncated and extended ciphertexts, wrong keys, wrong AAD, wrong lengths (swept, so that
//! no length panics), non-canonical S, bounds. Nothing here may panic, and every refusal is the uniform one.

mod common;

use common::*;
use oaiy_crypto::aead;
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::ed25519::{KeyRole, Signature, SigningKey, VerifyingKey};
use oaiy_crypto::kdf::{self, Context};
use oaiy_crypto::kit::RecoveryKit;
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::{PublicKey, SecretKey};
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

fn wrapping_key() -> Secret<32> {
    Secret::new([0x5a; 32])
}

fn aad() -> Aad {
    Aad::new(AadDomain::VaultWrap, &["user-1", "vw_00112233445566778899aabbccddeeff", "recovery-phrase", &"c".repeat(64)]).unwrap()
}

#[test]
fn a_wrapped_key_refuses_every_single_bit_flip_in_the_nonce_the_ciphertext_and_the_tag() {
    let key = Secret::new([0x33; 32]);
    let blob = aead::wrap_key(&wrapping_key(), &aad(), &key).unwrap();
    assert_eq!(blob.len(), 72);
    assert_eq!(aead::unwrap_key(&wrapping_key(), &aad(), &blob).unwrap(), key);
    for bit in 0..blob.len() * 8 {
        let mut bad = blob;
        bad[bit / 8] ^= 1 << (bit % 8);
        assert_eq!(
            aead::unwrap_key(&wrapping_key(), &aad(), &bad).unwrap_err(),
            Error::DecryptFailed,
            "bit {bit} (byte {} of the nonce, ciphertext or tag)",
            bit / 8
        );
    }
}

#[test]
fn a_wrapped_key_refuses_every_truncation_and_every_extension() {
    let blob = aead::wrap_key(&wrapping_key(), &aad(), &Secret::new([0x33; 32])).unwrap();
    for len in 0..blob.len() {
        assert_eq!(aead::unwrap_key(&wrapping_key(), &aad(), &blob[..len]).unwrap_err(), Error::DecryptFailed, "truncated to {len}");
        assert_eq!(aead::unwrap(&wrapping_key(), &aad(), &blob[..len]).unwrap_err(), Error::DecryptFailed, "truncated to {len} (generic unwrap)");
    }
    for extra in 1..=40 {
        let mut long = blob.to_vec();
        long.extend(std::iter::repeat_n(0u8, extra));
        assert_eq!(aead::unwrap_key(&wrapping_key(), &aad(), &long).unwrap_err(), Error::DecryptFailed, "extended by {extra}");
        assert_eq!(aead::unwrap(&wrapping_key(), &aad(), &long).unwrap_err(), Error::DecryptFailed, "extended by {extra} (generic unwrap)");
    }
}

#[test]
fn a_wrapped_key_refuses_a_wrong_key_and_a_wrong_aad_in_every_position() {
    let blob = aead::wrap_key(&wrapping_key(), &aad(), &Secret::new([0x33; 32])).unwrap();
    assert!(aead::unwrap_key(&Secret::new([0x5b; 32]), &aad(), &blob).is_err(), "wrong key");
    let mut key = [0x5a; 32];
    for i in 0..32 {
        key[i] ^= 0x80;
        assert!(aead::unwrap_key(&Secret::new(key), &aad(), &blob).is_err(), "key byte {i}");
        key[i] ^= 0x80;
    }
    // the AAD string with any one byte changed to another permitted character, or dropped, or doubled
    let text = std::str::from_utf8(aad().as_bytes()).unwrap().to_string();
    for i in 0..text.len() {
        let mut changed = text.clone().into_bytes();
        changed[i] = if changed[i] == b'x' { b'y' } else { b'x' };
        let nonce: [u8; 24] = blob[..24].try_into().unwrap();
        assert!(aead::open(&wrapping_key(), &nonce, &changed, &blob[24..]).is_err(), "AAD byte {i}");
    }
    let nonce: [u8; 24] = blob[..24].try_into().unwrap();
    for bad in [&text[1..], &text[..text.len() - 1], "", &format!("{text}|"), &format!("{text}{text}")] {
        assert!(aead::open(&wrapping_key(), &nonce, bad.as_bytes(), &blob[24..]).is_err(), "{bad:?}");
    }
}

#[test]
fn wrap_never_wraps_nothing_and_unwrap_never_opens_a_blob_of_40_bytes_or_fewer() {
    assert_eq!(aead::wrap(&wrapping_key(), &aad(), b"").unwrap_err(), Error::InvalidLength("plaintext"));
    // a genuine empty-plaintext blob (nonce || tag) made with the raw primitive: refused, as FormLogic's unwrapKey refuses it
    let nonce = [7u8; 24];
    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&aead::seal(&wrapping_key(), &nonce, aad().as_bytes(), b"").unwrap());
    assert_eq!(blob.len(), 40);
    assert_eq!(aead::unwrap(&wrapping_key(), &aad(), &blob).unwrap_err(), Error::DecryptFailed);
    // the raw primitive itself handles the empty message
    assert!(aead::open(&wrapping_key(), &nonce, aad().as_bytes(), &blob[24..]).unwrap().is_empty());
    // a wrapped plaintext of one byte is the smallest blob: 41 bytes
    assert_eq!(aead::wrap(&wrapping_key(), &aad(), b"x").unwrap().len(), 41);
}

#[test]
fn a_raw_ciphertext_of_any_length_from_zero_to_eighty_refuses_without_panicking() {
    let mut rng = Rng(11);
    let nonce = [3u8; 24];
    for len in 0..=80 {
        let garbage = rng.bytes(len);
        assert_eq!(aead::open(&wrapping_key(), &nonce, b"aad", &garbage).unwrap_err(), Error::DecryptFailed, "length {len}");
    }
}

#[test]
fn a_multi_block_message_refuses_every_bit_flip_in_the_ciphertext_and_tag() {
    let mut rng = Rng(12);
    let message = rng.bytes(200); // three ChaCha blocks and a partial one
    let nonce = [9u8; 24];
    let sealed = aead::seal(&wrapping_key(), &nonce, b"context", &message).unwrap();
    assert_eq!(sealed.len(), 216);
    for bit in 0..sealed.len() * 8 {
        let mut bad = sealed.clone();
        bad[bit / 8] ^= 1 << (bit % 8);
        assert!(aead::open(&wrapping_key(), &nonce, b"context", &bad).is_err(), "bit {bit}");
    }
    let mut bad_nonce = nonce;
    for bit in 0..24 * 8 {
        bad_nonce[bit / 8] ^= 1 << (bit % 8);
        assert!(aead::open(&wrapping_key(), &bad_nonce, b"context", &sealed).is_err(), "nonce bit {bit}");
        bad_nonce[bit / 8] ^= 1 << (bit % 8);
    }
    // the same key and nonce over two messages must not give related ciphertexts of the tag (a sanity check that the AAD is authenticated)
    let other = aead::seal(&wrapping_key(), &nonce, b"other context", &message).unwrap();
    assert_eq!(sealed[..200], other[..200], "same keystream, as any stream cipher under one nonce");
    assert_ne!(sealed[200..], other[200..], "but the tag covers the AAD");
}

#[test]
fn a_sealed_box_refuses_every_bit_flip_every_truncation_and_every_extension() {
    let recipient = SecretKey::from_bytes([0x61; 32]);
    let sealed = sealbox::seal(&recipient.public_key(), &[0x77; 32]).unwrap();
    assert_eq!(sealed.len(), 80);
    assert_eq!(sealbox::open(&recipient, &sealed).unwrap().expose(), &[0x77; 32]);
    for bit in 0..sealed.len() * 8 {
        let mut bad = sealed.clone();
        bad[bit / 8] ^= 1 << (bit % 8);
        assert_eq!(sealbox::open(&recipient, &bad).unwrap_err(), Error::DecryptFailed, "bit {bit}");
    }
    for len in 0..sealed.len() {
        assert_eq!(sealbox::open(&recipient, &sealed[..len]).unwrap_err(), Error::DecryptFailed, "truncated to {len}");
    }
    for extra in 1..=32 {
        let mut long = sealed.clone();
        long.extend(std::iter::repeat_n(0u8, extra));
        assert_eq!(sealbox::open(&recipient, &long).unwrap_err(), Error::DecryptFailed, "extended by {extra}");
    }
    assert_eq!(sealbox::open(&SecretKey::from_bytes([0x62; 32]), &sealed).unwrap_err(), Error::DecryptFailed, "another recipient");
    // the smallest sealed box (an empty message) is 48 bytes, and 47 is refused
    let empty = sealbox::seal(&recipient.public_key(), b"").unwrap();
    assert_eq!(empty.len(), 48);
    assert!(sealbox::open(&recipient, &empty).unwrap().is_empty());
    assert!(sealbox::open(&recipient, &empty[..47]).is_err());
}

#[test]
fn a_signature_refuses_every_bit_flip_in_the_signature_the_message_and_the_key() {
    let key = SigningKey::from_seed(KeyRole::Hazmat, &Secret::new([0x21; 32]));
    let public = key.verifying_key();
    let message = b"the message that was signed";
    let signature = key.sign_raw(message).unwrap();
    public.verify_raw(message, &signature).unwrap();
    for bit in 0..512 {
        let mut bad = signature.to_bytes();
        bad[bit / 8] ^= 1 << (bit % 8);
        assert!(public.verify_raw(message, &Signature::from_bytes(&bad)).is_err(), "signature bit {bit}");
    }
    for bit in 0..message.len() * 8 {
        let mut bad = message.to_vec();
        bad[bit / 8] ^= 1 << (bit % 8);
        assert!(public.verify_raw(&bad, &signature).is_err(), "message bit {bit}");
    }
    for bit in 0..256 {
        let mut bytes = public.to_bytes();
        bytes[bit / 8] ^= 1 << (bit % 8);
        if let Ok(other) = VerifyingKey::from_bytes(&bytes) {
            assert!(other.verify_raw(message, &signature).is_err(), "key bit {bit}");
        }
    }
    // a truncated or extended message and the empty message
    assert!(public.verify_raw(&message[1..], &signature).is_err());
    assert!(public.verify_raw(b"", &signature).is_err());
}

/// The group order L, little-endian.
const L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];

fn add_le(a: &[u8; 32], b: &[u8; 32]) -> ([u8; 32], bool) {
    let mut out = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let sum = u16::from(a[i]) + u16::from(b[i]) + carry;
        out[i] = sum as u8;
        carry = sum >> 8;
    }
    (out, carry != 0)
}

#[test]
fn a_signature_with_s_not_below_the_group_order_is_refused_for_fifty_random_keys() {
    let mut rng = Rng(21);
    for i in 0..50 {
        let key = SigningKey::from_seed(KeyRole::Hazmat, &Secret::new(rng.array::<32>()));
        let message = rng.bytes(i % 40);
        let signature = key.sign_raw(&message).unwrap().to_bytes();
        key.verifying_key().verify_raw(&message, &Signature::from_bytes(&signature)).unwrap();
        let s: [u8; 32] = signature[32..].try_into().unwrap();
        // S + L is the same scalar modulo L, so the plain equation holds; canonical-S checking refuses it
        let (malleated, overflow) = add_le(&s, &L);
        assert!(!overflow);
        let mut bad = signature;
        bad[32..].copy_from_slice(&malleated);
        assert!(key.verifying_key().verify_raw(&message, &Signature::from_bytes(&bad)).is_err(), "S + L, key {i}");
        // S with any of the top three bits set is not below L either
        for top in [0x20u8, 0x40, 0x80] {
            let mut bad = signature;
            bad[63] |= top;
            assert!(key.verifying_key().verify_raw(&message, &Signature::from_bytes(&bad)).is_err(), "top bit {top:#x}, key {i}");
        }
    }
}

#[test]
fn every_wrong_length_of_every_fixed_size_input_is_refused_without_a_panic() {
    let mut rng = Rng(31);
    for len in 0..=100usize {
        let bytes = rng.bytes(len);
        let ok = |n: usize| len == n;
        assert_eq!(Signature::from_slice(&bytes).is_ok(), ok(64), "signature {len}");
        if len != 32 {
            assert_eq!(VerifyingKey::from_slice(&bytes).unwrap_err(), Error::InvalidLength("ed25519 public key"));
            assert_eq!(PublicKey::from_slice(&bytes).unwrap_err(), Error::InvalidLength("x25519 public key"));
            assert_eq!(SecretKey::from_slice(&bytes).err(), Some(Error::InvalidLength("x25519 secret key")));
            assert!(Secret::<32>::from_slice(&bytes).is_err());
        }
        assert_eq!(Secret::<64>::from_slice(&bytes).is_ok(), ok(64));
        assert_eq!(Secret::<16>::from_slice(&bytes).is_ok(), ok(16));
        let mut out = vec![0u8; len];
        let key: Secret<32> = Secret::new([1; 32]);
        let context = Context::new("flrecov1").unwrap();
        assert_eq!(kdf::derive_subkey_into(&key, 1, &context, &mut out).is_ok(), matches!(len, 16 | 32 | 64), "kdf output {len}");
        assert!(aead::unwrap_key(&key, &aad(), &bytes).is_err());
        assert!(sealbox::open(&SecretKey::from_bytes([2; 32]), &bytes).is_err());
    }
}

#[test]
fn recovery_kit_codes_refuse_every_single_character_substitution_and_the_trailing_bits() {
    let kit = RecoveryKit::from_bytes(Secret::new(*b"0123456789abcdef0123456789abcdef"));
    let code = kit.encode().expose().to_string();
    assert_eq!(RecoveryKit::decode(&code).unwrap().key(), kit.key());
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let positions: Vec<usize> = code.char_indices().filter(|(i, c)| *i >= 6 && *c != '-').map(|(i, _)| i).collect();
    assert_eq!(positions.len(), 56);
    let (mut refused, mut total) = (0, 0);
    for (n, position) in positions.iter().enumerate() {
        for substitute in alphabet {
            if *substitute == code.as_bytes()[*position] {
                continue;
            }
            let mut bad = code.clone().into_bytes();
            bad[*position] = *substitute;
            let bad = String::from_utf8(bad).unwrap();
            total += 1;
            match RecoveryKit::decode(&bad) {
                Err(Error::KitChecksum) | Err(Error::KitFormat) => refused += 1,
                other => panic!("position {n}: {other:?} accepted a substitution"),
            }
        }
    }
    assert_eq!(refused, total);
    // the last body character carries one key bit and four bits that must be zero: JavaScript ignores them, this crate does not
    let mut chars: Vec<char> = code.chars().collect();
    let last_body = chars.len() - 6; // "...-XXXX-CCCC": four checksum characters and a hyphen come after the last body character
    let original = chars[last_body];
    let index = alphabet.iter().position(|a| char::from(*a) == original).unwrap();
    // the 52nd character holds one key bit (the top bit of its five) and four padding bits: flip a padding bit, and the key stays the same
    chars[last_body] = char::from(alphabet[index ^ 0b10]);
    let text: String = chars.into_iter().collect();
    assert_eq!(RecoveryKit::decode(&text).unwrap_err(), Error::KitFormat, "non-zero trailing bits");
    for bad in ["", "FLRK1", "FLRK2-AAAA", "flrk1", "AAAA-AAAA", &code[1..], &code[..code.len() - 1], &format!("{code}A"), &"A".repeat(300)] {
        assert!(RecoveryKit::decode(bad).is_err(), "{bad:?}");
    }
    // a code with the checksum of another key
    let other = RecoveryKit::from_bytes(Secret::new([0xee; 32])).encode().expose().to_string();
    let mixed = format!("{}{}", &code[..code.len() - 4], &other[other.len() - 4..]);
    assert_eq!(RecoveryKit::decode(&mixed).unwrap_err(), Error::KitChecksum);
}

#[test]
fn hkdf_and_the_kdf_refuse_what_they_cannot_give() {
    let mut too_long = vec![0u8; 255 * 32 + 1];
    assert_eq!(kdf::hkdf_sha256(b"ikm", Some(b"salt"), b"info", &mut too_long).unwrap_err(), Error::HkdfLength);
    let key: Secret<32> = Secret::new([1; 32]);
    let context = Context::new("flrecov1").unwrap();
    for len in [0usize, 1, 15, 17, 31, 33, 63, 65, 128] {
        assert_eq!(kdf::derive_subkey_into(&key, 1, &context, &mut vec![0u8; len]).unwrap_err(), Error::KdfContext, "{len}");
    }
    // different masters, ids and contexts give different keys
    let a = kdf::derive_subkey(&key, 1, &context).unwrap();
    assert_ne!(a, kdf::derive_subkey(&key, 2, &context).unwrap());
    assert_ne!(a, kdf::derive_subkey(&key, 1, &Context::new("flrecov2").unwrap()).unwrap());
    assert_ne!(a, kdf::derive_subkey(&Secret::new([2; 32]), 1, &context).unwrap());
}
