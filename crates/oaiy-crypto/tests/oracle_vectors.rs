//! libsodium as the oracle (`tests/vectors/libsodium-oracle.json`, made by `scripts/oracle.mjs` with libsodium 1.0.x, and recomputed by
//! `scripts/oracle_check.py` with Python `cryptography`, hashlib and a hand-written Salsa20 family: 456 recomputations): random-looking
//! inputs with libsodium's outputs for `crypto_kdf`, XChaCha20-Poly1305, Ed25519 signing, X25519, Argon2id and (with OpenSSL's) HKDF,
//! and libsodium's verdict on the Ed25519 negative corpus of 80 cases (the RFC 8032 vectors malleated in every way, every point of
//! small order in every encoding, the twelve ed25519-speccheck cases). This crate must give libsodium's answer on every one of them.

mod common;

use std::collections::BTreeSet;

use common::*;
use oaiy_crypto::aead;
use oaiy_crypto::argon;
use oaiy_crypto::ed25519::{KeyRole, Signature, SigningKey, VerifyingKey};
use oaiy_crypto::kdf::{self, Context};
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::{self, PublicKey, SecretKey};
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

fn oracle() -> serde_json::Value {
    json(ORACLE)
}

fn s(value: &serde_json::Value) -> &str {
    value.as_str().unwrap()
}

#[test]
fn crypto_kdf_133_vectors_at_16_32_and_64_bytes() {
    let o = oracle();
    let cases = o["kdf"].as_array().unwrap();
    assert_eq!(cases.len(), 133);
    let mut lengths = BTreeSet::new();
    for case in cases {
        let master: Secret<32> = secret(s(&case["key"]));
        let id: u64 = s(&case["id"]).parse().unwrap();
        let context = Context::new(s(&case["ctx"])).unwrap();
        let len = case["len"].as_u64().unwrap() as usize;
        lengths.insert(len);
        let mut out = vec![0u8; len];
        kdf::derive_subkey_into(&master, id, &context, &mut out).unwrap();
        assert_eq!(hex(&out), s(&case["out"]), "{} id {id} len {len}", s(&case["ctx"]));
        if len == 32 {
            assert_eq!(hex(kdf::derive_subkey(&master, id, &context).unwrap().expose()), s(&case["out"]));
        }
    }
    assert_eq!(lengths, BTreeSet::from([16, 32, 64]));
    // subkey ids up to u64::MAX, as libsodium's uint64_t
    assert!(cases.iter().any(|c| s(&c["id"]) == "18446744073709551615"));
}

/// libsodium takes any eight bytes as a context; this crate takes eight of `[a-z0-9]`, as every row of the registry is, so that a
/// context cannot be misspelt into a different derivation with a capital or a symbol.
#[test]
fn a_context_is_eight_lowercase_alphanumerics_and_stricter_than_libsodiums() {
    for bad in ["", "flrecov", "flrecov11", "FLRECOV1", "flrecov!", "flrecov\0", "fl recov", "flrecové"] {
        assert_eq!(Context::new(bad).unwrap_err(), Error::KdfContext, "{bad:?}");
    }
    assert!(Context::new("abcdefgh").is_ok() && Context::new("00000000").is_ok());
}

#[test]
fn xchacha20_poly1305_30_vectors_with_empty_and_long_inputs() {
    let o = oracle();
    let cases = o["xchacha"].as_array().unwrap();
    assert_eq!(cases.len(), 30);
    for case in cases {
        let key: Secret<32> = secret(s(&case["key"]));
        let nonce: [u8; 24] = arr(s(&case["nonce"]));
        let aad = unhex(s(&case["aad"]));
        let plaintext = unhex(s(&case["pt"]));
        let sealed = aead::seal(&key, &nonce, &aad, &plaintext).unwrap();
        assert_eq!(hex(&sealed), s(&case["ct"]));
        assert_eq!(aead::open(&key, &nonce, &aad, &sealed).unwrap().expose(), plaintext.as_slice());
    }
    assert!(cases.iter().any(|c| s(&c["pt"]).is_empty()) && cases.iter().any(|c| s(&c["aad"]).is_empty()));
}

#[test]
fn ed25519_signatures_equal_libsodiums_and_the_libsodium_secret_key_form_is_checked() {
    let o = oracle();
    for case in o["ed25519_sign"].as_array().unwrap() {
        let seed: Secret<32> = secret(s(&case["seed"]));
        let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
        assert_eq!(hex(&key.verifying_key().to_bytes()), s(&case["pk"]));
        let message = unhex(s(&case["msg"]));
        assert_eq!(hex(&key.sign_raw(&message).unwrap().to_bytes()), s(&case["sig"]));
        // libsodium's 64-byte secret key (seed || public key) gives the same key
        let sk64: Secret<64> = secret(s(&case["sk64"]));
        let from64 = SigningKey::from_libsodium_secret_key(KeyRole::Hazmat, &sk64).unwrap();
        assert_eq!(from64.verifying_key(), key.verifying_key());
        assert_eq!(from64.seed(), seed);
        // a public half that is not the seed's is refused, whichever byte of it is wrong
        for i in [32usize, 40, 63] {
            let mut bad = *sk64.expose();
            bad[i] ^= 1;
            assert_eq!(SigningKey::from_libsodium_secret_key(KeyRole::Hazmat, &Secret::new(bad)).unwrap_err(), Error::InvalidKey, "byte {i}");
        }
        // and another key's public half likewise
        let mut swapped = *sk64.expose();
        swapped[32..].copy_from_slice(&arr::<32>(s(&o["ed25519_sign"][0]["pk"])));
        if swapped[32..] != sk64.expose()[32..] {
            assert!(SigningKey::from_libsodium_secret_key(KeyRole::Hazmat, &Secret::new(swapped)).is_err());
        }
    }
}

#[test]
fn ed25519_negative_corpus_gives_libsodiums_verdict_on_all_80_cases() {
    let o = oracle();
    let cases = o["ed25519_verify"].as_array().unwrap();
    assert_eq!(cases.len(), 80);
    let mut accepted = Vec::new();
    let mut python_accepts_we_refuse = Vec::new();
    for case in cases {
        let name = s(&case["name"]);
        let (pk, msg, sig) = (unhex(s(&case["pk"])), unhex(s(&case["msg"])), unhex(s(&case["sig"])));
        let ours = match (VerifyingKey::from_slice(&pk), Signature::from_slice(&sig)) {
            (Ok(key), Ok(signature)) => key.verify_raw(&msg, &signature).is_ok(),
            _ => false,
        };
        assert_eq!(ours, case["libsodium"].as_bool().unwrap(), "{name}: this crate must answer as libsodium does");
        assert_eq!(case["libsodium"], case["openssl"], "{name}: node's OpenSSL agrees with libsodium on this corpus");
        if ours {
            accepted.push(name.to_string());
        }
        if case["python_openssl"].as_bool().unwrap() && !ours {
            python_accepts_we_refuse.push(name.to_string());
        }
    }
    // the three positive controls and the one speccheck case every strict verifier accepts
    assert_eq!(accepted.len(), 4, "{accepted:?}");
    assert!(accepted.iter().filter(|n| n.contains("(valid: the positive control)")).count() == 3);
    assert!(accepted.iter().any(|n| n == "ed25519-speccheck case 3"));
    // design finding A5: Python's OpenSSL accepts signatures that no strict verifier does; this crate is not one of them
    let recorded: Vec<String> =
        o["meta"]["ed25519_divergences_libsodium_vs_python_openssl"].as_array().unwrap().iter().map(|n| s(n).to_string()).collect();
    assert_eq!(python_accepts_we_refuse.len(), 16);
    assert_eq!(python_accepts_we_refuse.len(), recorded.len());
    assert!(python_accepts_we_refuse.iter().any(|n| n.contains("[order 1, canonical 01000000]") && n.contains("R=identity")));
}

/// What strict verification buys when the public key is acceptable. A key of mixed order (a·B plus a point of order 8) is a good key for every
/// verifier, and with a small-order R the plain cofactorless equation can be made to hold for it (the generator does it for R = identity and
/// for R = -3T). dalek's plain `verify` accepts both; libsodium, node's OpenSSL and this crate refuse them, because R is of small order. Without
/// `verify_strict` these two cases are the only ones in the corpus that would pass.
#[test]
fn a_small_order_r_under_a_key_of_mixed_order_is_what_strict_verification_refuses() {
    use ed25519_dalek::Verifier;
    let o = oracle();
    let cases: Vec<&serde_json::Value> =
        o["ed25519_verify"].as_array().unwrap().iter().filter(|c| s(&c["name"]).starts_with("mixed-order A")).collect();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let (pk, msg, sig) = (arr::<32>(s(&case["pk"])), unhex(s(&case["msg"])), arr::<64>(s(&case["sig"])));
        // the key is acceptable: canonical, on the curve, not of small order
        let key = VerifyingKey::from_bytes(&pk).expect("a key of mixed order is a valid key");
        // the plain equation holds
        let plain = ed25519_dalek::VerifyingKey::from_bytes(&pk).unwrap();
        assert!(plain.verify(&msg, &ed25519_dalek::Signature::from_bytes(&sig)).is_ok(), "{}: the non-strict verifier accepts it", s(&case["name"]));
        // and this crate refuses, as libsodium does
        assert_eq!(key.verify_raw(&msg, &Signature::from_bytes(&sig)).unwrap_err(), Error::SignatureInvalid, "{}", s(&case["name"]));
        assert_eq!(case["libsodium"], false);
        assert_eq!(case["python_openssl"], true, "Python's OpenSSL accepts it: design finding A5");
    }
}

#[test]
fn every_point_of_small_order_in_every_encoding_is_refused_as_a_public_key() {
    let o = oracle();
    let keys = o["small_order_keys"].as_array().unwrap();
    assert_eq!(keys.len(), 14, "eight points, and the non-canonical encodings of the ones that have them");
    let canonical = keys.iter().filter(|k| k["canonical"].as_bool().unwrap()).count();
    assert_eq!(canonical, 8, "the eight points of order dividing 8");
    for key in keys {
        let bytes: [u8; 32] = arr(s(&key["enc"]));
        let expected = if key["canonical"].as_bool().unwrap() { Error::SmallOrderKey } else { Error::NonCanonicalKey };
        assert_eq!(VerifyingKey::from_bytes(&bytes).unwrap_err(), expected, "{}", s(&key["name"]));
        assert_eq!(key["libsodium_accepts_identity_sig"], false, "libsodium refuses it too");
    }
}

#[test]
fn a_non_canonical_encoding_of_a_good_key_is_refused_and_the_canonical_one_is_not() {
    // y in 0..19 has an encoding y+p below 2^255 that decodes to the same point (dalek reduces y modulo p; libsodium refuses the encoding).
    // Where the point is of large order, the canonical encoding is a fine public key and its alias must be refused.
    let mut aliases_refused = 0;
    for y in 0u8..19 {
        for sign in [0u8, 1] {
            let mut canonical = [0u8; 32];
            canonical[0] = y;
            canonical[31] = sign << 7;
            let mut alias = [0xffu8; 32];
            alias[0] = 0xed + y;
            alias[31] = 0x7f | (sign << 7);
            if VerifyingKey::from_bytes(&canonical).is_ok() {
                assert_eq!(VerifyingKey::from_bytes(&alias).unwrap_err(), Error::NonCanonicalKey, "y = {y}, sign = {sign}");
                aliases_refused += 1;
            }
        }
    }
    assert!(aliases_refused >= 2, "the probe found points of large order whose non-canonical alias exists ({aliases_refused})");
}

#[test]
fn x25519_24_shared_secrets_equal_libsodiums() {
    let o = oracle();
    for case in o["x25519_dh"].as_array().unwrap() {
        let a = SecretKey::from_bytes(arr(s(&case["sk"])));
        let b = SecretKey::from_bytes(arr(s(&case["peer_sk"])));
        assert_eq!(hex(a.public_key().as_bytes()), s(&case["pk_of_sk"]));
        assert_eq!(hex(b.public_key().as_bytes()), s(&case["peer_pk"]));
        assert_eq!(hex(a.diffie_hellman(&b.public_key()).unwrap().expose()), s(&case["shared"]));
        assert_eq!(hex(b.diffie_hellman(&a.public_key()).unwrap().expose()), s(&case["shared"]));
    }
}

#[test]
fn the_low_order_table_is_the_set_libsodium_refuses() {
    let o = oracle();
    let cases = o["x25519_low_order"].as_array().unwrap();
    assert_eq!(cases.len(), 14, "seven u-coordinates, each with the ignored top bit clear and set");
    let mut masked = BTreeSet::new();
    for case in cases {
        assert_eq!(case["libsodium_scalarmult_rejects"], true);
        assert_eq!(case["python_rejects"], true, "OpenSSL refuses an all-zero result too");
        let mut enc: [u8; 32] = arr(s(&case["enc"]));
        assert!(x25519::is_low_order(&enc), "{}", s(&case["enc"]));
        assert_eq!(PublicKey::from_bytes(&enc).unwrap_err(), Error::LowOrderPoint);
        enc[31] &= 0x7f;
        masked.insert(enc);
    }
    let table: BTreeSet<[u8; 32]> = x25519::LOW_ORDER.iter().copied().collect();
    assert_eq!(table, masked, "the table of this crate is exactly libsodium's list");
    // and nothing else is: a thousand ordinary points pass the check
    let mut rng = Rng(5);
    for _ in 0..1000 {
        assert!(!x25519::is_low_order(&rng.array::<32>()));
    }
}

#[test]
fn sealed_boxes_of_small_order_ephemeral_keys_are_refused_as_libsodium_refuses_them() {
    let o = oracle();
    let cases = o["sealedbox_low_order"].as_array().unwrap();
    assert_eq!(cases.len(), 14);
    for case in cases {
        assert_eq!(case["libsodium_opens"], false);
        let recipient = SecretKey::from_libsodium_seed(&secret::<32>(s(&case["recipient_seed"])));
        let blob = unhex(s(&case["blob"]));
        assert_eq!(&blob[..32], unhex(s(&case["epk"])).as_slice());
        assert_eq!(sealbox::open(&recipient, &blob).unwrap_err(), Error::DecryptFailed, "{}", s(&case["epk"]));
    }
}

/// The forgery a small-order ephemeral key allows. The shared secret is all zero for every recipient, so the attacker knows the box key and can make
/// a box that authenticates under it (the generator checks each one with libsodium's own `crypto_secretbox_open_easy`). `crypto_box_seal_open`
/// refuses them; an implementation without the all-zero check would open them, and these tests are the ones that fail if this crate lost it.
#[test]
fn a_forged_sealed_box_under_a_small_order_ephemeral_key_is_refused() {
    let o = oracle();
    let cases = o["sealedbox_forged_low_order"].as_array().unwrap();
    assert_eq!(cases.len(), 14);
    for case in cases {
        assert_eq!(case["libsodium_opens"], false);
        let recipient = SecretKey::from_libsodium_seed(&secret::<32>(s(&case["recipient_seed"])));
        let forged = unhex(s(&case["forged"]));
        assert_eq!(&forged[..32], unhex(s(&case["epk"])).as_slice());
        assert_eq!(forged.len(), 32 + 16 + unhex(s(&case["msg"])).len());
        assert_eq!(sealbox::open(&recipient, &forged).unwrap_err(), Error::DecryptFailed, "epk {}", s(&case["epk"]));
    }
}

#[test]
fn sealed_boxes_libsodium_made_open() {
    let o = oracle();
    for case in o["sealedbox_kat"].as_array().unwrap() {
        let recipient = SecretKey::from_libsodium_seed(&secret::<32>(s(&case["recipient_seed"])));
        assert_eq!(hex(recipient.public_key().as_bytes()), s(&case["recipient_pk"]));
        let opened = sealbox::open(&recipient, &unhex(s(&case["sealed"]))).unwrap();
        assert_eq!(hex(opened.expose()), s(&case["msg"]));
    }
}

#[test]
fn hkdf_sha256_56_vectors_against_openssl() {
    let o = oracle();
    let cases = o["hkdf"].as_array().unwrap();
    assert_eq!(cases.len(), 56);
    for case in cases {
        let salt = unhex(s(&case["salt"]));
        let mut okm = vec![0u8; case["len"].as_u64().unwrap() as usize];
        kdf::hkdf_sha256(&unhex(s(&case["ikm"])), (!salt.is_empty()).then_some(salt.as_slice()), &unhex(s(&case["info"])), &mut okm).unwrap();
        assert_eq!(hex(&okm), s(&case["okm"]));
    }
}

#[test]
fn argon2id_across_the_bounds_equals_libsodium_and_openssl() {
    let o = oracle();
    let cases = o["argon2id"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        // libsodium and node's OpenSSL agreed when the file was made (checked by the generator)
        assert_eq!(case["out"], case["node_openssl"]);
        let out = argon::argon2id13(&unhex(s(&case["pwd"])), &unhex(s(&case["salt"])), case["ops"].as_u64().unwrap(), case["mem"].as_u64().unwrap())
            .unwrap();
        assert_eq!(hex(out.expose()), s(&case["out"]), "ops {} mem {}", case["ops"], case["mem"]);
    }
    // both corners of the box are among them
    assert!(cases.iter().any(|c| c["ops"] == 10 && c["mem"].as_u64() == Some(256 * 1024 * 1024)));
    assert!(cases.iter().any(|c| c["ops"] == 3 && c["mem"].as_u64() == Some(64 * 1024 * 1024)));
    assert!(cases.iter().any(|c| s(&c["pwd"]).is_empty()), "the empty password");
}
