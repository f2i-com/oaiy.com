//! Known-answer tests for every primitive from public sources, extracted from the published texts by `tests/vectors/scripts/extract_public.py`
//! and recomputed there by an independent implementation before they were written to `public-vectors.json`:
//! RFC 5869 (HKDF-SHA256), RFC 8032 (Ed25519, the five vectors of section 7.1), RFC 7748 (X25519, section 5.2 and 6.1 and the iterated
//! vectors), the XChaCha20-Poly1305 test vector of draft-irtf-cfrg-xchacha-03 (A.3.1), and the 128-bit vectors of the BIP-39
//! reference (trezor's `vectors.json`). RFC 9106 (Argon2id) is in `src/argon.rs`, because its parameters (four lanes, a secret, associated
//! data) are outside this crate's public bounds.

mod common;

use common::*;
use oaiy_crypto::aead;
use oaiy_crypto::bip39::{self, Entropy};
use oaiy_crypto::ed25519::{KeyRole, Signature, SigningKey, VerifyingKey};
use oaiy_crypto::kdf::hkdf_sha256;
use oaiy_crypto::x25519::{PublicKey, SecretKey};
use oaiy_crypto::zeroize::Secret;
use oaiy_crypto::Error;

fn vectors() -> serde_json::Value {
    json(PUBLIC_VECTORS)
}

fn s(value: &serde_json::Value) -> &str {
    value.as_str().unwrap()
}

#[test]
fn provenance_names_every_source_with_its_hash() {
    let v = vectors();
    let sources = v["meta"]["sources"].as_object().unwrap();
    for key in ["rfc5869", "rfc8032", "rfc7748", "rfc9106", "xchacha_draft_03", "bip39_vectors", "bip39_english"] {
        let source = &sources[key];
        assert!(s(&source["url"]).starts_with("https://"), "{key}");
        assert_eq!(s(&source["sha256"]).len(), 64, "{key}");
    }
    assert_eq!(s(&sources["bip39_english"]["sha256"]), bip39::WORDLIST_SHA256, "the official list is the embedded list");
}

#[test]
fn rfc_5869_hkdf_sha256_test_cases_1_to_3() {
    let v = vectors();
    let cases = v["hkdf_sha256_rfc5869"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let salt = unhex(s(&case["salt"]));
        let mut okm = vec![0u8; case["len"].as_u64().unwrap() as usize];
        // test case 3 has an empty salt: "not provided" is a string of HashLen zeros, and so is an empty one
        hkdf_sha256(&unhex(s(&case["ikm"])), (!salt.is_empty()).then_some(salt.as_slice()), &unhex(s(&case["info"])), &mut okm).unwrap();
        assert_eq!(hex(&okm), s(&case["okm"]), "test case {}", case["case"]);
        // Some(empty) is the same as None
        let mut again = vec![0u8; okm.len()];
        hkdf_sha256(&unhex(s(&case["ikm"])), Some(&salt), &unhex(s(&case["info"])), &mut again).unwrap();
        assert_eq!(again, okm);
    }
    // RFC 5869 section 2.3: at most 255 blocks of output
    let mut max = vec![0u8; 255 * 32];
    assert!(hkdf_sha256(b"ikm", None, b"", &mut max).is_ok());
    let mut too_long = vec![0u8; 255 * 32 + 1];
    assert_eq!(hkdf_sha256(b"ikm", None, b"", &mut too_long).unwrap_err(), Error::HkdfLength);
}

#[test]
fn rfc_8032_ed25519_test_vectors() {
    let v = vectors();
    let cases = v["ed25519_rfc8032"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for case in cases {
        let name = s(&case["name"]);
        let key = SigningKey::from_seed(KeyRole::Hazmat, &secret(s(&case["seed"])));
        let public = key.verifying_key();
        assert_eq!(hex(&public.to_bytes()), s(&case["public"]), "{name} public key");
        let message = unhex(s(&case["message"]));
        let signature = key.sign_raw(&message).unwrap();
        assert_eq!(hex(&signature.to_bytes()), s(&case["signature"]), "{name} signature");
        public.verify_raw(&message, &signature).unwrap();
        VerifyingKey::from_bytes(&arr(s(&case["public"])))
            .unwrap()
            .verify_raw(&message, &Signature::from_slice(&unhex(s(&case["signature"]))).unwrap())
            .unwrap();
        // any single flipped bit of the signature, the message or the key is refused
        let mut sig = signature.to_bytes();
        sig[0] ^= 1;
        assert!(public.verify_raw(&message, &Signature::from_bytes(&sig)).is_err(), "{name} flipped R");
        let mut other = message.clone();
        other.push(0);
        assert!(public.verify_raw(&other, &signature).is_err(), "{name} extended message");
    }
    assert_eq!(unhex(s(&cases[3]["message"])).len(), 1023, "TEST 1024 is the 1023-byte message");
}

fn x25519(scalar: &str, u: &str) -> String {
    let secret = SecretKey::from_bytes(arr(scalar));
    let public = PublicKey::from_bytes(&arr(u)).unwrap();
    hex(secret.diffie_hellman(&public).unwrap().expose())
}

#[test]
fn rfc_7748_x25519_section_5_2_and_6_1() {
    let v = vectors();
    let x = &v["x25519_rfc7748"];
    for case in x["scalarmult"].as_array().unwrap() {
        assert_eq!(x25519(s(&case["scalar"]), s(&case["u"])), s(&case["out"]));
    }
    let dh = &x["dh"];
    let alice = SecretKey::from_bytes(arr(s(&dh["alice_private"])));
    let bob = SecretKey::from_bytes(arr(s(&dh["bob_private"])));
    assert_eq!(hex(alice.public_key().as_bytes()), s(&dh["alice_public"]));
    assert_eq!(hex(bob.public_key().as_bytes()), s(&dh["bob_public"]));
    assert_eq!(hex(alice.diffie_hellman(&bob.public_key()).unwrap().expose()), s(&dh["shared"]));
    assert_eq!(hex(bob.diffie_hellman(&alice.public_key()).unwrap().expose()), s(&dh["shared"]));
    // the iterated vectors: k = u = 9; each step k, u = X25519(k, u), k
    let it = &x["iterated"];
    let (mut k, mut u) = (s(&it["k"]).to_string(), s(&it["u"]).to_string());
    for i in 1..=1000 {
        let next = x25519(&k, &u);
        u = k;
        k = next;
        if i == 1 {
            assert_eq!(k, s(&it["after_1"]));
        }
    }
    assert_eq!(k, s(&it["after_1000"]));
}

/// The million-iteration vector of RFC 7748 (about a minute unoptimised): `cargo test --release -p oaiy-crypto -- --ignored`.
#[test]
#[ignore = "one million X25519 iterations; run with --release --ignored"]
fn rfc_7748_x25519_one_million_iterations() {
    let v = vectors();
    let it = &v["x25519_rfc7748"]["iterated"];
    let (mut k, mut u): ([u8; 32], [u8; 32]) = (arr(s(&it["k"])), arr(s(&it["u"])));
    for _ in 0..1_000_000 {
        let next = SecretKey::from_bytes(k).diffie_hellman(&PublicKey::from_bytes(&u).unwrap()).unwrap();
        u = k;
        k = *next.expose();
    }
    assert_eq!(hex(&k), s(&it["after_1000000"]));
}

#[test]
fn xchacha20_poly1305_draft_irtf_cfrg_xchacha_03_appendix_a_3_1() {
    let v = vectors();
    let x = &v["xchacha20poly1305_draft03"];
    let key: Secret<32> = secret(s(&x["key"]));
    let nonce: [u8; 24] = arr(s(&x["nonce"]));
    let aad = unhex(s(&x["aad"]));
    let plaintext = unhex(s(&x["plaintext"]));
    assert!(plaintext.starts_with(b"Ladies and Gentlemen of the class of '99"));
    let sealed = aead::seal(&key, aead::Nonce::from_bytes_for_tests(nonce), &aad, &plaintext).unwrap();
    assert_eq!(hex(&sealed[..sealed.len() - 16]), s(&x["ciphertext"]));
    assert_eq!(hex(&sealed[sealed.len() - 16..]), s(&x["tag"]));
    assert_eq!(aead::open(&key, &nonce, &aad, &sealed).unwrap().expose(), plaintext.as_slice());
}

#[test]
fn bip39_official_128_bit_vectors() {
    let v = vectors();
    let cases = v["bip39_trezor_128"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let entropy = Entropy::from_bytes(arr(s(&case["entropy"])));
        assert_eq!(bip39::encode(&entropy).expose(), s(&case["mnemonic"]));
        assert_eq!(bip39::decode(s(&case["mnemonic"])).unwrap(), entropy);
    }
    // the four the design names
    assert_eq!(s(&cases[0]["mnemonic"]), "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about");
    assert_eq!(s(&cases[3]["mnemonic"]), "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong");
}
