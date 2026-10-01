//! Nonces (review M-5): `aead::wrap` draws a fresh random 24-byte nonce for every call, and nothing tested it. The reviewer's mutant that made `wrap` use an
//! all-zero nonce, so that every wrap under one key reused its nonce (which with XChaCha20-Poly1305 gives away the XOR of two plaintexts and the authentication key),
//! passed the whole suite.
//!
//! What can be tested about a random nonce from outside: that two wraps of the same plaintext differ; that over 10,000 wraps under one key no nonce repeats (a
//! collision among 10,000 random 192-bit values has a probability below 2^-160, so one is a bug and not bad luck); that the nonce is not all zero; that every byte of
//! it varies (a counter, or a nonce with a fixed part, fails this: each of the 24 positions must take at least 200 of the 256 values over 10,000 draws; a uniform
//! byte takes all of them with overwhelming probability); and that the bits are balanced (the total of set bits is within five standard deviations of half).
//! That the bytes come from the operating system's generator is `getrandom`'s job and is checked by its own tests; what is checked here is that they are what a
//! generator's output looks like, and differ every time.

use std::collections::HashSet;

use oaiy_crypto::aead::{self, NONCE_LEN};
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::zeroize::Secret;

const WRAPS: usize = 10_000;

fn aad() -> Aad {
    Aad::new(AadDomain::VaultWrap, &["user", "wrapper", "recovery-phrase", "x"]).unwrap()
}

#[test]
fn ten_thousand_wraps_of_one_plaintext_under_one_key_use_ten_thousand_different_random_nonces() {
    let key = Secret::<32>::new([0x42; 32]);
    let aad = aad();
    let plaintext = [0x99u8; 32];
    let mut nonces = HashSet::new();
    let mut blobs = HashSet::new();
    let mut seen_per_position = vec![HashSet::new(); NONCE_LEN];
    let mut set_bits = 0u64;
    for i in 0..WRAPS {
        let blob = aead::wrap(&key, &aad, &plaintext).unwrap();
        assert_eq!(blob.len(), NONCE_LEN + plaintext.len() + aead::TAG_LEN);
        let nonce: [u8; NONCE_LEN] = blob[..NONCE_LEN].try_into().unwrap();
        assert_ne!(nonce, [0u8; NONCE_LEN], "wrap {i} used an all-zero nonce");
        assert!(nonces.insert(nonce), "wrap {i} repeated a nonce");
        assert!(blobs.insert(blob.clone()), "wrap {i} produced a blob that an earlier wrap had produced");
        for (position, byte) in nonce.iter().enumerate() {
            seen_per_position[position].insert(*byte);
        }
        set_bits += nonce.iter().map(|b| u64::from(b.count_ones())).sum::<u64>();
        // and it opens, with the nonce it carries
        if i % 500 == 0 {
            assert_eq!(aead::unwrap(&key, &aad, &blob).unwrap().expose(), plaintext);
        }
    }
    assert_eq!(nonces.len(), WRAPS);
    for (position, seen) in seen_per_position.iter().enumerate() {
        assert!(seen.len() >= 200, "byte {position} of the nonce took only {} of 256 values in {WRAPS} wraps: it is not random", seen.len());
    }
    let bits = (WRAPS * NONCE_LEN * 8) as f64;
    let deviation = (bits / 4.0).sqrt(); // the standard deviation of the number of set bits
    assert!((set_bits as f64 - bits / 2.0).abs() < 5.0 * deviation, "{set_bits} of {bits} nonce bits are set: not balanced");
}

#[test]
fn wrapping_a_key_twice_gives_different_blobs_and_different_nonces_and_both_unwrap() {
    let wrapping = Secret::<32>::new([0x11; 32]);
    let key = Secret::<32>::new([0x22; 32]);
    let aad = aad();
    let first = aead::wrap_key(&wrapping, &aad, &key).unwrap();
    let second = aead::wrap_key(&wrapping, &aad, &key).unwrap();
    assert_ne!(first, second);
    assert_ne!(first[..NONCE_LEN], second[..NONCE_LEN], "the same nonce twice");
    assert_ne!(first[NONCE_LEN..], second[NONCE_LEN..], "the same ciphertext twice: the nonce did not change it");
    assert_eq!(aead::unwrap_key(&wrapping, &aad, &first).unwrap(), key);
    assert_eq!(aead::unwrap_key(&wrapping, &aad, &second).unwrap(), key);
}
