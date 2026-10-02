//! Review test (F1): `sealed1` against libsodium. `rv/sealed/seal_gen.php` (PHP, libsodium) writes sealed boxes and damaged copies with libsodium's own verdict on each; this test
//! opens every one with the crate and requires the same verdict and the same plaintext, then seals messages of many sizes with the crate and writes them for
//! `rv/sealed/seal_check.php` to open with `sodium_crypto_box_seal_open`. Set `RV_SEAL_IN` and `RV_SEAL_OUT`; without them the test says so and does nothing.

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::keys::{X25519Public, X25519Secret};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn huge_and_degenerate_inputs_are_refused_without_a_panic_and_a_big_box_round_trips() {
    let secret = X25519Secret::generate().unwrap();
    for n in [0usize, 1, 31, 32, 47, 48, 49, 100, 8 << 20] {
        assert!(secret.open_sealed(&vec![0x41; n]).is_err(), "{n} bytes of 0x41");
        assert!(secret.open_sealed(&vec![0; n]).is_err(), "{n} zero bytes");
    }
    assert!(oaiy_relay_core::sealed::open_token(&secret, &"A".repeat(100_000)).is_err());
    assert!(oaiy_relay_core::sealed::open_token(&secret, "").is_err());
    let plain = vec![0x5au8; 8 << 20];
    let sealed = secret.public_key().seal(&plain).unwrap();
    assert_eq!(secret.open_sealed(&sealed).unwrap().expose(), &plain[..]);
    let mut damaged = sealed.clone();
    *damaged.last_mut().unwrap() ^= 1;
    assert!(secret.open_sealed(&damaged).is_err());
    let mut damaged = sealed;
    damaged[40] ^= 0x80;
    assert!(secret.open_sealed(&damaged).is_err());
}

#[test]
fn libsodiums_sealed_boxes_open_here_and_the_crates_open_in_libsodium() {
    let (Ok(input), Ok(output)) = (std::env::var("RV_SEAL_IN"), std::env::var("RV_SEAL_OUT")) else {
        eprintln!("SKIPPED: RV_SEAL_IN and RV_SEAL_OUT are not set");
        return;
    };
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let sk: [u8; 32] = unhex(doc["sk"].as_str().unwrap()).try_into().unwrap();
    let pk: [u8; 32] = unhex(doc["pk"].as_str().unwrap()).try_into().unwrap();
    let secret = X25519Secret::from_secret(&Secret::new(sk));
    assert_eq!(secret.public_key().to_bytes(), pk, "the clamped public key of libsodium's seed keypair");

    let mut disagreements = Vec::new();
    let mut valid = 0;
    for c in doc["cases"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let sealed = unhex(c["sealed"].as_str().unwrap());
        let theirs = c["plain"].as_str().map(str::to_string);
        let ours = secret.open_sealed(&sealed).ok().map(|p| hex(p.expose()));
        if theirs.is_some() {
            valid += 1;
        }
        if ours != theirs {
            disagreements.push(format!("{label}: libsodium {theirs:?}, crate {ours:?}"));
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} disagreements out of {} cases ({valid} valid): {disagreements:#?}",
        disagreements.len(),
        doc["cases"].as_array().unwrap().len()
    );
    assert!(valid >= 12);

    // the other direction: boxes made here, for libsodium to open
    let recipient = X25519Public::from_bytes(&pk).unwrap();
    let mut made = Vec::new();
    for n in [0usize, 1, 2, 31, 32, 33, 63, 111, 255, 256, 1000, 65536] {
        let plain: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        let sealed = recipient.seal(&plain).unwrap();
        assert_eq!(sealed.len(), n + 48);
        made.push(serde_json::json!({ "plain": hex(&plain), "sealed": hex(&sealed) }));
    }
    std::fs::write(output, serde_json::to_vec(&made).unwrap()).unwrap();
}
