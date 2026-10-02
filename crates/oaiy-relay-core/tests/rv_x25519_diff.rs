//! Review test (F1): which X25519 public keys the crate refuses and what its Diffie-Hellman gives, against libsodium (`rv/x25519/x25519_gen.php`: `sodium_crypto_scalarmult` on 3,000 random u-coordinates, the fourteen
//! low-order encodings and the values at the edge of the field). Set `RV_X_IN`; without it the test says so and does nothing.

use oaiy_crypto::x25519::SecretKey;
use oaiy_relay_core::keys::X25519Public;

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    let v: Vec<u8> = (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
    v.try_into().unwrap()
}

#[test]
fn the_crate_refuses_what_libsodium_refuses_and_agrees_on_the_rest() {
    let Ok(input) = std::env::var("RV_X_IN") else {
        eprintln!("SKIPPED: RV_X_IN is not set");
        return;
    };
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let sk = SecretKey::from_bytes(unhex(doc["sk"].as_str().unwrap()));
    let (mut refused, mut accepted) = (0, 0);
    for c in doc["cases"].as_array().unwrap() {
        let u: [u8; 32] = unhex(c["u"].as_str().unwrap());
        let theirs = c["dh"].as_str();
        let key = X25519Public::from_bytes(&u);
        match theirs {
            None => {
                refused += 1;
                assert!(key.is_err(), "{}: libsodium refuses, the crate's key type accepts", c["label"]);
            }
            Some(dh) => {
                accepted += 1;
                let key = key.unwrap_or_else(|e| panic!("{}: libsodium accepts, the crate refuses ({e:?})", c["label"]));
                let peer = oaiy_crypto::x25519::PublicKey::from_bytes(&key.to_bytes()).unwrap();
                let ours = sk.diffie_hellman(&peer).unwrap_or_else(|e| panic!("{}: dh refused ({e:?})", c["label"]));
                let hex: String = ours.expose().iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(hex, dh, "{}: shared secret", c["label"]);
            }
        }
    }
    assert!(refused >= 14 && accepted >= 3000, "{refused} {accepted}");
}
