//! Review test (F1): the crate's Ed25519 verdicts on hand-built edge cases, written to a JSON file so that a script can set them beside libsodium's (PHP), OpenSSL's (Node and
//! Python `cryptography`). The cases are made by `rv/ed25519/gen_cases.py`; this test only runs them. Set `RV_ED_CASES` (input) and `RV_ED_OUT` (output); without them the
//! test says so and does nothing.

use oaiy_crypto::ed25519::Signature;
use oaiy_relay_core::keys::VerifyKey;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

#[test]
fn the_crates_verdicts_on_the_edge_cases() {
    let (Ok(input), Ok(output)) = (std::env::var("RV_ED_CASES"), std::env::var("RV_ED_OUT")) else {
        eprintln!("SKIPPED: RV_ED_CASES and RV_ED_OUT are not set");
        return;
    };
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for c in cases.as_array().unwrap() {
        let pk: [u8; 32] = unhex(c["pk"].as_str().unwrap()).try_into().unwrap();
        let msg = unhex(c["msg"].as_str().unwrap());
        let sig: [u8; 64] = unhex(c["sig"].as_str().unwrap()).try_into().unwrap();
        let (key_ok, verify) = match VerifyKey::from_bytes(&pk) {
            Err(e) => (format!("refused:{e:?}"), serde_json::json!("keyrefused")),
            Ok(k) => ("ok".to_string(), serde_json::json!(k.verify_raw(&msg, &Signature::from_bytes(&sig)).is_ok())),
        };
        out.insert(c["id"].as_str().unwrap().to_string(), serde_json::json!({ "verify": verify, "key": key_ok }));
    }
    std::fs::write(output, serde_json::to_vec(&out).unwrap()).unwrap();

    // With libsodium's verdicts (`verdict_php.php`) beside them the test is also a check: the crate accepts exactly what libsodium accepts (a key it refuses to build counts as a
    // refusal), which is what makes the relay (libsodium) and the clients (this crate) judge one signature alike.
    if let Ok(path) = std::env::var("RV_ED_SODIUM") {
        let theirs: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut differ = Vec::new();
        for (id, ours) in &out {
            let ours_accepts = ours["verify"] == serde_json::json!(true);
            let theirs_accepts = theirs[id]["verify"] == serde_json::json!(true);
            if ours_accepts != theirs_accepts {
                differ.push(id.clone());
            }
        }
        assert!(differ.is_empty(), "the crate and libsodium judge these differently: {differ:#?}");
    }
}
