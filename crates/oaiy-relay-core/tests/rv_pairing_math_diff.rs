//! Review test (F1): the arithmetic of pairing v3 (`pairing::math`) against an independent recomputation from the README text (`rv/pairing_math/gen_math.py`, Python's
//! `hashlib` and `hmac` and its own HKDF): the pid and MAC key, the typed code, the SAS with its check character, the two MACs and the approval receipt's text, for 300 random
//! inputs, the reading of 60 variants of a typed code and the judgement of SAS entries. Set `RV_MATH_IN`; without it the test says so and does nothing.

use oaiy_crypto::kdf::hmac_sha256;
use oaiy_relay_core::pairing::math::{self, PairingSecret, SasEntry};

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    let v: Vec<u8> = (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
    v.try_into().unwrap()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn the_pairing_arithmetic_equals_an_independent_recomputation() {
    let Ok(input) = std::env::var("RV_MATH_IN") else {
        eprintln!("SKIPPED: RV_MATH_IN is not set");
        return;
    };
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let mut n = 0;
    for c in doc["cases"].as_array().unwrap() {
        let e = &c["expect"];
        let secret = PairingSecret::new(unhex::<16>(c["secret"].as_str().unwrap()));
        let d = secret.derive().unwrap();
        assert_eq!(hex(&d.pid), e["pid"].as_str().unwrap());
        assert_eq!(hex(d.mac_key.expose()), e["mac_key"].as_str().unwrap());
        assert_eq!(secret.typed_code(), e["typed"].as_str().unwrap());
        let sas =
            math::sas(&unhex(c["dk"].as_str().unwrap()), &unhex(c["pk"].as_str().unwrap()), &unhex(c["nonce"].as_str().unwrap()), &d.pid).unwrap();
        assert_eq!(hex(&sas.raw), e["sas_raw"].as_str().unwrap());
        assert_eq!(sas.chars12, e["sas_chars12"].as_str().unwrap());
        assert_eq!(sas.check.to_string(), e["sas_check"].as_str().unwrap());
        assert_eq!(sas.display(), e["sas_display"].as_str().unwrap());
        assert_eq!(math::offer_mac(&d.mac_key, c["offer_text"].as_str().unwrap()).unwrap(), e["offer_mac"].as_str().unwrap());
        assert_eq!(math::response_mac(&d.mac_key, c["claims_text"].as_str().unwrap()).unwrap(), e["response_mac"].as_str().unwrap());
        math::verify_offer_mac(&d.mac_key, c["offer_text"].as_str().unwrap(), e["offer_mac"].as_str().unwrap()).unwrap();
        math::verify_response_mac(&d.mac_key, c["claims_text"].as_str().unwrap(), e["response_mac"].as_str().unwrap()).unwrap();
        // a MAC of the other kind over the same text does not verify
        assert!(math::verify_offer_mac(&d.mac_key, c["claims_text"].as_str().unwrap(), e["response_mac"].as_str().unwrap()).is_err());
        let grants: Vec<String> = c["grants"].as_array().unwrap().iter().map(|g| g.as_str().unwrap().to_string()).collect();
        let text = math::receipt_text(
            c["app_id"].as_str().unwrap(),
            &grants,
            c["issued_at"].as_u64().unwrap(),
            c["phone_thumbprint"].as_str().unwrap(),
            &math::pid_text(&d.pid),
        )
        .unwrap();
        assert_eq!(text, e["receipt_text"].as_str().unwrap());
        // the MAC is not the plain HMAC of the text: the domain and the zero byte are in it
        let plain = hmac_sha256(d.mac_key.expose(), c["offer_text"].as_str().unwrap().as_bytes()).unwrap();
        assert_ne!(oaiy_relay_core::b64::encode(&plain), e["offer_mac"].as_str().unwrap());
        // the typed code of the secret reads back to the secret
        assert_eq!(hex(math::parse_typed_code(&secret.typed_code()).unwrap().expose()), c["secret"].as_str().unwrap());
        n += 1;
    }
    assert_eq!(n, 300);

    let mut parsed = 0;
    for t in doc["typed"].as_array().unwrap() {
        let got = math::parse_typed_code(t["input"].as_str().unwrap()).ok().map(|s| hex(s.expose()));
        let want = t["expect"].as_str().map(str::to_string);
        assert_eq!(got, want, "typed code {:?}", t["input"]);
        if want.is_some() {
            parsed += 1;
        }
    }
    assert!(parsed >= 8, "{parsed}");

    let s = &doc["sas"];
    let sas = math::sas(
        &unhex(s["dk"].as_str().unwrap()),
        &unhex(s["pk"].as_str().unwrap()),
        &unhex(s["nonce"].as_str().unwrap()),
        &unhex(s["pid"].as_str().unwrap()),
    )
    .unwrap();
    assert_eq!(sas.display(), s["display"].as_str().unwrap());
    for e in s["entries"].as_array().unwrap() {
        let got = match math::judge_sas_entry(&sas, e["entry"].as_str().unwrap()) {
            SasEntry::Incomplete => "Incomplete",
            SasEntry::Invalid => "Invalid",
            SasEntry::BadCheck => "BadCheck",
            SasEntry::Wrong => "Wrong",
            SasEntry::Right => "Right",
        };
        assert_eq!(got, e["expect"].as_str().unwrap(), "entry {:?}", e["entry"]);
    }
}
