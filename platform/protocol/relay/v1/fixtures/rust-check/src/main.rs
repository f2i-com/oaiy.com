// Opens the recorded sealed tokens of fixtures/sealed-token.json with the RustCrypto `crypto_box` crate (the one the desktop
// depends on), and checks that the boxes that must be refused are refused.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use crypto_box::SecretKey;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn key(v: &Value) -> [u8; 32] {
    let raw = URL_SAFE_NO_PAD.decode(v.as_str().unwrap()).unwrap();
    raw.try_into().unwrap()
}

fn main() {
    let path = std::env::args().nth(1).expect("path of sealed-token.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let sk = SecretKey::from(key(&doc["recipient"]["x25519Secret"]));
    let (mut checks, mut bad) = (0, 0);
    let mut check = |label: String, ok: bool| {
        checks += 1;
        if !ok {
            bad += 1;
            println!("MISMATCH {label}");
        }
    };
    for (i, c) in doc["opens"].as_array().unwrap().iter().enumerate() {
        let sealed = URL_SAFE_NO_PAD.decode(c["sealedToken"].as_str().unwrap()).unwrap();
        match sk.unseal(&sealed) {
            Ok(plain) => {
                check(format!("opens[{i}] length"), plain.len() as u64 == c["plaintextLength"].as_u64().unwrap());
                let hash: String = Sha256::digest(&plain).iter().map(|b| format!("{b:02x}")).collect();
                check(format!("opens[{i}] sha256"), hash == c["plaintextSha256"].as_str().unwrap());
                check(format!("opens[{i}] is a device token"), plain.starts_with(b"oaiyrt1."));
            }
            Err(_) => check(format!("opens[{i}] opens"), false),
        }
    }
    for (i, c) in doc["refused"].as_array().unwrap().iter().enumerate() {
        let sealed = URL_SAFE_NO_PAD.decode(c["sealedToken"].as_str().unwrap()).unwrap_or_default();
        check(format!("refused[{i}] {}", c["label"]), sk.unseal(&sealed).is_err());
    }
    let wrong = SecretKey::from(key(&doc["wrongRecipient"]["x25519Secret"]));
    let first = URL_SAFE_NO_PAD.decode(doc["opens"][0]["sealedToken"].as_str().unwrap()).unwrap();
    check("the wrong recipient does not open the first token".to_string(), wrong.unseal(&first).is_err());
    println!("{checks} checks, {bad} mismatches");
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
