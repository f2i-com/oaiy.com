// Opens the recorded sealed tokens of fixtures/sealed-token.json with the RustCrypto `crypto_box` crate (the one the desktop
// depends on), and checks that the boxes that must be refused are refused.
//
// `SecretKey::unseal` does NOT refuse an ephemeral key of small order: a box that authenticates under the all-zero shared secret
// (refused[8]) opens with it. A reader must therefore check the shared secret itself, as `open` below does with curve25519-dalek
// (which crypto_box builds on): an all-zero result is a refusal, as in libsodium's crypto_box_beforenm.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use crypto_box::SecretKey;
use curve25519_dalek::montgomery::MontgomeryPoint;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn key(v: &Value) -> [u8; 32] {
    let raw = URL_SAFE_NO_PAD.decode(v.as_str().unwrap()).unwrap();
    raw.try_into().unwrap()
}

/// What a reader must do: refuse a short box and a zero shared secret, then open.
fn open(secret: &[u8; 32], sk: &SecretKey, sealed: &[u8]) -> Option<Vec<u8>> {
    if sealed.len() < 48 {
        return None;
    }
    let epk: [u8; 32] = sealed[..32].try_into().unwrap();
    if MontgomeryPoint(epk).mul_clamped(*secret).0 == [0u8; 32] {
        return None;
    }
    sk.unseal(sealed).ok()
}

fn main() {
    let path = std::env::args().nth(1).expect("path of sealed-token.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let secret = key(&doc["recipient"]["x25519Secret"]);
    let sk = SecretKey::from(secret);
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
        match open(&secret, &sk, &sealed) {
            Some(plain) => {
                check(format!("opens[{i}] length"), plain.len() as u64 == c["plaintextLength"].as_u64().unwrap());
                let hash: String = Sha256::digest(&plain).iter().map(|b| format!("{b:02x}")).collect();
                check(format!("opens[{i}] sha256"), hash == c["plaintextSha256"].as_str().unwrap());
                check(format!("opens[{i}] is a device token"), plain.starts_with(b"oaiyrt1."));
            }
            None => check(format!("opens[{i}] opens"), false),
        }
    }
    let mut unseal_alone_opens = 0;
    for (i, c) in doc["refused"].as_array().unwrap().iter().enumerate() {
        let sealed = URL_SAFE_NO_PAD.decode(c["sealedToken"].as_str().unwrap()).unwrap_or_default();
        check(format!("refused[{i}] {}", c["label"]), open(&secret, &sk, &sealed).is_none());
        if sk.unseal(&sealed).is_ok() {
            unseal_alone_opens += 1;
            println!("note: unseal alone OPENS refused[{i}]: only the shared-secret check refuses it");
        }
    }
    // refused[8] (u = 0) opens with unseal alone. refused[9] (an order-8 point) does not: crypto_box multiplies by the scalar reduced
    // modulo the group order, so a torsion point gives it a non-zero shared secret where RFC 7748 and libsodium give zero.
    check("unseal alone opens the small-order box that authenticates under the zero key (the reason the reader checks)".to_string(), unseal_alone_opens >= 1);
    let wrong = SecretKey::from(key(&doc["wrongRecipient"]["x25519Secret"]));
    let first = URL_SAFE_NO_PAD.decode(doc["opens"][0]["sealedToken"].as_str().unwrap()).unwrap();
    check("the wrong recipient does not open the first token".to_string(), wrong.unseal(&first).is_err());
    println!("{checks} checks, {bad} mismatches");
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
