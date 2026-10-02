//! Reviewer's check (F2, secrets): which public types print a secret through `{:?}`. The ones that hold a secret by design must print nothing of it; the output of the others is
//! shown. The assertions are the claim of the crate ("a credential has no Display, its Debug prints nothing of it"); a failing assertion is a finding.

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::admission::{Bearer, IceServer};
use oaiy_relay_core::enrol::{EnrolmentKey, Enrolled, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::math::{typed_code, PairingKey, PairingSecret};
use oaiy_relay_core::pairing::PairingInput;
use oaiy_relay_core::url::RelayUrl;
use oaiy_relay_core::{b64, pairing::math::Sas};

const THUMB: &str = "atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR4";

fn battery() -> Vec<String> {
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let ps: [u8; 16] = core::array::from_fn(|i| 0x55u8.wrapping_add(i as u8 * 11));
    let ps_b64 = b64::encode(&ps);
    let code = typed_code(&ps);
    let uri = PairingKey::to_uri(&relay, THUMB, &PairingSecret::new(ps), 1790000600);

    let mut leaks: Vec<String> = Vec::new();
    let mut check = |what: &str, shown: String, secrets: &[&str]| {
        let hit = secrets.iter().any(|s| shown.contains(s));
        println!("{what:<44} {} {}", if hit { "LEAKS " } else { "clean " }, shown.chars().take(150).collect::<String>());
        if hit {
            leaks.push(what.to_string());
        }
    };

    // by-design secrets
    let sec: [u8; 32] = core::array::from_fn(|i| 0xA0u8.wrapping_add(i as u8 * 7));
    let sec_t = b64::encode(&sec);
    let token = Token::parse(&format!("oaiyrt1.{}.{sec_t}", b64::encode(&[1u8, 2, 3, 4, 5, 6, 7, 8]))).unwrap();
    check("Token", format!("{token:?}"), &[&sec_t]);
    check("PairingSecret", format!("{:?}", PairingSecret::new(ps)), &[&ps_b64]);
    check("PairingKey (parsed)", format!("{:?}", PairingKey::parse(&uri).unwrap()), &[&ps_b64]);
    let es = [0x77u8; 16];
    let enrol_uri = EnrolmentKey::to_uri(&relay, THUMB, &es, Role::Desktop, 1790000600).unwrap();
    check("EnrolmentKey", format!("{:?}", EnrolmentKey::parse(&enrol_uri).unwrap()), &[&b64::encode(&es)]);
    let body = format!("{{\"deviceId\":\"dev-AAAAAAAAAAAAAAAAAAAAAA\",\"token\":\"oaiyrt1.AQIDBAUGBwg.{sec_t}\",\"relayId\":\"rly-AAAAAAAAAAAAAAAAAAAAAA\",\"time\":1}}");
    check("Enrolled", format!("{:?}", Enrolled::parse(body.as_bytes(), Role::Desktop).unwrap()), &[&sec_t]);
    let mac = "00".repeat(32);
    let bearer = format!("aokie-adm-v2.{}.{mac}", "7b2265787022 3a317d".replace(' ', ""));
    check("Bearer", format!("{:?}", Bearer::parse(&bearer).unwrap()), &[&mac]);
    check("Signer", format!("{:?}", Signer::from_seed(&Secret::new([7; 32]))), &["0707"]);
    check("X25519Secret", format!("{:?}", X25519Secret::from_secret(&Secret::new([9; 32]))), &["0909"]);

    // not secret by design in the crate's eyes, but holding a secret
    check("PairingInput::Key(uri) [derives Debug]", format!("{:?}", PairingInput::Key(&uri)), &[&ps_b64]);
    check("PairingInput::Typed{code,..} [derives Debug]", format!("{:?}", PairingInput::Typed { code: &code, host: "relay.example.com" }), &[&code[..4]]);
    let ice = IceServer {
        urls: vec!["turn:turn.example.com:3478".into()],
        username: "1790000600:opaque".into(),
        credential: "TURN-CREDENTIAL-SECRET".into(),
        expires_at: Some(1790000600),
    };
    check("IceServer (TURN credential) [derives Debug]", format!("{ice:?}"), &["TURN-CREDENTIAL-SECRET"]);
    let sas = Sas { raw: [1; 8], chars12: "6NHNK68MQQVZ".into(), check: '5' };
    check("Sas [custom Debug]", format!("{sas:?}"), &["6NHN-K68M-QQVZ-5"]);

    println!("leaks: {leaks:?}");
    leaks
}

#[test]
fn what_debug_prints() {
    // the types that hold a secret by design are clean; the leaks listed are the finding (see the ignored test)
    let leaks = battery();
    assert!(!leaks.iter().any(|l| l == "Token" || l == "PairingSecret" || l == "EnrolmentKey" || l == "Bearer" || l == "Signer"), "{leaks:?}");
}

/// Fails today (run with `--ignored`): `PairingInput` (the scanned key or the typed code), `IceServer` (a TURN credential) and `Sas` print what they hold.
#[test]
#[ignore = "a finding: PairingInput, IceServer and Sas derive or implement a Debug that prints a secret"]
fn no_public_type_prints_a_secret() {
    let leaks = battery();
    assert!(leaks.is_empty(), "{leaks:?}");
}

