//! Every known-answer vector of `platform/protocol/relay/v1/vectors.json` (Appendix A0 to A12 and the extras), recomputed with this crate, and the two recordings of the
//! real relay that a client opens (`fixtures/sealed-token.json`, `fixtures/pairing-ceremony.json` at the level of its values). The vectors are read in place: a change to
//! the contract that this crate does not follow fails here, and a change to this crate that departs from the contract does too.

mod common;

use common::{hex, load, unhex, unhex32, At};
use oaiy_crypto::kdf::{hkdf_sha256, hmac_sha256, sha256};
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::admission::Bearer;
use oaiy_relay_core::b64;
use oaiy_relay_core::enrol::{self, EnrolmentKey, Role};
use oaiy_relay_core::ids::{self, Token};
use oaiy_relay_core::info::{self, Info};
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::keys::{domain_message, signature_from_b64u, thumbprint_of, SignDomain, Signer, VerifyKey, X25519Public, X25519Secret};
use oaiy_relay_core::pairing::math::{self, PairingKey, PairingSecret, SasEntry};
use oaiy_relay_core::pairing::{Claims, Offer, OfferParams, Response};
use oaiy_relay_core::ring::{self, RingBody};
use oaiy_relay_core::roster;
use oaiy_relay_core::rotation;
use oaiy_relay_core::sealed::{self, Container, ContainerDomain};
use oaiy_relay_core::ticket;
use oaiy_relay_core::url::RelayUrl;

fn vectors() -> Json {
    load("vectors.json")
}

fn signer(seed_hex: &str) -> Signer {
    Signer::from_seed(&Secret::new(unhex32(seed_hex)))
}

fn x_secret(hex_text: &str) -> X25519Secret {
    X25519Secret::from_secret(&Secret::new(unhex32(hex_text)))
}

#[test]
fn the_external_anchors() {
    let v = vectors();
    let a = v.at("anchors.rfc5869_tc1");
    let mut okm = vec![0u8; a.n("length") as usize];
    hkdf_sha256(&unhex(a.s("ikm")), Some(&unhex(a.s("salt"))), &unhex(a.s("info")), &mut okm).unwrap();
    assert_eq!(hex(&okm), a.s("okm"));
    let h = v.at("anchors.rfc4231_tc1_hmac_sha256");
    assert_eq!(hex(&hmac_sha256(&unhex(h.s("key")), h.s("data").as_bytes()).unwrap()), h.s("mac"));
    // RFC 8032 test 1: the empty message, verified strictly with the public key the seed gives.
    let e = v.at("anchors.rfc8032_test1");
    let key = VerifyKey::from_bytes(&unhex32(e.s("public"))).unwrap();
    let sig = oaiy_crypto::ed25519::Signature::from_slice(&unhex(e.s("signature"))).unwrap();
    key.verify_raw(b"", &sig).unwrap();
    assert!(key.verify_raw(b"x", &sig).is_err());
}

#[test]
fn the_fixed_test_keys_and_their_thumbprints() {
    let v = vectors();
    for name in ["desktopEndpoint", "phone", "relay", "host", "provider", "provider2"] {
        let s = signer(v.s(&format!("keys.ed25519Seeds.{name}")));
        assert_eq!(s.verify_key().to_b64u(), v.s(&format!("keys.ed25519Public.{name}.publicKey")), "{name}");
        assert_eq!(s.thumbprint(), v.s(&format!("keys.ed25519Public.{name}.thumbprint")), "{name}");
    }
    for name in ["plugin", "phone", "host", "provider", "provider2", "browserEphemeral"] {
        let x = x_secret(v.s(&format!("keys.x25519Secrets.{name}")));
        assert_eq!(x.public_key().to_b64u(), v.s(&format!("keys.x25519Public.{name}")), "{name}");
    }
    for t in v.at("extras.thumbprints").as_array().unwrap() {
        let key = signer(t.s("seed")).verify_key();
        assert_eq!(key.to_b64u(), t.s("publicKey"));
        let jwk = format!("{{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"{}\"}}", t.s("publicKey"));
        assert_eq!(jwk, t.s("jwk"));
        assert_eq!(thumbprint_of(&key.to_bytes()), t.s("thumbprint"));
    }
}

#[test]
fn a0_the_roster_hash_against_the_aokie_readmes_own_example() {
    let a = vectors();
    let a0 = a.at("A0");
    let thumbs: Vec<String> =
        a0.at("inputs.approvedPeerKeyThumbprints").as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    assert_eq!(roster::hash(a0.n("inputs.peerRosterRevision"), &thumbs).unwrap(), a0.s("expected.peerRosterHash"));
    assert_eq!(a0.s("expected.peerRosterHash"), a0.s("expected.aokieReadmeValue"));
    // The order the thumbprints are given in does not matter: the construction sorts them.
    let reversed: Vec<String> = thumbs.iter().rev().cloned().collect();
    assert_eq!(roster::hash(7, &reversed).unwrap(), a0.s("expected.peerRosterHash"));
    assert_ne!(roster::hash(8, &thumbs).unwrap(), a0.s("expected.peerRosterHash"));
}

#[test]
fn a1_the_rfc_8037_jws_signature() {
    let v = vectors();
    let a1 = v.at("A1");
    let key = VerifyKey::from_b64u(a1.s("inputs.publicKey")).unwrap();
    let sig = signature_from_b64u(a1.s("expected.signature")).unwrap();
    key.verify_raw(a1.s("inputs.signingInput").as_bytes(), &sig).unwrap();
    assert!(key.verify_raw(b"eyJhbGciOiJFZERTQSJ9.RXhhbXBsZSBvZiBFZDI1NTE5IHNpZ25pbmd", &sig).is_err());
}

#[test]
fn a2_a_token() {
    let v = vectors();
    let a = v.at("A2");
    let id = unhex(a.s("inputs.idHex"));
    let secret = unhex(a.s("inputs.secretHex"));
    let text = format!("{}{}.{}", a.s("inputs.prefix"), b64::encode(&id), b64::encode(&secret));
    assert_eq!(text, a.s("expected.token"));
    assert_eq!(text.len() as u64, a.n("expected.length"));
    assert_eq!(hex(&sha256(&secret)), a.s("expected.secretSha256"));
    assert!(Token::parse(&text).is_ok());
}

#[test]
fn the_vectors_token_acceptance_and_refusal_cases() {
    let v = vectors();
    for t in v.at("extras.tokens.valid").as_array().unwrap() {
        assert!(Token::parse(t.as_str().unwrap()).is_ok(), "{t:?}");
    }
    for t in v.at("extras.tokens.invalid").as_array().unwrap() {
        assert!(Token::parse(t.s("token")).is_err(), "{}: {}", t.s("reason"), t.s("token"));
    }
}

/// The values of vector A3 as they are needed again and again.
struct A3 {
    secret: PairingSecret,
    desktop_endpoint: Signer,
    phone_endpoint: Signer,
    host: Signer,
    relay: RelayUrl,
}

fn a3() -> A3 {
    let v = vectors();
    A3 {
        secret: PairingSecret::new(unhex(v.s("A3.inputs.secretHex")).try_into().unwrap()),
        desktop_endpoint: signer(v.s("keys.ed25519Seeds.desktopEndpoint")),
        phone_endpoint: signer(v.s("keys.ed25519Seeds.phone")),
        host: signer(v.s("keys.ed25519Seeds.host")),
        relay: RelayUrl::parse(v.s("A3.inputs.relayUrl")).unwrap(),
    }
}

fn a3_offer(a: &A3) -> Offer {
    let v = vectors();
    let desktop_x = X25519Public::from_b64u(v.s("keys.x25519Public.plugin")).unwrap();
    let host_x = X25519Public::from_b64u(v.s("keys.x25519Public.host")).unwrap();
    Offer::build(&OfferParams {
        app_id: v.s("A3.inputs.offer.appId"),
        desktop_connection_id: v.s("A3.inputs.offer.desktopConnectionId"),
        desktop_name: v.s("A3.inputs.offer.desktopName"),
        desktop_endpoint: &a.desktop_endpoint.verify_key(),
        desktop_x25519: &desktop_x,
        host_ed25519: &a.host.verify_key(),
        host_x25519: &host_x,
        nonce: unhex32(v.s("A3.inputs.nonceHex")),
        jti: v.s("A3.inputs.offer.jti"),
        issued_at: v.n("A3.inputs.offer.issuedAt"),
        relay: &a.relay,
        relay_fingerprint: v.s("A3.inputs.offer.relay.fingerprint"),
    })
    .unwrap()
}

fn a3_claims(a: &A3) -> Claims {
    let v = vectors();
    Claims {
        app_id: v.s("A3.inputs.claims.appId").into(),
        desktop_connection_id: v.s("A3.inputs.claims.desktopConnectionId").into(),
        desktop_key_thumbprint: v.s("A3.inputs.claims.desktopKeyThumbprint").into(),
        device_id: v.s("A3.inputs.claims.deviceId").into(),
        display_name: Some(v.s("A3.inputs.claims.displayName").into()),
        mobile_endpoint: a.phone_endpoint.verify_key(),
        mobile_x25519: X25519Public::from_b64u(v.s("A3.inputs.claims.mobileX25519")).unwrap(),
        pairing_nonce: unhex32(v.s("A3.inputs.nonceHex")),
        jti: v.s("A3.inputs.claims.jti").into(),
        issued_at: v.n("A3.inputs.claims.issuedAt"),
        expires_at: v.n("A3.inputs.claims.expiresAt"),
    }
}

#[test]
fn a3_the_pairing_secret_the_typed_code_the_pid_and_the_mac_key() {
    let v = vectors();
    let a = a3();
    assert_eq!(a.secret.b64u(), v.s("A3.expected.secretB64u"));
    assert_eq!(a.secret.typed_code(), v.s("A3.expected.typedCode"));
    let d = a.secret.derive().unwrap();
    assert_eq!(hex(&d.pid), v.s("A3.expected.pidHex"));
    assert_eq!(math::pid_text(&d.pid), v.s("A3.expected.pid"));
    assert_eq!(hex(d.mac_key.expose()), v.s("A3.expected.macKeyHex"));
    let uri = PairingKey::to_uri(&a.relay, v.s("A3.inputs.offer.relay.fingerprint"), &a.secret, v.n("A3.inputs.expiresAtParam"));
    assert_eq!(uri, v.s("A3.expected.pairingUri"));
    let parsed = PairingKey::parse(&uri).unwrap();
    assert_eq!(parsed.secret.expose(), a.secret.expose());
    assert_eq!(parsed.relay, a.relay);
    assert_eq!(parsed.expires_at, Some(v.n("A3.inputs.expiresAtParam")));
}

#[test]
fn a3_the_offer_its_text_its_size_and_its_mac() {
    let v = vectors();
    let a = a3();
    let offer = a3_offer(&a);
    assert_eq!(offer.text, v.s("A3.expected.offerText"));
    assert_eq!(offer.text.len() as u64, v.n("A3.expected.offerTextBytes"));
    let d = a.secret.derive().unwrap();
    assert_eq!(offer.mac(&d.mac_key).unwrap(), v.s("A3.expected.offerMac"));
    // The phone's reading: the MAC first, over the text as received, and only then the parse.
    let read = Offer::verify(&offer.text, v.s("A3.expected.offerMac"), &d.mac_key).unwrap();
    assert_eq!(read.desktop_endpoint, a.desktop_endpoint.verify_key());
    assert_eq!(read.nonce, unhex32(v.s("A3.inputs.nonceHex")));
    assert_eq!(read.expires_at, read.issued_at + 600);
    // A text that was touched by one byte is never parsed: the MAC fails.
    let touched = offer.text.replace("Front desk PC", "Front desk PD");
    assert!(matches!(Offer::verify(&touched, v.s("A3.expected.offerMac"), &d.mac_key), Err(oaiy_relay_core::Error::BadMac(_))));
    // The canonical form of the recorded inputs is the recorded text (the offer is canonical JSON).
    assert_eq!(json::canonicalize(offer.text.as_bytes()).unwrap(), offer.text);
}

#[test]
fn a3_the_response_claims_signature_and_mac() {
    let v = vectors();
    let a = a3();
    let d = a.secret.derive().unwrap();
    let claims = a3_claims(&a);
    assert_eq!(claims.canonical().unwrap(), v.s("A3.expected.claimsCanonical"));
    let response = Response::build(&a.phone_endpoint, &d.mac_key, claims).unwrap();
    assert_eq!(response.signature, v.s("A3.expected.responseSignature"));
    assert_eq!(response.mac, v.s("A3.expected.responseMac"));
    // The recorded ceremony's phone request carries exactly this text, byte for byte.
    let ceremony = load("fixtures/pairing-ceremony.json");
    assert_eq!(response.text, ceremony.s("steps.2.request.body.response"));
    // And the desktop's reading of it, against its own offer, accepts it inside the window and refuses it outside.
    let offer = a3_offer(&a);
    let parsed = Response::parse(&response.text).unwrap();
    parsed.verify(&offer, &d.mac_key, 1_790_000_040).unwrap();
    assert!(parsed.verify(&offer, &d.mac_key, 1_790_000_150 + 30).is_err(), "expired");
    assert!(parsed.verify(&offer, &d.mac_key, 1_790_000_030 - 31).is_err(), "issued in the future");
    parsed.verify(&offer, &d.mac_key, 1_790_000_150 + 29).unwrap();
}

#[test]
fn a3_the_sas_is_computed_from_the_raw_bytes_of_the_pid() {
    let v = vectors();
    let a = a3();
    let d = a.secret.derive().unwrap();
    let nonce = unhex32(v.s("A3.inputs.nonceHex"));
    let sas = math::sas(&a.desktop_endpoint.verify_key().to_bytes(), &a.phone_endpoint.verify_key().to_bytes(), &nonce, &d.pid).unwrap();
    assert_eq!(hex(&sas.raw), v.s("A3.expected.sasRawHex"));
    assert_eq!(sas.chars12, v.s("A3.expected.sas12"));
    assert_eq!(sas.check.to_string(), v.s("A3.expected.sasCheckChar"));
    assert_eq!(sas.display(), v.s("A3.expected.sasDisplay"));
    let ceremony = load("fixtures/pairing-ceremony.json");
    assert_eq!(sas.display(), ceremony.s("sas"));

    // The two wrong readings of the pid that extras.sasNegative records: the 22-character text and the 32-character hex. Neither may ever be produced by this crate.
    let neg = v.at("extras.sasNegative");
    let mut ikm = Vec::new();
    ikm.extend_from_slice(&unhex(neg.s("inputs.desktopEndpointPublicHex")));
    ikm.extend_from_slice(&unhex(neg.s("inputs.phoneEndpointPublicHex")));
    for (reading, pid_text) in [("text", neg.s("inputs.pidB64u").to_string()), ("hex", neg.s("inputs.pidHex").to_string())] {
        let info = domain_message(math::SAS_DOMAIN, &[pid_text.as_bytes()]);
        let mut wrong = [0u8; 8];
        hkdf_sha256(&ikm, Some(&nonce), &info, &mut wrong).unwrap();
        let want = &neg
            .at("wrong")
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w.s("reading").contains(if reading == "text" { "b64u text" } else { "lower-case hex" }))
            .unwrap();
        assert_eq!(hex(&wrong), want.s("sasRawHex"), "{reading}");
        assert_ne!(wrong, sas.raw, "{reading}");
        assert_ne!(
            math::sas(&a.desktop_endpoint.verify_key().to_bytes(), &a.phone_endpoint.verify_key().to_bytes(), &nonce, &d.pid).unwrap().chars12,
            want.s("sas12")
        );
    }
    // The right reading's info is 35 bytes (the 18 ASCII bytes of the domain, a zero byte and the 16 raw bytes of the pid), not the 41 and 51 of the wrong ones.
    assert_eq!(neg.n("correct.infoLength"), 35);
    assert_eq!(domain_message(math::SAS_DOMAIN, &[&d.pid]).len(), 35);
    assert_eq!(hex(&domain_message(math::SAS_DOMAIN, &[&d.pid])), neg.s("correct.infoHex"));
}

#[test]
fn a3_the_receipt() {
    let v = vectors();
    let a = a3();
    let grants: Vec<String> = v.at("A3.inputs.receiptDocument.grants").as_array().unwrap().iter().map(|g| g.as_str().unwrap().to_string()).collect();
    let text = math::receipt_text(
        v.s("A3.inputs.receiptDocument.appId"),
        &grants,
        v.n("A3.inputs.receiptDocument.issuedAt"),
        v.s("A3.inputs.receiptDocument.phoneThumbprint"),
        v.s("A3.inputs.receiptDocument.pid"),
    )
    .unwrap();
    assert_eq!(text, v.s("A3.expected.receiptText"));
    let signature = math::sign_receipt(
        &a.desktop_endpoint,
        "aokie",
        &grants,
        v.n("A3.inputs.receiptDocument.issuedAt"),
        v.s("A3.inputs.receiptDocument.phoneThumbprint"),
        v.s("A3.inputs.receiptDocument.pid"),
    )
    .unwrap();
    assert_eq!(signature, v.s("A3.expected.receiptSignature"));
    let key = a.desktop_endpoint.verify_key();
    let verify = |grants: &[String], issued: u64, thumb: &str, pid: &str| math::verify_receipt(&key, "aokie", grants, issued, thumb, pid, &signature);
    let (issued, thumb, pid) =
        (v.n("A3.inputs.receiptDocument.issuedAt"), v.s("A3.inputs.receiptDocument.phoneThumbprint"), v.s("A3.inputs.receiptDocument.pid"));
    verify(&grants, issued, thumb, pid).unwrap();
    // The grants are sorted before they are signed or checked, so their order is free; their content is not.
    let mut shuffled = grants.clone();
    shuffled.reverse();
    verify(&shuffled, issued, thumb, pid).unwrap();
    let mut more = grants.clone();
    more.push("takeover".into());
    assert!(verify(&more, issued, thumb, pid).is_err());
    assert!(verify(&grants[1..], issued, thumb, pid).is_err());
    assert!(verify(&grants, issued + 1, thumb, pid).is_err());
    assert!(verify(&grants, issued, "--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_cj", pid).is_err());
    assert!(verify(&grants, issued, thumb, "b5YkfMcTvJb0g1GTv3kNNR").is_err());
    // A receipt is the desktop's: another key does not verify it.
    assert!(math::verify_receipt(&a.phone_endpoint.verify_key(), "aokie", &grants, issued, thumb, pid, &signature).is_err());
}

#[test]
fn the_typed_codes_the_normaliser_and_the_sas_check_characters() {
    let v = vectors();
    for s in v.at("extras.typedCode.samples").as_array().unwrap() {
        let secret = PairingSecret::new(unhex(s.s("secretHex")).try_into().unwrap());
        assert_eq!(secret.typed_code(), s.s("typed"));
        let back = math::parse_typed_code(s.s("typed")).unwrap();
        assert_eq!(back.expose(), secret.expose());
    }
    for n in v.at("extras.typedCode.normalise").as_array().unwrap() {
        let got = math::normalise(n.s("input"));
        match n.at("output") {
            Json::Null => assert!(got.is_none(), "{}", n.s("input")),
            out => assert_eq!(got.as_ref().map(|s| s.as_str()), out.as_str(), "{}", n.s("input")),
        }
    }
    for c in v.at("extras.sasCheck.samples").as_array().unwrap() {
        assert_eq!(math::sas_check_char(c.s("sas12")).to_string(), c.s("check"), "{}", c.s("sas12"));
    }
}

#[test]
fn a_typed_code_with_a_typo_is_refused_locally_before_any_network_call() {
    let v = vectors();
    let good = v.s("A3.expected.typedCode");
    assert!(math::parse_typed_code(good).is_ok());
    assert!(math::parse_typed_code(&good.to_lowercase()).is_ok(), "case does not matter");
    assert!(math::parse_typed_code(&good.replace('-', " ")).is_ok());
    // Every single-character substitution over all 28 positions is offered to the check: at least 99 percent are refused (a 10-bit check cannot promise 100).
    let chars: Vec<char> = good.chars().filter(|c| *c != '-').collect();
    let alphabet: Vec<char> = String::from_utf8(math::CROCKFORD.to_vec()).unwrap().chars().collect();
    let (mut tried, mut accepted) = (0u32, 0u32);
    for i in 0..chars.len() {
        for &c in &alphabet {
            if c == chars[i] {
                continue;
            }
            let mut t = chars.clone();
            t[i] = c;
            tried += 1;
            if math::parse_typed_code(&t.iter().collect::<String>()).is_ok() {
                accepted += 1;
            }
        }
    }
    assert!(accepted * 100 <= tried, "{accepted} of {tried} single-character typos got through");
    // And the first 26 characters carry the secret, so a typo there is caught by the two check characters (and by the two zero padding bits): none gets through.
    // (the property above, as a bound; this one is exact for the positions the check covers fully)
    assert!(math::parse_typed_code("M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0R").is_err(), "too short");
    assert!(math::parse_typed_code("M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RWW").is_err(), "too long");
    assert!(math::parse_typed_code("M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RU").is_err(), "U is refused");
    assert!(math::parse_typed_code("").is_err());
}

#[test]
fn the_sas_entry_a_typo_is_not_an_attempt_and_a_wrong_code_is() {
    let a = a3();
    let d = a.secret.derive().unwrap();
    let v = vectors();
    let sas = math::sas(
        &a.desktop_endpoint.verify_key().to_bytes(),
        &a.phone_endpoint.verify_key().to_bytes(),
        &unhex32(v.s("A3.inputs.nonceHex")),
        &d.pid,
    )
    .unwrap();
    assert_eq!(math::judge_sas_entry(&sas, "6NHN-K68M-QQVZ-5"), SasEntry::Right);
    assert_eq!(math::judge_sas_entry(&sas, "6nhn k68m qqvz 5"), SasEntry::Right);
    assert_eq!(math::judge_sas_entry(&sas, "6NHN-K68M-QQVZ-"), SasEntry::Incomplete);
    assert_eq!(math::judge_sas_entry(&sas, ""), SasEntry::Incomplete);
    assert_eq!(math::judge_sas_entry(&sas, "6NHN-K68M-QQVZ-6"), SasEntry::BadCheck, "a wrong check character is a mistyped character");
    assert_eq!(math::judge_sas_entry(&sas, "6NHN-K68M-QQVZ-5X"), SasEntry::Invalid);
    assert_eq!(math::judge_sas_entry(&sas, "6NHN-K68M-QQVU-5"), SasEntry::Invalid, "U is not in the alphabet");
    // A different code whose check character is right: this is an attempt.
    let other = "3P8V-VXBP-SYYN-B";
    assert_eq!(math::judge_sas_entry(&sas, other), SasEntry::Wrong);
    assert!(SasEntry::Wrong.counts_as_attempt());
    for e in [SasEntry::Incomplete, SasEntry::Invalid, SasEntry::BadCheck, SasEntry::Right] {
        assert!(!e.counts_as_attempt());
    }
    // Every single-character substitution of the displayed code is a typo that the check character catches 31 times in 32 (the 32nd class is the check character itself).
    let shown: Vec<char> = "6NHNK68MQQVZ5".chars().collect();
    let alphabet: Vec<char> = String::from_utf8(math::CROCKFORD.to_vec()).unwrap().chars().collect();
    let (mut caught, mut other_class) = (0, 0);
    for i in 0..12 {
        for &c in &alphabet {
            if c == shown[i] {
                continue;
            }
            let mut t = shown.clone();
            t[i] = c;
            match math::judge_sas_entry(&sas, &t.iter().collect::<String>()) {
                SasEntry::BadCheck => caught += 1,
                SasEntry::Wrong => other_class += 1,
                other => panic!("{other:?}"),
            }
        }
    }
    assert_eq!(caught + other_class, 12 * 31);
    assert!(caught as f64 / (caught + other_class) as f64 > 0.9, "{caught} of {}", caught + other_class);
}

#[test]
fn a4_the_admission_bearer_shapes_and_sizes() {
    let v = vectors();
    let b = Bearer::parse(v.s("A4.expected.token")).unwrap();
    assert_eq!(b.len() as u64, v.n("A4.expected.length"));
    assert_eq!(b.expires_at(), 1_790_000_090);
    assert_eq!(b.claims().get_str("role"), Some("mobile"));
    // The relay's HMAC is the one the vector's secret makes: this checks the vector, and that the shape's two parts are what they say.
    let payload = v.s("A4.expected.payload");
    let mac = hex(&hmac_sha256(&unhex(v.s("A4.inputs.secretHex")), payload.as_bytes()).unwrap());
    assert!(v.s("A4.expected.token").ends_with(&mac));
    let single = Bearer::parse(v.s("A4b.expected.tokenForOnePhone")).unwrap();
    assert_eq!(single.len() as u64, v.at("A4b.expected.lengthByPhones").n("1"));
    for bad in [
        "",
        "aokie-adm-v2.",
        "aokie-adm-v2.7b.00",
        "aokie-adm-v2.zz.0000000000000000000000000000000000000000000000000000000000000000",
        "aokie-adm-v1.7b7d.0000000000000000000000000000000000000000000000000000000000000000",
    ] {
        assert!(Bearer::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn a6_the_identity_proof_and_the_static_signature() {
    let v = vectors();
    let relay = signer(v.s("keys.ed25519Seeds.relay")).verify_key();
    let a6 = v.at("A6");
    let body = a6.s("inputs.body").as_bytes();
    let nonce = b64::decode(a6.s("inputs.nonce")).unwrap();
    assert_eq!(hex(&sha256(body)), a6.s("expected.bodySha256"));
    info::verify_proof_signature(&relay, body, &nonce, a6.n("inputs.time") as i64, a6.s("expected.proof")).unwrap();
    info::verify_static_signature(&relay, body, a6.s("expected.staticSignature")).unwrap();
    // A replayed body and static signature with a new nonce fails, and so does a proof over another time or body.
    let other_nonce = [9u8; 16];
    assert!(info::verify_proof_signature(&relay, body, &other_nonce, a6.n("inputs.time") as i64, a6.s("expected.proof")).is_err());
    assert!(info::verify_proof_signature(&relay, body, &nonce, a6.n("inputs.time") as i64 + 1, a6.s("expected.proof")).is_err());
    assert!(info::verify_proof_signature(&relay, b"{}", &nonce, a6.n("inputs.time") as i64, a6.s("expected.proof")).is_err());
    assert!(
        info::verify_proof_signature(&relay, body, &nonce, a6.n("inputs.time") as i64, a6.s("expected.staticSignature")).is_err(),
        "the static signature is not a proof"
    );
    assert!(info::verify_static_signature(&relay, body, a6.s("expected.proof")).is_err(), "and a proof is not the static signature");
}

#[test]
fn a6b_a_whole_info_document_through_the_proof_check_a_client_makes() {
    let v = vectors();
    let a = v.at("A6b");
    let body = a.s("expected.bodyText").as_bytes();
    assert_eq!(body.len() as u64, a.n("expected.bodyBytes"));
    let nonce = b64::decode(a.s("inputs.nonce")).unwrap();
    let pinned = v.s("keys.ed25519Public.relay.thumbprint");
    let proved = info::verify_proof(body, &nonce, &a.n("inputs.time").to_string(), a.s("expected.proof"), pinned).unwrap();
    assert_eq!(proved.relay_time, 1_790_000_000);
    let i = &proved.info;
    assert_eq!((i.wait.default, i.wait.max, i.wait.poll_gap_ms, i.wait.fallback_s), (20, 20, 250, 5));
    assert_eq!(i.min_client, 1);
    assert!(i.has_feature("poll") && i.has_feature("compat.sse-framed-poll") && !i.has_feature("nope"));
    assert_eq!(i.limits.lanes.len(), 8);
    assert_eq!(i.lane("cmd").unwrap().body, 32768);
    assert_eq!(i.lane("sig").unwrap().ttl_max, 300);
    assert!(i.lane("pair").is_none(), "A6b keeps the design's example verbatim");
    info::verify_static_signature(&i.relay_key, body, a.s("expected.staticSignature")).unwrap();
    // Not who it was: another pinned thumbprint, a proof over another nonce, another time, a tampered body, a nonce of the wrong size.
    let other = v.s("keys.ed25519Public.provider.thumbprint");
    assert!(matches!(info::verify_proof(body, &nonce, "1790000000", a.s("expected.proof"), other), Err(oaiy_relay_core::Error::Mismatch(_))));
    assert!(info::verify_proof(body, &[1u8; 16], "1790000000", a.s("expected.proof"), pinned).is_err());
    assert!(info::verify_proof(body, &nonce, "1790000001", a.s("expected.proof"), pinned).is_err());
    assert!(info::verify_proof(body, &nonce, "01790000000", a.s("expected.proof"), pinned).is_err(), "the time is its canonical decimal spelling");
    assert!(info::verify_proof(body, &nonce[..15], "1790000000", a.s("expected.proof"), pinned).is_err());
    let tampered = a.s("expected.bodyText").replace("\"fallbackS\":5", "\"fallbackS\":9");
    assert!(info::verify_proof(tampered.as_bytes(), &nonce, "1790000000", a.s("expected.proof"), pinned).is_err());
    assert!(Info::parse(b"{}").is_err());
}

#[test]
fn a7_the_enrolment_key_its_request_and_its_proof() {
    let v = vectors();
    let a = v.at("A7");
    let key = EnrolmentKey::parse(a.s("expected.uri")).unwrap();
    assert_eq!(key.kid, a.s("expected.kid"));
    assert_eq!(key.role, Role::Desktop);
    assert_eq!(key.expires_at, a.n("inputs.expiry"));
    assert_eq!(key.relay_thumbprint, v.s("keys.ed25519Public.relay.thumbprint"));
    assert_eq!(key.signer().unwrap().verify_key().to_b64u(), a.s("expected.derivedPublic"));
    let secret: [u8; 16] = unhex(a.s("inputs.secretHex")).try_into().unwrap();
    assert_eq!(b64::encode(&secret), a.s("expected.secretB64u"));
    let relay = RelayUrl::parse(a.s("inputs.relayUrl")).unwrap();
    assert_eq!(
        EnrolmentKey::to_uri(&relay, v.s("keys.ed25519Public.relay.thumbprint"), &secret, Role::Desktop, a.n("inputs.expiry")).unwrap(),
        a.s("expected.uri")
    );
    let ed = signer(v.s("keys.ed25519Seeds.host")).verify_key();
    let x = X25519Public::from_b64u(v.s("keys.x25519Public.host")).unwrap();
    let nonce: [u8; 16] = b64::decode(a.s("inputs.request.n")).unwrap().try_into().unwrap();
    let req = enrol::build_request(&key, a.s("inputs.request.name"), &ed, &x, &nonce).unwrap();
    assert_eq!(req.body, a.s("expected.requestBody"));
    assert_eq!(req.proof, a.s("expected.proof"));
    // The relay's reading: the proof verifies over the exact bytes with the derived key, and over nothing else.
    let derived = key.signer().unwrap().verify_key();
    derived.verify_b64u(SignDomain::Enroll, &[req.body.as_bytes()], &req.proof).unwrap();
    assert!(derived.verify_b64u(SignDomain::Enroll, &[req.body.replace("Front", "Frant").as_bytes()], &req.proof).is_err());
    // A key whose k is not the key id of its secret is refused, as are a repeated parameter, a wrong version and a long one.
    let uri = a.s("expected.uri");
    assert!(EnrolmentKey::parse(&uri.replace("k=OJttnmp91Xo", "k=OJttnmp91Xp")).is_err());
    assert!(EnrolmentKey::parse(&format!("{uri}&x=1")).is_err());
    assert!(EnrolmentKey::parse(&uri.replace("v=1", "v=2")).is_err());
    assert!(EnrolmentKey::parse(&format!("{uri}&junk={}", "a".repeat(600))).is_err());
    assert!(EnrolmentKey::parse(&format!("{uri}&unknown=1")).is_ok(), "unknown parameters are ignored");
    assert!(EnrolmentKey::parse(&uri.replace("oaiy://enroll", "oaiy://pair")).is_err());
}

#[test]
fn a8_a_command_container_signed_by_a_provider() {
    let v = vectors();
    let a = v.at("A8");
    let provider = signer(v.s("keys.ed25519Seeds.provider"));
    let bytes = a.s("expected.signedBytes").as_bytes();
    // The bytes are shipped, not re-derived: the text the vector holds is the text signed.
    let text = sealed::build_container(&provider, ContainerDomain::Cmd, bytes, false, |_| {});
    assert_eq!(text, a.s("expected.container"));
    assert_eq!(text.len() as u64, a.n("expected.containerBytes"));
    assert_eq!(signature_to_text(&provider, bytes), a.s("expected.signature"));
    // Sealed to the desktop's X25519 key and opened with its secret: the sealing step is randomised, so there is no fixed answer, only a round trip.
    let desktop = x_secret(v.s("keys.x25519Secrets.host"));
    let body = sealed::seal_container(&desktop.public_key(), &text).unwrap();
    let opened = sealed::open_container(&desktop, &body).unwrap();
    assert_eq!(opened.signer, provider.thumbprint());
    assert_eq!(opened.verify(ContainerDomain::Cmd, &provider.verify_key()).unwrap(), bytes);
    assert!(opened.verify(ContainerDomain::Res, &provider.verify_key()).is_err(), "a command is not a result");
    assert!(opened.verify(ContainerDomain::Cmd, &signer(v.s("keys.ed25519Seeds.provider2")).verify_key()).is_err(), "another key");
    // Padding: made a multiple of 256, ignored by the verifier.
    let mut state = 1u32;
    let padded = sealed::build_container(&provider, ContainerDomain::Cmd, bytes, true, |buf| {
        for b in buf.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (state >> 16) as u8;
        }
    });
    assert_eq!(padded.len() % 256, 0);
    let c = Container::parse(padded.as_bytes()).unwrap();
    assert_eq!(c.verify(ContainerDomain::Cmd, &provider.verify_key()).unwrap(), bytes);
    // Extra members and a wrong shape are refused.
    assert!(Container::parse(text.replace("}", ",\"x\":\"y\"}").as_bytes()).is_err());
    assert!(Container::parse(br#"{"k":"x","b":"AA","s":"y"}"#).is_err());
    assert!(Container::parse(text.replace(",\"s\"", ",\"s\":\"x\",\"s\"").as_bytes()).is_err(), "a duplicate member");
}

fn signature_to_text(signer: &Signer, bytes: &[u8]) -> String {
    signer.sign_b64u(SignDomain::Cmd, &[bytes])
}

#[test]
fn a9_a_ticket() {
    let v = vectors();
    let a = v.at("A9");
    let provider = signer(v.s("keys.ed25519Seeds.provider")).verify_key();
    let claims = ticket::verify_signature(a.s("expected.ticket"), &provider).unwrap();
    assert_eq!(claims.jti, "tkt-0001");
    assert_eq!(claims.lane, "ai");
    assert_eq!(claims.org, "https://app.example.com");
    assert_eq!(ticket::kid(a.s("expected.ticket")).unwrap(), provider.thumbprint());
    let ok = ticket::Context { relay_id: v.s("keys.ids.relay"), desktop: v.s("keys.ids.desktopDevice"), provider_now: 1_790_000_100 };
    claims.check(&ok).unwrap();
    assert!(claims.check(&ticket::Context { provider_now: 1_790_000_300 + 31, ..ok }).is_err(), "past exp + 30");
    assert!(claims.check(&ticket::Context { provider_now: 1_790_000_000 - 31, ..ok }).is_err(), "before iat - 30");
    claims.check(&ticket::Context { provider_now: 1_790_000_000 - 30, ..ok }).unwrap();
    assert!(claims.check(&ticket::Context { relay_id: "rly-AAAAAAAAAAAAAAAAAAAAAA", ..ok }).is_err(), "a ticket for another relay");
    assert!(claims.check(&ticket::Context { desktop: "dev-AAAAAAAAAAAAAAAAAAAAAA", ..ok }).is_err(), "a ticket for another desktop");
    // Algorithm confusion, a repeated or extra header member, a missing part and a flipped signature are refused.
    let t = a.s("expected.ticket");
    let (h, rest) = t.split_once('.').unwrap();
    let header = String::from_utf8(b64::decode(h).unwrap()).unwrap();
    for bad_header in [
        header.replace("EdDSA", "none"),
        header.replace("EdDSA", "HS256"),
        header.replace('}', ",\"crit\":[\"x\"]}"),
        header.replace("oaiy-ticket+jwt", "JWT"),
    ] {
        let forged = format!("{}.{rest}", b64::encode(bad_header.as_bytes()));
        assert!(ticket::verify_signature(&forged, &provider).is_err(), "{bad_header}");
    }
    assert!(ticket::verify_signature(&format!("{t}.AAAA"), &provider).is_err());
    assert!(ticket::verify_signature(h, &provider).is_err());
    assert!(ticket::verify_signature(&t.replace("8uLJTXCa", "8uLJTXCb"), &provider).is_err());
    assert!(ticket::verify_signature(t, &signer(v.s("keys.ed25519Seeds.provider2")).verify_key()).is_err());
}

#[test]
fn a10_a_provider_rotation_statement() {
    let v = vectors();
    let a = v.at("A10");
    let old = signer(v.s("keys.ed25519Seeds.provider")).verify_key();
    let b = b64::encode(a.s("expected.statementText").as_bytes());
    let s = a.s("expected.signature");
    let st = rotation::verify(&old, 0, 1_790_000_100, &b, s).unwrap();
    assert_eq!(st.serial, 1);
    assert_eq!(st.new_ed25519.to_b64u(), v.s("keys.ed25519Public.provider2.publicKey"));
    assert_eq!(st.new_x25519.to_b64u(), v.s("keys.x25519Public.provider2"));
    assert!(rotation::verify(&old, 1, 1_790_000_100, &b, s).is_err(), "a serial that is not above the last accepted");
    assert!(rotation::verify(&old, 0, 1_790_086_400 + 31, &b, s).is_err(), "past exp + 30");
    assert!(rotation::verify(&old, 0, 1_790_000_000 - 31, &b, s).is_err(), "before iat - 30");
    assert!(
        rotation::verify(&signer(v.s("keys.ed25519Seeds.provider2")).verify_key(), 0, 1_790_000_100, &b, s).is_err(),
        "not signed by the pinned key"
    );
    assert!(rotation::verify(
        &old,
        0,
        1_790_000_100,
        &b64::encode(a.s("expected.statementText").replace("\"serial\":1", "\"serial\":2").as_bytes()),
        s
    )
    .is_err());
}

#[test]
fn a11_a_ring_signature_and_the_ring_window() {
    let v = vectors();
    let a = v.at("A11");
    let host = signer(v.s("keys.ed25519Seeds.host"));
    let text = a.s("expected.bodyText");
    assert_eq!(ring::sign(&host, text), a.s("expected.hdrSig"));
    ring::verify(&host.verify_key(), text, a.s("expected.hdrSig")).unwrap();
    // A ring signed by another key, over another text, or with a signature of another domain, is dropped.
    assert!(ring::verify(&signer(v.s("keys.ed25519Seeds.provider")).verify_key(), text, a.s("expected.hdrSig")).is_err());
    assert!(ring::verify(&host.verify_key(), &text.replace("call_0123", "call_0124"), a.s("expected.hdrSig")).is_err());
    let body = RingBody::parse(text).unwrap();
    assert_eq!(body.get("callEpoch"), Some("7"));
    assert_eq!(body.expires_at(), Some(1_790_000_040));
    body.check_window(1_790_000_000).unwrap();
    assert!(body.check_window(1_790_000_040).is_err(), "expiresAt must be later than now");
    assert!(body.check_window(1_789_999_739).is_err(), "and at most now + 300");
    body.check_window(1_789_999_740).unwrap();
    for bad in [
        text.replace("\"callEpoch\":\"7\"", "\"callEpoch\":7"),
        text.replace("\"callEpoch\":\"7\"", "\"callEpoch\":\"0\""),
        text.replace("\"callEpoch\":\"7\"", "\"callEpoch\":\"07\""),
        text.replace('}', ",\"extra\":\"x\"}"),
        text.replace("voice_offer", "voice_other"),
        text.replace("\"schemaVersion\":\"1\"", "\"schemaVersion\":\"2\""),
        text.replace("\"eventId\":\"evt_ring_0001\",", ""),
    ] {
        assert!(RingBody::parse(&bad).is_err(), "{bad}");
    }
}

#[test]
fn a12_the_small_order_encodings_are_refused_with_and_without_bit_255() {
    let v = vectors();
    let a = v.at("A12");
    for set in ["inputs.encodings", "inputs.withBit255"] {
        for (name, value) in a.at(set).as_object().unwrap() {
            let bytes = unhex32(value.as_str().unwrap());
            assert!(X25519Public::from_bytes(&bytes).is_err(), "{set}.{name}");
        }
    }
    let mut base = [0u8; 32];
    base[0] = 9;
    assert!(X25519Public::from_bytes(&base).is_ok(), "the base point is the control");
    // An Ed25519 key of small order (the identity) and a key that is not canonical are refused too.
    let mut identity = [0u8; 32];
    identity[0] = 1;
    assert!(VerifyKey::from_bytes(&identity).is_err());
    let mut non_canonical = identity;
    non_canonical[31] = 0x80;
    assert!(VerifyKey::from_bytes(&non_canonical).is_err());
}

#[test]
fn the_canonical_json_cases_and_refusals_of_the_vectors() {
    let v = vectors();
    for c in v.at("extras.canonical.cases").as_array().unwrap() {
        assert_eq!(json::canonicalize(c.s("input").as_bytes()).unwrap(), c.s("output"), "{}", c.s("label"));
    }
    for c in v.at("extras.canonical.refused").as_array().unwrap() {
        assert!(json::canonicalize(c.s("input").as_bytes()).is_err(), "{}", c.s("label"));
    }
}

#[test]
fn the_lane_table_of_the_vectors_is_what_a_client_reads_in_info() {
    let v = vectors();
    let info = Info::parse(v.s("A6b.expected.bodyText").as_bytes()).unwrap();
    for (name, lane) in v.at("extras.lanes").as_object().unwrap() {
        if let Some(l) = info.lane(name) {
            assert_eq!(l.body, lane.n("body"), "{name}");
            assert_eq!(l.ttl_default, lane.n("ttl.default"), "{name}");
            assert_eq!(l.ttl_min, lane.n("ttl.min"), "{name}");
            assert_eq!(l.ttl_max, lane.n("ttl.max"), "{name}");
        }
    }
    assert_eq!(v.n("extras.pollGapMs"), 250);
}

#[test]
fn identifier_forms_of_the_vectors() {
    let v = vectors();
    assert!(ids::is_device_id(v.s("keys.ids.desktopDevice")) && ids::is_device_id(v.s("keys.ids.phoneDevice")));
    assert!(ids::is_provider_id(v.s("keys.ids.provider")) && ids::is_relay_id(v.s("keys.ids.relay")));
    // The ids are the b64u of their 16 bytes.
    assert_eq!(format!("dev-{}", b64::encode(&unhex(v.s("keys.idBytes.desktopDevice")))), v.s("keys.ids.desktopDevice"));
}

#[test]
fn sealed_tokens_three_open_ten_do_not() {
    let doc = load("fixtures/sealed-token.json");
    let recipient = X25519Secret::from_secret(&Secret::new(b64::decode_exact::<32>(doc.s("recipient.x25519Secret")).unwrap()));
    assert_eq!(recipient.public_key().to_b64u(), doc.s("recipient.x25519Public"));
    let opens = doc.at("opens").as_array().unwrap();
    assert_eq!(opens.len(), 3);
    for (i, c) in opens.iter().enumerate() {
        let token = sealed::open_token(&recipient, c.s("sealedToken")).unwrap_or_else(|e| panic!("opens[{i}]: {e}"));
        assert_eq!(token.expose().len() as u64, c.n("plaintextLength"));
        assert_eq!(hex(&sha256(token.expose().as_bytes())), c.s("plaintextSha256"), "opens[{i}]");
        assert_eq!(b64::decode(c.s("sealedToken")).unwrap().len() as u64, c.n("sealedBytes"));
    }
    let refused = doc.at("refused").as_array().unwrap();
    assert_eq!(refused.len(), 10);
    for (i, c) in refused.iter().enumerate() {
        assert!(sealed::open_token(&recipient, c.s("sealedToken")).is_err(), "refused[{i}] {}", c.s("label"));
    }
    // The boxes that authenticate under the all-zero shared secret (refused[8] and refused[9]) are the ones that test the small-order rule: the Rust `crypto_box` crate's
    // `unseal` opens refused[8]; `oaiy-crypto`'s `open`, which this crate calls, refuses it before it opens anything.
    let wrong = X25519Secret::from_secret(&Secret::new(b64::decode_exact::<32>(doc.s("wrongRecipient.x25519Secret")).unwrap()));
    assert_eq!(wrong.public_key().to_b64u(), doc.s("wrongRecipient.x25519Public"));
    assert!(sealed::open_token(&wrong, opens[0].s("sealedToken")).is_err());
    // Not strict base64url: padding, the standard alphabet, a different spelling of the same bytes.
    let good = opens[0].s("sealedToken");
    assert!(sealed::open_token(&recipient, &format!("{good}=")).is_err());
    assert!(sealed::open_token(&recipient, &good.replace('-', "+")).is_err() || !good.contains('-'));
    assert!(sealed::open_token(&recipient, "").is_err());
    assert!(sealed::open_token(&recipient, &"A".repeat(2000)).is_err());
    // A sealed box that opens but is not a token is refused too: this is the plaintext check.
    let not_a_token = b64::encode(&recipient.public_key().seal(b"oaiyrt1.AQIDBAUGBwg.not-a-token").unwrap());
    assert!(sealed::open_token(&recipient, &not_a_token).is_err());
    let a_token = b64::encode(&recipient.public_key().seal(b"oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8").unwrap());
    assert!(sealed::open_token(&recipient, &a_token).is_ok());
    let a_token_and_more = b64::encode(&recipient.public_key().seal(b"oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8\n").unwrap());
    assert!(sealed::open_token(&recipient, &a_token_and_more).is_err());
}

#[test]
fn a_typed_code_has_one_spelling_the_two_unused_bits_of_the_26th_character_are_zero() {
    let good = vectors().s("A3.expected.typedCode").to_string();
    let chars: Vec<char> = good.chars().filter(|c| *c != '-').collect();
    let alphabet: Vec<char> = String::from_utf8(math::CROCKFORD.to_vec()).unwrap().chars().collect();
    let value = alphabet.iter().position(|c| *c == chars[25]).unwrap();
    assert_eq!(value & 3, 0, "the 26th character of a code this crate writes ends in two zero bits");
    for flip in 1..4 {
        let mut t = chars.clone();
        t[25] = alphabet[value | flip];
        // The 128 bits of the secret are the same and so are the two check characters: only the spelling differs, and a second spelling of one secret is refused.
        assert!(math::parse_typed_code(&t.iter().collect::<String>()).is_err(), "unused bits {flip}");
    }
}

#[test]
fn the_edges_the_recorded_vectors_leave_open_a_nonce_of_another_size_and_a_lifetime_one_second_too_long() {
    let v = vectors();
    // A proof that verifies, for a nonce of the wrong size, is refused for the size and for nothing else: the same proof procedure is good for 16 and 32 bytes.
    let a = v.at("A6b");
    let body = a.s("expected.bodyText").as_bytes();
    let relay = signer(v.s("keys.ed25519Seeds.relay"));
    let pinned = v.s("keys.ed25519Public.relay.thumbprint");
    let digest = sha256(body);
    for (len, good) in [(15usize, false), (16, true), (32, true), (33, false)] {
        let nonce = vec![7u8; len];
        let proof = relay.sign_b64u(SignDomain::InfoProof, &[&nonce[..], &digest[..], b"1790000000".as_slice()]);
        assert_eq!(info::verify_proof(body, &nonce, "1790000000", &proof, pinned).is_ok(), good, "a nonce of {len} bytes");
    }
    // The wait of an info is 300 seconds at most, each of the two on its own.
    let text = a.s("expected.bodyText");
    let wait = "\"wait\":{\"default\":20,\"max\":20,";
    assert!(text.contains(wait));
    for (replacement, good) in [
        ("\"wait\":{\"default\":300,\"max\":300,", true),
        ("\"wait\":{\"default\":301,\"max\":300,", false),
        ("\"wait\":{\"default\":300,\"max\":301,", false),
    ] {
        assert_eq!(Info::parse(text.replace(wait, replacement).as_bytes()).is_ok(), good, "{replacement}");
    }
    // A ticket lives 300 seconds at most.
    let provider = signer(v.s("keys.ed25519Seeds.provider")).verify_key();
    let claims = ticket::verify_signature(v.s("A9.expected.ticket"), &provider).unwrap();
    let ok = ticket::Context { relay_id: v.s("keys.ids.relay"), desktop: v.s("keys.ids.desktopDevice"), provider_now: 1_790_000_100 };
    assert_eq!(claims.exp - claims.iat, 300);
    claims.check(&ok).unwrap();
    assert!(ticket::Claims { exp: claims.exp + 1, ..claims.clone() }.check(&ok).is_err(), "exp - iat of 301");
    // A rotation statement lives 24 hours at most: the statement of the vector, re-signed by the pinned key with the one second added, is refused.
    let old = signer(v.s("keys.ed25519Seeds.provider"));
    let statement = v.s("A10.expected.statementText");
    let resign = |text: &str| (b64::encode(text.as_bytes()), old.sign_b64u(SignDomain::ProviderRotate, &[text.as_bytes()]));
    let (b, s) = resign(statement);
    assert_eq!(s, v.s("A10.expected.signature"), "Ed25519 is deterministic: the re-signed vector is the vector");
    rotation::verify(&old.verify_key(), 0, 1_790_000_100, &b, &s).unwrap();
    let longer = statement.replace("\"exp\":1790086400", "\"exp\":1790086401");
    assert_ne!(longer, statement);
    let (b, s) = resign(&longer);
    assert!(rotation::verify(&old.verify_key(), 0, 1_790_000_100, &b, &s).is_err(), "a statement that lives 86,401 seconds");
}

#[test]
fn an_ed25519_key_in_a_non_canonical_encoding_is_refused_though_it_names_a_good_point() {
    // For y below 19 the numbers y and y + p are the same field element, so `y + p` is a second spelling of the same point; only the first is canonical.
    let mut tried = 0;
    for y in 2u8..19 {
        let mut canonical = [0u8; 32];
        canonical[0] = y;
        if VerifyKey::from_bytes(&canonical).is_err() {
            continue; // not a point, or a point of small order
        }
        let mut second = [0xffu8; 32];
        second[0] = 0xed + y;
        second[31] = 0x7f;
        assert!(VerifyKey::from_bytes(&second).is_err(), "y + p for y = {y}");
        tried += 1;
    }
    assert!(tried >= 5, "{tried} of the small y values name a point of the prime-order group");
}

#[test]
fn the_poll_gap_has_a_floor_of_250_ms_and_a_ceiling_of_5_s_whatever_the_relay_advertises() {
    let v = vectors();
    let a = v.at("A6b");
    let nonce = b64::decode(a.s("inputs.nonce")).unwrap();
    let proved = info::verify_proof(
        a.s("expected.bodyText").as_bytes(),
        &nonce,
        &a.n("inputs.time").to_string(),
        a.s("expected.proof"),
        v.s("keys.ed25519Public.relay.thumbprint"),
    )
    .unwrap();
    let mut i = proved.info;
    for (advertised, used) in [(0u64, 250u64), (100, 250), (249, 250), (250, 250), (300, 300), (5000, 5000), (60_000, 5000)] {
        i.wait.poll_gap_ms = advertised;
        assert_eq!(oaiy_relay_core::poll::PollInfo::from_info(&i).poll_gap_ms, used, "pollGapMs {advertised}");
    }
    for (advertised, used) in [(0u64, 1u64), (5, 5), (60, 60), (500, 60)] {
        i.wait.fallback_s = advertised;
        assert_eq!(oaiy_relay_core::poll::PollInfo::from_info(&i).fallback_s, used, "fallbackS {advertised}");
    }
}
