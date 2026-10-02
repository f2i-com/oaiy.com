//! Review tests (F3, the reviewer's own): attacks on the phone party of pairing v3 and on the documents it reads.
//!
//! `attack_*` tests assert the secure behaviour and pass today. `finding_*` tests assert what the README or the design asks for and FAIL today; they are `#[ignore]`d so that the
//! suite stays green: `cargo test -p oaiy-relay-core --test rv_pairing_phone -- --ignored` shows each failure.

mod common;

use std::sync::Mutex;
use std::time::Duration;

use common::env::quick;
use common::pair::{client_for_target, grants, world, World};
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::b64;
use oaiy_relay_core::client::*;
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::json::Json;
use oaiy_relay_core::keys::{thumbprint_of, SignDomain, Signer, X25519Public, X25519Secret};
use oaiy_relay_core::pairing::math::{self, Derived};
use oaiy_relay_core::pairing::phone::{store_paired, Outcome, Paired, PhoneIdentity};
use oaiy_relay_core::pairing::{NewOffer, Offer, PairEvent, PairingError, PairingInput, PairingKey, PairingTarget, PhonePairing, Response};
use oaiy_relay_core::testing::stub::{Fault, StubConfig};

fn cancel() -> Cancel {
    Cancel::new()
}

const TOKEN: &str = "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8";

struct Phone {
    pairing: PhonePairing,
    thumb: String,
    x25519: X25519Public,
}

/// A phone with fixed keys (so that two pairings can be made by one key), made for `offer` over the world's stub.
fn phone_with_keys(w: &World, offer: &NewOffer, seed: u8) -> Phone {
    let endpoint = Signer::from_seed(&Secret::new([seed; 32]));
    let xs = X25519Secret::from_secret(&Secret::new([seed.wrapping_add(1); 32]));
    let (thumb, x25519) = (endpoint.thumbprint(), xs.public_key());
    let identity = PhoneIdentity { endpoint, x25519: xs, device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(), display_name: Some("Test phone".into()) };
    let target = PairingTarget::from_input(PairingInput::Key(&offer.pairing_uri)).unwrap();
    let client = client_for_target(&w.env, &target, u64::from(seed));
    Phone { pairing: PhonePairing::new(client, target, identity).unwrap(), thumb, x25519 }
}

fn derived(offer: &NewOffer) -> Derived {
    PairingKey::parse(&offer.pairing_uri).unwrap().secret.derive().unwrap()
}

fn grants_body(g: Option<&[&str]>) -> Option<Json> {
    g.map(|g| Json::Arr(g.iter().map(|x| Json::str(*x)).collect()))
}

/// The answer of a relay that says `approved`.
fn approved_body(sealed: &str, issued: u64, sig: &str, grants: Option<&[&str]>) -> String {
    let mut receipt = vec![("issuedAt", Json::int(issued)), ("signature", Json::str(sig))];
    if let Some(g) = grants_body(grants) {
        receipt.push(("grants", g));
    }
    Json::obj([
        ("v", Json::int(1)),
        ("state", Json::str("approved")),
        ("deviceId", Json::str("dev-AAAAAAAAAAAAAAAAAAAAAA")),
        ("sealedToken", Json::str(sealed)),
        ("receipt", Json::Obj(receipt.into_iter().map(|(k, v)| (k.to_string(), v)).collect())),
        ("time", Json::int(1)),
    ])
    .to_compact()
}

fn answered_body(offer: &NewOffer, hold: bool) -> String {
    let mut m = vec![
        ("v", Json::int(1)),
        ("state", Json::str("answered")),
        ("offer", Json::str(offer.offer.text.clone())),
        ("mac", Json::str(offer.mac.clone())),
        ("exp", Json::int(offer.offer.expires_at)),
        ("time", Json::int(1)),
    ];
    if hold {
        m.push(("hold", Json::obj([("granted", Json::Bool(true))])));
    }
    Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()).to_compact()
}

// ------------------------------------------------------------------------------------------------------------------ the receipt

#[test]
fn attack_the_receipt_is_bound_to_app_grants_time_phone_pid_signer_and_domain() {
    let desktop = Signer::generate().unwrap();
    let (phone, other) = (Signer::generate().unwrap(), Signer::generate().unwrap());
    let (pid, pid2) = (b64::encode(&[7u8; 16]), b64::encode(&[8u8; 16]));
    let g: Vec<String> = ["state_read", "caller_read", "rtc_signal"].iter().map(|s| s.to_string()).collect();
    let sig = math::sign_receipt(&desktop, "aokie", &g, 1000, &phone.thumbprint(), &pid).unwrap();
    let verify = |app: &str, grants: &[String], at: u64, thumb: &str, pid: &str, sig: &str, key: &Signer| {
        math::verify_receipt(&key.verify_key(), app, grants, at, thumb, pid, sig)
    };
    assert!(verify("aokie", &g, 1000, &phone.thumbprint(), &pid, &sig, &desktop).is_ok(), "control");
    // Order of the grants is not part of the document (sorted), a different set is.
    let reordered: Vec<String> = g.iter().rev().cloned().collect();
    assert!(verify("aokie", &reordered, 1000, &phone.thumbprint(), &pid, &sig, &desktop).is_ok());
    let more: Vec<String> = g.iter().cloned().chain(["takeover".to_string()]).collect();
    let fewer: Vec<String> = g[..2].to_vec();
    let dup: Vec<String> = g.iter().cloned().chain(["state_read".to_string()]).collect();
    for (what, r) in [
        ("grants plus takeover", verify("aokie", &more, 1000, &phone.thumbprint(), &pid, &sig, &desktop)),
        ("fewer grants", verify("aokie", &fewer, 1000, &phone.thumbprint(), &pid, &sig, &desktop)),
        ("a repeated grant", verify("aokie", &dup, 1000, &phone.thumbprint(), &pid, &sig, &desktop)),
        ("other app", verify("other", &g, 1000, &phone.thumbprint(), &pid, &sig, &desktop)),
        ("one second later", verify("aokie", &g, 1001, &phone.thumbprint(), &pid, &sig, &desktop)),
        ("other phone", verify("aokie", &g, 1000, &other.thumbprint(), &pid, &sig, &desktop)),
        ("other pid", verify("aokie", &g, 1000, &phone.thumbprint(), &pid2, &sig, &desktop)),
        ("the pid as raw hex instead of text", verify("aokie", &g, 1000, &phone.thumbprint(), &"07".repeat(16), &sig, &desktop)),
        ("other signer", verify("aokie", &g, 1000, &phone.thumbprint(), &pid, &sig, &other)),
    ] {
        assert!(r.is_err(), "{what} verified");
    }
    // Domain separation: the same document signed under any other domain of the protocol, or by the phone's key, is not a receipt.
    let text = math::receipt_text("aokie", &g, 1000, &phone.thumbprint(), &pid).unwrap();
    for domain in [
        SignDomain::PairingResponse,
        SignDomain::Info,
        SignDomain::InfoProof,
        SignDomain::Enroll,
        SignDomain::Cmd,
        SignDomain::Res,
        SignDomain::Sync,
        SignDomain::ProviderRotate,
        SignDomain::Ring,
    ] {
        let s = desktop.sign_b64u(domain, &[text.as_bytes()]);
        assert!(verify("aokie", &g, 1000, &phone.thumbprint(), &pid, &s, &desktop).is_err(), "{domain:?}");
    }
    // And a receipt is not accepted where a response signature is (the claims text under the approval domain).
    let s = desktop.sign_b64u(SignDomain::PairingApproval, &[text.as_bytes()]);
    assert!(desktop.verify_key().verify_b64u(SignDomain::PairingResponse, &[text.as_bytes()], &s).is_err());
    assert!(desktop.verify_key().verify_b64u(SignDomain::PairingApproval, &[text.as_bytes()], &s).is_ok());
}

#[test]
fn attack_a_hostile_relay_cannot_get_a_profile_or_a_token_opened_with_a_receipt_that_is_for_something_else() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = phone_with_keys(&w, &offer, 71);
    p.pairing.fetch_offer(&cancel()).unwrap();
    p.pairing.respond(&cancel()).unwrap();
    let pid = p.pairing.pid().to_string();
    let now = w.env.client.relay_now_or_local() as u64;
    let desk = &w.identity.endpoint;
    let g: Vec<String> = vec!["state_read".into()];
    let sealed = b64::encode(&p.x25519.seal(TOKEN.as_bytes()).unwrap());
    assert!(Token::parse(TOKEN).is_ok());
    let stranger = Signer::generate().unwrap();
    let other_pid = b64::encode(&[9u8; 16]);
    let sign = |signer: &Signer, app: &str, grants: &[String], at: u64, thumb: &str, pid: &str| math::sign_receipt(signer, app, grants, at, thumb, pid).unwrap();
    let text = math::receipt_text("aokie", &g, now, &p.thumb, &pid).unwrap();
    // Each is a relay's answer `approved` that a phone must refuse with the box never opened (the box here IS for this phone: only the receipt is wrong).
    let scenarios: Vec<(&str, u64, String, Option<Vec<&str>>, PairingError)> = vec![
        ("signed by a key that is not the desktop's", now, sign(&stranger, "aokie", &g, now, &p.thumb, &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("another pairing's pid", now, sign(desk, "aokie", &g, now, &p.thumb, &other_pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("another phone's key", now, sign(desk, "aokie", &g, now, &stranger.thumbprint(), &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("another app", now, sign(desk, "otherapp", &g, now, &p.thumb, &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("grants the desktop did not sign", now, sign(desk, "aokie", &g, now, &p.thumb, &pid), Some(vec!["state_read", "takeover"]), PairingError::ReceiptInvalid),
        ("fewer grants than signed", now, sign(desk, "aokie", &["state_read".to_string(), "caller_read".to_string()], now, &p.thumb, &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("a grant that no desktop knows", now, sign(desk, "aokie", &["bogus".to_string()], now, &p.thumb, &pid), Some(vec!["bogus"]), PairingError::ReceiptInvalid),
        ("dated 31 s in the future", now + 31, sign(desk, "aokie", &g, now + 31, &p.thumb, &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        ("dated 31 s before the phone asked", now - 31, sign(desk, "aokie", &g, now - 31, &p.thumb, &pid), Some(vec!["state_read"]), PairingError::ReceiptInvalid),
        (
            "signed under the response domain",
            now,
            desk.sign_b64u(SignDomain::PairingResponse, &[text.as_bytes()]),
            Some(vec!["state_read"]),
            PairingError::ReceiptInvalid,
        ),
        ("the grants are not on the wire and not given", now, sign(desk, "aokie", &g, now, &p.thumb, &pid), None, PairingError::ReceiptGrantsUnknown),
    ];
    for (what, issued, sig, wire_grants, expected) in scenarios {
        w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved_body(&sealed, issued, &sig, wire_grants.as_deref())));
        let result = p.pairing.wait_outcome(None, &cancel());
        assert_eq!(result.err(), Some(expected), "{what}");
    }
    // Control, last (a paired phone is done): a receipt that is right in every member pairs.
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved_body(&sealed, now, &sign(desk, "aokie", &g, now, &p.thumb, &pid), Some(&["state_read"]))));
    assert!(matches!(p.pairing.wait_outcome(None, &cancel()), Ok(Outcome::Paired(_))), "control");
}

/// A phone that is given a receipt, right in every member, dated `delta` seconds from the moment it answered.
fn approved_with_receipt_at(delta: i64) -> Result<Outcome, PairingError> {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = phone_with_keys(&w, &offer, 81);
    p.pairing.fetch_offer(&cancel()).unwrap();
    p.pairing.respond(&cancel()).unwrap();
    let issued = (w.env.client.relay_now_or_local() + delta) as u64;
    let g = vec!["state_read".to_string()];
    let sig = math::sign_receipt(&w.identity.endpoint, "aokie", &g, issued, &p.thumb, p.pairing.pid()).unwrap();
    let sealed = b64::encode(&p.x25519.seal(TOKEN.as_bytes()).unwrap());
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved_body(&sealed, issued, &sig, Some(&["state_read"]))));
    p.pairing.wait_outcome(None, &cancel())
}

#[test]
fn attack_the_receipt_window_is_30_seconds_either_side_of_the_exchange() {
    assert!(approved_with_receipt_at(30).is_ok(), "30 s ahead of the relay's clock");
    assert!(approved_with_receipt_at(31).is_err(), "31 s ahead");
    assert!(approved_with_receipt_at(-30).is_ok(), "30 s before the phone asked");
    assert!(approved_with_receipt_at(-31).is_err(), "31 s before");
}

#[test]
fn attack_the_grants_a_caller_supplies_cannot_stand_in_for_what_the_desktop_signed_and_the_relays_win_when_given() {
    let mut w = world(quick()); // the shipped relay: no grants on the wire
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 72);
    p.fetch_offer(&cancel()).unwrap();
    let sas = p.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    let mut wrong = grants();
    wrong.reverse();
    wrong.pop();
    // Whoever supplies the grants out of band (the UI, an attacker with a say in it) cannot get a profile with grants other than the signed ones...
    assert_eq!(p.wait_outcome(Some(&wrong), &cancel()).err(), Some(PairingError::ReceiptInvalid));
    let mut with_extra = grants();
    with_extra.push("monitor".into());
    assert_eq!(p.wait_outcome(Some(&with_extra), &cancel()).err(), Some(PairingError::ReceiptInvalid));
    // ...and the right ones, in any order, pair, with the signed set in the profile.
    let mut shuffled = grants();
    shuffled.rotate_left(2);
    let Ok(Outcome::Paired(paired)) = p.wait_outcome(Some(&shuffled), &cancel()) else { panic!("not paired") };
    let mut got = paired.profile.grants.clone();
    got.sort();
    let mut want = grants();
    want.sort();
    assert_eq!(got, want);
}

#[test]
fn attack_grants_from_the_relay_win_over_grants_from_the_caller_and_are_only_as_good_as_the_signature() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 73);
    p.fetch_offer(&cancel()).unwrap();
    let sas = p.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    // A caller that passes other grants out of band is ignored when the relay returns them: the receipt (checked over the relay's list) is what decides.
    let Ok(Outcome::Paired(paired)) = p.wait_outcome(Some(&["takeover".to_string()]), &cancel()) else { panic!("not paired") };
    assert_eq!(paired.profile.grants, grants());
}

#[test]
fn attack_a_phone_could_find_the_signed_grants_by_search_if_the_relay_never_returns_them() {
    // Evidence for the options of the contract finding: with no change of the relay, the phone can learn the set the desktop signed by trying the subsets of the 14 known names
    // (2^14 = 16,384 verifications, worst case). The signature is valid for exactly one document, so what is found is what was approved, whatever the relay says.
    let desktop = Signer::generate().unwrap();
    let phone = Signer::generate().unwrap();
    let pid = b64::encode(&[7u8; 16]);
    let signed: Vec<String> =
        ["state_read", "caller_read", "captions_read", "assistance_read", "assistance_respond", "rtc_signal"].iter().map(|s| s.to_string()).collect();
    let sig = math::sign_receipt(&desktop, "aokie", &signed, 1000, &phone.thumbprint(), &pid).unwrap();
    let start = std::time::Instant::now();
    let (mut found, mut tried) = (None, 0u32);
    for mask in 0u32..(1 << 14) {
        let candidate: Vec<String> = (0..14).filter(|i| mask & (1 << i) != 0).map(|i| oaiy_relay_core::ids::KNOWN_GRANTS[i].to_string()).collect();
        tried += 1;
        if math::verify_receipt(&desktop.verify_key(), "aokie", &candidate, 1000, &phone.thumbprint(), &pid, &sig).is_ok() {
            found = Some(candidate);
            break;
        }
    }
    println!("grant search: {tried} candidates in {:?} (debug build)", start.elapsed());
    let mut got = found.expect("the signed set was not among the subsets");
    got.sort();
    let mut want = signed;
    want.sort();
    assert_eq!(got, want);
}

// ------------------------------------------------------------------------------------------------------------------ the offer

fn crafted(offer: &NewOffer, f: impl Fn(&str) -> String) -> (String, String) {
    let text = f(&offer.offer.text);
    let mac = math::offer_mac(&derived(offer).mac_key, &text).unwrap();
    (text, mac)
}

#[test]
fn attack_an_offer_with_the_right_mac_is_still_refused_for_a_downgrade_a_small_order_key_or_a_bad_shape() {
    let w = &mut world(quick());
    let offer = w.new_offer();
    let o = &offer.offer;
    let mac_key = derived(&offer).mac_key;
    let identity = {
        let mut b = [0u8; 32];
        b[0] = 1;
        b
    };
    let (dpub, dthumb) = (o.desktop_endpoint.to_b64u(), o.desktop_endpoint.thumbprint());
    let (hpub, hthumb) = (o.host_ed25519.to_b64u(), o.host_ed25519.thumbprint());
    let zero_x = "A".repeat(43);
    let id_b64 = b64::encode(&identity);
    let id_thumb = thumbprint_of(&identity);
    let cases: Vec<(&str, Box<dyn Fn(&str) -> String>)> = vec![
        ("schemaVersion 2", Box::new(|t| t.replace("\"schemaVersion\":3", "\"schemaVersion\":2"))),
        ("schemaVersion 4", Box::new(|t| t.replace("\"schemaVersion\":3", "\"schemaVersion\":4"))),
        ("schemaVersion 3.0", Box::new(|t| t.replace("\"schemaVersion\":3", "\"schemaVersion\":3.0"))),
        ("kind of the response", Box::new(|t| t.replace("\"kind\":\"aokie_mobile_pairing\"", "\"kind\":\"aokie_mobile_pairing_response\""))),
        ("small-order desktop endpoint key (identity point) with its own thumbprint", {
            let (a, b, c, d) = (dpub.clone(), dthumb.clone(), id_b64.clone(), id_thumb.clone());
            Box::new(move |t| t.replace(&a, &c).replace(&b, &d))
        }),
        ("small-order host identity key with its own thumbprint", {
            let (a, b, c, d) = (hpub.clone(), hthumb.clone(), id_b64.clone(), id_thumb.clone());
            Box::new(move |t| t.replace(&a, &c).replace(&b, &d))
        }),
        ("all-zero desktop X25519 key", {
            let (a, z) = (o.desktop_x25519.to_b64u(), zero_x.clone());
            Box::new(move |t| t.replace(&format!("\"desktopX25519\":\"{a}\""), &format!("\"desktopX25519\":\"{z}\"")))
        }),
        ("all-zero host X25519 key", {
            let (a, z) = (o.host_x25519.to_b64u(), zero_x.clone());
            Box::new(move |t| t.replace(&format!("\"x25519\":\"{a}\""), &format!("\"x25519\":\"{z}\"")))
        }),
        ("a thumbprint that is not the key's", {
            let (b, other) = (dthumb.clone(), Signer::generate().unwrap().thumbprint());
            Box::new(move |t| t.replacen(&b, &other, 1))
        }),
        ("a host identity thumbprint that is not the key's", {
            let (b, other) = (hthumb.clone(), Signer::generate().unwrap().thumbprint());
            Box::new(move |t| t.replacen(&b, &other, 1))
        }),
        ("a lifetime of 601 s", {
            let e = o.expires_at;
            Box::new(move |t| t.replace(&format!("\"expiresAt\":{e}"), &format!("\"expiresAt\":{}", e + 1)))
        }),
        ("an extra member", Box::new(|t| t.replacen('{', "{\"extra\":1,", 1))),
        ("a repeated member", Box::new(|t| t.replacen('{', "{\"appId\":\"aokie\",", 1))),
        ("a missing member", Box::new(|t| t.replace("\"jti\":", "\"jtx\":"))),
        ("a relay over plain http (not loopback)", Box::new(|t| t.replace("\"url\":\"https://", "\"url\":\"http://"))),
        ("a relay url with a path", Box::new(|t| t.replace("\"url\":\"https://relay.stub.test\"", "\"url\":\"https://relay.stub.test/x\""))),
        ("a relay url with userinfo", Box::new(|t| t.replace("\"url\":\"https://relay.stub.test\"", "\"url\":\"https://u@relay.stub.test\""))),
        ("a desktop name with a control character", Box::new(|t| t.replace("Front desk PC", "Front\\u0007desk"))),
        ("a desktop name of 61 characters", Box::new(|t| t.replace("Front desk PC", &"n".repeat(61)))),
        ("a jti without the prefix", Box::new(|t| t.replace("\"jti\":\"pair-", "\"jti\":\"pxir-"))),
    ];
    for (what, f) in &cases {
        let (text, mac) = crafted(&offer, f);
        assert_ne!(text, o.text, "{what}: the alteration did not change the offer (a test bug)");
        let r = Offer::verify(&text, &mac, &mac_key);
        assert!(r.is_err(), "{what}: the offer was accepted");
    }
    // Control: the unaltered offer, and one with only whitespace changed (the MAC is over the text as it is, so it needs its own MAC and then parses).
    assert!(Offer::verify(&o.text, &offer.mac, &mac_key).is_ok());
    let (spaced, spaced_mac) = crafted(&offer, |t| t.replace(',', ", "));
    assert!(Offer::verify(&spaced, &spaced_mac, &mac_key).is_ok());
    assert!(Offer::verify(&spaced, &offer.mac, &mac_key).is_err(), "the MAC covers the text as stored");
}

#[test]
fn attack_a_phone_refuses_an_offer_that_names_another_relay_key_before_it_posts_anything() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let other = Signer::generate().unwrap().thumbprint();
    let (text, mac) = crafted(&offer, |t| t.replace(&w.env.stub.relay_thumbprint(), &other));
    let body = Json::obj([
        ("v", Json::int(1)),
        ("state", Json::str("open")),
        ("offer", Json::str(text)),
        ("mac", Json::str(mac)),
        ("exp", Json::int(offer.offer.expires_at)),
        ("time", Json::int(1)),
    ])
    .to_compact();
    // Scanned key: the key says one relay key, the MAC-verified offer another.
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 74);
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], body.clone()));
    assert!(matches!(p.fetch_offer(&cancel()), Err(PairingError::Protocol(oaiy_relay_core::Error::Mismatch(_)))));
    // Typed code: the offer's key is what the relay is proved against, and the relay does not have it.
    let mut t = w.phone(PairingInput::Typed { code: &offer.typed_code, host: "relay.stub.test" }, 75);
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], body));
    assert_eq!(t.fetch_offer(&cancel()).err(), Some(PairingError::Suspect));
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "nothing was posted");
}

#[test]
fn attack_the_windows_of_an_offer_and_of_a_response_are_30_seconds_either_side_and_not_a_second_more() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let (i, e) = (offer.offer.issued_at as i64, offer.offer.expires_at as i64);
    assert!(offer.offer.check_window(i - 30).is_ok() && offer.offer.check_window(i - 31).is_err());
    assert!(offer.offer.check_window(e + 29).is_ok() && offer.offer.check_window(e + 30).is_err());
    let phone = Signer::generate().unwrap();
    let t = now as u64;
    let claims = oaiy_relay_core::pairing::Claims {
        app_id: "aokie".into(),
        desktop_connection_id: offer.offer.desktop_connection_id.clone(),
        desktop_key_thumbprint: offer.offer.desktop_endpoint.thumbprint(),
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        display_name: None,
        mobile_endpoint: phone.verify_key(),
        mobile_x25519: X25519Secret::generate().unwrap().public_key(),
        pairing_nonce: offer.offer.nonce,
        jti: offer.offer.jti.clone(),
        issued_at: t,
        expires_at: t + 120,
    };
    let mac_key = derived(&offer).mac_key;
    let r = Response::build(&phone, &mac_key, claims).unwrap();
    assert!(r.verify(&offer.offer, &mac_key, now - 30).is_ok() && r.verify(&offer.offer, &mac_key, now - 31).is_err());
    assert!(r.verify(&offer.offer, &mac_key, now + 120 + 29).is_ok() && r.verify(&offer.offer, &mac_key, now + 120 + 30).is_err());
    // A relay time at the edge of what a header can say does not overflow the arithmetic.
    for hostile in [i64::MAX / 2, 9_999_999_999_999_999, 0, -1] {
        let _ = offer.offer.check_window(hostile);
        let _ = r.verify(&offer.offer, &mac_key, hostile);
    }
}

// ------------------------------------------------------------------------------------------------------------------ the key and the typed code

#[test]
fn attack_a_pairing_key_that_is_not_exactly_the_documented_form_is_refused() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let good = offer.pairing_uri.clone();
    assert!(PairingKey::parse(&good).is_ok());
    let s = good.split("&s=").nth(1).unwrap().split('&').next().unwrap().to_string();
    let f = good.split("&f=").nth(1).unwrap().split('&').next().unwrap().to_string();
    let bad: Vec<(&str, String)> = vec![
        ("v=2", good.replace("v=3", "v=2")),
        ("v=4", good.replace("v=3", "v=4")),
        ("v=03", good.replace("v=3", "v=03")),
        ("no v", good.replace("v=3&", "")),
        ("V=3", good.replace("v=3", "V=3")),
        ("scheme in capitals", good.replace("oaiy://", "OAIY://")),
        ("another host", good.replace("oaiy://pair?", "oaiy://enroll?")),
        ("a slash before the query", good.replace("oaiy://pair?", "oaiy://pair/?")),
        ("plain http relay", good.replace("u=https%3A%2F%2F", "u=http%3A%2F%2F")),
        ("userinfo in the relay", good.replace("relay.stub.test", "user%40relay.stub.test")),
        ("a path in the relay", good.replace("relay.stub.test", "relay.stub.test%2Fx")),
        ("a query in the relay", good.replace("relay.stub.test", "relay.stub.test%3Fx")),
        ("a fragment in the relay", good.replace("relay.stub.test", "relay.stub.test%23x")),
        ("a repeated s", format!("{good}&s={s}")),
        ("a repeated u", format!("{good}&u=https%3A%2F%2Fevil.example")),
        ("an s of 21 characters", good.replace(&s, &s[..21])),
        ("an s with a padding character", good.replace(&s, &format!("{}=", &s[..21]))),
        ("an s that is another spelling of the same bytes", good.replace(&s, &format!("{}B", &s[..21]))),
        ("an f of 42 characters", good.replace(&f, &f[..42])),
        ("an x with a leading zero", good.replace("&x=", "&x=0")),
        ("an x that is a float", good.replace("&x=", "&x=1e3&y=")),
        ("a parameter without a value", format!("{good}&flag")),
        ("more than 512 characters", format!("{good}&pad={}", "a".repeat(512))),
    ];
    for (what, text) in bad {
        let r = PairingKey::parse(&text);
        // ("an x that is a float" splits into x=1e3 and a harmless y: that one is refused by x)
        assert!(r.is_err(), "{what}: {text}");
    }
    // Unknown parameters are ignored, as the README says (even repeated ones).
    assert!(PairingKey::parse(&format!("{good}&unknown=1&unknown=2")).is_ok());
}

#[test]
fn attack_the_typed_route_builds_only_an_https_origin_of_the_host_the_owner_typed() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let code = offer.typed_code.clone();
    let host = |h: &str| PairingTarget::from_input(PairingInput::Typed { code: &code, host: h }).map(|t| t.relay.origin());
    assert_eq!(host("relay.stub.test").unwrap(), "https://relay.stub.test");
    assert_eq!(host("  RELAY.Stub.Test/ ").unwrap(), "https://relay.stub.test");
    assert_eq!(host("relay.stub.test:8443").unwrap(), "https://relay.stub.test:8443");
    assert_eq!(host("https://relay.stub.test").unwrap(), "https://relay.stub.test");
    assert_eq!(host("relay.stub.test\n").unwrap(), "https://relay.stub.test", "surrounding whitespace, a pasted newline included, is trimmed");
    for bad in [
        "",
        "user@relay.stub.test",
        "relay.stub.test/path",
        "relay.stub.test?x=1",
        "relay.stub.test#x",
        "relay stub test",
        "relay.stub.test\\evil.example",
        "http://relay.stub.test",
        "ftp://relay.stub.test",
        "relay.stub.test:0",
        "relay.stub.test:65536",
        "relay.stub.test:",
        "\u{ff52}elay.stub.test",
    ] {
        assert!(host(bad).is_err(), "{bad:?} gave {:?}", host(bad));
    }
    // A host with a trailing dot is another origin than the offer's, so the offer is refused and nothing is posted (no fuzzy match).
    let dotted = host("relay.stub.test.").unwrap();
    assert_eq!(dotted, "https://relay.stub.test.");
    assert_ne!(dotted, "https://relay.stub.test");
}

// ------------------------------------------------------------------------------------------------------------------ the order of the phone's steps

#[test]
fn attack_the_phones_steps_cannot_be_taken_out_of_order() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 76);
    // Nothing before the offer is fetched.
    assert!(matches!(p.respond(&cancel()), Err(PairingError::WrongState(_))));
    assert!(matches!(p.wait_outcome(None, &cancel()), Err(PairingError::WrongState(_))));
    assert!(p.sas().is_err());
    p.fetch_offer(&cancel()).unwrap();
    assert!(matches!(p.fetch_offer(&cancel()), Err(PairingError::WrongState(_))), "the offer is fetched once");
    // Not before the response is posted: a relay that says `approved` to a phone that never answered gets nothing.
    assert!(matches!(p.wait_outcome(None, &cancel()), Err(PairingError::WrongState(_))));
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")));
    let sas = p.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    let outcome = p.wait_outcome(None, &cancel()).unwrap();
    assert!(matches!(outcome, Outcome::Paired(_)));
    // After the end nothing runs again.
    assert!(matches!(p.respond(&cancel()), Err(PairingError::WrongState(_))));
    assert!(matches!(p.wait_outcome(None, &cancel()), Err(PairingError::WrongState(_))));
    assert!(matches!(p.fetch_offer(&cancel()), Err(PairingError::WrongState(_))));
}

#[test]
fn attack_a_relay_that_answers_forever_cannot_keep_the_phone_waiting_past_the_offer_and_paces_it() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 77);
    p.fetch_offer(&cancel()).unwrap();
    p.respond(&cancel()).unwrap();
    w.env.clock.clear_sleeps();
    w.env.stub.clear_log();
    w.env.stub.fail_next_on("/v1/pair/", 4000, Fault::Respond(200, vec![], answered_body(&offer, false)));
    let outcome = p.wait_outcome(None, &cancel()).unwrap();
    assert!(matches!(outcome, Outcome::Expired));
    let gets = w.env.stub.log().iter().filter(|r| r.method == "GET" && r.target.starts_with("/v1/pair/")).count();
    assert!((1000..=1800).contains(&gets), "{gets} requests for 1,500 seconds");
    assert!(w.env.clock.sleeps().iter().all(|d| *d >= Duration::from_secs(1)), "never faster than a second");
}

// ------------------------------------------------------------------------------------------------------------------ what is stored, and in what order

struct Secrets {
    fail_put: bool,
    calls: Mutex<Vec<String>>,
}
impl SecretStore for Secrets {
    fn get(&self, _: &str) -> Result<Option<zeroize::Zeroizing<Vec<u8>>>, StoreError> {
        Ok(None)
    }
    fn put(&self, name: &str, _: &[u8]) -> Result<(), StoreError> {
        self.calls.lock().unwrap().push(format!("put {name}"));
        if self.fail_put {
            Err(StoreError("no".into()))
        } else {
            Ok(())
        }
    }
    fn delete(&self, name: &str) -> Result<(), StoreError> {
        self.calls.lock().unwrap().push(format!("delete {name}"));
        Ok(())
    }
}
struct Profiles {
    fail: bool,
    calls: Mutex<Vec<String>>,
}
impl ProfileStore for Profiles {
    fn load(&self) -> Result<Option<RelayProfile>, StoreError> {
        Ok(None)
    }
    fn save(&self, _: &RelayProfile) -> Result<(), StoreError> {
        self.calls.lock().unwrap().push("save".into());
        if self.fail {
            Err(StoreError("no".into()))
        } else {
            Ok(())
        }
    }
    fn clear(&self) -> Result<(), StoreError> {
        self.calls.lock().unwrap().push("clear".into());
        Ok(())
    }
}

fn a_paired(seed: u64) -> Paired {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), seed);
    p.fetch_offer(&cancel()).unwrap();
    let sas = p.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    match p.wait_outcome(None, &cancel()).unwrap() {
        Outcome::Paired(x) => x,
        _ => panic!("not paired"),
    }
}

#[test]
fn attack_a_pairing_is_stored_token_first_and_a_half_made_one_is_taken_back() {
    let paired = a_paired(78);
    // The token cannot be stored: no profile is written.
    let (s, p) = (Secrets { fail_put: true, calls: Mutex::new(vec![]) }, Profiles { fail: false, calls: Mutex::new(vec![]) });
    assert_eq!(store_paired(&paired, &s, &p).err(), Some(PairingError::Store));
    assert!(p.calls.lock().unwrap().is_empty(), "a profile was written without its token");
    // The profile cannot be stored: the token goes back out.
    let (s, p) = (Secrets { fail_put: false, calls: Mutex::new(vec![]) }, Profiles { fail: true, calls: Mutex::new(vec![]) });
    assert_eq!(store_paired(&paired, &s, &p).err(), Some(PairingError::Store));
    assert_eq!(*s.calls.lock().unwrap(), vec!["put relay.token".to_string(), "delete relay.token".to_string()]);
    // Both work: token first, then the profile, and no delete.
    let (s, p) = (Secrets { fail_put: false, calls: Mutex::new(vec![]) }, Profiles { fail: false, calls: Mutex::new(vec![]) });
    store_paired(&paired, &s, &p).unwrap();
    assert_eq!(*s.calls.lock().unwrap(), vec!["put relay.token".to_string()]);
    assert_eq!(*p.calls.lock().unwrap(), vec!["save".to_string()]);
}

#[test]
fn attack_a_phone_refuses_an_offer_whose_window_is_over_or_not_yet_begun_even_while_the_relay_serves_it() {
    // The implementer's test of an expired offer (pairing_stub.rs) passes through the stub's own 404 and never reaches the phone's window check.
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    for (what, issued) in [("issued 1,000 s ago", now - 1000), ("issued 1,000 s ahead", now + 1000), ("issued 631 s ago", now - 631), ("issued 31 s ahead", now + 31)] {
        let offer = w.desktop.create_offer(&mut w.rng, issued).unwrap();
        w.desktop.open(&w.env.client, &w.token, &offer, &cancel()).unwrap();
        let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 90);
        let r = p.fetch_offer(&cancel());
        assert!(matches!(r, Err(PairingError::Protocol(oaiy_relay_core::Error::OutsideWindow(_)))), "{what}: {r:?}");
    }
    // Control: 599 s ago and 30 s ahead are inside the window.
    for issued in [now - 599, now + 30] {
        let offer = w.desktop.create_offer(&mut w.rng, issued).unwrap();
        w.desktop.open(&w.env.client, &w.token, &offer, &cancel()).unwrap();
        let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 91);
        assert!(p.fetch_offer(&cancel()).is_ok());
    }
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "nothing was posted");
}

#[test]
fn attack_a_pairing_cannot_be_made_with_a_client_of_another_relay_than_the_target() {
    let w = world(quick());
    let mut w = w;
    let offer = w.new_offer();
    let target = PairingTarget::from_input(PairingInput::Key(&offer.pairing_uri)).unwrap();
    // A client for another relay than the key names: the pairing would send the phone's keys to the one and verify against the other.
    let elsewhere = common::env::client_for(&w.env.stub, &w.env.clock, "https://other.stub.test", 92);
    let identity = common::pair::phone_identity(93);
    let r = PhonePairing::new(elsewhere, target, identity);
    assert!(matches!(r, Err(PairingError::Protocol(oaiy_relay_core::Error::Mismatch(_)))), "{:?}", r.err());
}

#[test]
fn attack_the_phone_posts_nothing_when_its_proof_of_the_relay_has_lapsed() {
    // README 8: the phone's keys are posted only to a relay proved a short while ago (600 s). A phone that took its time over the confirmation screens must prove again first.
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 94);
    p.fetch_offer(&cancel()).unwrap();
    w.env.clock.advance(Duration::from_secs(601));
    w.env.stub.clear_log();
    let r = p.respond(&cancel());
    assert!(matches!(r, Err(PairingError::Client(ClientError::NotProved))), "{:?}", r.err());
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "the phone's keys went to a relay that was not proved");
}

/// A phone whose relay answers `approved` with `member` left out of the answer (`None`: nothing left out).
fn approved_answer_without(member: Option<&str>) -> Result<Outcome, PairingError> {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut p = phone_with_keys(&w, &offer, 95);
    p.pairing.fetch_offer(&cancel()).unwrap();
    p.pairing.respond(&cancel()).unwrap();
    let now = w.env.client.relay_now_or_local() as u64;
    let g = vec!["state_read".to_string()];
    let sig = math::sign_receipt(&w.identity.endpoint, "aokie", &g, now, &p.thumb, p.pairing.pid()).unwrap();
    let sealed = b64::encode(&p.x25519.seal(TOKEN.as_bytes()).unwrap());
    let good = approved_body(&sealed, now, &sig, Some(&["state_read"]));
    let body = match member {
        None => good,
        Some(member) => {
            let Json::Obj(m) = oaiy_relay_core::json::parse(good.as_bytes()).unwrap() else { panic!() };
            Json::Obj(m.into_iter().filter(|(k, _)| k != member).collect()).to_compact()
        }
    };
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], body));
    p.pairing.wait_outcome(None, &cancel())
}

#[test]
fn attack_an_approved_answer_without_a_device_or_a_box_is_not_an_answer() {
    // Control: the whole answer pairs.
    assert!(matches!(approved_answer_without(None), Ok(Outcome::Paired(_))));
    // A malformed answer is a failure like any other (retried with a backoff, never acted on, never a profile): the real rendezvous then ends by itself.
    for member in ["deviceId", "sealedToken", "receipt"] {
        let r = approved_answer_without(Some(member));
        assert!(matches!(r, Ok(Outcome::Expired)), "{member}: {:?}", r.err());
    }
}
// ------------------------------------------------------------------------------------------------------------------ findings

#[test]
#[ignore = "finding: respond() after a reject re-sends the same, stale, claims"]
fn finding_a_phone_that_is_rejected_answers_again_with_the_claims_it_had_and_so_is_rejected_again() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 79);
    p.fetch_offer(&cancel()).unwrap();
    p.respond(&cancel()).unwrap();
    // The desktop was asleep: the response reached it after its 120 s (+ 30 s of slack) and it rejected it for its window.
    w.env.client.pair_reject(&w.token, &offer.pid, "window", &cancel()).unwrap();
    w.env.clock.advance(Duration::from_secs(200));
    assert!(matches!(p.wait_outcome(None, &cancel()).unwrap(), Outcome::Rejected));
    p.respond(&cancel()).unwrap();
    let texts: Vec<String> = w
        .env
        .stub
        .log()
        .iter()
        .filter(|r| r.target.ends_with("/response"))
        .map(|r| oaiy_relay_core::json::parse(r.body.as_bytes()).unwrap().get_str("response").unwrap().to_string())
        .collect();
    assert_eq!(texts.len(), 2);
    let now = w.env.client.relay_now_or_local();
    let verdict = w.desktop.receive_response(&format!("{}.2", offer.pid), &texts[1], now);
    assert!(matches!(verdict, PairEvent::AwaitingSas { .. }), "the second answer carries the first answer's claims, 200 s old: {verdict:?}");
}

#[test]
#[ignore = "finding: wait_outcome has no pause when a relay answers a requested hold at once"]
fn finding_a_relay_that_answers_a_hold_at_once_makes_the_phone_poll_without_a_pause() {
    let mut w = world(StubConfig { receipt_includes_grants: true, wait_default: 20, wait_max: 20, ..Default::default() });
    let offer = w.new_offer();
    let mut p = w.phone(PairingInput::Key(&offer.pairing_uri), 80);
    p.fetch_offer(&cancel()).unwrap();
    p.respond(&cancel()).unwrap();
    w.env.clock.clear_sleeps();
    w.env.stub.clear_log();
    // 300 answers of "still answered", each at once and each claiming the hold was granted (a proxy that does not hold, a relay behind a buffering CDN, a hostile relay).
    w.env.stub.fail_next_on("/v1/pair/", 300, Fault::Respond(200, vec![], answered_body(&offer, true)));
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], r#"{"v":1,"state":"denied","time":1}"#.into()));
    assert!(matches!(p.wait_outcome(None, &cancel()).unwrap(), Outcome::Denied));
    let gets = w.env.stub.log().iter().filter(|r| r.method == "GET" && r.target.contains("wait=20")).count();
    assert_eq!(gets, 301);
    let sleeps = w.env.clock.sleeps().len();
    assert!(sleeps >= 250, "{gets} requests in a row with {sleeps} pauses between them");
}

