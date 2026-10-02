//! Pairing v3 between this crate's desktop party and its phone party through the in-process stub relay: the whole ceremony by scanned key and by typed code, the gap in the contract
//! about the receipt's grants, and the adversarial list of the design (a wrong MAC, three wrong codes, a typo that does not count, a replay, a revoked key, an expired offer or response,
//! keys of small order, a relay that swaps the offer, an offer answered twice, a relay that says "approved" with no desktop decision behind it, a receipt that does not verify, a typed
//! route to a host that serves no valid offer).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::env::{held_requests, quick};
use common::pair::{grants, phone_identity, world, World};
use oaiy_relay_core::admission::{MobileExpect, MobileRequest, Transport};
use oaiy_relay_core::b64;
use oaiy_relay_core::client::*;
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::keys::{SignDomain, Signer};
use oaiy_relay_core::pairing::math::{self, PairingKey, PairingSecret};
use oaiy_relay_core::pairing::phone::{store_paired, Outcome};
use oaiy_relay_core::pairing::{Claims, PairEvent, PairingError, PairingInput, PairingTarget, PhonePairing, Response, SasOutcome};
use oaiy_relay_core::testing::stub::{Fault, StubConfig};
use oaiy_relay_core::testing::{FakeClock, SeededRng};

fn fixed(w: &World) -> StubConfig {
    let _ = w;
    StubConfig { receipt_includes_grants: true, ..quick() }
}

fn cancel() -> Cancel {
    Cancel::new()
}

fn paired(outcome: Outcome) -> oaiy_relay_core::pairing::phone::Paired {
    match outcome {
        Outcome::Paired(p) => p,
        Outcome::Denied => panic!("denied"),
        Outcome::Expired => panic!("expired"),
        Outcome::Rejected => panic!("rejected"),
    }
}

#[test]
fn a_whole_pairing_by_scanning_the_key_ends_with_a_phone_that_polls_and_is_admitted() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 1);
    let summary = phone.fetch_offer(&cancel()).unwrap();
    assert_eq!((summary.desktop_name.as_str(), summary.relay_host.as_str(), summary.app_id.as_str()), ("Front desk PC", "relay.stub.test", "aokie"));
    // The phone showed the owner the relay's host and the desktop's name before anything was posted.
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "nothing posted before the owner confirmed");
    let sas = phone.respond(&cancel()).unwrap();
    let events = w.deliver();
    assert!(matches!(&events[0], PairEvent::AwaitingSas { phone_name: Some(n), .. } if n == "Test phone"), "{events:?}");
    // The owner types the phone's 13 characters on the desktop.
    let outcome = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    let SasOutcome::Approved { device_id } = outcome else { panic!("{outcome:?}") };
    let result = paired(phone.wait_outcome(None, &cancel()).unwrap());
    assert_eq!(result.profile.device_id, device_id);
    assert_eq!(result.profile.kind, ProfileKind::Phone);
    assert_eq!(result.profile.grants, grants());
    let peer = result.profile.peer.clone().unwrap();
    assert_eq!(peer.desktop_endpoint, w.identity.endpoint.verify_key(), "the pin comes from the MAC-verified offer");
    assert_eq!(peer.host_ed25519, w.identity.host_ed25519);
    assert_eq!(result.profile.relay_thumbprint, w.env.stub.relay_thumbprint());
    // The token is stored first, then the profile.
    let secrets = MemorySecretStore::new();
    let profiles = MemoryProfileStore::new();
    store_paired(&result, &secrets, &profiles).unwrap();
    assert_eq!(profiles.load().unwrap().unwrap(), result.profile);
    // It is a token the relay accepts: a poll, and an admission read as the shipped phone reads it.
    let client = common::env::client_for(&w.env.stub, &w.env.clock, &w.env.stub.public_url(), 77);
    client.prove(&cancel()).unwrap();
    w.env.clock.advance(Duration::from_secs(1));
    let reply = client.poll(&result.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
    assert_eq!(reply.status, Some(200));
    let request = MobileRequest {
        app_id: "aokie".into(),
        device_id: result.profile.device_id.clone(),
        display_name: Some("Test phone".into()),
        holder_thumbprint: phone.endpoint_thumbprint(),
        transports: Some(vec![Transport::RelayPoll]),
    };
    let expect = MobileExpect { app_id: "aokie", device_id: &result.profile.device_id, holder_thumbprint: &request.holder_thumbprint };
    let admission = client.admission_mobile(&result.token, &request, &expect, &cancel()).unwrap();
    assert_eq!(admission.scopes, grants());
    assert_eq!(admission.expected_peer_thumbprint, w.identity.endpoint.thumbprint(), "advisory, and here it is the desktop's");
    // The desktop party has nothing left in flight and has consumed the response's nonce and jti.
    assert_eq!(w.desktop.open_count(), 0);
    assert_eq!(w.desktop.consumed().len(), 1);
}

#[test]
fn a_whole_pairing_by_typed_code_proves_the_relay_against_the_fingerprint_inside_the_offer() {
    let mut w = world(fixed(&world(quick())));
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Typed { code: &offer.typed_code.to_lowercase(), host: "relay.stub.test" }, 2);
    w.env.stub.clear_log();
    let summary = phone.fetch_offer(&cancel()).unwrap();
    assert_eq!(summary.desktop_name, "Front desk PC");
    // The offer was fetched first (no key to prove against yet), and the proof followed it.
    let log = w.env.stub.log();
    assert!(log[0].target.starts_with("/v1/pair/"), "{log:?}");
    assert_eq!(log[1].target, "/v1/info");
    assert!(log.iter().all(|r| r.authorization.is_none()));
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    assert!(matches!(phone.wait_outcome(None, &cancel()).unwrap(), Outcome::Paired(_)));
}

#[test]
fn a_typed_code_with_a_typo_is_refused_before_any_request_is_made() {
    let mut w = world(quick());
    let offer = w.new_offer();
    w.env.stub.clear_log();
    let mut typo: Vec<char> = offer.typed_code.chars().collect();
    typo[2] = if typo[2] == 'A' { 'B' } else { 'A' };
    let typo: String = typo.into_iter().collect();
    assert!(PairingTarget::from_input(PairingInput::Typed { code: &typo, host: "relay.stub.test" }).is_err());
    assert!(w.env.stub.log().is_empty(), "a typo costs nothing: no request, no attempt");
}

#[test]
fn the_receipt_covers_grants_the_relay_does_not_return_so_the_phone_fails_closed_without_them() {
    // The shipped relay returns `{issuedAt, signature}`: the grants are in the signed document and not on the wire to the phone.
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 3);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    // 1. Nothing to verify the receipt against: no profile, no token.
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::ReceiptGrantsUnknown));
    // 2. A guess is not good enough: the receipt does not verify over other grants.
    assert_eq!(phone.wait_outcome(Some(&["state_read".to_string()]), &cancel()).err(), Some(PairingError::ReceiptInvalid));
    let mut more = grants();
    more.push("takeover".into());
    assert_eq!(phone.wait_outcome(Some(&more), &cancel()).err(), Some(PairingError::ReceiptInvalid));
    // 3. The grants the desktop approved, from somewhere the relay cannot forge, verify.
    assert!(matches!(phone.wait_outcome(Some(&grants()), &cancel()).unwrap(), Outcome::Paired(_)));
}

#[test]
fn a_wrong_mac_is_rejected_and_the_rendezvous_is_open_again_for_the_real_phone() {
    let mut w = world(quick());
    let offer = w.new_offer();
    // Someone who has read the offer on the relay (it is public) but does not know the secret answers it with a MAC key of its own.
    let attacker = Signer::generate().unwrap();
    let claims = |nonce: [u8; 32], jti: &str| Claims {
        app_id: "aokie".into(),
        desktop_connection_id: offer.offer.desktop_connection_id.clone(),
        desktop_key_thumbprint: offer.offer.desktop_endpoint.thumbprint(),
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        display_name: Some("Evil".into()),
        mobile_endpoint: attacker.verify_key(),
        mobile_x25519: oaiy_relay_core::keys::X25519Secret::generate().unwrap().public_key(),
        pairing_nonce: nonce,
        jti: jti.into(),
        issued_at: w.env.client.relay_now_or_local() as u64,
        expires_at: w.env.client.relay_now_or_local() as u64 + 120,
    };
    let bad_mac_key = PairingSecret::new([9; 16]).derive().unwrap().mac_key;
    let evil = Response::build(&attacker, &bad_mac_key, claims(offer.offer.nonce, &offer.offer.jti)).unwrap();
    let anon = common::env::client_for(&w.env.stub, &w.env.clock, &w.env.stub.public_url(), 5);
    anon.prove(&cancel()).unwrap();
    anon.pair_respond(&offer.pid, &evil.text, &cancel()).unwrap();
    let events = w.deliver();
    assert_eq!(events, vec![PairEvent::Rejected { pid: offer.pid.clone(), reason: "mac" }]);
    // The relay returned the rendezvous to `open`: the genuine phone can still pair.
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 6);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    let again = w.deliver();
    assert!(again.iter().any(|e| matches!(e, PairEvent::AwaitingSas { .. })), "{again:?}");
    assert!(matches!(
        w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap(),
        SasOutcome::Approved { .. }
    ));
}

#[test]
fn an_attacker_who_has_the_secret_but_signs_with_a_key_other_than_the_one_in_the_claims_is_rejected_on_the_signature() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let secret_text = offer.pairing_uri.split("&s=").nth(1).unwrap().split('&').next().unwrap();
    let derived = PairingSecret::new(b64::decode_exact::<16>(secret_text).unwrap()).derive().unwrap();
    let (phone_key, other_key) = (Signer::generate().unwrap(), Signer::generate().unwrap());
    let now = w.env.client.relay_now_or_local() as u64;
    let claims = Claims {
        app_id: "aokie".into(),
        desktop_connection_id: offer.offer.desktop_connection_id.clone(),
        desktop_key_thumbprint: offer.offer.desktop_endpoint.thumbprint(),
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        display_name: None,
        mobile_endpoint: phone_key.verify_key(),
        mobile_x25519: oaiy_relay_core::keys::X25519Secret::generate().unwrap().public_key(),
        pairing_nonce: offer.offer.nonce,
        jti: offer.offer.jti.clone(),
        issued_at: now,
        expires_at: now + 120,
    };
    // MAC right, signature by another key.
    let canonical = claims.canonical().unwrap();
    let text = format!(
        "{{\"kind\":\"aokie_mobile_pairing_response\",\"schemaVersion\":3,\"claims\":{canonical},\"signature\":{},\"mac\":{}}}",
        json::quote(&other_key.sign_b64u(SignDomain::PairingResponse, &[canonical.as_bytes()])),
        json::quote(&math::response_mac(&derived.mac_key, &canonical).unwrap())
    );
    assert_eq!(w.desktop.receive_response(&offer.pid, &text, now as i64), PairEvent::Rejected { pid: offer.pid.clone(), reason: "signature" });
}

#[test]
fn a_response_with_a_key_of_small_order_is_rejected_whatever_else_in_it_is_right() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let secret_text = offer.pairing_uri.split("&s=").nth(1).unwrap().split('&').next().unwrap();
    let derived = PairingSecret::new(b64::decode_exact::<16>(secret_text).unwrap()).derive().unwrap();
    let phone_key = Signer::generate().unwrap();
    let now = w.env.client.relay_now_or_local() as u64;
    let good = Claims {
        app_id: "aokie".into(),
        desktop_connection_id: offer.offer.desktop_connection_id.clone(),
        desktop_key_thumbprint: offer.offer.desktop_endpoint.thumbprint(),
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        display_name: None,
        mobile_endpoint: phone_key.verify_key(),
        mobile_x25519: oaiy_relay_core::keys::X25519Secret::generate().unwrap().public_key(),
        pairing_nonce: offer.offer.nonce,
        jti: offer.offer.jti.clone(),
        issued_at: now,
        expires_at: now + 120,
    };
    let canonical = good.canonical().unwrap();
    let zero_x = "A".repeat(43);
    // The identity point of X25519, with the MAC and the signature made over the claims as altered.
    let altered = canonical.replace(&good.mobile_x25519.to_b64u(), &zero_x);
    let signed = |c: &str| {
        format!(
            "{{\"kind\":\"aokie_mobile_pairing_response\",\"schemaVersion\":3,\"claims\":{c},\"signature\":{},\"mac\":{}}}",
            json::quote(&phone_key.sign_b64u(SignDomain::PairingResponse, &[c.as_bytes()])),
            json::quote(&math::response_mac(&derived.mac_key, c).unwrap())
        )
    };
    assert_eq!(w.desktop.receive_response(&offer.pid, &signed(&altered), now as i64), PairEvent::Rejected { pid: offer.pid.clone(), reason: "key" });
    // (Each text is its own item: a second body under the id of one already judged is the same item delivered again, and is ignored.)
    // The identity point of Ed25519 as the phone's endpoint key.
    let identity_key = format!("A{}", "Q".chars().chain(std::iter::repeat_n('A', 41)).collect::<String>());
    let altered = canonical.replace(&phone_key.verify_key().to_b64u(), &identity_key);
    assert_eq!(
        w.desktop.receive_response(&format!("{}.2", offer.pid), &signed(&altered), now as i64),
        PairEvent::Rejected { pid: offer.pid.clone(), reason: "key" }
    );
    // Control: the unaltered claims are accepted.
    assert!(matches!(w.desktop.receive_response(&format!("{}.3", offer.pid), &signed(&canonical), now as i64), PairEvent::AwaitingSas { .. }));
}

#[test]
fn three_wrong_codes_deny_and_burn_and_a_typo_never_counts() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 7);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    let try_code = |w: &mut World, code: &str| w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, code, &grants(), &cancel()).unwrap();
    // Approving before the code is typed is refused by the party itself.
    assert_eq!(w.desktop.approve_body(&offer.pid, &grants(), 1_790_000_000), Err(PairingError::SasRequired));
    // Typos and unfinished entries cost nothing, however many.
    for _ in 0..10 {
        assert_eq!(try_code(&mut w, "6NHN-K68M"), SasOutcome::Incomplete);
        assert_eq!(try_code(&mut w, ""), SasOutcome::Incomplete);
        assert_eq!(try_code(&mut w, "????-????-????-?"), SasOutcome::Invalid);
    }
    let real = sas.display();
    let mut bad_check = real.clone();
    bad_check.pop();
    bad_check.push(if real.ends_with('A') { 'B' } else { 'A' });
    assert_eq!(try_code(&mut w, &bad_check), SasOutcome::BadCheck);
    // A wrong code with a good check character: an attempt, three of them.
    let wrong = |seed: &str| format!("{seed}{}", math::sas_check_char(seed));
    assert_eq!(
        try_code(
            &mut w,
            &format!(
                "{}-{}-{}-{}",
                &wrong("3P8VVXBPSYYN")[0..4],
                &wrong("3P8VVXBPSYYN")[4..8],
                &wrong("3P8VVXBPSYYN")[8..12],
                &wrong("3P8VVXBPSYYN")[12..]
            )
        ),
        SasOutcome::Wrong { attempts_left: 2 }
    );
    assert_eq!(try_code(&mut w, &wrong("8D47W7XJQDS0")), SasOutcome::Wrong { attempts_left: 1 });
    assert_eq!(try_code(&mut w, &wrong("EDN6328D9EK3")), SasOutcome::Denied);
    // The right code after that is too late: the pairing is gone, and the relay's rendezvous with it.
    assert_eq!(
        w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &real, &grants(), &cancel()).err(),
        Some(PairingError::WrongState("no response is waiting for a code"))
    );
    assert!(matches!(phone.wait_outcome(Some(&grants()), &cancel()).unwrap(), Outcome::Expired | Outcome::Denied));
}

#[test]
fn a_replayed_response_a_revoked_key_and_a_response_outside_its_window_are_rejected() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 8);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    let item = {
        let reply = w.env.client.poll(&w.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
        Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap()
    };
    let now = w.env.client.relay_now_or_local();
    // A revoked key.
    let thumb = Response::parse(&item.body).unwrap().claims.mobile_endpoint.thumbprint();
    w.desktop.revoke_phone_key(&thumb);
    assert_eq!(w.desktop.receive_response(&item.id, &item.body, now), PairEvent::Rejected { pid: offer.pid.clone(), reason: "revoked" });
    // After the 120 seconds of the claims and 30 of slack, the response is no longer one.
    let mut fresh = world(quick());
    let o2 = fresh.new_offer();
    let mut p2 = fresh.phone(PairingInput::Key(&o2.pairing_uri), 9);
    p2.fetch_offer(&cancel()).unwrap();
    p2.respond(&cancel()).unwrap();
    let reply = fresh.env.client.poll(&fresh.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
    let item2 = Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap();
    let t = fresh.env.client.relay_now_or_local();
    assert_eq!(fresh.desktop.receive_response(&item2.id, &item2.body, t + 151), PairEvent::Rejected { pid: o2.pid.clone(), reason: "window" });
    assert_eq!(
        fresh.desktop.receive_response(&format!("{}.2", o2.pid), &item2.body, t - 31),
        PairEvent::Rejected { pid: o2.pid.clone(), reason: "window" },
        "issued in the future"
    );
    // Inside both windows the same response is taken (a response is judged once per item id, and `.3` is a new one).
    assert!(matches!(fresh.desktop.receive_response(&format!("{}.3", o2.pid), &item2.body, t), PairEvent::AwaitingSas { .. }));
    // (That a nonce an approval consumed is not a response any more is `a_nonce_that_an_approval_consumed_cannot_be_used_again`, and that a party that restored the consumed set refuses
    // it is `attack_a_consumed_nonce_and_jti_are_refused_by_a_party_that_restored_them`: the lines that stood here only showed that a party with no such offer ignores an item.)
}

#[test]
fn a_nonce_that_an_approval_consumed_cannot_be_used_again() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 10);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    let reply = w.env.client.poll(&w.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
    let item = Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap();
    let now = w.env.client.relay_now_or_local();
    assert!(matches!(w.desktop.receive_response(&item.id, &item.body, now), PairEvent::AwaitingSas { .. }));
    assert!(matches!(
        w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap(),
        SasOutcome::Approved { .. }
    ));
    // The same response again, as a relay that replays it would: the pairing is done, and the nonce is consumed.
    assert!(matches!(w.desktop.receive_response(&item.id, &item.body, now), PairEvent::Ignored(_)));
    assert_eq!(w.desktop.consumed().len(), 1);
}

#[test]
fn an_offer_that_the_relay_swapped_is_never_parsed_and_nothing_is_posted() {
    let mut w = world(quick());
    let offer = w.new_offer();
    // A hostile relay serves another desktop's offer text, with the MAC it has: the MAC covers the text, so this fails before anything in the text is read.
    let swapped = offer.offer.text.replace("Front desk PC", "Front desk PD");
    let body = Json::obj([
        ("v", Json::int(1)),
        ("state", Json::str("open")),
        ("offer", Json::str(swapped)),
        ("mac", Json::str(offer.mac.clone())),
        ("exp", Json::int(offer.offer.expires_at)),
        ("time", Json::int(w.env.client.relay_now_or_local())),
    ])
    .to_compact();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], body));
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 11);
    assert!(matches!(phone.fetch_offer(&cancel()), Err(PairingError::Protocol(oaiy_relay_core::Error::BadMac(_)))));
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "nothing was answered");
    // And an offer whose relay.url is another relay's: the right MAC (the desktop wrote it) and the wrong place.
    let mut w2 = world(quick());
    let o2 = w2.new_offer();
    let mut elsewhere = common::pair::client_for_target(&w2.env, &PairingTarget::from_input(PairingInput::Key(&o2.pairing_uri)).unwrap(), 12);
    let _ = &mut elsewhere;
    let other_url = oaiy_relay_core::url::RelayUrl::parse("https://other.stub.test").unwrap();
    let target = PairingTarget::from_input(PairingInput::Typed { code: &o2.typed_code, host: "other.stub.test" }).unwrap();
    assert_eq!(target.relay, other_url);
    let client = common::pair::client_for_target(&w2.env, &target, 13);
    let mut p2 = PhonePairing::new(client, target, phone_identity(14)).unwrap();
    assert!(
        matches!(p2.fetch_offer(&cancel()), Err(PairingError::Protocol(oaiy_relay_core::Error::Mismatch(_)))),
        "the offer says it is for another relay"
    );
}

#[test]
fn an_offer_answered_twice_is_the_second_phones_already_answered_and_a_lost_202_is_retried_with_the_same_text() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut first = w.phone(PairingInput::Key(&offer.pairing_uri), 15);
    first.fetch_offer(&cancel()).unwrap();
    // The first phone's 202 is lost on the way: it asks again with the same text, which the relay answers 202 again, and nothing is posted twice.
    w.env.stub.clear_log();
    first.respond(&cancel()).unwrap();
    first.respond(&cancel()).unwrap();
    assert_eq!(w.env.stub.log().iter().filter(|r| r.target.ends_with("/response")).count(), 2);
    assert_eq!(w.env.stub.live_items(&w.profile.device_id), 1, "one pair item, not two");
    // A second phone that has the same key (it was shown the same QR) finds the offer answered.
    let mut second = w.phone(PairingInput::Key(&offer.pairing_uri), 16);
    assert_eq!(second.fetch_offer(&cancel()).err(), Some(PairingError::AlreadyAnswered));
}

#[test]
fn a_hostile_relay_that_says_approved_gets_a_phone_that_stores_nothing() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 17);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    let approved = |sealed: &str, issued: u64, sig: &str, grants: &[&str]| {
        Json::obj([
            ("v", Json::int(1)),
            ("state", Json::str("approved")),
            ("deviceId", Json::str("dev-AAAAAAAAAAAAAAAAAAAAAA")),
            ("sealedToken", Json::str(sealed)),
            (
                "receipt",
                Json::obj([
                    ("issuedAt", Json::int(issued)),
                    ("signature", Json::str(sig)),
                    ("grants", Json::Arr(grants.iter().map(|g| Json::str(*g)).collect())),
                ]),
            ),
            ("time", Json::int(1)),
        ])
        .to_compact()
    };
    let now = w.env.client.relay_now_or_local() as u64;
    let garbage_sig = "A".repeat(86);
    // 1. No desktop decision at all: a made-up receipt and a made-up box.
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved("AAAA", now, &garbage_sig, &["state_read"])));
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::ReceiptInvalid));
    // 2. A receipt signed by a key that is not the desktop's (the relay's own, say), over the right document.
    let relay_like = Signer::generate().unwrap();
    let thumb = phone.endpoint_thumbprint();
    let forged = math::sign_receipt(&relay_like, "aokie", &["state_read".to_string()], now, &thumb, phone.pid()).unwrap();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved("AAAA", now, &forged, &["state_read"])));
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::ReceiptInvalid));
    // 3. A receipt that is the desktop's own but dated before the phone asked (a receipt kept from some other pairing).
    let old = math::sign_receipt(&w.identity.endpoint, "aokie", &["state_read".to_string()], now - 1000, &thumb, phone.pid()).unwrap();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved("AAAA", now - 1000, &old, &["state_read"])));
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::ReceiptInvalid));
    // 4. A good receipt and a box that is not for this phone (sealed to another key): the token does not open.
    let good = math::sign_receipt(&w.identity.endpoint, "aokie", &["state_read".to_string()], now, &thumb, phone.pid()).unwrap();
    let stranger = oaiy_relay_core::keys::X25519Secret::generate().unwrap().public_key();
    let box_for_another = b64::encode(&stranger.seal(b"oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8").unwrap());
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], approved(&box_for_another, now, &good, &["state_read"])));
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::TokenInvalid));
}

#[test]
fn a_denied_pairing_and_a_rejected_response_are_told_apart() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 18);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    w.deliver();
    // The owner says no.
    w.desktop.deny(&w.env.client, &w.token, &offer.pid, &cancel()).unwrap();
    assert!(matches!(phone.wait_outcome(Some(&grants()), &cancel()).unwrap(), Outcome::Denied));
    // A response the desktop rejected returns the rendezvous to open: the phone is told, and may answer again (a second item, `pid.2`).
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 19);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    let reply = w.env.client.poll(&w.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
    assert_eq!(Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap().id, offer.pid);
    w.env.client.pair_reject(&w.token, &offer.pid, "test", &cancel()).unwrap();
    assert!(matches!(phone.wait_outcome(Some(&grants()), &cancel()).unwrap(), Outcome::Rejected));
    phone.respond(&cancel()).unwrap();
    w.env.clock.advance(Duration::from_secs(1));
    let reply = w.env.client.poll(&w.token, &PollRequest { since: 1, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
    assert_eq!(Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap().id, format!("{}.2", offer.pid));
}

#[test]
fn a_relay_that_is_not_the_pinned_one_is_refused_before_anything_is_posted() {
    // The key's `f` is the thumbprint of another relay key: the proof fails, the phone does not post its keys.
    let mut w = world(quick());
    let offer = w.new_offer();
    let tampered = offer.pairing_uri.replace(&w.env.stub.relay_thumbprint(), "SWUejy55xcz8xgSi-GX15ERZzyuXnb6voSaopq2Jakw");
    let mut phone = w.phone(PairingInput::Key(&tampered), 20);
    w.env.stub.clear_log();
    assert_eq!(phone.fetch_offer(&cancel()).err(), Some(PairingError::Suspect));
    assert!(
        w.env.stub.log().iter().all(|r| !r.target.ends_with("/response") && !r.target.starts_with("/v1/pair")),
        "no pairing request left the phone"
    );
    // A phone that calls respond() with no offer is told it cannot.
    assert!(matches!(phone.respond(&cancel()), Err(PairingError::WrongState(_))));
}

#[test]
fn the_typed_route_to_a_host_that_serves_no_valid_offer_is_not_found() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Typed { code: &offer.typed_code, host: "relay.stub.test" }, 21);
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(404, vec![], r#"{"error":{"code":"not_found","message":"Not found."}}"#.into()));
    assert_eq!(phone.fetch_offer(&cancel()).err(), Some(PairingError::NotFound));
    // A host that answers 200 with something that is not an offer.
    let mut phone = w.phone(PairingInput::Typed { code: &offer.typed_code, host: "relay.stub.test" }, 22);
    w.env.stub.fail_next_on(
        "/v1/pair/",
        1,
        Fault::Respond(
            200,
            vec![],
            "{\"v\":1,\"state\":\"open\",\"offer\":\"{}\",\"mac\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\",\"exp\":1,\"time\":1}".into(),
        ),
    );
    assert!(phone.fetch_offer(&cancel()).is_err());
    // An offer that expired, 700 seconds on. The honest relay no longer has the rendezvous (it answers 404, which is `NotFound`), which says nothing about the phone's own check, so a
    // relay that still serves the offer, with its right MAC, is played too: the phone's window refuses it whatever the relay says.
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 23);
    w.env.clock.advance(Duration::from_secs(700));
    assert_eq!(phone.fetch_offer(&cancel()).err(), Some(PairingError::NotFound), "the stub forgot the rendezvous");
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 24);
    let still_served = Json::obj([
        ("v", Json::int(1)),
        ("state", Json::str("open")),
        ("offer", Json::str(offer.offer.text.clone())),
        ("mac", Json::str(offer.mac.clone())),
        ("exp", Json::int(offer.offer.expires_at)),
        ("time", Json::int(w.env.client.relay_now_or_local())),
    ])
    .to_compact();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], still_served));
    let err = phone.fetch_offer(&cancel()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(oaiy_relay_core::Error::OutsideWindow(_))), "{err:?}");
    assert!(w.env.stub.log().iter().all(|r| !r.target.ends_with("/response")), "an expired offer is not answered");
}

#[test]
fn the_desktop_party_retries_opening_a_rendezvous_with_the_same_request_and_gets_the_original_answer() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let again = w.desktop.open(&w.env.client, &w.token, &offer, &cancel()).unwrap();
    assert_eq!(again.pid, offer.pid);
    assert_eq!(again.exp, offer.offer.expires_at);
    // A different offer under the same pid is a conflict.
    let mut other = offer.create.clone();
    other.mac = b64::encode(&[7u8; 32]);
    assert_eq!(w.env.client.pair_create(&w.token, &other, &cancel()).unwrap_err().code(), Some("conflict"));
}

#[test]
fn a_receipt_dated_in_the_future_is_not_an_answer_to_this_pairing_though_it_is_signed() {
    // The desktop's own clock is 200 seconds ahead of the relay's and the phone's (it samples the relay's time and then its monotonic clock runs on): the receipt it signs is dated
    // 200 seconds ahead of what the phone knows the time to be, and with 30 seconds of slack that is not an answer to a pairing that was asked a moment ago. The signature is right:
    // only the date can refuse it.
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let desk_clock = Arc::new(FakeClock::new(common::env::T0));
    let desk_client = Arc::new(RelayClient::new(
        w.profile.relay.clone(),
        Some(w.profile.relay_thumbprint.clone()),
        Arc::new(w.env.stub.clone()),
        desk_clock.clone(),
        Box::new(SeededRng::new(55)),
        ClientConfig::default(),
    ));
    desk_client.prove(&cancel()).unwrap();
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 62);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    assert!(matches!(w.deliver()[0], PairEvent::AwaitingSas { .. }));
    desk_clock.advance(Duration::from_secs(200));
    let outcome = w.desktop.confirm_sas(&desk_client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { .. }));
    assert_eq!(phone.wait_outcome(None, &cancel()).err(), Some(PairingError::ReceiptInvalid));

    // The same with the desktop 20 seconds ahead (inside the 30 of slack): the phone pairs.
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let desk_clock = Arc::new(FakeClock::new(common::env::T0));
    let desk_client = Arc::new(RelayClient::new(
        w.profile.relay.clone(),
        Some(w.profile.relay_thumbprint.clone()),
        Arc::new(w.env.stub.clone()),
        desk_clock.clone(),
        Box::new(SeededRng::new(56)),
        ClientConfig::default(),
    ));
    desk_client.prove(&cancel()).unwrap();
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 63);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    assert!(matches!(w.deliver()[0], PairEvent::AwaitingSas { .. }));
    desk_clock.advance(Duration::from_secs(20));
    w.desktop.confirm_sas(&desk_client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    assert!(matches!(phone.wait_outcome(None, &cancel()).unwrap(), Outcome::Paired(_)));
}

#[test]
fn a_phone_that_waits_for_the_owner_holds_one_pairing_read_at_a_time_as_the_relays_own_count_shows() {
    // MOB-21a, the phone's side: while the owner has not typed the code the phone's read of the rendezvous is held by the relay, and the relay's count (`GET /v1/admin/status`) shows one
    // such request, never two, and no poll (the phone has no token yet).
    let mut w = world(StubConfig { wait_default: 2, wait_max: 2, receipt_includes_grants: true, ..Default::default() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 61);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    assert!(matches!(w.deliver()[0], PairEvent::AwaitingSas { .. }));
    assert_eq!(held_requests(&w.env.stub, &w.token), (0, 0, 0), "nothing is held before the phone waits");
    let (stub, token) = (w.env.stub.clone(), w.token.clone());
    let (most, with_one, result) = std::thread::scope(|s| {
        let waiting = s.spawn(|| phone.wait_outcome(None, &cancel()));
        let start = std::time::Instant::now();
        let (mut most, mut with_one) = (0u64, 0u32);
        while start.elapsed() < Duration::from_millis(2600) {
            let (live, polls, pairs) = held_requests(&stub, &token);
            assert_eq!((polls, live), (0, pairs), "the phone holds no poll");
            most = most.max(pairs);
            with_one += u32::from(pairs == 1);
            std::thread::sleep(Duration::from_millis(15));
        }
        // The owner types the code: the approval wakes the held read, which ends the wait.
        let outcome = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
        assert!(matches!(outcome, SasOutcome::Approved { .. }));
        (most, with_one, waiting.join().unwrap())
    });
    assert_eq!(most, 1, "never more than one pairing read held");
    assert!(with_one >= 30, "the phone's read was held in {with_one} of the samples");
    assert!(matches!(result.unwrap(), Outcome::Paired(_)));
    assert_eq!(held_requests(&w.env.stub, &w.token), (0, 0, 0), "nothing is held once the phone has its answer");
}

#[test]
fn a_pairing_key_that_expired_before_the_offer_did_is_refused_with_thirty_seconds_of_slack() {
    // The key carries its own expiry (`x`), which can be earlier than the offer's (a QR that was photographed and kept): it is judged by the phone against the relay's time, with the
    // same 30 seconds of slack as the offer's window, whatever the offer's own window says.
    let mut w = world(quick());
    let offer = w.new_offer();
    let key = PairingKey::parse(&offer.pairing_uri).unwrap();
    w.env.clock.advance(Duration::from_secs(100));
    let now = w.env.client.relay_now_or_local();
    for (x_offset, accepted) in [(-29i64, true), (-30, false), (-200, false), (30, true)] {
        let uri = PairingKey::to_uri(&key.relay, key.relay_thumbprint.as_deref().unwrap(), &key.secret, (now + x_offset) as u64);
        let mut phone = w.phone(PairingInput::Key(&uri), 70);
        let result = phone.fetch_offer(&cancel());
        assert_eq!(result.is_ok(), accepted, "a key that expires {x_offset} s from now: {:?}", result.as_ref().err());
        if !accepted {
            assert!(matches!(result.unwrap_err(), PairingError::Protocol(oaiy_relay_core::Error::OutsideWindow(_))));
        }
    }
}

#[test]
fn wrong_codes_typed_after_the_right_one_never_deny_a_pairing_whose_approval_is_on_its_way() {
    // The right code was typed and the decision sent, but its answer was lost: the relay may have approved already, and a denial after that is a conflict. Wrong entries from now on
    // cost nothing and never deny or burn; the right code again approves with the same receipt.
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 71);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    assert!(matches!(w.deliver()[0], PairEvent::AwaitingSas { .. }));
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Drop);
    assert!(
        w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).is_err(),
        "the decision's answer was lost"
    );
    w.env.stub.clear_log();
    let wrong = math::sas(&[1u8; 32], &[2u8; 32], &[3u8; 32], &[4u8; 16]).unwrap().display();
    assert_ne!(wrong, sas.display());
    for _ in 0..4 {
        let outcome = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &wrong, &grants(), &cancel()).unwrap();
        assert_eq!(outcome, SasOutcome::Wrong { attempts_left: 3 }, "no attempt is used up once the approval has been sent");
    }
    assert!(
        w.env.stub.log().iter().all(|r| !r.target.ends_with("/burn") && !r.body.contains("\"approve\":false")),
        "nothing was denied or burned: {:?}",
        w.env.stub.log()
    );
    let outcome = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { .. }), "{outcome:?}");
    assert!(matches!(phone.wait_outcome(None, &cancel()).unwrap(), Outcome::Paired(_)));
}
