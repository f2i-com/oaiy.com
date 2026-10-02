//! Review tests (F3, the reviewer's own, not the implementer's): attacks on the desktop party of pairing v3 and on the arithmetic it relies on.
//!
//! `attack_*` tests assert the secure behaviour and pass today. `finding_*` tests assert the behaviour the README or the design asks for and FAIL today; they are `#[ignore]`d so
//! that the suite stays green, and `cargo test -p oaiy-relay-core --test rv_pairing_desktop -- --ignored` shows each failure.

mod common;

use common::env::quick;
use common::pair::{grants, world, World};
use oaiy_relay_core::client::*;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::desktop::SasStep;
use oaiy_relay_core::pairing::math::{self, Sas, SasEntry};
use oaiy_relay_core::pairing::{Claims, DesktopPairing, NewOffer, PairEvent, PairingError, PairingInput, PairingKey, Response, SasOutcome};
use oaiy_relay_core::testing::stub::Fault;

fn cancel() -> Cancel {
    Cancel::new()
}

/// What an attacker who has read the QR (so knows `s`) can derive.
fn derived(offer: &NewOffer) -> math::Derived {
    PairingKey::parse(&offer.pairing_uri).unwrap().secret.derive().unwrap()
}

fn claims_for(offer: &NewOffer, phone: &Signer, now: u64) -> Claims {
    Claims {
        app_id: "aokie".into(),
        desktop_connection_id: offer.offer.desktop_connection_id.clone(),
        desktop_key_thumbprint: offer.offer.desktop_endpoint.thumbprint(),
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        display_name: Some("Rogue phone".into()),
        mobile_endpoint: phone.verify_key(),
        mobile_x25519: X25519Secret::generate().unwrap().public_key(),
        pairing_nonce: offer.offer.nonce,
        jti: offer.offer.jti.clone(),
        issued_at: now,
        expires_at: now + 120,
    }
}

fn response_text(offer: &NewOffer, phone: &Signer, claims: Claims) -> String {
    Response::build(phone, &derived(offer).mac_key, claims).unwrap().text
}

/// A pending pairing that has a good response (from an attacker with `s`) and so waits for the SAS; returns the offer and the SAS the "phone" would show.
fn awaiting_sas(w: &mut World) -> (NewOffer, Sas) {
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    let event = w.desktop.receive_response(&offer.pid, &text, now);
    assert!(matches!(event, PairEvent::AwaitingSas { .. }), "{event:?}");
    let sas = w.desktop.expected_sas(&offer.pid).unwrap();
    (offer, sas)
}

fn code_of(sas: &Sas) -> String {
    format!("{}{}", sas.chars12, sas.check)
}

/// 13 characters that are well formed (the check character is right) but are not `sas`.
fn wrong_code(sas: &Sas, salt: usize) -> String {
    let mut chars: Vec<char> = sas.chars12.chars().collect();
    let i = salt % 12;
    chars[i] = if chars[i] == '0' { '1' } else { '0' };
    let twelve: String = chars.into_iter().collect();
    format!("{twelve}{}", math::sas_check_char(&twelve))
}

// ------------------------------------------------------------------------------------------------------------------ the SAS gate

#[test]
fn attack_the_sas_gate_cannot_be_reordered_skipped_or_borrowed_from_another_pairing() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    // Nothing pending: no approval, no SAS entry.
    assert_eq!(w.desktop.approve_body("AAAAAAAAAAAAAAAAAAAAAA", &grants(), now), Err(PairingError::UnknownPairing));
    let (a, sas_a) = awaiting_sas(&mut w);
    let (b, sas_b) = awaiting_sas(&mut w);
    assert_ne!(code_of(&sas_a), code_of(&sas_b));
    // 1. Approval before the code: refused, and it stays refused after unfinished or malformed entries.
    assert_eq!(w.desktop.approve_body(&a.pid, &grants(), now), Err(PairingError::SasRequired));
    for typed in ["", "6NHN", "????-????-????-?", "ZZZZZZZZZZZZZZ"] {
        let step = w.desktop.submit_sas(&a.pid, typed).unwrap();
        assert!(matches!(step, SasStep::Incomplete | SasStep::Invalid), "{typed:?}: {step:?}");
    }
    assert_eq!(w.desktop.approve_body(&a.pid, &grants(), now), Err(PairingError::SasRequired));
    // 2. The right code of ANOTHER pairing is a wrong code here, and counts.
    assert_eq!(w.desktop.submit_sas(&a.pid, &code_of(&sas_b)).unwrap(), SasStep::Wrong { attempts_left: 2 });
    assert_eq!(w.desktop.approve_body(&a.pid, &grants(), now), Err(PairingError::SasRequired));
    // 3. The attempts of one pairing are not the attempts of the other.
    assert_eq!(w.desktop.submit_sas(&b.pid, &wrong_code(&sas_b, 0)).unwrap(), SasStep::Wrong { attempts_left: 2 });
    // 4. The right code after two wrong ones still approves (three attempts, not two), and only then is a body made.
    assert_eq!(w.desktop.submit_sas(&a.pid, &wrong_code(&sas_a, 1)).unwrap(), SasStep::Wrong { attempts_left: 1 });
    assert_eq!(w.desktop.submit_sas(&a.pid, &code_of(&sas_a)).unwrap(), SasStep::Confirmed);
    let body = w.desktop.approve_body(&a.pid, &grants(), now).unwrap();
    assert!(body.contains("\"approve\":true") && body.contains("\"receipt\""));
    // 5. A third wrong entry ends B: no further code, no approval, no SAS to read.
    assert_eq!(w.desktop.submit_sas(&b.pid, &wrong_code(&sas_b, 1)).unwrap(), SasStep::Wrong { attempts_left: 1 });
    assert_eq!(w.desktop.submit_sas(&b.pid, &wrong_code(&sas_b, 2)).unwrap(), SasStep::Exhausted);
    assert_eq!(w.desktop.submit_sas(&b.pid, &code_of(&sas_b)).err(), Some(PairingError::WrongState("no response is waiting for a code")));
    assert_eq!(w.desktop.approve_body(&b.pid, &grants(), now), Err(PairingError::WrongState("no response is waiting for a decision")));
    assert!(w.desktop.expected_sas(&b.pid).is_none());
}

#[test]
fn attack_malformed_entries_never_count_and_normalisation_does_not_widen_the_code() {
    let mut w = world(quick());
    let (a, sas) = awaiting_sas(&mut w);
    // A thousand typos, short entries, pasted newlines and look-alike letters: none is an attempt.
    for i in 0..1000 {
        let typed = match i % 5 {
            0 => "6NHN-K68M".to_string(),
            1 => format!("{}\n", code_of(&sas)),
            2 => format!("{}X", code_of(&sas)),
            3 => format!("\u{410}{}", code_of(&sas)),
            _ => format!("{}{}", sas.chars12, if sas.check == 'Z' { 'Y' } else { 'Z' }),
        };
        let step = w.desktop.submit_sas(&a.pid, &typed).unwrap();
        assert!(!matches!(step, SasStep::Wrong { .. } | SasStep::Exhausted | SasStep::Confirmed), "entry {i} {typed:?}: {step:?}");
    }
    // Separators, case and the look-alikes of Crockford are the same code; 'U' is not part of any code.
    let lower = format!("{}-{}-{}-{}", &sas.chars12[0..4], &sas.chars12[4..8], &sas.chars12[8..12], sas.check).to_lowercase();
    assert_eq!(math::judge_sas_entry(&sas, &lower), SasEntry::Right);
    assert_eq!(math::judge_sas_entry(&sas, &format!(" {} ", code_of(&sas))), SasEntry::Right);
    assert_eq!(math::judge_sas_entry(&sas, &code_of(&sas).replace('0', "O").replace('1', "l")), SasEntry::Right);
    assert_eq!(math::judge_sas_entry(&sas, "UUUUUUUUUUUUU"), SasEntry::Invalid);
    // Still three attempts.
    assert_eq!(w.desktop.submit_sas(&a.pid, &wrong_code(&sas, 0)).unwrap(), SasStep::Wrong { attempts_left: 2 });
}

#[test]
fn attack_single_character_errors_in_the_typed_code_and_in_the_sas_are_caught_locally() {
    // The design asks (9.2): every single-character substitution of the 28-character code is offered to the checksum and at least 99 percent are rejected locally.
    const ALPHABET: &str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    let (mut total, mut accepted) = (0u32, 0u32);
    for seed in 0..40u8 {
        let secret: [u8; 16] = std::array::from_fn(|i| (i as u8).wrapping_mul(seed.wrapping_add(7)).wrapping_add(seed ^ 0x5a));
        let code = math::typed_code(&secret).replace('-', "");
        assert_eq!(code.len(), 28);
        assert_eq!(math::parse_typed_code(&code).unwrap().expose(), &secret);
        for pos in 0..28 {
            for c in ALPHABET.chars() {
                if code.chars().nth(pos) == Some(c) {
                    continue;
                }
                let mut v: Vec<char> = code.chars().collect();
                v[pos] = c;
                let text: String = v.into_iter().collect();
                total += 1;
                if math::parse_typed_code(&text).is_ok() {
                    accepted += 1;
                }
            }
        }
    }
    assert!(accepted * 100 <= total, "{accepted} of {total} single-character errors passed the checksum");
    // The SAS: a substituted data character is caught by the check character with probability 31/32 (the rest are counted attempts, as the design says), and the check character
    // itself, substituted, is always caught.
    let w = &mut world(quick());
    let (_, sas) = awaiting_sas(w);
    let (mut caught, mut counted, mut checks_caught) = (0, 0, 0);
    for pos in 0..12 {
        for c in ALPHABET.chars() {
            if sas.chars12.chars().nth(pos) == Some(c) {
                continue;
            }
            let mut v: Vec<char> = sas.chars12.chars().collect();
            v[pos] = c;
            let typed: String = v.into_iter().collect::<String>() + &sas.check.to_string();
            match math::judge_sas_entry(&sas, &typed) {
                SasEntry::BadCheck => caught += 1,
                SasEntry::Wrong => counted += 1,
                other => panic!("{typed}: {other:?}"),
            }
        }
    }
    for c in ALPHABET.chars().filter(|c| *c != sas.check) {
        let typed = format!("{}{}", sas.chars12, c);
        assert_eq!(math::judge_sas_entry(&sas, &typed), SasEntry::BadCheck);
        checks_caught += 1;
    }
    assert_eq!(checks_caught, 31);
    assert!(caught > 10 * counted, "{caught} caught, {counted} counted");
}

// ------------------------------------------------------------------------------------------------------------------ the response

#[test]
fn attack_every_binding_of_a_response_to_its_offer_is_checked_even_when_the_mac_and_signature_are_right() {
    // The attacker has `s` (a photographed QR): the MAC and the signature are good, so only the binding can stop a response that is for something else.
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    type Alter = Box<dyn Fn(&mut Claims)>;
    let cases: Vec<(&str, Alter)> = vec![
        ("app", Box::new(|c: &mut Claims| c.app_id = "otherapp".into())),
        ("desktop connection", Box::new(|c: &mut Claims| c.desktop_connection_id = "dev-AAAAAAAAAAAAAAAAAAAAAQ".into())),
        ("desktop key", Box::new(|c: &mut Claims| c.desktop_key_thumbprint = Signer::generate().unwrap().thumbprint())),
        ("nonce", Box::new(|c: &mut Claims| c.pairing_nonce[0] ^= 1)),
        ("jti", Box::new(|c: &mut Claims| c.jti = "pair-AAAAAAAAAAAAAAAA".into())),
    ];
    for (what, alter) in cases {
        let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
        let phone = Signer::generate().unwrap();
        let mut claims = claims_for(&offer, &phone, now as u64);
        alter(&mut claims);
        let text = response_text(&offer, &phone, claims);
        assert_eq!(w.desktop.receive_response(&offer.pid, &text, now), PairEvent::Rejected { pid: offer.pid.clone(), reason: "binding" }, "{what}");
    }
    // Control: the same attacker, unaltered claims, is a response like any other.
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    assert!(matches!(w.desktop.receive_response(&offer.pid, &text, now), PairEvent::AwaitingSas { .. }));
}

#[test]
fn attack_a_second_response_cannot_replace_the_one_the_owner_is_checking() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let (first, second) = (Signer::generate().unwrap(), Signer::generate().unwrap());
    let t1 = response_text(&offer, &first, claims_for(&offer, &first, now as u64));
    let t2 = response_text(&offer, &second, claims_for(&offer, &second, now as u64));
    assert!(matches!(w.desktop.receive_response(&offer.pid, &t1, now), PairEvent::AwaitingSas { .. }));
    let sas1 = w.desktop.expected_sas(&offer.pid).unwrap();
    // A second valid response (the relay posts it as `pid.2`): ignored. The SAS and the approval are still the first phone's.
    assert!(matches!(w.desktop.receive_response(&format!("{}.2", offer.pid), &t2, now), PairEvent::Ignored(_)));
    assert_eq!(w.desktop.expected_sas(&offer.pid).unwrap(), sas1);
    // Same item again, and the first text under the next id: ignored.
    assert!(matches!(w.desktop.receive_response(&offer.pid, &t1, now), PairEvent::Ignored(_)));
    assert!(matches!(w.desktop.receive_response(&format!("{}.3", offer.pid), &t1, now), PairEvent::Ignored(_)));
    assert_eq!(w.desktop.submit_sas(&offer.pid, &code_of(&sas1)).unwrap(), SasStep::Confirmed);
    let body = w.desktop.approve_body(&offer.pid, &grants(), now).unwrap();
    assert!(body.contains(&first.verify_key().to_b64u()) && !body.contains(&second.verify_key().to_b64u()), "{body}");
}

#[test]
fn attack_a_response_that_is_malformed_in_any_member_is_rejected_before_any_check_of_its_value() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let phone = Signer::generate().unwrap();
    let good = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    let bad: Vec<(&str, String)> = vec![
        ("schemaVersion 2", good.replace("\"schemaVersion\":3", "\"schemaVersion\":2")),
        ("schemaVersion 3.0", good.replace("\"schemaVersion\":3", "\"schemaVersion\":3.0")),
        ("kind", good.replace("aokie_mobile_pairing_response", "aokie_mobile_pairing")),
        ("extra member", good.replacen('{', "{\"x\":1,", 1)),
        ("duplicate member", good.replacen('{', "{\"kind\":\"aokie_mobile_pairing_response\",", 1)),
        ("extra claim", good.replace("\"claims\":{", "\"claims\":{\"extra\":1,")),
        ("lifetime 121", good.replace(&format!("\"expiresAt\":{}", now + 120), &format!("\"expiresAt\":{}", now + 121))),
        ("display name with a control character", good.replace("Rogue phone", "Rogue\\u0007phone")),
        ("display name of 61 characters", good.replace("Rogue phone", &"x".repeat(61))),
        ("signature not base64url", good.replacen("\"signature\":\"", "\"signature\":\"=", 1)),
        ("empty", String::new()),
        ("an array", "[]".into()),
    ];
    for (i, (what, text)) in bad.iter().enumerate() {
        let id = if i == 0 { offer.pid.clone() } else { format!("{}.{}", offer.pid, 2 + (i % 2)) };
        // Each is a distinct item, but the id space is three wide: use a fresh party for each, so that "already judged" cannot be the reason.
        let mut party = DesktopPairing::new(w.identity.clone(), "aokie", w.profile.relay.clone(), &w.profile.relay_thumbprint);
        let offer2 = party.create_offer_with(*derived_secret(&offer).expose(), offer.offer.nonce, offer.offer.jti.clone(), now).unwrap();
        assert_eq!(offer2.pid, offer.pid);
        let event = party.receive_response(&id, text, now);
        assert!(matches!(event, PairEvent::Rejected { reason: "malformed" | "key", .. }), "{what}: {event:?}");
    }
    // Control.
    let mut party = DesktopPairing::new(w.identity.clone(), "aokie", w.profile.relay.clone(), &w.profile.relay_thumbprint);
    party.create_offer_with(*derived_secret(&offer).expose(), offer.offer.nonce, offer.offer.jti.clone(), now).unwrap();
    assert!(matches!(party.receive_response(&offer.pid, &good, now), PairEvent::AwaitingSas { .. }));
}

fn derived_secret(offer: &NewOffer) -> math::PairingSecret {
    PairingKey::parse(&offer.pairing_uri).unwrap().secret
}

#[test]
fn attack_items_that_are_not_for_the_party_are_ignored_whoever_sends_them() {
    let mut w = world(quick());
    let (offer, _) = awaiting_sas(&mut w);
    let item = |id: &str, lane: &str, from: &str| Item {
        seq: 1,
        id: id.to_string(),
        lane: lane.to_string(),
        from: from.to_string(),
        at: 0,
        exp: 0,
        hdr: "{}".into(),
        body: "{}".into(),
        rp: None,
    };
    for (id, lane, from) in [
        (offer.pid.as_str(), "cmd", "relay"),
        (offer.pid.as_str(), "pair", "dev-AAAAAAAAAAAAAAAAAAAAAA"),
        (offer.pid.as_str(), "pair", "prov-AAAAAAAAAAAAAAAAAAAAAA"),
        ("not-a-pid", "pair", "relay"),
        ("AAAAAAAAAAAAAAAAAAAAAA.4", "pair", "relay"),
        ("AAAAAAAAAAAAAAAAAAAAAA.2.x", "pair", "relay"),
    ] {
        let event = w.desktop.on_pair_item(&w.env.client, &w.token, &item(id, lane, from), &cancel()).unwrap();
        assert!(matches!(event, PairEvent::Ignored(_)), "{id} {lane} {from}: {event:?}");
    }
}

#[test]
fn attack_a_relay_that_replays_a_judged_response_under_a_new_id_or_after_the_decision_gets_nothing() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 61);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    let now = w.env.client.relay_now_or_local();
    let body = {
        let reply = w.env.client.poll(&w.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &cancel()).unwrap();
        Item::from_json(&reply.body.unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap().body
    };
    assert!(matches!(w.desktop.receive_response(&offer.pid, &body, now), PairEvent::AwaitingSas { .. }));
    assert!(matches!(w.desktop.receive_response(&format!("{}.2", offer.pid), &body, now), PairEvent::Ignored(_)), "while the owner types");
    assert!(matches!(
        w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap(),
        SasOutcome::Approved { .. }
    ));
    assert!(matches!(w.desktop.receive_response(&format!("{}.3", offer.pid), &body, now), PairEvent::Ignored(_)), "after the decision");
    // And a replay into a party that never saw the offer (a restart): no such pairing, so nothing to judge.
    let mut fresh = DesktopPairing::new(w.identity.clone(), "aokie", w.profile.relay.clone(), &w.profile.relay_thumbprint);
    assert!(matches!(fresh.receive_response(&offer.pid, &body, now), PairEvent::Ignored("no such pairing")));
}

#[test]
fn attack_a_consumed_nonce_and_jti_are_refused_by_a_party_that_restored_them() {
    // The consumed set is only reachable through `restore_consumed` on a party that has the same pending offer again: the implementer's test of it (pairing_stub.rs) uses a party with
    // no pending offer, which ignores the item for another reason.
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let (secret, nonce, jti) = ([3u8; 16], [4u8; 32], "pair-AAAAAAAAAAAAAAAA".to_string());
    let offer = w.desktop.create_offer_with(secret, nonce, jti.clone(), now).unwrap();
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    assert!(matches!(w.desktop.receive_response(&offer.pid, &text, now), PairEvent::AwaitingSas { .. }));
    let sas = w.desktop.expected_sas(&offer.pid).unwrap();
    assert_eq!(w.desktop.submit_sas(&offer.pid, &code_of(&sas)).unwrap(), SasStep::Confirmed);
    w.desktop.approve_body(&offer.pid, &grants(), now).unwrap();
    let consumed = w.desktop.consumed();
    assert_eq!(consumed.len(), 1);
    let mut b = DesktopPairing::new(w.identity.clone(), "aokie", w.profile.relay.clone(), &w.profile.relay_thumbprint);
    b.restore_consumed(consumed);
    let again = b.create_offer_with(secret, nonce, jti, now).unwrap();
    assert_eq!(again.pid, offer.pid);
    assert_eq!(b.receive_response(&offer.pid, &text, now), PairEvent::Rejected { pid: offer.pid.clone(), reason: "replayed" });
}

#[test]
fn attack_three_wrong_codes_tell_the_relay_to_deny_and_to_burn() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 66);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.env.stub.clear_log();
    let wrong = |i: usize| {
        let mut chars: Vec<char> = sas.chars12.chars().collect();
        chars[i] = if chars[i] == '0' { '1' } else { '0' };
        let twelve: String = chars.into_iter().collect();
        format!("{twelve}{}", math::sas_check_char(&twelve))
    };
    for i in 0..2 {
        assert!(matches!(
            w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &wrong(i), &grants(), &cancel()).unwrap(),
            SasOutcome::Wrong { .. }
        ));
    }
    assert_eq!(w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &wrong(2), &grants(), &cancel()).unwrap(), SasOutcome::Denied);
    let log = w.env.stub.log();
    assert!(log.iter().any(|r| r.target.ends_with("/decision") && r.body.contains("\"approve\":false")), "the denial was not sent: {log:?}");
    assert!(log.iter().any(|r| r.target.ends_with("/burn")), "the rendezvous was not burned");
    // And nothing was sent before the third entry.
    assert_eq!(log.iter().filter(|r| r.target.ends_with("/decision") || r.target.ends_with("/burn")).count(), 2);
}

#[test]
fn attack_the_approval_refuses_grants_that_the_relay_or_the_phone_would_refuse() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    for (what, bad) in [
        ("an unknown name", vec!["state_read".to_string(), "bogus".to_string()]),
        ("a repeated name", vec!["state_read".to_string(), "state_read".to_string()]),
        ("seventeen names", (0..17).map(|i| format!("g{i}")).collect::<Vec<_>>()),
    ] {
        let (a, sas) = awaiting_sas(&mut w);
        assert_eq!(w.desktop.submit_sas(&a.pid, &code_of(&sas)).unwrap(), SasStep::Confirmed);
        let r = w.desktop.approve_body(&a.pid, &bad, now);
        assert!(matches!(r, Err(PairingError::Protocol(_))), "{what}: {r:?}");
    }
}

#[test]
fn attack_a_valid_response_in_an_item_of_another_lane_or_another_sender_is_not_a_response() {
    // The pending offer waits for a response (so that nothing but the item filter can stop the item), and the body is a response an attacker with `s` could make.
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    let item = |lane: &str, from: &str| Item {
        seq: 1,
        id: offer.pid.clone(),
        lane: lane.to_string(),
        from: from.to_string(),
        at: 0,
        exp: 0,
        hdr: "{}".into(),
        body: text.clone(),
        rp: None,
    };
    for (lane, from) in [
        ("cmd", "relay"),
        ("res", "relay"),
        ("sync", "relay"),
        ("pair", "dev-AAAAAAAAAAAAAAAAAAAAAA"),
        ("pair", "prov-AAAAAAAAAAAAAAAAAAAAAA"),
        ("pair", ""),
    ] {
        let event = w.desktop.on_pair_item(&w.env.client, &w.token, &item(lane, from), &cancel()).unwrap();
        assert!(matches!(event, PairEvent::Ignored(_)), "{lane} from {from:?}: {event:?}");
    }
    assert_eq!(w.desktop.open_count(), 1);
    // The same item as the relay posts it is a response.
    let event = w.desktop.on_pair_item(&w.env.client, &w.token, &item("pair", "relay"), &cancel()).unwrap();
    assert!(matches!(event, PairEvent::AwaitingSas { .. }), "{event:?}");
}

#[test]
fn attack_a_decision_answer_that_does_not_say_what_was_decided_is_refused() {
    let w = world(quick());
    // `approved` with no device, `denied` with a device, a device that is not one, a state that is not one: none is an answer to a decision.
    for body in [
        r#"{"v":1,"state":"approved","time":1}"#,
        r#"{"v":1,"state":"denied","deviceId":"dev-AAAAAAAAAAAAAAAAAAAAAA","time":1}"#,
        r#"{"v":1,"state":"approved","deviceId":"phone-1","time":1}"#,
        r#"{"v":1,"state":"open","time":1}"#,
    ] {
        w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], body.to_string()));
        let r = w.env.client.pair_decision(&w.token, "AAAAAAAAAAAAAAAAAAAAAA", "{\"approve\":true}", &cancel());
        assert!(matches!(r, Err(ClientError::BadAnswer(_))), "{body}: {r:?}");
    }
    w.env.stub.fail_next_on(
        "/v1/pair/",
        1,
        Fault::Respond(200, vec![], r#"{"v":1,"state":"approved","deviceId":"dev-AAAAAAAAAAAAAAAAAAAAAA","time":1}"#.into()),
    );
    let ok = w.env.client.pair_decision(&w.token, "AAAAAAAAAAAAAAAAAAAAAA", "{\"approve\":true}", &cancel()).unwrap();
    assert_eq!(ok.state, "approved");
}

// ------------------------------------------------------------------------------------------------------------------ findings

#[test]
#[ignore = "finding: the desktop never expires a pending offer"]
fn finding_the_desktop_accepts_a_fresh_response_to_an_offer_that_expired_long_ago() {
    // An attacker who has the secret of an old QR (photographed, a screenshot) and a relay that delivers what it is given (a hostile or compromised one): the offer's window is 600 s
    // (README 10.1) and the desktop is the only party that knows the offer is over, but `receive_response` judges only the response's own 120 s window.
    let mut w = world(quick());
    let offer = w.new_offer();
    let later = offer.offer.expires_at as i64 + 4000;
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, later as u64));
    let event = w.desktop.receive_response(&offer.pid, &text, later);
    assert!(matches!(event, PairEvent::Rejected { .. }), "an hour after the offer expired, a response was taken: {event:?}");
}

#[test]
#[ignore = "finding: confirm_sas cannot be retried after a failed decision"]
fn finding_confirm_sas_strands_the_pairing_after_one_failed_decision_request() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 62);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    // The network drops the decision once.
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Drop);
    assert!(w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).is_err());
    // The owner types the code again (it was right the first time).
    let again = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel());
    assert!(matches!(again, Ok(SasOutcome::Approved { .. })), "the retry of the helper that is the whole gate: {again:?}");
}

#[test]
fn attack_the_lower_level_retry_of_a_failed_decision_works_and_sends_the_same_receipt() {
    // The workaround the finding above leaves: approve_body again (same body, same receipt) and the idempotent decision.
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 63);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Drop);
    assert!(w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).is_err());
    let now = w.env.client.relay_now_or_local();
    w.env.clock.advance(std::time::Duration::from_secs(5));
    let body = w.desktop.approve_body(&offer.pid, &grants(), now + 5).unwrap();
    let ack = w.env.client.pair_decision(&w.token, &offer.pid, &body, &cancel()).unwrap();
    assert_eq!(ack.state, "approved");
    w.desktop.approved(&offer.pid);
    assert_eq!(w.env.stub.log().iter().filter(|r| r.target.ends_with("/decision")).count(), 2, "the dropped request and its retry");
    assert!(matches!(phone.wait_outcome(Some(&grants()), &cancel()), Ok(oaiy_relay_core::pairing::phone::Outcome::Paired(_))));
}

#[test]
#[ignore = "finding: pair_decision, pair_reject and pair_burn put the pid into the path without validating it"]
fn finding_the_desktops_three_pairing_calls_do_not_validate_the_pid_they_put_in_the_path() {
    // `pair_create`, `pair_fetch` and `pair_respond` refuse a pid that is not 22 characters of the alphabet; these three, which carry the desktop's token, do not (and
    // `DesktopPairing::burn` passes whatever it is given, pending or not).
    let w = world(quick());
    let evil = "AAAAAAAAAAAAAAAAAAAAAA/../../../v1/devices/revoke?x=";
    w.env.stub.clear_log();
    let _ = w.env.client.pair_burn(&w.token, evil, &cancel());
    let _ = w.env.client.pair_reject(&w.token, evil, "x", &cancel());
    let _ = w.env.client.pair_decision(&w.token, evil, "{\"approve\":false}", &cancel());
    let sent: Vec<String> = w.env.stub.log().iter().map(|r| r.target.clone()).filter(|t| t.contains("..")).collect();
    assert!(sent.is_empty(), "requests carrying the desktop's token were sent to {sent:?}");
}

#[test]
#[ignore = "finding: the stub answers 409 to the same approval again, the real relay 200 (README 10.1 table)"]
fn finding_the_stub_does_not_answer_the_same_approval_again_as_the_relay_does() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 65);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel()).unwrap();
    let body = w.env.stub.log().iter().rev().find(|r| r.target.ends_with("/decision")).unwrap().body.clone();
    let again = w.env.client.pair_decision(&w.token, &offer.pid, &body, &cancel());
    assert!(again.is_ok(), "the relay answers the same approval again with 200 (PHP Pairing::decide), the stub: {again:?}");
}

#[test]
#[ignore = "finding: a confirmed pairing approves for any typed text"]
fn finding_once_the_code_was_right_any_later_entry_approves() {
    let mut w = world(quick());
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 64);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    // The owner types the right code, but the host passes a grant list that is refused: the pairing stays confirmed and unapproved.
    let refused = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &["not_a_grant".to_string()], &cancel());
    assert!(matches!(refused, Err(PairingError::Protocol(_))), "{refused:?}");
    // Anything at all then approves it: the typed text is no longer looked at.
    let empty = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, "", &grants(), &cancel());
    assert!(!matches!(empty, Ok(SasOutcome::Approved { .. })), "an empty entry approved a pairing: {empty:?}");
}

#[test]
#[ignore = "finding: create_offer_with replaces a pending pairing of the same pid and resets its attempts"]
fn finding_a_second_offer_with_the_same_secret_resets_the_attempt_counter() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let secret = [7u8; 16];
    let offer = w.desktop.create_offer_with(secret, [9u8; 32], "pair-AAAAAAAAAAAAAAAA".into(), now).unwrap();
    let phone = Signer::generate().unwrap();
    let text = response_text(&offer, &phone, claims_for(&offer, &phone, now as u64));
    assert!(matches!(w.desktop.receive_response(&offer.pid, &text, now), PairEvent::AwaitingSas { .. }));
    let sas = w.desktop.expected_sas(&offer.pid).unwrap();
    assert_eq!(w.desktop.submit_sas(&offer.pid, &wrong_code(&sas, 0)).unwrap(), SasStep::Wrong { attempts_left: 2 });
    assert_eq!(w.desktop.submit_sas(&offer.pid, &wrong_code(&sas, 1)).unwrap(), SasStep::Wrong { attempts_left: 1 });
    // The same secret again (a host that derives it from something it keeps): the pending pairing is replaced, and the same response is judged anew with three attempts.
    let again = w.desktop.create_offer_with(secret, [9u8; 32], "pair-AAAAAAAAAAAAAAAA".into(), now);
    assert!(again.is_err(), "a pid that is in flight was replaced");
}

#[test]
fn attack_the_sas_depends_on_both_endpoint_keys_in_order_the_nonce_and_the_raw_pid() {
    // A relay that substitutes the phone's key (so that the desktop would show other numbers than the phone) is seen by the owner: the SAS of two phone keys differ.
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let offer = w.desktop.create_offer(&mut w.rng, now).unwrap();
    let d = derived(&offer);
    let (k1, k2) = (Signer::generate().unwrap().verify_key().to_bytes(), Signer::generate().unwrap().verify_key().to_bytes());
    let dk = offer.offer.desktop_endpoint.to_bytes();
    let s1 = math::sas(&dk, &k1, &offer.offer.nonce, &d.pid).unwrap();
    let s2 = math::sas(&dk, &k2, &offer.offer.nonce, &d.pid).unwrap();
    let s3 = math::sas(&k1, &dk, &offer.offer.nonce, &d.pid).unwrap();
    let mut other_nonce = offer.offer.nonce;
    other_nonce[31] ^= 1;
    let s4 = math::sas(&dk, &k1, &other_nonce, &d.pid).unwrap();
    let mut other_pid = d.pid;
    other_pid[0] ^= 1;
    let s5 = math::sas(&dk, &k1, &offer.offer.nonce, &other_pid).unwrap();
    for (name, s) in [("other phone key", &s2), ("keys swapped", &s3), ("other nonce", &s4), ("other pid", &s5)] {
        assert_ne!(s.raw, s1.raw, "{name}");
    }
    // The formula of README 10.1, computed here once more: HKDF(IKM = desktop || phone, salt = nonce, info = "oaiy/pairing/3/sas" 0x00 pid-raw, L = 8).
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(&dk);
    ikm[32..].copy_from_slice(&k1);
    let mut info = b"oaiy/pairing/3/sas\0".to_vec();
    info.extend_from_slice(&d.pid);
    assert_eq!(info.len(), 35);
    let mut raw = [0u8; 8];
    oaiy_crypto::kdf::hkdf_sha256(&ikm, Some(&offer.offer.nonce), &info, &mut raw).unwrap();
    assert_eq!(raw, s1.raw);
    // Reading the pid as its 22-character text (the mistake of Interpretation 21) gives another value.
    let mut wrong_info = b"oaiy/pairing/3/sas\0".to_vec();
    wrong_info.extend_from_slice(math::pid_text(&d.pid).as_bytes());
    assert_eq!(wrong_info.len(), 41);
    let mut wrong = [0u8; 8];
    oaiy_crypto::kdf::hkdf_sha256(&ikm, Some(&offer.offer.nonce), &wrong_info, &mut wrong).unwrap();
    assert_ne!(wrong, s1.raw);
    // Twelve characters carry the top 60 bits, and the display is 13 characters in groups of four.
    assert_eq!(s1.chars12.len(), 12);
    assert_eq!(s1.display().len(), 16);
}
