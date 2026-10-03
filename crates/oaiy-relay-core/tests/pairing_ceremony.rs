//! `fixtures/pairing-ceremony.json` replayed step by step by this crate's two parties: the desktop's and the phone's requests are built from the vectors' keys and values and must
//! be **exactly** the ones the real relay recorded (the offer, its MAC, the response with its claims, signature and MAC, the decision with its receipt), the relay's recorded answers
//! are fed back, and the ceremony ends with the SAS of the recording, a verified receipt and the token of `fixtures/sealed-token.json`.
//!
//! The replay of honest data passes whether or not a check is there, so a second group of tests plays the **same recording with one artifact damaged** (the offer's MAC or text, the
//! window, the response's signature, MAC or claims, the typed code, the receipt's signature, date or grants, the sealed token) and requires the party that must refuse to refuse, at the
//! step where it must, and nothing to have been sent that should not have been. A check that is removed from the crate fails one of them.

mod common;

use std::sync::{Arc, Mutex};

use common::{hex, load, unhex, unhex32, At};
use oaiy_crypto::kdf::sha256;
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::client::*;
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::keys::{Signer, X25519Public, X25519Secret};
use oaiy_relay_core::pairing::phone::Outcome;
use oaiy_relay_core::pairing::{
    DesktopIdentity, DesktopPairing, NewOffer, PairEvent, PairingError, PairingInput, PairingTarget, PhoneIdentity, PhonePairing, SasOutcome,
};
use oaiy_relay_core::testing::stub::{StubConfig, StubRelay};
use oaiy_relay_core::testing::{json_response, FakeClock, Logged, ScriptedHttp, SeededRng};
use oaiy_relay_core::url::RelayUrl;
use oaiy_relay_core::Error;

const T0: i64 = 1_790_000_000;

fn seed(v: &Json, path: &str) -> Signer {
    Signer::from_seed(&Secret::new(unhex32(v.s(path))))
}

/// The member or element of `v` at a dotted path (a number indexes an array), to change.
fn slot<'a>(v: &'a mut Json, path: &str) -> &'a mut Json {
    let mut cur = v;
    for part in path.split('.') {
        cur = match cur {
            Json::Arr(items) => items
                .get_mut(part.parse::<usize>().unwrap_or_else(|_| panic!("{path}: {part} is no index")))
                .unwrap_or_else(|| panic!("{path}: no {part}")),
            Json::Obj(members) => &mut members.iter_mut().find(|(k, _)| k == part).unwrap_or_else(|| panic!("{path}: no {part}")).1,
            _ => panic!("{path}: {part} is not inside a container"),
        };
    }
    cur
}

/// Changes the text at `path` with `f`.
fn edit_text(v: &mut Json, path: &str, f: impl FnOnce(&str) -> String) {
    let s = slot(v, path);
    let new = f(s.as_str().unwrap_or_else(|| panic!("{path}: not a string")));
    *s = Json::str(new);
}

/// `text` with the first character of the value of `"member":"` replaced by another one of the base64url alphabet (so it is still base64url, and not the same bytes).
fn flip_member(text: &str, member: &str) -> String {
    let marker = format!("\"{member}\":\"");
    let at = text.find(&marker).unwrap_or_else(|| panic!("no {member} in {text}")) + marker.len();
    let c = text.as_bytes()[at];
    let other = if c == b'A' { 'B' } else { 'A' };
    format!("{}{}{}", &text[..at], other, &text[at + 1..])
}

/// The recorded ceremony, served to the two parties by a scripted relay, up to and including the desktop's opening of the rendezvous.
struct Rig {
    vectors: Json,
    ceremony: Json,
    steps: Vec<Json>,
    clock: Arc<FakeClock>,
    http: Arc<ScriptedHttp>,
    next: Arc<Mutex<usize>>,
    /// Requests that were not the recording's next step (a strict rig panics instead), as `METHOD path`.
    off_script: Arc<Mutex<Vec<String>>>,
    relay: RelayUrl,
    relay_fingerprint: String,
    desktop_client: Arc<RelayClient>,
    token: Token,
    desktop: DesktopPairing,
    offer: NewOffer,
}

/// `strict`: any request that is not the recording's next step, any body that is not the recorded one and any token that is where none is recorded fails at once.
/// `tamper` changes the recorded steps before they are served (an answer of the relay that has been damaged on its way).
fn rig(strict: bool, tamper: impl FnOnce(&mut Vec<Json>)) -> Rig {
    let vectors = load("vectors.json");
    let ceremony = load("fixtures/pairing-ceremony.json");
    let mut steps: Vec<Json> = ceremony.at("steps").as_array().unwrap().to_vec();
    assert_eq!(steps.len(), 6);
    tamper(&mut steps);
    let clock = Arc::new(FakeClock::new(T0));
    // The identity proofs are not in the recording (it is of the pairing routes): an inner stub with the vectors' relay key answers `GET /v1/info`.
    let info_stub = {
        let c = clock.clone();
        StubRelay::with_clocks(
            StubConfig { public_url: "https://relay.example.com".into(), wait_default: 0, wait_max: 0, ..Default::default() },
            move || c.unix_now(),
            || std::time::Duration::ZERO,
        )
    };
    assert_eq!(info_stub.relay_thumbprint(), vectors.s("keys.ed25519Public.relay.thumbprint"));
    let next = Arc::new(Mutex::new(0usize));
    let off_script = Arc::new(Mutex::new(Vec::<String>::new()));
    let script = {
        let (next, steps, clock, off) = (next.clone(), steps.clone(), clock.clone(), off_script.clone());
        ScriptedHttp::new(move |req| {
            if req.url.ends_with("/v1/info") {
                return info_stub.handle(req);
            }
            let mut i = next.lock().unwrap();
            let path = req.url.strip_prefix("https://relay.example.com").unwrap().split('?').next().unwrap().to_string();
            let line = format!("{} {}", req.method.as_str(), path);
            let Some(step) = steps.get(*i) else {
                assert!(!strict, "a request beyond the recording: {line}");
                off.lock().unwrap().push(line);
                return Ok(json_response(404, &[], r#"{"error":{"code":"not_found","message":"Not found."}}"#, clock.unix_now()));
            };
            let name = step.s("step").to_string();
            if req.method.as_str() != step.s("request.method") || path != step.s("request.path") {
                assert!(
                    !strict,
                    "step {}: {name}: the recording has {} {}, the party sent {line}",
                    *i,
                    step.s("request.method"),
                    step.s("request.path")
                );
                off.lock().unwrap().push(line);
                return Ok(json_response(404, &[], r#"{"error":{"code":"not_found","message":"Not found."}}"#, clock.unix_now()));
            }
            if strict {
                let authorised = req.header("authorization").is_some();
                assert_eq!(authorised, step.s("from") == "desktop", "step {}: only the desktop carries a token ({name})", *i);
            }
            if let Some(body) = step.get("request").and_then(|r| r.get("body")) {
                let sent = String::from_utf8(req.body.clone().expect("a body")).unwrap();
                let (sent, recorded) = (json::canonicalize(sent.as_bytes()).unwrap(), json::canonicalize(body.to_compact().as_bytes()).unwrap());
                if strict {
                    assert_eq!(sent, recorded, "step {}: {name}", *i);
                } else if sent != recorded {
                    // Not the recorded request: it is not answered with the recorded answer (a denial is not the approval of the recording).
                    off.lock().unwrap().push(line);
                    return Ok(json_response(404, &[], r#"{"error":{"code":"not_found","message":"Not found."}}"#, clock.unix_now()));
                }
            }
            *i += 1;
            let status = step.n("response.status") as u16;
            Ok(json_response(status, &[], &step.at("response.body").to_compact(), clock.unix_now()))
        })
    };
    let http: Arc<ScriptedHttp> = Arc::new(script);
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let relay_fingerprint = vectors.s("keys.ed25519Public.relay.thumbprint").to_string();

    // The desktop: the identity of the vectors, a token (any well-formed one: the recording does not carry it), and the secret, nonce and jti of Appendix A3.
    let desktop_identity = Arc::new(DesktopIdentity {
        device_id: vectors.s("keys.ids.desktopDevice").into(),
        name: "Front desk PC".into(),
        endpoint: seed(&vectors, "keys.ed25519Seeds.desktopEndpoint"),
        endpoint_x25519: X25519Public::from_b64u(vectors.s("keys.x25519Public.plugin")).unwrap(),
        host_ed25519: seed(&vectors, "keys.ed25519Seeds.host").verify_key(),
        host_x25519: X25519Public::from_b64u(vectors.s("keys.x25519Public.host")).unwrap(),
    });
    let desktop_client = Arc::new(RelayClient::new(
        relay.clone(),
        Some(relay_fingerprint.clone()),
        http.clone(),
        clock.clone(),
        Box::new(SeededRng::new(1)),
        ClientConfig::default(),
    ));
    let token = Token::parse(vectors.s("A2.expected.token")).unwrap();
    desktop_client.prove(&Cancel::new()).unwrap();
    let mut desktop = DesktopPairing::new(desktop_identity.clone(), "aokie", relay.clone(), &relay_fingerprint);
    let offer = desktop
        .create_offer_with(
            unhex(vectors.s("A3.inputs.secretHex")).try_into().unwrap(),
            unhex32(vectors.s("A3.inputs.nonceHex")),
            vectors.s("A3.inputs.offer.jti").into(),
            T0,
        )
        .unwrap();
    assert_eq!(offer.typed_code, vectors.s("A3.expected.typedCode"));
    assert_eq!(offer.pairing_uri, vectors.s("A3.expected.pairingUri"));
    // Step 0: the desktop opens the rendezvous. The request is the recorded one, and the answer the recorded one.
    let created = desktop.open(&desktop_client, &token, &offer, &Cancel::new()).unwrap();
    assert_eq!((created.pid.as_str(), created.exp), (ceremony.s("pid"), 1_790_000_600));
    Rig { vectors, ceremony, steps, clock, http, next, off_script, relay, relay_fingerprint, desktop_client, token, desktop, offer }
}

impl Rig {
    fn used(&self) -> usize {
        *self.next.lock().unwrap()
    }

    fn off_script(&self) -> Vec<String> {
        self.off_script.lock().unwrap().clone()
    }

    fn requests(&self) -> Vec<Logged> {
        self.http.log()
    }

    /// The requests that carried this text in their body, as `METHOD path`.
    fn sent_with(&self, text: &str) -> Vec<String> {
        self.requests()
            .iter()
            .filter(|l| l.body.as_deref().is_some_and(|b| b.contains(text)))
            .map(|l| format!("{} {}", l.method, l.url.strip_prefix("https://relay.example.com").unwrap_or(&l.url)))
            .collect()
    }

    fn client(&self, seed: u64, pin: Option<String>) -> Arc<RelayClient> {
        Arc::new(RelayClient::new(
            self.relay.clone(),
            pin,
            self.http.clone(),
            self.clock.clone(),
            Box::new(SeededRng::new(seed)),
            ClientConfig::default(),
        ))
    }

    fn identity(&self) -> PhoneIdentity {
        PhoneIdentity {
            endpoint: seed(&self.vectors, "keys.ed25519Seeds.phone"),
            x25519: X25519Secret::from_secret(&Secret::new(unhex32(self.vectors.s("keys.x25519Secrets.phone")))),
            device_id: self.vectors.s("keys.ids.phoneDevice").into(),
            display_name: Some("Test phone".into()),
        }
    }

    /// The phone, from the key the desktop shows.
    fn phone(&self) -> PhonePairing {
        let target = PairingTarget::from_input(PairingInput::Key(&self.offer.pairing_uri)).unwrap();
        assert_eq!(target.fingerprint.as_deref(), Some(self.relay_fingerprint.as_str()));
        PhonePairing::new(self.client(2, None), target, self.identity()).unwrap()
    }

    /// The phone, from the typed code and the relay's host: no pairing key, so no key expiry (`x`) beside the offer's own window.
    fn phone_typed(&self) -> PhonePairing {
        let target = PairingTarget::from_input(PairingInput::Typed { code: &self.offer.typed_code, host: "relay.example.com" }).unwrap();
        PhonePairing::new(self.client(4, None), target, self.identity()).unwrap()
    }

    /// The phone fetches the offer and answers it at T0 + 30, as the recording does (steps 1 and 2).
    fn phone_answers(&self) -> (PhonePairing, oaiy_relay_core::pairing::math::Sas) {
        let mut phone = self.phone();
        phone.fetch_offer(&Cancel::new()).unwrap();
        self.clock.advance(std::time::Duration::from_secs(30));
        let sas = phone.respond(&Cancel::new()).unwrap();
        (phone, sas)
    }

    /// The desktop's poll returns the `pair` item (step 3).
    fn the_pair_item(&self) -> Item {
        let reply = self.desktop_client.poll(&self.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
        let adoption = oaiy_relay_core::poll::assess(reply.status, reply.body.as_ref(), 0).expect("an accepted item");
        Item::from_json(&reply.body.as_ref().unwrap().get("items").unwrap().as_array().unwrap()[adoption.accepted[0]]).unwrap()
    }

    /// The party verifies the response: the event, which is `AwaitingSas` when every check held.
    fn desktop_receives(&mut self, item: &Item) -> PairEvent {
        self.desktop.on_pair_item(&self.desktop_client, &self.token, item, &Cancel::new()).unwrap()
    }

    fn grants(&self) -> Vec<String> {
        self.steps[4].at("request.body.grants").as_array().unwrap().iter().map(|g| g.as_str().unwrap().to_string()).collect()
    }
}

#[test]
fn the_recorded_ceremony_is_reproduced_request_for_request_and_ends_with_the_recorded_sas_and_token() {
    let mut r = rig(true, |_| {});
    // Steps 1 and 2: the phone, from the key the desktop shows, proves the relay, fetches the offer, answers it at T0 + 30.
    let mut phone = r.phone();
    let summary = phone.fetch_offer(&Cancel::new()).unwrap();
    assert_eq!((summary.desktop_name.as_str(), summary.relay_host.as_str()), ("Front desk PC", "relay.example.com"));
    r.clock.advance(std::time::Duration::from_secs(30));
    let sas = phone.respond(&Cancel::new()).unwrap();
    assert_eq!(sas.display(), r.ceremony.s("sas"), "the phone shows the recorded SAS");

    // Step 3: the desktop's poll returns the `pair` item; the party verifies the response.
    let item = r.the_pair_item();
    assert_eq!((item.lane.as_str(), item.from.as_str(), item.id.as_str()), ("pair", "relay", r.ceremony.s("pid")));
    let event = r.desktop_receives(&item);
    assert_eq!(
        event,
        PairEvent::AwaitingSas {
            pid: r.ceremony.s("pid").into(),
            phone_name: Some("Test phone".into()),
            phone_thumbprint: r.vectors.s("keys.ed25519Public.phone.thumbprint").into()
        }
    );
    assert_eq!(r.desktop.expected_sas(&r.offer.pid).unwrap().display(), r.ceremony.s("sas"));

    // Step 4: the owner types the 13 characters at T0 + 40 and the desktop decides, with the recorded grants in the recorded order: the body is the recorded one, byte for byte.
    r.clock.advance(std::time::Duration::from_secs(10));
    let grants = r.grants();
    let pid = r.offer.pid.clone();
    let outcome = r.desktop.confirm_sas(&r.desktop_client, &r.token, &pid, &r.ceremony.s("sas").to_lowercase(), &grants, &Cancel::new()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { ref device_id } if device_id == r.steps[4].s("response.body.deviceId")), "{outcome:?}");
    let decision = r.requests().into_iter().find(|l| l.url.ends_with("/decision")).and_then(|l| l.body).expect("the decision");
    assert_eq!(decision, r.steps[4].at("request.body").to_compact(), "the decision, the receipt included, is the recorded bytes");

    // Step 5: the phone reads the outcome. The recording's receipt has no grants (the shipped relay does not return them), so the phone is given the ones the desktop approved.
    let paired = match phone.wait_outcome(Some(&grants), &Cancel::new()).unwrap() {
        Outcome::Paired(p) => p,
        _ => panic!("not paired"),
    };
    let sealed_tokens = load("fixtures/sealed-token.json");
    assert_eq!(hex(&sha256(paired.token.expose().as_bytes())), sealed_tokens.s("opens.0.plaintextSha256"), "the token of the recorded ceremony");
    assert_eq!(paired.profile.device_id, r.steps[5].s("response.body.deviceId"));
    assert_eq!(paired.profile.relay_id, r.vectors.s("keys.ids.relay"));
    assert_eq!(paired.profile.peer.as_ref().unwrap().desktop_endpoint.to_b64u(), r.vectors.s("keys.ed25519Public.desktopEndpoint.publicKey"));
    assert_eq!(r.used(), 6, "every recorded step was used, and no more");
    assert!(r.off_script().is_empty());
}

// ---- the same recording with one artifact damaged

#[test]
fn an_offer_whose_mac_is_wrong_is_refused_before_anything_in_it_is_read_and_nothing_is_posted() {
    let r = rig(false, |s| {
        edit_text(s.get_mut(1).unwrap(), "response.body.mac", |m| format!("{}{}", if m.starts_with('A') { 'B' } else { 'A' }, &m[1..]))
    });
    let mut phone = r.phone();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::BadMac(_))), "{err:?}");
    assert_eq!(r.used(), 2, "the rendezvous was opened and the offer fetched; the phone went no further");
    assert!(r.sent_with("response").iter().all(|l| !l.ends_with("/response")), "no response was posted");
    assert!(matches!(phone.respond(&Cancel::new()), Err(PairingError::WrongState(_))), "a phone with no verified offer cannot answer");
}

#[test]
fn an_offer_whose_text_was_changed_under_its_mac_is_refused() {
    let r = rig(false, |s| edit_text(s.get_mut(1).unwrap(), "response.body.offer", |t| t.replace("Front desk PC", "Front desk PD")));
    let mut phone = r.phone();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::BadMac(_))), "{err:?}");
    assert_eq!(r.used(), 2);
}

#[test]
fn an_offer_is_judged_against_its_own_window_at_relay_time_with_thirty_seconds_of_slack() {
    // The offer expires at T0 + 600. At T0 + 629 it is still taken (29 seconds of slack used) and at T0 + 630 it is not.
    let r = rig(false, |_| {});
    r.clock.advance(std::time::Duration::from_secs(629));
    let mut phone = r.phone();
    phone.fetch_offer(&Cancel::new()).expect("29 seconds after the offer expired is inside the slack");

    let r = rig(false, |_| {});
    r.clock.advance(std::time::Duration::from_secs(630));
    let mut phone = r.phone();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::OutsideWindow(_))), "30 seconds after it expired the offer is no offer: {err:?}");
    assert_eq!(r.used(), 2);
    assert!(r.sent_with("response").iter().all(|l| !l.ends_with("/response")), "nothing was answered");
}

#[test]
fn an_offer_is_judged_against_its_own_window_also_when_there_is_no_key_expiry_beside_it() {
    // The scanned key carries the offer's expiry (`x`), which refuses a late phone by itself; a typed code has none, so here the offer's own window is the only judge.
    let r = rig(false, |_| {});
    r.clock.advance(std::time::Duration::from_secs(629));
    let mut phone = r.phone_typed();
    phone.fetch_offer(&Cancel::new()).expect("29 seconds past the expiry is inside the slack");
    let r = rig(false, |_| {});
    r.clock.advance(std::time::Duration::from_secs(630));
    let mut phone = r.phone_typed();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::OutsideWindow(_))), "{err:?}");
    assert!(r.sent_with("response").iter().all(|l| !l.ends_with("/response")), "nothing was answered");
}

#[test]
fn an_offer_for_another_relay_than_the_one_the_owner_named_is_refused_though_its_mac_and_key_are_right() {
    // The offer is made by the desktop's own party for `https://other.example.com` (the same relay key and the same secret, so a MAC that verifies); the key the phone scanned names
    // `https://relay.example.com`. The phone must see that the offer is for another relay, and answer nothing.
    let r = rig(false, |_| {});
    let other_relay = RelayUrl::parse("https://other.example.com").unwrap();
    let mut other_party = DesktopPairing::new(
        Arc::new(DesktopIdentity {
            device_id: r.vectors.s("keys.ids.desktopDevice").into(),
            name: "Front desk PC".into(),
            endpoint: seed(&r.vectors, "keys.ed25519Seeds.desktopEndpoint"),
            endpoint_x25519: X25519Public::from_b64u(r.vectors.s("keys.x25519Public.plugin")).unwrap(),
            host_ed25519: seed(&r.vectors, "keys.ed25519Seeds.host").verify_key(),
            host_x25519: X25519Public::from_b64u(r.vectors.s("keys.x25519Public.host")).unwrap(),
        }),
        "aokie",
        other_relay,
        &r.relay_fingerprint,
    );
    let offer = other_party
        .create_offer_with(
            unhex(r.vectors.s("A3.inputs.secretHex")).try_into().unwrap(),
            unhex32(r.vectors.s("A3.inputs.nonceHex")),
            r.vectors.s("A3.inputs.offer.jti").into(),
            T0,
        )
        .unwrap();
    assert_eq!(offer.pid, r.offer.pid, "the same secret: the same rendezvous");
    let served = {
        let mut steps = r.steps.clone();
        edit_text(&mut steps[1], "response.body.offer", |_| offer.offer.text.clone());
        edit_text(&mut steps[1], "response.body.mac", |_| offer.mac.clone());
        steps
    };
    let r = rig(false, |s| *s = served);
    let mut phone = r.phone();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::Mismatch(_))), "{err:?}");
    assert!(r.sent_with("response").iter().all(|l| !l.ends_with("/response")));
}

#[test]
fn a_key_that_names_another_relay_key_than_the_offer_does_is_refused_before_the_offer_is_used() {
    // The key's `f` (what the phone proves the relay against) is the right relay's; the offer, which carries its own MAC, was made by a desktop that was told another fingerprint.
    let r = rig(false, |_| {});
    let other = "SWUejy55xcz8xgSi-GX15ERZzyuXnb6voSaopq2Jakw";
    let mut other_party = DesktopPairing::new(
        Arc::new(DesktopIdentity {
            device_id: r.vectors.s("keys.ids.desktopDevice").into(),
            name: "Front desk PC".into(),
            endpoint: seed(&r.vectors, "keys.ed25519Seeds.desktopEndpoint"),
            endpoint_x25519: X25519Public::from_b64u(r.vectors.s("keys.x25519Public.plugin")).unwrap(),
            host_ed25519: seed(&r.vectors, "keys.ed25519Seeds.host").verify_key(),
            host_x25519: X25519Public::from_b64u(r.vectors.s("keys.x25519Public.host")).unwrap(),
        }),
        "aokie",
        r.relay.clone(),
        other,
    );
    let offer = other_party
        .create_offer_with(
            unhex(r.vectors.s("A3.inputs.secretHex")).try_into().unwrap(),
            unhex32(r.vectors.s("A3.inputs.nonceHex")),
            r.vectors.s("A3.inputs.offer.jti").into(),
            T0,
        )
        .unwrap();
    // Served in place of the recorded offer (the rendezvous of the recording is the same pid: the secret is the same).
    let served = {
        let mut steps = r.steps.clone();
        edit_text(&mut steps[1], "response.body.offer", |_| offer.offer.text.clone());
        edit_text(&mut steps[1], "response.body.mac", |_| offer.mac.clone());
        steps
    };
    let r = rig(false, |s| *s = served);
    let uri_with_the_right_key = offer.pairing_uri.replace(other, &r.relay_fingerprint);
    assert_ne!(uri_with_the_right_key, offer.pairing_uri);
    let target = PairingTarget::from_input(PairingInput::Key(&uri_with_the_right_key)).unwrap();
    let mut phone = PhonePairing::new(r.client(3, None), target, r.identity()).unwrap();
    let err = phone.fetch_offer(&Cancel::new()).unwrap_err();
    assert!(matches!(err, PairingError::Protocol(Error::Mismatch(_))), "{err:?}");
    assert!(r.sent_with("response").iter().all(|l| !l.ends_with("/response")));
}

#[test]
fn a_response_with_a_flipped_signature_is_rejected_by_the_desktop_and_the_relay_is_told() {
    let mut r = rig(false, |s| edit_text(s.get_mut(3).unwrap(), "response.body.items.0.body", |b| flip_member(b, "signature")));
    let _phone = r.phone_answers();
    let item = r.the_pair_item();
    let event = r.desktop_receives(&item);
    assert_eq!(event, PairEvent::Rejected { pid: r.offer.pid.clone(), reason: "signature" });
    assert!(r.desktop.expected_sas(&r.offer.pid).is_none(), "no code is asked for");
    assert_eq!(
        r.off_script(),
        vec![format!("POST /v1/pair/{}/reject", r.offer.pid)],
        "the desktop told the relay to take the response back, and sent nothing else"
    );
}

#[test]
fn a_response_with_a_flipped_mac_is_rejected_by_the_desktop_and_the_relay_is_told() {
    let mut r = rig(false, |s| edit_text(s.get_mut(3).unwrap(), "response.body.items.0.body", |b| flip_member(b, "mac")));
    let _phone = r.phone_answers();
    let item = r.the_pair_item();
    assert_eq!(r.desktop_receives(&item), PairEvent::Rejected { pid: r.offer.pid.clone(), reason: "mac" });
    assert!(r.desktop.expected_sas(&r.offer.pid).is_none());
    assert_eq!(r.off_script(), vec![format!("POST /v1/pair/{}/reject", r.offer.pid)]);
}

#[test]
fn a_response_outside_its_own_window_is_rejected_by_the_desktop() {
    // The claims were issued at T0 + 30 and live 120 seconds, with 30 of slack: the desktop that reads them at T0 + 181 refuses them, at T0 + 179 it takes them.
    let mut r = rig(false, |_| {});
    let _phone = r.phone_answers();
    let item = r.the_pair_item();
    r.clock.advance(std::time::Duration::from_secs(149));
    assert!(matches!(r.desktop_receives(&item), PairEvent::AwaitingSas { .. }), "T0 + 179");
    let mut r = rig(false, |_| {});
    let _phone = r.phone_answers();
    let item = r.the_pair_item();
    r.clock.advance(std::time::Duration::from_secs(151));
    assert_eq!(r.desktop_receives(&item), PairEvent::Rejected { pid: r.offer.pid.clone(), reason: "window" }, "T0 + 181");
}

#[test]
fn a_wrong_code_approves_nothing_and_the_third_denies_and_burns() {
    let mut r = rig(false, |_| {});
    let (_phone, sas) = r.phone_answers();
    let item = r.the_pair_item();
    assert!(matches!(r.desktop_receives(&item), PairEvent::AwaitingSas { .. }));
    let right = sas.display().to_lowercase();
    // A code that is well formed (the check character is right) and wrong.
    let wrong = {
        let other = oaiy_relay_core::pairing::math::sas(&[1u8; 32], &[2u8; 32], &[3u8; 32], &[4u8; 16]).unwrap();
        other.display().to_lowercase()
    };
    assert_ne!(wrong, right);
    let grants = r.grants();
    let pid = r.offer.pid.clone();
    // No approval can be made for a pairing whose code has not been typed, by any call.
    assert_eq!(r.desktop.approve_body(&pid, &grants, T0 + 40).unwrap_err(), PairingError::SasRequired);
    assert!(r.sent_with("approve").is_empty());
    let confirm = |r: &mut Rig, typed: &str| r.desktop.confirm_sas(&r.desktop_client, &r.token, &pid, typed, &grants, &Cancel::new()).unwrap();
    assert_eq!(confirm(&mut r, &wrong), SasOutcome::Wrong { attempts_left: 2 });
    assert_eq!(confirm(&mut r, &wrong), SasOutcome::Wrong { attempts_left: 1 });
    assert!(r.sent_with("approve").is_empty(), "no decision of any kind was sent for two wrong codes");
    assert_eq!(confirm(&mut r, &wrong), SasOutcome::Denied);
    assert_eq!(
        r.off_script(),
        vec![format!("POST /v1/pair/{pid}/decision"), format!("POST /v1/pair/{pid}/burn")],
        "the third wrong code posts the denial and then burns the rendezvous"
    );
    assert_eq!(r.sent_with("\"approve\":false"), vec![format!("POST /v1/pair/{pid}/decision")]);
    assert!(r.sent_with("\"approve\":true").is_empty(), "and nothing was ever approved");
    // The pairing is gone: not even the right code approves it now.
    assert!(matches!(
        r.desktop.confirm_sas(&r.desktop_client, &r.token, &pid, &right, &grants, &Cancel::new()),
        Err(PairingError::UnknownPairing | PairingError::WrongState(_))
    ));
    assert!(r.sent_with("\"approve\":true").is_empty());
}

/// The ceremony up to the phone's reading of the outcome (step 5), with `tamper` applied to the steps; returns the phone's answer.
fn phone_reads_the_outcome(tamper: impl FnOnce(&mut Vec<Json>)) -> (Rig, Result<Outcome, PairingError>) {
    let mut r = rig(false, tamper);
    let (mut phone, sas) = r.phone_answers();
    let item = r.the_pair_item();
    assert!(matches!(r.desktop_receives(&item), PairEvent::AwaitingSas { .. }));
    r.clock.advance(std::time::Duration::from_secs(10));
    let (grants, pid) = (r.grants(), r.offer.pid.clone());
    let outcome = r.desktop.confirm_sas(&r.desktop_client, &r.token, &pid, &sas.display().to_lowercase(), &grants, &Cancel::new()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { .. }));
    let result = phone.wait_outcome(Some(&grants), &Cancel::new());
    (r, result)
}

#[test]
fn the_receipt_the_recording_carries_is_the_one_that_pairs() {
    let (_r, result) = phone_reads_the_outcome(|_| {});
    assert!(matches!(result, Ok(Outcome::Paired(_))), "the untouched recording pairs: {:?}", result.err());
}

#[test]
fn a_receipt_with_a_flipped_signature_pairs_nothing() {
    let (_r, result) = phone_reads_the_outcome(|s| {
        edit_text(s.get_mut(5).unwrap(), "response.body.receipt.signature", |m| format!("{}{}", if m.starts_with('A') { 'B' } else { 'A' }, &m[1..]))
    });
    assert!(matches!(result, Err(PairingError::ReceiptInvalid)), "{:?}", result.map(|_| "paired"));
}

#[test]
fn a_receipt_with_another_date_than_the_one_signed_pairs_nothing() {
    // The date is inside the window (T0 + 41 for a receipt signed at T0 + 40), so only the signature can tell.
    let (_r, result) = phone_reads_the_outcome(|s| *slot(s.get_mut(5).unwrap(), "response.body.receipt.issuedAt") = Json::int(1_790_000_041i64));
    assert!(matches!(result, Err(PairingError::ReceiptInvalid)), "{:?}", result.map(|_| "paired"));
}

#[test]
fn a_receipt_that_covers_other_grants_than_the_phone_was_told_pairs_nothing() {
    // The shipped relay returns no grants with the receipt; one that returns a list (the signature protects it) is believed over the phone's own, and a list that is not the one
    // the desktop signed fails the receipt.
    let (_r, result) = phone_reads_the_outcome(|s| {
        if let Json::Obj(m) = slot(s.get_mut(5).unwrap(), "response.body.receipt") {
            m.retain(|(k, _)| k != "grants");
            m.push(("grants".to_string(), Json::Arr(vec![Json::str("state_read")])));
        }
    });
    assert!(matches!(result, Err(PairingError::ReceiptInvalid)), "{:?}", result.map(|_| "paired"));
}

#[test]
fn a_sealed_token_that_does_not_open_is_not_a_pairing_even_when_the_receipt_is_right() {
    let (_r, result) = phone_reads_the_outcome(|s| {
        edit_text(s.get_mut(5).unwrap(), "response.body.sealedToken", |t| {
            let mut b = t.to_string().into_bytes();
            b[40] = if b[40] == b'A' { b'B' } else { b'A' };
            String::from_utf8(b).unwrap()
        })
    });
    assert!(matches!(result, Err(PairingError::TokenInvalid)), "{:?}", result.map(|_| "paired"));
}
