//! `fixtures/pairing-ceremony.json` replayed step by step by this crate's two parties: the desktop's and the phone's requests are built from the vectors' keys and values and must
//! be **exactly** the ones the real relay recorded (the offer, its MAC, the response with its claims, signature and MAC, the decision with its receipt), the relay's recorded answers
//! are fed back, and the ceremony ends with the SAS of the recording, a verified receipt and the token of `fixtures/sealed-token.json`.

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
use oaiy_relay_core::pairing::{DesktopIdentity, DesktopPairing, PairEvent, PairingInput, PairingTarget, PhoneIdentity, PhonePairing, SasOutcome};
use oaiy_relay_core::testing::stub::{StubConfig, StubRelay};
use oaiy_relay_core::testing::{json_response, FakeClock, ScriptedHttp, SeededRng};
use oaiy_relay_core::url::RelayUrl;

const T0: i64 = 1_790_000_000;

fn seed(v: &Json, path: &str) -> Signer {
    Signer::from_seed(&Secret::new(unhex32(v.s(path))))
}

#[test]
fn the_recorded_ceremony_is_reproduced_request_for_request_and_ends_with_the_recorded_sas_and_token() {
    let vectors = load("vectors.json");
    let ceremony = load("fixtures/pairing-ceremony.json");
    let steps: Vec<Json> = ceremony.at("steps").as_array().unwrap().to_vec();
    assert_eq!(steps.len(), 6);
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
    let seen_requests = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
    let script = {
        let (next, steps, clock, seen) = (next.clone(), steps.clone(), clock.clone(), seen_requests.clone());
        ScriptedHttp::new(move |req| {
            if req.url.ends_with("/v1/info") {
                return info_stub.handle(req);
            }
            let mut i = next.lock().unwrap();
            let step = steps.get(*i).unwrap_or_else(|| panic!("a request beyond the recording: {} {}", req.method.as_str(), req.url));
            *i += 1;
            let path = req.url.strip_prefix("https://relay.example.com").unwrap().split('?').next().unwrap().to_string();
            assert_eq!(req.method.as_str(), step.s("request.method"), "step {}: {}", *i - 1, step.s("step"));
            assert_eq!(path, step.s("request.path"), "step {}: {}", *i - 1, step.s("step"));
            let authorised = req.header("authorization").is_some();
            assert_eq!(authorised, step.s("from") == "desktop", "step {}: only the desktop carries a token ({})", *i - 1, step.s("step"));
            if let Some(body) = step.get("request").and_then(|r| r.get("body")) {
                let sent = String::from_utf8(req.body.clone().expect("a body")).unwrap();
                assert_eq!(
                    json::canonicalize(sent.as_bytes()).unwrap(),
                    json::canonicalize(body.to_compact().as_bytes()).unwrap(),
                    "step {}: {}",
                    *i - 1,
                    step.s("step")
                );
                seen.lock().unwrap().push((step.s("step").to_string(), sent));
            }
            let status = step.n("response.status") as u16;
            Ok(json_response(status, &[], &step.at("response.body").to_compact(), clock.unix_now()))
        })
    };
    let http: Arc<ScriptedHttp> = Arc::new(script);
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let relay_fingerprint = vectors.s("keys.ed25519Public.relay.thumbprint").to_string();
    let make_client = |seed: u64, pin: Option<String>| {
        Arc::new(RelayClient::new(relay.clone(), pin, http.clone(), clock.clone(), Box::new(SeededRng::new(seed)), ClientConfig::default()))
    };

    // The desktop: the identity of the vectors, a token (any well-formed one: the recording does not carry it), and the secret, nonce and jti of Appendix A3.
    let desktop_identity = Arc::new(DesktopIdentity {
        device_id: vectors.s("keys.ids.desktopDevice").into(),
        name: "Front desk PC".into(),
        endpoint: seed(&vectors, "keys.ed25519Seeds.desktopEndpoint"),
        endpoint_x25519: X25519Public::from_b64u(vectors.s("keys.x25519Public.plugin")).unwrap(),
        host_ed25519: seed(&vectors, "keys.ed25519Seeds.host").verify_key(),
        host_x25519: X25519Public::from_b64u(vectors.s("keys.x25519Public.host")).unwrap(),
    });
    let desktop_client = make_client(1, Some(relay_fingerprint.clone()));
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

    // Steps 1 and 2: the phone, from the key the desktop shows, proves the relay, fetches the offer, answers it at T0 + 30.
    let target = PairingTarget::from_input(PairingInput::Key(&offer.pairing_uri)).unwrap();
    assert_eq!(target.fingerprint.as_deref(), Some(relay_fingerprint.as_str()));
    let phone_client = make_client(2, None);
    let identity = PhoneIdentity {
        endpoint: seed(&vectors, "keys.ed25519Seeds.phone"),
        x25519: X25519Secret::from_secret(&Secret::new(unhex32(vectors.s("keys.x25519Secrets.phone")))),
        device_id: vectors.s("keys.ids.phoneDevice").into(),
        display_name: Some("Test phone".into()),
    };
    let mut phone = PhonePairing::new(phone_client.clone(), target, identity).unwrap();
    let summary = phone.fetch_offer(&Cancel::new()).unwrap();
    assert_eq!((summary.desktop_name.as_str(), summary.relay_host.as_str()), ("Front desk PC", "relay.example.com"));
    clock.advance(std::time::Duration::from_secs(30));
    let sas = phone.respond(&Cancel::new()).unwrap();
    assert_eq!(sas.display(), ceremony.s("sas"), "the phone shows the recorded SAS");

    // Step 3: the desktop's poll returns the `pair` item; the party verifies the response.
    let reply = desktop_client.poll(&token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
    let adoption = oaiy_relay_core::poll::assess(reply.status, reply.body.as_ref(), 0).expect("an accepted item");
    let item = Item::from_json(&reply.body.as_ref().unwrap().get("items").unwrap().as_array().unwrap()[adoption.accepted[0]]).unwrap();
    assert_eq!((item.lane.as_str(), item.from.as_str(), item.id.as_str()), ("pair", "relay", ceremony.s("pid")));
    let event = desktop.on_pair_item(&desktop_client, &token, &item, &Cancel::new()).unwrap();
    assert_eq!(
        event,
        PairEvent::AwaitingSas {
            pid: ceremony.s("pid").into(),
            phone_name: Some("Test phone".into()),
            phone_thumbprint: vectors.s("keys.ed25519Public.phone.thumbprint").into()
        }
    );
    assert_eq!(desktop.expected_sas(&offer.pid).unwrap().display(), ceremony.s("sas"));

    // Step 4: the owner types the 13 characters at T0 + 40 and the desktop decides, with the recorded grants in the recorded order: the body is the recorded one, byte for byte.
    clock.advance(std::time::Duration::from_secs(10));
    let grants: Vec<String> = steps[4].at("request.body.grants").as_array().unwrap().iter().map(|g| g.as_str().unwrap().to_string()).collect();
    let outcome = desktop.confirm_sas(&desktop_client, &token, &offer.pid, &ceremony.s("sas").to_lowercase(), &grants, &Cancel::new()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { ref device_id } if device_id == steps[4].s("response.body.deviceId")), "{outcome:?}");
    let decision = seen_requests.lock().unwrap().iter().find(|(s, _)| s == "desktop approves").unwrap().1.clone();
    assert_eq!(decision, steps[4].at("request.body").to_compact(), "the decision, the receipt included, is the recorded bytes");

    // Step 5: the phone reads the outcome. The recording's receipt has no grants (the shipped relay does not return them), so the phone is given the ones the desktop approved.
    let paired = match phone.wait_outcome(Some(&grants), &Cancel::new()).unwrap() {
        Outcome::Paired(p) => p,
        _ => panic!("not paired"),
    };
    let sealed_tokens = load("fixtures/sealed-token.json");
    assert_eq!(hex(&sha256(paired.token.expose().as_bytes())), sealed_tokens.s("opens.0.plaintextSha256"), "the token of the recorded ceremony");
    assert_eq!(paired.profile.device_id, steps[5].s("response.body.deviceId"));
    assert_eq!(paired.profile.relay_id, vectors.s("keys.ids.relay"));
    assert_eq!(paired.profile.peer.as_ref().unwrap().desktop_endpoint.to_b64u(), vectors.s("keys.ed25519Public.desktopEndpoint.publicKey"));
    assert_eq!(*next.lock().unwrap(), 6, "every recorded step was used, and no more");
}
