//! Tests of the hardening round that followed the independent review: each one pins a behaviour that was missing, wrong or only claimed, and is written so that the mutant of the
//! fix (see `mutation/`) makes it fail. They run against the in-process stub relay with the fake clock, so a loop of minutes costs no time.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::env::{env, quick, run_loop};
use common::pair::world;
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::b64;
use oaiy_relay_core::client::*;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::desktop::{MAX_CONSUMED, MAX_PENDING};
use oaiy_relay_core::pairing::math::{self, PairingSecret};
use oaiy_relay_core::pairing::{PairEvent, PairingError};
use oaiy_relay_core::poll::Outcome;
use oaiy_relay_core::testing::stub::{Fault, StubConfig};
use oaiy_relay_core::testing::SeededRng;
use oaiy_relay_core::url::RelayUrl;

fn secs(d: &Duration) -> f64 {
    d.as_secs_f64()
}

fn within_jitter(d: &Duration, base: f64) -> bool {
    (base..=base * 1.2 + 1e-9).contains(&secs(d))
}

#[test]
fn a_429_for_the_identity_proof_is_flow_and_never_unreachable() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.clock.clear_sleeps();
    // The relay answers the next three proof requests with `429` and `Retry-After: 2`: it answered, so the loop paces them as flow (P5: max(clamp(2), min(30, 2^(n-1))) = 2, 2, 4) and
    // reports nothing, where a failure would be paced 1, 2, 4 and report `unreachable` at the third.
    let body = r#"{"error":{"code":"rate_limited","message":"Too many requests.","retryAfter":2}}"#;
    e.stub.fail_next_on("/v1/info", 3, Fault::Respond(429, vec![("Retry-After".into(), "2".into())], body.into()));
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("three flow answers", |ev| ev.iter().filter(|x| matches!(x, Event::Answer { outcome: Outcome::Flow, .. })).count() >= 3);
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    let sink = running.sink.clone();
    let (_, _) = running.finish();
    assert!(sink.reports().is_empty(), "{:?}", sink.reports());
    assert!(!sink.states().contains(&ConnectionState::Unreachable));
    let sleeps: Vec<Duration> = e.clock.sleeps().into_iter().filter(|s| secs(s) >= 1.0).collect();
    for (i, base) in [2.0, 2.0, 4.0].iter().enumerate() {
        assert!(within_jitter(&sleeps[i], *base), "pause {i} is {base} s: {:?}", sleeps[i]);
    }
}

#[test]
fn a_retry_after_is_honoured_in_whatever_case_the_adapter_returns_it() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.clock.clear_sleeps();
    e.stub.fail_next_on("/v1/poll", 1, Fault::Respond(429, vec![("RETRY-AFTER".into(), "40".into())], "{}".into()));
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("the flow answer", |ev| ev.iter().any(|x| matches!(x, Event::Answer { outcome: Outcome::Flow, .. })));
    let _ = running.finish();
    assert!(e.clock.sleeps().iter().any(|s| within_jitter(s, 40.0)), "{:?}", e.clock.sleeps());
}

#[test]
fn the_headers_of_an_answer_reach_the_host_in_lower_case_whatever_the_adapter_returned() {
    // An OkHttp adapter returns the names as the server wrote them: the client normalises them as the response comes in, so that what it hands on (`PollReply::headers`, which the
    // decision core reads) is in lower case and a consumer never has to know what the adapter did.
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let stub = e.stub.clone();
    let adapter = oaiy_relay_core::testing::ScriptedHttp::new(move |req| {
        if req.url.contains("/v1/poll") {
            return Ok(HttpResponse {
                status: 429,
                headers: vec![
                    ("Retry-After".into(), "7".into()),
                    ("X-OAIY-Time".into(), "1790000000".into()),
                    ("CONTENT-TYPE".into(), "application/json".into()),
                ],
                body: b"{}".to_vec(),
            });
        }
        stub.handle(req)
    });
    let client = RelayClient::new(
        e.client.url().clone(),
        Some(e.stub.relay_thumbprint()),
        Arc::new(adapter),
        e.clock.clone(),
        Box::new(SeededRng::new(77)),
        ClientConfig::default(),
    );
    client.prove(&Cancel::new()).unwrap();
    let reply = client.poll(&token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
    assert_eq!(reply.status, Some(429));
    let names: Vec<&str> = reply.headers.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(names, ["retry-after", "x-oaiy-time", "content-type"], "lower case, in the order they came");
    assert!(reply.headers.iter().any(|(k, v)| k == "retry-after" && v == "7"));
}

/// An adapter whose reads of a rendezvous with a wait take 12 seconds of the phone's clock and answer `answered` with a hold that was granted (and, if asked, superseded), so that the
/// phone's pacing can be told apart by what the relay says and not only by how long the request took. Everything else goes to the stub.
struct HeldPairReads {
    stub: oaiy_relay_core::testing::stub::StubRelay,
    clock: Arc<oaiy_relay_core::testing::FakeClock>,
    answered: oaiy_relay_core::pairing::NewOffer,
    superseded: bool,
    reads: std::sync::atomic::AtomicUsize,
}

impl HttpClient for HeldPairReads {
    fn send(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError> {
        if req.method.as_str() == "GET" && req.url.contains("/v1/pair/") && req.url.contains("wait=") {
            self.clock.advance(Duration::from_secs(12));
            let n = self.reads.fetch_add(1, Ordering::SeqCst);
            let now = self.clock.unix_now();
            let body = if n < 3 {
                let mut hold = vec![("granted", oaiy_relay_core::json::Json::Bool(true))];
                if self.superseded {
                    hold.push(("superseded", oaiy_relay_core::json::Json::Bool(true)));
                }
                oaiy_relay_core::json::Json::obj([
                    ("v", oaiy_relay_core::json::Json::int(1)),
                    ("state", oaiy_relay_core::json::Json::str("answered")),
                    ("offer", oaiy_relay_core::json::Json::str(self.answered.offer.text.clone())),
                    ("mac", oaiy_relay_core::json::Json::str(self.answered.mac.clone())),
                    ("exp", oaiy_relay_core::json::Json::int(self.answered.offer.expires_at)),
                    ("time", oaiy_relay_core::json::Json::int(now)),
                    ("hold", oaiy_relay_core::json::Json::Obj(hold.into_iter().map(|(k, v)| (k.to_string(), v)).collect())),
                ])
                .to_compact()
            } else {
                format!("{{\"v\":1,\"state\":\"denied\",\"time\":{now}}}")
            };
            return Ok(oaiy_relay_core::testing::json_response(200, &[], &body, now));
        }
        self.stub.handle(req)
    }
}

#[test]
fn a_read_the_relay_says_it_held_is_believed_only_when_it_was_not_superseded_and_took_a_hold_s_time() {
    // The phone asks again at once only for a read that the relay says it held (`hold.granted`), that was not `superseded` (another read of the rendezvous took its place: it did not
    // wait for the owner) and that took as long as a hold takes by the phone's own clock (here 12 s of the wait asked for). A superseded read is paced at 10 s like any other answer
    // that leaves the rendezvous as it was. And (README 10.1) when the relay's `wait.max` is below ten seconds a granted hold is followed by the difference to ten: a phone that asked
    // again at once after a two-second hold would spend the 60 counted reads of a rendezvous in two minutes.
    for (wait_max, superseded, pause) in
        [(20u64, false, None), (20, true, Some(10.0)), (2, false, Some(8.0)), (9, false, Some(1.0)), (10, false, None), (2, true, Some(10.0))]
    {
        let mut w = world(StubConfig { wait_default: wait_max, wait_max, receipt_includes_grants: true, ..Default::default() });
        let offer = w.new_offer();
        let target = oaiy_relay_core::pairing::PairingTarget::from_input(oaiy_relay_core::pairing::PairingInput::Key(&offer.pairing_uri)).unwrap();
        let adapter = Arc::new(HeldPairReads {
            stub: w.env.stub.clone(),
            clock: w.env.clock.clone(),
            answered: offer,
            superseded,
            reads: std::sync::atomic::AtomicUsize::new(0),
        });
        let client = Arc::new(RelayClient::new(
            target.relay.clone(),
            None,
            adapter,
            w.env.clock.clone(),
            Box::new(SeededRng::new(5)),
            ClientConfig::default(),
        ));
        let mut phone = oaiy_relay_core::pairing::PhonePairing::new(client, target, common::pair::phone_identity(5)).unwrap();
        phone.fetch_offer(&Cancel::new()).unwrap();
        phone.respond(&Cancel::new()).unwrap();
        w.env.clock.clear_sleeps();
        assert!(matches!(phone.wait_outcome(None, &Cancel::new()).unwrap(), oaiy_relay_core::pairing::phone::Outcome::Denied));
        let sleeps = w.env.clock.sleeps();
        match pause {
            None => assert!(sleeps.is_empty(), "wait.max {wait_max}, superseded {superseded}: {sleeps:?}"),
            Some(base) => {
                assert_eq!(sleeps.len(), 3, "wait.max {wait_max}, superseded {superseded}: {sleeps:?}");
                for s in &sleeps {
                    assert!(within_jitter(s, base), "wait.max {wait_max}, superseded {superseded}: a pause of {base} s with jitter, got {s:?}");
                }
            }
        }
    }
}

#[test]
fn after_a_429_of_the_pairing_reads_the_phone_waits_the_larger_of_ten_seconds_and_the_retry_after() {
    // README 10.1: max(10, Retry-After) with the jitter of P6. A relay that says 3 seconds is waited ten; one that says 40 is waited 40 (up to 20 percent more).
    for (retry_after, base) in [("3", 10.0), ("40", 40.0), ("10", 10.0), ("11", 11.0)] {
        let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
        let offer = w.new_offer();
        let mut phone = w.phone(oaiy_relay_core::pairing::PairingInput::Key(&offer.pairing_uri), 9);
        phone.fetch_offer(&Cancel::new()).unwrap();
        phone.respond(&Cancel::new()).unwrap();
        w.env.clock.clear_sleeps();
        let body = format!("{{\"error\":{{\"code\":\"rate_limited\",\"message\":\"Too many reads.\",\"retryAfter\":{retry_after}}}}}");
        w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(429, vec![("Retry-After".into(), retry_after.into())], body));
        w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], "{\"v\":1,\"state\":\"denied\",\"time\":1}".into()));
        assert!(matches!(phone.wait_outcome(None, &Cancel::new()).unwrap(), oaiy_relay_core::pairing::phone::Outcome::Denied));
        let sleeps = w.env.clock.sleeps();
        assert_eq!(sleeps.len(), 1, "Retry-After {retry_after}: {sleeps:?}");
        assert!(within_jitter(&sleeps[0], base), "Retry-After {retry_after}: {base} s with jitter, got {:?}", sleeps[0]);
    }
}
/// Every spelling a secret can be printed in: the raw bytes as hex, as decimal numbers, as base64url, and as the 6-bit digits are not printed either.
fn spellings(bytes: &[u8]) -> Vec<String> {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let dec = bytes.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ");
    vec![hex.clone(), hex.to_uppercase(), dec, b64::encode(bytes), format!("{bytes:?}")]
}

#[test]
fn no_secret_holding_type_prints_its_secret_in_any_spelling() {
    let seed = [0x5au8; 32];
    let signer = Signer::from_seed(&Secret::new(seed));
    let x_secret = [0x5bu8; 32];
    let x = X25519Secret::from_secret(&Secret::new(x_secret));
    let pairing = [0x77u8; 16];
    let ps = PairingSecret::new(pairing);
    let typed = ps.typed_code();
    let sas = math::Sas::from_parts([0xC1; 8], "6NHNK68MQQVZ", '5');
    let shown: Vec<(&str, String, Vec<String>)> = vec![
        ("Signer", format!("{signer:?} {signer:#?}"), spellings(&seed)),
        ("X25519Secret", format!("{x:?} {x:#?}"), spellings(&x_secret)),
        ("PairingSecret", format!("{ps:?} {ps:#?}"), spellings(&pairing).into_iter().chain([typed.clone(), typed.replace('-', "")]).collect()),
        ("Sas", format!("{sas:?} {sas:#?}"), vec!["6NHNK68MQQVZ".into(), "6NHN-K68M-QQVZ-5".into(), "c1c1c1".into()]),
    ];
    for (what, printed, secrets) in shown {
        for s in secrets {
            assert!(!printed.contains(&s), "the Debug of {what} prints {s}: {printed}");
        }
    }
}

#[test]
fn the_party_remembers_no_more_than_its_bounds_whatever_a_relay_delivers() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    // A limit on the pairings in flight: the 33rd offer is refused until one finishes or expires.
    for _ in 0..MAX_PENDING {
        w.desktop.create_offer(now).unwrap();
    }
    assert_eq!(w.desktop.remembered().0, MAX_PENDING);
    assert!(matches!(w.desktop.create_offer(now), Err(PairingError::WrongState(_))));
    // Ten minutes and the slack later every one of them has expired and is forgotten.
    let later = now + 700;
    w.desktop.create_offer(later).unwrap();
    assert_eq!(w.desktop.remembered().0, 1);
    // Items that name pairings that do not exist are judged by no one and remembered by no one.
    for i in 0..5000u32 {
        let mut raw = [0u8; 16];
        raw[..4].copy_from_slice(&i.to_le_bytes());
        let event = w.desktop.receive_response(&b64::encode(&raw), "{}", later);
        assert!(matches!(event, PairEvent::Ignored("no such pairing")), "{event:?}");
    }
    assert_eq!(w.desktop.remembered().1, 0, "nothing was remembered about pairings that do not exist");
    // The consumed nonces are capped, the oldest forgotten first.
    w.desktop.restore_consumed((0..5000).map(|i| (format!("n{i}"), format!("j{i}"))));
    let kept = w.desktop.consumed();
    assert_eq!(kept.len(), MAX_CONSUMED);
    assert!(kept.contains(&("n4999".to_string(), "j4999".to_string())) && !kept.contains(&("n0".to_string(), "j0".to_string())));
}

#[test]
fn an_offer_whose_secret_is_in_flight_is_never_replaced() {
    let mut w = world(quick());
    let now = w.env.client.relay_now_or_local();
    let secret = Secret::<16>::random().unwrap();
    let first = w.desktop.create_offer_with(*secret.expose(), [9u8; 32], "pair-AAAAAAAAAAAAAAAA".into(), now).unwrap();
    let again = w.desktop.create_offer_with(*secret.expose(), [9u8; 32], "pair-AAAAAAAAAAAAAAAA".into(), now);
    assert!(matches!(again, Err(PairingError::WrongState(_))), "the pairing in flight was replaced");
    assert_eq!(w.desktop.remembered().0, 1);
    drop(first);
}

fn poll_request() -> PollRequest {
    PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }
}

#[test]
fn a_proof_goes_stale_when_the_wall_clock_moves_by_more_than_the_proof_allows() {
    // The proof is good for 600 seconds by the monotonic clock and by the wall clock alike; a wall clock that goes back by more than a minute leaves nothing to measure it by.
    for (step, good) in
        [(0i64, true), (599, true), (600, true), (601, false), (8 * 3600, false), (-59, true), (-60, true), (-61, false), (-8 * 3600, false)]
    {
        let e = env(quick());
        let (token, _) = e.enrol_desktop();
        e.clock.step_wall(step);
        let r = e.client.poll(&token, &poll_request(), &Cancel::new());
        assert_eq!(r.is_ok(), good, "the wall clock stepped by {step} s: {:?}", r.as_ref().map(|r| r.status));
        if !good {
            assert!(matches!(r, Err(ClientError::NotProved)), "{r:?}");
            // And a fresh proof makes it good again.
            e.client.prove(&Cancel::new()).unwrap();
            assert!(e.client.poll(&token, &poll_request(), &Cancel::new()).is_ok());
        }
    }
}

/// An adapter that steps the wall clock once, in the middle of the first poll it carries (the device slept while the request was out), and notes every URL it is given.
struct SleepsDuringAPoll {
    inner: Arc<dyn HttpClient>,
    clock: Arc<oaiy_relay_core::testing::FakeClock>,
    slept: AtomicBool,
    urls: Mutex<Vec<String>>,
}

impl HttpClient for SleepsDuringAPoll {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        self.urls.lock().unwrap().push(request.url.clone());
        if request.url.contains("/v1/poll") && !self.slept.swap(true, Ordering::SeqCst) {
            self.clock.step_wall(8 * 3600);
        }
        self.inner.send(request)
    }
}

#[test]
fn a_loop_that_slept_proves_the_relay_again_before_it_sends_a_token() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let http = Arc::new(SleepsDuringAPoll {
        inner: Arc::new(e.stub.clone()),
        clock: e.clock.clone(),
        slept: AtomicBool::new(false),
        urls: Mutex::new(Vec::new()),
    });
    let client = Arc::new(RelayClient::new(
        RelayUrl::parse(&e.stub.public_url()).unwrap(),
        Some(e.stub.relay_thumbprint()),
        http.clone(),
        e.clock.clone(),
        Box::new(SeededRng::new(5)),
        ClientConfig::default(),
    ));
    let running = run_loop(&client, &token, MemoryPollStore::new());
    running.wait_for("polls after the sleep", |_| http.urls.lock().unwrap().iter().filter(|u| u.contains("/v1/poll")).count() >= 3);
    let _ = running.finish();
    let urls = http.urls.lock().unwrap().clone();
    let kinds: Vec<&str> = urls.iter().map(|u| if u.contains("/v1/info") { "info" } else { "poll" }).collect();
    // info (the first proof), poll (during which the device slept), then the proof again, and only then the next poll that carries the token.
    assert_eq!(&kinds[..4], ["info", "poll", "info", "poll"], "{urls:?}");
}
