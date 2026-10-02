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
use oaiy_relay_core::testing::stub::Fault;
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
    let sas = math::Sas { raw: [0xC1; 8], chars12: "6NHNK68MQQVZ".into(), check: '5' };
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
