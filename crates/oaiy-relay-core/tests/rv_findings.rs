//! The reviewer's tests (not part of the crate's own suite): each one states what the README or the code's own documentation says should happen and fails on `relay-core` where the
//! reviewer found it does not. They are `#[ignore]`d so that the branch's default test run stays green; run them with `cargo test -p oaiy-relay-core --test rv_findings -- --ignored`.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::env::{env, quick, run_loop};
use common::pair::{grants, world};
use oaiy_relay_core::client::*;
use oaiy_relay_core::pairing::phone::Outcome;
use oaiy_relay_core::pairing::{PairEvent, PairingInput, SasOutcome};
use oaiy_relay_core::testing::stub::{Fault, StubConfig};
use oaiy_relay_core::testing::SeededRng;
use oaiy_relay_core::url::RelayUrl;

fn cancel() -> Cancel {
    Cancel::new()
}

/// F-clock: the proof's age is measured on the monotonic clock only, which on Linux and Android (`Instant`) and macOS does not run while the device sleeps.
#[test]
fn rv_a_proof_eight_hours_old_by_the_wall_clock_is_not_good_enough_to_send_a_token() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let req = PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 };
    assert!(e.client.poll(&token, &req, &Cancel::new()).is_ok());
    // The device sleeps for eight hours: the monotonic clock does not move, the wall clock does.
    e.clock.step_wall(8 * 3600);
    e.stub.clear_log();
    let r = e.client.poll(&token, &req, &Cancel::new());
    let sent_a_token = e.stub.log().iter().any(|l| l.authorization.is_some());
    assert!(!sent_a_token && r.is_err(), "a bearer was sent on a proof that is 8 hours old by the wall clock: {r:?}");
}

struct Rec {
    inner: Arc<dyn HttpClient>,
    clock: Arc<dyn Clock>,
    log: Mutex<Vec<(String, Duration)>>,
}

impl HttpClient for Rec {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        self.log.lock().unwrap().push((request.url.clone(), self.clock.monotonic()));
        self.inner.send(request)
    }
}

/// F-P1: README P1: a client that cancels a running poll starts the next no sooner than 250 ms after it started the one it cancels. The loop's branch for it is guarded by
/// `network_changed`, which a proof (always due after a network change) has consumed before the branch is reached.
#[test]
fn rv_a_poll_that_replaces_another_after_a_network_change_waits_for_the_250_ms_of_p1() {
    let e = env(StubConfig { wait_default: 5, wait_max: 5, ..quick() });
    let (token, _) = e.enrol_desktop();
    let rec = Arc::new(Rec { inner: Arc::new(e.stub.clone()), clock: e.clock.clone(), log: Mutex::new(Vec::new()) });
    let client = Arc::new(RelayClient::new(
        RelayUrl::parse(&e.stub.public_url()).unwrap(),
        Some(e.stub.relay_thumbprint()),
        rec.clone(),
        e.clock.clone(),
        Box::new(SeededRng::new(5)),
        ClientConfig::default(),
    ));
    let running = run_loop(&client, &token, MemoryPollStore::new());
    // wait until the first (held) poll is in flight
    let started = std::time::Instant::now();
    loop {
        if rec.log.lock().unwrap().iter().any(|(u, _)| u.contains("/v1/poll")) {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(5), "no poll started");
        std::thread::sleep(Duration::from_millis(2));
    }
    running.handle.network_changed();
    // wait for the replacement poll
    let started = std::time::Instant::now();
    loop {
        if rec.log.lock().unwrap().iter().filter(|(u, _)| u.contains("/v1/poll")).count() >= 2 {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(10), "no replacement poll");
        std::thread::sleep(Duration::from_millis(2));
    }
    let times: Vec<Duration> = rec.log.lock().unwrap().iter().filter(|(u, _)| u.contains("/v1/poll")).map(|(_, t)| *t).collect();
    let _ = running.finish();
    let gap = times[1].saturating_sub(times[0]);
    assert!(gap >= Duration::from_millis(250), "the replacing poll started {gap:?} after the poll it cancelled (README P1: no sooner than 250 ms)");
}

/// F-respond: after the desktop rejects a response for its window (a desktop that was offline), the phone is told `Rejected` ("the phone may answer again"), but `respond()` resends the
/// first response text, whose claims have lapsed, so the pairing can never complete.
#[test]
fn rv_a_phone_answers_afresh_after_its_response_was_rejected_for_its_window() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 1);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    // The desktop was offline for 200 seconds: the claims (issuedAt .. issuedAt + 120, 30 s slack) have lapsed when it polls.
    w.env.clock.advance(Duration::from_secs(200));
    let events = w.deliver();
    assert!(matches!(events.as_slice(), [PairEvent::Rejected { reason: "window", .. }]), "{events:?}");
    let out = phone.wait_outcome(None, &cancel()).unwrap();
    assert!(matches!(out, Outcome::Rejected), "the relay returned the rendezvous to open");
    phone.respond(&cancel()).unwrap();
    let again = w.deliver();
    assert!(
        again.iter().any(|e| matches!(e, PairEvent::AwaitingSas { .. })),
        "the second answer must be a fresh response, not the lapsed one: {again:?}"
    );
}

/// F-confirm: `approve_body` documents that a retry of a decision whose answer was lost sends the same receipt, but `confirm_sas`, the documented whole gate, refuses the retry
/// because `approve_body` already moved the pairing out of the phase `submit_sas` reads.
#[test]
fn rv_confirm_sas_is_retryable_when_the_decision_post_was_lost() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 1);
    phone.fetch_offer(&cancel()).unwrap();
    let sas = phone.respond(&cancel()).unwrap();
    w.deliver();
    w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Drop);
    let first = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel());
    assert!(first.is_err(), "the decision never reached the relay: {first:?}");
    let second = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &grants(), &cancel());
    assert!(matches!(second, Ok(SasOutcome::Approved { .. })), "a retry with the same, correct code must send the same receipt again: {second:?}");
}

/// F-pairwait: after a held `GET /v1/pair/{pid}?wait=..&state=answered` the phone asks again at once, whatever came back. A relay (or a proxy that does not hold) that answers at once
/// makes it a loop with no pause; the real relay bounds it with a 30-per-minute bucket per address.
#[test]
fn rv_the_phone_does_not_spin_on_a_relay_that_answers_a_pairing_wait_at_once() {
    let mut w = world(StubConfig { wait_default: 1, wait_max: 1, receipt_includes_grants: true, ..quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 1);
    phone.fetch_offer(&cancel()).unwrap();
    phone.respond(&cancel()).unwrap();
    // The body the relay gives for an answered rendezvous, replayed at once for every further read.
    let probe = HttpRequest {
        method: Method::Get,
        url: format!("{}/v1/pair/{}", w.env.stub.public_url(), offer.pid),
        headers: vec![],
        body: None,
        timeout: Duration::from_secs(5),
        max_response_bytes: 1 << 20,
        cancel: cancel(),
    };
    let body = String::from_utf8(w.env.stub.handle(&probe).unwrap().body).unwrap();
    w.env.stub.clear_log();
    w.env.stub.fail_next_on("/v1/pair/", 5000, Fault::Respond(200, vec![], body));
    let cancel_flag = cancel();
    let flag = cancel_flag.clone();
    let stub = w.env.stub.clone();
    let clock = w.env.clock.clone();
    let watcher = std::thread::spawn(move || {
        let t = std::time::Instant::now();
        while t.elapsed() < Duration::from_millis(1500) && stub.log().len() < 200 {
            std::thread::sleep(Duration::from_millis(5));
        }
        flag.cancel();
    });
    let _ = phone.wait_outcome(None, &cancel_flag);
    watcher.join().unwrap();
    let reads = w.env.stub.log().iter().filter(|r| r.target.starts_with("/v1/pair/") && r.method == "GET").count();
    let elapsed = clock.monotonic();
    // At least one second of the client's own time per request after an unheld answer (the relay's bucket is 30 a minute per address).
    assert!(reads <= 1 || elapsed.as_secs_f64() / (reads as f64) >= 1.0, "{reads} reads in {elapsed:?} of the client's clock");
}
