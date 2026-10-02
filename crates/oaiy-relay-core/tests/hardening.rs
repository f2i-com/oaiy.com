//! Tests of the hardening round that followed the independent review: each one pins a behaviour that was missing, wrong or only claimed, and is written so that the mutant of the
//! fix (see `mutation/`) makes it fail. They run against the in-process stub relay with the fake clock, so a loop of minutes costs no time.

mod common;

use std::time::Duration;

use common::env::{env, quick, run_loop};
use oaiy_relay_core::client::*;
use oaiy_relay_core::poll::Outcome;
use oaiy_relay_core::testing::stub::Fault;

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
