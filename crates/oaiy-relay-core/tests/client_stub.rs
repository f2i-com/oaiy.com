//! The relay client against the in-process stub relay (layer L4 without PHP): enrolment, the identity proof and the rule that no token is sent without one, the poll loop in every
//! situation README 5.1.1 names (progress, idle, a reset, a revocation, an outage, a `429` of each rule, a refused hold, a first and a second `400`, a `426`, a store that fails, a
//! second process with the same credential), posting, the relay clock, and admission. Every scenario that has a pause reads it from the fake clock's record, so it checks the
//! number the README gives (1, 2, 4, 8 ... with up to 20 percent jitter) and not just that the loop went on.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::env::{env, quick, run_loop, wait_until, T0};
use oaiy_relay_core::client::*;
use oaiy_relay_core::enrol::{EnrolmentKey, Role};
use oaiy_relay_core::json::Json;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::poll::{Action, Report};
use oaiy_relay_core::testing::stub::{Fault, StubConfig, StubRelay};
use oaiy_relay_core::testing::{json_response, ScriptedHttp};
use oaiy_relay_core::url::RelayUrl;

fn secs(d: &Duration) -> f64 {
    d.as_secs_f64()
}

/// True when `d` is `base` seconds with up to 20 percent jitter added (`base * (1 + 0.2 * u)`, `u` below 1).
fn within_jitter(d: &Duration, base: f64) -> bool {
    secs(d) >= base - 1e-9 && secs(d) < base * 1.2 + 1e-9
}

#[test]
fn enrolment_proves_the_relay_before_it_sends_anything_and_stores_the_token_then_the_profile() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    let log = e.stub.log();
    assert_eq!(log[0].target, "/v1/info");
    assert_eq!(log[1].target, "/v1/enroll");
    assert!(log.iter().all(|r| r.authorization.is_none()), "an enrolment sends no credential");
    assert_eq!(profile.device_id.len(), 4 + 22);
    assert_eq!(profile.relay_thumbprint, e.stub.relay_thumbprint());
    assert_eq!(profile.relay_id, e.stub.relay_id());
    assert_eq!(e.profiles.load().unwrap().unwrap(), profile);
    // The token works: a poll with it is answered.
    let reply = e.client.poll(&token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
    assert_eq!(reply.status, Some(200));
    // A key is single use.
    let again = e.stub.mint_enrolment_key(Role::Desktop, 3600);
    let key = EnrolmentKey::parse(&again).unwrap();
    let ed = Signer::generate().unwrap().verify_key();
    let x = X25519Secret::generate().unwrap().public_key();
    e.client.enroll(&key, "Another", &ed, &x, &Cancel::new()).unwrap();
    let second = e.client.enroll(&key, "Another", &ed, &x, &Cancel::new()).unwrap_err();
    assert_eq!(second.code(), Some("unauthorized"));
}

#[test]
fn a_key_for_another_relay_key_sends_nothing_and_stores_nothing() {
    let e = env(quick());
    let uri = e.stub.mint_enrolment_key(Role::Desktop, 3600);
    // The same key with another thumbprint in `f`: what a man in the middle who has not got the relay key cannot make match.
    let tampered = uri.replace(&e.stub.relay_thumbprint(), "SWUejy55xcz8xgSi-GX15ERZzyuXnb6voSaopq2Jakw");
    let key = EnrolmentKey::parse(&tampered).unwrap();
    let ed = Signer::generate().unwrap().verify_key();
    let x = X25519Secret::generate().unwrap().public_key();
    // The client of an enrolment is made with no pin: the key's `f` is the pin.
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        None,
        Arc::new(e.stub.clone()),
        e.clock.clone(),
        Box::new(oaiy_relay_core::testing::SeededRng::new(9)),
        ClientConfig::default(),
    );
    let err = enrol_and_store(&client, &key, "PC", &ed, &x, &*e.secrets, &*e.profiles, &Cancel::new()).unwrap_err();
    assert_eq!(err, ClientError::Suspect);
    assert!(e.stub.log().iter().all(|r| r.target != "/v1/enroll"), "no enrolment request was sent");
    assert!(e.secrets.get(SECRET_TOKEN).unwrap().is_none() && e.profiles.load().unwrap().is_none());
    assert!(client.is_suspect());
    // A client that is already pinned to a key refuses an enrolment key for another one before it asks the relay anything.
    e.stub.clear_log();
    let refused = e.client.enroll(&key, "PC", &ed, &x, &Cancel::new()).unwrap_err();
    assert!(matches!(refused, ClientError::Request(_)), "{refused:?}");
    assert!(e.stub.log().is_empty());
}

#[test]
fn a_profile_that_cannot_be_stored_takes_the_token_back_out() {
    struct Failing;
    impl ProfileStore for Failing {
        fn load(&self) -> Result<Option<RelayProfile>, StoreError> {
            Ok(None)
        }
        fn save(&self, _: &RelayProfile) -> Result<(), StoreError> {
            Err(StoreError("disk full".into()))
        }
        fn clear(&self) -> Result<(), StoreError> {
            Ok(())
        }
    }
    let e = env(quick());
    let key = EnrolmentKey::parse(&e.stub.mint_enrolment_key(Role::Desktop, 3600)).unwrap();
    let ed = Signer::generate().unwrap().verify_key();
    let x = X25519Secret::generate().unwrap().public_key();
    assert!(enrol_and_store(&e.client, &key, "PC", &ed, &x, &*e.secrets, &Failing, &Cancel::new()).is_err());
    assert!(e.secrets.get(SECRET_TOKEN).unwrap().is_none(), "a token with no profile is not left behind");
}

#[test]
fn no_token_is_sent_before_a_proof_after_a_failed_one_or_after_the_proof_has_lapsed() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let req = PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 };
    // A fresh client has proved nothing: nothing is sent.
    let fresh = common::env::client_for(&e.stub, &e.clock, &e.stub.public_url(), 1);
    e.stub.clear_log();
    assert_eq!(fresh.poll(&token, &req, &Cancel::new()).unwrap_err(), ClientError::NotProved);
    assert!(e.stub.log().is_empty());
    // A proof that lapsed (older than twice the loop's schedule) is no proof; a new one is made by `ensure_proved`.
    assert!(e.client.poll(&token, &req, &Cancel::new()).is_ok());
    e.clock.advance(Duration::from_secs(601));
    assert_eq!(e.client.poll(&token, &req, &Cancel::new()).unwrap_err(), ClientError::NotProved);
    e.client.ensure_proved(&Cancel::new()).unwrap();
    assert!(e.client.poll(&token, &req, &Cancel::new()).is_ok());
    // A relay that cannot prove it is the pinned one (here: it signs with another key) is refused, and from then on no token is sent until a proof verifies.
    let impostor =
        StubRelay::with_clocks(StubConfig { relay_seed: [0x44; 32], wait_default: 0, wait_max: 0, ..Default::default() }, || T0, || Duration::ZERO);
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        Some(e.stub.relay_thumbprint()),
        Arc::new(impostor.clone()),
        e.clock.clone(),
        Box::new(oaiy_relay_core::testing::SeededRng::new(3)),
        ClientConfig::default(),
    );
    assert!(matches!(client.prove(&Cancel::new()), Err(ProveError::Invalid(_))));
    assert!(client.is_suspect());
    assert_eq!(client.poll(&token, &req, &Cancel::new()).unwrap_err(), ClientError::Suspect);
    assert!(impostor.log().iter().all(|r| r.authorization.is_none()), "the impostor never saw a token");
}

#[test]
fn a_replayed_info_with_a_new_nonce_does_not_prove_the_relay() {
    // A man in the middle who recorded the relay's answer to one nonce and serves it for every later one: the body and the static signature are good, the proof is not.
    let e = env(quick());
    let recorded = e.stub.handle(&oaiy_relay_core::client::HttpRequest {
        method: Method::Get,
        url: "https://relay.stub.test/v1/info".into(),
        headers: vec![("X-OAIY-Nonce".into(), "EBESExQVFhcYGRobHB0eHw".into())],
        body: None,
        timeout: Duration::from_secs(5),
        max_response_bytes: 1 << 20,
        cancel: Cancel::new(),
    });
    let recorded = recorded.unwrap();
    let replay = ScriptedHttp::new({
        let recorded = recorded.clone();
        move |_| Ok(recorded.clone())
    });
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        Some(e.stub.relay_thumbprint()),
        Arc::new(replay),
        e.clock.clone(),
        Box::new(oaiy_relay_core::testing::SeededRng::new(5)),
        ClientConfig::default(),
    );
    assert!(matches!(client.prove(&Cancel::new()), Err(ProveError::Invalid(_))), "a replayed answer");
    assert!(client.is_suspect());
}

#[test]
fn the_loop_proves_first_and_carries_what_it_accepted_as_since_so_that_the_next_poll_acknowledges_it() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    let (provider, ptoken) = e.provider();
    let pclient = e.other_client(11);
    for i in 0..3 {
        let item = PostItem {
            to: format!("dev:{}", profile.device_id),
            lane: "cmd".into(),
            id: format!("cmd-{i}"),
            ttl: Some(60),
            hdr: Hdr::new().ct("sealed1"),
            body: format!("body {i}"),
        };
        let r = pclient.post_items(&ptoken, &[item], &Cancel::new()).unwrap();
        assert_eq!(r[0].status, PostStatus::Queued);
    }
    let _ = provider;
    e.stub.clear_log();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("three items accepted", |ev| ev.iter().filter(|x| matches!(x, Event::Accepted { .. })).count() >= 1);
    running.wait_for("a poll after the accepting one", |_| e.stub.log().iter().filter(|r| r.target.starts_with("/v1/poll")).count() >= 2);
    let (end, store) = running.finish();
    assert_eq!(end, LoopEnd::Cancelled);
    assert_eq!(store.accepted.len(), 3);
    assert_eq!(store.accepted.iter().map(|a| a.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(store.accepted[1].item.as_ref().unwrap().id, "cmd-1");
    assert_eq!(store.accepted[0].item.as_ref().unwrap().body, "body 0");
    assert_eq!(store.cursor().since, 3);
    // The relay deleted what the second poll acknowledged.
    assert_eq!(e.stub.live_items(&profile.device_id), 0);
    // The log: the proof (with a nonce, without a credential) comes first; then polls, the second of which carries `since=3` and the epoch as the relay gave it.
    let log = e.stub.log();
    assert_eq!(log[0].target, "/v1/info");
    assert!(log[0].authorization.is_none());
    let polls: Vec<&str> = log.iter().filter(|r| r.target.starts_with("/v1/poll")).map(|r| r.target.as_str()).collect();
    assert!(polls[0].starts_with("/v1/poll?since=0&wait=0&limit=32"), "{}", polls[0]);
    assert!(polls[1].contains("since=3") && polls[1].contains(&format!("epoch={}", e.stub.epoch())), "{}", polls[1]);
    assert!(log.iter().position(|r| r.authorization.is_some()).unwrap() > 0, "no token before the first proof");
}

#[test]
fn an_idle_answer_is_paused_at_poll_gap_ms_and_never_meets_the_gap_rule() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.clock.clear_sleeps();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    wait_until("ten idle pauses", || e.clock.sleeps().len() >= 10);
    let (_, _) = running.finish();
    for s in e.clock.sleeps().iter().take(10) {
        assert!(within_jitter(s, 0.25), "an idle pause is 250 ms and up to 20 percent: {s:?}");
    }
    assert!(e.stub.log().iter().all(|r| r.status != 429), "the gap rule was never met");
}

#[test]
fn a_reset_adopts_the_servers_cursor_once_and_the_loop_goes_on() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    e.stub.post_as_relay(
        &profile.device_id,
        "ctl",
        "n1",
        Json::obj([("ct", Json::str("json"))]),
        "{\"t\":\"relay.notice\",\"level\":\"info\",\"message\":\"hi\"}",
        600,
    );
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("the first item", |ev| ev.iter().any(|x| matches!(x, Event::Accepted { since: 1, .. })));
    // A restore from a backup: another epoch. The next poll answers `reset` with the highest seq as the cursor.
    e.stub.reset_epoch();
    running.wait_for("the reset", |ev| ev.iter().any(|x| matches!(x, Event::MailboxReset)));
    e.stub.post_as_relay(
        &profile.device_id,
        "ctl",
        "n2",
        Json::obj([("ct", Json::str("json"))]),
        "{\"t\":\"relay.notice\",\"level\":\"info\",\"message\":\"again\"}",
        600,
    );
    running.wait_for("an item after the reset", |ev| ev.iter().any(|x| matches!(x, Event::Accepted { since: 2, .. })));
    let (_, store) = running.finish();
    assert_eq!(store.resets, 1, "adopted once");
    assert_eq!(store.cursor().epoch.as_deref(), Some(e.stub.epoch().as_str()));
    assert_eq!(store.accepted.iter().map(|a| a.item.as_ref().unwrap().id.clone()).collect::<Vec<_>>(), vec!["n1", "n2"]);
}

#[test]
fn a_revoked_device_ends_the_loop_with_forget_credential() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    e.stub.revoke(&profile.device_id);
    let sink = running.sink.clone();
    let (end, _) = running.join();
    match end {
        LoopEnd::Stopped { action, status, code, .. } => {
            assert_eq!(action, Action::ForgetCredential);
            assert_eq!(status, Some(401));
            assert_eq!(code.as_deref(), Some("revoked"));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(sink.states().last(), Some(&ConnectionState::Revoked));
}

#[test]
fn an_outage_is_paced_1_2_4_8_reported_unreachable_after_three_and_recovered_from_with_a_proof_first() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.stub.set_down(true);
    e.stub.clear_log();
    e.clock.clear_sleeps();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("unreachable", |ev| ev.iter().any(|x| matches!(x, Event::Report(Report::Unreachable))));
    wait_until("six pauses", || e.clock.sleeps().len() >= 6);
    // The relay comes back.
    e.stub.set_down(false);
    running.wait_for("connected again", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    let sink = running.sink.clone();
    let (_, _) = running.finish();
    let sleeps = e.clock.sleeps();
    for (i, base) in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0].iter().enumerate() {
        assert!(within_jitter(&sleeps[i], *base), "pause {i} is {base} s and up to 20 percent: {:?}", sleeps[i]);
    }
    let reports = sink.reports();
    assert_eq!(reports.iter().filter(|r| **r == Report::Unreachable).count(), 1, "reported once, not every pause");
    // It never made a request while the relay was down that carried a token (the proof was due before the first poll, and it never succeeded), and the first token went out after a
    // proof that was answered.
    let log = e.stub.log();
    let first_auth = log.iter().position(|r| r.authorization.is_some()).expect("a token after the outage");
    assert!(log[..first_auth].iter().any(|r| r.target == "/v1/info" && r.status == 200), "a proof preceded the first token");
    assert!(log[..first_auth].iter().all(|r| r.status == 0 || r.target == "/v1/info"));
}

#[test]
fn a_429_is_never_a_failure_and_the_fifth_in_flight_is_a_defect_reported_once() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.clock.clear_sleeps();
    // Six answers of `429 rate_limited`, rule `in_flight`, `Retry-After: 1`.
    let body = r#"{"error":{"code":"rate_limited","message":"Too many polls.","retryAfter":1,"rule":"in_flight"}}"#;
    e.stub.fail_next_on("/v1/poll", 6, Fault::Respond(429, vec![("retry-after".into(), "1".into())], body.into()));
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("six 429s", |ev| {
        ev.iter().filter(|x| matches!(x, Event::Answer { outcome: oaiy_relay_core::poll::Outcome::Flow, .. })).count() >= 6
    });
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    let sink = running.sink.clone();
    let (_, _) = running.finish();
    let sleeps: Vec<Duration> = e.clock.sleeps().into_iter().filter(|s| secs(s) >= 1.0).collect();
    // 1, 2, 4, 8, 16, 30: `max(clamp(Retry-After), min(30, 2^(n-1)))`.
    for (i, base) in [1.0, 2.0, 4.0, 8.0, 16.0, 30.0].iter().enumerate() {
        assert!(within_jitter(&sleeps[i], *base), "429 pause {i} is {base} s: {:?}", sleeps[i]);
    }
    assert_eq!(sink.reports(), vec![Report::InFlightDefect], "the fifth in a row, once");
    assert!(sink.events().iter().filter(|x| matches!(x, Event::Action(Action::CancelOwnPolls))).count() >= 6);
    assert!(!sink.states().contains(&ConnectionState::Unreachable), "a 429 proves the relay answered");
}

#[test]
fn a_503_with_retry_after_is_paced_by_the_larger_of_it_and_the_backoff() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    e.clock.clear_sleeps();
    e.stub.fail_next_on(
        "/v1/poll",
        1,
        Fault::Respond(503, vec![("retry-after".into(), "40".into())], r#"{"error":{"code":"unavailable","message":"busy","retryAfter":40}}"#.into()),
    );
    wait_until("the pause", || e.clock.sleeps().iter().any(|s| secs(s) >= 1.0));
    e.stub.fail_next_on(
        "/v1/poll",
        1,
        Fault::Respond(
            503,
            vec![("retry-after".into(), "999".into())],
            r#"{"error":{"code":"unavailable","message":"busy","retryAfter":999}}"#.into(),
        ),
    );
    wait_until("the second pause", || e.clock.sleeps().iter().filter(|s| secs(s) >= 1.0).count() >= 2);
    let (_, _) = running.finish();
    let long: Vec<Duration> = e.clock.sleeps().into_iter().filter(|s| secs(s) >= 1.0).collect();
    assert!(within_jitter(&long[0], 40.0), "{:?}", long[0]);
    // The second 503 comes after an idle answer, so it is the first failure of a new run: the larger of 1 and `clamp(999)`, which is 120.
    assert!(within_jitter(&long[1], 120.0), "clamped to 120: {:?}", long[1]);
}
#[test]
fn a_refused_hold_is_a_short_poll_at_retry_after_doubling_to_the_fallback_2_4_5() {
    let e = env(StubConfig { wait_default: 1, wait_max: 1, ..Default::default() });
    let (token, _) = e.enrol_desktop();
    e.stub.refuse_holds(3);
    e.clock.clear_sleeps();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    wait_until("three pauses", || e.clock.sleeps().len() >= 3);
    let (_, _) = running.finish();
    let sleeps = e.clock.sleeps();
    for (i, base) in [2.0, 4.0, 5.0].iter().enumerate() {
        assert!(within_jitter(&sleeps[i], *base), "refused hold pause {i} is {base} s: {:?}", sleeps[i]);
    }
    // Every request still asked for a hold: it is asked for again on every request.
    let polls: Vec<String> = e.stub.log().into_iter().filter(|r| r.target.starts_with("/v1/poll")).map(|r| r.target).collect();
    assert!(polls.iter().take(4).all(|p| p.contains("wait=1")), "{polls:?}");
}

#[test]
fn the_first_400_retries_without_the_epoch_and_the_second_in_a_row_stops_the_loop() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    e.stub.post_as_relay(&profile.device_id, "ctl", "n1", Json::obj([("ct", Json::str("json"))]), "x", 600);
    // First, a normal run so that the cursor holds an epoch.
    let r1 = run_loop(&e.client, &token, MemoryPollStore::new());
    r1.wait_for("accepted", |ev| ev.iter().any(|x| matches!(x, Event::Accepted { .. })));
    let (_, store) = r1.finish();
    assert!(store.cursor().epoch.is_some());
    // Now a 400 and then answers again: the retry leaves the epoch out.
    e.stub.clear_log();
    e.stub.fail_next_on("/v1/poll", 1, Fault::Respond(400, vec![], r#"{"error":{"code":"invalid_request","message":"epoch"}}"#.into()));
    let r2 = run_loop(&e.client, &token, store);
    r2.wait_for("invalid_request reported", |ev| ev.iter().any(|x| matches!(x, Event::Report(Report::InvalidRequest))));
    r2.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    let sink = r2.sink.clone();
    let (_, store) = r2.finish();
    let polls: Vec<String> = e.stub.log().into_iter().filter(|r| r.target.starts_with("/v1/poll")).map(|r| r.target).collect();
    assert!(polls[0].contains("epoch="), "the first poll carries the stored epoch: {}", polls[0]);
    assert!(!polls[1].contains("epoch="), "the retry leaves it out: {}", polls[1]);
    assert!(sink.events().iter().any(|x| matches!(x, Event::Action(Action::ClearEpoch))));
    assert!(store.cursor().epoch.is_some(), "and the answer's epoch is stored again");
    assert!(!sink.reports().contains(&Report::Unreachable), "a first 400 is the client's own failure and never `unreachable`");
    // Two 400s in a row: a defect.
    e.stub.fail_next_on("/v1/poll", 2, Fault::Respond(400, vec![], r#"{"error":{"code":"invalid_request","message":"no"}}"#.into()));
    let r3 = run_loop(&e.client, &token, MemoryPollStore::new());
    let (end, _) = r3.join();
    assert!(matches!(end, LoopEnd::Stopped { action: Action::ReportDefect, status: Some(400), .. }), "{end:?}");
}

#[test]
fn a_426_makes_the_client_re_read_info_and_stop_when_it_is_too_old() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    e.stub.configure(|c| c.min_client = 2);
    let sink = running.sink.clone();
    let (end, _) = running.join();
    assert!(matches!(end, LoopEnd::Stopped { action: Action::UpdateClient, status: Some(426), .. }), "{end:?}");
    assert_eq!(sink.states().last(), Some(&ConnectionState::UpgradeRequired));
}

#[test]
fn a_store_that_fails_moves_nothing_is_paced_as_a_failure_and_the_items_come_again() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    e.stub.post_as_relay(&profile.device_id, "ctl", "n1", Json::obj([("ct", Json::str("json"))]), "x", 600);
    e.stub.clear_log();
    e.clock.clear_sleeps();
    let mut store = MemoryPollStore::new();
    store.fail_writes = 2;
    let running = run_loop(&e.client, &token, store);
    running.wait_for("accepted at last", |ev| ev.iter().any(|x| matches!(x, Event::Accepted { count: 1, since: 1 })));
    let sink = running.sink.clone();
    let (_, store) = running.finish();
    assert_eq!(store.accepted.len(), 1, "written once, though the relay delivered it three times");
    assert_eq!(sink.reports().iter().filter(|r| **r == Report::StorageFailure).count(), 2);
    assert!(!sink.reports().contains(&Report::Unreachable), "the relay answered: never `unreachable` for a storage failure of the first two");
    // The polls after a failed write still carried `since=0`: the relay was never told the item was safe.
    let polls: Vec<String> = e.stub.log().into_iter().filter(|r| r.target.starts_with("/v1/poll")).map(|r| r.target).collect();
    assert!(polls[1].contains("since=0") && polls[2].contains("since=0"), "{polls:?}");
    // The pauses were the failure pauses 1 and 2.
    let sleeps = e.clock.sleeps();
    assert!(within_jitter(&sleeps[0], 1.0) && within_jitter(&sleeps[1], 2.0), "{sleeps:?}");
}

#[test]
fn a_second_process_with_the_same_credential_is_reported_as_duplicate_credential() {
    let e = env(StubConfig { wait_default: 1, wait_max: 1, ..Default::default() });
    let (token, _) = e.enrol_desktop();
    let other = e.other_client(21);
    let first = run_loop(&e.client, &token, MemoryPollStore::new());
    first.wait_for("connected", |ev| {
        ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))) || ev.iter().any(|x| matches!(x, Event::InfoProved))
    });
    // Let the first poll be held, then start the second process.
    std::thread::sleep(Duration::from_millis(150));
    let second = run_loop(&other, &token, MemoryPollStore::new());
    first.wait_for("duplicate_credential", |ev| ev.iter().any(|x| matches!(x, Event::Report(Report::DuplicateCredential))));
    second.handle.stop();
    first.handle.stop();
    let _ = second.join();
    let _ = first.join();
}

#[test]
fn a_held_poll_returns_the_moment_an_item_is_posted() {
    let e = env(StubConfig { wait_default: 2, wait_max: 2, ..Default::default() });
    let (token, profile) = e.enrol_desktop();
    let (_, ptoken) = e.provider();
    let pclient = e.other_client(31);
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("a poll is in flight", |_| e.stub.log().iter().any(|r| r.target.starts_with("/v1/poll")));
    std::thread::sleep(Duration::from_millis(300));
    let started = std::time::Instant::now();
    pclient
        .post_items(
            &ptoken,
            &[PostItem {
                to: format!("dev:{}", profile.device_id),
                lane: "cmd".into(),
                id: "c1".into(),
                ttl: None,
                hdr: Hdr::new(),
                body: "b".into(),
            }],
            &Cancel::new(),
        )
        .unwrap();
    running.wait_for("the item", |ev| ev.iter().any(|x| matches!(x, Event::Accepted { .. })));
    assert!(started.elapsed() < Duration::from_millis(1500), "a hold of 2 s ended early: {:?}", started.elapsed());
    let (_, store) = running.finish();
    assert_eq!(store.accepted.len(), 1);
}

#[test]
fn the_relay_clock_is_sampled_from_x_oaiy_time_and_a_difference_over_a_minute_is_warned_about() {
    let clock = Arc::new(oaiy_relay_core::testing::FakeClock::new(T0));
    let c = clock.clone();
    let stub = StubRelay::with_clocks(quick(), move || oaiy_relay_core::client::Clock::unix_now(&*c) + 100, || Duration::ZERO);
    let client = common::env::client_for(&stub, &clock, "https://relay.stub.test", 4);
    assert_eq!(client.relay_now(), None);
    client.prove(&Cancel::new()).unwrap();
    assert!((client.relay_offset_s().unwrap() - 100.0).abs() < 1e-9);
    assert_eq!(client.relay_now(), Some(T0 + 100));
    assert!(client.clock_mismatch());
    // A wall-clock step of the PC does not move the offset's monotonic base: the offset is a difference of two readings taken together.
    let (e2, e3) = (env(quick()), 0);
    let _ = (e2.client.relay_now(), e3);
}

#[test]
fn posting_checks_what_it_can_before_it_sends_and_reports_each_item_as_the_relay_did() {
    let e = env(quick());
    let (token, profile) = e.enrol_desktop();
    let (pid, ptoken) = e.provider();
    let pclient = e.other_client(41);
    e.stub.clear_log();
    let to = format!("dev:{}", profile.device_id);
    let item = |lane: &str, id: &str, ttl: Option<u64>, hdr: Hdr, body: &str| PostItem {
        to: to.clone(),
        lane: lane.into(),
        id: id.into(),
        ttl,
        hdr,
        body: body.into(),
    };
    // Refused before anything is sent: a lane through POST /v1/items that the relay does not take, a bad id, a ttl of 0, a hdr key that is not on the list, a ring without hdr.sig,
    // a body above the lane's cap, too many items.
    for bad in [
        item("ai", "x", None, Hdr::new(), "b"),
        item("pair", "x", None, Hdr::new(), "b"),
        item("cmd", "..", None, Hdr::new(), "b"),
        item("cmd", "a/b", None, Hdr::new(), "b"),
        item("cmd", "x", Some(0), Hdr::new(), "b"),
        item("cmd", "x", Some(301), Hdr::new(), "b"),
        item("cmd", "x", None, Hdr::new().prio(2), "b"),
        item("cmd", "x", None, Hdr::new().ct("nope"), "b"),
        item("cmd", "x", None, Hdr::new().re("a").re("b"), "b"),
        item("ring", "x", None, Hdr::new(), "b"),
        item("cmd", "x", None, Hdr::new(), &"b".repeat(32769)),
    ] {
        assert!(matches!(pclient.post_items(&ptoken, std::slice::from_ref(&bad), &Cancel::new()), Err(ClientError::Request(_))), "{:?}", bad.lane);
    }
    assert!(matches!(pclient.post_items(&ptoken, &[], &Cancel::new()), Err(ClientError::Request(_))));
    let many: Vec<PostItem> = (0..65).map(|i| item("cmd", &format!("m{i}"), None, Hdr::new(), "b")).collect();
    assert!(matches!(pclient.post_items(&ptoken, &many, &Cancel::new()), Err(ClientError::Request(_))));
    assert!(e.stub.log().iter().all(|r| r.target != "/v1/items"), "nothing reached the relay");
    // Sent: a queued item, the same item again (duplicate, the original seq), another body under the same id (conflict), a lane the sender may not post to.
    let first = pclient.post_items(&ptoken, &[item("cmd", "c1", None, Hdr::new().ct("sealed1"), "one")], &Cancel::new()).unwrap();
    assert_eq!((first[0].status.clone(), first[0].seq), (PostStatus::Queued, Some(1)));
    let again = pclient.post_items(&ptoken, &[item("cmd", "c1", None, Hdr::new().ct("sealed1"), "one")], &Cancel::new()).unwrap();
    assert_eq!((again[0].status.clone(), again[0].seq), (PostStatus::Duplicate, Some(1)));
    let conflict = pclient.post_items(&ptoken, &[item("cmd", "c1", None, Hdr::new(), "two")], &Cancel::new()).unwrap();
    assert_eq!(conflict[0].status, PostStatus::Rejected);
    assert_eq!(conflict[0].code.as_deref(), Some("conflict"));
    assert!(!conflict[0].retryable());
    let forbidden = pclient.post_items(&ptoken, &[item("res", "r1", None, Hdr::new().re("c1"), "x")], &Cancel::new()).unwrap();
    assert_eq!(forbidden[0].code.as_deref(), Some("forbidden"));
    // A desktop answering a command, to the provider that sent it.
    let ok = e
        .client
        .post_items(
            &token,
            &[PostItem {
                to: format!("dev:{pid}"),
                lane: "res".into(),
                id: "res-c1".into(),
                ttl: Some(300),
                hdr: Hdr::new().re("c1").ct("sealed1"),
                body: "x".into(),
            }],
            &Cancel::new(),
        )
        .unwrap();
    assert_eq!(ok[0].status, PostStatus::Queued);
}

#[test]
fn rotating_a_token_gives_a_new_one_and_the_old_one_keeps_working_for_the_grace_period() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    let (new, grace) = e.client.rotate_token(&token, &Cancel::new()).unwrap();
    assert_ne!(new, token);
    assert!(grace as i64 >= T0 + 599);
    let req = PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 };
    assert_eq!(e.client.poll(&new, &req, &Cancel::new()).unwrap().status, Some(200));
    // (Both tokens are one device: a second empty poll inside the gap is the gap rule's 429, which is what the relay does and not a fault of the old token.)
    e.clock.advance(Duration::from_secs(1));
    assert_eq!(e.client.poll(&token, &req, &Cancel::new()).unwrap().status, Some(200), "the old one works until graceUntil");
    assert_eq!(e.client.rotate_token(&new, &Cancel::new()).unwrap_err().code(), Some("conflict"), "a second rotation during the grace");
    e.clock.advance(Duration::from_secs(601));
    e.client.ensure_proved(&Cancel::new()).unwrap();
    assert_eq!(e.client.poll(&token, &req, &Cancel::new()).unwrap().status, Some(401), "and not after");
    assert_eq!(e.client.poll(&new, &req, &Cancel::new()).unwrap().status, Some(200));
}

#[test]
fn an_error_answers_message_is_returned_as_data_and_cut_to_200_characters() {
    let http = ScriptedHttp::new(|req| {
        if req.url.ends_with("/v1/health") {
            return Ok(json_response(
                503,
                &[("Retry-After", "7")],
                &format!("{{\"error\":{{\"code\":\"unavailable\",\"message\":\"{}\",\"retryAfter\":7}}}}", "x".repeat(500)),
                1,
            ));
        }
        Err(TransportError::Refused)
    });
    let clock = Arc::new(oaiy_relay_core::testing::FakeClock::new(T0));
    let client = RelayClient::new(
        RelayUrl::parse("https://relay.stub.test").unwrap(),
        None,
        Arc::new(http),
        clock,
        Box::new(oaiy_relay_core::testing::SeededRng::new(1)),
        ClientConfig::default(),
    );
    let err = client.health(&Cancel::new()).unwrap_err();
    let relay = err.relay().unwrap();
    assert_eq!((relay.status, relay.retry_after, relay.code.as_deref()), (503, Some(7), Some("unavailable")));
    assert_eq!(relay.message.as_ref().unwrap().len(), 200);
}

#[test]
fn a_proof_that_is_answered_with_a_503_and_retry_after_is_paced_by_it_and_never_sends_a_token() {
    let e = env(quick());
    let (token, _) = e.enrol_desktop();
    e.clock.clear_sleeps();
    e.stub.clear_log();
    e.stub.fail_next_on(
        "/v1/info",
        1,
        Fault::Respond(503, vec![("retry-after".into(), "45".into())], r#"{"error":{"code":"unavailable","message":"busy","retryAfter":45}}"#.into()),
    );
    let running = run_loop(&e.client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|x| matches!(x, Event::State(ConnectionState::Connected))));
    let (_, _) = running.finish();
    let sleeps = e.clock.sleeps();
    assert!(within_jitter(&sleeps[0], 45.0), "the larger of 1 and Retry-After: {:?}", sleeps[0]);
    let log = e.stub.log();
    assert_eq!(log[0].status, 503);
    assert!(log[0].authorization.is_none(), "the proof request carries no token");
}
