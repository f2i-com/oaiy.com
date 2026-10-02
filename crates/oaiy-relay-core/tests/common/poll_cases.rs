//! Runs a table in the shape of `poll-client.json` against `oaiy_relay_core::poll`: shared by the test of the committed table and the test of random tables.

#![allow(dead_code)]

use super::At;
use oaiy_relay_core::json::Json;
use oaiy_relay_core::poll::{self, Action, Answer, Counters, DecideInput, PollInfo, ProofDue, ProofResult};

/// The `state` object of a case as counters.
pub fn counters(v: &Json) -> Counters {
    Counters { n429: v.n("n429") as u32, n_fail: v.n("nFail") as u32, n_refused: v.n("nRefused") as u32, n400: v.n("n400") as u32 }
}

/// A number as `f64`.
pub fn float(v: &Json) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v:?}"))
}

fn check_decision(c: &Json, d: &poll::Decision) {
    let id = c.s("id");
    let want = c.at("expect");
    assert_eq!(d.outcome.as_str(), want.s("outcome"), "{id}: outcome");
    assert!((d.base_s - float(want.at("baseS"))).abs() < 1e-9, "{id}: baseS {} want {}", d.base_s, float(want.at("baseS")));
    assert!((d.pause_s - float(want.at("pauseS"))).abs() < 1e-6, "{id}: pauseS {} want {}", d.pause_s, float(want.at("pauseS")));
    assert_eq!(d.counters, counters(want.at("state")), "{id}: state after");
    let action = match want.at("action") {
        Json::Null => None,
        a => Some(a.as_str().unwrap()),
    };
    assert_eq!(d.action.map(Action::as_str), action, "{id}: action");
    let mut reports: Vec<&str> = d.reports.iter().map(|r| r.as_str()).collect();
    reports.sort_unstable();
    let mut wanted: Vec<&str> = want.at("report").as_array().unwrap().iter().map(|r| r.as_str().unwrap()).collect();
    wanted.sort_unstable();
    assert_eq!(reports, wanted, "{id}: reports");
    if let Some(since) = want.get("since") {
        assert_eq!(Some(d.since), since.as_u64(), "{id}: since");
    }
}

/// Runs every case of `doc` (all six members of each); returns how many were checked.
pub fn run_poll_cases(doc: &Json) -> usize {
    let mut checked = 0;
    for c in doc.at("cases").as_array().unwrap() {
        let id = c.s("id");
        if let Some(r) = c.get("replace") {
            assert_eq!(poll::replace_wait_ms(r.n("msSinceLastStart")), c.n("expect.waitMs"), "{id}");
            checked += 1;
            continue;
        }
        if let Some(p) = c.get("proofDue") {
            let got = poll::proof_due(&ProofDue {
                process_start: p.get("processStart").and_then(Json::as_bool).unwrap_or(false),
                network_changed: p.get("networkChanged").and_then(Json::as_bool).unwrap_or(false),
                longest_pause_s: p.n("longestPauseS"),
                seconds_since_proof: p.n("secondsSinceProof"),
            });
            assert_eq!(got, c.at("expect.due").as_bool().unwrap(), "{id}");
            checked += 1;
            continue;
        }
        let u = float(c.at("u"));
        let state = counters(c.at("state"));
        let since = c.get("since").map_or(0, |s| s.as_u64().unwrap());
        if let Some(p) = c.get("proof") {
            let result = match p.s("result") {
                "verified" => ProofResult::Verified,
                "none" => ProofResult::NoAnswer,
                "invalid" => ProofResult::Invalid,
                other => panic!("{id}: {other}"),
            };
            check_decision(c, &poll::decide_proof(state, result, u, since));
            checked += 1;
            continue;
        }
        let info = PollInfo { poll_gap_ms: c.n("info.pollGapMs"), fallback_s: c.n("info.fallbackS") };
        let response = c.at("response");
        let status = match response.at("status") {
            Json::Null => None,
            s => Some(s.as_u64().unwrap() as u16),
        };
        let headers: Vec<(String, String)> = response
            .get("headers")
            .and_then(Json::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.to_lowercase(), v.as_str().unwrap().to_string())).collect())
            .unwrap_or_default();
        let body = match response.get("body") {
            None | Some(Json::Null) => None,
            Some(b) => Some(b),
        };
        let input = DecideInput {
            counters: state,
            info,
            answer: Answer { status, headers: &headers, body },
            since,
            persisted: c.get("persisted").and_then(Json::as_bool).unwrap_or(true),
            we_replaced: c.get("weReplaced").and_then(Json::as_bool).unwrap_or(true),
            min_client_above_ours: c.get("minClientAboveOurs").and_then(Json::as_bool).unwrap_or(false),
            now_epoch: c.get("nowEpoch").map(|n| n.as_int().unwrap() as i64),
            u,
        };
        check_decision(c, &poll::decide(&input));
        // What the driver persists before it sends the next poll is what the decision adopts: the two agree on every case that has an answer.
        let adoption = poll::assess(status, body, since);
        let d = poll::decide(&DecideInput { persisted: true, ..input });
        match (&adoption, d.outcome) {
            (Some(a), poll::Outcome::Progress) => assert_eq!(a.since, d.since, "{id}"),
            (None, poll::Outcome::Progress) => panic!("{id}: progress with nothing to persist"),
            (Some(_), other) => panic!("{id}: {other:?} with something to persist"),
            (None, _) => {}
        }
        checked += 1;
    }
    checked
}
