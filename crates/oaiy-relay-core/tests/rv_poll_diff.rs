//! Reviewer's differential harness (not part of the crate's own suite): reads the JSON lines of `RV_POLL_IN` (cases written by the reviewer's own generator), runs
//! `oaiy_relay_core::poll::decide` / `decide_proof` on each, and writes one JSON line per case to `RV_POLL_OUT`. Skipped when the variables are not set.

use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::poll::{self, Answer, Counters, DecideInput, PollInfo, ProofResult};

fn counters(v: &Json) -> Counters {
    let n = |k: &str| v.get(k).and_then(Json::as_int).unwrap() as u32;
    Counters { n429: n("n429"), n_fail: n("nFail"), n_refused: n("nRefused"), n400: n("n400") }
}

fn f64_of(v: &Json) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn rv_poll_differential() {
    let (Ok(inp), Ok(outp)) = (std::env::var("RV_POLL_IN"), std::env::var("RV_POLL_OUT")) else {
        eprintln!("SKIPPED: RV_POLL_IN / RV_POLL_OUT not set");
        return;
    };
    let text = std::fs::read_to_string(inp).unwrap();
    let mut out = String::new();
    for line in text.lines() {
        let c = json::parse(line.as_bytes()).unwrap();
        let id = c.get_str("id").unwrap().to_string();
        let u = f64_of(c.get("u").unwrap());
        let state = counters(c.get("state").unwrap());
        let d = if let Some(p) = c.get("proof").and_then(Json::as_str) {
            let r = match p {
                "verified" => ProofResult::Verified,
                "none" => ProofResult::NoAnswer,
                _ => ProofResult::Invalid,
            };
            let asked = c.get("asked").and_then(Json::as_int).map(|a| a as u64);
            poll::decide_proof(state, r, u, 0, asked)
        } else {
            let info = c.get("info").unwrap();
            let info = PollInfo {
                poll_gap_ms: info.get("pollGapMs").and_then(Json::as_int).unwrap() as u64,
                fallback_s: info.get("fallbackS").and_then(Json::as_int).unwrap() as u64,
            };
            let status = c.get("status").and_then(Json::as_int).map(|s| s as u16);
            let headers: Vec<(String, String)> = c
                .get("headers")
                .and_then(Json::as_object)
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect())
                .unwrap_or_default();
            let body_text = c.get("bodyText").and_then(Json::as_str);
            let body = body_text.and_then(|t| json::parse(t.as_bytes()).ok());
            let input = DecideInput {
                counters: state,
                info,
                answer: Answer { status, headers: &headers, body: body.as_ref() },
                since: c.get("since").and_then(Json::as_int).unwrap() as u64,
                persisted: c.get("persisted").and_then(Json::as_bool).unwrap(),
                we_replaced: c.get("weReplaced").and_then(Json::as_bool).unwrap(),
                min_client_above_ours: c.get("minClientAboveOurs").and_then(Json::as_bool).unwrap(),
                now_epoch: c.get("nowEpoch").and_then(Json::as_int).map(|n| n as i64),
                u,
            };
            let d = poll::decide(&input);
            // the driver's own contract: what it persists is what assess() names.
            let adoption = poll::assess(status, body.as_ref(), input.since);
            match (&adoption, d.outcome) {
                (Some(a), poll::Outcome::Progress) => assert_eq!(a.since, d.since, "{id}"),
                (None, poll::Outcome::Progress) => panic!("{id}: progress with nothing to persist"),
                (Some(_), poll::Outcome::Failure) => {}
                (Some(_), other) => panic!("{id}: {other:?} with something to persist"),
                (None, _) => {}
            }
            d
        };
        let reports: Vec<String> = d.reports.iter().map(|r| format!("\"{}\"", r.as_str())).collect();
        out.push_str(&format!(
            "{{\"id\":\"{}\",\"outcome\":\"{}\",\"baseS\":{:?},\"pauseS\":{:?},\"state\":{{\"n429\":{},\"nFail\":{},\"nRefused\":{},\"n400\":{}}},\"action\":{},\"report\":[{}],\"since\":{}}}\n",
            id,
            d.outcome.as_str(),
            d.base_s,
            d.pause_s,
            d.counters.n429,
            d.counters.n_fail,
            d.counters.n_refused,
            d.counters.n400,
            d.action.map_or("null".to_string(), |a| format!("\"{}\"", a.as_str())),
            reports.join(","),
            d.since
        ));
    }
    std::fs::write(outp, out).unwrap();
}
