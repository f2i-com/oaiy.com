//! `fixtures/poll-client/poll-client.json` against this crate's `decide`: every one of its cases, all six members of each (outcome, base pause, pause, counters after, action,
//! reports, and `since` where the case has it), the constants the table says, and the table's own digests. The table is read independently by `verify_poll_client.py` and
//! `.mjs`; this is the third reading, and the one that ships. A client's rules that depart from the README by one number, one clamp or one counter fail here.

mod common;

use std::collections::BTreeSet;

use common::{float, hex, load, run_poll_cases, At};
use oaiy_crypto::kdf::sha256;
use oaiy_relay_core::json::Json;
use oaiy_relay_core::poll::{self, Action, Report};

/// The numbers README 5.1.1 states, written here from the README (and the same list the two readers hold): the table's `constants` block must be exactly these, and so must
/// this crate's own constants.
const EXPECTED: [(&str, u64); 13] = [
    ("replaceMinMs", 250),
    ("clampMin", 1),
    ("clampMax", 120),
    ("retryAfterBodyMax", 86_400),
    ("retryAfterDigitsMax", 6),
    ("backoff429Cap", 30),
    ("backoffFailureCap", 60),
    ("unreachableAfter", 3),
    ("inFlightDefectAfter", 5),
    ("refusedHoldDefaultS", 2),
    ("proofEveryS", 300),
    ("proofAfterPauseS", 60),
    ("pollTimeoutExtraS", 10),
];

#[test]
fn the_tables_constants_are_the_readmes_and_this_crates() {
    let doc = load("fixtures/poll-client/poll-client.json");
    let k = doc.at("constants");
    for (name, want) in EXPECTED {
        assert_eq!(k.n(name), want, "table constant {name}");
    }
    assert_eq!(float(k.at("jitter")), 0.2);
    assert_eq!(k.as_object().unwrap().len(), EXPECTED.len() + 1, "the table has a constant this test does not know");
    // This crate's own copies.
    assert_eq!(poll::REPLACE_MIN_MS, 250);
    assert_eq!((poll::CLAMP_MIN_S, poll::CLAMP_MAX_S), (1, 120));
    assert_eq!(poll::RETRY_AFTER_BODY_MAX, 86_400);
    assert_eq!(poll::RETRY_AFTER_DIGITS_MAX, 6);
    assert_eq!(poll::JITTER, 0.2);
    assert_eq!((poll::BACKOFF_429_CAP_S, poll::BACKOFF_FAILURE_CAP_S), (30, 60));
    assert_eq!((poll::UNREACHABLE_AFTER, poll::IN_FLIGHT_DEFECT_AFTER), (3, 5));
    assert_eq!(poll::REFUSED_HOLD_DEFAULT_S, 2);
    assert_eq!((poll::PROOF_EVERY_S, poll::PROOF_AFTER_PAUSE_S), (300, 60));
    assert_eq!(poll::POLL_TIMEOUT_EXTRA_S, 10);
}

#[test]
fn the_tables_own_digests_say_no_case_is_missing_relabelled_or_moved() {
    let doc = load("fixtures/poll-client/poll-client.json");
    let cases = doc.at("cases").as_array().unwrap();
    assert_eq!(doc.n("caseCount") as usize, cases.len());
    assert_eq!(cases.len(), 162, "a case went missing or was added: update this number with the table");
    let ids: Vec<&str> = cases.iter().map(|c| c.s("id")).collect();
    let unique: BTreeSet<&str> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "ids are unique");
    let sorted: Vec<&str> = unique.iter().copied().collect();
    assert_eq!(hex(&sha256(sorted.join("\n").as_bytes())), doc.s("idsSha256"));
    let layout: Vec<String> = cases.iter().map(|c| format!("{}|{}", c.s("id"), c.s("rule"))).collect();
    assert_eq!(hex(&sha256(layout.join("\n").as_bytes())), doc.s("layoutSha256"));
    // Every rule of 5.1.1 that a case can check has a case.
    let rules: BTreeSet<&str> = cases.iter().map(|c| c.s("rule")).collect();
    for rule in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"] {
        assert!(rules.contains(rule), "no case for {rule}");
    }
}

#[test]
fn every_case_of_the_table() {
    let doc = load("fixtures/poll-client/poll-client.json");
    assert_eq!(run_poll_cases(&doc), 162);
}

#[test]
fn a_float_a_boolean_or_a_minus_zero_is_not_an_integer_in_an_answer() {
    // The README's "integer" is a spelling: the rules read `1.0`, `true` and `-0` as no integer at all (Interpretation 23 for `-0`).
    let body = |cursor: &str| oaiy_relay_core::json::parse(format!(r#"{{"epoch":"eVp54C0-EJY","cursor":{cursor},"items":[]}}"#).as_bytes()).unwrap();
    for ok in ["0", "7", "9007199254740991"] {
        assert!(poll::valid_200(Some(&body(ok))), "{ok}");
    }
    for bad in ["1.0", "1e0", "-0", "true", "\"5\"", "null", "-1", "9007199254740992", "123456789012345678901234567890123456789012345"] {
        assert!(!poll::valid_200(Some(&body(bad))), "{bad}");
    }
    assert!(!poll::valid_200(Some(&Json::Null)));
    assert!(!poll::valid_200(None));
}

#[test]
fn the_reports_and_actions_are_exactly_the_readmes_words() {
    let reports: Vec<&str> =
        [Report::Unreachable, Report::InFlightDefect, Report::DuplicateCredential, Report::InvalidRequest, Report::StorageFailure]
            .iter()
            .map(|r| r.as_str())
            .collect();
    assert_eq!(reports, ["unreachable", "in_flight_defect", "duplicate_credential", "invalid_request", "storage_failure"]);
    let actions: Vec<&str> = [
        Action::ForgetCredential,
        Action::RefreshOrReenrol,
        Action::UpdateClient,
        Action::ReportDefect,
        Action::ClearEpoch,
        Action::CancelOwnPolls,
        Action::ReportRelayChanged,
    ]
    .iter()
    .map(|a| a.as_str())
    .collect();
    assert_eq!(
        actions,
        ["forget_credential", "refresh_or_reenrol", "update_client", "report_defect", "clear_epoch", "cancel_own_polls", "report_relay_changed"]
    );
}
