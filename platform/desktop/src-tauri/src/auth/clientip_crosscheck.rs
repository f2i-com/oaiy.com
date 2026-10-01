//! The client-address rules (design 4.5.4) against an independent implementation.
//!
//! `scripts/clientip-reference.py` implements the same rules in Python, on `ipaddress` (which shares no code with
//! Rust's standard library), and writes cases with the answers it computes: an address (is it one, and which),
//! a network (is it one, and which), and a whole client derivation (a peer, `X-Forwarded-For` lines, the trusted
//! list, and who the client is, how its throttle bucket is keyed, whether a forwarded address was believed or the
//! header could not be used). This test reads such a file and fails on the first difference it finds, listing up to
//! twenty.
//!
//! - A committed golden file (`clientip_golden.json`, generated with seed 20260930) always runs.
//! - `OAIY_CLIENTIP_CASES=<file>` runs a bigger file the local script `scripts/check-exposure.mjs` makes (tens of
//!   thousands of cases, a new seed each time if it is told to). Without the variable that test passes with a
//!   note, so a plain `cargo test` does not need Python.
//!
//! A mistake here is an authentication bypass (a forged `X-Forwarded-For` believed as a client, a network that
//! trusts more than it says), which is why the rules are held to a second implementation and not only to their
//! own tests.

use std::net::IpAddr;

use serde_json::Value;

use super::clientip::{bucket_key, client_ip, parse_forwarded_ip, Cidr, TrustedProxies};

/// The golden cases: 1,800 made by the reference with `--seed 20260930`.
const GOLDEN: &str = include_str!("clientip_golden.json");

fn value_of(ip: IpAddr) -> (bool, u128) {
    match ip {
        IpAddr::V4(v4) => (true, u128::from(u32::from(v4))),
        IpAddr::V6(v6) => (false, u128::from(v6)),
    }
}

fn hex(text: &Value) -> u128 {
    u128::from_str_radix(text.as_str().expect("a hex string"), 16).expect("hex")
}

/// Check every case; the differences, as text.
pub(crate) fn check_cases(text: &str) -> (usize, Vec<String>) {
    let cases: Vec<Value> = serde_json::from_str(text).expect("the cases are JSON");
    let mut differences = Vec::new();
    for case in &cases {
        let kind = case["kind"].as_str().unwrap_or("");
        match kind {
            "ip" => {
                let text = case["text"].as_str().unwrap();
                let got = parse_forwarded_ip(text).map(value_of);
                let want = case["ok"]
                    .as_bool()
                    .unwrap()
                    .then(|| (case["v4"].as_bool().unwrap(), hex(&case["value"])));
                if got != want {
                    differences.push(format!("ip {text:?}: rust {got:?}, reference {want:?}"));
                }
            }
            "cidr" => {
                let text = case["text"].as_str().unwrap();
                let got = Cidr::parse(text).map(|c| c.parts());
                let want = case["ok"].as_bool().unwrap().then(|| {
                    (
                        case["v4"].as_bool().unwrap(),
                        hex(&case["value"]),
                        case["prefix"].as_u64().unwrap() as u8,
                    )
                });
                if got != want {
                    differences.push(format!("cidr {text:?}: rust {got:?}, reference {want:?}"));
                }
            }
            "client" => {
                let peer: IpAddr = case["peer"].as_str().unwrap().parse().unwrap();
                let lines: Vec<&str> = case["xff"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|l| l.as_str().unwrap())
                    .collect();
                let entries: Vec<&str> = case["trusted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|l| l.as_str().unwrap())
                    .collect();
                let (trusted, _rejected) = TrustedProxies::parse_list(&entries.join(","));
                let got = client_ip(peer, &lines, &trusted);
                let (v4, value) = value_of(got.ip);
                let want = (
                    case["v4"].as_bool().unwrap(),
                    hex(&case["value"]),
                    case["key"].as_str().unwrap().to_string(),
                    case["via"].as_bool().unwrap(),
                    case["fell"].as_bool().unwrap(),
                );
                let have = (v4, value, got.key.clone(), got.via_proxy, got.fell_back);
                if have != want || bucket_key(got.ip) != got.key {
                    differences.push(format!(
                        "client peer {peer} xff {lines:?} trusted {entries:?}: rust {have:?}, reference {want:?}"
                    ));
                }
            }
            other => differences.push(format!("unknown case kind {other:?}")),
        }
    }
    (cases.len(), differences)
}

fn report(what: &str, total: usize, differences: &[String]) {
    assert!(
        differences.is_empty(),
        "{what}: {} of {total} cases differ from the reference; the first {}:\n{}",
        differences.len(),
        differences.len().min(20),
        differences
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_client_address_rules_agree_with_the_independent_implementation_on_the_golden_cases() {
    let (total, differences) = check_cases(GOLDEN);
    assert!(total >= 1800, "the golden file holds {total} cases");
    report("golden", total, &differences);
}

#[test]
fn the_client_address_rules_agree_with_the_independent_implementation_on_a_generated_file() {
    let Some(path) = std::env::var_os("OAIY_CLIENTIP_CASES") else {
        eprintln!("OAIY_CLIENTIP_CASES is not set: the generated cross-check is run by scripts/check-exposure.mjs");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("the cases file is readable");
    let (total, differences) = check_cases(&text);
    let clients = text.matches("\"kind\":\"client\"").count();
    assert!(
        clients >= 20_000,
        "{clients} client derivations in {total} cases: the cross-check needs at least 20,000"
    );
    eprintln!(
        "clientip cross-check: {total} cases ({clients} client derivations), {} differences",
        differences.len()
    );
    report("generated", total, &differences);
}
