//! Reviewer's corpus runners (F2, encoding): each test reads a corpus written by the reviewer's Python driver from the directory in `RV_DIR` (one input per line, as hex of
//! the raw bytes), runs this crate's function on every line and writes one result line per input next to it. The driver compares the results with PHP's `B64.php`, `Ids.php`,
//! `json_decode`, Python's `base64` and `json`, Node's `Buffer` and `JSON`, and `serde_json`. The tests are ignored (they are the drivers of `tools/run-differentials.ps1`) and fail loudly when `RV_DIR` is not set.

use std::fmt::Write as _;
use std::path::PathBuf;

use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::pairing::math::{self, Sas};
use oaiy_relay_core::{b64, ids, url};

fn dir() -> PathBuf {
    PathBuf::from(std::env::var_os("RV_DIR").expect("RV_DIR is not set: run this through tools/run-differentials.ps1"))
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        write!(s, "{x:02x}").unwrap();
    }
    s
}

fn lines(name: &str) -> (PathBuf, Vec<String>) {
    let d = dir();
    let text = std::fs::read_to_string(d.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    (d, text.lines().map(str::to_string).collect())
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_b64() {
    let (d, input) = lines("b64.in");
    let mut out = String::new();
    for l in &input {
        let raw = unhex(l);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        let dec = match b64::decode(s) {
            Ok(b) => format!("OK:{}", hex(&b)),
            Err(e) => format!("ERR:{e:?}"),
        };
        let e16 = b64::decode_exact::<16>(s).is_ok();
        let e32 = b64::decode_exact::<32>(s).is_ok();
        let e64 = b64::decode_exact::<64>(s).is_ok();
        // encode of the decoded value must give the text back (the single-spelling property)
        let round = match b64::decode(s) {
            Ok(b) => b64::encode(&b) == s,
            Err(_) => true,
        };
        writeln!(out, "{dec}\t{}{}{}\t{}", u8::from(e16), u8::from(e32), u8::from(e64), u8::from(round)).unwrap();
    }
    std::fs::write(d.join("b64.rust.out"), out).unwrap();
}

fn jerr(e: json::JsonError) -> String {
    format!("ERR:{e:?}")
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_json() {
    let (d, input) = lines("json.in");
    let mut out = String::new();
    for l in &input {
        let raw = unhex(l);
        let g = match json::parse(&raw) {
            Ok(v) => format!("OK:{}", hex(v.to_compact().as_bytes())),
            Err(e) => jerr(e),
        };
        let c = match json::parse_canonical(&raw) {
            Ok(v) => match v.to_canonical() {
                Ok(t) => format!("OK:{}", hex(t.as_bytes())),
                Err(e) => jerr(e),
            },
            Err(e) => jerr(e),
        };
        let serde = if serde_json::from_slice::<serde_json::Value>(&raw).is_ok() { "OK" } else { "ERR" };
        writeln!(out, "{g}\t{c}\t{serde}").unwrap();
    }
    std::fs::write(d.join("json.rust.out"), out).unwrap();
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_typed() {
    let (d, input) = lines("typed.in");
    let mut out = String::new();
    for l in &input {
        let raw = unhex(l);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        let norm = math::normalise(s).map_or("NONE".to_string(), |n| format!("OK:{}", n.as_str()));
        let parsed = match math::parse_typed_code(s) {
            Ok(secret) => format!("OK:{}", hex(secret.expose())),
            Err(e) => format!("ERR:{e}"),
        };
        writeln!(out, "{norm}\t{parsed}").unwrap();
    }
    std::fs::write(d.join("typed.rust.out"), out).unwrap();
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_sas_entry() {
    let (d, input) = lines("sas.in");
    let mut out = String::new();
    for l in &input {
        // "<12 expected chars>\t<hex of typed>"
        let (chars12, typed) = l.split_once('\t').unwrap();
        let raw = unhex(typed);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        let expected = Sas::from_parts([0; 8], chars12.to_string(), math::sas_check_char(chars12));
        writeln!(out, "{:?}", math::judge_sas_entry(&expected, s)).unwrap();
    }
    std::fs::write(d.join("sas.rust.out"), out).unwrap();
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_ids() {
    let (d, input) = lines("ids.in");
    let mut out = String::new();
    for l in &input {
        let (f, h) = l.split_once('\t').unwrap();
        let raw = unhex(h);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        let r = match f {
            "device" => ids::is_device_id(s),
            "provider" => ids::is_provider_id(s),
            "relay" => ids::is_relay_id(s),
            "principal" => ids::is_principal_id(s),
            "pid" => ids::is_pid(s),
            "epoch" => ids::is_epoch(s),
            "item" => ids::is_item_id(s),
            "app" => ids::is_app_id(s),
            "thumb" => ids::is_thumbprint(s),
            "grant" => ids::is_grant(s),
            "token" => ids::Token::parse(s).is_ok(),
            "jti" => ids::is_pairing_jti(s),
            _ => panic!("{f}"),
        };
        writeln!(out, "{}", u8::from(r)).unwrap();
    }
    std::fs::write(d.join("ids.rust.out"), out).unwrap();
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_name() {
    let (d, input) = lines("name.in");
    let mut out = String::new();
    for l in &input {
        let (max, h) = l.split_once('\t').unwrap();
        let raw = unhex(h);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        writeln!(out, "{}", hex(ids::clean_name(s, max.parse().unwrap()).as_bytes())).unwrap();
    }
    std::fs::write(d.join("name.rust.out"), out).unwrap();
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn corpus_url() {
    let (d, input) = lines("url.in");
    let mut out = String::new();
    for l in &input {
        let raw = unhex(l);
        let Ok(s) = std::str::from_utf8(&raw) else {
            out.push_str("SKIP\n");
            continue;
        };
        let a = url::RelayUrl::parse_with(s, false).map_or("ERR".to_string(), |u| u.origin());
        let b = url::RelayUrl::parse_with(s, true).map_or("ERR".to_string(), |u| u.origin());
        let p = url::percent_decode(s).map_or("ERR".to_string(), |t| hex(t.as_bytes()));
        writeln!(out, "{a}\t{b}\t{p}").unwrap();
    }
    std::fs::write(d.join("url.rust.out"), out).unwrap();
}

// A Json value is not used directly here; this keeps the import honest if the file is trimmed.
#[allow(dead_code)]
fn _json_type_check(_: &Json) {}
