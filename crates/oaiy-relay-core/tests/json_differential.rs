//! This crate's JSON parser and canonical writer against `serde_json`, an independent implementation, on every JSON file of the protocol package and on generated and
//! damaged documents. `serde_json` (members in a sorted map, compact output, only quote, backslash and control characters escaped, `/` and non-ASCII as they are) writes
//! exactly the protocol's canonical form for a document of integers, so it is the oracle for the writer; for the parser the property is that the two agree on what is
//! valid, except for the things this parser refuses on purpose (a repeated member name, and nesting past 64 levels, which `serde_json` takes to 128).

mod common;

use common::{protocol_dir, Rng};
use oaiy_relay_core::json::{self, Json, JsonError, Number};
use serde_json::Value;

fn to_serde(j: &Json) -> Value {
    match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Num(Number::Int(n)) => {
            if let Ok(v) = i64::try_from(*n) {
                Value::from(v)
            } else if let Ok(v) = u64::try_from(*n) {
                Value::from(v)
            } else {
                // Beyond 64 bits serde_json holds the value as a float; so does the document it parsed.
                serde_json::from_str(&n.to_string()).unwrap()
            }
        }
        Json::Num(Number::Big(t) | Number::Other(t)) => serde_json::from_str(t).unwrap(),
        Json::Str(s) => Value::String(s.clone()),
        Json::Arr(a) => Value::Array(a.iter().map(to_serde).collect()),
        Json::Obj(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), to_serde(v))).collect()),
    }
}

fn json_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "rust-check") {
                continue;
            }
            json_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "json") {
            out.push(p);
        }
    }
}

#[test]
fn every_json_file_of_the_protocol_package_parses_the_same_in_both() {
    let mut files = Vec::new();
    json_files(&protocol_dir(), &mut files);
    assert!(files.len() > 70, "{} files", files.len());
    for f in files {
        let bytes = std::fs::read(&f).unwrap();
        let mine = json::parse(&bytes).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let theirs: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(to_serde(&mine), theirs, "{}", f.display());
    }
}

/// A random value of at most `depth` more levels, with integers inside the canonical range.
fn gen(rng: &mut Rng, depth: u32) -> Json {
    let kind = rng.below(if depth == 0 { 4 } else { 7 });
    match kind {
        0 => Json::Null,
        1 => Json::Bool(rng.one_in(2)),
        2 => Json::int(match rng.below(6) {
            0 => 0i128,
            1 => i128::from(rng.next() as i64),
            2 => i128::from(rng.next()),
            3 => json::CANONICAL_MIN,
            4 => json::CANONICAL_MAX,
            _ => i128::from(rng.below(1000)) - 500,
        }),
        3 => Json::Str(gen_string(rng)),
        4 | 5 => Json::Arr((0..rng.below(5)).map(|_| gen(rng, depth - 1)).collect()),
        _ => {
            let mut members: Vec<(String, Json)> = Vec::new();
            for _ in 0..rng.below(5) {
                let k = gen_string(rng);
                if !members.iter().any(|(e, _)| *e == k) {
                    members.push((k, gen(rng, depth - 1)));
                }
            }
            Json::Obj(members)
        }
    }
}

fn gen_string(rng: &mut Rng) -> String {
    const POOL: [&str; 22] = [
        "a",
        "B",
        "z",
        "_",
        "-",
        "/",
        "\\",
        "\"",
        "\u{0}",
        "\u{1f}",
        "\u{7f}",
        "\n",
        "\t",
        " ",
        "\u{e9}",
        "\u{65e5}",
        "\u{2028}",
        "\u{1f600}",
        "\u{10ffff}",
        "\u{ffff}",
        "9",
        "~",
    ];
    (0..rng.below(8)).map(|_| *rng.pick(&POOL)).collect()
}

/// A JSON text of `v` with random whitespace and, in strings, random choices of how a character is written (a `\u` escape where one is allowed, `\/`).
fn write_odd(rng: &mut Rng, v: &Json, out: &mut String) {
    let ws = |rng: &mut Rng, out: &mut String| {
        for _ in 0..rng.below(3) {
            out.push(*rng.pick(&[' ', '\t', '\n', '\r']));
        }
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(Number::Int(n)) => out.push_str(&n.to_string()),
        Json::Num(_) => unreachable!(),
        Json::Str(s) => write_odd_string(rng, s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ws(rng, out);
                write_odd(rng, e, out);
                ws(rng, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, e)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ws(rng, out);
                write_odd_string(rng, k, out);
                ws(rng, out);
                out.push(':');
                ws(rng, out);
                write_odd(rng, e, out);
                ws(rng, out);
            }
            out.push('}');
        }
    }
}

fn write_odd_string(rng: &mut Rng, s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '/' if rng.one_in(2) => out.push_str("\\/"),
            c if (c as u32) < 0x20 || rng.one_in(6) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&if rng.one_in(2) { format!("\\u{unit:04x}") } else { format!("\\u{unit:04X}") });
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[test]
fn generated_documents_canonicalise_exactly_as_serde_json_writes_them() {
    for seed in 0..400u64 {
        let mut rng = Rng(seed);
        let tree = gen(&mut rng, 4);
        let mut text = String::new();
        write_odd(&mut rng, &tree, &mut text);
        let mine = json::canonicalize(text.as_bytes()).unwrap_or_else(|e| panic!("seed {seed}: {e}: {text}"));
        let oracle = serde_json::to_string(&serde_json::from_str::<Value>(&text).unwrap_or_else(|e| panic!("seed {seed}: {e}: {text}"))).unwrap();
        assert_eq!(mine, oracle, "seed {seed}: {text}");
        // And what this crate wrote reads back as the same tree, in general mode as in canonical mode.
        assert_eq!(json::parse(mine.as_bytes()).unwrap().to_canonical().unwrap(), mine, "seed {seed}");
        assert_eq!(json::parse_canonical(mine.as_bytes()).unwrap().to_canonical().unwrap(), mine, "seed {seed}");
    }
}

#[test]
fn damaged_documents_are_judged_alike_except_for_what_this_parser_refuses_on_purpose() {
    let mut agreed = 0u32;
    for seed in 0..1500u64 {
        let mut rng = Rng(0x5eed_0000 + seed);
        let tree = gen(&mut rng, 3);
        let mut text = String::new();
        write_odd(&mut rng, &tree, &mut text);
        let mut bytes = text.into_bytes();
        for _ in 0..1 + rng.below(3) {
            if bytes.is_empty() {
                break;
            }
            let at = rng.below(bytes.len() as u64) as usize;
            match rng.below(4) {
                0 => bytes[at] = rng.next() as u8,
                1 => {
                    bytes.remove(at);
                }
                2 => bytes.insert(at, *rng.pick(b"{}[]\",:\\-0.eE tfn ux")),
                _ => bytes.truncate(at),
            }
        }
        let mine = json::parse(&bytes);
        let theirs = serde_json::from_slice::<Value>(&bytes);
        match (&mine, &theirs) {
            (Ok(m), Ok(t)) => {
                assert_eq!(&to_serde(m), t, "seed {seed}: {:?}", String::from_utf8_lossy(&bytes));
                agreed += 1;
            }
            (Err(_), Err(_)) => agreed += 1,
            (Err(JsonError::DuplicateKey), Ok(_)) => agreed += 1,
            // A number spelling beyond what an f64 holds (`1e999`): serde_json converts every non-integer to an f64 and refuses it, this parser keeps the spelling it was given
            // (a float is never read as a value here, only classified as "not an integer").
            (Ok(_), Err(e)) if e.to_string().starts_with("number out of range") => agreed += 1,
            // Anything else that differs is a bug in one of them.
            (Ok(_), Err(e)) => panic!("seed {seed}: accepted here, refused by serde_json ({e}): {:?}", String::from_utf8_lossy(&bytes)),
            (Err(e), Ok(_)) => panic!("seed {seed}: refused here ({e}), accepted by serde_json: {:?}", String::from_utf8_lossy(&bytes)),
        }
    }
    assert_eq!(agreed, 1500);
}
