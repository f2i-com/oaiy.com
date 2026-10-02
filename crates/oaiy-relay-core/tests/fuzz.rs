//! Fuzzing of every decoder that reads bytes a stranger chose (a relay's answers, a scanned code, what a phone typed, a sealed value, an offer, a response): none may panic, loop
//! or allocate without a bound, whatever it is given, and each must keep the promises its output makes (a value that decodes encodes back to the text it came from; a canonical form is
//! its own canonical form; a pause is within its limits).
//!
//! The generator is seeded (SplitMix64) and the seed, the target, the round and the input are printed when a round fails, so a finding is a reproduction. Each target is given
//! random bytes, random printable text, random multi-byte text, and the valid corpus of the protocol package damaged by flips, insertions, deletions, truncations, duplications, splices from
//! other corpus members, and runs of brackets (nesting far beyond the parser's limit). `OAIY_FUZZ_ROUNDS` raises the number of rounds per target (default 2000).

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

use common::{hex, load, At, Rng};
use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::admission::{Bearer, MobileAdmission, MobileExpect, PluginAdmission, PluginRequest};
use oaiy_relay_core::client::loopback::LoopbackHttp;
use oaiy_relay_core::client::{Cancel, Health, HttpClient, HttpRequest, Item, Method, RelayProfile};
use oaiy_relay_core::enrol::EnrolmentKey;
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::info::{self, Info};
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::keys::{Signer, VerifyKey, X25519Secret};
use oaiy_relay_core::pairing::math::{self, PairingKey, SasEntry};
use oaiy_relay_core::pairing::offer::Offer;
use oaiy_relay_core::pairing::response::Response;
use oaiy_relay_core::pairing::{PairingInput, PairingTarget};
use oaiy_relay_core::poll::{self, Answer, Counters, DecideInput, PollInfo};
use oaiy_relay_core::ring::{self, RingBody};
use oaiy_relay_core::sealed::{self, Container};
use oaiy_relay_core::url::{self, RelayUrl};
use oaiy_relay_core::{b64, rotation, ticket};

fn rounds() -> u64 {
    std::env::var("OAIY_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// A damaged copy of `base`.
fn mutate(rng: &mut Rng, base: &[u8], donors: &[Vec<u8>]) -> Vec<u8> {
    const INTERESTING: [u8; 14] = [b'"', b'\\', b'{', b'}', b'[', b']', 0, 0xff, 0xed, 0xc0, b'-', b'=', b' ', b'\n'];
    let mut v = base.to_vec();
    for _ in 0..=rng.below(4) {
        let len = v.len() as u64;
        match rng.below(10) {
            0 if len > 0 => {
                let i = rng.below(len) as usize;
                v[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(len + 1) as usize;
                v.insert(i, rng.next() as u8);
            }
            2 if len > 0 => {
                v.remove(rng.below(len) as usize);
            }
            3 => v.truncate(rng.below(len + 1) as usize),
            4 if len > 0 => {
                let (a, b) = (rng.below(len) as usize, rng.below(len) as usize);
                let (a, b) = (a.min(b), a.max(b));
                let chunk = v[a..b].to_vec();
                let at = rng.below(len + 1) as usize;
                for (k, byte) in chunk.into_iter().enumerate() {
                    v.insert(at + k, byte);
                }
            }
            5 if !donors.is_empty() => {
                let d = rng.pick(donors);
                if !d.is_empty() {
                    let a = rng.below(d.len() as u64) as usize;
                    let b = a + rng.below((d.len() - a) as u64 + 1) as usize;
                    let at = rng.below(len + 1) as usize;
                    for (k, byte) in d[a..b].iter().enumerate() {
                        v.insert(at + k, *byte);
                    }
                }
            }
            6 if len > 0 => {
                let i = rng.below(len) as usize;
                v[i] = *rng.pick(&INTERESTING);
            }
            7 => {
                let c = *rng.pick(b"[{(\"");
                let n = rng.below(200) as usize + 1;
                let at = rng.below(len + 1) as usize;
                for k in 0..n {
                    v.insert(at + k, c);
                }
            }
            8 if len > 1 => {
                let (a, b) = (rng.below(len) as usize, rng.below(len) as usize);
                v.swap(a, b);
            }
            _ => {}
        }
        if v.len() > 1 << 16 {
            v.truncate(1 << 16);
        }
    }
    v
}

fn random_input(rng: &mut Rng) -> Vec<u8> {
    match rng.below(4) {
        0 => {
            let n = rng.below(300) as usize;
            rng.bytes(n)
        }
        1 => (0..rng.below(300)).map(|_| 0x20 + rng.below(0x5f) as u8).collect(),
        2 => {
            // Multi-byte text, with surrogates' encodings and noncharacters among it.
            let mut s = String::new();
            for _ in 0..rng.below(80) {
                let c = *rng.pick(&['a', 'é', 'ß', '€', '𝄞', '\u{FFFE}', '\u{10FFFF}', '\u{0}', '\n', '"', '\\', '/', 'Z', '-', '_', '=']);
                s.push(c);
            }
            s.into_bytes()
        }
        _ => {
            let mut v = b"{\"a\":".to_vec();
            for _ in 0..rng.below(400) {
                const PIECES: [&[u8]; 8] = [b"[", b"{\"k\":", b"\"", b"\\u", b"\\ud800", b"1e999", b"-0", b"123456789012345678901234567890"];
                v.extend_from_slice(PIECES[rng.below(PIECES.len() as u64) as usize]);
            }
            v
        }
    }
}

/// Runs `f` on the corpus and on `rounds()` inputs; a panic (an assertion included) fails with what is needed to reproduce it.
fn fuzz(name: &str, seed: u64, corpus: &[Vec<u8>], f: impl Fn(&[u8])) {
    let mut rng = Rng(seed);
    let guard = |round: u64, input: &[u8]| {
        if catch_unwind(AssertUnwindSafe(|| f(input))).is_err() {
            let shown = if input.len() > 600 { &input[..600] } else { input };
            panic!("{name}: seed {seed}, round {round}, {} bytes, hex {}, text {:?}", input.len(), hex(shown), text(shown));
        }
    };
    for (i, c) in corpus.iter().enumerate() {
        guard(u64::MAX - i as u64, c);
    }
    for round in 0..rounds() {
        let input = if corpus.is_empty() || rng.one_in(4) {
            random_input(&mut rng)
        } else {
            {
                let base = rng_pick(&mut rng, corpus);
                mutate(&mut rng, &base, corpus)
            }
        };
        guard(round, &input);
    }
}

fn rng_pick(rng: &mut Rng, corpus: &[Vec<u8>]) -> Vec<u8> {
    rng.pick(corpus).clone()
}

fn s(t: &str) -> Vec<u8> {
    t.as_bytes().to_vec()
}

struct Corpus {
    json: Vec<Vec<u8>>,
    typed: Vec<Vec<u8>>,
    pairing_uri: Vec<Vec<u8>>,
    enrol: Vec<Vec<u8>>,
    token: Vec<Vec<u8>>,
    container: Vec<Vec<u8>>,
    sealed: Vec<Vec<u8>>,
    ticket: Vec<Vec<u8>>,
    rotation: Vec<Vec<u8>>,
    ring: Vec<Vec<u8>>,
    offer: Vec<Vec<u8>>,
    response: Vec<Vec<u8>>,
    info: Vec<Vec<u8>>,
    b64: Vec<Vec<u8>>,
    mobile_admissions: Vec<(Vec<u8>, String, String, String)>,
    plugin_admissions: Vec<(Vec<u8>, PluginRequest)>,
    keys: Vec<(String, String)>,
}

fn corpus() -> Corpus {
    use oaiy_relay_core::admission::Transport;
    let v = load("vectors.json");
    let ceremony = load("fixtures/pairing-ceremony.json");
    let sealed_fixture = load("fixtures/sealed-token.json");
    let admissions = load("fixtures/aokie/admission.json");
    let mut json = vec![
        s(r#"{"a":1,"b":[true,false,null,"x"],"c":{"d":-5}}"#),
        s(r#"{"b":1,"a":2,"s":"é𝄞\n"}"#),
        s("[1,2,3]"),
        s("18446744073709551615"),
        s("-9223372036854775808"),
        s(r#""a\u0000b""#),
        s(r#"{"a":1,"a":2}"#),
        s("1e3"),
        s("-0"),
        s(r#""\ud800""#),
    ];
    let mut mobile = Vec::new();
    let mut plugin = Vec::new();
    for case in admissions.at("cases").as_array().unwrap() {
        let (req, answer) = (case.at("request.body"), case.at("response.body"));
        let transports = |t: Option<&Json>| {
            t.map(|t| {
                t.as_array().unwrap().iter().map(|x| if x.as_str() == Some("relay") { Transport::Relay } else { Transport::RelayPoll }).collect()
            })
        };
        if case.s("role") == "plugin" {
            plugin.push((
                answer.to_compact().into_bytes(),
                PluginRequest {
                    app_id: req.get_str("appId").unwrap().into(),
                    plugin_id: req.get_str("pluginId").unwrap().into(),
                    display_name: req.get_str("displayName").map(str::to_string),
                    endpoint: VerifyKey::from_b64u(req.at("endpointPublicKey.publicKey").as_str().unwrap()).unwrap(),
                    approved_peers: req
                        .get("approvedPeerKeyThumbprints")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| t.as_str().unwrap().to_string())
                        .collect(),
                    revision: req.n("peerRosterRevision"),
                    transports: transports(req.get("supportedTransports")),
                },
            ));
        } else {
            mobile.push((
                answer.to_compact().into_bytes(),
                req.s("appId").to_string(),
                req.s("deviceId").to_string(),
                req.s("holderKeyThumbprint").to_string(),
            ));
        }
        json.push(answer.to_compact().into_bytes());
    }
    let response_text = {
        let r = ceremony.at("steps.2.request.body.response");
        r.as_str().map(str::to_string).unwrap_or_else(|| r.to_compact())
    };
    json.push(s(&response_text));
    json.push(s(v.s("A3.expected.offerText")));
    let thumbs = [v.s("keys.ed25519Public.relay.thumbprint").to_string(), v.s("keys.ed25519Public.phone.thumbprint").to_string()];
    Corpus {
        json,
        typed: vec![
            s(v.s("A3.expected.typedCode")),
            s("0000-0000-0000-0000-0000-0000-00"),
            s("zzzz-zzzz-zzzz-zzzz-zzzz-zzzz-zz"),
            s("7ZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZ"),
        ],
        pairing_uri: vec![s(v.s("A3.expected.pairingUri"))],
        enrol: vec![s(v.s("A7.expected.uri"))],
        token: vec![s(v.s("A2.expected.token")), s(v.s("A4.expected.token"))],
        container: vec![s(v.s("A8.expected.container"))],
        sealed: vec![s(sealed_fixture.s("opens.0.sealedToken")), s(v.s("A4.expected.token"))],
        ticket: vec![s(v.s("A9.expected.ticket"))],
        rotation: vec![s(v.s("A10.expected.statementText"))],
        ring: vec![s(v.s("A11.expected.bodyText"))],
        offer: vec![s(v.s("A3.expected.offerText"))],
        response: vec![s(&response_text)],
        info: vec![s(v.s("A6b.expected.bodyText"))],
        b64: vec![
            s(v.s("A3.expected.secretB64u")),
            s(v.s("A3.expected.offerMac")),
            s(v.s("A3.expected.responseSignature")),
            s("Zg"),
            s("Zm9v"),
            s("Zm9vYg"),
        ],
        mobile_admissions: mobile,
        plugin_admissions: plugin,
        keys: thumbs.iter().map(|t| (t.clone(), t.clone())).collect(),
    }
}

// ------------------------------------------------------------------------------------------------------------------------- JSON and the encodings

#[test]
fn json_in_every_mode_never_panics_and_what_it_accepts_is_stable() {
    let c = corpus();
    fuzz("json", 0xA1, &c.json, |b| {
        if let Ok(v) = json::parse(b) {
            // What parses prints as compact text that parses to the same value.
            let again = json::parse(v.to_compact().as_bytes()).expect("compact output parses");
            assert_eq!(v, again);
        }
        match json::canonicalize(b) {
            Ok(canon) => {
                // A canonical text is its own canonical form, and is what `parse_canonical` accepts.
                assert_eq!(json::canonicalize(canon.as_bytes()).expect("canonical output canonicalizes"), canon);
                assert!(json::parse_canonical(canon.as_bytes()).is_ok());
                let direct = json::parse_canonical(b).and_then(|v| v.to_canonical());
                assert_eq!(direct.ok(), Some(canon));
            }
            Err(_) => {
                // The canonical mode is stricter than the plain one, never looser.
                if let Ok(v) = json::parse_canonical(b) {
                    assert!(json::parse(b).is_ok(), "{v:?}");
                }
            }
        }
    });
}

#[test]
fn base64url_decodes_only_what_it_encodes_back_to_the_same_text() {
    let c = corpus();
    fuzz("base64url", 0xB2, &c.b64, |b| {
        let t = text(b);
        if let Ok(bytes) = b64::decode(&t) {
            assert_eq!(b64::encode(&bytes), t, "no padding, no unused bits, no other alphabet");
            assert_eq!(t.len(), b64::encoded_len(bytes.len()));
        }
        let _ = b64::decode_exact::<32>(&t);
        let _ = b64::decode_exact::<64>(&t);
        let _ = b64::decode_exact::<16>(&t);
    });
}

#[test]
fn urls_and_percent_escapes_never_panic_and_a_url_that_parses_prints_back_the_same() {
    fuzz("urls", 0xC3, &[s("https://relay.example.com"), s("https://relay.example.com:8443"), s("http://127.0.0.1:9999"), s("HTTPS://A.B")], |b| {
        let t = text(b);
        for lax in [false, true] {
            if let Ok(u) = RelayUrl::parse_with(&t, lax) {
                let again = RelayUrl::parse_with(&u.origin(), lax).expect("an origin parses");
                assert_eq!(u, again);
                assert!(lax || u.is_https());
            }
        }
        if let Ok(d) = url::percent_decode(&t) {
            let _ = url::percent_encode(&d);
        }
    });
}

// ------------------------------------------------------------------------------------------------------------------------- what a person types or scans

#[test]
fn typed_codes_pairing_keys_enrolment_keys_and_sas_entries_never_panic_and_a_code_that_parses_is_the_code() {
    let c = corpus();
    fuzz("typed code", 0xD4, &c.typed, |b| {
        let t = text(b);
        if let Ok(secret) = math::parse_typed_code(&t) {
            assert_eq!(math::normalise(&math::typed_code(secret.expose())), math::normalise(&t), "a code that reads is the code it names");
        }
        let _ = math::normalise(&t);
    });
    fuzz("pairing key", 0xD5, &c.pairing_uri, |b| {
        let t = text(b);
        let _ = PairingKey::parse(&t);
        let _ = PairingTarget::from_input(PairingInput::Key(&t));
        let _ = PairingTarget::from_input(PairingInput::Typed { code: &t, host: &t });
    });
    fuzz("enrolment key", 0xD6, &c.enrol, |b| {
        let _ = EnrolmentKey::parse(&text(b));
    });
    let sas = math::sas(&[1; 32], &[2; 32], &[3; 32], &[4; 16]).unwrap();
    let shown = sas.display();
    fuzz("sas entry", 0xD7, &[s(&shown), s(&shown.to_lowercase()), s("0000-0000-0000-0")], |b| {
        let t = text(b);
        let judged = math::judge_sas_entry(&sas, &t);
        assert_eq!(judged == SasEntry::Right, math::normalise(&t).is_some_and(|n| n == format!("{}{}", sas.chars12, sas.check)), "{t:?}");
    });
}

#[test]
fn tokens_bearers_and_sealed_values_never_panic_and_never_open_for_a_stranger() {
    let c = corpus();
    let stranger = X25519Secret::generate().unwrap();
    fuzz("token", 0xE1, &c.token, |b| {
        let t = text(b);
        let _ = Token::parse(&t);
        let _ = Bearer::parse(&t);
    });
    fuzz("sealed token", 0xE2, &c.sealed, |b| {
        assert!(sealed::open_token(&stranger, &text(b)).is_err(), "no sealed value opens for a key it was not sealed to");
        let _ = stranger.open_sealed(b);
    });
    fuzz("container", 0xE3, &c.container, |b| {
        let _ = Container::parse(b);
        assert!(sealed::open_container(&stranger, &text(b)).is_err());
    });
}

// ------------------------------------------------------------------------------------------------------------------------- the relay's and the peers' documents

#[test]
fn signed_documents_never_panic_and_a_stranger_signs_nothing_that_verifies() {
    let c = corpus();
    let key = Signer::generate().unwrap().verify_key();
    fuzz("ticket", 0xF1, &c.ticket, |b| {
        assert!(ticket::verify_signature(&text(b), &key).is_err());
    });
    fuzz("rotation", 0xF2, &c.rotation, |b| {
        assert!(rotation::verify(&key, 0, 1_790_000_000, &text(b), &text(b)).is_err());
    });
    fuzz("ring", 0xF3, &c.ring, |b| {
        let t = text(b);
        let _ = RingBody::parse(&t);
        assert!(ring::verify(&key, &t, &t).is_err());
    });
    fuzz("info", 0xF4, &c.info, |b| {
        let _ = Info::parse(b);
        let t = text(b);
        assert!(info::verify_proof(b, b, &t, &t, &c.keys[0].0).is_err());
        let _ = info::parse_time_header(&t);
    });
}

#[test]
fn offers_responses_and_admissions_never_panic_and_nothing_forged_verifies() {
    let c = corpus();
    let mac_key = Secret::new([9u8; 32]);
    fuzz("offer", 0x101, &c.offer, |b| {
        let t = text(b);
        let _ = Offer::parse(&t);
        assert!(Offer::verify(&t, &t, &mac_key).is_err(), "a MAC that is not the MAC of the text");
    });
    fuzz("response", 0x102, &c.response, |b| {
        let _ = Response::parse(&text(b));
    });
    let t0 = 1_790_000_000;
    for (n, (body, app, device, holder)) in c.mobile_admissions.iter().enumerate() {
        let expect = MobileExpect { app_id: app, device_id: device, holder_thumbprint: holder };
        fuzz(&format!("mobile admission {n}"), 0x103 + n as u64, std::slice::from_ref(body), |b| {
            for lax in [false, true] {
                let _ = MobileAdmission::parse_with(b, &expect, t0, lax);
            }
        });
    }
    for (n, (body, request)) in c.plugin_admissions.iter().enumerate() {
        fuzz(&format!("plugin admission {n}"), 0x113 + n as u64, std::slice::from_ref(body), |b| {
            for lax in [false, true] {
                let _ = PluginAdmission::parse_with(b, request, t0, lax);
            }
        });
    }
    fuzz(
        "health, profile, item",
        0x123,
        &[s(r#"{"ok":true,"status":"ok"}"#), s(r#"{"seq":1,"id":"a","lane":"cmd","from":"x","body":"","hdr":"{}"}"#)],
        |b| {
            let _ = Health::parse(b);
            let _ = RelayProfile::from_json(b);
            if let Ok(v) = json::parse(b) {
                let _ = Item::from_json(&v);
            }
        },
    );
}

// ------------------------------------------------------------------------------------------------------------------------- the poll rules

fn random_body(rng: &mut Rng) -> Option<Json> {
    let doc = match rng.below(4) {
        0 => return None,
        1 => json::parse(&random_input(rng)).ok(),
        _ => {
            let items: Vec<String> = (0..rng.below(5))
                .map(|_| {
                    let seq = match rng.below(5) {
                        0 => "-1".to_string(),
                        1 => "1e2".to_string(),
                        2 => "18446744073709551616".to_string(),
                        3 => "\"7\"".to_string(),
                        _ => rng.below(20).to_string(),
                    };
                    format!("{{\"seq\":{seq}}}")
                })
                .collect();
            let epoch = *rng.pick(&["abcdefghijk", "abcdefghij", "abcdefghijkl", "abcdefghij=", ""]);
            let cursor = *rng.pick(&["0", "5", "9007199254740991", "9007199254740992", "-1", "1.5", "null"]);
            let extra = *rng.pick(&[
                "",
                ",\"reset\":true",
                ",\"reset\":false",
                ",\"hold\":{\"superseded\":true}",
                ",\"hold\":{\"refused\":true,\"retryAfter\":3}",
                ",\"error\":{\"code\":\"x\",\"retryAfter\":7}",
            ]);
            json::parse(format!("{{\"items\":[{}],\"epoch\":\"{epoch}\",\"cursor\":{cursor}{extra}}}", items.join(",")).as_bytes()).ok()
        }
    };
    doc
}

#[test]
fn the_poll_rules_never_panic_and_keep_every_pause_within_its_limits() {
    let mut rng = Rng(0x201);
    let statuses: [Option<u16>; 14] = [
        None,
        Some(200),
        Some(200),
        Some(200),
        Some(400),
        Some(401),
        Some(403),
        Some(408),
        Some(426),
        Some(429),
        Some(429),
        Some(500),
        Some(302),
        Some(0),
    ];
    let header_values = [
        "0",
        "1",
        "120",
        "121",
        "999999",
        "9999999",
        "-1",
        "1e3",
        " 5 ",
        "Sun, 06 Nov 1994 08:49:37 GMT",
        "Sun, 06 Nov 1994 08:49:37 PST",
        "",
        "\u{e9}",
    ];
    for round in 0..rounds() * 10 {
        let status = *rng.pick(&statuses);
        let mut headers: Vec<(String, String)> = Vec::new();
        for name in ["retry-after", "date", "x-oaiy-time"] {
            if rng.one_in(2) {
                headers.push((name.to_string(), (*rng.pick(&header_values)).to_string()));
            }
        }
        let body = random_body(&mut rng);
        let since = *rng.pick(&[0u64, 1, 5, 9_007_199_254_740_991, u64::MAX]);
        let counters =
            Counters { n429: rng.below(40) as u32, n_fail: rng.below(40) as u32, n_refused: rng.below(40) as u32, n400: rng.below(3) as u32 };
        let info = PollInfo { poll_gap_ms: rng.below(10_000), fallback_s: rng.below(120) };
        let answer = Answer { status, headers: &headers, body: body.as_ref() };
        let result = catch_unwind(AssertUnwindSafe(|| {
            let adoption = poll::assess(status, body.as_ref(), since);
            let persisted = rng.below(4) != 0;
            let u = [0.0, 0.5, 0.999_999, 1.0, -1.0, f64::NAN, f64::INFINITY][rng.below(7) as usize];
            let now_epoch = if rng.one_in(2) { Some(*rng.pick(&[0i64, 784_111_777, i64::MAX, i64::MIN])) } else { None };
            let d = poll::decide(&DecideInput {
                counters,
                info,
                answer,
                since,
                persisted,
                we_replaced: rng.one_in(2),
                min_client_above_ours: rng.one_in(2),
                now_epoch,
                u,
            });
            assert!(d.base_s.is_finite() && (0.0..=120.0).contains(&d.base_s), "base {}", d.base_s);
            assert!(d.pause_s >= d.base_s && d.pause_s <= d.base_s * 1.2 + 1e-9, "pause {} of base {}", d.pause_s, d.base_s);
            assert!(
                d.counters.n429 <= counters.n429 + 1
                    && d.counters.n_fail <= counters.n_fail + 1
                    && d.counters.n_refused <= counters.n_refused + 1
                    && d.counters.n400 <= counters.n400 + 1
            );
            if let Some(a) = &adoption {
                assert!(a.since <= poll::MAX_SEQ || a.since == since);
                assert!(a.accepted.windows(2).all(|w| w[0] < w[1]));
            }
            let _ = poll::retry_after(&answer, now_epoch);
        }));
        assert!(
            result.is_ok(),
            "poll rules: round {round}: status {status:?}, headers {headers:?}, body {body:?}, since {since}, counters {counters:?}"
        );
    }
    fuzz("http date", 0x202, &[s("Sun, 06 Nov 1994 08:49:37 GMT"), s("Thu, 01 Jan 1970 00:00:00 GMT")], |b| {
        let _ = poll::parse_http_date(&text(b));
    });
}

// ------------------------------------------------------------------------------------------------------------------------- the transport

#[test]
fn the_loopback_client_survives_a_server_that_answers_with_anything() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seeds: Vec<Vec<u8>> = vec![
        s("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}"),
        s("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n"),
        s("HTTP/1.0 200 OK\r\n\r\nuntil the connection closes"),
        s("HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n"),
        s("HTTP/1.1 301 Moved\r\nLocation: http://127.0.0.1:1/\r\nContent-Length: 0\r\n\r\n"),
    ];
    let n = (rounds() / 8).max(60);
    let server_seeds = seeds.clone();
    let server = std::thread::spawn(move || {
        let mut rng = Rng(0x301);
        for _ in 0..n {
            let Ok((mut stream, _)) = listener.accept() else { return };
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buf = [0u8; 4096];
            let mut got = Vec::new();
            while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => got.extend_from_slice(&buf[..k]),
                }
            }
            let reply = if rng.one_in(5) {
                random_input(&mut rng)
            } else {
                {
                    let base = rng.pick(&server_seeds).clone();
                    mutate(&mut rng, &base, &server_seeds)
                }
            };
            let _ = stream.write_all(&reply);
            // Closes: a response that promised more than it sent ends here.
        }
    });
    let cap = 4096;
    for i in 0..n {
        let request = HttpRequest {
            method: Method::Get,
            url: format!("http://{addr}/v1/poll?since=0"),
            headers: vec![("Accept".into(), "application/json".into())],
            body: None,
            timeout: Duration::from_secs(3),
            max_response_bytes: cap,
            cancel: Cancel::new(),
        };
        let result = catch_unwind(AssertUnwindSafe(|| LoopbackHttp.send(&request)));
        match result {
            Ok(Ok(response)) => assert!(response.body.len() <= cap, "round {i}: a body beyond the cap"),
            Ok(Err(_)) => {}
            Err(_) => panic!("the loopback client panicked in round {i}"),
        }
    }
    server.join().unwrap();
}
