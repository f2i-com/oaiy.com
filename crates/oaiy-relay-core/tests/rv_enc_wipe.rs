//! Reviewer's measurement (F2, secrets): does a secret that this crate parses or derives leave an unwiped copy in a heap block that is freed? A counting allocator scans every
//! block that is freed (or moved by `realloc`) while armed for the secret in several spellings (its raw bytes, its base64url text, its 6-bit digits, its Crockford text); a block
//! that held a secret and was wiped by `Zeroizing`/`Secret` has zeros in it and does not match. Positive and negative controls show that the detector works. Stack copies and
//! registers are not seen. `unwiped_copies_of_secrets` prints the table (run with `--nocapture --test-threads 1`); `no_scenario_leaves_an_unwiped_copy` asserts it and is ignored
//! because it fails today (a finding).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::admission::Bearer;
use oaiy_relay_core::enrol::{Enrolled, EnrolmentKey, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::X25519Secret;
use oaiy_relay_core::pairing::math::{parse_typed_code, typed_code, PairingKey, PairingSecret};
use oaiy_relay_core::url::RelayUrl;
use oaiy_relay_core::{b64, sealed};
use zeroize::Zeroizing;

struct Needle {
    name: &'static str,
    bytes: Vec<u8>,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static SET: AtomicPtr<Vec<Needle>> = AtomicPtr::new(std::ptr::null_mut());
static FREED: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];
static MOVED: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];

fn contains(block: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && block.len() >= needle.len() && block.windows(needle.len()).any(|w| w == needle)
}

unsafe fn scan(p: *mut u8, size: usize, counters: &[AtomicUsize; 16]) {
    if !ARMED.load(Ordering::Relaxed) || p.is_null() || size == 0 || size > 1 << 20 {
        return;
    }
    let set = SET.load(Ordering::Acquire);
    if set.is_null() {
        return;
    }
    let block = std::slice::from_raw_parts(p, size);
    for (i, n) in (*set).iter().enumerate() {
        if contains(block, &n.bytes) {
            counters[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Detector;
unsafe impl GlobalAlloc for Detector {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        scan(p, l.size(), &FREED);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        scan(p, l.size(), &MOVED);
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Detector = Detector;

fn six_bit(text: &str) -> Vec<u8> {
    text.bytes().map(|c| "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".find(c as char).unwrap() as u8).collect()
}

/// Runs `f` armed for `needles`, drops its result while still armed, prints one line and records how many freed blocks held any needle.
fn measure<R>(results: &mut Vec<(String, usize)>, label: &str, needles: Vec<Needle>, f: impl FnOnce() -> R) {
    let set = Box::into_raw(Box::new(needles));
    for c in FREED.iter().chain(MOVED.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    SET.store(set, Ordering::Release);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    drop(out);
    ARMED.store(false, Ordering::SeqCst);
    SET.store(std::ptr::null_mut(), Ordering::Release);
    // SAFETY: `set` came from Box::into_raw above and nothing else frees it.
    let needles = unsafe { Box::from_raw(set) };
    let mut line = String::new();
    let mut total = 0;
    for (i, n) in needles.iter().enumerate() {
        let (a, b) = (FREED[i].load(Ordering::Relaxed), MOVED[i].load(Ordering::Relaxed));
        total += a;
        line.push_str(&format!(" [{}: freed {} moved {}]", n.name, a, b));
    }
    println!("{label:<46}{line}");
    results.push((label.to_string(), total));
}

fn scenarios() -> Vec<(String, usize)> {
    let mut r: Vec<(String, usize)> = Vec::new();
    // ---- controls
    let probe = b"PROBE-PROBE-PROBE-0123456789".to_vec();
    measure(&mut r, "control: a plain String is freed (positive)", vec![Needle { name: "probe", bytes: probe.clone() }], || {
        let s = String::from_utf8(probe.clone()).unwrap();
        std::hint::black_box(&s);
    });
    measure(&mut r, "control: a Zeroizing<String> is freed (negative)", vec![Needle { name: "probe", bytes: probe.clone() }], || {
        let s = Zeroizing::new(String::from_utf8(probe.clone()).unwrap());
        std::hint::black_box(&s);
    });

    // ---- the device token
    let id = [0x41u8, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48];
    let sec: [u8; 32] = core::array::from_fn(|i| 0xA0u8.wrapping_add(i as u8 * 7));
    let (id_t, sec_t) = (b64::encode(&id), b64::encode(&sec));
    let token_text = Zeroizing::new(format!("oaiyrt1.{id_t}.{sec_t}"));
    let token_needles = || {
        vec![
            Needle { name: "raw32", bytes: sec.to_vec() },
            Needle { name: "b64 text", bytes: sec_t.clone().into_bytes() },
            Needle { name: "6-bit digits", bytes: six_bit(&sec_t) },
            Needle { name: "token text", bytes: token_text.as_bytes().to_vec() },
        ]
    };
    measure(&mut r, "Token::parse(text), then drop", token_needles(), || Token::parse(&token_text).unwrap());

    // ---- sealed token
    let recipient = X25519Secret::from_secret(&Secret::new([9u8; 32]));
    let sealed_text = sealed::seal_container(&recipient.public_key(), &token_text).unwrap();
    measure(&mut r, "sealed::open_token(box)", token_needles(), || sealed::open_token(&recipient, &sealed_text).unwrap());

    // ---- the enrolment answer
    let body = Zeroizing::new(
        format!(
            "{{\"deviceId\":\"dev-AAAAAAAAAAAAAAAAAAAAAA\",\"token\":\"{}\",\"relayId\":\"rly-AAAAAAAAAAAAAAAAAAAAAA\",\"time\":1790000000}}",
            token_text.as_str()
        )
        .into_bytes(),
    );
    measure(&mut r, "Enrolled::parse(body)", token_needles(), || Enrolled::parse(&body, Role::Desktop).unwrap());

    // ---- the admission bearer
    let mac_raw: [u8; 32] = core::array::from_fn(|i| 0x30u8.wrapping_add(i as u8 * 5));
    let claims = br#"{"exp":1790000000}"#;
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let bearer_text = Zeroizing::new(format!("aokie-adm-v2.{}.{}", hex(claims), hex(&mac_raw)));
    measure(
        &mut r,
        "Bearer::parse(text)",
        vec![Needle { name: "mac raw32", bytes: mac_raw.to_vec() }, Needle { name: "bearer text", bytes: bearer_text.as_bytes().to_vec() }],
        || Bearer::parse(&bearer_text).unwrap(),
    );

    // ---- the pairing secret
    let ps: [u8; 16] = core::array::from_fn(|i| 0x55u8.wrapping_add(i as u8 * 11));
    let ps_b64 = b64::encode(&ps);
    let code = Zeroizing::new(typed_code(&ps));
    let flat = Zeroizing::new(code.replace('-', ""));
    let pairing_needles = || {
        vec![
            Needle { name: "raw16", bytes: ps.to_vec() },
            Needle { name: "b64 text", bytes: ps_b64.clone().into_bytes() },
            Needle { name: "6-bit digits", bytes: six_bit(&ps_b64) },
            Needle { name: "crockford 26", bytes: flat.as_bytes()[..26].to_vec() },
        ]
    };
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let thumb = "atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR4";
    let pairing_uri = Zeroizing::new(PairingKey::to_uri(&relay, thumb, &PairingSecret::new(ps), 1790000600));
    measure(&mut r, "math::parse_typed_code(code)", pairing_needles(), || parse_typed_code(&code).unwrap());
    measure(&mut r, "math::typed_code(&secret) [String wrapped]", pairing_needles(), || Zeroizing::new(typed_code(&ps)));
    measure(&mut r, "PairingSecret::b64u() [String wrapped]", pairing_needles(), || Zeroizing::new(PairingSecret::new(ps).b64u()));
    measure(&mut r, "PairingKey::to_uri(..) [String wrapped]", pairing_needles(), || {
        Zeroizing::new(PairingKey::to_uri(&relay, thumb, &PairingSecret::new(ps), 1790000600))
    });
    measure(&mut r, "PairingKey::parse(uri)", pairing_needles(), || PairingKey::parse(&pairing_uri).unwrap());
    measure(&mut r, "PairingSecret::derive()", pairing_needles(), || PairingSecret::new(ps).derive().unwrap());

    // ---- the enrolment key
    let es: [u8; 16] = core::array::from_fn(|i| 0x77u8.wrapping_add(i as u8 * 13));
    let es_b64 = b64::encode(&es);
    let enrol_needles = || {
        vec![
            Needle { name: "raw16", bytes: es.to_vec() },
            Needle { name: "b64 text", bytes: es_b64.clone().into_bytes() },
            Needle { name: "6-bit digits", bytes: six_bit(&es_b64) },
        ]
    };
    let enrol_uri = Zeroizing::new(EnrolmentKey::to_uri(&relay, thumb, &es, Role::Desktop, 1790000600).unwrap());
    measure(&mut r, "EnrolmentKey::to_uri(..) [String wrapped]", enrol_needles(), || {
        Zeroizing::new(EnrolmentKey::to_uri(&relay, thumb, &es, Role::Desktop, 1790000600).unwrap())
    });
    measure(&mut r, "EnrolmentKey::parse(uri)", enrol_needles(), || EnrolmentKey::parse(&enrol_uri).unwrap());
    measure(&mut r, "EnrolmentKey::parse(uri).signer()", enrol_needles(), || EnrolmentKey::parse(&enrol_uri).unwrap().signer().unwrap());
    r
}

#[test]
fn unwiped_copies_of_secrets() {
    let r = scenarios();
    // the controls must behave as controls, or the table means nothing
    assert!(r[0].1 > 0, "the positive control was not detected");
    assert_eq!(r[1].1, 0, "the negative control was detected");
}

/// Fails today (run with `--ignored`): every scenario of the table must show no unwiped copy of a secret. A failure names the scenarios that still leave one.
#[test]
fn no_scenario_leaves_an_unwiped_copy() {
    let bad: Vec<String> = scenarios().into_iter().filter(|(l, n)| *n > 0 && !l.starts_with("control")).map(|(l, n)| format!("{l}: {n}")).collect();
    assert!(bad.is_empty(), "{bad:#?}");
}
