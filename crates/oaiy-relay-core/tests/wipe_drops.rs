//! What is wiped when a value that held a secret is dropped, measured the way the reviewer's `rv_enc_wipe.rs` measures what the parsers leave behind: a counting allocator scans every
//! heap block that is given back (and every one that a `realloc` moves) while it is armed, for the secret in the spellings it has; a block that was wiped holds zeros and does not
//! match. Positive and negative controls show that the detector sees what it should. One test only, because the allocator counts the whole process. Stack copies are not seen.
//!
//! The values here are the ones the crate wipes by a `Drop` of its own: a request that carries a bearer (`HttpRequest`), the offer the owner is shown (`NewOffer`) and the party that
//! made it, a TURN credential (`IceServer`), the short authentication string (`Sas`) and a parsed document that held a token (`json::Wiped`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oaiy_relay_core::admission::IceServer;
use oaiy_relay_core::b64;
use oaiy_relay_core::client::{Cancel, HttpRequest, Method};
use oaiy_relay_core::json;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::math::{typed_code, Sas};
use oaiy_relay_core::pairing::{DesktopIdentity, DesktopPairing};
use oaiy_relay_core::url::RelayUrl;
use zeroize::Zeroizing;

struct Needle {
    name: &'static str,
    bytes: Vec<u8>,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static SET: AtomicPtr<Vec<Needle>> = AtomicPtr::new(std::ptr::null_mut());
static FREED: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];

fn contains(block: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && block.len() >= needle.len() && block.windows(needle.len()).any(|w| w == needle)
}

/// # Safety
/// `p` is a block of `size` bytes that the allocator is about to free or move.
unsafe fn scan(p: *mut u8, size: usize) {
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
            FREED[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Detector;
unsafe impl GlobalAlloc for Detector {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        scan(p, l.size());
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        scan(p, l.size());
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static A: Detector = Detector;

/// Runs `f` armed for `needles`, drops its result while still armed, and returns how many freed (or moved) blocks held a needle, by name.
fn measure<R>(needles: Vec<Needle>, f: impl FnOnce() -> R) -> Vec<(&'static str, usize)> {
    let set = Box::into_raw(Box::new(needles));
    for c in FREED.iter() {
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
    needles.iter().enumerate().map(|(i, n)| (n.name, FREED[i].load(Ordering::Relaxed))).collect()
}

fn leaks(r: &[(&'static str, usize)]) -> usize {
    r.iter().map(|(_, n)| n).sum()
}

fn six_bit(text: &str) -> Vec<u8> {
    text.bytes().map(|c| "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".find(c as char).unwrap() as u8).collect()
}

#[test]
fn what_a_drop_wipes() {
    // ---- controls: the detector sees a plain String that is freed, and does not see one that was wiped.
    let probe = b"PROBE-PROBE-PROBE-0123456789".to_vec();
    let positive = measure(vec![Needle { name: "probe", bytes: probe.clone() }], || {
        let s = String::from_utf8(probe.clone()).unwrap();
        std::hint::black_box(&s);
    });
    assert!(leaks(&positive) > 0, "the positive control was not detected");
    let negative = measure(vec![Needle { name: "probe", bytes: probe.clone() }], || {
        let s = Zeroizing::new(String::from_utf8(probe.clone()).unwrap());
        std::hint::black_box(&s);
    });
    assert_eq!(leaks(&negative), 0, "the negative control was detected");

    // ---- a request that carries a bearer: the header value is wiped when the request is dropped.
    let token = Zeroizing::new(format!("oaiyrt1.{}.{}", b64::encode(&[1u8, 2, 3, 4, 5, 6, 7, 8]), b64::encode(&[0xB7u8; 32])));
    let secret_half = Zeroizing::new(token.rsplit('.').next().unwrap().to_string());
    let token_needles = || {
        vec![Needle { name: "token text", bytes: token.as_bytes().to_vec() }, Needle { name: "secret half", bytes: secret_half.as_bytes().to_vec() }]
    };
    let r = measure(token_needles(), || {
        let mut bearer = String::with_capacity(8 + token.len());
        bearer.push_str("Bearer ");
        bearer.push_str(&token);
        HttpRequest {
            method: Method::Get,
            url: "https://relay.example.com/v1/poll".into(),
            headers: vec![("Authorization".into(), bearer)],
            body: None,
            timeout: Duration::from_secs(1),
            max_response_bytes: 10,
            cancel: Cancel::new(),
        }
    });
    assert_eq!(leaks(&r), 0, "a dropped request left the bearer in a freed block: {r:?}");
    // The control: a header value that is not wiped is seen (so the measurement above means something).
    let r = measure(token_needles(), || {
        let mut bearer = String::with_capacity(8 + token.len());
        bearer.push_str("Bearer ");
        bearer.push_str(&token);
        bearer
    });
    assert!(leaks(&r) > 0, "the detector did not see a bearer in a String that was freed");

    // ---- the offer the owner is shown, and the party that made it.
    let secret = [0x55u8, 0x42, 0x31, 0x20, 0x1f, 0x0e, 0x0d, 0x1c, 0x2b, 0x3a, 0x49, 0x58, 0x67, 0x76, 0x85, 0x94];
    let flat = Zeroizing::new(typed_code(&secret).replace('-', ""));
    let secret_b64 = Zeroizing::new(b64::encode(&secret));
    let offer_needles = || {
        vec![
            Needle { name: "raw16", bytes: secret.to_vec() },
            Needle { name: "b64 text", bytes: secret_b64.as_bytes().to_vec() },
            Needle { name: "6-bit digits", bytes: six_bit(&secret_b64) },
            Needle { name: "crockford 26", bytes: flat.as_bytes()[..26].to_vec() },
        ]
    };
    let identity = Arc::new(DesktopIdentity {
        device_id: "dev-AAAAAAAAAAAAAAAAAAAAAA".into(),
        name: "Desk".into(),
        endpoint: Signer::generate().unwrap(),
        endpoint_x25519: X25519Secret::generate().unwrap().public_key(),
        host_ed25519: Signer::generate().unwrap().verify_key(),
        host_x25519: X25519Secret::generate().unwrap().public_key(),
    });
    let thumb = Signer::generate().unwrap().thumbprint();
    let relay = RelayUrl::parse("https://relay.example.com").unwrap();
    let r = measure(offer_needles(), || {
        let mut party = DesktopPairing::new(identity.clone(), "aokie", relay.clone(), &thumb);
        let offer = party.create_offer_with(secret, [9u8; 32], "pair-AAAAAAAAAAAAAAAA".into(), 1_790_000_000).unwrap();
        // The owner is shown the key and the code; both are dropped with the offer, and the party with its pending pairing.
        std::hint::black_box((&offer.pairing_uri, &offer.typed_code));
        drop(offer);
        party
    });
    assert_eq!(leaks(&r), 0, "making an offer, showing it and dropping it left the secret in a freed block: {r:?}");

    // ---- a TURN credential.
    let credential = "TURN-CREDENTIAL-WIPE-ME-0123456789";
    let r = measure(vec![Needle { name: "credential", bytes: credential.as_bytes().to_vec() }], || IceServer {
        urls: vec!["turns:turn.example.com:443".into()],
        username: "1790000000:abc".into(),
        credential: credential.to_string(),
        expires_at: Some(1_790_000_000),
    });
    assert_eq!(leaks(&r), 0, "a dropped ICE server left its credential in a freed block: {r:?}");

    // ---- the short authentication string.
    let r = measure(
        vec![
            Needle { name: "chars12", bytes: b"6NHNK68MQQVZ".to_vec() },
            Needle { name: "raw", bytes: vec![0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8] },
        ],
        || Sas::from_parts([0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8], String::from_utf8(b"6NHNK68MQQVZ".to_vec()).unwrap(), '5'),
    );
    assert_eq!(leaks(&r), 0, "a dropped Sas left its characters in a freed block: {r:?}");

    // ---- a parsed document that held a token.
    let body = format!("{{\"deviceId\":\"dev-AAAAAAAAAAAAAAAAAAAAAA\",\"token\":\"{}\"}}", token.as_str());
    let r = measure(token_needles(), || json::parse_wiped(body.as_bytes()).unwrap());
    assert_eq!(leaks(&r), 0, "a dropped document left the token in a freed block: {r:?}");
    let r = measure(token_needles(), || json::parse(body.as_bytes()).unwrap());
    assert!(leaks(&r) > 0, "the detector did not see a token in a tree that was not wiped");
}
