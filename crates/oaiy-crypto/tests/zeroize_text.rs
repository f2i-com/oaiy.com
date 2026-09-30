//! Text secrets and the memory they pass through: where does a recovery phrase or a kit code go that is *not* in the final, zeroizing result? (Review M-4.)
//!
//! `zeroize.rs` checks the final block of a secret: the one whose size is the secret's. A `String` that is built by pushing grows by moving to a bigger block,
//! and the block it leaves is freed, unwiped, with the first part of the text in it: the reviewer found 'legal winner thank year wave sausage w' (19 and 38
//! bytes) in freed memory after `bip39::decode`, ten of twelve words of the phrase in a freed 64-byte block after `encode`, and `FLRK1-CEIR-...` after the kit's
//! `encode` and `decode`. The probe here is a global allocator that looks for a distinctive run of the secret's text in **every** block at the moment it is
//! freed or moved by `realloc`, whatever its size, while one function runs. A positive control (a plain copy, and a `String` that grows) shows that the probe
//! sees what it is meant to see; the paths under test must leave none: phrase encode, decode and `phrase_wrap_key`, and kit encode, decode and decode of a
//! code as a person types it (lower case, spaces instead of hyphens).
//!
//! This is the only test in this binary, so no other thread's blocks are looked at. The `unsafe` is test code: the `GlobalAlloc` wrapper, and the read of a block
//! that is still allocated when it is handed back.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use oaiy_crypto::argon;
use oaiy_crypto::bip39::{self, Entropy};
use oaiy_crypto::kit::RecoveryKit;
use oaiy_crypto::zeroize::Secret;

struct Scan;

const NEEDLE_MAX: usize = 32;

/// Whether the probe is looking.
static ARMED: AtomicBool = AtomicBool::new(false);
static NEEDLE: [AtomicU8; NEEDLE_MAX] = [const { AtomicU8::new(0) }; NEEDLE_MAX];
static NEEDLE_LEN: AtomicUsize = AtomicUsize::new(0);
/// Blocks freed or moved while armed that still held the needle.
static HITS: AtomicUsize = AtomicUsize::new(0);
/// The sizes of the first eight such blocks (for the message of a failure).
static HIT_SIZES: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];

fn record(size: usize) {
    let n = HITS.fetch_add(1, Ordering::SeqCst);
    if let Some(slot) = HIT_SIZES.get(n) {
        slot.store(size, Ordering::SeqCst);
    }
}

/// # Safety
/// `ptr` must be valid for reads of `len` bytes (a block that is still allocated).
unsafe fn holds_needle(ptr: *const u8, len: usize) -> bool {
    let n = NEEDLE_LEN.load(Ordering::SeqCst);
    if n == 0 || len < n {
        return false;
    }
    let mut needle = [0u8; NEEDLE_MAX];
    for (i, slot) in needle.iter_mut().enumerate().take(n) {
        *slot = NEEDLE[i].load(Ordering::SeqCst);
    }
    std::slice::from_raw_parts(ptr, len).windows(n).any(|w| w == &needle[..n])
}

// SAFETY: every method forwards to the system allocator with the arguments it was given; before a block is freed or moved it is read for the needle, while it is
// still allocated and valid for `layout.size()` bytes (the caller of `dealloc` and `realloc` guarantees that).
unsafe impl GlobalAlloc for Scan {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ARMED.load(Ordering::SeqCst) && holds_needle(ptr, layout.size()) {
            record(layout.size());
        }
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let had = ARMED.load(Ordering::SeqCst) && holds_needle(ptr, layout.size());
        let moved = System.realloc(ptr, layout, new_size);
        if had && moved != ptr {
            record(layout.size()); // the old block was freed with the needle in it
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Scan = Scan;

/// Runs `f` with the probe looking for `needle`; returns how many freed or moved blocks still held it, and their sizes.
fn probe(needle: &[u8], f: impl FnOnce()) -> (usize, Vec<usize>) {
    assert!(needle.len() <= NEEDLE_MAX);
    for (i, b) in needle.iter().enumerate() {
        NEEDLE[i].store(*b, Ordering::SeqCst);
    }
    NEEDLE_LEN.store(needle.len(), Ordering::SeqCst);
    HITS.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    let hits = HITS.load(Ordering::SeqCst);
    let sizes = HIT_SIZES.iter().take(hits.min(8)).map(|s| s.load(Ordering::SeqCst)).collect();
    (hits, sizes)
}

const PHRASE: &str = "legal winner thank year wave sausage worth useful legal winner thank yellow";

#[test]
fn text_secrets_leave_no_copy_in_freed_or_moved_memory() {
    let needle = &PHRASE.as_bytes()[..12]; // "legal winner"

    // the controls: the probe sees a plain copy that is freed, and a String that grows (its old block is freed with the start of the text in it)
    let original = PHRASE.to_string();
    let (hits, _) = probe(needle, || drop(std::hint::black_box(original.clone())));
    assert_eq!(hits, 1, "control: a plain copy of the phrase, freed, is seen");
    let (hits, sizes) = probe(needle, || {
        let mut grown = String::new();
        for word in PHRASE.split(' ') {
            grown.push_str(word);
            grown.push(' ');
        }
        drop(std::hint::black_box(grown));
    });
    assert!(
        hits >= 2,
        "control: a String that grows leaves copies in the blocks it moved out of, and the probe sees them ({hits} seen, sizes {sizes:?})"
    );

    // --- the phrase ---
    let (hits, sizes) = probe(needle, || {
        let entropy = bip39::decode(PHRASE).unwrap();
        drop(entropy);
    });
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "bip39::decode left copies of the phrase in freed memory (block sizes {sizes:?})");

    let entropy = bip39::decode(PHRASE).unwrap();
    let (hits, sizes) = probe(needle, || {
        let shown = bip39::encode(&entropy);
        assert_eq!(shown.expose(), PHRASE);
        drop(shown);
    });
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "bip39::encode left copies of the phrase in freed memory (block sizes {sizes:?})");

    let (hits, sizes) = probe(needle, || {
        let key = bip39::phrase_wrap_key(PHRASE, &[7u8; 16], 3, argon::MEM_MIN).unwrap();
        drop(key);
    });
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "bip39::phrase_wrap_key left copies of the phrase in freed memory (block sizes {sizes:?})");

    // the phrase as a person types it: capitals, extra and odd white space
    let typed = "  LEGAL Winner\tthank\u{a0}year wave  sausage\nworth useful legal winner thank yellow ".to_string();
    let (hits, sizes) = probe(needle, || drop(bip39::decode(&typed).unwrap()));
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "decoding a typed phrase left copies in freed memory (block sizes {sizes:?})");
    let _ = Entropy::from_bytes([0; 16]);
    // NFKD can lengthen a character a very great deal (U+FDFA becomes eighteen characters), so a crafted input outgrows the buffers that were reserved for it. The text
    // then moves to a bigger buffer by hand, and the buffer it leaves must be wiped: the start of the phrase is in it.
    let long = format!("{PHRASE} {}", "\u{fdfa}".repeat(60));
    let (hits, sizes) = probe(needle, || assert!(bip39::decode(&long).is_err()));
    assert_eq!(
        (hits, &sizes[..]),
        (0, &[][..]),
        "a phrase whose NFKD form outgrew the reserved buffer left copies in freed memory (block sizes {sizes:?})"
    );

    // --- the kit code: 0x11 * 32 is "CEIRCEIR..." in Base32 ---
    let kit = RecoveryKit::from_bytes(Secret::new([0x11; 32]));
    let code = kit.encode().expose().to_string();
    let kneedle = b"CEIRCEIRCEIR";
    let (hits, sizes) = probe(kneedle, || drop(kit.encode()));
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "RecoveryKit::encode left copies of the code in freed memory (block sizes {sizes:?})");
    let (hits, sizes) = probe(b"-CEIR-CEIR-", || drop(kit.encode()));
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "RecoveryKit::encode left copies of the hyphenated code in freed memory (block sizes {sizes:?})");
    let (hits, sizes) = probe(kneedle, || drop(RecoveryKit::decode(&code).unwrap()));
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "RecoveryKit::decode left copies of the code in freed memory (block sizes {sizes:?})");
    let typed = code.to_lowercase().replace('-', " ");
    let (hits, sizes) = probe(kneedle, || drop(RecoveryKit::decode(&typed).unwrap()));
    assert_eq!((hits, &sizes[..]), (0, &[][..]), "decoding a typed kit code left copies in freed memory (block sizes {sizes:?})");
}
