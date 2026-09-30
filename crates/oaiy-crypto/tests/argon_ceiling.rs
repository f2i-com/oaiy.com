//! Design 4.1.2 and attack C1: "Argon2id bounds, checked before any derivation ... anything else is `kdf_params_out_of_range` and nothing is
//! allocated". FormLogic's browser client has only the floor, so a hostile server can choose a cost of terabytes.
//!
//! This test binary installs a counting allocator (that is the `unsafe` in this file: a `GlobalAlloc` wrapper around the system
//! allocator, test code only; the library forbids unsafe code) and proves the claim as stated: after refusing a hostile cost, no
//! allocation of even a megabyte was asked for, and the same allocator does see the 64 MiB of an honest derivation, so the counter is
//! not blind. It is the only test in this binary, so that no other thread's allocations are counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use oaiy_crypto::argon::{self, MEM_MAX, MEM_MIN};
use oaiy_crypto::bip39::{self, Entropy};
use oaiy_crypto::Error;

struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the arguments it was given and only records the requested size.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn c1_a_hostile_argon2_cost_is_refused_before_anything_is_allocated() {
    let salt = [0x11u8; 16];
    let entropy = Entropy::from_bytes([0x22; 16]);
    let words = bip39::encode(&entropy);
    let tib = 1024u64 * 1024 * 1024 * 1024;
    let hostile: [(u64, u64); 12] = [
        (3, 4 * tib),
        (3, tib),
        (3, 100 * 1024 * 1024 * 1024),
        (3, MEM_MAX + 1024),
        (3, u64::from(u32::MAX) * 1024 + MEM_MIN), // would wrap a 32-bit KiB count
        (3, u64::MAX),
        (11, MEM_MIN),
        (1_000_000_000, MEM_MIN),
        (u64::MAX, u64::MAX),
        (2, MEM_MIN),
        (3, MEM_MIN - 1024),
        (0, 0),
    ];
    LARGEST.store(0, Ordering::Relaxed);
    for (ops, mem) in hostile {
        assert_eq!(argon::argon2id13(b"password", &salt, ops, mem).unwrap_err(), Error::KdfParamsOutOfRange, "ops {ops} mem {mem}");
        assert_eq!(bip39::phrase_wrap_key(words.expose(), &salt, ops, mem).unwrap_err(), Error::KdfParamsOutOfRange, "phrase: ops {ops} mem {mem}");
        assert_eq!(bip39::wrap_key(&entropy, &salt, ops, mem).unwrap_err(), Error::KdfParamsOutOfRange, "entropy: ops {ops} mem {mem}");
    }
    // a salt that is not 16 bytes is refused the same way, and before the memory too
    for len in [0usize, 15, 17, 64] {
        assert_eq!(argon::argon2id13(b"password", &vec![0u8; len], 3, MEM_MIN).unwrap_err(), Error::KdfParamsOutOfRange, "salt {len}");
    }
    let asked = LARGEST.load(Ordering::Relaxed);
    assert!(asked < 1024 * 1024, "the refusals asked the allocator for {asked} bytes in one piece");

    // the counter is not blind: an honest derivation at the floor asks for its 64 MiB
    LARGEST.store(0, Ordering::Relaxed);
    let key = argon::argon2id13(b"password", &salt, 3, MEM_MIN).unwrap();
    assert!(LARGEST.load(Ordering::Relaxed) >= MEM_MIN as usize, "an honest derivation allocates its memory");
    assert_eq!(key.expose().len(), 32);
}
