//! "Zeroization" of the keystore's buffers, observed: a global allocator that looks at every block in a size window at the moment it is freed (or
//! moved by `realloc`, which frees the old block) and records whether it was all zero. The secret in this test is 30,011 bytes, a size nothing else
//! in the process allocates near, so a block in the window is one of the keystore's copies of it: the blob buffer, the read-back buffer, the opened
//! value, the check buffer. Every one of them must be zero when it goes back to the allocator, on `put` and on `get`, for the keyfile (which holds the
//! value in the clear in its blob) and for DPAPI (whose blob is ciphertext, so only the value-sized copies are secret).
//!
//! The control shows the probe is not blind: a plain `Vec` of that size, freed, is not zero. This is the only test in this binary, so no other
//! thread's blocks are counted. The `unsafe` here is test code: a `GlobalAlloc` wrapper around the system allocator that forwards every call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use oaiy_keystore::{open_at, Name, ProviderChoice};

struct Probe;

static LOW: AtomicUsize = AtomicUsize::new(usize::MAX);
static HIGH: AtomicUsize = AtomicUsize::new(0);
static SEEN: AtomicUsize = AtomicUsize::new(0);
static NONZERO: AtomicBool = AtomicBool::new(false);
static NONZERO_SIZE: AtomicUsize = AtomicUsize::new(0);

fn look(ptr: *mut u8, size: usize) {
    if size >= LOW.load(Ordering::SeqCst) && size <= HIGH.load(Ordering::SeqCst) {
        SEEN.fetch_add(1, Ordering::SeqCst);
        // SAFETY: the caller of `dealloc` and `realloc` guarantees `ptr` is valid for `size` bytes, and the block is still allocated here.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
        if bytes.iter().any(|b| *b != 0) {
            NONZERO.store(true, Ordering::SeqCst);
            NONZERO_SIZE.store(size, Ordering::SeqCst);
        }
    }
}

// SAFETY: every method forwards to the system allocator with the arguments it was given; `dealloc` and `realloc` first look at the old block.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        look(ptr, layout.size());
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        look(ptr, layout.size());
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Probe = Probe;

/// Runs `f` with the window `[low, high]` armed; returns (blocks seen, any of them not zero).
fn probe(low: usize, high: usize, f: impl FnOnce()) -> (usize, bool) {
    SEEN.store(0, Ordering::SeqCst);
    NONZERO.store(false, Ordering::SeqCst);
    LOW.store(low, Ordering::SeqCst);
    HIGH.store(high, Ordering::SeqCst);
    f();
    LOW.store(usize::MAX, Ordering::SeqCst);
    HIGH.store(0, Ordering::SeqCst);
    (SEEN.load(Ordering::SeqCst), NONZERO.load(Ordering::SeqCst))
}

const N: usize = 30_011;

#[test]
fn the_keystores_copies_of_a_secret_are_zero_when_they_are_freed() {
    // the control first: a plain Vec in the window is not zero at its free
    let (seen, nonzero) = probe(N, N + 100, || drop(vec![0xABu8; N + 7]));
    assert_eq!((seen, nonzero), (1, true), "control: the probe sees a plain Vec and finds the bytes in it");

    // the keyfile window reaches twice the secret's size, so that a buffer that grew by doubling (and left a smaller copy behind when it moved) would be seen
    let mut providers = vec![(ProviderChoice::Keyfile, 2 * N + 200, "keyfile")];
    if cfg!(windows) {
        // the DPAPI blob is ciphertext and its size is not the secret's; only blocks of exactly the value's size are copies of it
        providers.push((ProviderChoice::DpapiFile, N, "dpapi"));
    }
    for (choice, high, label) in providers {
        let dir = std::env::temp_dir().join(format!("oaiy-keystore-probe-{label}-{}", std::process::id()));
        let store = open_at(dir.join("keys"), choice).unwrap();
        let name = Name::new("probe.secret").unwrap();
        let value: Vec<u8> = (0..N).map(|i| (i as u8) | 1).collect(); // no zero byte anywhere: any block still holding it is visible
        assert_eq!(value.len(), N);

        let (seen, nonzero) = probe(N, high, || store.put(&name, &value).unwrap());
        // the keyfile makes several copies of the value in the clear (blob, read-back, opened value, check buffer); DPAPI only the opened value it verifies with
        let expected = if label == "keyfile" { 3 } else { 1 };
        assert!(seen >= expected, "{label}: put freed {seen} blocks of the secret's size, expected at least {expected}");
        assert!(!nonzero, "{label}: put freed a buffer that still held the secret (size {})", NONZERO_SIZE.load(Ordering::SeqCst));

        let (seen, nonzero) = probe(N, high, || {
            let got = store.get(&name).unwrap().unwrap();
            assert_eq!(got.len(), N);
            drop(got);
        });
        assert!(seen >= 1, "{label}: get freed {seen} blocks of the secret's size");
        assert!(!nonzero, "{label}: get freed a buffer that still held the secret");

        // a failed read frees its buffers zeroed too: a damaged file is read into memory, found wrong, and released
        let path = dir.join("keys").join(format!("probe.secret.{}", if label == "keyfile" { "kf" } else { "ks" }));
        let mut damaged = std::fs::read(&path).unwrap();
        let last = damaged.len() - 1;
        damaged[last] ^= 1;
        std::fs::write(&path, &damaged).unwrap();
        let (_, nonzero) = probe(N, high, || assert!(store.get(&name).is_err()));
        assert!(!nonzero, "{label}: a failed get freed a buffer that still held the secret");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
