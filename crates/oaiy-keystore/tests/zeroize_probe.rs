//! "Zeroization" of the keystore's buffers, observed: a global allocator that looks at every block at the moment it is freed (or moved by `realloc`, which
//! frees the old block) and records whether any byte of the secret is still in it. The secret is 30,011 bytes and holds a distinctive 32-byte run (bytes
//! 64 to 95, no zero byte anywhere in the value); a block that contains that run when it is released is a copy of the secret that was not wiped, whatever its
//! size, so a buffer that grew by doubling and left a smaller copy behind is seen as surely as a whole one. The probe also counts the blocks of the secret's
//! size that were freed, so the test cannot pass by never looking.
//!
//! The control shows the probe is not blind: a plain clone of the value, freed, contains the run. This is the only test in this binary, so no other thread's
//! blocks are counted. The `unsafe` here is test code: a `GlobalAlloc` wrapper around the system allocator that forwards every call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use oaiy_keystore::{open_at, Name, ProviderChoice};

struct Probe;

/// Whether the probe is looking.
static ARMED: AtomicBool = AtomicBool::new(false);
/// Blocks freed while armed whose size is that of the secret or a little more (the blob is the secret plus framing).
static SEEN: AtomicUsize = AtomicUsize::new(0);
/// Whether a block freed while armed still held the distinctive run.
static LEAKED: AtomicBool = AtomicBool::new(false);
static LEAKED_SIZE: AtomicUsize = AtomicUsize::new(0);

const N: usize = 30_011;

/// The value: no zero byte, and a run at 64..96 that nothing else in the process contains.
fn value() -> Vec<u8> {
    (0..N).map(|i| (i as u8) | 1).collect()
}

fn marker() -> [u8; 32] {
    std::array::from_fn(|i| ((64 + i) as u8) | 1)
}

fn look(ptr: *mut u8, size: usize) {
    if !ARMED.load(Ordering::SeqCst) {
        return;
    }
    if (N..=N + 200).contains(&size) {
        SEEN.fetch_add(1, Ordering::SeqCst);
    }
    if size < 96 {
        return;
    }
    // SAFETY: the caller of `dealloc` and `realloc` guarantees `ptr` is valid for `size` bytes, and the block is still allocated here.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
    let run = marker();
    if bytes.windows(run.len()).any(|w| w == run) {
        LEAKED.store(true, Ordering::SeqCst);
        LEAKED_SIZE.store(size, Ordering::SeqCst);
    }
}

// SAFETY: every method forwards to the system allocator with the arguments it was given; `dealloc` and `realloc` first look at the old block. While the probe is
// armed, a block that is handed out is first filled with zeros (`alloc`, and the new tail of a `realloc`): memory that nobody has written since it was allocated can
// hold what an earlier user of the same chunk left (the unwiped copies the controls make on purpose, a path buffer with room to spare), and a probe that looked at
// it would report that as a leak of the code under test. What is found in a block is what the code wrote into it while the probe was looking.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let block = System.alloc(layout);
        if !block.is_null() && ARMED.load(Ordering::SeqCst) {
            std::ptr::write_bytes(block, 0, layout.size());
        }
        block
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
        let block = System.realloc(ptr, layout, new_size);
        if !block.is_null() && new_size > layout.size() && ARMED.load(Ordering::SeqCst) {
            std::ptr::write_bytes(block.add(layout.size()), 0, new_size - layout.size());
        }
        block
    }
}

#[global_allocator]
static ALLOCATOR: Probe = Probe;

/// Runs `f` with the probe armed; returns (blocks of the secret's size freed, whether any freed block still held the secret's run).
fn probe(f: impl FnOnce()) -> (usize, bool) {
    SEEN.store(0, Ordering::SeqCst);
    LEAKED.store(false, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    (SEEN.load(Ordering::SeqCst), LEAKED.load(Ordering::SeqCst))
}

#[test]
fn the_keystores_copies_of_a_secret_are_zero_when_they_are_freed() {
    // the control first: a plain copy of the value, freed, still holds the run, and the probe sees it
    let original = value();
    let (seen, leaked) = probe(|| drop(original.clone()));
    assert_eq!((seen, leaked), (1, true), "control: the probe sees a plain copy and finds the secret in it");
    // and the control for a buffer that grows: a Vec that is extended in steps leaves smaller copies in the blocks it moves out of
    let (_, leaked) = probe(|| {
        let mut grown: Vec<u8> = Vec::new();
        for chunk in original.chunks(1000) {
            grown.extend_from_slice(chunk);
        }
        drop(grown);
    });
    assert!(leaked, "control: a buffer that grew freed a block with the secret in it");

    let mut providers = Vec::new();
    #[cfg(unix)]
    providers.push((ProviderChoice::Keyfile, "keyfile"));
    #[cfg(all(windows, feature = "unsafe-keyfile"))]
    providers.push((ProviderChoice::KeyfileUnsafe, "keyfile"));
    if cfg!(windows) {
        providers.push((ProviderChoice::DpapiFile, "dpapi"));
    }
    for (choice, label) in providers {
        let dir = std::env::temp_dir().join(format!("oaiy-keystore-probe-{label}-{}", std::process::id()));
        let store = open_at(dir.join("keys"), choice).unwrap();
        let name = Name::new("probe.secret").unwrap();

        let (seen, leaked) = probe(|| {
            store.put(&name, &original).unwrap();
        });
        // the keyfile makes several copies of the value in the clear (blob, read-back, opened value, check buffer); DPAPI only the opened value it verifies with
        let expected = if label == "keyfile" { 3 } else { 1 };
        assert!(seen >= expected, "{label}: put freed {seen} blocks of the secret's size, expected at least {expected}");
        assert!(!leaked, "{label}: put freed a buffer that still held the secret (size {})", LEAKED_SIZE.load(Ordering::SeqCst));

        let (seen, leaked) = probe(|| {
            let got = store.get(&name).unwrap().unwrap();
            assert_eq!(got.len(), N);
            drop(got);
        });
        assert!(seen >= 1, "{label}: get freed {seen} blocks of the secret's size");
        assert!(!leaked, "{label}: get freed a buffer that still held the secret (size {})", LEAKED_SIZE.load(Ordering::SeqCst));

        // a failed read frees its buffers zeroed too: a damaged file is read into memory, found wrong, and released
        let path = dir.join("keys").join(format!("probe.secret.{}", if label == "keyfile" { "kf" } else { "ks" }));
        let mut damaged = std::fs::read(&path).unwrap();
        let last = damaged.len() - 1;
        damaged[last] ^= 1;
        std::fs::write(&path, &damaged).unwrap();
        let (_, leaked) = probe(|| assert!(store.get(&name).is_err()));
        assert!(!leaked, "{label}: a failed get freed a buffer that still held the secret (size {})", LEAKED_SIZE.load(Ordering::SeqCst));

        drop(store); // the store holds its folder open, and Windows will not remove an open folder
        let _ = std::fs::remove_dir_all(&dir);
    }
}
