//! Zeroization of every secret type, and constant-time comparison.
//!
//! The library is `#![forbid(unsafe_code)]`. This test file is the only place `unsafe` appears in the crate, and it is test code with two jobs
//! that cannot be done in safe Rust: (1) a global allocator that looks at the bytes of a block at the moment it is freed, so that "the memory was
//! zero before it went back to the allocator" is observed, not assumed (`heap_secrets_are_zero_when_they_are_freed`, with a control that shows a
//! plain `Vec` is not zero at that moment, so the probe is not blind); (2) `drop_in_place` on a value in a slot we own and a volatile read of the
//! slot afterwards, for the secrets that live inline (`inline_secrets_are_zero_after_drop`). Both read only memory that is still ours and was
//! initialised (padding of the one struct that has some is read as bytes, the way the `zeroize` crate's own tests read it).

use std::alloc::{GlobalAlloc, Layout, System};
use std::mem::{size_of, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use oaiy_crypto::aead;
use oaiy_crypto::argon::{self, MEM_MIN};
use oaiy_crypto::bip39::Entropy;
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::ed25519::{KeyRole, SigningKey};
use oaiy_crypto::kit::RecoveryKit;
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::SecretKey;
use oaiy_crypto::zeroize::{ct_eq, Secret, SecretString, SecretVec};
use zeroize::ZeroizeOnDrop;

struct Probe;

/// The size of the block whose contents are checked when it is freed (0: none).
static TARGET: AtomicUsize = AtomicUsize::new(0);
/// 0: not freed yet; 1: all zero when freed; 2: not all zero when freed.
static VERDICT: AtomicU8 = AtomicU8::new(0);

// SAFETY: every method forwards to the system allocator with the arguments it was given. `dealloc` first reads the `layout.size()` bytes of a block
// that is still allocated (the caller guarantees `ptr` is valid for that layout) and records whether they were all zero.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() != 0 && layout.size() == TARGET.load(Ordering::SeqCst) {
            let bytes = std::slice::from_raw_parts(ptr, layout.size());
            VERDICT.store(if bytes.iter().all(|b| *b == 0) { 1 } else { 2 }, Ordering::SeqCst);
        }
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Probe = Probe;

/// Runs `f` with the probe aimed at blocks of `size` bytes and returns the verdict on the last such block freed while it ran.
fn verdict_for(size: usize, f: impl FnOnce()) -> u8 {
    TARGET.store(size, Ordering::SeqCst);
    VERDICT.store(0, Ordering::SeqCst);
    f();
    TARGET.store(0, Ordering::SeqCst);
    VERDICT.load(Ordering::SeqCst)
}

// The heap cases share the two statics above, so they run one after another in one test; the sizes are primes that no other allocation in this
// binary has.
const N1: usize = 7_919;
const N2: usize = 7_927;
const N3: usize = 7_933;
const N4: usize = 7_937;
const N5: usize = 7_949;

#[test]
fn heap_secrets_are_zero_when_they_are_freed() {
    // the control: a plain Vec of that size is NOT zero when it is freed, so the probe would notice
    assert_eq!(verdict_for(N1, || drop(vec![0xABu8; N1])), 2, "control: a plain Vec keeps its bytes until it is freed");

    assert_eq!(verdict_for(N1, || drop(SecretVec::new(vec![0xABu8; N1]))), 1, "SecretVec");
    assert_eq!(verdict_for(N2, || drop(SecretString::new("x".repeat(N2)))), 1, "SecretString");

    // a decrypted message of N3 bytes
    let key: Secret<32> = Secret::new([0x5a; 32]);
    let nonce = [1u8; 24];
    let sealed = aead::seal(&key, aead::Nonce::from_bytes_for_tests(nonce), b"aad", &vec![0xCDu8; N3]).unwrap();
    assert_eq!(verdict_for(N3, || drop(aead::open(&key, &nonce, b"aad", &sealed).unwrap())), 1, "aead::open");
    let recipient = SecretKey::from_bytes([0x61; 32]);
    let box_ = sealbox::seal(&recipient.public_key(), &vec![0xEFu8; N4]).unwrap();
    assert_eq!(verdict_for(N4, || drop(sealbox::open(&recipient, &box_).unwrap())), 1, "sealbox::open");
    let aad = Aad::new(AadDomain::Enc, &["a"]).unwrap();
    let wrapped = aead::wrap(&key, &aad, &vec![0x99u8; N5]).unwrap();
    assert_eq!(verdict_for(N5, || drop(aead::unwrap(&key, &aad, &wrapped).unwrap())), 1, "aead::unwrap");

    // the memory Argon2 works in: 64 MiB of blocks, wiped before they are released ("zeroize ... the Argon2 block", 4.5.2 step 8)
    let block_memory = MEM_MIN as usize;
    assert_eq!(
        verdict_for(block_memory, || drop(argon::argon2id13(b"correct horse battery staple", &[7u8; 16], 3, MEM_MIN).unwrap())),
        1,
        "argon2 blocks"
    );
    // and the control for that: the same derivation with memory that nothing wipes (what the argon2 crate's own `hash_password_into` does)
    let control = verdict_for(block_memory, || {
        let params = argon2::Params::new((MEM_MIN / 1024) as u32, 3, 1, Some(32)).unwrap();
        let mut out = [0u8; 32];
        let mut blocks = vec![argon2::Block::default(); params.block_count()];
        argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params)
            .hash_password_into_with_memory(b"correct horse battery staple", &[7u8; 16], &mut out, &mut blocks[..])
            .unwrap();
    });
    assert_eq!(control, 2, "control: an unwiped block vector is freed with the derived blocks still in it");
}

/// Reads `size_of::<T>()` bytes at `ptr` with volatile loads.
unsafe fn bytes_at<T>(ptr: *const T) -> Vec<u8> {
    let base = ptr as *const u8;
    (0..size_of::<T>()).map(|i| ptr::read_volatile(base.add(i))).collect()
}

/// Moves `value` into a slot, returns its bytes, drops it in place and returns the bytes again.
fn before_and_after<T>(value: T) -> (Vec<u8>, Vec<u8>) {
    let mut slot = MaybeUninit::<T>::new(value);
    let pointer = slot.as_mut_ptr();
    // SAFETY: `slot` holds an initialised `T` that this function owns; it is read, then dropped exactly once with drop_in_place, and the storage is
    // never used as a `T` again (`MaybeUninit` does not drop it a second time). The bytes read afterwards are those of the same, still owned, slot.
    unsafe {
        let before = bytes_at(pointer);
        ptr::drop_in_place(pointer);
        let after = bytes_at(pointer);
        (before, after)
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[test]
fn inline_secrets_are_zero_after_drop() {
    // the secret containers
    let (before, after) = before_and_after(Secret::<32>::new([0xC3; 32]));
    assert!(before.iter().all(|b| *b == 0xC3) && after.iter().all(|b| *b == 0), "Secret<32>");
    let (before, after) = before_and_after(Secret::<16>::new([0xC4; 16]));
    assert!(before.iter().all(|b| *b == 0xC4) && after.iter().all(|b| *b == 0), "Secret<16>");
    let (before, after) = before_and_after(Secret::<64>::new([0xC5; 64]));
    assert!(before.iter().all(|b| *b == 0xC5) && after.iter().all(|b| *b == 0), "Secret<64>");
    // the types that wrap one
    let (before, after) = before_and_after(SecretKey::from_bytes([0xC6; 32]));
    assert!(before.contains(&0xC6) && after.iter().all(|b| *b == 0), "x25519::SecretKey");
    let (before, after) = before_and_after(Entropy::from_bytes([0xC7; 16]));
    assert!(before.iter().all(|b| *b == 0xC7) && after.iter().all(|b| *b == 0), "bip39::Entropy");
    let (before, after) = before_and_after(RecoveryKit::from_bytes(Secret::new([0xC8; 32])));
    assert!(before.iter().all(|b| *b == 0xC8) && after.iter().all(|b| *b == 0), "RecoveryKit");
    // an Ed25519 signing key: find the seed (32 distinct bytes) in the struct, and require that place, and no other, to be zero afterwards
    let seed: [u8; 32] = std::array::from_fn(|i| 0xA0 + i as u8);
    let (before, after) = before_and_after(SigningKey::from_seed(KeyRole::Writer, &Secret::new(seed)));
    let at = find(&before, &seed).expect("the seed is in the key");
    assert!(after[at..at + 32].iter().all(|b| *b == 0), "the seed's place is zero after drop");
    assert!(find(&after, &seed[..8]).is_none(), "no part of the seed remains anywhere in the key");
    // explicit wipe
    let mut secret = Secret::<32>::new([0xC9; 32]);
    secret.wipe();
    assert_eq!(secret.expose(), &[0u8; 32]);
    // a duplicate is a second, independent secret: wiping one leaves the other
    let mut first = Secret::<32>::new([0xCA; 32]);
    let second = first.duplicate();
    first.wipe();
    assert_eq!(second.expose(), &[0xCA; 32]);
}

fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}

#[test]
fn every_secret_type_is_zeroize_on_drop_and_prints_nothing() {
    assert_zeroize_on_drop::<Secret<16>>();
    assert_zeroize_on_drop::<Secret<32>>();
    assert_zeroize_on_drop::<Secret<64>>();
    assert_zeroize_on_drop::<SecretVec>();
    assert_zeroize_on_drop::<SecretString>();
    assert_zeroize_on_drop::<SecretKey>();
    assert_zeroize_on_drop::<SigningKey>();
    assert_zeroize_on_drop::<Entropy>();
    assert_zeroize_on_drop::<RecoveryKit>();

    let printed = [
        format!("{:?}", Secret::<32>::new([0xAB; 32])),
        format!("{:?}", SecretVec::new(vec![0xAB; 8])),
        format!("{:?}", SecretString::new("correct horse battery staple".into())),
        format!("{:?}", SecretKey::from_bytes([0xAB; 32])),
        format!("{:?}", SigningKey::from_seed(KeyRole::Vault, &Secret::new([0xAB; 32]))),
        format!("{:?}", Entropy::from_bytes([0xAB; 16])),
        format!("{:?}", RecoveryKit::from_bytes(Secret::new([0xAB; 32]))),
    ];
    for text in &printed {
        for leak in ["171", "0xab", "abab", "AB", "correct", "horse", "staple"] {
            assert!(!text.contains(leak), "{text} shows {leak}");
        }
        assert!(text.contains("redacted"), "{text}");
    }
    // a public key prints (it is public); a signing key prints its role and no bytes
    assert!(printed[4].contains("Vault"));
}

#[test]
fn comparison_of_secrets_is_constant_time_equality() {
    assert!(ct_eq(b"", b""));
    assert!(ct_eq(b"abc", b"abc"));
    assert!(!ct_eq(b"abc", b"abd"));
    assert!(!ct_eq(b"abc", b"ab"));
    assert!(!ct_eq(b"ab", b"abc"));
    assert!(!ct_eq(b"", b"a"));
    // a difference in any single bit of a 32-byte value is found, wherever it is
    let base = [0x5au8; 32];
    for bit in 0..256 {
        let mut other = base;
        other[bit / 8] ^= 1 << (bit % 8);
        assert!(!ct_eq(&base, &other), "bit {bit}");
        assert!(Secret::new(base) != Secret::new(other), "Secret bit {bit}");
    }
    assert!(Secret::new(base) == Secret::new(base));
    assert!(SecretVec::new(vec![1, 2, 3]) == SecretVec::new(vec![1, 2, 3]));
    assert!(SecretVec::new(vec![1, 2, 3]) != SecretVec::new(vec![1, 2]));
    assert!(SecretString::new("a".into()) == SecretString::new("a".into()));
    assert!(SecretString::new("a".into()) != SecretString::new("b".into()));
}
