//! Keys on the dead stack (review L-7): does a 32-byte key that a function used survive, in the stack space the function has given back, after it returns?
//!
//! The containers of this crate wipe themselves, the heap is covered by `zeroize.rs` and `zeroize_text.rs`, and the reviewer's probe found one more place: after
//! `kdf::derive` returned, **four copies** of the 32-byte master key were still in the dead stack below the caller, left by the BLAKE2b state (its key block is
//! copied when the state is built, and the state is moved into `finalize`, which copies it again; dropping wipes one place, not the copies the moves left). The
//! design's test P9 ("no UMK in a memory scan after connect") would have tripped on it: the UMK is the master key of `kdf::derive` for the backup keys.
//!
//! The probe: a function that calls the code under test is followed, in the same frame, by a function whose local array covers the stack below; a scan of that
//! array (reads of stack memory that nothing has written since, which is the point) counts the places where the key's 32 bytes still are. Controls: a callee that
//! deliberately leaves a copy is found (positive), and a callee that does not is not (negative), so that the scan is neither blind nor seeing things. Every
//! primitive that takes a key is probed, not only the one that leaked: HKDF, HMAC, Ed25519 (from a seed and signing), X25519, the AEAD, sealed boxes and Argon2id.
//!
//! This is the only test in this binary. The `unsafe` is test code: the read of bytes of uninitialised stack, which is undefined in Rust's abstract machine and is
//! what every stack-scanning test of a zeroizing library does (the `zeroize` crate's own tests read the dead part of a struct the same way); a volatile read, so
//! that it is not optimised away. The counts, not the bytes, are what is asserted.

use std::mem::MaybeUninit;

use oaiy_crypto::aead;
use oaiy_crypto::argon;
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::ed25519::{KeyRole, SigningKey};
use oaiy_crypto::kdf::{self, Context};
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::SecretKey;
use oaiy_crypto::zeroize::Secret;

/// How much stack below the caller is looked at.
const WINDOW: usize = 96 * 1024;

/// Counts the places in the stack below the caller where `needle` is, in memory that nothing has written since the callee returned.
#[inline(never)]
fn scan_dead_stack(needle: &[u8]) -> usize {
    let frame = MaybeUninit::<[u8; WINDOW]>::uninit();
    let base = frame.as_ptr().cast::<u8>();
    let mut copy = vec![0u8; WINDOW];
    for (i, slot) in copy.iter_mut().enumerate() {
        // SAFETY: `base..base + WINDOW` is this frame's own storage, which is in bounds; its bytes are whatever the callee that returned before this call left there,
        // which is what is being looked for. A volatile read, so that the compiler does not assume the memory is uninitialised and drop the loop.
        *slot = unsafe { std::ptr::read_volatile(base.add(i)) };
    }
    copy.windows(needle.len()).filter(|w| *w == needle).count()
}

/// Overwrites the stack below the caller with zeros, so that a scan sees only what the next call left.
#[inline(never)]
fn scrub() {
    let mut pad = [0u8; 2 * WINDOW];
    std::hint::black_box(&mut pad);
}

#[inline(never)]
fn run_and_scan(needle: &[u8], f: impl FnOnce()) -> usize {
    scrub();
    f();
    scan_dead_stack(needle)
}

/// The positive control: a callee that leaves a copy of the key in its own frame.
#[inline(never)]
fn leaves_a_copy(key: &[u8; 32]) {
    let mut pad = [0u8; 4096];
    pad[2000..2032].copy_from_slice(key);
    std::hint::black_box(&pad);
}

#[test]
fn no_primitive_leaves_its_key_in_the_dead_stack() {
    let secret_bytes: [u8; 32] = std::array::from_fn(|i| 0xC0 + i as u8);
    // built outside the closures, so that only the library's own copies are counted
    let master = Secret::<32>::new(secret_bytes);
    let seed = Secret::<32>::new(secret_bytes);
    let x_secret = SecretKey::from_bytes(secret_bytes);
    let x_peer = SecretKey::from_bytes([0x62; 32]).public_key();
    let box_key = SecretKey::from_bytes(secret_bytes);
    let box_bytes = sealbox::seal(&box_key.public_key(), b"hi").unwrap();
    let aad = Aad::new(AadDomain::VaultWrap, &["u", "w", "recovery-phrase", "x"]).unwrap();
    let context = Context::new("flbkrcp1").unwrap();

    let positive = run_and_scan(&secret_bytes, || leaves_a_copy(master.expose()));
    let negative = run_and_scan(&secret_bytes, || {
        std::hint::black_box(master.expose().len());
    });
    assert!(positive >= 1, "control: a callee that leaves a copy of the key is found ({positive})");
    assert_eq!(negative, 0, "control: a callee that leaves none finds none");

    let mut counts: Vec<(&str, usize)> = Vec::new();
    counts.push((
        "kdf::derive",
        run_and_scan(&secret_bytes, || {
            drop(kdf::derive(&master, kdf::Purpose::BackupRecipient).unwrap());
        }),
    ));
    counts.push((
        "kdf::derive_subkey",
        run_and_scan(&secret_bytes, || {
            drop(kdf::derive_subkey(&master, 7, &context).unwrap());
        }),
    ));
    counts.push((
        "kdf::derive_subkey_into (64 bytes)",
        run_and_scan(&secret_bytes, || {
            let mut out = [0u8; 64];
            kdf::derive_subkey_into(&master, 1, &context, &mut out).unwrap();
        }),
    ));
    counts.push((
        "kdf::hkdf_sha256",
        run_and_scan(&secret_bytes, || {
            let mut out = [0u8; 64];
            kdf::hkdf_sha256(master.expose(), Some(b"salt"), b"info", &mut out).unwrap();
        }),
    ));
    counts.push((
        "kdf::hmac_sha256",
        run_and_scan(&secret_bytes, || {
            let _ = kdf::hmac_sha256(master.expose(), b"msg").unwrap();
        }),
    ));
    counts.push((
        "ed25519 from_seed",
        run_and_scan(&secret_bytes, || {
            let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
            std::hint::black_box(&key);
        }),
    ));
    counts.push((
        "ed25519 sign",
        run_and_scan(&secret_bytes, || {
            let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
            let _ = key.sign_raw(b"message").unwrap();
        }),
    ));
    counts.push((
        "x25519 diffie_hellman",
        run_and_scan(&secret_bytes, || {
            drop(x_secret.diffie_hellman(&x_peer).unwrap());
        }),
    ));
    counts.push((
        "aead wrap and unwrap",
        run_and_scan(&secret_bytes, || {
            let wrapped = aead::wrap(&master, &aad, b"a payload").unwrap();
            let _ = aead::unwrap(&master, &aad, &wrapped).unwrap();
        }),
    ));
    counts.push((
        "argon2id13 (the key as the password)",
        run_and_scan(&secret_bytes, || {
            let _ = argon::argon2id13(&secret_bytes, &[3u8; 16], 3, argon::MEM_MIN).unwrap();
        }),
    ));
    counts.push((
        "sealbox open",
        run_and_scan(&secret_bytes, || {
            drop(sealbox::open(&box_key, &box_bytes).unwrap());
        }),
    ));

    // What is allowed: nothing, except that a function that *returns a key* (Ed25519 `from_seed`) moves it out of its own frame, and in an optimised build the frame it was
    // moved out of can keep the one copy that the returned value is a copy of. That copy is the key itself, the one the caller holds; what is asserted is that the
    // expansion (dalek's, which left two more) leaves nothing besides it.
    let allowed = |name: &str| usize::from(name == "ed25519 from_seed");
    let leaks: Vec<String> = counts.iter().filter(|(name, n)| *n > allowed(name)).map(|(name, n)| format!("{name}: {n}")).collect();
    assert!(leaks.is_empty(), "copies of a 32-byte key are left in the dead stack: {leaks:?} (all counts: {counts:?})");
}
