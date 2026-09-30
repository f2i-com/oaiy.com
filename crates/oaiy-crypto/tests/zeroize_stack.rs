//! Keys on the dead stack (review L-7, and low 1 of the second review): does a 32-byte key that a function used, **or that it made**, survive in the stack space the
//! function has given back, after it returns?
//!
//! The containers of this crate wipe themselves, the heap is covered by `zeroize.rs` and `zeroize_text.rs`, and this file looks at the one place they cannot reach. Two
//! things are counted, for every primitive that takes or makes a key:
//!
//! - **the input**: the master key, the seed, the password, the wrapping key. After `kdf::derive` returned, the reviewer found four copies of the 32-byte master key below
//!   the caller (the BLAKE2b state copies the key into a block when it is built and is moved into `finalize`, which copies it again; dropping wipes one place, and the moves
//!   left the rest), and in an optimised build the same after HKDF and HMAC, and after Ed25519 `from_seed`. Every one of these must be **zero**, in every build configuration
//!   (see below), which is what `zeroize::scrub_stack` is for. The exception is a function that **returns a key object that holds the input** (Ed25519 `from_seed`): it
//!   leaves none with assertions on, and one in an optimised build.
//! - **the output**, the derived key. A function that returns a key **by value** leaves a copy of it in the frame that made it, once that frame has returned; nothing inside the
//!   function can wipe the frame it is still running in. The floor is what a function with no cryptography in it leaves when it builds a `Secret` and returns it through a
//!   `Result` (`plain_return`, below): none in an optimised build and one in a build with no optimisation. The `*_into(&mut Secret)` variants write the key where the caller says and
//!   must leave **none**, in every configuration; the by-value functions are allowed one more than the floor, and the test prints every count. The keys that are **made at
//!   random** (`Secret::random`, the `generate` functions) come back by value too, and are allowed two more than the floor.
//!
//! The configurations this is run in (see the README): `cargo test` (dev: no optimisation, assertions on), `cargo test --release` (optimised), and
//! `cargo test --profile vault-probe` (no optimisation **and no assertions**, the one that `debug_assertions` used to be mistaken for). The counts of each are printed with `--nocapture`.
//!
//! The probe: a function that calls the code under test is followed, in the same frame, by one whose local array covers the stack below; a scan of that array (reads of stack
//! memory that nothing has written since, which is the point) counts the places where the key's 32 bytes still are. Controls: a callee that deliberately leaves a copy is
//! found (positive), a callee that does not is not (negative), so the scan is neither blind nor seeing things.
//!
//! This is the only test in this binary. The `unsafe` is test code: the read of bytes of uninitialised stack, which is undefined in Rust's abstract machine and is what every
//! stack-scanning test of a zeroizing library does (the `zeroize` crate's own tests read the dead part of a struct the same way); a volatile read, so that it is not optimised
//! away. The counts, not the bytes, are what is asserted.

use std::mem::MaybeUninit;

use oaiy_crypto::aead;
use oaiy_crypto::argon;
use oaiy_crypto::bip39::{self, Entropy};
use oaiy_crypto::canon::{Aad, AadDomain};
use oaiy_crypto::ed25519::{KeyRole, SigningKey};
use oaiy_crypto::kdf::{self, Context, Purpose};
use oaiy_crypto::kit::RecoveryKit;
use oaiy_crypto::sealbox;
use oaiy_crypto::x25519::SecretKey;
use oaiy_crypto::zeroize::Secret;

/// How much stack below the caller is looked at: more than the 96 KiB that `scrub_stack` covers, so that what it missed would be seen.
const WINDOW: usize = 192 * 1024;

/// Counts the places in the stack below the caller where each needle is, in memory that nothing has written since the callee returned.
#[inline(never)]
fn scan_dead_stack(needles: &[&[u8]]) -> Vec<usize> {
    let frame = MaybeUninit::<[u8; WINDOW]>::uninit();
    let base = frame.as_ptr().cast::<u8>();
    let mut copy = vec![0u8; WINDOW];
    for (i, slot) in copy.iter_mut().enumerate() {
        // SAFETY: `base..base + WINDOW` is this frame's own storage, which is in bounds; its bytes are whatever the callee that returned before this call left there,
        // which is what is being looked for. A volatile read, so that the compiler does not assume the memory is uninitialised and drop the loop.
        *slot = unsafe { std::ptr::read_volatile(base.add(i)) };
    }
    needles.iter().map(|needle| copy.windows(needle.len()).filter(|w| w == needle).count()).collect()
}

/// Overwrites the stack below the caller with zeros, so that a scan sees only what the next call left.
#[inline(never)]
fn scrub() {
    let mut pad = [0u8; 2 * WINDOW];
    std::hint::black_box(&mut pad);
}

/// A frame of some depth between the caller and the code under test, so that the code does not always run at the same offset (the counts must not depend on it).
#[inline(never)]
fn padded(f: &mut dyn FnMut()) {
    let pad = [0u8; 512];
    std::hint::black_box(&pad);
    f();
}

#[inline(never)]
fn run_and_scan(needles: &[&[u8]], deep: bool, mut f: impl FnMut()) -> Vec<usize> {
    scrub();
    if deep {
        padded(&mut f);
    } else {
        f();
    }
    scan_dead_stack(needles)
}

/// The positive control: a callee that leaves a copy of the key in its own frame.
#[inline(never)]
fn leaves_a_copy(key: &[u8; 32]) {
    let mut pad = [0u8; 4096];
    pad[2000..2032].copy_from_slice(key);
    std::hint::black_box(&pad);
}

/// The floor for a by-value result: no cryptography at all, a `Secret` built from a value and returned by move through `Result`, unwrapped and dropped, as any caller of a
/// function that returns a key does.
#[inline(never)]
fn plain_return(x: &[u8; 32]) -> Result<Secret<32>, oaiy_crypto::Error> {
    let mut out = *x;
    let secret = Secret::new(out);
    zeroize::Zeroize::zeroize(&mut out);
    Ok(secret)
}

/// Like `run_and_scan`, for a function that makes a key at random: the needle is what it made, so `f` returns a copy of it **on the heap** (a copy in the frame would be the test's own).
#[inline(never)]
fn run_and_scan_made(deep: bool, mut f: impl FnMut() -> Vec<u8>) -> usize {
    scrub();
    let mut needle = Vec::new();
    if deep {
        padded(&mut || needle = f());
    } else {
        needle = f();
    }
    scan_dead_stack(&[&needle])[0]
}

struct Row {
    name: &'static str,
    input: usize,
    output: usize,
}

#[test]
fn no_primitive_leaves_its_key_in_the_dead_stack_and_the_into_variants_leave_no_derived_key() {
    let configuration = if cfg!(debug_assertions) { "assertions on" } else { "assertions off" };
    let master_bytes: [u8; 32] = std::array::from_fn(|i| 0xC0 + i as u8);
    let inner_bytes: [u8; 32] = std::array::from_fn(|i| 0x10 + i as u8);
    // built outside the closures, so that only the library's own copies are counted
    let master = Secret::<32>::new(master_bytes);
    let seed = Secret::<32>::new(master_bytes);
    let x_secret = SecretKey::from_bytes(master_bytes);
    let x_peer = SecretKey::from_bytes([0x62; 32]).public_key();
    let box_key = SecretKey::from_bytes(master_bytes);
    let box_bytes = sealbox::seal(&box_key.public_key(), b"hi").unwrap();
    let aad = Aad::new(AadDomain::VaultWrap, &["u", "w", "recovery-phrase", "x"]).unwrap();
    let wrapped = aead::wrap_key(&master, &aad, &Secret::new(inner_bytes)).unwrap();
    let context = Context::new("flbkrcp1").unwrap();
    let entropy = Entropy::from_bytes([0x5a; 16]);
    let salt = [3u8; 16];
    let kit = RecoveryKit::from_bytes(Secret::new(master_bytes));

    // the derived values, computed once, so that they can be searched for
    let d_backup = *kdf::derive(&master, Purpose::BackupRecipient).unwrap().expose();
    let d_hkdf = *kdf::hkdf_sha256_secret::<32>(master.expose(), Some(b"salt"), b"info").unwrap().expose();
    let d_kit = *kit.wrap_key().unwrap().expose();
    let d_dh = *x_secret.diffie_hellman(&x_peer).unwrap().expose();
    let d_ikm = *argon::argon2id13(entropy.expose(), &salt, 3, argon::MEM_MIN).unwrap().expose();
    let d_phrase = *bip39::wrap_key(&entropy, &salt, 3, argon::MEM_MIN).unwrap().expose();

    let mut table: Vec<Row> = Vec::new();
    let mut controls: Vec<(String, usize, usize)> = Vec::new();
    for deep in [false, true] {
        let tag = if deep { "deep" } else { "top" };
        let positive = run_and_scan(&[&master_bytes], deep, || leaves_a_copy(master.expose()))[0];
        let negative = run_and_scan(&[&master_bytes], deep, || {
            std::hint::black_box(master.expose().len());
        })[0];
        assert!(positive >= 1, "{tag} control: a callee that leaves a copy of the key is found ({positive})");
        assert_eq!(negative, 0, "{tag} control: a callee that leaves none finds none");
        let floor = run_and_scan(&[&d_backup], deep, || drop(plain_return(&d_backup).unwrap()))[0];
        controls.push((tag.to_string(), positive, floor));

        let mut row = |name: &'static str, input: &[u8], output: &[u8], mut f: Box<dyn FnMut() + '_>| {
            let counts = run_and_scan(&[input, output], deep, &mut f);
            table.push(Row { name, input: counts[0], output: counts[1] });
            println!("ZEROIZE_STACK {configuration}, {tag}: {name}: input={} output={}", counts[0], counts[1]);
        };

        // the keys that come back: by value, and in place
        row("kdf::derive", &master_bytes, &d_backup, Box::new(|| drop(kdf::derive(&master, Purpose::BackupRecipient).unwrap())));
        row(
            "kdf::derive_into",
            &master_bytes,
            &d_backup,
            Box::new(|| {
                let mut out = Secret::zeroed();
                kdf::derive_into(&master, Purpose::BackupRecipient, &mut out).unwrap();
            }),
        );
        row(
            "kdf::hkdf_sha256_secret",
            &master_bytes,
            &d_hkdf,
            Box::new(|| drop(kdf::hkdf_sha256_secret::<32>(master.expose(), Some(b"salt"), b"info").unwrap())),
        );
        row(
            "kdf::hkdf_sha256_secret_into",
            &master_bytes,
            &d_hkdf,
            Box::new(|| {
                let mut out = Secret::<32>::zeroed();
                kdf::hkdf_sha256_secret_into(master.expose(), Some(b"salt"), b"info", &mut out).unwrap();
            }),
        );
        row("kit.wrap_key", &master_bytes, &d_kit, Box::new(|| drop(kit.wrap_key().unwrap())));
        row(
            "kit.wrap_key_into",
            &master_bytes,
            &d_kit,
            Box::new(|| {
                let mut out = Secret::zeroed();
                kit.wrap_key_into(&mut out).unwrap();
            }),
        );
        row("x25519 diffie_hellman", &master_bytes, &d_dh, Box::new(|| drop(x_secret.diffie_hellman(&x_peer).unwrap())));
        row(
            "x25519 diffie_hellman_into",
            &master_bytes,
            &d_dh,
            Box::new(|| {
                let mut out = Secret::zeroed();
                x_secret.diffie_hellman_into(&x_peer, &mut out).unwrap();
            }),
        );
        row("aead::unwrap_key", &master_bytes, &inner_bytes, Box::new(|| drop(aead::unwrap_key(&master, &aad, &wrapped).unwrap())));
        row(
            "aead::unwrap_key_into",
            &master_bytes,
            &inner_bytes,
            Box::new(|| {
                let mut out = Secret::zeroed();
                aead::unwrap_key_into(&master, &aad, &wrapped, &mut out).unwrap();
            }),
        );
        row("bip39::wrap_key", &[0x5a; 16], &d_phrase, Box::new(|| drop(bip39::wrap_key(&entropy, &salt, 3, argon::MEM_MIN).unwrap())));
        row(
            "bip39::wrap_key_into",
            &[0x5a; 16],
            &d_phrase,
            Box::new(|| {
                let mut out = Secret::zeroed();
                bip39::wrap_key_into(&entropy, &salt, 3, argon::MEM_MIN, &mut out).unwrap();
            }),
        );
        row(
            "argon2id13 (the key as the password)",
            &master_bytes,
            &d_ikm,
            Box::new(|| drop(argon::argon2id13(&master_bytes, &salt, 3, argon::MEM_MIN).unwrap())),
        );

        // the keys that go in, and nothing comes back that is a key
        row(
            "kdf::derive_subkey_into (64 bytes)",
            &master_bytes,
            &[0xA5; 32], // the output is the caller's own array: not counted
            Box::new(|| {
                let mut out = [0u8; 64];
                kdf::derive_subkey_into(&master, 1, &context, &mut out).unwrap();
            }),
        );
        row(
            "kdf::hkdf_sha256",
            &master_bytes,
            &[0xA5; 32], // the output is the caller's own array: not counted
            Box::new(|| {
                let mut out = [0u8; 64];
                kdf::hkdf_sha256(master.expose(), Some(b"salt"), b"info", &mut out).unwrap();
            }),
        );
        row(
            "kdf::hmac_sha256",
            &master_bytes,
            &master_bytes,
            Box::new(|| {
                let _ = kdf::hmac_sha256(master.expose(), b"msg").unwrap();
            }),
        );
        row(
            "kdf::hmac_sha256_verify",
            &master_bytes,
            &master_bytes,
            Box::new(|| {
                let _ = std::hint::black_box(kdf::hmac_sha256_verify(master.expose(), b"msg", &[0u8; 32]));
            }),
        );
        row(
            "ed25519 from_seed",
            &master_bytes,
            &master_bytes,
            Box::new(|| {
                let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
                std::hint::black_box(&key);
            }),
        );
        row(
            "ed25519 from_seed + sign",
            &master_bytes,
            &master_bytes,
            Box::new(|| {
                let key = SigningKey::from_seed(KeyRole::Hazmat, &seed);
                let _ = key.sign_raw(b"message").unwrap();
            }),
        );
        row(
            "aead wrap + unwrap",
            &master_bytes,
            &master_bytes,
            Box::new(|| {
                let w = aead::wrap(&master, &aad, b"a payload").unwrap();
                let _ = aead::unwrap(&master, &aad, &w).unwrap();
            }),
        );
        row("sealbox open", &master_bytes, &master_bytes, Box::new(|| drop(sealbox::open(&box_key, &box_bytes).unwrap())));
    }
    // the keys that are made at random: by value, so held to the floor like the rest
    let mut made: Vec<(&str, usize)> = Vec::new();
    for deep in [false, true] {
        let tag = if deep { "deep" } else { "top" };
        let mut one = |name: &'static str, f: &mut dyn FnMut() -> Vec<u8>| {
            let count = run_and_scan_made(deep, &mut *f);
            println!("ZEROIZE_STACK {configuration}, {tag}: {name}: made={count}");
            made.push((name, count));
        };
        one("Secret::random", &mut || Secret::<32>::random().unwrap().expose().to_vec());
        one("x25519 SecretKey::generate", &mut || SecretKey::generate().unwrap().to_secret().expose().to_vec());
        one("ed25519 SigningKey::generate", &mut || SigningKey::generate(KeyRole::Hazmat).unwrap().seed().expose().to_vec());
        one("bip39 Entropy::random", &mut || Entropy::random().unwrap().expose().to_vec());
        one("RecoveryKit::generate", &mut || RecoveryKit::generate().unwrap().key().expose().to_vec());
    }
    println!("ZEROIZE_STACK {configuration}: controls (tag, positive, by-value floor) {controls:?}");

    // what is asserted
    let floor = controls.iter().map(|(_, _, floor)| *floor).max().unwrap_or(0);
    let mut failures: Vec<String> = Vec::new();
    for row in &table {
        let is_into = row.name.ends_with("_into");
        // the input: never, except that Ed25519 `from_seed` **returns** a key that is the seed (the dalek key holds it), by value. Measured with the scrub after the expansion:
        // none with assertions on and none with no optimisation, and **one** in an optimised build (the copy that the frame of `from_seed` keeps when it moves the key out).
        // Without the scrub (mutant N06) it is one more in an optimised build and one in a build with assertions on: over the allowance in every configuration, so the
        // debug suite kills it too, not only the release lane (review low 1).
        let input_allowed = if row.name.starts_with("ed25519 from_seed") { usize::from(!cfg!(debug_assertions)) } else { 0 };
        if row.input > input_allowed {
            failures.push(format!("{}: {} copies of the input key", row.name, row.input));
        }
        // the output: none from an `_into`; from a by-value function no more than one above the floor of a function with no cryptography in it
        if is_into && row.output > 0 {
            failures.push(format!("{}: {} copies of the derived key", row.name, row.output));
        }
        let by_value_derived = matches!(
            row.name,
            "kdf::derive"
                | "kdf::hkdf_sha256_secret"
                | "kit.wrap_key"
                | "x25519 diffie_hellman"
                | "aead::unwrap_key"
                | "bip39::wrap_key"
                | "argon2id13 (the key as the password)"
        );
        if by_value_derived && row.output > floor + 1 {
            failures.push(format!("{}: {} copies of the derived key, more than one above the floor of {floor}", row.name, row.output));
        }
    }
    // the keys made at random come back by value, so they are held to the floor as well, with room for the copy that `from_seed` keeps in an optimised build: measured, none to two
    for (name, count) in &made {
        if *count > floor + 2 {
            failures.push(format!("{name}: {count} copies of the key that was made, more than two above the floor of {floor}"));
        }
    }
    assert!(failures.is_empty(), "{configuration}: {failures:#?}");
}
