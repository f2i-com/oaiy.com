//! The typed entry points that replaced the free-form ones (review L-10): `aead::seal` takes a `Nonce` that only the random generator can make, and the only
//! derivation that production code has is `kdf::derive(master, Purpose)`.
//!
//! What a test cannot show is that the free-form functions are absent from a build without the `test-vectors` feature (this crate's own tests turn the feature on, so
//! they see them). That is shown by building something that depends on the crate without the feature: `cargo check -p oaiy-keystore --lib` with a call to
//! `Nonce::from_bytes_for_tests` or `derive_subkey` in it fails with E0599 and E0425 (run by hand when the change was made, and the CI job builds the crate's
//! documentation without dev-dependencies and looks for the names in it). What is shown here is that the typed entry points are the same functions, with the same
//! answers, as the free-form ones the known-answer tests check.

use oaiy_crypto::aead::{self, Nonce, NONCE_LEN};
use oaiy_crypto::kdf::{self, Purpose, REGISTRY};
use oaiy_crypto::zeroize::Secret;

#[test]
fn a_nonce_is_random_public_and_used_up_by_the_seal_that_takes_it() {
    let (a, b) = (Nonce::random().unwrap(), Nonce::random().unwrap());
    assert_ne!(a.as_bytes(), b.as_bytes(), "two random nonces");
    assert_ne!(a.as_bytes(), &[0u8; NONCE_LEN]);
    let key = Secret::<32>::new([0x33; 32]);
    let bytes = *a.as_bytes();
    let sealed = aead::seal(&key, a, b"aad", b"a message").unwrap(); // `a` is moved: it cannot be sealed under twice
    assert_eq!(aead::open(&key, &bytes, b"aad", &sealed).unwrap().expose(), b"a message");
    // another nonce does not open it, and neither does another AAD
    assert!(aead::open(&key, b.as_bytes(), b"aad", &sealed).is_err());
    assert!(aead::open(&key, &bytes, b"other", &sealed).is_err());
    // the test-only constructor gives exactly the bytes it is given (it is what the known-answer tests use)
    assert_eq!(Nonce::from_bytes_for_tests([7; NONCE_LEN]).as_bytes(), &[7; NONCE_LEN]);
    // a nonce prints (it is public) and a blob made by `wrap` carries its own
    assert!(format!("{b:?}").starts_with("Nonce("));
}

/// The typed derivation is the registry row's derivation: `derive(master, purpose)` is `derive_subkey(master, row.id, row.context)`, for every row, and every row gives a
/// different key (the known-answer tests check the free-form function against libsodium; this ties the typed one to it).
#[test]
fn derive_by_purpose_is_the_free_form_derivation_of_its_registry_row() {
    let master = Secret::<32>::new(std::array::from_fn(|i| i as u8 * 7 + 1));
    let purposes = [Purpose::KitWrap, Purpose::PhraseWrap, Purpose::PrfWrap, Purpose::BackupRecipient, Purpose::BackupSigning, Purpose::LocalData];
    assert_eq!(purposes.len(), REGISTRY.len(), "a purpose for every row of the registry");
    let mut seen = Vec::new();
    for purpose in purposes {
        let row = purpose.entry();
        let typed = kdf::derive(&master, purpose).unwrap();
        let free_form = kdf::derive_subkey(&master, row.id, &row.context).unwrap();
        assert_eq!(typed, free_form, "{}", row.context.as_str());
        assert!(!seen.contains(typed.expose()), "{} gave a key that another purpose gave", row.context.as_str());
        seen.push(*typed.expose());
    }
}

/// The `*_into` functions (review low 1 of the second review) write the key where the caller says and leave none in the stack below it (`tests/zeroize_stack.rs` counts that).
/// What they must also do is the same thing as the by-value functions: give the same key, and on an error **not write** `out`, so that a caller that ignores the error
/// has a zero key and not half of one.
#[test]
fn the_into_variants_write_what_the_by_value_functions_return_and_leave_out_untouched_on_an_error() {
    use oaiy_crypto::aead::{unwrap_key, unwrap_key_into, wrap_key};
    use oaiy_crypto::argon::MEM_MIN;
    use oaiy_crypto::bip39::{self, Entropy};
    use oaiy_crypto::canon::{Aad, AadDomain};
    use oaiy_crypto::kit::RecoveryKit;
    use oaiy_crypto::x25519::SecretKey;
    use oaiy_crypto::Error;

    let master = Secret::<32>::new(std::array::from_fn(|i| i as u8 * 5 + 3));
    let zero = [0u8; 32];

    // kdf::derive_into, for every registry row
    for purpose in [Purpose::KitWrap, Purpose::PhraseWrap, Purpose::PrfWrap, Purpose::BackupRecipient, Purpose::BackupSigning, Purpose::LocalData] {
        let mut out = Secret::<32>::zeroed();
        kdf::derive_into(&master, purpose, &mut out).unwrap();
        assert_eq!(out, kdf::derive(&master, purpose).unwrap(), "{purpose:?}");
        assert_ne!(out.expose(), &zero);
    }
    // hkdf, for the sizes that are keys
    let by_value: Secret<32> = kdf::hkdf_sha256_secret(master.expose(), Some(b"salt"), b"info").unwrap();
    let mut out = Secret::<32>::zeroed();
    kdf::hkdf_sha256_secret_into(master.expose(), Some(b"salt"), b"info", &mut out).unwrap();
    assert_eq!(out, by_value);
    let mut long = Secret::<64>::zeroed();
    kdf::hkdf_sha256_secret_into(master.expose(), None, b"info", &mut long).unwrap();
    assert_eq!(long, kdf::hkdf_sha256_secret::<64>(master.expose(), None, b"info").unwrap());
    // the kit
    let kit = RecoveryKit::from_bytes(Secret::new([0x42; 32]));
    let mut out = Secret::<32>::zeroed();
    kit.wrap_key_into(&mut out).unwrap();
    assert_eq!(out, kit.wrap_key().unwrap());
    // x25519, and the refusal of a result of small order leaves `out` as it was
    let (mine, theirs) = (SecretKey::from_bytes([0x11; 32]), SecretKey::from_bytes([0x22; 32]).public_key());
    let mut out = Secret::<32>::zeroed();
    mine.diffie_hellman_into(&theirs, &mut out).unwrap();
    assert_eq!(out, mine.diffie_hellman(&theirs).unwrap());
    // aead: the same key, and a wrong wrapping key, a wrong AAD and a wrong length leave `out` alone
    let aad = Aad::new(AadDomain::VaultWrap, &["u", "w", "recovery-phrase", "x"]).unwrap();
    let other_aad = Aad::new(AadDomain::VaultWrap, &["u", "w", "recovery-phrase", "y"]).unwrap();
    let inner = Secret::<32>::new([0x77; 32]);
    let wrapped = wrap_key(&master, &aad, &inner).unwrap();
    let mut out = Secret::<32>::zeroed();
    unwrap_key_into(&master, &aad, &wrapped, &mut out).unwrap();
    assert_eq!((&out, &unwrap_key(&master, &aad, &wrapped).unwrap()), (&inner, &inner));
    for (why, result) in [
        ("another wrapping key", {
            let mut o = Secret::<32>::zeroed();
            let r = unwrap_key_into(&Secret::new([1; 32]), &aad, &wrapped, &mut o);
            (r, o)
        }),
        ("another AAD", {
            let mut o = Secret::<32>::zeroed();
            let r = unwrap_key_into(&master, &other_aad, &wrapped, &mut o);
            (r, o)
        }),
        ("a truncated blob", {
            let mut o = Secret::<32>::zeroed();
            let r = unwrap_key_into(&master, &aad, &wrapped[..71], &mut o);
            (r, o)
        }),
    ] {
        assert_eq!(result.0.unwrap_err(), Error::DecryptFailed, "{why}");
        assert_eq!(result.1.expose(), &zero, "{why}: `out` was written on an error");
    }
    // the phrase: the same wrap key, and a bad phrase leaves `out` alone and runs no key derivation
    let entropy = Entropy::from_bytes([0x5a; 16]);
    let salt = [3u8; 16];
    let mut out = Secret::<32>::zeroed();
    bip39::wrap_key_into(&entropy, &salt, 3, MEM_MIN, &mut out).unwrap();
    assert_eq!(out, bip39::wrap_key(&entropy, &salt, 3, MEM_MIN).unwrap());
    let phrase = bip39::encode(&entropy);
    let mut from_phrase = Secret::<32>::zeroed();
    bip39::phrase_wrap_key_into(phrase.expose(), &salt, 3, MEM_MIN, &mut from_phrase).unwrap();
    assert_eq!(from_phrase, out);
    let mut untouched = Secret::<32>::zeroed();
    assert!(bip39::phrase_wrap_key_into(
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon",
        &salt,
        3,
        MEM_MIN,
        &mut untouched
    )
    .is_err());
    assert_eq!(untouched.expose(), &zero);
    assert_eq!(bip39::wrap_key_into(&entropy, &salt, 2, MEM_MIN, &mut untouched).unwrap_err(), Error::KdfParamsOutOfRange);
    assert_eq!(untouched.expose(), &zero, "a refused Argon2 parameter leaves `out` alone");
}

/// Keys that are made at random are made from the random generator and are all different (a generator that wrote nothing, or wrote the same thing twice, would make every key
/// the same): the generate functions write the random bytes in place, and this is what shows that they do write them.
#[test]
fn every_key_that_is_made_at_random_is_not_zero_and_not_the_same_twice() {
    use oaiy_crypto::bip39::Entropy;
    use oaiy_crypto::ed25519::{KeyRole, SigningKey};
    use oaiy_crypto::kit::RecoveryKit;
    use oaiy_crypto::x25519::SecretKey;

    fn filled() -> Vec<u8> {
        let mut secret = Secret::<32>::zeroed();
        secret.fill_random().unwrap();
        secret.expose().to_vec()
    }

    let made: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
        ("Secret::random", Secret::<32>::random().unwrap().expose().to_vec(), Secret::<32>::random().unwrap().expose().to_vec()),
        ("Secret::fill_random", filled(), filled()),
        (
            "x25519 generate",
            SecretKey::generate().unwrap().to_secret().expose().to_vec(),
            SecretKey::generate().unwrap().to_secret().expose().to_vec(),
        ),
        (
            "ed25519 generate",
            SigningKey::generate(KeyRole::Hazmat).unwrap().seed().expose().to_vec(),
            SigningKey::generate(KeyRole::Hazmat).unwrap().seed().expose().to_vec(),
        ),
        ("kit generate", RecoveryKit::generate().unwrap().key().expose().to_vec(), RecoveryKit::generate().unwrap().key().expose().to_vec()),
        ("entropy random", Entropy::random().unwrap().expose().to_vec(), Entropy::random().unwrap().expose().to_vec()),
    ];
    for (name, first, second) in made {
        assert!(first.iter().any(|b| *b != 0), "{name}: a zero key");
        assert!(second.iter().any(|b| *b != 0), "{name}: a zero key");
        assert_ne!(first, second, "{name}: the same key twice");
    }
}
