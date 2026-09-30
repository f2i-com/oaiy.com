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
