//! The signature an update is checked against.
//!
//! The release workflow signs each installer an update can install (minisign, through the Tauri
//! CLI) and this copy of OAIY carries the public key (`plugins.updater.pubkey` in
//! `tauri.conf.json`). A download is an update only when [`verify_package`] accepts it, and
//! that is the only way to make a [`VerifiedPackage`]: the state that lets an update be
//! installed holds one, so an installer whose signature failed, or was never checked, cannot
//! reach the install step. (The updater plugin checks a download with the same key too, inside
//! its own `download`; this is the second check, over the same bytes, with code that has tests.)
//!
//! `signature` is the CONTENT of the `.sig` file: the base64 of the minisign signature file, and
//! `pubkey` the base64 of the minisign public key file, both as Tauri writes them.

use base64::Engine as _;
use minisign_verify::{PublicKey, Signature};

/// An installer whose signature verified against the key this copy of OAIY carries.
#[derive(Debug, PartialEq, Eq)]
pub struct VerifiedPackage {
    version: String,
    bytes: Vec<u8>,
}

impl VerifiedPackage {
    /// The version the feed announced, and the signature was checked for.
    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The key this build carries cannot be read.
    BadPublicKey,
    /// The signature the feed gave is not a minisign signature.
    BadSignature,
    /// The bytes are not what the signature signs (or it was made with another key).
    DoesNotMatch,
    /// The signature was made for another version than the feed announced (a feed pairing a newer version with an older release).
    SignedForOtherVersion { signed: String, announced: String },
    /// Versions must be signed, and this signature does not say which one it is for.
    NoSignedVersion,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::BadPublicKey => write!(f, "This copy of OAIY has no usable key to check updates with, so the update was refused."),
            VerifyError::BadSignature => write!(f, "The update's signature is not in a form that can be read, so the update was refused."),
            VerifyError::DoesNotMatch => write!(f, "The downloaded update does not match its signature, so it was thrown away. It may be damaged, or not made by OAIY."),
            VerifyError::SignedForOtherVersion { signed, announced } => write!(f, "The update was signed for version {signed} but the update information named {announced}, so it was refused."),
            VerifyError::NoSignedVersion => write!(f, "The update's signature does not say which version it is for, so it was refused."),
        }
    }
}

impl std::error::Error for VerifyError {}

fn decode_text(b64: &str) -> Option<String> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim()).ok()?;
    String::from_utf8(bytes).ok()
}

/// The version a signature was made for: the `version:` field of its trusted comment (tab-separated `key:value` pairs).
fn signed_version(trusted_comment: &str) -> Option<&str> {
    trusted_comment.split('\t').find_map(|field| field.strip_prefix("version:"))
}

/// Check `bytes` against `signature` with `pubkey`, and the version the signature was made for against `announced_version`.
///
/// `require_signed_version` is `plugins.updater.requireSignedVersion`: when on, a signature that
/// names no version is refused. A signature that DOES name one must always match what the feed
/// announced, whatever that setting: a mismatch is a feed that pairs a new version number with an
/// older, genuinely signed release.
pub fn verify_package(bytes: Vec<u8>, signature: &str, pubkey: &str, announced_version: &str, require_signed_version: bool) -> Result<VerifiedPackage, VerifyError> {
    let key = decode_text(pubkey).and_then(|text| PublicKey::decode(&text).ok()).ok_or(VerifyError::BadPublicKey)?;
    let signature = decode_text(signature).and_then(|text| Signature::decode(&text).ok()).ok_or(VerifyError::BadSignature)?;
    // Legacy (non-prehashed) signatures are accepted as the updater plugin accepts them.
    key.verify(&bytes, &signature, true).map_err(|_| VerifyError::DoesNotMatch)?;
    // Only now is the trusted comment usable: verify() also checked the global signature that covers it.
    match signed_version(signature.trusted_comment()) {
        Some(signed) => {
            let same = match (super::version::parse(signed), super::version::parse(announced_version)) {
                (Ok(signed), Ok(announced)) => signed == announced,
                _ => signed == announced_version,
            };
            if !same {
                return Err(VerifyError::SignedForOtherVersion { signed: signed.to_string(), announced: announced_version.to_string() });
            }
        }
        None if require_signed_version => return Err(VerifyError::NoSignedVersion),
        None => {}
    }
    Ok(VerifiedPackage { version: announced_version.to_string(), bytes })
}

/// Test-only signing in the minisign format, so the tests need no fixture and no key file.
#[cfg(test)]
pub(crate) mod testing {
    use base64::Engine as _;
    use ed25519_dalek::{Signer, SigningKey};

    pub struct Keys {
        pub pubkey: String,
        key: SigningKey,
        key_id: [u8; 8],
    }

    impl Keys {
        /// A key made from `seed` (another seed is another key).
        pub fn new(seed: u8) -> Keys {
            Keys::with_id(seed, seed)
        }

        /// A key made from `seed`, named `id`: two seeds with one id are two keys a signature cannot tell apart by name.
        pub fn with_id(seed: u8, id: u8) -> Keys {
            let key = SigningKey::from_bytes(&[seed; 32]);
            let key_id = [id; 8];
            let mut file = vec![0x45, 0x64];
            file.extend_from_slice(&key_id);
            file.extend_from_slice(key.verifying_key().as_bytes());
            let text = format!("untrusted comment: minisign public key: {}\n{}\n", key_id.iter().map(|b| format!("{b:02X}")).collect::<String>(), base64::engine::general_purpose::STANDARD.encode(file));
            Keys { pubkey: base64::engine::general_purpose::STANDARD.encode(text), key, key_id }
        }

        /// The content of a `.sig` file for `data` with `trusted_comment`.
        pub fn sign(&self, data: &[u8], trusted_comment: &str) -> String {
            let signature = self.key.sign(data).to_bytes();
            let mut first = vec![0x45, 0x64];
            first.extend_from_slice(&self.key_id);
            first.extend_from_slice(&signature);
            let mut covered = signature.to_vec();
            covered.extend_from_slice(trusted_comment.as_bytes());
            let global = self.key.sign(&covered).to_bytes();
            let text = format!(
                "untrusted comment: signature from tauri secret key\n{}\ntrusted comment: {trusted_comment}\n{}\n",
                base64::engine::general_purpose::STANDARD.encode(first),
                base64::engine::general_purpose::STANDARD.encode(global)
            );
            base64::engine::general_purpose::STANDARD.encode(text)
        }
    }

    /// The trusted comment the Tauri CLI writes for a release: a time, the file, the version it was built as.
    pub fn comment(version: &str) -> String {
        format!("timestamp:1790000000\tfile:oaiy-setup.exe\tversion:{version}")
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{comment, Keys};
    use super::*;

    const INSTALLER: &[u8] = b"pretend this is a 100 MB installer";

    fn accepted(keys: &Keys, data: &[u8], sig: &str, announced: &str, require: bool) -> Result<VerifiedPackage, VerifyError> {
        verify_package(data.to_vec(), sig, &keys.pubkey, announced, require)
    }

    #[test]
    fn an_installer_signed_with_the_key_this_build_carries_verifies() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        let package = accepted(&keys, INSTALLER, &sig, "0.2.0", true).unwrap();
        assert_eq!(package.bytes(), INSTALLER);
        assert_eq!(package.version(), "0.2.0");
        assert_eq!(package.len(), INSTALLER.len());
    }

    #[test]
    fn an_installer_that_was_changed_by_a_byte_is_refused() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        let mut tampered = INSTALLER.to_vec();
        tampered[10] ^= 1;
        assert_eq!(accepted(&keys, &tampered, &sig, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
        // Cut short, or with something added.
        assert_eq!(accepted(&keys, &INSTALLER[..INSTALLER.len() - 1], &sig, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
        let mut longer = INSTALLER.to_vec();
        longer.push(0);
        assert_eq!(accepted(&keys, &longer, &sig, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
        assert_eq!(accepted(&keys, b"", &sig, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
    }

    #[test]
    fn an_installer_signed_with_another_key_is_refused() {
        let ours = Keys::new(7);
        let theirs = Keys::new(9);
        let sig = theirs.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(accepted(&ours, INSTALLER, &sig, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
        // A signature that names our key id but was made by another key: the check itself refuses it, not just the name.
        let ours = Keys::with_id(7, 1);
        let forger = Keys::with_id(9, 1);
        let forged = forger.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(accepted(&ours, INSTALLER, &forged, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
        assert!(accepted(&ours, INSTALLER, &ours.sign(INSTALLER, &comment("0.2.0")), "0.2.0", true).is_ok());
    }

    #[test]
    fn a_trusted_comment_that_was_changed_is_refused() {
        // The comment is covered by the global signature: pointing the signature at another version by editing it fails.
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.1.5"));
        let text = String::from_utf8(base64::engine::general_purpose::STANDARD.decode(&sig).unwrap()).unwrap();
        let edited = text.replace("version:0.1.5", "version:0.2.0");
        let forged = base64::engine::general_purpose::STANDARD.encode(edited);
        assert_eq!(accepted(&keys, INSTALLER, &forged, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
    }

    #[test]
    fn a_signature_for_another_version_than_the_feed_announced_is_refused() {
        // The feed says 9.9.9 but points at the genuinely signed 0.1.5: a downgrade dressed as an update.
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.1.5"));
        assert_eq!(
            accepted(&keys, INSTALLER, &sig, "9.9.9", true).unwrap_err(),
            VerifyError::SignedForOtherVersion { signed: "0.1.5".into(), announced: "9.9.9".into() }
        );
        // Also when versions are not required: a version that IS there must match.
        assert!(matches!(accepted(&keys, INSTALLER, &sig, "9.9.9", false), Err(VerifyError::SignedForOtherVersion { .. })));
        // The same version spelled with a v is the same version.
        assert!(accepted(&keys, INSTALLER, &keys.sign(INSTALLER, &comment("v0.2.0")), "0.2.0", true).is_ok());
    }

    #[test]
    fn a_signature_that_names_no_version_is_refused_only_when_versions_are_required() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, "timestamp:1790000000\tfile:oaiy-setup.exe");
        assert_eq!(accepted(&keys, INSTALLER, &sig, "0.2.0", true).unwrap_err(), VerifyError::NoSignedVersion);
        assert!(accepted(&keys, INSTALLER, &sig, "0.2.0", false).is_ok());
    }

    #[test]
    fn a_key_or_a_signature_that_cannot_be_read_is_refused_in_words_and_never_accepted() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        for bad in ["", "not base64!", "aGVsbG8="] {
            assert_eq!(verify_package(INSTALLER.to_vec(), &sig, bad, "0.2.0", true).unwrap_err(), VerifyError::BadPublicKey, "key {bad:?}");
            assert_eq!(verify_package(INSTALLER.to_vec(), bad, &keys.pubkey, "0.2.0", true).unwrap_err(), VerifyError::BadSignature, "signature {bad:?}");
        }
        // A public key file where the signature belongs, and the other way round.
        assert_eq!(verify_package(INSTALLER.to_vec(), &keys.pubkey, &keys.pubkey, "0.2.0", true).unwrap_err(), VerifyError::BadSignature);
        assert_eq!(verify_package(INSTALLER.to_vec(), &sig, &sig, "0.2.0", true).unwrap_err(), VerifyError::BadPublicKey);
        for error in [VerifyError::BadPublicKey, VerifyError::BadSignature, VerifyError::DoesNotMatch, VerifyError::NoSignedVersion] {
            assert!(error.to_string().contains("refused") || error.to_string().contains("thrown away"), "{error}");
        }
    }

    #[test]
    fn the_key_this_build_carries_is_a_readable_public_key() {
        // tauri.conf.json's plugins.updater.pubkey: what every installed copy verifies with.
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        let pubkey = conf["plugins"]["updater"]["pubkey"].as_str().unwrap();
        let text = decode_text(pubkey).unwrap();
        assert!(text.starts_with("untrusted comment: minisign public key: "), "{text}");
        assert!(PublicKey::decode(&text).is_ok());
        // Not a secret key by mistake.
        assert!(!text.to_lowercase().contains("secret"));
        // And a signature made by some other key does not verify against it.
        let stranger = Keys::new(3);
        let sig = stranger.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(verify_package(INSTALLER.to_vec(), &sig, pubkey, "0.2.0", true).unwrap_err(), VerifyError::DoesNotMatch);
    }
}
