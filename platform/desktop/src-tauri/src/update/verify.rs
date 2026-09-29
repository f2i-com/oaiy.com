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
//!
//! What a signature proves is that the key signed THESE BYTES. It does not by itself say which
//! release they belong to: the version in the feed is text, and a feed that pairs a newer version
//! number with an older, genuinely signed installer would install as a downgrade dressed as an
//! update. What ties the bytes to a release is the name of the file the signature was made for,
//! which the Tauri CLI writes into the signature's trusted comment (`file:OAIY_0.2.0_x64-setup.exe`) and
//! the key's signature covers. So the name has to be an installer of this platform's kind, for the
//! announced version ([`Target::signed_name_fits`]). (The CLI in use writes no `version:` field;
//! if one is there it must agree too, and `requireSignedVersion` can insist on one.)

use base64::Engine as _;
use minisign_verify::{PublicKey, Signature};

use super::target::{NameProblem, Target};

/// An installer whose signature verified against the key this copy of OAIY carries, for the version the feed
/// announced and the platform this copy runs on.
#[derive(Debug, PartialEq, Eq)]
pub struct VerifiedPackage {
    version: String,
    target: Target,
    bytes: Vec<u8>,
}

impl VerifiedPackage {
    /// The version the feed announced, and the signature was checked for.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The kind of installer it was checked to be.
    pub fn target(&self) -> Target {
        self.target
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

    /// The last look before the installer is given the bytes (the desktop's hand-off calls it, and nothing goes to the installer
    /// unless it is Ok): they are for `update_version`, the version of the update handle the installer belongs to, and for the
    /// kind of installer `current` (this platform's: None when no release is built for it), and they start the way that kind of
    /// file does. A package that was made by [`verify_package`] passes; this is for the day one does not (a mix-up of
    /// handles, a package built for another platform, a change made to the bytes in memory).
    pub fn check_for_hand_off(&self, update_version: &str, current: Option<Target>) -> Result<(), String> {
        if self.version != update_version {
            return Err(format!("the downloaded update is for version {}, not {update_version}", self.version));
        }
        let Some(current) = current else { return Err("there is no installer of OAIY for this kind of computer".to_string()) };
        if self.target != current {
            return Err(format!("the downloaded update is {}, not {}", self.target.what(), current.what()));
        }
        if !current.looks_like(&self.bytes) {
            return Err(format!("the downloaded update does not start like {}", current.what()));
        }
        Ok(())
    }
}

/// What a signature has to agree with: the version the feed announced, the kind of installer this platform takes,
/// and whether the signature must also name a version of its own.
#[derive(Debug, Clone, Copy)]
pub struct Expected<'a> {
    pub version: &'a str,
    pub target: Target,
    /// `plugins.updater.requireSignedVersion`: a signature with no `version:` field is refused.
    pub require_signed_version: bool,
}

impl<'a> Expected<'a> {
    pub fn new(version: &'a str, target: Target) -> Expected<'a> {
        Expected { version, target, require_signed_version: false }
    }

    pub fn requiring_signed_version(self) -> Expected<'a> {
        Expected { require_signed_version: true, ..self }
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
    /// The signature does not say which file it was made for.
    NoSignedFile,
    /// The signature was made for a file that is not an installer of this platform's kind (another platform's, an MSI, a path).
    SignedFileNotThisKind { file: String, expected: &'static str },
    /// The signature was made for a file of another version than the feed announced (a feed pairing a newer version with an older release).
    SignedFileNotThisVersion { file: String, announced: String },
    /// The bytes do not start the way this platform's kind of installer does (the signature and its name were right).
    NotThisKindOfFile { expected: &'static str },
    /// The signature says it was made for another version than the feed announced.
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
            VerifyError::NoSignedFile => write!(f, "The update's signature does not say which file it was made for, so the update was refused."),
            VerifyError::SignedFileNotThisKind { file, expected } => write!(f, "The update's signature was made for a file called {file}, which is not {expected}, so the update was refused."),
            VerifyError::SignedFileNotThisVersion { file, announced } => write!(f, "The update's signature was made for a file called {file}, which is not version {announced}, so the update was refused."),
            VerifyError::NotThisKindOfFile { expected } => write!(f, "The downloaded update is not {expected}, so it was thrown away."),
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

/// One field of a trusted comment (tab-separated `key:value` pairs, `key` given with its colon): None when it is not there, and
/// when it is there twice, because a comment that says two things says nothing.
fn field<'a>(trusted_comment: &'a str, key: &str) -> Option<&'a str> {
    let mut found = trusted_comment.split('\t').filter_map(|part| part.strip_prefix(key));
    let first = found.next()?;
    if found.next().is_some() {
        return None;
    }
    Some(first)
}

/// The file a signature was made for: the `file:` field of its trusted comment.
fn signed_file(trusted_comment: &str) -> Option<&str> {
    field(trusted_comment, "file:")
}

/// The version a signature says it was made for: the `version:` field, if there is one (the Tauri CLI in use writes none).
fn signed_version(trusted_comment: &str) -> Option<&str> {
    field(trusted_comment, "version:")
}

/// Check `bytes` against `signature` with `pubkey`, and what the signature says it was made for against `expected`.
///
/// After the signature itself (which also covers the trusted comment, so what is read from it next cannot have been edited):
///
/// - the file name in it must be an installer of `expected.target`'s kind (its ending) for `expected.version` (one whole
///   underscore-separated part of it): [`Target::signed_name_fits`]. This is what a downgrade meets, since the older
///   installer's signature names the older version;
/// - the bytes must start the way that kind of file does ([`Target::looks_like`]: `MZ` for the Windows setup, `\x7fELF` for the
///   AppImage), so that a file of the wrong kind cannot come through under the right name;
/// - a `version:` field, when there is one, must be the announced version, whatever `require_signed_version` says;
///   with that on, a signature without one is refused.
pub fn verify_package(bytes: Vec<u8>, signature: &str, pubkey: &str, expected: &Expected) -> Result<VerifiedPackage, VerifyError> {
    let key = decode_text(pubkey).and_then(|text| PublicKey::decode(&text).ok()).ok_or(VerifyError::BadPublicKey)?;
    let signature = decode_text(signature).and_then(|text| Signature::decode(&text).ok()).ok_or(VerifyError::BadSignature)?;
    // Legacy (non-prehashed) signatures are accepted as the updater plugin accepts them.
    key.verify(&bytes, &signature, true).map_err(|_| VerifyError::DoesNotMatch)?;
    // Only now is the trusted comment usable: verify() also checked the global signature that covers it.
    let comment = signature.trusted_comment();
    let file = signed_file(comment).ok_or(VerifyError::NoSignedFile)?;
    expected.target.signed_name_fits(file, expected.version).map_err(|problem| match problem {
        NameProblem::NotThisKind => VerifyError::SignedFileNotThisKind { file: file.to_string(), expected: expected.target.what() },
        NameProblem::NotThisVersion => VerifyError::SignedFileNotThisVersion { file: file.to_string(), announced: expected.version.to_string() },
    })?;
    // The signature can be right for a name and the bytes still not the kind of file that name says (the signer signed what it was
    // given): the updater on Windows runs whatever it is handed, and the one on Linux writes it over the AppImage.
    if !expected.target.looks_like(&bytes) {
        return Err(VerifyError::NotThisKindOfFile { expected: expected.target.what() });
    }
    match signed_version(comment) {
        Some(signed) => {
            let same = match (super::version::parse(signed), super::version::parse(expected.version)) {
                (Ok(signed), Ok(announced)) => signed == announced,
                _ => signed == expected.version,
            };
            if !same {
                return Err(VerifyError::SignedForOtherVersion { signed: signed.to_string(), announced: expected.version.to_string() });
            }
        }
        None if expected.require_signed_version => return Err(VerifyError::NoSignedVersion),
        None => {}
    }
    Ok(VerifiedPackage { version: expected.version.to_string(), target: expected.target, bytes })
}

/// Test-only signing in the minisign format, so the tests need no fixture and no key file.
#[cfg(test)]
pub(crate) mod testing {
    use super::super::target::Target;
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

    /// The trusted comment the Tauri CLI writes for the Windows setup of a release: a time and the file, as the bundler
    /// named it, and no version (the CLI in use writes none).
    pub fn comment(version: &str) -> String {
        comment_for(Target::WindowsSetup, version)
    }

    /// The same for either installer.
    pub fn comment_for(target: Target, version: &str) -> String {
        let file = match target {
            Target::WindowsSetup => format!("OAIY_{version}_x64-setup.exe"),
            Target::LinuxAppImage => format!("OAIY_{version}_amd64.AppImage"),
        };
        format!("timestamp:1790000000\tfile:{file}")
    }

    /// What a later CLI might write: the file, and the version besides.
    pub fn comment_with_version(version: &str) -> String {
        format!("{}\tversion:{version}", comment(version))
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{comment, comment_for, comment_with_version, Keys};
    use super::*;

    /// Stand-ins that start the way a Windows executable and a Linux executable do (an update checks that much of what it is given).
    const INSTALLER: &[u8] = b"MZ pretend this is a 100 MB installer";
    const APPIMAGE: &[u8] = b"\x7fELF pretend this is a 100 MB AppImage";

    fn setup(version: &str) -> Expected<'_> {
        Expected::new(version, Target::WindowsSetup)
    }

    fn accepted(keys: &Keys, data: &[u8], sig: &str, expected: &Expected) -> Result<VerifiedPackage, VerifyError> {
        verify_package(data.to_vec(), sig, &keys.pubkey, expected)
    }

    #[test]
    fn an_installer_signed_with_the_key_this_build_carries_verifies() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        let package = accepted(&keys, INSTALLER, &sig, &setup("0.2.0")).unwrap();
        assert_eq!(package.bytes(), INSTALLER);
        assert_eq!(package.version(), "0.2.0");
        assert_eq!(package.target(), Target::WindowsSetup);
        assert_eq!(package.len(), INSTALLER.len());
    }

    #[test]
    fn an_installer_that_was_changed_by_a_byte_is_refused() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        let mut tampered = INSTALLER.to_vec();
        tampered[10] ^= 1;
        assert_eq!(accepted(&keys, &tampered, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
        // Cut short, or with something added.
        assert_eq!(accepted(&keys, &INSTALLER[..INSTALLER.len() - 1], &sig, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
        let mut longer = INSTALLER.to_vec();
        longer.push(0);
        assert_eq!(accepted(&keys, &longer, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
        assert_eq!(accepted(&keys, b"", &sig, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
    }

    #[test]
    fn an_installer_signed_with_another_key_is_refused() {
        let ours = Keys::new(7);
        let theirs = Keys::new(9);
        let sig = theirs.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(accepted(&ours, INSTALLER, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
        // A signature that names our key id but was made by another key: the check itself refuses it, not just the name.
        let ours = Keys::with_id(7, 1);
        let forger = Keys::with_id(9, 1);
        let forged = forger.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(accepted(&ours, INSTALLER, &forged, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
        assert!(accepted(&ours, INSTALLER, &ours.sign(INSTALLER, &comment("0.2.0")), &setup("0.2.0")).is_ok());
    }

    #[test]
    fn a_trusted_comment_that_was_changed_is_refused() {
        // The comment is covered by the global signature: pointing the signature at another version by editing the file name fails.
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.1.5"));
        let text = String::from_utf8(base64::engine::general_purpose::STANDARD.decode(&sig).unwrap()).unwrap();
        let edited = text.replace("OAIY_0.1.5_x64-setup.exe", "OAIY_0.2.0_x64-setup.exe");
        assert_ne!(edited, text);
        let forged = base64::engine::general_purpose::STANDARD.encode(edited);
        assert_eq!(accepted(&keys, INSTALLER, &forged, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
    }

    #[test]
    fn the_downgrade_a_reviewer_ran_is_refused_an_old_signed_installer_announced_as_a_newer_version() {
        // The feed says 9.9.9 and points at a v9.9.9 address, but the file behind it is the genuinely signed 0.1.0 installer.
        let keys = Keys::new(7);
        let old = keys.sign(INSTALLER, &comment("0.1.0"));
        assert_eq!(
            accepted(&keys, INSTALLER, &old, &setup("9.9.9")).unwrap_err(),
            VerifyError::SignedFileNotThisVersion { file: "OAIY_0.1.0_x64-setup.exe".into(), announced: "9.9.9".into() }
        );
        // Whether or not versions are required of signatures.
        assert!(matches!(accepted(&keys, INSTALLER, &old, &setup("9.9.9").requiring_signed_version()), Err(VerifyError::SignedFileNotThisVersion { .. })));
        // A version that only looks like it is not it.
        for announced in ["0.1", "0.1.00", "1.0", "10.1.0", "0.1.0-rc.1", "0.1.0+build"] {
            assert!(matches!(accepted(&keys, INSTALLER, &old, &setup(announced)), Err(VerifyError::SignedFileNotThisVersion { .. })), "announced {announced}");
        }
        // The matching one is accepted.
        assert!(accepted(&keys, INSTALLER, &old, &setup("0.1.0")).is_ok());
        // ...and the message says what happened, in words.
        let message = accepted(&keys, INSTALLER, &old, &setup("9.9.9")).unwrap_err().to_string();
        assert!(message.contains("OAIY_0.1.0_x64-setup.exe") && message.contains("9.9.9") && message.contains("refused"), "{message}");
    }

    #[test]
    fn each_platform_takes_only_its_own_kind_of_file() {
        let keys = Keys::new(7);
        let windows = keys.sign(INSTALLER, &comment_for(Target::WindowsSetup, "0.2.0"));
        let linux = keys.sign(APPIMAGE, &comment_for(Target::LinuxAppImage, "0.2.0"));
        // Each verifies for its own platform...
        assert!(accepted(&keys, INSTALLER, &windows, &Expected::new("0.2.0", Target::WindowsSetup)).is_ok());
        assert!(accepted(&keys, APPIMAGE, &linux, &Expected::new("0.2.0", Target::LinuxAppImage)).is_ok());
        // ...and not for the other: the Windows setup where the AppImage belongs, and the AppImage where the setup does.
        assert!(matches!(accepted(&keys, INSTALLER, &windows, &Expected::new("0.2.0", Target::LinuxAppImage)), Err(VerifyError::SignedFileNotThisKind { .. })));
        assert!(matches!(accepted(&keys, APPIMAGE, &linux, &Expected::new("0.2.0", Target::WindowsSetup)), Err(VerifyError::SignedFileNotThisKind { .. })));
        // Other files the pipeline makes, and things that are not file names.
        for file in ["OAIY_0.2.0_x64_en-US.msi", "OAIY_0.2.0_amd64.deb", "OAIY-0.2.0-1.x86_64.rpm", "OAIY_0.2.0_x64-setup.exe.zip", "OAIY_0.2.0", "installer.bin", "../OAIY_0.2.0_x64-setup.exe", "C:\\OAIY_0.2.0_x64-setup.exe"] {
            let sig = keys.sign(INSTALLER, &format!("timestamp:1790000000\tfile:{file}"));
            assert!(matches!(accepted(&keys, INSTALLER, &sig, &setup("0.2.0")), Err(VerifyError::SignedFileNotThisKind { .. })), "{file}");
        }
        let message = accepted(&keys, APPIMAGE, &linux, &setup("0.2.0")).unwrap_err().to_string();
        assert!(message.contains("OAIY_0.2.0_amd64.AppImage") && message.contains("Windows installer"), "{message}");
    }

    #[test]
    fn the_appimage_is_held_to_its_ending_and_its_version_and_nothing_more() {
        // The AppImage's name has changed between Tauri versions (amd64, x86_64, no architecture), so only these two are looked at.
        let keys = Keys::new(7);
        for file in ["OAIY_0.2.0_amd64.AppImage", "OAIY_0.2.0_x86_64.AppImage", "oaiy_0.2.0.AppImage", "OAIY_0.2.0_aarch64.AppImage"] {
            let sig = keys.sign(APPIMAGE, &format!("timestamp:1790000000\tfile:{file}"));
            assert!(accepted(&keys, APPIMAGE, &sig, &Expected::new("0.2.0", Target::LinuxAppImage)).is_ok(), "{file}");
            assert!(matches!(accepted(&keys, APPIMAGE, &sig, &Expected::new("9.9.9", Target::LinuxAppImage)), Err(VerifyError::SignedFileNotThisVersion { .. })), "{file}");
        }
    }

    #[test]
    fn bytes_that_are_not_this_platforms_kind_of_file_are_refused_though_the_signature_and_its_name_are_right() {
        // Signed by the right key, under the right name, for the right version: but the file is not that kind of installer
        // (the signer signed what it was given, or a mix-up put the wrong file under the right name).
        let keys = Keys::new(7);
        let wrong: [(Target, &[u8]); 7] = [
            (Target::WindowsSetup, APPIMAGE),
            (Target::WindowsSetup, b"<html>404</html>"),
            (Target::WindowsSetup, b"M"),
            (Target::WindowsSetup, b""),
            (Target::LinuxAppImage, INSTALLER),
            (Target::LinuxAppImage, b"#!/bin/sh\nrm -rf ~\n"),
            (Target::LinuxAppImage, b"\x7fEL"),
        ];
        for (target, bytes) in wrong {
            let sig = keys.sign(bytes, &comment_for(target, "0.2.0"));
            assert_eq!(accepted(&keys, bytes, &sig, &Expected::new("0.2.0", target)).unwrap_err(), VerifyError::NotThisKindOfFile { expected: target.what() }, "{target:?} {bytes:?}");
        }
        let message = VerifyError::NotThisKindOfFile { expected: Target::LinuxAppImage.what() }.to_string();
        assert!(message.contains("Linux AppImage") && message.contains("thrown away"), "{message}");
    }

    #[test]
    fn the_installer_is_handed_only_bytes_for_its_own_version_and_this_platforms_kind_of_file() {
        let keys = Keys::new(7);
        let windows = accepted(&keys, INSTALLER, &keys.sign(INSTALLER, &comment("0.2.0")), &setup("0.2.0")).unwrap();
        let linux = accepted(&keys, APPIMAGE, &keys.sign(APPIMAGE, &comment_for(Target::LinuxAppImage, "0.2.0")), &Expected::new("0.2.0", Target::LinuxAppImage)).unwrap();
        assert_eq!(windows.check_for_hand_off("0.2.0", Some(Target::WindowsSetup)), Ok(()));
        assert_eq!(linux.check_for_hand_off("0.2.0", Some(Target::LinuxAppImage)), Ok(()));
        // The update handle is for another version than the bytes were verified for.
        let other = windows.check_for_hand_off("0.3.0", Some(Target::WindowsSetup)).unwrap_err();
        assert!(other.contains("0.2.0") && other.contains("0.3.0"), "{other}");
        // No release is built for this kind of computer.
        assert!(windows.check_for_hand_off("0.2.0", None).unwrap_err().contains("no installer"));
        // The other platform's installer: the Linux updater writes whatever it is given over the AppImage.
        let mixed = windows.check_for_hand_off("0.2.0", Some(Target::LinuxAppImage)).unwrap_err();
        assert!(mixed.contains("Windows installer") && mixed.contains("Linux AppImage"), "{mixed}");
        assert!(linux.check_for_hand_off("0.2.0", Some(Target::WindowsSetup)).is_err());
        // Bytes that no longer start the way that kind of file does (built in place: the fields are private to this module).
        for (target, bytes) in [(Target::WindowsSetup, &b"<html>"[..]), (Target::WindowsSetup, APPIMAGE), (Target::LinuxAppImage, INSTALLER), (Target::LinuxAppImage, b"")] {
            let altered = VerifiedPackage { version: "0.2.0".into(), target, bytes: bytes.to_vec() };
            assert!(altered.check_for_hand_off("0.2.0", Some(target)).unwrap_err().contains("does not start like"), "{target:?} {bytes:?}");
        }
    }

    #[test]
    fn a_signature_that_says_no_file_or_two_is_refused() {
        let keys = Keys::new(7);
        for comment in ["timestamp:1790000000", "timestamp:1790000000\tfilename:OAIY_0.2.0_x64-setup.exe", "timestamp:1790000000\tfile:OAIY_0.2.0_x64-setup.exe\tfile:OAIY_0.2.0_x64-setup.exe"] {
            let sig = keys.sign(INSTALLER, comment);
            assert_eq!(accepted(&keys, INSTALLER, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::NoSignedFile, "{comment:?}");
        }
    }

    #[test]
    fn a_version_the_signature_names_besides_must_agree_and_may_be_required() {
        // (What a later Tauri CLI might write.) Present, it must be the announced one, whatever requireSignedVersion says.
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &format!("{}\tversion:0.1.5", comment("0.2.0")));
        assert_eq!(accepted(&keys, INSTALLER, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::SignedForOtherVersion { signed: "0.1.5".into(), announced: "0.2.0".into() });
        assert!(matches!(accepted(&keys, INSTALLER, &sig, &setup("0.2.0").requiring_signed_version()), Err(VerifyError::SignedForOtherVersion { .. })));
        // The same version spelled with a v is the same version.
        assert!(accepted(&keys, INSTALLER, &keys.sign(INSTALLER, &format!("{}\tversion:v0.2.0", comment("0.2.0"))), &setup("0.2.0").requiring_signed_version()).is_ok());
        assert!(accepted(&keys, INSTALLER, &keys.sign(INSTALLER, &comment_with_version("0.2.0")), &setup("0.2.0").requiring_signed_version()).is_ok());
        // Absent, it is refused only when versions are required.
        let plain = keys.sign(INSTALLER, &comment("0.2.0"));
        assert_eq!(accepted(&keys, INSTALLER, &plain, &setup("0.2.0").requiring_signed_version()).unwrap_err(), VerifyError::NoSignedVersion);
        assert!(accepted(&keys, INSTALLER, &plain, &setup("0.2.0")).is_ok());
    }

    #[test]
    fn a_key_or_a_signature_that_cannot_be_read_is_refused_in_words_and_never_accepted() {
        let keys = Keys::new(7);
        let sig = keys.sign(INSTALLER, &comment("0.2.0"));
        for bad in ["", "not base64!", "aGVsbG8="] {
            assert_eq!(verify_package(INSTALLER.to_vec(), &sig, bad, &setup("0.2.0")).unwrap_err(), VerifyError::BadPublicKey, "key {bad:?}");
            assert_eq!(verify_package(INSTALLER.to_vec(), bad, &keys.pubkey, &setup("0.2.0")).unwrap_err(), VerifyError::BadSignature, "signature {bad:?}");
        }
        // A public key file where the signature belongs, and the other way round.
        assert_eq!(verify_package(INSTALLER.to_vec(), &keys.pubkey, &keys.pubkey, &setup("0.2.0")).unwrap_err(), VerifyError::BadSignature);
        assert_eq!(verify_package(INSTALLER.to_vec(), &sig, &sig, &setup("0.2.0")).unwrap_err(), VerifyError::BadPublicKey);
        for error in [VerifyError::BadPublicKey, VerifyError::BadSignature, VerifyError::DoesNotMatch, VerifyError::NoSignedVersion, VerifyError::NoSignedFile] {
            assert!(error.to_string().contains("refused") || error.to_string().contains("thrown away"), "{error}");
        }
    }

    // A real installer's signature: two stand-in files (an `MZ` file and an ELF one, 3000 bytes each, not installers) signed by
    // `tauri signer sign` (the Tauri CLI this repository uses, 2.11) under the names the bundler gives an installer, with a
    // throwaway key. See testdata/README.txt.
    const WINDOWS_PAYLOAD: &[u8] = include_bytes!("testdata/windows-setup.bin");
    const WINDOWS_SIGNATURE: &str = include_str!("testdata/windows-setup.bin.sig");
    const LINUX_PAYLOAD: &[u8] = include_bytes!("testdata/linux-appimage.bin");
    const LINUX_SIGNATURE: &str = include_str!("testdata/linux-appimage.bin.sig");
    const REAL_PUBKEY: &str = include_str!("testdata/throwaway.key.pub");

    // Payloads that are not signed under their own names are not made here: see the real ones below.
    fn real(target: Target, version: &str) -> Result<VerifiedPackage, VerifyError> {
        let (payload, signature) = match target {
            Target::WindowsSetup => (WINDOWS_PAYLOAD, WINDOWS_SIGNATURE),
            Target::LinuxAppImage => (LINUX_PAYLOAD, LINUX_SIGNATURE),
        };
        verify_package(payload.to_vec(), signature.trim(), REAL_PUBKEY.trim(), &Expected::new(version, target))
    }

    #[test]
    fn a_signature_made_by_the_tauri_cli_verifies_for_its_own_version_and_kind_and_a_changed_byte_or_another_key_does_not() {
        let (signature, pubkey) = (WINDOWS_SIGNATURE.trim(), REAL_PUBKEY.trim());
        let package = real(Target::WindowsSetup, "0.1.0").expect("what the pipeline signs verifies");
        assert_eq!(package.bytes(), WINDOWS_PAYLOAD);
        let appimage = real(Target::LinuxAppImage, "0.1.0").expect("the AppImage too");
        assert_eq!((appimage.bytes(), appimage.target(), appimage.version()), (LINUX_PAYLOAD, Target::LinuxAppImage, "0.1.0"));
        assert_eq!((package.target(), package.version()), (Target::WindowsSetup, "0.1.0"));
        // The signature is the CLI's: prehashed, made by a key with its own name, for the file the bundler names.
        let parsed = Signature::decode(&decode_text(signature).unwrap()).unwrap();
        assert!(parsed.trusted_comment().starts_with("timestamp:") && parsed.trusted_comment().ends_with("\tfile:OAIY_0.1.0_x64-setup.exe"), "{}", parsed.trusted_comment());
        let parsed = Signature::decode(&decode_text(LINUX_SIGNATURE.trim()).unwrap()).unwrap();
        assert!(parsed.trusted_comment().ends_with("\tfile:OAIY_0.1.0_amd64.AppImage"), "{}", parsed.trusted_comment());
        for at in [0, WINDOWS_PAYLOAD.len() / 2, WINDOWS_PAYLOAD.len() - 1] {
            let mut changed = WINDOWS_PAYLOAD.to_vec();
            changed[at] ^= 1;
            assert_eq!(verify_package(changed, signature, pubkey, &setup("0.1.0")).unwrap_err(), VerifyError::DoesNotMatch, "byte {at}");
        }
        assert_eq!(verify_package(WINDOWS_PAYLOAD[..WINDOWS_PAYLOAD.len() - 1].to_vec(), signature, pubkey, &setup("0.1.0")).unwrap_err(), VerifyError::DoesNotMatch);
        // Not with the key OAIY's updates are really signed with: this signature is the throwaway key's.
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        let production = conf["plugins"]["updater"]["pubkey"].as_str().unwrap();
        assert_ne!(production, pubkey);
        assert_eq!(verify_package(WINDOWS_PAYLOAD.to_vec(), signature, production, &setup("0.1.0")).unwrap_err(), VerifyError::DoesNotMatch);
    }

    #[test]
    fn the_downgrade_with_a_real_signature_is_refused_and_so_is_the_wrong_platforms_file() {
        // A real, genuinely signed 0.1.0 installer announced as 9.9.9 (the reviewer's downgrade), and as a version that only resembles it.
        for announced in ["9.9.9", "0.2.0", "0.1.1", "1.0.0", "0.1.0-rc.1"] {
            assert!(matches!(real(Target::WindowsSetup, announced), Err(VerifyError::SignedFileNotThisVersion { .. })), "{announced}");
            assert!(matches!(real(Target::LinuxAppImage, announced), Err(VerifyError::SignedFileNotThisVersion { .. })), "{announced}");
        }
        // The Windows setup and its real signature, where the AppImage belongs.
        let mixed = verify_package(WINDOWS_PAYLOAD.to_vec(), WINDOWS_SIGNATURE.trim(), REAL_PUBKEY.trim(), &Expected::new("0.1.0", Target::LinuxAppImage));
        assert!(matches!(mixed, Err(VerifyError::SignedFileNotThisKind { .. })), "{mixed:?}");
        let mixed = verify_package(LINUX_PAYLOAD.to_vec(), LINUX_SIGNATURE.trim(), REAL_PUBKEY.trim(), &Expected::new("0.1.0", Target::WindowsSetup));
        assert!(matches!(mixed, Err(VerifyError::SignedFileNotThisKind { .. })), "{mixed:?}");
    }

    #[test]
    fn the_cli_in_use_writes_no_version_into_a_signature_so_requiring_one_would_refuse_every_release() {
        // This is why plugins.updater.requireSignedVersion is off, and why the release is tied to its signature by the name of the
        // file it was made for instead. When the Tauri CLI records `version:` in the trusted comment, regenerate testdata/, flip
        // these two assertions, and turn requireSignedVersion on.
        let (signature, pubkey) = (WINDOWS_SIGNATURE.trim(), REAL_PUBKEY.trim());
        assert_eq!(verify_package(WINDOWS_PAYLOAD.to_vec(), signature, pubkey, &setup("0.1.0").requiring_signed_version()).unwrap_err(), VerifyError::NoSignedVersion);
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        assert!(!conf["plugins"]["updater"]["requireSignedVersion"].as_bool().unwrap_or(false), "turned on, it would make every update fail: the signatures carry no version");
    }

    /// The pipeline's real output, verified the way an installed OAIY verifies a download: an installer made by `tauri build`
    /// with the updater artifacts on (or signed with `tauri signer sign`), the `.sig` beside it, and the public key of the key
    /// that signed it. Run on purpose, with the paths in the environment (it needs an installer, so it is not part of a plain
    /// `cargo test`). A `.AppImage` is taken as the Linux installer, anything else as the Windows one:
    ///
    /// ```text
    /// OAIY_TEST_INSTALLER=<...-setup.exe> OAIY_TEST_PUBKEY_FILE=<key>.pub OAIY_TEST_VERSION=0.1.0 \
    ///   cargo test --locked --lib real_installer -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs an installer built by the pipeline and its key: see the doc comment"]
    fn a_real_installer_from_the_pipeline_verifies_with_the_key_that_signed_it_and_with_no_other() {
        let installer = std::env::var("OAIY_TEST_INSTALLER").expect("OAIY_TEST_INSTALLER is the path of the setup.exe");
        let pubkey_file = std::env::var("OAIY_TEST_PUBKEY_FILE").expect("OAIY_TEST_PUBKEY_FILE is the path of the .pub file");
        let version = std::env::var("OAIY_TEST_VERSION").unwrap_or_else(|_| "0.1.0".to_string());
        let target = if installer.ends_with(".AppImage") { Target::LinuxAppImage } else { Target::WindowsSetup };
        let pubkey = std::fs::read_to_string(&pubkey_file).unwrap().trim().to_string();
        let signature = std::fs::read_to_string(format!("{installer}.sig")).expect("the .sig beside the installer");
        let bytes = std::fs::read(&installer).unwrap();

        // What the Tauri CLI wrote into the signature besides the signature itself (a comment: nothing secret).
        let text = decode_text(&signature).expect("the .sig is the base64 of a minisign signature file");
        let parsed = Signature::decode(&text).expect("a minisign signature");
        println!("untrusted comment: {}", parsed.untrusted_comment());
        println!("trusted comment: {}", parsed.trusted_comment());
        println!("file in the trusted comment: {:?}", signed_file(parsed.trusted_comment()));
        println!("version in the trusted comment: {:?}", signed_version(parsed.trusted_comment()));
        println!("installer: {} bytes", bytes.len());

        // It verifies with the key that signed it, for the version and the kind (versions are not required: this CLI does not write one).
        let expected = Expected::new(&version, target);
        let package = verify_package(bytes.clone(), &signature, &pubkey, &expected).expect("the installer verifies with its key");
        assert_eq!(package.len(), bytes.len());
        // Not as another version...
        assert!(matches!(verify_package(bytes.clone(), &signature, &pubkey, &Expected::new("9.9.9", target)), Err(VerifyError::SignedFileNotThisVersion { .. })));
        // ...not with one byte changed, in the middle or at the end...
        for at in [bytes.len() / 2, bytes.len() - 1] {
            let mut changed = bytes.clone();
            changed[at] ^= 1;
            assert_eq!(verify_package(changed, &signature, &pubkey, &expected).unwrap_err(), VerifyError::DoesNotMatch, "byte {at}");
        }
        // ...and not with the production key, which did not sign it.
        let conf: serde_json::Value = serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        let production = conf["plugins"]["updater"]["pubkey"].as_str().unwrap();
        assert_ne!(production, pubkey, "this must be run with a throwaway key, never the production one");
        assert_eq!(verify_package(bytes, &signature, production, &expected).unwrap_err(), VerifyError::DoesNotMatch);
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
        assert_eq!(verify_package(INSTALLER.to_vec(), &sig, pubkey, &setup("0.2.0")).unwrap_err(), VerifyError::DoesNotMatch);
    }
}
