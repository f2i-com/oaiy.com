//! How one secret becomes the bytes of one file, and back: the blob formats of the two providers, each with the name bound in.
//!
//! **`windows-dpapi-file`** (`<name>.ks`): `"OAIYKS1" || 0x01 || DPAPI blob`, the blob made by `CryptProtectData` for the current user (never
//! `LOCAL_MACHINE`, never a prompt) with optional entropy `SHA-256("oaiy-ks:1|" + name)`. The entropy is the binding: a blob copied to another name
//! does not unprotect, so a file dropped under the wrong name fails instead of yielding another secret (`Os`).
//!
//! **`keyfile`** (`<name>.kf`): `"OAIYKF1" || 0x01 || tag(32) || u32be(len) || value || check(32)` with `tag = SHA-256("oaiy-ks:1|" + name)` and
//! `check = SHA-256("oaiy-ks:1:check|" || tag || value)`. The value is in the clear in this file (that is what a keyfile is: the weakest
//! provider, protected by `0600` in a `0700` directory and by nothing else); the tag makes a moved file fail with `WrongName` and the check makes a
//! damaged one fail with `Corrupt`, in that order of precedence: a copy is intact, so it is a wrong name, not damage.

use oaiy_crypto::kdf::sha256;
#[cfg(any(unix, feature = "unsafe-keyfile", test))]
use oaiy_crypto::zeroize::ct_eq;
use zeroize::Zeroizing;

use crate::error::KeyError;
use crate::name::Name;

/// How strongly a provider protects a secret at rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strength {
    /// Bound to the operating system account (DPAPI): another user, another machine, a reset password cannot read it.
    OsAccount,
    /// The value sits in a file that only file permissions protect. The weakest, and labelled so.
    FilePermissions,
}

/// What a provider says about itself, for a status page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderInfo {
    /// The identifier: `windows-dpapi-file`, `keyfile`.
    pub id: &'static str,
    /// How strong.
    pub strength: Strength,
    /// One line for a person.
    pub description: &'static str,
}

/// The name entropy and the tag: `SHA-256("oaiy-ks:1|" + name)`.
pub(crate) fn name_binding(name: &Name) -> [u8; 32] {
    let mut text = b"oaiy-ks:1|".to_vec();
    text.extend_from_slice(name.as_str().as_bytes());
    sha256(&text)
}

/// One provider's way of making a file's bytes.
pub(crate) trait Codec: Send + Sync {
    /// What it is.
    fn info(&self) -> ProviderInfo;
    /// The file extension, without the dot.
    fn extension(&self) -> &'static str;
    /// The bytes of the file for `value` under `name`.
    fn seal(&self, name: &Name, value: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError>;
    /// The value of a file's bytes under `name`. Never `Ok` for a blob of another name.
    fn open(&self, name: &Name, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError>;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the keyfile
// ---------------------------------------------------------------------------------------------------------------------------------------

// The plaintext provider is compiled in only where it can be used: on Unix (modes), in a build with the `unsafe-keyfile` feature, and in this crate's own unit tests. A Windows build
// without the feature has no code that stores a value in the clear, whatever a setting says.
#[cfg(any(unix, feature = "unsafe-keyfile", test))]
const KEYFILE_MAGIC: &[u8; 8] = b"OAIYKF1\x01";
#[cfg(any(unix, feature = "unsafe-keyfile", test))]
const HEADER: usize = 8 + 32 + 4;
#[cfg(any(unix, feature = "unsafe-keyfile", test))]
const CHECK: usize = 32;

/// The keyfile provider.
#[cfg(any(unix, feature = "unsafe-keyfile", test))]
pub(crate) struct KeyfileCodec;

#[cfg(any(unix, feature = "unsafe-keyfile", test))]
fn check_of(tag: &[u8; 32], value: &[u8]) -> [u8; 32] {
    let mut text = b"oaiy-ks:1:check|".to_vec();
    text.extend_from_slice(tag);
    text.extend_from_slice(value);
    let digest = sha256(&text);
    zeroize::Zeroize::zeroize(&mut text);
    digest
}

#[cfg(any(unix, feature = "unsafe-keyfile", test))]
impl Codec for KeyfileCodec {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "keyfile",
            strength: Strength::FilePermissions,
            description: "a file per secret, 0600 in a 0700 directory; the weakest provider (no operating-system protection)",
        }
    }

    fn extension(&self) -> &'static str {
        "kf"
    }

    fn seal(&self, name: &Name, value: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
        let tag = name_binding(name);
        let mut out = Zeroizing::new(Vec::with_capacity(HEADER + value.len() + CHECK));
        out.extend_from_slice(KEYFILE_MAGIC);
        out.extend_from_slice(&tag);
        out.extend_from_slice(&u32::try_from(value.len()).map_err(|_| KeyError::InvalidValue("too large"))?.to_be_bytes());
        out.extend_from_slice(value);
        out.extend_from_slice(&check_of(&tag, value));
        Ok(out)
    }

    fn open(&self, name: &Name, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
        if blob.len() < HEADER + CHECK || &blob[..8] != KEYFILE_MAGIC {
            return Err(KeyError::Corrupt("not a keyfile blob"));
        }
        let mut tag = [0u8; 32];
        tag.copy_from_slice(&blob[8..40]);
        let len = u32::from_be_bytes([blob[40], blob[41], blob[42], blob[43]]) as usize;
        if blob.len() != HEADER + len + CHECK {
            return Err(KeyError::Corrupt("length"));
        }
        let value = &blob[HEADER..HEADER + len];
        if !ct_eq(&blob[HEADER + len..], &check_of(&tag, value)) {
            return Err(KeyError::Corrupt("check"));
        }
        if !ct_eq(&tag, &name_binding(name)) {
            return Err(KeyError::WrongName);
        }
        Ok(Zeroizing::new(value.to_vec()))
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// DPAPI
// ---------------------------------------------------------------------------------------------------------------------------------------

#[cfg(windows)]
const DPAPI_MAGIC: &[u8; 8] = b"OAIYKS1\x01";

/// The Windows provider: a DPAPI blob per secret.
#[cfg(windows)]
pub(crate) struct DpapiCodec;

#[cfg(windows)]
impl Codec for DpapiCodec {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "windows-dpapi-file",
            strength: Strength::OsAccount,
            description: "a DPAPI blob per secret, user scope, the name bound in as entropy",
        }
    }

    fn extension(&self) -> &'static str {
        "ks"
    }

    fn seal(&self, name: &Name, value: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
        let blob = crate::dpapi::protect(value, &name_binding(name))?;
        let mut out = Vec::with_capacity(DPAPI_MAGIC.len() + blob.len());
        out.extend_from_slice(DPAPI_MAGIC);
        out.extend_from_slice(&blob);
        Ok(Zeroizing::new(out))
    }

    fn open(&self, name: &Name, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
        if blob.len() <= DPAPI_MAGIC.len() || &blob[..DPAPI_MAGIC.len()] != DPAPI_MAGIC {
            return Err(KeyError::Corrupt("not a DPAPI blob of this keystore"));
        }
        crate::dpapi::unprotect(&blob[DPAPI_MAGIC.len()..], &name_binding(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// Known answers for the two things every blob is bound by, computed with node's crypto (`scratchpad/vault-impl/kat.mjs`), not by this code: the name
    /// binding `SHA-256("oaiy-ks:1|" + name)` (the DPAPI entropy and the keyfile tag) and the whole keyfile, byte for byte. A file written today must be read
    /// by every later version, so the domain string, the field order and the length's byte order (big-endian) are pinned: a change to any of them would make
    /// every stored secret unreadable, and this test is what says so (KM11, KM12).
    const KATS: [(&str, &str, &str, &str); 2] = [
        (
            "archive.writer",
            "6f616979206b657973746f7265206b6e6f776e20616e73776572",
            "195a64d1c2e3a8d0c197bc2ddf1df260424530839be9fda34c33e23fdde2795c",
            "4f4149594b463101195a64d1c2e3a8d0c197bc2ddf1df260424530839be9fda34c33e23fdde2795c0000001a6f616979206b657973746f7265206b6e6f776e20616e73776572d57f32c601b74541a0f629ddf6506393553028cb8598860d71ade7a85424bc2e",
        ),
        (
            "vault.fk.f1",
            "00ff01",
            "71a3542662649453a76dc19689a899f8bbff6f57b7dd580805c66c809a7aa777",
            "4f4149594b46310171a3542662649453a76dc19689a899f8bbff6f57b7dd580805c66c809a7aa7770000000300ff01748ed4d0507837dfd14a1edb9e7296d0a9638617d76efe6359acb88933fb2a46",
        ),
    ];

    #[test]
    fn the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte() {
        for (name, value, tag, file) in KATS {
            let name = Name::new(name).unwrap();
            let value = unhex(value);
            assert_eq!(hex(&name_binding(&name)), tag, "the binding of {name}");
            let sealed = KeyfileCodec.seal(&name, &value).unwrap();
            assert_eq!(hex(&sealed), file, "the keyfile of {name}");
            // and the pinned bytes are read back: the reader agrees with the writer, and with the file
            assert_eq!(&**KeyfileCodec.open(&name, &unhex(file)).unwrap(), value.as_slice());
            // the length field is big-endian: the same bytes with it little-endian are not a blob
            let mut little = unhex(file);
            little[40..44].reverse();
            assert_ne!(little, unhex(file));
            assert!(KeyfileCodec.open(&name, &little).is_err(), "a little-endian length was accepted");
        }
    }
}
