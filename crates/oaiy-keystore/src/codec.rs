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

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::Path;

use oaiy_crypto::kdf::sha256;
use oaiy_crypto::zeroize::ct_eq;
use zeroize::Zeroizing;

use crate::error::KeyError;
use crate::name::Name;
use crate::perm::{self, Kind};

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
    /// Makes the directory, if it is not there, the way the provider wants it.
    fn prepare_dir(&self, dir: &Path) -> Result<(), KeyError> {
        fs::create_dir_all(dir).map_err(|e| KeyError::io("create the keys directory", e))
    }
    /// Checks the directory before every operation.
    fn check_dir(&self, _dir: &Path, _owner: Option<u32>) -> Result<(), KeyError> {
        Ok(())
    }
    /// The user this process runs as, learned by making a file of its own in the directory (`None` where owners are not compared).
    fn probe_owner(&self, _dir: &Path) -> Result<Option<u32>, KeyError> {
        Ok(None)
    }
    /// Checks a file before it is read.
    fn check_file(&self, _path: &Path, _owner: Option<u32>) -> Result<(), KeyError> {
        Ok(())
    }
    /// Creates a new file that must not exist.
    fn create_file(&self, path: &Path) -> io::Result<File> {
        OpenOptions::new().write(true).create_new(true).open(path)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the keyfile
// ---------------------------------------------------------------------------------------------------------------------------------------

const KEYFILE_MAGIC: &[u8; 8] = b"OAIYKF1\x01";
const HEADER: usize = 8 + 32 + 4;
const CHECK: usize = 32;

/// The keyfile provider.
pub(crate) struct KeyfileCodec;

fn check_of(tag: &[u8; 32], value: &[u8]) -> [u8; 32] {
    let mut text = b"oaiy-ks:1:check|".to_vec();
    text.extend_from_slice(tag);
    text.extend_from_slice(value);
    let digest = sha256(&text);
    zeroize::Zeroize::zeroize(&mut text);
    digest
}

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

    #[cfg(unix)]
    fn prepare_dir(&self, dir: &Path) -> Result<(), KeyError> {
        use std::os::unix::fs::DirBuilderExt;
        if !dir.exists() {
            // a new directory is made 0700; an existing one is checked, never repaired
            fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| KeyError::io("create the keys directory", e))?;
        }
        Ok(())
    }

    fn check_dir(&self, dir: &Path, owner: Option<u32>) -> Result<(), KeyError> {
        let meta = perm::read_meta(dir).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        perm::check(Kind::Dir, &meta, owner).map_err(|why| KeyError::Permissions(format!("{}: {why}", dir.display())))
    }

    #[cfg(unix)]
    fn probe_owner(&self, dir: &Path) -> Result<Option<u32>, KeyError> {
        use std::os::unix::fs::MetadataExt;
        let random: oaiy_crypto::zeroize::Secret<8> =
            oaiy_crypto::zeroize::Secret::random().map_err(|_| KeyError::io("random", io::Error::other("the random generator failed")))?;
        let probe = dir.join(format!(".probe.{}.tmp", oaiy_crypto::kdf::hex_lower(random.expose())));
        let file = self.create_file(&probe).map_err(|e| KeyError::io("create a probe file", e))?;
        let uid = file.metadata().map(|m| m.uid());
        drop(file);
        let _ = fs::remove_file(&probe);
        uid.map(Some).map_err(|e| KeyError::io("inspect a probe file", e))
    }

    fn check_file(&self, path: &Path, owner: Option<u32>) -> Result<(), KeyError> {
        let meta = perm::read_meta(path).map_err(|e| KeyError::io("inspect a key file", e))?;
        perm::check(Kind::File, &meta, owner).map_err(|why| KeyError::Permissions(format!("{}: {why}", path.display())))
    }

    #[cfg(unix)]
    fn create_file(&self, path: &Path) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)
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
