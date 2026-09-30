//! Which provider backs the store, and how it is chosen: on purpose, never by a silent fallback.
//!
//! `Auto` is DPAPI on Windows and **nothing** elsewhere: a Linux or macOS build without an operating-system keystore refuses to start rather than fall
//! back to a file (design 4.5.1: "fail closed, no plaintext fallback"). The keyfile is chosen by naming it (`OAIY_KEY_PROVIDER=keyfile`, or the
//! configuration that maps to it), it says what it is in its provider information, and it is the only provider that stores a value in the clear.
//!
//! Not built: `os-keyring` (the Secret Service of Linux desktops). The name is accepted and answers `ProviderUnavailable`, so that a machine
//! configured for it fails with a message and does not fall back.

use std::env;
use std::path::Path;

use crate::codec::KeyfileCodec;
use crate::error::KeyError;
use crate::store::{FileStore, KeyStore};

/// The environment variable that names the provider.
pub const ENV_PROVIDER: &str = "OAIY_KEY_PROVIDER";

/// The folder inside the data folder that holds the secrets.
pub const KEYS_DIR: &str = "keys";

/// A provider, as asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderChoice {
    /// The default of the platform: DPAPI on Windows, and an error elsewhere.
    Auto,
    /// `windows-dpapi-file`.
    DpapiFile,
    /// `os-keyring`: the Secret Service. Not built.
    OsKeyring,
    /// `keyfile`: a file per secret, the weakest provider, for headless machines and for tests.
    Keyfile,
}

impl ProviderChoice {
    /// Reads a provider name: `auto`, `windows-dpapi-file`, `os-keyring` or `keyfile` (exactly, in lower case).
    pub fn parse(text: &str) -> Result<ProviderChoice, KeyError> {
        match text {
            "auto" => Ok(ProviderChoice::Auto),
            "windows-dpapi-file" => Ok(ProviderChoice::DpapiFile),
            "os-keyring" => Ok(ProviderChoice::OsKeyring),
            "keyfile" => Ok(ProviderChoice::Keyfile),
            other => Err(KeyError::InvalidProvider(other.chars().take(64).collect())),
        }
    }

    /// `OAIY_KEY_PROVIDER`; unset is `Auto`, and a value that is not a provider is an error (a typo must not select a default).
    pub fn from_env() -> Result<ProviderChoice, KeyError> {
        match env::var(ENV_PROVIDER) {
            Ok(value) => ProviderChoice::parse(&value),
            Err(env::VarError::NotPresent) => Ok(ProviderChoice::Auto),
            Err(env::VarError::NotUnicode(_)) => Err(KeyError::InvalidProvider("(not text)".into())),
        }
    }

    /// The provider `Auto` resolves to on this platform, or why there is none.
    pub fn resolve(self) -> Result<ProviderChoice, KeyError> {
        match self {
            ProviderChoice::Auto if cfg!(windows) => Ok(ProviderChoice::DpapiFile),
            ProviderChoice::Auto => Err(KeyError::ProviderUnavailable(
                "no operating-system keystore is built for this platform; name the keyfile provider on purpose (OAIY_KEY_PROVIDER=keyfile)",
            )),
            other => Ok(other),
        }
    }
}

/// Opens the keystore in `<data_dir>/keys`, with the provider `choice` resolves to.
pub fn open(data_dir: &Path, choice: ProviderChoice) -> Result<Box<dyn KeyStore>, KeyError> {
    open_at(data_dir.join(KEYS_DIR), choice)
}

/// Opens the keystore in the given directory itself.
pub fn open_at(keys_dir: impl AsRef<Path>, choice: ProviderChoice) -> Result<Box<dyn KeyStore>, KeyError> {
    let keys_dir = keys_dir.as_ref();
    match choice.resolve()? {
        ProviderChoice::Keyfile => Ok(Box::new(FileStore::open(keys_dir, KeyfileCodec)?)),
        #[cfg(windows)]
        ProviderChoice::DpapiFile => Ok(Box::new(FileStore::open(keys_dir, crate::codec::DpapiCodec)?)),
        #[cfg(not(windows))]
        ProviderChoice::DpapiFile => Err(KeyError::ProviderUnavailable("DPAPI exists on Windows only")),
        ProviderChoice::OsKeyring => Err(KeyError::ProviderUnavailable("the os-keyring (Secret Service) provider is not built")),
        ProviderChoice::Auto => Err(KeyError::ProviderUnavailable("unresolved provider")),
    }
}
