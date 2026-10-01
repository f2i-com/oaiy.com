//! What a keystore refuses, and why, without ever putting a secret in the message.
//!
//! The one distinction that matters is the one the trait is built on: **a name that was never stored is `Ok(None)`; everything else that
//! stops a read is an error.** A store that cannot read its key must say so, and refuse to mint a new identity or to treat a missing file
//! as a first run (the rule the tunnel identity, the data-node key and the app-logic markers already follow).

use core::fmt;
use std::io;

/// A keystore failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum KeyError {
    /// A name outside `^[a-z0-9][a-z0-9._-]{0,79}$`, or a device name.
    InvalidName,
    /// A value that is empty or larger than 64 KiB.
    InvalidValue(&'static str),
    /// The file system refused: which step, and what it said.
    Io {
        /// The step (`read`, `create temp file`, `rename`...).
        op: &'static str,
        /// The operating system's error.
        source: io::Error,
    },
    /// A file that is not a blob of this keystore: a wrong magic, a truncation, a failed check.
    Corrupt(&'static str),
    /// A blob that is intact but was made for another name (a file moved or copied under another name).
    WrongName,
    /// The operating system's data protection refused (a blob of another user, another machine, a reset password).
    Os(&'static str, u32),
    /// A key file or its directory is looser than the keyfile provider allows, or is a symbolic link.
    Permissions(String),
    /// What `put` read back is not what it wrote, so nothing was replaced.
    Verify,
    /// The provider asked for does not exist in this build or on this machine.
    ProviderUnavailable(&'static str),
    /// `OAIY_KEY_PROVIDER` holds something that is not a provider.
    InvalidProvider(String),
    /// The keys folder holds values made by another provider than the one asked for (a folder remembers its provider in `.provider`, and a file of the other
    /// provider's kind is refused too). Nothing was read or changed: a secret that the other provider holds is not "never stored", and it is not replaced by a
    /// second value under the same name.
    ProviderMismatch {
        /// The provider the folder belongs to (as the folder says, cut to 64 characters).
        stored: String,
        /// The provider that was asked for.
        requested: &'static str,
    },
}

impl KeyError {
    /// A stable machine code for status pages and logs.
    pub const fn code(&self) -> &'static str {
        match self {
            KeyError::InvalidName => "key_invalid_name",
            KeyError::InvalidValue(_) => "key_invalid_value",
            KeyError::Io { .. } => "key_io",
            KeyError::Corrupt(_) => "key_corrupt",
            KeyError::WrongName => "key_wrong_name",
            KeyError::Os(_, _) => "key_os",
            KeyError::Permissions(_) => "key_permissions",
            KeyError::Verify => "key_verify_failed",
            KeyError::ProviderUnavailable(_) => "key_provider_unavailable",
            KeyError::InvalidProvider(_) => "key_invalid_provider",
            KeyError::ProviderMismatch { .. } => "key_provider_mismatch",
        }
    }

    pub(crate) fn io(op: &'static str, source: io::Error) -> KeyError {
        KeyError::Io { op, source }
    }
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyError::InvalidName => f.write_str("key_invalid_name"),
            KeyError::InvalidValue(why) => write!(f, "key_invalid_value: {why}"),
            KeyError::Io { op, source } => write!(f, "key_io: {op}: {source}"),
            KeyError::Corrupt(why) => write!(f, "key_corrupt: {why}"),
            KeyError::WrongName => f.write_str("key_wrong_name: the blob was made for another name"),
            KeyError::Os(what, code) => write!(f, "key_os: {what} failed with code {code:#x}"),
            KeyError::Permissions(why) => write!(f, "key_permissions: {why}"),
            KeyError::Verify => f.write_str("key_verify_failed: what was read back is not what was written"),
            KeyError::ProviderUnavailable(why) => write!(f, "key_provider_unavailable: {why}"),
            KeyError::InvalidProvider(value) => write!(f, "key_invalid_provider: {value:?}"),
            KeyError::ProviderMismatch { stored, requested } => write!(
                f,
                "key_provider_mismatch: the keys folder holds values of the {stored:?} provider and {requested:?} was asked for; open it with the provider it was made with (nothing was read or changed)"
            ),
        }
    }
}

impl std::error::Error for KeyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KeyError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
