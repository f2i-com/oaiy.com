//! oaiy-keystore: K1, the named-secret keystore of the OAIY desktop (design `vault.final.md` 4.5.1, work package V-02).
//!
//! ```text
//! let store = oaiy_keystore::open(&data_dir, ProviderChoice::from_env()?)?;
//! store.put(&Name::new("archive.writer")?, &seed)?;
//! match store.get(&Name::new("archive.writer")?)? {
//!     Some(seed) => { /* use it */ }
//!     None => { /* never stored: only now may a caller mint one */ }
//! }                                   // an Err is "could not read", and is never the same as None
//! ```
//!
//! What the crate promises, and where each promise is tested (`tests/keystore.rs` and the unit tests):
//!
//! - **`Ok(None)` is "never stored", an error is "could not read".** A corrupt file, a file moved to another name, a file another user protected, a
//!   locked file, a directory in place of a file, a keys directory that has vanished: all errors.
//! - **The name is bound into each blob.** DPAPI entropy on Windows, a tag inside the file for the keyfile; a blob copied under another name fails.
//! - **No plaintext beside the blob.** DPAPI files hold DPAPI blobs only; the keyfile provider holds the value in one file and nowhere else (no
//!   temporary file survives a success, a failure or a restart).
//! - **`put` is atomic and verified**: written to a temporary file, flushed, read back through the provider, compared, then renamed. **`Err` means nothing changed**:
//!   the previous value is still the value (every failure before the rename, and the rename itself). **`Ok` means the value is stored and every reader sees it**, and the
//!   [`Durability`] it carries says whether the folder was flushed after the rename: a file system that cannot flush a folder (an SMB share, FAT) or a flush that fails after the
//!   rename is `Ok(Durability::Unconfirmed)`, never an `Err`, because the change is made and an `Err` would tell the caller the opposite. `delete` is the same. The network share
//!   question is answered in the README ("Network shares and redirected profiles").
//! - **A store that is open and finds its folder emptied is an error.** The provider marker and the lock file are looked for before each operation; a folder that has been wiped while
//!   it was open answers with an error, never with `None` for every key that was in it.
//! - **Fail closed.** No provider is chosen for you where none is strong; the keyfile is named on purpose and refuses a directory or file with any group
//!   or other permission, a symbolic link, or another owner.
//! - **Values are zeroized** (`Zeroizing<Vec<u8>>` out, buffers wiped, the memory DPAPI allocated overwritten before it is freed).
//!
//! Unsafe code: two modules, on Windows only, each block with its safety argument: `dpapi` (the two DPAPI calls and the release of the memory they allocate) and `winfs` (which file
//! a handle is, how many names it has, where it really is). Everything else is `#![deny(unsafe_code)]`, and `rustix` (Unix) is used through its safe API only.

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod codec;
#[cfg(windows)]
#[allow(unsafe_code)]
mod dpapi;
mod error;
mod keydir;
mod name;
pub mod perm;
mod provider;
mod store;
#[cfg(windows)]
#[allow(unsafe_code)]
mod winfs;

pub use codec::{ProviderInfo, Strength};
pub use error::KeyError;
pub use name::{names, Name, MAX_NAME_LEN};
pub use provider::{open, open_at, ProviderChoice, ENV_PROVIDER, KEYS_DIR};
pub use store::{Durability, KeyStore, MAX_VALUE_LEN};
