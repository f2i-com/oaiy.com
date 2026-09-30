//! The trait every provider implements, and the one file-per-secret store that both providers share.
//!
//! Every rule of design 4.5.1 that is not about how a blob is made lives here, once: `Ok(None)` means the name was never stored and nothing else
//! does; `put` writes a temporary file (created new, never overwriting, with only what the provider makes of the value in it), flushes it, reads
//! it back through the provider and compares, and only then renames it over the old value, so a failure at any step leaves the previous value
//! and no file of the failed attempt; `delete` is idempotent; `list` shows valid names only; a keys directory that has disappeared is an error, not an
//! empty store.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use oaiy_crypto::kdf::hex_lower;
use oaiy_crypto::zeroize::{ct_eq, Secret};
use zeroize::Zeroizing;

use crate::codec::{Codec, ProviderInfo};
use crate::error::KeyError;
use crate::name::Name;

/// The most a value may be: 64 KiB.
pub const MAX_VALUE_LEN: usize = 64 * 1024;

/// A blob is never larger than a value plus a provider's framing; a file larger than this is not one of ours and is not read into memory.
const MAX_BLOB_LEN: usize = MAX_VALUE_LEN + 4096;

/// K1: named secrets. `Ok(None)` and `Err` are not the same thing.
pub trait KeyStore: Send + Sync {
    /// What backs this store, for the status page.
    fn provider(&self) -> ProviderInfo;

    /// The secret stored under `name`. **`Ok(None)` is "never stored"; an error is "could not read"**, and the two must never be confused: a caller that
    /// meets an error does not mint a new identity and does not treat the name as a first run.
    fn get(&self, name: &Name) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError>;

    /// Stores `value` (1 byte to 64 KiB) under `name`, replacing any earlier one. Atomic: the value is written, read back and compared before it replaces
    /// anything, and a failure leaves the previous value in place.
    fn put(&self, name: &Name, value: &[u8]) -> Result<(), KeyError>;

    /// Removes `name`. Removing a name that is not there is not an error. Call it only after the replacement of what it held has been verified.
    fn delete(&self, name: &Name) -> Result<(), KeyError>;

    /// The names that start with `prefix`, in order.
    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError>;
}

/// One file per secret in one directory.
pub(crate) struct FileStore<C: Codec> {
    dir: PathBuf,
    codec: C,
    owner: Option<u32>,
}

impl<C: Codec> FileStore<C> {
    /// Opens (creating it if need be) the keys directory, checks it, and removes the debris of an interrupted write.
    pub(crate) fn open(dir: PathBuf, codec: C) -> Result<Self, KeyError> {
        codec.prepare_dir(&dir)?;
        codec.check_dir(&dir, None)?;
        let owner = codec.probe_owner(&dir)?;
        codec.check_dir(&dir, owner)?;
        let store = FileStore { dir, codec, owner };
        store.remove_stale_temporaries();
        Ok(store)
    }

    fn path(&self, name: &Name) -> PathBuf {
        self.dir.join(format!("{}.{}", name.as_str(), self.codec.extension()))
    }

    /// The directory is checked before every operation: one that has gone is a failure, not an empty store.
    fn require_dir(&self) -> Result<(), KeyError> {
        let meta = fs::metadata(&self.dir).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        if !meta.is_dir() {
            return Err(KeyError::io("inspect the keys directory", io::Error::other("not a directory")));
        }
        self.codec.check_dir(&self.dir, self.owner)
    }

    /// Temporary files are `.<name>.<16 hex>.tmp`: they can never be a name, so `list` and `get` cannot see them. A crash between the write and the rename leaves one;
    /// with the keyfile provider it holds a value in the clear, so it is removed the next time the store is opened.
    fn remove_stale_temporaries(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else { return };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(text) = file_name.to_str() else { continue };
            if text.starts_with('.') && text.ends_with(".tmp") && entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    fn temporary_path(&self, name: &Name) -> Result<PathBuf, KeyError> {
        let random: Secret<8> = Secret::random().map_err(|_| KeyError::io("random", io::Error::other("the random generator failed")))?;
        Ok(self.dir.join(format!(".{}.{}.tmp", name.as_str(), hex_lower(random.expose()))))
    }

    /// Writes the blob to `tmp`, flushes it, reads it back through the provider and compares.
    fn write_and_verify(&self, tmp: &Path, name: &Name, value: &[u8], blob: &[u8]) -> Result<(), KeyError> {
        let mut file = self.codec.create_file(tmp).map_err(|e| KeyError::io("create the temporary file", e))?;
        file.write_all(blob).map_err(|e| KeyError::io("write the temporary file", e))?;
        file.sync_all().map_err(|e| KeyError::io("flush the temporary file", e))?;
        drop(file);
        let back = read_bounded(tmp)?.ok_or(KeyError::Verify)?;
        let opened = self.codec.open(name, &back)?;
        if !ct_eq(&opened, value) {
            return Err(KeyError::Verify);
        }
        Ok(())
    }
}

/// Reads a whole file into one buffer of exactly its size (a buffer that grows leaves earlier, smaller copies of a secret behind in freed memory),
/// refusing one that is too big to be a blob. `Ok(None)` if there is no such file.
fn read_bounded(path: &Path) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(KeyError::io("open a key file", e)),
    };
    let length = file.metadata().map_err(|e| KeyError::io("inspect a key file", e))?.len();
    if length > MAX_BLOB_LEN as u64 {
        return Err(KeyError::Corrupt("larger than any secret"));
    }
    let mut bytes = Zeroizing::new(vec![0u8; length as usize]);
    file.read_exact(&mut bytes).map_err(|e| KeyError::io("read a key file", e))?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra).map_err(|e| KeyError::io("read a key file", e))? != 0 {
        return Err(KeyError::Corrupt("the file changed while it was read"));
    }
    Ok(Some(bytes))
}

#[cfg(unix)]
fn sync_directory(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory(_dir: &Path) {}

impl<C: Codec> KeyStore for FileStore<C> {
    fn provider(&self) -> ProviderInfo {
        self.codec.info()
    }

    fn get(&self, name: &Name) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
        self.require_dir()?;
        let path = self.path(name);
        match self.codec.check_file(&path, self.owner) {
            Ok(()) => {}
            Err(KeyError::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(other) => return Err(other),
        }
        let Some(blob) = read_bounded(&path)? else { return Ok(None) };
        self.codec.open(name, &blob).map(Some)
    }

    fn put(&self, name: &Name, value: &[u8]) -> Result<(), KeyError> {
        if value.is_empty() {
            return Err(KeyError::InvalidValue("empty"));
        }
        if value.len() > MAX_VALUE_LEN {
            return Err(KeyError::InvalidValue("larger than 64 KiB"));
        }
        self.require_dir()?;
        let blob = self.codec.seal(name, value)?;
        let tmp = self.temporary_path(name)?;
        if let Err(error) = self.write_and_verify(&tmp, name, value, &blob) {
            let _ = fs::remove_file(&tmp);
            return Err(error);
        }
        if let Err(error) = fs::rename(&tmp, self.path(name)) {
            let _ = fs::remove_file(&tmp);
            return Err(KeyError::io("replace the key file", error));
        }
        sync_directory(&self.dir);
        Ok(())
    }

    fn delete(&self, name: &Name) -> Result<(), KeyError> {
        self.require_dir()?;
        match fs::remove_file(self.path(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KeyError::io("remove a key file", e)),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError> {
        self.require_dir()?;
        let suffix = format!(".{}", self.codec.extension());
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.dir).map_err(|e| KeyError::io("list the keys directory", e))? {
            let entry = entry.map_err(|e| KeyError::io("list the keys directory", e))?;
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let file_name = entry.file_name();
            let Some(stem) = file_name.to_str().and_then(|f| f.strip_suffix(suffix.as_str())) else { continue };
            if let Ok(name) = Name::new(stem) {
                if name.as_str().starts_with(prefix) {
                    names.push(name);
                }
            }
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{KeyfileCodec, Strength};
    use std::sync::atomic::{AtomicU8, Ordering};

    /// What the wrapped codec should do wrong next.
    const FAULT_NONE: u8 = 0;
    /// `open` returns a value that is not the one in the file (a provider that reads back something else).
    const FAULT_WRONG_VALUE: u8 = 1;
    /// `open` fails.
    const FAULT_OPEN_FAILS: u8 = 2;
    /// `create_file` fails (the disk is full, the directory went read-only).
    const FAULT_CREATE_FAILS: u8 = 3;

    struct Faulty {
        inner: KeyfileCodec,
        fault: AtomicU8,
    }

    impl Codec for Faulty {
        fn info(&self) -> ProviderInfo {
            ProviderInfo { id: "faulty", strength: Strength::FilePermissions, description: "test" }
        }
        fn extension(&self) -> &'static str {
            "kf"
        }
        fn seal(&self, name: &Name, value: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
            self.inner.seal(name, value)
        }
        fn open(&self, name: &Name, blob: &[u8]) -> Result<Zeroizing<Vec<u8>>, KeyError> {
            match self.fault.load(Ordering::SeqCst) {
                FAULT_WRONG_VALUE => Ok(Zeroizing::new(vec![0u8; self.inner.open(name, blob)?.len()])),
                FAULT_OPEN_FAILS => Err(KeyError::Corrupt("injected")),
                _ => self.inner.open(name, blob),
            }
        }
        fn create_file(&self, path: &Path) -> io::Result<File> {
            if self.fault.load(Ordering::SeqCst) == FAULT_CREATE_FAILS {
                return Err(io::Error::other("injected: no space left on device"));
            }
            self.inner.create_file(path)
        }
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            let dir = std::env::temp_dir().join(format!("oaiy-keystore-unit-{tag}-{}-{nanos}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn faulty(scratch: &Scratch) -> FileStore<Faulty> {
        FileStore::open(scratch.0.join("keys"), Faulty { inner: KeyfileCodec, fault: AtomicU8::new(FAULT_NONE) }).unwrap()
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut out: Vec<String> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        out.sort();
        out
    }

    fn name(text: &str) -> Name {
        Name::new(text).unwrap()
    }

    /// Design 4.5.1: "`put` reads back and compares, else `Err`" and "a failure leaves the previous value". A provider that reads back something else
    /// makes `put` fail with `Verify`, the old value is still there, and no file of the failed attempt is left.
    #[test]
    fn a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris() {
        let scratch = Scratch::new("verify");
        let store = faulty(&scratch);
        let n = name("archive.writer");
        store.put(&n, b"the old value").unwrap();
        let before = files(&store.dir);
        assert_eq!(before, vec!["archive.writer.kf".to_string()]);

        store.codec.fault.store(FAULT_WRONG_VALUE, Ordering::SeqCst);
        assert!(matches!(store.put(&n, b"the new value"), Err(KeyError::Verify)));
        store.codec.fault.store(FAULT_NONE, Ordering::SeqCst);

        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"the old value", "the previous value is still there");
        assert_eq!(files(&store.dir), before, "no temporary file of the failed attempt is left");
    }

    #[test]
    fn a_put_whose_read_back_fails_is_an_error_and_changes_nothing() {
        let scratch = Scratch::new("openfails");
        let store = faulty(&scratch);
        let n = name("backup.sig");
        store.put(&n, b"first").unwrap();
        store.codec.fault.store(FAULT_OPEN_FAILS, Ordering::SeqCst);
        assert!(store.put(&n, b"second").is_err());
        store.codec.fault.store(FAULT_NONE, Ordering::SeqCst);
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first");
        assert_eq!(files(&store.dir), vec!["backup.sig.kf".to_string()]);
    }

    #[test]
    fn a_put_that_cannot_create_its_temporary_file_is_an_error_and_changes_nothing() {
        let scratch = Scratch::new("create");
        let store = faulty(&scratch);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        store.codec.fault.store(FAULT_CREATE_FAILS, Ordering::SeqCst);
        assert!(matches!(store.put(&n, b"second"), Err(KeyError::Io { op: "create the temporary file", .. })));
        // and a name that was never stored is still not stored, not an empty file
        assert!(matches!(store.put(&name("never.stored"), b"x"), Err(KeyError::Io { .. })));
        store.codec.fault.store(FAULT_NONE, Ordering::SeqCst);
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first");
        assert!(store.get(&name("never.stored")).unwrap().is_none());
        assert_eq!(files(&store.dir), vec!["relay.token.kf".to_string()]);
    }

    /// A crash between the write and the rename leaves a temporary file. With the keyfile provider it holds a value in the clear, so opening the store removes it.
    #[test]
    fn opening_the_store_removes_the_debris_of_an_interrupted_put() {
        let scratch = Scratch::new("debris");
        let dir = scratch.0.join("keys");
        {
            let store = FileStore::open(dir.clone(), KeyfileCodec).unwrap();
            store.put(&name("keep.me"), b"kept").unwrap();
        }
        fs::write(dir.join(".keep.me.0123456789abcdef.tmp"), b"a value in the clear").unwrap();
        fs::write(dir.join(".other.fedcba9876543210.tmp"), b"another").unwrap();
        fs::write(dir.join("notes.txt"), b"not ours").unwrap();
        let store = FileStore::open(dir.clone(), KeyfileCodec).unwrap();
        assert_eq!(files(&dir), vec!["keep.me.kf".to_string(), "notes.txt".to_string()]);
        assert_eq!(&**store.get(&name("keep.me")).unwrap().unwrap(), b"kept");
    }
}
