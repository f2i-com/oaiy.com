//! The trait every provider implements, and the one file-per-secret store that both providers share.
//!
//! Every rule of design 4.5.1 that is not about how a blob is made lives here, once: `Ok(None)` means the name was never stored and nothing else
//! does; `put` writes a temporary file (created new, never overwriting, with only what the provider makes of the value in it), flushes it, reads
//! it back through the provider and compares, and only then renames it over the old value, so a failure at any step **before the rename** leaves the previous value
//! and no file of the failed attempt; `delete` is idempotent; `list` shows valid names only; a keys directory that has disappeared, or one whose provider marker has gone, is an error, not an
//! empty store. Every file is reached through the folder the store holds open ([`crate::keydir`]), never through a path looked up again.
//!
//! **What an error means, and what comes after the rename** (third review, M-A). `Err` from `put` or `delete` means that nothing changed: the previous value is still the value. Once the
//! rename (or the removal) has been made the change is made, and nothing that comes after it can be an error, or the contract would be false: the step after it is the flush of the
//! folder, which is what makes the new name survive a power cut, and it fails on file systems that cannot flush a folder (an SMB share: `FlushFileBuffers` on the folder handle is
//! "incorrect function", os error 1; FAT; some network and FUSE ones). That outcome is `Ok(Durability::Unconfirmed)`: the change is made and visible to every reader, and a power cut
//! right after it could undo the rename (the file itself was flushed before it). `Ok(Durability::Confirmed)` says the folder was flushed too.

use std::io::Write;
use std::path::Path;

use oaiy_crypto::kdf::hex_lower;
use oaiy_crypto::zeroize::{ct_eq, Secret};
use zeroize::Zeroizing;

use crate::codec::{Codec, ProviderInfo};
use crate::error::KeyError;
use crate::keydir::{KeyDir, STALE_AFTER};
use crate::name::Name;

/// The most a value may be: 64 KiB.
pub const MAX_VALUE_LEN: usize = 64 * 1024;

/// What the store can say about a change it has made (`put` and `delete`, when they succeed): whether the folder was flushed after it. **The change is made either way**: it is
/// visible to every reader, and the previous state is gone. The difference is the power cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// The change was made and the folder was flushed after it: the new name (or the absence of the old one) survives a power cut.
    Confirmed,
    /// The change was made, and the flush of the folder that makes it survive a power cut did not happen: the file system cannot flush a folder (an SMB share, FAT, some network and FUSE
    /// ones) or the flush failed after the change. The new value (or the removal) is stored and is what every reader sees; a power cut right after it could bring the previous state
    /// back (the file itself was flushed before it was renamed into place, so what comes back is whole). **A caller whose next step destroys the only other copy of what was replaced**
    /// (a rotation that has re-wrapped data under the new key, a delete that the caller is not able to redo) should treat this as "not safe yet": keep the old copy until a later
    /// change reports `Confirmed`, or until the next start finds the new value there.
    Unconfirmed,
}

/// K1: named secrets. `Ok(None)` and `Err` are not the same thing.
pub trait KeyStore: Send + Sync {
    /// What backs this store, for the status page.
    fn provider(&self) -> ProviderInfo;

    /// The secret stored under `name`. **`Ok(None)` is "never stored"; an error is "could not read"**, and the two must never be confused: a caller that
    /// meets an error does not mint a new identity and does not treat the name as a first run.
    fn get(&self, name: &Name) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError>;

    /// Stores `value` (1 byte to 64 KiB) under `name`, replacing any earlier one. Atomic: the value is written, read back and compared before it replaces
    /// anything. **`Err` means nothing changed**: the previous value is still the value, and no file of the attempt is left (every failure that comes before the rename,
    /// and the rename itself). **`Ok` means the value is stored** and every reader sees it; the [`Durability`] says whether the folder was flushed after the rename. A failure to flush the folder
    /// cannot be an `Err`, because the value is already in place: it is `Ok(Durability::Unconfirmed)`.
    fn put(&self, name: &Name, value: &[u8]) -> Result<Durability, KeyError>;

    /// Removes `name`. Removing a name that is not there is not an error. Call it only after the replacement of what it held has been verified. **`Err` means the name is still
    /// there**; `Ok` means it has gone, with the same [`Durability`] as `put` (`Unconfirmed`: a power cut right after could bring the file back).
    fn delete(&self, name: &Name) -> Result<Durability, KeyError>;

    /// The names that start with `prefix`, in order.
    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError>;
}

/// The file that says which provider made the values in the folder. It is written when the folder is first opened and never changes: a folder remembers its
/// provider and refuses to be opened with another (review M-2).
const MARKER_FILE: &str = ".provider";

/// The providers that keep one file per secret, by the extension of their files: what a folder can hold.
const PROVIDERS: [(&str, &str); 2] = [("ks", "windows-dpapi-file"), ("kf", "keyfile")];

/// One file per secret in one directory.
pub(crate) struct FileStore<C: Codec> {
    dir: KeyDir,
    codec: C,
}

impl<C: Codec> FileStore<C> {
    /// Opens (creating it if need be) the keys directory, checks it, makes sure that it belongs to this provider, and removes the debris of an interrupted
    /// write.
    pub(crate) fn open(dir: &Path, codec: C) -> Result<Self, KeyError> {
        Self::open_prepared(dir, codec, |_| {})
    }

    /// [`FileStore::open`], with `prepare` run on the folder once it is held and before anything is made in it (a test uses it to make the first flush fail).
    fn open_prepared(dir: &Path, codec: C, prepare: impl FnOnce(&KeyDir)) -> Result<Self, KeyError> {
        let store = FileStore { dir: KeyDir::open(dir)?, codec };
        prepare(&store.dir);
        {
            let _lock = store.dir.lock(true)?;
            store.claim_folder()?;
        }
        store.remove_stale_temporaries();
        Ok(store)
    }

    fn file_name(&self, name: &Name) -> String {
        format!("{}.{}", name.as_str(), self.codec.extension())
    }

    /// **The folder belongs to one provider** (review M-2). Reopened with another provider (a machine that was switched to `OAIY_KEY_PROVIDER=keyfile`, a
    /// build without DPAPI), a store used to read every secret of the first as `None`, list nothing, and on a `put` leave two live values under one name, one of
    /// them in the clear. Now the marker file names the provider, a folder that has none is adopted only if it holds no file of another provider's kind, and
    /// anything else is `ProviderMismatch` before anything is read. Must be called with the exclusive lock held.
    fn claim_folder(&self) -> Result<(), KeyError> {
        let info = self.codec.info();
        if let Some(bytes) = self.dir.read(MARKER_FILE)? {
            let stored = parse_marker(&bytes)?;
            if stored != info.id {
                return Err(KeyError::ProviderMismatch { stored, requested: info.id });
            }
            return Ok(());
        }
        // no marker: a new folder, or one made before folders remembered their provider. It may hold this provider's files, and no other's.
        for entry in self.dir.list()? {
            if !entry.is_file {
                continue;
            }
            let Some((stem, extension)) = entry.name.rsplit_once('.') else { continue };
            if extension == self.codec.extension() || Name::new(stem).is_err() {
                continue;
            }
            if let Some((_, other)) = PROVIDERS.iter().find(|(e, _)| *e == extension) {
                return Err(KeyError::ProviderMismatch { stored: (*other).to_owned(), requested: info.id });
            }
        }
        let tmp = self.temporary_name(&Name::new("provider")?)?;
        if let Err(error) = self.stage(&tmp, format!("{}\n", info.id).as_bytes()).and_then(|()| self.dir.rename(&tmp, MARKER_FILE)) {
            let _ = self.dir.remove(&tmp);
            return Err(error);
        }
        // The marker is in place. Flushing the folder makes the marker survive a power cut, and a folder that cannot be flushed (an SMB share: os error 1) is no reason for the open
        // to fail: a marker that a power cut takes back is made again by the next open, which adopts a folder that has none (above). Whatever the flush says, the store is open.
        let _ = self.dir.sync();
        Ok(())
    }

    /// **The marker must still be there, and still be this provider's** (third review, L-5). A folder that is wiped while it is open (`remove_dir_all` on Windows deletes everything in it and
    /// then fails to remove the folder, which the store holds) is an empty folder to a store that does not look, and `get` answered `Ok(None)`, "never stored", for every key that had
    /// been there. The marker is written when the folder is first opened and never changes, so a store that is open and does not find it has been emptied (or opened on another folder):
    /// an error, whatever the name. The lock file is looked for the same way (it is made when the folder is opened, and an operation does not make it again).
    fn check_marker(&self) -> Result<(), KeyError> {
        let info = self.codec.info();
        let Some(bytes) = self.dir.read(MARKER_FILE)? else {
            return Err(KeyError::Corrupt("the provider marker of the keys folder has gone: the folder was emptied while it was open"));
        };
        let stored = parse_marker(&bytes)?;
        if stored != info.id {
            return Err(KeyError::ProviderMismatch { stored, requested: info.id });
        }
        Ok(())
    }

    /// A value of another provider's kind under this name is an error, not "nothing there": it is a secret that this provider cannot read, and a `put` beside
    /// it would leave two live values.
    fn refuse_a_value_of_another_provider(&self, name: &Name) -> Result<(), KeyError> {
        for (extension, other) in PROVIDERS {
            if extension != self.codec.extension() && self.dir.exists(&format!("{}.{extension}", name.as_str()))? {
                return Err(KeyError::ProviderMismatch { stored: other.to_owned(), requested: self.codec.info().id });
            }
        }
        Ok(())
    }

    /// Temporary files are `.<name>.<16 hex>.tmp`: they can never be a name, so `list` and `get` cannot see them. A crash between the write and the rename leaves one;
    /// with the keyfile provider it holds a value in the clear, so it is removed the next time the store is opened. **Only debris is removed** (review L-6): a
    /// file of exactly that shape, older than a minute, that nobody has locked. The temporary file of a `put` in another process is young, and locked while it is
    /// written; opening the store used to delete it, and the put failed at its read-back with `key_verify_failed`.
    fn remove_stale_temporaries(&self) {
        let Ok(entries) = self.dir.list() else { return };
        for entry in entries {
            if entry.is_file && is_temporary_name(&entry.name) {
                self.dir.remove_if_stale(&entry.name, STALE_AFTER);
            }
        }
    }

    fn temporary_name(&self, name: &Name) -> Result<String, KeyError> {
        let random: Secret<8> = Secret::random().map_err(|_| KeyError::io("random", std::io::Error::other("the random generator failed")))?;
        Ok(format!(".{}.{}.tmp", name.as_str(), hex_lower(random.expose())))
    }

    /// Flushes the folder after a change that is already made: `Confirmed` if it was, `Unconfirmed` if the file system cannot flush a folder or the flush failed. Never an error, because
    /// the change cannot be undone by a failure that comes after it (see the header of this file).
    fn flushed_after_the_change(&self) -> Durability {
        self.dir.sync().unwrap_or(Durability::Unconfirmed)
    }

    /// Makes `tmp` (which must not exist) with `bytes` in it and flushes it. The file is locked while it is written, so that a process that is looking for debris
    /// can tell this one is not (a file system that cannot lock is no reason to fail: the age rule still protects a young file).
    fn stage(&self, tmp: &str, bytes: &[u8]) -> Result<(), KeyError> {
        let mut file = self.dir.create_new(tmp)?;
        let locked = file.try_lock().is_ok();
        self.dir.fire("staging");
        let written = file
            .write_all(bytes)
            .map_err(|e| KeyError::io("write the temporary file", e))
            .and_then(|()| file.sync_all().map_err(|e| KeyError::io("flush the temporary file", e)));
        if locked {
            let _ = file.unlock();
        }
        written
    }

    /// Writes the blob to `tmp`, flushes it, reads it back through the provider and compares.
    fn write_and_verify(&self, tmp: &str, name: &Name, value: &[u8], blob: &[u8]) -> Result<(), KeyError> {
        self.stage(tmp, blob)?;
        let back = self.dir.read(tmp)?.ok_or(KeyError::Verify)?;
        let opened = self.codec.open(name, &back)?;
        if !ct_eq(&opened, value) {
            return Err(KeyError::Verify);
        }
        Ok(())
    }
}

/// Whether a file name is one of the store's temporary files: `.<name>.<16 lower-case hex>.tmp`, and nothing else (a file someone else put there is not debris).
fn is_temporary_name(file_name: &str) -> bool {
    let Some(inner) = file_name.strip_prefix('.').and_then(|rest| rest.strip_suffix(".tmp")) else { return false };
    let Some((stem, random)) = inner.rsplit_once('.') else { return false };
    random.len() == 16 && random.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) && Name::new(stem).is_ok()
}

/// What the marker file says: the provider's identifier, and a line feed. Anything else is not a marker this keystore wrote.
fn parse_marker(bytes: &[u8]) -> Result<String, KeyError> {
    const NOT_A_MARKER: KeyError = KeyError::Corrupt("the provider marker of the keys folder is not one this keystore wrote");
    let text = std::str::from_utf8(bytes).map_err(|_| NOT_A_MARKER)?;
    match text.strip_suffix('\n') {
        Some(id) if (1..=64).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') => {
            Ok(id.to_owned())
        }
        _ => Err(NOT_A_MARKER),
    }
}

impl<C: Codec> KeyStore for FileStore<C> {
    fn provider(&self) -> ProviderInfo {
        self.codec.info()
    }

    fn get(&self, name: &Name) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
        self.dir.verify()?;
        self.check_marker()?;
        // the lock, shared, is what makes `None` mean "not there": a writer that is replacing the name holds it exclusively (review H-1)
        let blob = {
            let _lock = self.dir.lock(false)?;
            self.refuse_a_value_of_another_provider(name)?;
            self.dir.read(&self.file_name(name))?
        };
        let Some(blob) = blob else { return Ok(None) };
        self.codec.open(name, &blob).map(Some)
    }

    fn put(&self, name: &Name, value: &[u8]) -> Result<Durability, KeyError> {
        if value.is_empty() {
            return Err(KeyError::InvalidValue("empty"));
        }
        if value.len() > MAX_VALUE_LEN {
            return Err(KeyError::InvalidValue("larger than 64 KiB"));
        }
        self.dir.verify()?;
        self.check_marker()?;
        self.refuse_a_value_of_another_provider(name)?;
        let blob = self.codec.seal(name, value)?;
        let tmp = self.temporary_name(name)?;
        if let Err(error) = self.write_and_verify(&tmp, name, value, &blob) {
            let _ = self.dir.remove(&tmp);
            return Err(error);
        }
        self.dir.fire("before_rename");
        // the exclusive lock is held for the rename only, not for the write and the read-back: readers wait for one rename, never for a disk
        let renamed = self.dir.lock(true).and_then(|_lock| {
            self.dir.fire("rename_locked");
            self.dir.rename(&tmp, &self.file_name(name))
        });
        if let Err(error) = renamed {
            let _ = self.dir.remove(&tmp);
            return Err(error);
        }
        // The new value is in place, and what is left is the flush that makes the name survive a power cut. It cannot be an error now (third review, M-A): over SMB it fails on every
        // put with "incorrect function" after the rename, and a caller that saw `Err` kept the old key in memory while the new one was on disk and the old one was gone.
        Ok(self.flushed_after_the_change())
    }

    fn delete(&self, name: &Name) -> Result<Durability, KeyError> {
        self.dir.verify()?;
        self.check_marker()?;
        {
            let _lock = self.dir.lock(true)?;
            self.refuse_a_value_of_another_provider(name)?;
            self.dir.remove(&self.file_name(name))?;
        }
        Ok(self.flushed_after_the_change())
    }

    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError> {
        self.dir.verify()?;
        self.check_marker()?;
        let own = self.codec.extension();
        let mut names = Vec::new();
        let entries = {
            let _lock = self.dir.lock(false)?;
            self.dir.list()?
        };
        for entry in entries {
            if !entry.is_file {
                continue;
            }
            let Some((stem, extension)) = entry.name.rsplit_once('.') else { continue };
            let Ok(name) = Name::new(stem) else { continue };
            if extension != own {
                // a secret of another provider's kind is not "absent from the list": the caller would not know it exists
                if let Some((_, other)) = PROVIDERS.iter().find(|(e, _)| *e == extension) {
                    return Err(KeyError::ProviderMismatch { stored: (*other).to_owned(), requested: self.codec.info().id });
                }
                continue;
            }
            if name.as_str().starts_with(prefix) {
                names.push(name);
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
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// What the wrapped codec should do wrong next.
    const FAULT_NONE: u8 = 0;
    /// `open` returns a value that is not the one in the file (a provider that reads back something else).
    const FAULT_WRONG_VALUE: u8 = 1;
    /// `open` fails.
    const FAULT_OPEN_FAILS: u8 = 2;

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
    }

    pub(crate) struct Scratch(pub(crate) PathBuf);

    impl Scratch {
        pub(crate) fn new(tag: &str) -> Scratch {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            let dir = std::env::temp_dir().join(format!("oaiy-keystore-unit-{tag}-{}-{nanos}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            // a folder above the keys folder that others could rename in is refused (the umask of some systems makes it 0775)
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
            }
            Scratch(dir)
        }

        pub(crate) fn keys(&self) -> PathBuf {
            self.0.join("keys")
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn faulty(scratch: &Scratch) -> FileStore<Faulty> {
        FileStore::open(&scratch.keys(), Faulty { inner: KeyfileCodec, fault: AtomicU8::new(FAULT_NONE) }).unwrap()
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut out: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|f| f != ".lock" && f != ".provider") // the bookkeeping files of the store
            .collect();
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
        let before = files(&scratch.keys());
        assert_eq!(before, vec!["archive.writer.kf".to_string()]);

        store.codec.fault.store(FAULT_WRONG_VALUE, Ordering::SeqCst);
        assert!(matches!(store.put(&n, b"the new value"), Err(KeyError::Verify)));
        store.codec.fault.store(FAULT_NONE, Ordering::SeqCst);

        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"the old value", "the previous value is still there");
        assert_eq!(files(&scratch.keys()), before, "no temporary file of the failed attempt is left");
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
        assert_eq!(files(&scratch.keys()), vec!["backup.sig.kf".to_string()]);
    }

    #[test]
    fn a_put_that_cannot_create_its_temporary_file_is_an_error_and_changes_nothing() {
        let scratch = Scratch::new("create");
        let store = faulty(&scratch);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        store.dir.hooks.fail_create.store(true, Ordering::SeqCst);
        assert!(matches!(store.put(&n, b"second"), Err(KeyError::Io { op: "create the temporary file", .. })));
        // and a name that was never stored is still not stored, not an empty file
        assert!(matches!(store.put(&name("never.stored"), b"x"), Err(KeyError::Io { .. })));
        store.dir.hooks.fail_create.store(false, Ordering::SeqCst);
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first");
        assert!(store.get(&name("never.stored")).unwrap().is_none());
        assert_eq!(files(&scratch.keys()), vec!["relay.token.kf".to_string()]);
    }

    /// Makes the modification time of `path` `seconds` seconds ago.
    fn age(path: &Path, seconds: u64) {
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(seconds)).unwrap();
    }

    /// A crash between the write and the rename leaves a temporary file. With the keyfile provider it holds a value in the clear, so opening the store removes
    /// it (L-6: but only debris: a file of exactly the store's temporary shape, older than a minute, that nobody has locked).
    #[test]
    fn opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else() {
        let scratch = Scratch::new("debris");
        let dir = scratch.keys();
        {
            let store = FileStore::open(&dir, KeyfileCodec).unwrap();
            store.put(&name("keep.me"), b"kept").unwrap();
        }
        let old =
            [".keep.me.0123456789abcdef.tmp", ".other.fedcba9876543210.tmp", ".provider.00000000000000ff.tmp", ".just-over.00000000000000aa.tmp"];
        for debris in old {
            fs::write(dir.join(debris), b"a value in the clear").unwrap();
            age(&dir.join(debris), if debris.starts_with(".just-over") { 70 } else { 120 });
        }
        // the minute: a file of 50 seconds is a put that has been going a while, not debris; one of 70 is (this runs on Windows too, where the lock and the removal are LockFileEx and a
        // delete of a file that this process has open)
        fs::write(dir.join(".just-under.00000000000000bb.tmp"), b"a put in progress").unwrap();
        age(&dir.join(".just-under.00000000000000bb.tmp"), 50);
        // not debris: young (the put of another process that has not renamed yet), locked (an old file that someone is still writing), or not the store's shape
        fs::write(dir.join(".young.0123456789abcdef.tmp"), b"a put in progress").unwrap();
        fs::write(dir.join(".locked.0123456789abcdef.tmp"), b"a slow write").unwrap();
        age(&dir.join(".locked.0123456789abcdef.tmp"), 120);
        let held = fs::OpenOptions::new().write(true).open(dir.join(".locked.0123456789abcdef.tmp")).unwrap();
        held.lock().unwrap();
        let not_ours = [
            ".notes.tmp",
            ".upper.0123456789ABCDEF.tmp",
            ".short.0123456789abcde.tmp",
            ".Capital.0123456789abcdef.tmp",
            ".x.0123456789abcdef.temp",
            "notes.txt",
        ];
        for other in not_ours {
            fs::write(dir.join(other), b"someone else's file").unwrap();
            age(&dir.join(other), 120);
        }
        let store = FileStore::open(&dir, KeyfileCodec).unwrap();
        let mut survivors: Vec<String> =
            [".young.0123456789abcdef.tmp", ".locked.0123456789abcdef.tmp", ".just-under.00000000000000bb.tmp", "keep.me.kf"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        survivors.extend(not_ours.iter().map(|s| s.to_string()));
        survivors.sort();
        assert_eq!(files(&dir), survivors);
        assert_eq!(&**store.get(&name("keep.me")).unwrap().unwrap(), b"kept");
        // once the writer lets go, the old file is debris like any other
        held.unlock().unwrap();
        drop(held);
        drop(FileStore::open(&dir, KeyfileCodec).unwrap());
        assert!(!files(&dir).contains(&".locked.0123456789abcdef.tmp".to_string()));
        assert!(files(&dir).contains(&".young.0123456789abcdef.tmp".to_string()), "a young file is still not debris");
    }

    /// L-6, the reviewer's case made deterministic: another process opens the store while this one is between writing its temporary file and renaming it. The
    /// open used to delete the temporary file, and the put failed with `key_verify_failed` (870 of 874 puts on Linux, 1,201 of 1,265 on Windows, with a second
    /// process opening the store in a loop).
    #[test]
    fn a_put_in_progress_survives_another_process_opening_the_store() {
        use std::sync::{mpsc, Arc, Mutex};
        let scratch = Scratch::new("openduringput");
        let keys = scratch.keys();
        let store = Arc::new(FileStore::open(&keys, KeyfileCodec).unwrap());
        let n = name("vault.pins");
        store.put(&n, b"old").unwrap();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        store.dir.hooks.at("before_rename", move || {
            entered_tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        });
        let writer = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.put(&n, b"new"))
        };
        entered_rx.recv().unwrap(); // the temporary file is written and verified, and not renamed
        let temporaries = || files(&keys).into_iter().filter(|f| f.ends_with(".tmp")).count();
        assert_eq!(temporaries(), 1);
        drop(FileStore::open(&keys, KeyfileCodec).unwrap()); // another process opens the store
        assert_eq!(temporaries(), 1, "opening the store deleted the temporary file of a put that was in progress");
        go_tx.send(()).unwrap();
        writer.join().unwrap().expect("the put succeeds");
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"new");
        assert_eq!(temporaries(), 0);
    }
    /// KM16: the owner rule is applied where files and folders are judged: at open, before every operation, for a key file, for the lock file and for every folder
    /// above. Mutating any of the five to skip the owner (a probe that returned nothing, `None` where `Some(uid)` belongs) left the suite green, because the
    /// rule itself was tested only as a pure function. This process pretends to be the next user, so that everything it made belongs to "another user"; the
    /// same thing with a real second account needs root and is the `#[ignore]`d test in `tests/keystore.rs`.
    #[cfg(unix)]
    #[test]
    fn what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above() {
        let scratch = Scratch::new("owner");
        let keys = scratch.keys();
        let mut store = FileStore::open(&keys, KeyfileCodec).unwrap();
        store.put(&name("a.one"), b"1").unwrap();
        let me = store.dir.uid;
        let another = |error: KeyError| matches!(&error, KeyError::Permissions(why) if why.contains("another user"));
        assert!(another(KeyDir::open_as(&keys, me + 1).err().expect("open refuses a folder another user owns")), "open");
        // at open, on its own: a folder directly under /tmp (root's and sticky, so nothing above it is what refuses: the folders above are judged too, and
        // here they are the scratch folder's, which would refuse first)
        if std::env::temp_dir() == Path::new("/tmp") {
            let direct = std::env::temp_dir().join(format!("oaiy-keystore-owner-{}", std::process::id()));
            drop(FileStore::open(&direct, KeyfileCodec).unwrap());
            let refused = KeyDir::open_as(&direct, me + 1).err();
            fs::remove_dir_all(&direct).unwrap();
            assert!(
                matches!(&refused, Some(KeyError::Permissions(why)) if why.contains("another user") && !why.contains("above")),
                "open did not compare the owner of the folder itself: {refused:?}"
            );
        }
        store.dir.uid = me + 1;
        assert!(another(store.dir.verify().unwrap_err()), "the folder, before each operation");
        assert!(another(store.get(&name("a.one")).unwrap_err()), "get");
        assert!(another(store.dir.read("a.one.kf").expect_err("refused")), "a key file, judged on its own");
        assert!(another(store.dir.lock(false).err().expect("refused")), "the lock file");
        if me != 0 {
            // root owns the folders above in most places, and root is allowed; everyone else's are judged
            assert!(another(store.dir.check_ancestors().unwrap_err()), "the folders above");
        }
    }

    /// The lock is what keeps an old file that is still being written: a write that takes longer than a minute (a suspended laptop, a stalled disk) is older than
    /// the age rule, and only the lock tells the other process that the file is alive.
    #[test]
    fn a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock() {
        use std::sync::{mpsc, Arc, Mutex};
        let scratch = Scratch::new("slowwrite");
        let keys = scratch.keys();
        let store = Arc::new(FileStore::open(&keys, KeyfileCodec).unwrap());
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        store.dir.hooks.at("staging", move || {
            entered_tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        });
        let n = name("archive.writer");
        let writer = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.put(&n, b"a slow write"))
        };
        entered_rx.recv().unwrap(); // the temporary file exists and its writer holds its lock
        let temporary: Vec<String> = files(&keys).into_iter().filter(|f| f.ends_with(".tmp")).collect();
        assert_eq!(temporary.len(), 1);
        age(&keys.join(&temporary[0]), 600);
        drop(FileStore::open(&keys, KeyfileCodec).unwrap());
        assert_eq!(files(&keys).into_iter().filter(|f| f.ends_with(".tmp")).count(), 1, "an old file whose writer holds its lock was removed");
        go_tx.send(()).unwrap();
        writer.join().unwrap().expect("the slow put succeeds");
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"a slow write");
    }
    /// M-3, the reviewer's attack: between the moment the store has judged its folder and the moment it opens a file, someone who can rename folders puts a
    /// folder of their own at the path. A store that looks the path up again reads the attacker's values as its own (the reviewer's victim read 3,388 of
    /// them, and was told `None` 58,442 times). This store opened its folder once and opens every file relative to it: the read is of the real folder, the
    /// write goes to the real folder, nothing is ever written into the attacker's, and the very next operation says that the path leads elsewhere now.
    #[cfg(unix)]
    #[test]
    fn a_folder_swapped_in_after_the_check_is_never_read_or_written_in_place_of_the_real_one() {
        use std::sync::atomic::AtomicBool;
        let scratch = Scratch::new("swap");
        let keys = scratch.keys();
        let store = FileStore::open(&keys, KeyfileCodec).unwrap();
        let n = name("archive.writer");
        store.put(&n, b"the victim's value").unwrap();
        // the attacker's folder: a valid blob under the same name, made by a store of its own
        let attacker = scratch.0.join("attacker");
        FileStore::open(&attacker, KeyfileCodec).unwrap().put(&n, b"the attacker's value").unwrap();
        let attacker_before = fs::read(attacker.join("archive.writer.kf")).unwrap();
        let moved = scratch.0.join("victim-moved");
        let armed = std::sync::Arc::new(AtomicBool::new(true));
        let swap = {
            let (keys, attacker, moved, armed) = (keys.clone(), attacker.clone(), moved.clone(), armed.clone());
            move || {
                if armed.swap(false, Ordering::SeqCst) {
                    fs::rename(&keys, &moved).unwrap();
                    fs::rename(&attacker, &keys).unwrap();
                }
            }
        };

        store.dir.hooks.at("before_open", swap.clone());
        let read = store.get(&n).unwrap().expect("the victim's own value is still found");
        assert_eq!(&**read, b"the victim's value", "the read went through the descriptor of the real folder, not through the path");
        // the next operation notices that the path leads to another folder now, and says so: neither `None` nor the attacker's value
        assert!(matches!(store.get(&n), Err(KeyError::Io { op: "inspect the keys directory", .. })));
        assert!(matches!(store.put(&n, b"x"), Err(KeyError::Io { .. })));
        assert!(matches!(store.list(""), Err(KeyError::Io { .. })));
        assert!(matches!(store.delete(&n), Err(KeyError::Io { .. })));
        assert_eq!(fs::read(keys.join("archive.writer.kf")).unwrap(), attacker_before, "nothing was written into the attacker's folder");
        assert_eq!(files(&keys), vec!["archive.writer.kf".to_string()]);

        // the same swap in the middle of a put: the temporary file, its read-back and the rename are all in the real folder
        fs::rename(&keys, &attacker).unwrap();
        fs::rename(&moved, &keys).unwrap();
        armed.store(true, Ordering::SeqCst);
        store.dir.hooks.clear();
        store.dir.hooks.at("before_open", swap);
        store.put(&n, b"the new victim value").unwrap();
        assert_eq!(fs::read(keys.join("archive.writer.kf")).unwrap(), attacker_before, "the attacker's folder is at the path, and was not touched");
        assert_eq!(files(&keys), vec!["archive.writer.kf".to_string()], "no temporary file of the put is in the attacker's folder");
        let reopened = FileStore::open(&moved, KeyfileCodec).unwrap();
        assert_eq!(&**reopened.get(&n).unwrap().unwrap(), b"the new victim value", "the put landed in the real folder");
    }

    /// H-1, made deterministic. Another process is in the middle of replacing a name: it holds the lock and, for a moment, the name is not there (on Windows a
    /// `MoveFileEx` over an existing file does leave such a moment; the reviewer's reader was told `None`, "never stored", for a key that existed, and a caller
    /// that believes `None` mints a new identity). A reader must wait for the writer and then see the new value, never `None`.
    #[test]
    fn a_reader_waits_for_a_writer_in_the_middle_of_a_replace_and_is_never_told_that_nothing_is_stored() {
        use std::sync::Arc;
        use std::time::Duration;
        let scratch = Scratch::new("replace");
        let keys = scratch.keys();
        let store = Arc::new(FileStore::open(&keys, KeyfileCodec).unwrap());
        let n = name("archive.writer");
        store.put(&n, b"the old value").unwrap();

        let writer = KeyDir::open(&keys).unwrap(); // another process's view of the folder
        let lock = writer.lock(true).unwrap();
        writer.remove("archive.writer.kf").unwrap(); // the moment in which the name does not exist

        let reader = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.get(&n))
        };
        std::thread::sleep(Duration::from_millis(400));
        assert!(!reader.is_finished(), "the reader did not wait: it looked at the name while a writer was replacing it, and would have said `None`");

        // the writer finishes: the new blob is in place, and only then does it let go
        let blob = KeyfileCodec.seal(&n, b"the new value").unwrap();
        writer.create_new("archive.writer.kf").unwrap().write_all(&blob).unwrap();
        drop(lock);
        let got = reader.join().unwrap().unwrap().expect("the key exists: it is not `None`");
        assert_eq!(&**got, b"the new value");
    }

    /// M-1 at the level of the store: a path that leads to another tree now (here pointed there by the test hook, the way a re-pointed junction above the folder would: the open
    /// refuses a path with a junction on it, so that cannot be done to a store that is open) makes **every** operation an error. Not `None` for a key that exists in the held
    /// folder, not the other tree's value, not a write into either, and the same store works again when the path leads back.
    #[cfg(windows)]
    #[test]
    fn a_store_whose_path_leads_elsewhere_answers_with_errors_never_none_and_never_the_other_trees_value() {
        let scratch = Scratch::new("elsewhere");
        let (keys, decoy) = (scratch.0.join("a").join("keys"), scratch.0.join("b").join("keys"));
        let mut store = FileStore::open(&keys, KeyfileCodec).unwrap();
        let n = name("vault.pins");
        store.put(&n, b"the real value").unwrap();
        {
            let other = FileStore::open(&decoy, KeyfileCodec).unwrap();
            other.put(&n, b"the other tree's value").unwrap();
        }
        let before = (files(&keys), files(&decoy));

        store.dir.point_the_path_at(&decoy);
        let is_the_path = |error: KeyError| matches!(&error, KeyError::Io { op: "inspect the keys directory", source } if source.to_string().contains("replaced or moved"));
        assert!(is_the_path(store.get(&n).unwrap_err()), "get: not None and not the other tree's value");
        assert!(is_the_path(store.get(&name("never.stored")).unwrap_err()), "get of a name nobody has: still an error, not None");
        assert!(is_the_path(store.put(&n, b"a new value").unwrap_err()), "put");
        assert!(is_the_path(store.delete(&n).unwrap_err()), "delete");
        assert!(is_the_path(store.list("").unwrap_err()), "list");
        assert_eq!((files(&keys), files(&decoy)), before, "nothing was written, removed or made in either tree");

        store.dir.point_the_path_at(&keys);
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"the real value", "the same store, once the path leads back");
        assert!(decoy.join("vault.pins.kf").exists(), "and the other tree still has its own file");
    }

    /// H-1 through the real `put`, at the point that matters, on every platform (the review's two-process test is a weak guard: the lock-less store gave one false `None` in about
    /// 2.73 million reads, and the locked store none in 2.0 million, which proves little; this is the guard). The writer is paused **inside its locked rename step**: it holds the
    /// exclusive lock, the temporary file is verified, and the old file has just been taken away, which is the moment that a rename over an existing file leaves on Windows
    /// (`MoveFileEx` with `REPLACE_EXISTING`: the destination name is briefly absent; on Unix `rename` does not leave it, and the test puts it there by hand). A reader that starts now
    /// must **wait**, and must then find the new value: it must never return, and never return `None`, while the writer is between the two states. Without the exclusive lock
    /// around the rename the reader runs at once and says `None`; without the shared lock around the read it does the same.
    #[test]
    fn a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none() {
        use std::sync::{mpsc, Arc, Mutex};
        use std::time::Duration;
        let scratch = Scratch::new("midrename");
        let keys = scratch.keys();
        let store = Arc::new(FileStore::open(&keys, KeyfileCodec).unwrap());
        let n = name("vault.pins");
        store.put(&n, b"old").unwrap();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        let old_file = keys.join("vault.pins.kf");
        store.dir.hooks.at("rename_locked", move || {
            fs::remove_file(&old_file).unwrap(); // the destination name, briefly absent
            entered_tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        });
        let writer = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.put(&n, b"new"))
        };
        entered_rx
            .recv_timeout(Duration::from_secs(20))
            .expect("the writer never reached its locked rename step: the put does not rename under the lock");
        let reader = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.get(&n))
        };
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            !reader.is_finished(),
            "a reader that started in the middle of a rename returned at once, and would have said `None` for a key that exists"
        );
        go_tx.send(()).unwrap();
        writer.join().unwrap().expect("the put succeeds");
        let got = reader.join().unwrap().unwrap().expect("the key exists: the reader must not be told that it was never stored");
        assert_eq!(&**got, b"new");
    }

    /// The other half: readers share the lock, so one reader never waits for another, and a holder that never lets go is an error after the wait (naming the
    /// lock), not a caller that never returns. Nothing changes when the lock cannot be had.
    #[test]
    fn readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait() {
        use std::time::{Duration, Instant};
        let scratch = Scratch::new("lockwait");
        let keys = scratch.keys();
        let mut store = FileStore::open(&keys, KeyfileCodec).unwrap();
        store.dir.lock_wait = Duration::from_millis(300);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        let other = KeyDir::open(&keys).unwrap();

        {
            let _shared = other.lock(false).unwrap();
            assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first", "a reader does not wait for another reader");
            assert_eq!(store.list("").unwrap().len(), 1);
        }
        let before = files(&keys);
        let _exclusive = other.lock(true).unwrap();
        let started = Instant::now();
        let is_timeout =
            |e: KeyError| matches!(e, KeyError::Io { op: "lock the keys directory", source } if source.kind() == std::io::ErrorKind::TimedOut);
        assert!(is_timeout(store.get(&n).unwrap_err()), "get");
        assert!(is_timeout(store.put(&n, b"second").unwrap_err()), "put");
        assert!(is_timeout(store.delete(&n).unwrap_err()), "delete");
        assert!(is_timeout(store.list("").unwrap_err()), "list");
        assert!(started.elapsed() >= Duration::from_millis(1000), "each of the four waited the full 300 ms");
        assert!(started.elapsed() < Duration::from_secs(8), "and none waited much longer");
        assert_eq!(files(&keys), before, "a put that could not take the lock left no temporary file");
        drop(_exclusive);
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first", "and the old value is still there");
    }

    /// The exclusive lock is held for the rename only: a put that is between its write and its rename does not stop a reader, which sees the old value.
    #[test]
    fn a_put_does_not_hold_the_lock_while_it_writes() {
        use std::sync::{mpsc, Arc, Mutex};
        let scratch = Scratch::new("pausedput");
        let store = Arc::new(FileStore::open(&scratch.keys(), KeyfileCodec).unwrap());
        let n = name("vault.pins");
        store.put(&n, b"old").unwrap();
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = Mutex::new(go_rx);
        store.dir.hooks.at("before_rename", move || {
            entered_tx.send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        });
        let writer = {
            let (store, n) = (Arc::clone(&store), n.clone());
            std::thread::spawn(move || store.put(&n, b"new"))
        };
        entered_rx.recv().unwrap(); // the put has written and verified its temporary file, and has not renamed it
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"old", "a reader is not held up by a put that is still writing");
        go_tx.send(()).unwrap();
        writer.join().unwrap().unwrap();
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"new");
    }

    /// A put flushes the folder after the rename, and a delete after the removal: the new name, and the absence of the old one, survive a power cut (L-9: on
    /// Windows the flush was missing altogether).
    #[test]
    fn a_put_and_a_delete_flush_the_folder() {
        let scratch = Scratch::new("flush");
        let store = faulty(&scratch);
        let opened = store.dir.hooks.syncs.load(Ordering::SeqCst);
        assert_eq!(opened, 1, "opening a new folder flushes it once, after the marker file is renamed into place");
        let syncs = || store.dir.hooks.syncs.load(Ordering::SeqCst) - opened;
        store.put(&name("a.one"), b"1").unwrap();
        assert_eq!(syncs(), 1, "put");
        store.delete(&name("a.one")).unwrap();
        assert_eq!(syncs(), 2, "delete");
        assert!(store.put(&name("a.two"), b"").is_err());
        assert_eq!(syncs(), 2, "a refused put flushes nothing");
    }

    /// The raw operating system error of "this file system cannot flush a folder" (an SMB share: `ERROR_INVALID_FUNCTION`; Unix: `EINVAL`), and of "the flush was tried and failed" (a
    /// failing disk: `ERROR_IO_DEVICE`; `EIO`).
    const CANNOT_FLUSH: i32 = if cfg!(windows) { 1 } else { 22 };
    const FLUSH_FAILED: i32 = if cfg!(windows) { 1117 } else { 5 };

    /// Third review, M-A. On an SMB share `FlushFileBuffers` on the folder handle is "incorrect function" (os error 1), and every `put` and `delete` returned `Err` **after** the rename had
    /// committed: the contract says that a failed put leaves the previous value, and here the new one was in place, so a caller that followed the contract after a failed rotation kept the
    /// old key in memory while the new one was on disk and the old one was gone. A flush that fails after the change is not an error of the change: it is `Ok(Unconfirmed)`, whichever
    /// way it fails (a file system that cannot flush a folder, and a flush that was tried and failed), and the value is stored, and every reader sees it. Without the exception the
    /// first put here is an `Err` with the file on disk.
    #[test]
    fn a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error() {
        for code in [CANNOT_FLUSH, FLUSH_FAILED] {
            let scratch = Scratch::new("flushfails");
            let store = faulty(&scratch);
            let n = name("vault.pins");
            assert_eq!(store.put(&n, b"key version 0").unwrap(), Durability::Confirmed, "a flush that works is `Confirmed`");
            store.dir.hooks.fail_sync.store(code, Ordering::SeqCst);

            assert_eq!(store.put(&n, b"key version 1").unwrap(), Durability::Unconfirmed, "os error {code}: the put is made");
            assert_eq!(&**store.get(&n).unwrap().unwrap(), b"key version 1", "os error {code}: and it is what a reader sees");
            assert_eq!(files(&scratch.keys()), vec!["vault.pins.kf".to_string()], "os error {code}: no temporary file");
            assert_eq!(store.put(&n, b"key version 2 (the rotation)").unwrap(), Durability::Unconfirmed, "os error {code}: a rotation");
            assert_eq!(&**store.get(&n).unwrap().unwrap(), b"key version 2 (the rotation)");
            assert_eq!(store.delete(&n).unwrap(), Durability::Unconfirmed, "os error {code}: the delete is made");
            assert!(store.get(&n).unwrap().is_none(), "os error {code}: and the name is gone");
            assert!(files(&scratch.keys()).is_empty());
            // a delete of what is not there is as it ever was
            assert_eq!(store.delete(&n).unwrap(), Durability::Unconfirmed);

            // and the folder flushes again: `Confirmed`
            store.dir.hooks.fail_sync.store(0, Ordering::SeqCst);
            assert_eq!(store.put(&n, b"key version 3").unwrap(), Durability::Confirmed);
            assert_eq!(store.delete(&n).unwrap(), Durability::Confirmed);
        }
    }

    /// What the flush of the folder says, told apart at the level of the folder: a file system that cannot flush a folder is `Unconfirmed`, a flush that was tried and failed is an
    /// error (the store turns that into `Unconfirmed` for a change that is made; nothing else may swallow it: a mutant that made every failure a success was alive).
    #[test]
    fn the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed() {
        let scratch = Scratch::new("flushkinds");
        let store = faulty(&scratch);
        assert_eq!(store.dir.sync().unwrap(), Durability::Confirmed);
        store.dir.hooks.fail_sync.store(CANNOT_FLUSH, Ordering::SeqCst);
        assert_eq!(store.dir.sync().unwrap(), Durability::Unconfirmed, "a file system that says it cannot");
        store.dir.hooks.fail_sync.store(FLUSH_FAILED, Ordering::SeqCst);
        let failed = store.dir.sync().unwrap_err();
        assert!(
            matches!(&failed, KeyError::Io { op: "flush the keys directory", source } if source.raw_os_error() == Some(FLUSH_FAILED)),
            "{failed:?}"
        );
    }

    /// Third review, M-A: the first open of a new folder over SMB failed with "flush the keys directory" after it had made the lock file and the marker. The marker is in place, and
    /// whether the flush could make it survive a power cut is no reason for the open to fail (a marker that a power cut takes back is made again by the next open): the store opens, the
    /// folder is claimed, and it works.
    #[test]
    fn a_first_open_whose_flush_fails_still_opens_and_claims_the_folder() {
        for code in [CANNOT_FLUSH, FLUSH_FAILED] {
            let scratch = Scratch::new("openflush");
            let store = FileStore::open_prepared(&scratch.keys(), Faulty { inner: KeyfileCodec, fault: AtomicU8::new(FAULT_NONE) }, |dir| {
                dir.hooks.fail_sync.store(code, Ordering::SeqCst);
            })
            .unwrap_or_else(|e| panic!("os error {code}: the open failed: {e:?}"));
            assert_eq!(store.dir.hooks.syncs.load(Ordering::SeqCst), 1, "the marker was flushed (and the flush failed)");
            assert_eq!(fs::read(scratch.keys().join(MARKER_FILE)).unwrap(), b"faulty\n", "os error {code}: the folder is claimed");
            store.dir.hooks.fail_sync.store(0, Ordering::SeqCst);
            let n = name("vault.pins");
            assert_eq!(store.put(&n, b"1").unwrap(), Durability::Confirmed);
            drop(store);
            assert_eq!(&**faulty(&scratch).get(&n).unwrap().unwrap(), b"1", "and the next open finds it as it was");
        }
    }

    /// Third review, L-5. A store that is open and finds its folder emptied (on Windows `remove_dir_all` deletes everything in a folder that is held and then fails to remove the
    /// folder, which is the same thing) answered `Ok(None)`, "never stored", for every key that had been there, and a caller that believes `None` mints a new identity. The marker is
    /// written when the folder is first opened and never changes: a store that does not find it is an error, for every operation and for every name, whether or not the name was there.
    /// A marker that has been replaced by another provider's is as much an error. Without the look for the marker the first `get` is `Ok(None)`.
    #[test]
    fn a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none() {
        let scratch = Scratch::new("emptied");
        let keys = scratch.keys();
        let store = faulty(&scratch);
        let n = name("archive.writer");
        store.put(&n, b"the identity").unwrap();
        for entry in fs::read_dir(&keys).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap(); // the key file, the marker and the lock file
        }
        let emptied = |error: KeyError| matches!(&error, KeyError::Corrupt(why) if why.contains("marker"));
        assert!(emptied(store.get(&n).unwrap_err()), "get of a name that was there");
        assert!(emptied(store.get(&name("never.stored")).unwrap_err()), "get of a name that never was");
        assert!(emptied(store.put(&n, b"a new identity").unwrap_err()), "put");
        assert!(emptied(store.delete(&n).unwrap_err()), "delete");
        assert!(emptied(store.list("").unwrap_err()), "list");
        assert!(
            fs::read_dir(&keys).unwrap().next().is_none(),
            "and nothing was made in the emptied folder: no marker, no lock file, no temporary file"
        );

        // a marker that names another provider, put in its place by whoever can write there: refused, whatever the name
        let write_marker = |text: &[u8]| {
            fs::write(keys.join(MARKER_FILE), text).unwrap();
            #[cfg(unix)]
            fs::set_permissions(keys.join(MARKER_FILE), std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
        };
        write_marker(b"keyfile\n");
        let mismatch =
            |error: KeyError| matches!(&error, KeyError::ProviderMismatch { stored, requested } if stored == "keyfile" && *requested == "faulty");
        assert!(mismatch(store.get(&n).unwrap_err()), "get with another provider's marker");
        assert!(mismatch(store.put(&n, b"x").unwrap_err()), "put with another provider's marker");
        // the folder is empty, and a store opened again is a new store of an empty folder: `None` is the truth there, and the caller learns it only now
        write_marker(b"faulty\n");
        drop(store);
        assert!(faulty(&scratch).get(&n).unwrap().is_none());
    }

    /// Third review, L-5: the lock file is made when the folder is opened, and an operation only opens it. One that has gone is an error, not a lock that is made again beside the
    /// holders of the old one (on Unix a lock belongs to the file that was opened, so two lock files would not exclude each other). Without it an operation after the file is deleted
    /// makes a new one and succeeds.
    #[test]
    fn a_lock_file_that_has_gone_is_an_error_and_is_not_made_again() {
        let scratch = Scratch::new("lostlock");
        let store = faulty(&scratch);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        fs::remove_file(scratch.keys().join(".lock")).unwrap();
        let gone = |error: KeyError| matches!(&error, KeyError::Io { op: "open the lock file", .. });
        assert!(gone(store.get(&n).unwrap_err()), "get");
        assert!(gone(store.put(&n, b"second").unwrap_err()), "put");
        assert!(gone(store.delete(&n).unwrap_err()), "delete");
        assert!(gone(store.list("").unwrap_err()), "list");
        assert!(!scratch.keys().join(".lock").exists(), "and no lock file was made");
        assert_eq!(files(&scratch.keys()), vec!["relay.token.kf".to_string()], "nothing else was changed");
        // a new open makes it again, and the same key is there
        assert_eq!(&**faulty(&scratch).get(&n).unwrap().unwrap(), b"first");
    }

    /// The lock file has one name, like a key file (the third review: Unix refused a second name and Windows did not). A hard link to it is another way to reach the file from a place
    /// this store does not look after.
    #[test]
    fn a_lock_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone() {
        let scratch = Scratch::new("lockhardlink");
        let store = faulty(&scratch);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        let extra = scratch.0.join("another-name-for-the-lock");
        fs::hard_link(scratch.keys().join(".lock"), &extra).unwrap();
        let refused = |error: KeyError| matches!(&error, KeyError::Permissions(why) if why.contains("names (hard links)"));
        assert!(refused(store.get(&n).unwrap_err()), "get");
        assert!(refused(store.put(&n, b"second").unwrap_err()), "put");
        assert!(refused(store.dir.lock(true).err().expect("refused")), "the lock itself");
        fs::remove_file(&extra).unwrap();
        assert_eq!(&**store.get(&n).unwrap().unwrap(), b"first", "the same lock file with one name again");
    }

    /// The lock file is opened as itself, never through a link (Unix: `O_NOFOLLOW`): a symbolic link in its place is refused, and what it points at is not locked, written or made.
    /// Without `O_NOFOLLOW` the target is a regular file of ours with a good mode and the lock is taken on it.
    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_in_place_of_the_lock_file_is_refused() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let scratch = Scratch::new("locklink");
        let store = faulty(&scratch);
        let n = name("relay.token");
        store.put(&n, b"first").unwrap();
        let real = scratch.keys().join("a-file-of-ours.txt");
        fs::write(&real, b"not a lock").unwrap();
        fs::set_permissions(&real, fs::Permissions::from_mode(0o600)).unwrap();
        fs::remove_file(scratch.keys().join(".lock")).unwrap();
        symlink(&real, scratch.keys().join(".lock")).unwrap();
        let error = store.get(&n).unwrap_err();
        assert!(matches!(&error, KeyError::Permissions(why) if why.contains("symbolic link")), "{error:?}");
        assert_eq!(fs::read(&real).unwrap(), b"not a lock");
    }

    /// A name that another provider's extension has, and that is a link to nothing, is still an entry: a secret that this provider cannot read, which `get`, `put` and `delete` refuse.
    /// The look for it must not follow the link (a dangling one would read as "not there", and `get` would answer `None` for a name that has a value of another provider, `put` would
    /// leave two live values): a symbolic link on Unix, a junction whose target is gone on Windows.
    #[test]
    fn a_link_to_nothing_under_another_providers_extension_is_still_a_value_of_that_provider() {
        let scratch = Scratch::new("dangling");
        let store = faulty(&scratch);
        let link = scratch.keys().join("other.name.ks");
        #[cfg(unix)]
        std::os::unix::fs::symlink(scratch.0.join("nothing-here"), &link).unwrap();
        #[cfg(windows)]
        {
            let target = scratch.0.join("gone-target");
            fs::create_dir(&target).unwrap();
            let made = std::process::Command::new("cmd").args(["/C", "mklink", "/J"]).arg(&link).arg(&target).output().unwrap();
            assert!(made.status.success(), "mklink /J failed: {}", String::from_utf8_lossy(&made.stdout));
            fs::remove_dir(&target).unwrap();
        }
        let other = name("other.name");
        let mismatch = |error: KeyError| matches!(&error, KeyError::ProviderMismatch { stored, .. } if stored == "windows-dpapi-file");
        assert!(mismatch(store.get(&other).unwrap_err()), "get: not `None`");
        assert!(mismatch(store.put(&other, b"a second value").unwrap_err()), "put: not a second value beside it");
        assert!(mismatch(store.delete(&other).unwrap_err()), "delete");
        assert!(store.get(&name("some.other")).unwrap().is_none(), "the names that have no such entry are unaffected");
    }

    /// Debris is a file of exactly the store's shape: `.<name>.<16 hex>.tmp`. A longer random part is not (a file someone else put there), however old.
    #[test]
    fn a_file_with_a_longer_random_part_than_the_stores_is_not_its_temporary_file() {
        assert!(is_temporary_name(".keep.me.0123456789abcdef.tmp"));
        for other in [
            ".keep.me.0123456789abcdef0.tmp",
            ".keep.me.0123456789abcde.tmp",
            ".keep.me.0123456789ABCDEF.tmp",
            ".keep.me.0123456789abcdeg.tmp",
            ".keep.me..tmp",
            ".0123456789abcdef.tmp",
        ] {
            assert!(!is_temporary_name(other), "{other}");
        }
    }

    /// `list` answers in order, whatever order the file system keeps (NTFS keeps names sorted; ext4 does not).
    #[test]
    fn list_is_sorted() {
        let scratch = Scratch::new("sorted");
        let store = faulty(&scratch);
        let names: Vec<String> = (0..24).rev().map(|i| format!("n{:02}.x", (i * 7) % 24)).collect();
        for n in &names {
            store.put(&name(n), b"1").unwrap();
        }
        let listed: Vec<String> = store.list("").unwrap().iter().map(|n| n.as_str().to_string()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(listed, sorted);
    }
}
