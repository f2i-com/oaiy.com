//! The trait every provider implements, and the one file-per-secret store that both providers share.
//!
//! Every rule of design 4.5.1 that is not about how a blob is made lives here, once: `Ok(None)` means the name was never stored and nothing else
//! does; `put` writes a temporary file (created new, never overwriting, with only what the provider makes of the value in it), flushes it, reads
//! it back through the provider and compares, and only then renames it over the old value, so a failure at any step leaves the previous value
//! and no file of the failed attempt; `delete` is idempotent; `list` shows valid names only; a keys directory that has disappeared is an error, not an
//! empty store. Every file is reached through the folder the store holds open ([`crate::keydir`]), never through a path looked up again.

use std::io::Write;
use std::path::Path;

use oaiy_crypto::kdf::hex_lower;
use oaiy_crypto::zeroize::{ct_eq, Secret};
use zeroize::Zeroizing;

use crate::codec::{Codec, ProviderInfo};
use crate::error::KeyError;
use crate::keydir::KeyDir;
use crate::name::Name;

/// The most a value may be: 64 KiB.
pub const MAX_VALUE_LEN: usize = 64 * 1024;

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
        let store = FileStore { dir: KeyDir::open(dir)?, codec };
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
        self.dir.sync()
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
    /// with the keyfile provider it holds a value in the clear, so it is removed the next time the store is opened.
    fn remove_stale_temporaries(&self) {
        let Ok(entries) = self.dir.list() else { return };
        for entry in entries {
            if entry.is_file && entry.name.starts_with('.') && entry.name.ends_with(".tmp") {
                let _ = self.dir.remove(&entry.name);
            }
        }
    }

    fn temporary_name(&self, name: &Name) -> Result<String, KeyError> {
        let random: Secret<8> = Secret::random().map_err(|_| KeyError::io("random", std::io::Error::other("the random generator failed")))?;
        Ok(format!(".{}.{}.tmp", name.as_str(), hex_lower(random.expose())))
    }

    /// Makes `tmp` (which must not exist) with `bytes` in it and flushes it.
    fn stage(&self, tmp: &str, bytes: &[u8]) -> Result<(), KeyError> {
        let mut file = self.dir.create_new(tmp)?;
        file.write_all(bytes).map_err(|e| KeyError::io("write the temporary file", e))?;
        file.sync_all().map_err(|e| KeyError::io("flush the temporary file", e))
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
        // the lock, shared, is what makes `None` mean "not there": a writer that is replacing the name holds it exclusively (review H-1)
        let blob = {
            let _lock = self.dir.lock(false)?;
            self.refuse_a_value_of_another_provider(name)?;
            self.dir.read(&self.file_name(name))?
        };
        let Some(blob) = blob else { return Ok(None) };
        self.codec.open(name, &blob).map(Some)
    }

    fn put(&self, name: &Name, value: &[u8]) -> Result<(), KeyError> {
        if value.is_empty() {
            return Err(KeyError::InvalidValue("empty"));
        }
        if value.len() > MAX_VALUE_LEN {
            return Err(KeyError::InvalidValue("larger than 64 KiB"));
        }
        self.dir.verify()?;
        self.refuse_a_value_of_another_provider(name)?;
        let blob = self.codec.seal(name, value)?;
        let tmp = self.temporary_name(name)?;
        if let Err(error) = self.write_and_verify(&tmp, name, value, &blob) {
            let _ = self.dir.remove(&tmp);
            return Err(error);
        }
        self.dir.fire("before_rename");
        // the exclusive lock is held for the rename only, not for the write and the read-back: readers wait for one rename, never for a disk
        let renamed = self.dir.lock(true).and_then(|_lock| self.dir.rename(&tmp, &self.file_name(name)));
        if let Err(error) = renamed {
            let _ = self.dir.remove(&tmp);
            return Err(error);
        }
        self.dir.sync()
    }

    fn delete(&self, name: &Name) -> Result<(), KeyError> {
        self.dir.verify()?;
        {
            let _lock = self.dir.lock(true)?;
            self.refuse_a_value_of_another_provider(name)?;
            self.dir.remove(&self.file_name(name))?;
        }
        self.dir.sync()
    }

    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError> {
        self.dir.verify()?;
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

    /// A crash between the write and the rename leaves a temporary file. With the keyfile provider it holds a value in the clear, so opening the store removes it.
    #[test]
    fn opening_the_store_removes_the_debris_of_an_interrupted_put() {
        let scratch = Scratch::new("debris");
        let dir = scratch.keys();
        {
            let store = FileStore::open(&dir, KeyfileCodec).unwrap();
            store.put(&name("keep.me"), b"kept").unwrap();
        }
        fs::write(dir.join(".keep.me.0123456789abcdef.tmp"), b"a value in the clear").unwrap();
        fs::write(dir.join(".other.fedcba9876543210.tmp"), b"another").unwrap();
        fs::write(dir.join("notes.txt"), b"not ours").unwrap();
        let store = FileStore::open(&dir, KeyfileCodec).unwrap();
        assert_eq!(files(&dir), vec!["keep.me.kf".to_string(), "notes.txt".to_string()]);
        assert_eq!(&**store.get(&name("keep.me")).unwrap().unwrap(), b"kept");
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
}
