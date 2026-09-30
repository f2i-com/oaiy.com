//! The keys folder as something the store holds open, not a path it looks up again at every step.
//!
//! A path is checked, and then used, and in between anyone who can rename folders can put another in its place: the reviewer swapped a `keys` folder
//! for an attacker's 1.1 million times in 40 seconds against a check-then-open store, and the victim read 3,388 of the attacker's values as its own
//! and was told `None`, "never stored", 58,442 times. So every file of the store is reached through the folder the store opened:
//!
//! - **Unix:** a directory descriptor, opened once with `O_NOFOLLOW`, and every file opened, created, renamed and removed **relative to it** (`openat`,
//!   `renameat`, `unlinkat`). A file is opened with `O_NOFOLLOW | O_NONBLOCK` and judged by `fstat` **on the descriptor** that will be read: a regular file, owned
//!   by this user, with no group or other permission; a FIFO (which would block a read for ever), a symbolic link, a device, another user's file are
//!   refused. The folder is judged again before each operation (owner, mode) and compared with the path (same device and inode): a folder that was
//!   replaced, moved or removed is an error, never "nothing stored". At open the directories **above** are judged too, by walking `..` from the descriptor
//!   ([`crate::perm::check_ancestor`]): not owned by another user, not writable by others unless sticky.
//! - **Windows:** a handle to the folder held open **without `FILE_SHARE_DELETE`**, so that nobody can rename or remove the folder, or a **real** folder above it,
//!   while the store is open. That does not hold for a junction or a symbolic link above the folder: it is an entry that whoever can change its parent can delete and
//!   make point elsewhere, with the store open (review M-1: the store answered from another tree, and served an older copy as the current value). So the path is
//!   **refused if any folder on it is a reparse point** (walked before the folder is opened and again once it is held), every operation works in the held folder by
//!   its **real path** (the handle's final path), and the path is opened afresh before each operation and compared with the handle (volume and file index). Every
//!   file is opened as itself and refused if it is a reparse point, not a regular file, or has more than one name. The handle is also what the store flushes after a
//!   rename, so that the new name survives a power cut. The header of `keydir_windows.rs` says what is left: the attacker who can write in the keys folder itself.
//!
//! The two platform modules implement the same small set of operations; the store above them does not know which it is on.

use std::fs::{File, TryLockError};
use std::io::{self, Read};
use std::thread;
use std::time::{Duration, Instant};

use zeroize::Zeroizing;

use crate::error::KeyError;
use crate::store::MAX_VALUE_LEN;

#[cfg(unix)]
#[path = "keydir_unix.rs"]
mod sys;
#[cfg(windows)]
#[path = "keydir_windows.rs"]
mod sys;

pub(crate) use sys::KeyDir;

/// A blob is never larger than a value plus a provider's framing; a file larger than this is not one of ours and is not read into memory.
pub(crate) const MAX_BLOB_LEN: usize = MAX_VALUE_LEN + 4096;

/// The file whose advisory lock orders the operations of every process that uses the folder. It holds nothing; it is never read or written.
pub(crate) const LOCK_FILE: &str = ".lock";

/// How long an operation waits for another process's lock before it gives up with an error: a lock is held for the length of one rename, so a longer wait
/// means a process that is stuck, and an error is better than a caller that never returns. Ten seconds is generous on purpose: one stall under Windows Defender in the
/// reviewer's runs reached 8.5 seconds, and an operation that gives up is an error (it fails closed, nothing is changed, the caller may try again), so a slow machine
/// costs a caller a retry and never a wrong answer.
pub(crate) const LOCK_WAIT: Duration = Duration::from_secs(10);

/// An advisory lock on the folder, held until it is dropped. **Why there is one (review H-1):** replacing a file is not atomic for a reader on Windows
/// (`MoveFileEx` with `REPLACE_EXISTING` leaves a moment in which the name does not exist), and `Ok(None)` means "never stored", so a reader that happened
/// to open the name in that moment reported that a key that exists was never stored, and a caller that trusts `None` mints a new identity. Every operation that
/// can change what a name refers to (the rename of a put, the removal of a delete) now holds the lock exclusively, and every read holds it shared, so a reader
/// never sees the name between the two states: when it is told the file is not there, it is not there.
///
/// Each operation opens the lock file for itself, so the lock excludes other threads of this process as it does other processes (a lock belongs to the open
/// file, and two threads that shared one would not exclude each other). The operating system drops the lock if the process dies.
pub(crate) struct DirLock(File);

impl DirLock {
    /// Takes the lock on `file`, waiting at most `wait`.
    pub(super) fn acquire(file: File, exclusive: bool, wait: Duration) -> Result<DirLock, KeyError> {
        let deadline = Instant::now() + wait;
        let mut pause = Duration::from_micros(200);
        loop {
            match if exclusive { file.try_lock() } else { file.try_lock_shared() } {
                Ok(()) => return Ok(DirLock(file)),
                Err(TryLockError::WouldBlock) => {
                    if Instant::now() >= deadline {
                        let why = io::Error::new(io::ErrorKind::TimedOut, "another process holds the lock on the keys directory");
                        return Err(KeyError::io("lock the keys directory", why));
                    }
                    thread::sleep(pause);
                    pause = (pause * 2).min(Duration::from_millis(10));
                }
                Err(TryLockError::Error(e)) => return Err(KeyError::io("lock the keys directory", e)),
            }
        }
    }
}

impl Drop for DirLock {
    fn drop(&mut self) {
        // Windows releases a lock when the handle closes, but not at once; unlocking by hand is prompt everywhere
        let _ = self.0.unlock();
    }
}

/// A temporary file is debris only when it is older than this **and** nobody holds its lock (review L-6): the file of a `put` that is in progress is young and
/// is locked while it is written, and opening the store in another process used to delete it, so that the put failed with a misleading `key_verify_failed`
/// (870 of 874 puts on Linux, about 1,200 of 1,265 on Windows, with a second process opening the store in a loop).
pub(crate) const STALE_AFTER: Duration = Duration::from_secs(60);

/// Whether `file` (a temporary file, open for writing) is debris: old enough, and its lock is free. When this is true the lock is held by `file`, so
/// that nobody starts to use it before the caller has removed it.
pub(super) fn claim_if_stale(file: &File, min_age: Duration) -> bool {
    let old = file.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age >= min_age);
    old && file.try_lock().is_ok()
}

/// One entry of the folder.
pub(crate) struct Entry {
    /// The file name (only names that are valid text are listed).
    pub name: String,
    /// A regular file: not a directory, a link or anything else.
    pub is_file: bool,
}

/// Reads a whole file into one buffer of exactly its size (a buffer that grows leaves earlier, smaller copies of a secret behind in freed memory),
/// refusing one that is too big to be a blob.
pub(super) fn read_blob(mut file: File) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    let length = file.metadata().map_err(|e| KeyError::io("inspect a key file", e))?.len();
    read_exactly(&mut file, length)
}

/// Reads `length` bytes from `reader`, which must end there: a reader that has more (the file grew between the moment its size was asked and the read) is an
/// error and not a truncated secret, and one that has less is an error too. A `length` that no blob can have is refused before anything is allocated.
pub(super) fn read_exactly(reader: &mut impl Read, length: u64) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    if length > MAX_BLOB_LEN as u64 {
        return Err(KeyError::Corrupt("larger than any secret"));
    }
    let mut bytes = Zeroizing::new(vec![0u8; length as usize]);
    reader.read_exact(&mut bytes).map_err(|e| KeyError::io("read a key file", e))?;
    let mut extra = [0u8; 1];
    if reader.read(&mut extra).map_err(|e| KeyError::io("read a key file", e))? != 0 {
        return Err(KeyError::Corrupt("the file changed while it was read"));
    }
    Ok(bytes)
}

/// What a test runs at a named point of the code.
#[cfg(test)]
type Action = std::sync::Arc<dyn Fn() + Send + Sync>;

/// Hooks of `KeyDir::open` (Windows): a test that acts between the steps of an open names the path it is opening, so that tests that run at the same time do not see
/// each other's.
#[cfg(all(test, windows))]
pub(crate) mod open_hooks {
    use super::Action;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    type Entry = (PathBuf, &'static str, Action);
    static HOOKS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

    /// Runs `action` when an open of exactly `path` passes the named point (`after_walk`: the path was walked for links and the folder is not open yet).
    pub(crate) fn at(path: &Path, point: &'static str, action: impl Fn() + Send + Sync + 'static) {
        HOOKS.lock().unwrap().push((path.to_path_buf(), point, Arc::new(action)));
    }

    pub(crate) fn fire(path: &Path, point: &str) {
        let action = {
            let mut hooks = HOOKS.lock().unwrap();
            hooks.iter().position(|(p, q, _)| p == path && *q == point).map(|at| hooks.remove(at).2)
        };
        if let Some(action) = action {
            action();
        }
    }
}

/// Test hooks: a failure to inject, a count, and moments at which a test can do what an attacker or another process would.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Hooks {
    /// Makes the next `create_new` fail, as a full disk does.
    pub fail_create: std::sync::atomic::AtomicBool,
    /// How many times the folder was flushed.
    pub syncs: std::sync::atomic::AtomicUsize,
    points: std::sync::Mutex<Vec<(&'static str, Action)>>,
}

#[cfg(test)]
impl Hooks {
    /// Runs `action` whenever the code passes the named point (`before_open`: after the folder was judged and before a file is opened).
    pub(crate) fn at(&self, point: &'static str, action: impl Fn() + Send + Sync + 'static) {
        self.points.lock().unwrap().push((point, std::sync::Arc::new(action)));
    }

    #[cfg_attr(windows, allow(dead_code))] // the tests that clear hooks act as an attacker who renames folders, which Windows does not allow
    pub(crate) fn clear(&self) {
        self.points.lock().unwrap().clear();
    }

    fn fire(&self, point: &'static str) {
        let actions: Vec<_> = self.points.lock().unwrap().iter().filter(|(p, _)| *p == point).map(|(_, a)| std::sync::Arc::clone(a)).collect();
        for action in actions {
            action();
        }
    }
}

impl KeyDir {
    /// A point in the code at which a test may act; nothing outside tests.
    #[cfg(test)]
    pub(crate) fn fire(&self, point: &'static str) {
        self.hooks.fire(point);
    }

    #[cfg(not(test))]
    #[inline(always)]
    pub(crate) fn fire(&self, _point: &'static str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// KM08: the size of a file is asked, and the file is read; in between it can grow. The extra byte is what says so: without the check the first `length`
    /// bytes of a file that changed are returned as if they were the file.
    #[test]
    fn a_file_that_grew_or_shrank_while_it_was_read_is_an_error_not_a_secret() {
        let ok = read_exactly(&mut Cursor::new(vec![7u8; 4]), 4).unwrap();
        assert_eq!(&**ok, &[7u8; 4]);
        let grew = read_exactly(&mut Cursor::new(vec![7u8; 5]), 4).unwrap_err();
        assert!(matches!(grew, KeyError::Corrupt("the file changed while it was read")), "{grew:?}");
        let shrank = read_exactly(&mut Cursor::new(vec![7u8; 3]), 4).unwrap_err();
        assert!(matches!(shrank, KeyError::Io { op: "read a key file", .. }), "{shrank:?}");
        let empty = read_exactly(&mut Cursor::new(Vec::new()), 0).unwrap();
        assert!(empty.is_empty());
    }

    /// A length that no blob can have is refused before the reader is touched and before any buffer exists.
    #[test]
    fn a_length_no_blob_can_have_is_refused_before_anything_is_read_or_allocated() {
        struct NeverRead;
        impl Read for NeverRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                panic!("the reader was touched");
            }
        }
        let error = read_exactly(&mut NeverRead, MAX_BLOB_LEN as u64 + 1).unwrap_err();
        assert!(matches!(error, KeyError::Corrupt("larger than any secret")), "{error:?}");
        assert!(read_exactly(&mut Cursor::new(vec![0u8; MAX_BLOB_LEN]), MAX_BLOB_LEN as u64).is_ok(), "the largest blob is read");
    }
}
