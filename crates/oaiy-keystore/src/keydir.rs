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
//! - **Windows:** a handle to the folder held open **without `FILE_SHARE_DELETE`**, so that nobody can rename or remove the folder, or a folder above it,
//!   while the store is open: the same guarantee by other means. The folder is opened as itself (`FILE_FLAG_OPEN_REPARSE_POINT`) and refused if it is a
//!   junction or a symbolic link, and every file is opened as itself and refused if it is a reparse point or not a regular file. The handle is also what
//!   the store flushes after a rename, so that the new name survives a power cut.
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
/// means a process that is stuck, and an error is better than a caller that never returns.
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
    if length > MAX_BLOB_LEN as u64 {
        return Err(KeyError::Corrupt("larger than any secret"));
    }
    let mut bytes = Zeroizing::new(vec![0u8; length as usize]);
    file.read_exact(&mut bytes).map_err(|e| KeyError::io("read a key file", e))?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra).map_err(|e| KeyError::io("read a key file", e))? != 0 {
        return Err(KeyError::Corrupt("the file changed while it was read"));
    }
    Ok(bytes)
}

/// What a test runs at a named point of the code.
#[cfg(test)]
type Action = std::sync::Arc<dyn Fn() + Send + Sync>;

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
    #[cfg_attr(windows, allow(dead_code))] // the tests that use hooks act as an attacker who renames folders, which Windows does not allow
    pub(crate) fn at(&self, point: &'static str, action: impl Fn() + Send + Sync + 'static) {
        self.points.lock().unwrap().push((point, std::sync::Arc::new(action)));
    }

    #[cfg_attr(windows, allow(dead_code))]
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
