//! The lock on `<data>/auth`: one process at a time.
//!
//! The running server is the only writer of the auth files, and memory is authoritative, so two
//! processes on one data folder would each overwrite the other's credentials. The lock is an exclusive
//! `fs2` lock on `<data>/auth/.lock`, held for the life of the server or the desktop; the operating
//! system releases it when the process ends however it ends, so a crash leaves no stale lock.
//!
//! The holder's process id goes in `.lock.pid`, a file of its own: on Windows a locked file cannot be
//! read by another handle, and the error that names the holder must be readable.

use std::fs::{File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use fs2::FileExt as _;

/// Why the folder could not be locked.
#[derive(Debug)]
pub enum LockError {
    /// Another process holds it (its id, when it could be read).
    InUse { pid: Option<u32> },
    /// The folder or the lock file could not be made or opened.
    Io { path: PathBuf, source: io::Error },
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::InUse { pid: Some(pid) } => write!(f, "data folder in use by process {pid}"),
            LockError::InUse { pid: None } => write!(f, "data folder in use by another process"),
            LockError::Io { path, source } => write!(f, "cannot lock {}: {source}", path.display()),
        }
    }
}

impl std::error::Error for LockError {}

/// The lock. Dropping it releases it.
#[derive(Debug)]
pub struct AuthLock {
    file: File,
    dir: PathBuf,
}

fn open_private(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path)
}

impl AuthLock {
    /// Lock `auth_dir` (`<data>/auth`), making it first (owner-only) if it is not there.
    pub fn acquire(auth_dir: &Path) -> Result<AuthLock, LockError> {
        crate::secret_file::create_private_dir(auth_dir).map_err(|source| LockError::Io {
            path: auth_dir.to_path_buf(),
            source,
        })?;
        let lock_path = auth_dir.join(".lock");
        let pid_path = auth_dir.join(".lock.pid");
        let file = open_private(&lock_path).map_err(|source| LockError::Io {
            path: lock_path.clone(),
            source,
        })?;
        match file.try_lock_exclusive() {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == fs2::lock_contended_error().raw_os_error()
                    || e.kind() == io::ErrorKind::WouldBlock =>
            {
                let pid = std::fs::read_to_string(&pid_path)
                    .ok()
                    .and_then(|t| t.trim().parse().ok());
                return Err(LockError::InUse { pid });
            }
            Err(source) => {
                return Err(LockError::Io {
                    path: lock_path,
                    source,
                })
            }
        }
        let pid = std::process::id().to_string();
        // Best effort: the lock is what matters; the id only makes the refusal readable.
        let _ = file.set_len(0);
        let _ = (&file).write_all(pid.as_bytes());
        if let Ok(mut side) = open_private(&pid_path) {
            let _ = side.set_len(0);
            let _ = side.write_all(pid.as_bytes());
        }
        Ok(AuthLock {
            file,
            dir: auth_dir.to_path_buf(),
        })
    }

    /// The locked folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for AuthLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::TempDir;

    #[test]
    fn a_second_open_of_a_locked_folder_fails_and_names_the_holder() {
        let dir = TempDir::new("lock-twice");
        let auth = dir.0.join("auth");
        let first = AuthLock::acquire(&auth).expect("the first lock");
        match AuthLock::acquire(&auth) {
            Err(LockError::InUse { pid }) => assert_eq!(pid, Some(std::process::id())),
            other => panic!("a second lock on a held folder: {other:?}"),
        }
        let message = AuthLock::acquire(&auth).unwrap_err().to_string();
        assert_eq!(
            message,
            format!("data folder in use by process {}", std::process::id())
        );
        drop(first);
        AuthLock::acquire(&auth).expect("the lock is free again once the holder lets go");
    }

    #[test]
    fn different_folders_lock_independently() {
        let dir = TempDir::new("lock-two");
        let a = AuthLock::acquire(&dir.0.join("a")).unwrap();
        let b = AuthLock::acquire(&dir.0.join("b")).unwrap();
        assert_ne!(a.dir(), b.dir());
    }

    #[test]
    fn an_unreadable_holder_is_still_a_refusal() {
        let dir = TempDir::new("lock-nopid");
        let auth = dir.0.join("auth");
        let _held = AuthLock::acquire(&auth).unwrap();
        std::fs::remove_file(auth.join(".lock.pid")).unwrap();
        match AuthLock::acquire(&auth) {
            Err(e @ LockError::InUse { pid: None }) => {
                assert_eq!(e.to_string(), "data folder in use by another process")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_folder_that_cannot_be_made_is_an_io_error_naming_it() {
        let dir = TempDir::new("lock-io");
        // A file where the folder belongs.
        std::fs::write(dir.0.join("auth"), "in the way").unwrap();
        match AuthLock::acquire(&dir.0.join("auth")) {
            Err(LockError::Io { path, .. }) => assert!(path.ends_with("auth")),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_folder_and_the_lock_files_are_private() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("lock-mode");
        let auth = dir.0.join("auth");
        let _held = AuthLock::acquire(&auth).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&auth), 0o700);
        assert_eq!(mode(&auth.join(".lock")), 0o600);
        assert_eq!(mode(&auth.join(".lock.pid")), 0o600);
    }
}
