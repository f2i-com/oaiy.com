//! The keys folder on Unix: a directory descriptor, and every operation relative to it (see `keydir.rs` for why).

use std::fs::{self, File};
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rustix::fs::{fsync, openat, renameat, statat, unlinkat, AtFlags, Dir, FileType, Mode, OFlags, CWD};
use rustix::io::Errno;
use zeroize::Zeroizing;

use super::{claim_if_stale, read_blob, DirLock, Entry, LOCK_FILE, LOCK_WAIT};
use crate::error::KeyError;
use crate::perm::{self, FileKind, Kind};

/// The most directories above the keys folder that are walked (a path is never this deep; a loop of links cannot make the walk endless).
const MAX_DEPTH: usize = 4096;

/// The keys folder, held open.
pub(crate) struct KeyDir {
    path: PathBuf,
    dir: File,
    /// The user this process runs as, which every file and folder of the store must belong to (a test may pretend to be someone else).
    pub(crate) uid: u32,
    dev: u64,
    ino: u64,
    /// How long [`KeyDir::lock`] waits for another holder.
    pub(crate) lock_wait: Duration,
    #[cfg(test)]
    pub(crate) hooks: super::Hooks,
}

fn errno(op: &'static str, e: Errno) -> KeyError {
    KeyError::io(op, io::Error::from(e))
}

impl KeyDir {
    /// Opens the folder (making it `0700` if it is not there), judges it and every directory above it, and holds it.
    pub(crate) fn open(path: &Path) -> Result<KeyDir, KeyError> {
        KeyDir::open_as(path, rustix::process::geteuid().as_raw())
    }

    /// [`KeyDir::open`] for the user `uid` (the one this process runs as, except in a test that shows the owner rules are applied).
    pub(crate) fn open_as(path: &Path, uid: u32) -> Result<KeyDir, KeyError> {
        match fs::symlink_metadata(path) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // a new folder is made 0700; an existing one is checked, never repaired
                match fs::DirBuilder::new().recursive(true).mode(0o700).create(path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(KeyError::io("create the keys directory", e)),
                }
            }
            Err(e) => return Err(KeyError::io("inspect the keys directory", e)),
        }
        let fd = openat(CWD, path, OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC, Mode::empty()).map_err(|e| match e {
            Errno::LOOP => KeyError::Permissions(format!("{}: is a symbolic link", path.display())),
            other => errno("open the keys directory", other),
        })?;
        let dir = File::from(fd);
        let meta = dir.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        perm::check(Kind::Dir, &perm::meta_of(&meta), Some(uid)).map_err(|why| KeyError::Permissions(format!("{}: {why}", path.display())))?;
        let found = KeyDir {
            path: path.to_path_buf(),
            dir,
            uid,
            dev: meta.dev(),
            ino: meta.ino(),
            lock_wait: LOCK_WAIT,
            #[cfg(test)]
            hooks: super::Hooks::default(),
        };
        found.check_ancestors()?;
        Ok(found)
    }

    fn display(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }

    /// Walks `..` from the descriptor to the root, judging every directory on the way: the directories that hold the folder the store is in, as the file
    /// system has them, whatever symbolic links the path went through to get here. Whoever can rename or remove the folder can put another in its place.
    pub(crate) fn check_ancestors(&self) -> Result<(), KeyError> {
        let mut current = self.dir.try_clone().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        let (mut dev, mut ino) = (self.dev, self.ino);
        for level in 1..=MAX_DEPTH {
            let parent = openat(&current, "..", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty())
                .map_err(|e| errno("inspect the directories above the keys directory", e))?;
            let parent = File::from(parent);
            let meta = parent.metadata().map_err(|e| KeyError::io("inspect the directories above the keys directory", e))?;
            if meta.dev() == dev && meta.ino() == ino {
                return Ok(()); // the root is its own parent
            }
            perm::check_ancestor(&perm::meta_of(&meta), self.uid)
                .map_err(|why| KeyError::Permissions(format!("{}: the directory {level} above the keys directory {why}", self.path.display())))?;
            (dev, ino) = (meta.dev(), meta.ino());
            current = parent;
        }
        Err(KeyError::Permissions(format!("{}: more than {MAX_DEPTH} directories above the keys directory", self.path.display())))
    }

    /// Judged before every operation: the folder is still what was opened, still private, and still where the path says.
    pub(crate) fn verify(&self) -> Result<(), KeyError> {
        let meta = self.dir.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        perm::check(Kind::Dir, &perm::meta_of(&meta), Some(self.uid))
            .map_err(|why| KeyError::Permissions(format!("{}: {why}", self.path.display())))?;
        let by_path = fs::symlink_metadata(&self.path).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        if by_path.dev() != self.dev || by_path.ino() != self.ino {
            return Err(KeyError::io("inspect the keys directory", io::Error::other("the keys directory was replaced or moved while it was open")));
        }
        Ok(())
    }

    /// Takes the advisory lock of the folder (see [`DirLock`]), shared for a read and exclusive for a change.
    pub(crate) fn lock(&self, exclusive: bool) -> Result<DirLock, KeyError> {
        let flags = OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        let fd = openat(&self.dir, LOCK_FILE, flags, Mode::RUSR | Mode::WUSR).map_err(|e| match e {
            Errno::LOOP => KeyError::Permissions(format!("{}: is a symbolic link", self.display(LOCK_FILE))),
            other => errno("open the lock file", other),
        })?;
        let file = File::from(fd);
        let meta = file.metadata().map_err(|e| KeyError::io("inspect the lock file", e))?;
        perm::check(Kind::File, &perm::meta_of(&meta), Some(self.uid))
            .map_err(|why| KeyError::Permissions(format!("{}: {why}", self.display(LOCK_FILE))))?;
        DirLock::acquire(file, exclusive, self.lock_wait)
    }

    /// The contents of one file of the folder, or `None` if the folder has no such entry.
    pub(crate) fn read(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
        self.fire("before_open");
        let fd = match openat(&self.dir, name, OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC, Mode::empty()) {
            Ok(fd) => fd,
            Err(Errno::NOENT) => return Ok(None),
            Err(Errno::LOOP) => return Err(KeyError::Permissions(format!("{}: is a symbolic link", self.display(name)))),
            Err(e) => return Err(errno("open a key file", e)),
        };
        let file = File::from(fd);
        let meta = file.metadata().map_err(|e| KeyError::io("inspect a key file", e))?;
        perm::check(Kind::File, &perm::meta_of(&meta), Some(self.uid))
            .map_err(|why| KeyError::Permissions(format!("{}: {why}", self.display(name))))?;
        read_blob(file).map(Some)
    }

    /// Makes a new file that must not exist, `0600`.
    pub(crate) fn create_new(&self, name: &str) -> Result<File, KeyError> {
        #[cfg(test)]
        if self.hooks.fail_create.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(KeyError::io("create the temporary file", io::Error::other("injected: no space left on device")));
        }
        let flags = OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let fd = openat(&self.dir, name, flags, Mode::RUSR | Mode::WUSR).map_err(|e| errno("create the temporary file", e))?;
        Ok(File::from(fd))
    }

    /// Renames `from` to `to`, replacing `to`.
    pub(crate) fn rename(&self, from: &str, to: &str) -> Result<(), KeyError> {
        renameat(&self.dir, from, &self.dir, to).map_err(|e| errno("replace the key file", e))
    }

    /// Removes the temporary file `name` if it is debris: one of ours, older than `min_age`, and not locked by anyone (see [`STALE_AFTER`](super::STALE_AFTER)).
    /// Best effort: whether it was removed is the answer, and no failure is an error.
    pub(crate) fn remove_if_stale(&self, name: &str, min_age: Duration) -> bool {
        let flags = OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        let Ok(fd) = openat(&self.dir, name, flags, Mode::empty()) else { return false };
        let file = File::from(fd);
        let Ok(meta) = file.metadata() else { return false };
        let meta = perm::meta_of(&meta);
        if meta.kind != FileKind::Regular || meta.uid != self.uid || !claim_if_stale(&file, min_age) {
            return false;
        }
        unlinkat(&self.dir, name, AtFlags::empty()).is_ok()
    }

    /// Removes a file; one that is not there is not an error.
    pub(crate) fn remove(&self, name: &str) -> Result<(), KeyError> {
        match unlinkat(&self.dir, name, AtFlags::empty()) {
            Ok(()) | Err(Errno::NOENT) => Ok(()),
            Err(e) => Err(errno("remove a key file", e)),
        }
    }

    /// Whether the folder has an entry of any kind called `name` (a link, a FIFO and a directory count).
    pub(crate) fn exists(&self, name: &str) -> Result<bool, KeyError> {
        match statat(&self.dir, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => Ok(true),
            Err(Errno::NOENT) => Ok(false),
            Err(e) => Err(errno("inspect the keys directory", e)),
        }
    }

    /// The entries of the folder, through a descriptor of their own (the read position of a shared descriptor would be shared by every lister).
    pub(crate) fn list(&self) -> Result<Vec<Entry>, KeyError> {
        let op = "list the keys directory";
        let fd = openat(&self.dir, ".", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty())
            .map_err(|e| errno(op, e))?;
        let reader = Dir::new(fd).map_err(|e| errno(op, e))?;
        let mut out = Vec::new();
        for entry in reader {
            let entry = entry.map_err(|e| errno(op, e))?;
            let Ok(name) = entry.file_name().to_str() else { continue };
            if name == "." || name == ".." {
                continue;
            }
            let is_file = match entry.file_type() {
                FileType::RegularFile => true,
                FileType::Unknown => statat(&self.dir, name, AtFlags::SYMLINK_NOFOLLOW)
                    .map(|s| FileType::from_raw_mode(s.st_mode) == FileType::RegularFile)
                    .unwrap_or(false),
                _ => false,
            };
            out.push(Entry { name: name.to_owned(), is_file });
        }
        Ok(out)
    }

    /// Flushes the folder, so that a rename or a removal survives a power cut.
    pub(crate) fn sync(&self) -> Result<(), KeyError> {
        #[cfg(test)]
        self.hooks.syncs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match fsync(&self.dir) {
            Ok(()) | Err(Errno::INVAL) => Ok(()), // a file system that cannot flush a folder (some network and FUSE ones) says so
            Err(e) => Err(errno("flush the keys directory", e)),
        }
    }
}
