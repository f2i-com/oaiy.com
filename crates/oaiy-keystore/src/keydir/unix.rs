//! The keys folder on Unix: a directory descriptor, and every operation relative to it (see `mod.rs` for why).

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use rustix::fs::{fsync, openat, renameat, statat, unlinkat, AtFlags, Dir, FileType, Mode, OFlags, CWD};
use rustix::io::Errno;
use zeroize::Zeroizing;

use super::{claim_if_stale, read_blob, DirLock, Entry, LOCK_FILE, LOCK_WAIT};
use crate::error::KeyError;
use crate::perm::{self, FileKind, Kind};
use crate::store::Durability;

/// The most directories above the keys folder that are walked (a path is never this deep; a loop of links cannot make the walk endless).
const MAX_DEPTH: usize = 4096;

/// Resolves a relative path against the current directory without following symlinks and strips trailing slashes and `.` segments
/// (leaving bare `/` intact), so `O_NOFOLLOW` and `lstat` always inspect the final entry itself rather than following it via a trailing `/`.
fn normalize_path(path: &Path) -> Result<PathBuf, KeyError> {
    let abs = std::path::absolute(path).map_err(|e| KeyError::io("inspect the keys directory", e))?;
    let mut clean: PathBuf = abs.components().collect();
    let bytes = clean.as_os_str().as_bytes();
    if bytes.len() > 1 && bytes.ends_with(b"/") {
        let trimmed = bytes.iter().rposition(|&b| b != b'/').map_or(&b"/"[..], |i| &bytes[..=i]);
        clean = PathBuf::from(OsStr::from_bytes(trimmed));
    }
    Ok(clean)
}

/// Walks `..` from `start` to the root, judging every directory on the way with [`perm::check_ancestor`].
fn check_ancestors_of(start: &File, mut dev: u64, mut ino: u64, display_path: &Path, uid: u32) -> Result<(), KeyError> {
    let mut current = start.try_clone().map_err(|e| KeyError::io("inspect the keys directory", e))?;
    for level in 1..=MAX_DEPTH {
        let parent = openat(&current, "..", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty())
            .map_err(|e| errno("inspect the directories above the keys directory", e))?;
        let parent = File::from(parent);
        let meta = parent.metadata().map_err(|e| KeyError::io("inspect the directories above the keys directory", e))?;
        if meta.dev() == dev && meta.ino() == ino {
            return Ok(()); // the root is its own parent
        }
        perm::check_ancestor(&perm::meta_of(&meta), uid)
            .map_err(|why| KeyError::Permissions(format!("{}: the directory {level} above the keys directory {why}", display_path.display())))?;
        (dev, ino) = (meta.dev(), meta.ino());
        current = parent;
    }
    Err(KeyError::Permissions(format!("{}: more than {MAX_DEPTH} directories above the keys directory", display_path.display())))
}

/// Walks every syntactic component of `path` (which must be absolute and normalized) with `openat` and `statat(..., SYMLINK_NOFOLLOW)`.
/// Every parent directory along the syntactic path is checked with [`perm::check_ancestor`]. An intermediate symbolic link is allowed only
/// when its parent passes `check_ancestor`, the link itself belongs to `uid` or root, and the link target's physical ancestor chain also passes
/// `check_ancestor`. A symbolic link on the final component is always refused.
///
/// When `allow_missing` is true, returns `Ok(None)` as soon as a missing component is met (after checking its parent). Otherwise opens the
/// final component with `O_DIRECTORY | O_NOFOLLOW` and returns `Ok(Some(dir))`.
fn walk_syntactic(path: &Path, uid: u32, allow_missing: bool) -> Result<Option<File>, KeyError> {
    let root_fd = openat(CWD, "/", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty())
        .map_err(|e| errno("inspect the directories above the keys directory", e))?;
    let mut dir = File::from(root_fd);
    let comps: Vec<Component<'_>> = path.components().filter(|c| !matches!(c, Component::RootDir | Component::CurDir)).collect();
    if comps.is_empty() {
        return Ok(Some(dir));
    }
    let mut parent_path = PathBuf::from("/");
    let mut so_far = PathBuf::from("/");
    for (idx, comp) in comps.iter().enumerate() {
        let is_final = idx + 1 == comps.len();
        let name = Path::new(comp.as_os_str());
        so_far.push(name);

        let parent_meta = dir.metadata().map_err(|e| KeyError::io("inspect the directories above the keys directory", e))?;
        perm::check_ancestor(&perm::meta_of(&parent_meta), uid).map_err(|why| {
            KeyError::Permissions(format!("{}: the directory {} above the keys directory {why}", path.display(), parent_path.display()))
        })?;

        let st = match statat(&dir, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(st) => st,
            Err(Errno::NOENT) if allow_missing => return Ok(None),
            Err(e) => return Err(errno("inspect the directories above the keys directory", e)),
        };
        let ft = FileType::from_raw_mode(st.st_mode);

        if is_final {
            if ft == FileType::Symlink {
                return Err(KeyError::Permissions(format!("{}: is a symbolic link", path.display())));
            }
            if ft != FileType::Directory {
                return Err(KeyError::Permissions(format!("{}: is not a directory", path.display())));
            }
            let fd = openat(&dir, name, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC, Mode::empty())
                .map_err(|e| match e {
                    Errno::LOOP => KeyError::Permissions(format!("{}: is a symbolic link", path.display())),
                    Errno::NOTDIR => KeyError::Permissions(format!("{}: is not a directory", path.display())),
                    other => errno("open the keys directory", other),
                })?;
            return Ok(Some(File::from(fd)));
        }

        if ft == FileType::Symlink {
            // The parent already passed `check_ancestor`; also refuse a symlink owned by another user (e.g. in sticky `/tmp`).
            if st.st_uid != 0 && st.st_uid != uid {
                return Err(KeyError::Permissions(format!(
                    "{}: the symbolic link {} above the keys directory belongs to another user (uid {})",
                    path.display(),
                    so_far.display(),
                    st.st_uid
                )));
            }
            let next_fd = openat(&dir, name, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NONBLOCK | OFlags::CLOEXEC, Mode::empty()).map_err(|e| match e {
                Errno::NOTDIR => KeyError::Permissions(format!("{}: {} is not a directory", path.display(), so_far.display())),
                other => errno("inspect the directories above the keys directory", other),
            })?;
            let next = File::from(next_fd);
            let next_meta = next.metadata().map_err(|e| KeyError::io("inspect the directories above the keys directory", e))?;
            check_ancestors_of(&next, next_meta.dev(), next_meta.ino(), path, uid)?;
            dir = next;
        } else {
            let next_fd = openat(&dir, name, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC, Mode::empty())
                .map_err(|e| match e {
                    Errno::LOOP => KeyError::Permissions(format!("{}: {} is a symbolic link", path.display(), so_far.display())),
                    Errno::NOTDIR => KeyError::Permissions(format!("{}: {} is not a directory", path.display(), so_far.display())),
                    other => errno("inspect the directories above the keys directory", other),
                })?;
            dir = File::from(next_fd);
        }
        parent_path = so_far.clone();
    }
    Ok(Some(dir))
}

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
        let clean = normalize_path(path)?;
        // Walk syntactic components before creating anything so a symlink inside an untrusted parent or on the final component is refused first.
        let dir = match walk_syntactic(&clean, uid, true)? {
            Some(dir) => dir,
            None => {
                // a new folder is made 0700; an existing one is checked, never repaired
                match fs::DirBuilder::new().recursive(true).mode(0o700).create(&clean) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(KeyError::io("create the keys directory", e)),
                }
                walk_syntactic(&clean, uid, false)?
                    .ok_or_else(|| KeyError::io("open the keys directory", io::Error::from(io::ErrorKind::NotFound)))?
            }
        };
        let meta = dir.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        perm::check(Kind::Dir, &perm::meta_of(&meta), Some(uid)).map_err(|why| KeyError::Permissions(format!("{}: {why}", clean.display())))?;
        let found = KeyDir {
            path: clean,
            dir,
            uid,
            dev: meta.dev(),
            ino: meta.ino(),
            lock_wait: LOCK_WAIT,
            #[cfg(test)]
            hooks: super::Hooks::default(),
        };
        found.check_ancestors()?;
        // Re-walk syntactic components after open and confirm the path still resolves to the exact directory descriptor we hold.
        let recheck = walk_syntactic(&found.path, uid, false)?
            .ok_or_else(|| KeyError::io("inspect the keys directory", io::Error::from(io::ErrorKind::NotFound)))?;
        let recheck_meta = recheck.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        if recheck_meta.dev() != found.dev || recheck_meta.ino() != found.ino {
            return Err(KeyError::io("inspect the keys directory", io::Error::other("the keys directory was replaced or moved while it was opened")));
        }
        // the lock file is made here, once, and an operation only opens it: one that has gone is an error, not a lock that is made again beside the holders of the old one
        drop(found.open_lock(true)?);
        Ok(found)
    }

    fn display(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }

    /// Walks `..` from the descriptor to the root, judging every directory on the way: the directories that hold the folder the store is in, as the file
    /// system has them, whatever symbolic links the path went through to get here. Whoever can rename or remove the folder can put another in its place.
    pub(crate) fn check_ancestors(&self) -> Result<(), KeyError> {
        check_ancestors_of(&self.dir, self.dev, self.ino, &self.path, self.uid)
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
        DirLock::acquire(self.open_lock(false)?, exclusive, self.lock_wait)
    }

    /// Opens the lock file (`O_NOFOLLOW`), making it only if `create`, and judges it as any key file is judged: a regular file, ours, no group or other permission, **one name**.
    fn open_lock(&self, create: bool) -> Result<File, KeyError> {
        let mut flags = OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        if create {
            flags |= OFlags::CREATE;
        }
        let fd = openat(&self.dir, LOCK_FILE, flags, Mode::RUSR | Mode::WUSR).map_err(|e| match e {
            Errno::LOOP => KeyError::Permissions(format!("{}: is a symbolic link", self.display(LOCK_FILE))),
            other => errno("open the lock file", other),
        })?;
        let file = File::from(fd);
        let meta = file.metadata().map_err(|e| KeyError::io("inspect the lock file", e))?;
        perm::check(Kind::File, &perm::meta_of(&meta), Some(self.uid))
            .map_err(|why| KeyError::Permissions(format!("{}: {why}", self.display(LOCK_FILE))))?;
        Ok(file)
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

    /// Flushes the folder, so that a rename or a removal survives a power cut. A file system that cannot flush a folder says so (`EINVAL`, `ENOTSUP`, `ENOSYS`; some network and FUSE
    /// ones) and that is `Unconfirmed`, not an error.
    pub(crate) fn sync(&self) -> Result<Durability, KeyError> {
        #[cfg(test)]
        self.hooks.syncs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        #[cfg(test)]
        if let Some(injected) = self.hooks.injected_flush_error() {
            return super::flushed(Err(injected));
        }
        super::flushed(fsync(&self.dir).map_err(io::Error::from))
    }
}
