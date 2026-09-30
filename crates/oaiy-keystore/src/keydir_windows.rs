//! The keys folder on Windows: a handle to the folder held open without `FILE_SHARE_DELETE`, so that it cannot be renamed or removed while the store is
//! open (see `keydir.rs` for why), and every file opened as itself, never through a junction or a symbolic link.

use std::fs::{self, File, Metadata, OpenOptions};
use std::io;
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use zeroize::Zeroizing;

use super::{read_blob, DirLock, Entry, LOCK_FILE, LOCK_WAIT};
use crate::error::KeyError;

const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
const FILE_SHARE_READ: u32 = 1;
const FILE_SHARE_WRITE: u32 = 2;

/// The keys folder, held open.
pub(crate) struct KeyDir {
    path: PathBuf,
    held: File,
    /// How long [`KeyDir::lock`] waits for another holder.
    pub(crate) lock_wait: Duration,
    #[cfg(test)]
    pub(crate) hooks: super::Hooks,
}

/// What a Windows file is, from its attributes: a reparse point (a junction, a symbolic link, a mount point) is refused wherever it stands.
fn refuse_reparse_point(meta: &Metadata, what: &str) -> Result<(), KeyError> {
    if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(KeyError::Permissions(format!("{what}: is a junction or a symbolic link")));
    }
    Ok(())
}

fn require_directory(meta: &Metadata, what: &str) -> Result<(), KeyError> {
    refuse_reparse_point(meta, what)?;
    if meta.file_attributes() & FILE_ATTRIBUTE_DIRECTORY == 0 {
        return Err(KeyError::Permissions(format!("{what}: is not a directory")));
    }
    Ok(())
}

fn require_file(meta: &Metadata, what: &str) -> Result<(), KeyError> {
    refuse_reparse_point(meta, what)?;
    if meta.file_attributes() & FILE_ATTRIBUTE_DIRECTORY != 0 {
        return Err(KeyError::Permissions(format!("{what}: is a directory")));
    }
    Ok(())
}

impl KeyDir {
    /// Opens the folder (making it if it is not there), refuses a junction or a link, and holds it.
    pub(crate) fn open(path: &Path) -> Result<KeyDir, KeyError> {
        match fs::symlink_metadata(path) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(path).map_err(|e| KeyError::io("create the keys directory", e))?;
            }
            Err(e) => return Err(KeyError::io("inspect the keys directory", e)),
        }
        // no FILE_SHARE_DELETE: while this handle is open nobody can rename or remove the folder, or one above it. Write access is what flushing the folder needs.
        let held = OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .map_err(|e| KeyError::io("open the keys directory", e))?;
        let meta = held.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&meta, &path.display().to_string())?;
        Ok(KeyDir {
            path: path.to_path_buf(),
            held,
            lock_wait: LOCK_WAIT,
            #[cfg(test)]
            hooks: super::Hooks::default(),
        })
    }

    fn at(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Judged before every operation: the handle is still a folder, and the path still leads to a folder that is not a link.
    pub(crate) fn verify(&self) -> Result<(), KeyError> {
        let what = self.path.display().to_string();
        let meta = self.held.metadata().map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&meta, &what)?;
        let by_path = fs::symlink_metadata(&self.path).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&by_path, &what)
    }

    /// Takes the advisory lock of the folder (see [`DirLock`]), shared for a read and exclusive for a change (`LockFileEx` on the lock file).
    pub(crate) fn lock(&self, exclusive: bool) -> Result<DirLock, KeyError> {
        let path = self.at(LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(&path)
            .map_err(|e| KeyError::io("open the lock file", e))?;
        let meta = file.metadata().map_err(|e| KeyError::io("inspect the lock file", e))?;
        require_file(&meta, &path.display().to_string())?;
        DirLock::acquire(file, exclusive, self.lock_wait)
    }

    /// The contents of one file of the folder, or `None` if the folder has no such entry.
    pub(crate) fn read(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
        self.fire("before_open");
        let path = self.at(name);
        let file = match OpenOptions::new().read(true).custom_flags(FILE_FLAG_OPEN_REPARSE_POINT).open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(KeyError::io("open a key file", e)),
        };
        let meta = file.metadata().map_err(|e| KeyError::io("inspect a key file", e))?;
        require_file(&meta, &path.display().to_string())?;
        read_blob(file).map(Some)
    }

    /// Makes a new file that must not exist.
    pub(crate) fn create_new(&self, name: &str) -> Result<File, KeyError> {
        #[cfg(test)]
        if self.hooks.fail_create.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(KeyError::io("create the temporary file", io::Error::other("injected: no space left on device")));
        }
        OpenOptions::new().read(true).write(true).create_new(true).open(self.at(name)).map_err(|e| KeyError::io("create the temporary file", e))
    }

    /// Renames `from` to `to`, replacing `to`.
    pub(crate) fn rename(&self, from: &str, to: &str) -> Result<(), KeyError> {
        fs::rename(self.at(from), self.at(to)).map_err(|e| KeyError::io("replace the key file", e))
    }

    /// Removes a file; one that is not there is not an error.
    pub(crate) fn remove(&self, name: &str) -> Result<(), KeyError> {
        match fs::remove_file(self.at(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KeyError::io("remove a key file", e)),
        }
    }

    /// The entries of the folder.
    pub(crate) fn list(&self) -> Result<Vec<Entry>, KeyError> {
        let op = "list the keys directory";
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.path).map_err(|e| KeyError::io(op, e))? {
            let entry = entry.map_err(|e| KeyError::io(op, e))?;
            let Ok(name) = entry.file_name().into_string() else { continue };
            let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
            out.push(Entry { name, is_file });
        }
        Ok(out)
    }

    /// Flushes the folder, so that a rename or a removal survives a power cut (`FlushFileBuffers` on the folder handle).
    pub(crate) fn sync(&self) -> Result<(), KeyError> {
        #[cfg(test)]
        self.hooks.syncs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.held.sync_all() {
            Ok(()) => Ok(()),
            // a file system that cannot flush a folder (FAT, some network shares) says so
            Err(e) if matches!(e.kind(), io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported) => Ok(()),
            Err(e) => Err(KeyError::io("flush the keys directory", e)),
        }
    }
}
