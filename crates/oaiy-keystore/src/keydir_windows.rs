//! The keys folder on Windows. What holds, and what does not (review M-1):
//!
//! - **The folder is held open without `FILE_SHARE_DELETE`**, so while the store is open nobody can rename or remove it, **or a real folder above it** (a folder
//!   cannot be renamed or removed while anything inside it is open). That is the guarantee that the path to the folder cannot be swapped for another.
//! - **A junction, a symbolic link or a mount point is not a real folder**, and is not held by that: a junction is an entry in its parent that points somewhere,
//!   and whoever can change the parent can delete it and make it point somewhere else, with the store open (the reviewer did, and the store answered from another
//!   tree: `None` for keys that exist, a write into the other tree, and an older copy of the same user's keys served as the current one). So **the path is refused
//!   if any folder on it, the keys folder itself included, is a reparse point**, at open. The walk is repeated once the folder is held, because it cannot have
//!   been swapped after that but could have been between the first walk and the open (a hook in the tests does exactly that).
//! - **Every operation works in the folder that is held, by the folder's real path** (`winfs::final_path` of the handle: no junction in it, and it cannot change
//!   while the handle is open), never by the path the caller gave. And before each operation the path is opened afresh and its identity (volume and file index)
//!   compared with the handle's: a path that leads somewhere else now is an error, never "nothing stored" and never another tree's value.
//! - **What is left**, and is not claimed away: the window between the first walk and the open, closed by the second walk and the identity comparison except
//!   for an attacker who flips the path back and forth faster than a check takes (every later operation compares again); and whoever can **write to the keys
//!   folder itself** can delete a key file (a caller then sees `None`) or put back an older copy of a DPAPI blob (which the same user can unprotect: a rollback).
//!   Windows gives `E:\` and other roots that nobody set up "Modify" to Authenticated Users, which is exactly that: **keep the data folder under the user's
//!   profile** (`%LOCALAPPDATA%`, `%APPDATA%`), whose access control is the user, SYSTEM and Administrators. The store does not read or set ACLs.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use zeroize::Zeroizing;

use super::{claim_if_stale, read_blob, DirLock, Entry, LOCK_FILE, LOCK_WAIT};
use crate::error::KeyError;
use crate::winfs::{self, FileId, Info};

const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
const FILE_SHARE_READ: u32 = 1;
const FILE_SHARE_WRITE: u32 = 2;
const FILE_SHARE_ALL: u32 = 7;

/// The keys folder, held open.
pub(crate) struct KeyDir {
    /// The path the caller gave, made absolute: what [`KeyDir::verify`] compares with the handle. It is never used to reach a file.
    path: PathBuf,
    /// The real path of the folder that is held: where every file is, for as long as the handle is open.
    root: PathBuf,
    held: File,
    id: FileId,
    /// How long [`KeyDir::lock`] waits for another holder.
    pub(crate) lock_wait: Duration,
    #[cfg(test)]
    pub(crate) hooks: super::Hooks,
}

fn refuse_reparse_point(info: &Info, what: &str) -> Result<(), KeyError> {
    if info.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(KeyError::Permissions(format!("{what}: is a junction or a symbolic link")));
    }
    Ok(())
}

fn require_directory(info: &Info, what: &str) -> Result<(), KeyError> {
    refuse_reparse_point(info, what)?;
    if info.attributes & FILE_ATTRIBUTE_DIRECTORY == 0 {
        return Err(KeyError::Permissions(format!("{what}: is not a directory")));
    }
    Ok(())
}

/// A key file is a file with one name: a reparse point, a folder and a file that has another hard link (another way to reach the same bytes, from a place this
/// store does not look after) are refused.
fn require_file(info: &Info, what: &str, one_name: bool) -> Result<(), KeyError> {
    refuse_reparse_point(info, what)?;
    if info.attributes & FILE_ATTRIBUTE_DIRECTORY != 0 {
        return Err(KeyError::Permissions(format!("{what}: is a directory")));
    }
    if one_name && info.links > 1 {
        return Err(KeyError::Permissions(format!("{what}: has {} names (hard links): a key file has exactly one", info.links)));
    }
    Ok(())
}

/// Refuses the path if any folder on it is a junction, a symbolic link or a mount point (a reparse point of any kind). With `allow_missing`, the walk ends without
/// an error at the first component that is not there (the keys folder may not have been made yet, and what is there must be judged before anything is made
/// through it: a folder made through a junction is made in the tree the junction points to).
fn refuse_links_on_the_way(path: &Path, allow_missing: bool) -> Result<(), KeyError> {
    let mut so_far = PathBuf::new();
    for component in path.components() {
        so_far.push(component.as_os_str());
        if matches!(component, Component::Prefix(_) | Component::RootDir) {
            continue;
        }
        match fs::symlink_metadata(&so_far) {
            Ok(meta) if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 => {
                return Err(KeyError::Permissions(format!(
                    "{}: {} is a junction or a symbolic link: the keys directory is reached through real directories only",
                    path.display(),
                    so_far.display()
                )));
            }
            Ok(_) => {}
            Err(e) if allow_missing && e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(KeyError::io("inspect the directories above the keys directory", e)),
        }
    }
    Ok(())
}

/// A folder opened as itself (never through a reparse point at the end), sharing everything but deletion.
fn open_folder(path: &Path, share: u32, write: bool) -> io::Result<File> {
    OpenOptions::new().read(true).write(write).share_mode(share).custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT).open(path)
}

impl KeyDir {
    /// Opens the folder (making it if it is not there), refuses a path with a junction or a link on it, and holds it.
    pub(crate) fn open(path: &Path) -> Result<KeyDir, KeyError> {
        let absolute = std::path::absolute(path).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        refuse_links_on_the_way(&absolute, true)?;
        match fs::symlink_metadata(&absolute) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(&absolute).map_err(|e| KeyError::io("create the keys directory", e))?;
            }
            Err(e) => return Err(KeyError::io("inspect the keys directory", e)),
        }
        refuse_links_on_the_way(&absolute, false)?;
        #[cfg(test)]
        super::open_hooks::fire(&absolute, "after_walk");
        // no FILE_SHARE_DELETE: while this handle is open nobody can rename or remove the folder, or a real folder above it. Write access is what flushing the
        // folder needs.
        let held = open_folder(&absolute, FILE_SHARE_READ | FILE_SHARE_WRITE, true).map_err(|e| KeyError::io("open the keys directory", e))?;
        let shown = absolute.display().to_string();
        let info = winfs::info(&held).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&info, &shown)?;
        let root = winfs::final_path(&held).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        let found = KeyDir {
            path: absolute,
            root,
            held,
            id: info.id,
            lock_wait: LOCK_WAIT,
            #[cfg(test)]
            hooks: super::Hooks::default(),
        };
        // the path leads to what was opened, and (walked again, now that the folder is held and nothing above it can be swapped) through real folders only
        found.path_leads_here()?;
        refuse_links_on_the_way(&found.path, false)?;
        Ok(found)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Opens the path afresh and compares what it leads to with what is held: the same volume and file index, and a folder that is not a reparse point.
    fn path_leads_here(&self) -> Result<(), KeyError> {
        let shown = self.path.display().to_string();
        let by_path = open_folder(&self.path, FILE_SHARE_ALL, false).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        let info = winfs::info(&by_path).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&info, &shown)?;
        if info.id != self.id {
            return Err(KeyError::io("inspect the keys directory", io::Error::other("the keys directory was replaced or moved while it was open")));
        }
        Ok(())
    }

    /// Judged before every operation: the handle is still a folder, and the path still leads to it.
    pub(crate) fn verify(&self) -> Result<(), KeyError> {
        let shown = self.path.display().to_string();
        let info = winfs::info(&self.held).map_err(|e| KeyError::io("inspect the keys directory", e))?;
        require_directory(&info, &shown)?;
        self.path_leads_here()
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
        let info = winfs::info(&file).map_err(|e| KeyError::io("inspect the lock file", e))?;
        require_file(&info, &self.shown(LOCK_FILE), false)?;
        DirLock::acquire(file, exclusive, self.lock_wait)
    }

    /// The file as a person should read it in a message: in the folder as the caller named it.
    fn shown(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }

    /// The contents of one file of the folder, or `None` if the folder has no such entry.
    pub(crate) fn read(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>, KeyError> {
        self.fire("before_open");
        let file = match OpenOptions::new().read(true).custom_flags(FILE_FLAG_OPEN_REPARSE_POINT).open(self.at(name)) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(KeyError::io("open a key file", e)),
        };
        let info = winfs::info(&file).map_err(|e| KeyError::io("inspect a key file", e))?;
        require_file(&info, &self.shown(name), true)?;
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

    /// Removes the temporary file `name` if it is debris: a regular file, older than `min_age`, and not locked by anyone (see [`STALE_AFTER`](super::STALE_AFTER)).
    /// Best effort: whether it was removed is the answer, and no failure is an error.
    pub(crate) fn remove_if_stale(&self, name: &str, min_age: Duration) -> bool {
        let path = self.at(name);
        let Ok(file) = OpenOptions::new().read(true).write(true).custom_flags(FILE_FLAG_OPEN_REPARSE_POINT).open(&path) else { return false };
        let Ok(info) = winfs::info(&file) else { return false };
        if require_file(&info, "", false).is_err() || !claim_if_stale(&file, min_age) {
            return false;
        }
        fs::remove_file(&path).is_ok()
    }

    /// Removes a file; one that is not there is not an error.
    pub(crate) fn remove(&self, name: &str) -> Result<(), KeyError> {
        match fs::remove_file(self.at(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KeyError::io("remove a key file", e)),
        }
    }

    /// Whether the folder has an entry of any kind called `name` (a link and a directory count).
    pub(crate) fn exists(&self, name: &str) -> Result<bool, KeyError> {
        match fs::symlink_metadata(self.at(name)) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(KeyError::io("inspect the keys directory", e)),
        }
    }

    /// The entries of the folder.
    pub(crate) fn list(&self) -> Result<Vec<Entry>, KeyError> {
        let op = "list the keys directory";
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(|e| KeyError::io(op, e))? {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            let dir = std::env::temp_dir().join(format!("oaiy-keystore-kdw-{tag}-{}-{nanos}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn junction(link: &Path, target: &Path) {
        let made = std::process::Command::new("cmd").args(["/C", "mklink", "/J"]).arg(link).arg(target).output().unwrap();
        assert!(made.status.success(), "mklink /J failed: {}", String::from_utf8_lossy(&made.stdout));
    }

    fn write(dir: &KeyDir, name: &str, bytes: &[u8]) {
        dir.create_new(name).unwrap().write_all(bytes).unwrap();
    }

    /// M-1 (b) and (c): the path and the folder that is held come apart (here by hand: the store's own `path` is pointed at another folder, which is what a junction above
    /// the folder that was re-pointed would do to it). The next operation says so, and whatever it does, it does in the folder that is held, never in the one that the path
    /// leads to now: a read finds the held folder's file and not the decoy's, a write goes into the held folder, a listing is of the held folder. Without the comparison the
    /// store answers from the other tree; without the real path it reads and writes there.
    #[test]
    fn a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders() {
        let s = Scratch::new("elsewhere");
        let (held, decoy) = (s.0.join("held").join("keys"), s.0.join("decoy").join("keys"));
        fs::create_dir_all(&held).unwrap();
        fs::create_dir_all(&decoy).unwrap();
        fs::write(decoy.join("x.ks"), b"the decoy's value").unwrap();
        fs::write(decoy.join("only-in-decoy.ks"), b"decoy").unwrap();
        let mut dir = KeyDir::open(&held).unwrap();
        write(&dir, "x.ks", b"the held folder's value");
        dir.verify().unwrap();

        dir.path = decoy.clone();
        let error = dir.verify().unwrap_err();
        assert!(
            matches!(&error, KeyError::Io { op: "inspect the keys directory", source } if source.to_string().contains("replaced or moved")),
            "{error:?}"
        );
        assert_eq!(
            &**dir.read("x.ks").unwrap().unwrap(),
            b"the held folder's value",
            "a read is of the held folder, not of the tree the path leads to"
        );
        assert!(dir.read("only-in-decoy.ks").unwrap().is_none(), "and a name that only the decoy has is not found");
        write(&dir, "new.ks", b"written");
        dir.rename("new.ks", "renamed.ks").unwrap();
        assert!(
            held.join("renamed.ks").exists() && !decoy.join("renamed.ks").exists() && !decoy.join("new.ks").exists(),
            "a write is into the held folder"
        );
        let mut names: Vec<String> = dir.list().unwrap().into_iter().map(|e| e.name).collect();
        names.sort();
        assert_eq!(names, ["renamed.ks", "x.ks"], "a listing is of the held folder");
        drop(dir.lock(true).unwrap());
        assert!(held.join(LOCK_FILE).exists() && !decoy.join(LOCK_FILE).exists(), "and so is the lock");
        dir.remove("x.ks").unwrap();
        assert!(decoy.join("x.ks").exists(), "a removal leaves the decoy's file alone");
    }

    /// M-1 (a), and the window of the open: the path is walked before the folder is opened, and a junction can be put in place of a folder on it in between (here by a
    /// hook, between the walk and the open: the folder is renamed away and a junction to an attacker's tree, which has a keys folder of its own, takes its name). The
    /// open then leads to the attacker's folder, and the comparison of the path with the handle agrees with it, since both go through the junction. Only the walk that is
    /// made again once the folder is held sees it. Without that walk the store opens, and answers from the attacker's tree.
    #[test]
    fn a_junction_put_in_place_of_a_folder_between_the_walk_and_the_open_is_found_by_the_second_walk() {
        let s = Scratch::new("window");
        let path = s.0.join("p").join("keys");
        let attacker = s.0.join("attacker");
        fs::create_dir_all(attacker.join("keys")).unwrap();
        fs::write(attacker.join("keys").join("x.ks"), b"the attacker's value").unwrap();
        let (p, moved, target) = (s.0.join("p"), s.0.join("p-moved"), attacker.clone());
        super::super::open_hooks::at(&std::path::absolute(&path).unwrap(), "after_walk", move || {
            fs::rename(&p, &moved).unwrap();
            junction(&p, &target);
        });
        let error = KeyDir::open(&path).err().expect("a junction on the way is refused, however late it appears");
        assert!(matches!(&error, KeyError::Permissions(why) if why.contains("junction")), "{error:?}");
        assert_eq!(fs::read_dir(attacker.join("keys")).unwrap().count(), 1, "nothing was made in the attacker's tree");
        fs::remove_dir(s.0.join("p")).unwrap();
    }
}
