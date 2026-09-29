//! Writing a secret to disk so that nobody else on the machine can read it, not even for a moment.
//!
//! Provider API keys, the linked account's key, paired-app tokens, the relay's bearer, the chat and
//! node identity keys and the Hugging Face token are all plain files, because OAIY has no keystore
//! (that is a later task). They used to be written with `fs::write`, which creates a file with the
//! process's default permissions (0644 under the usual umask). Some of the stores narrowed the file
//! afterwards, so the secret was readable by every account on the machine for as long as that
//! took; the rest were never narrowed, and a file written once, like an identity key, stayed that
//! way for good. [`write`] makes the file private from its first byte instead, and replaces the old
//! one atomically.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, Write as _};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Write `contents` to `path`, replacing whatever is there, so that the file is private from the
/// moment it exists.
///
/// - The bytes go to a new file beside `path`, which is created owner-only in the very call that
///   creates it (unix: `open` with mode 0600). There is no moment at which it exists with the
///   default permissions, and the umask can only take permissions away from that.
/// - The staging file is unique to this call, so the GUI and a headless server sharing a data folder,
///   or two threads, cannot trample each other's. It is created with `create_new`, which refuses a
///   name that already exists, even as a symlink somebody planted.
/// - It is synced and then renamed over `path`. A reader sees the old file or the new one, never half
///   of either, and a crash leaves one of the two: an empty identity key would be a hard failure.
///   `rename` replaces an existing file on every platform std supports, so the old file is never
///   removed first (which would leave a moment with no key at all). On Windows a target that is
///   busy for the moment (another writer, an indexer or a virus scanner) is retried briefly.
/// - The staging file is removed on every failure, so the secret never lingers in a second file.
/// - A missing folder is made first, owner-only too (see [`create_private_dir`]).
///
/// # Windows
///
/// There is no mode to set there, so nothing narrows the file: it inherits the access-control list
/// of the folder it is created in. In the default data location that is a folder under the user's
/// profile, which other accounts cannot read. A data folder the user has moved somewhere else keeps
/// whatever ACL that folder has. Writing ACLs is a later task (with the keystore); this keeps the
/// behaviour the stores had on Windows and adds the atomic replace.
pub fn write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    write_with(path, contents.as_ref(), |_| {})
}

/// [`write`], calling `at_creation` with the staging file after it exists and before anything is
/// written to it: the moment a plain `fs::write` would have left it readable by everyone. The tests
/// look at its permissions there.
fn write_with(path: &Path, contents: &[u8], at_creation: impl FnOnce(&File)) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{} has no file name", path.display())))?;
    create_private_dir(dir)?;

    static STAGED: AtomicU64 = AtomicU64::new(0);
    let staging = dir.join(format!(
        ".{}.{}-{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        STAGED.fetch_add(1, Ordering::Relaxed)
    ));

    let staged = (|| {
        let mut file = create_new_owner_only(&staging)?;
        at_creation(&file);
        file.write_all(contents)?;
        file.sync_all()
    })();
    if let Err(e) = staged {
        let _ = std::fs::remove_file(&staging);
        return Err(e);
    }
    if let Err(e) = rename_over(&staging, path) {
        let _ = std::fs::remove_file(&staging);
        return Err(e);
    }
    sync_dir(dir);
    Ok(())
}

/// `rename`, which replaces `to` if it is there.
///
/// On Windows a replace is refused while another handle is in the middle of replacing the same
/// file or holds it without delete sharing: a second writer of the same file (the GUI and a
/// headless server share a data folder), a search indexer or a virus scanner that has just looked
/// at it. Those clear within moments, so it tries a few times before giving up. It does not retry
/// when a folder stands where the file belongs, which no wait would change.
#[cfg(windows)]
fn rename_over(from: &Path, to: &Path) -> io::Result<()> {
    let mut attempt: u64 = 0;
    loop {
        match std::fs::rename(from, to) {
            Err(e) if attempt < 20 && is_transient(&e) && !to.is_dir() => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(10 * attempt.min(10)));
            }
            done => return done,
        }
    }
}

/// Access denied, sharing violation or lock violation: what Windows answers while a file is busy.
#[cfg(windows)]
fn is_transient(e: &io::Error) -> bool {
    e.kind() == io::ErrorKind::PermissionDenied || matches!(e.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(not(windows))]
fn rename_over(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::rename(from, to)
}

/// Make `dir`, and any of its parents that are missing, for a secret to live in: owner-only (unix
/// mode 0700) for every folder this call creates.
///
/// A folder that already exists is left exactly as it is, whatever its permissions: the data folder
/// can be relocated by the user, and changing the permissions of a folder we did not make is not ours
/// to decide. Making the folder private only matters where we make it.
///
/// On Windows this is `create_dir_all`: the new folders inherit their parent's ACL (see [`write`]).
pub fn create_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    owner_only_dir(&mut builder);
    builder.create(dir)
}

/// Make `dir`, owner-only, and fail if it already exists (`create_dir`, not `create_dir_all`): for a
/// scratch folder in a shared place like the system temp folder, where finding the name taken means
/// somebody else made it, and putting a secret in it would be putting it in their hands.
///
/// The parent must exist. On Windows the new folder inherits its parent's ACL (see [`write`]).
pub fn create_new_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    owner_only_dir(&mut builder);
    builder.create(dir)
}

#[cfg(unix)]
fn owner_only_dir(builder: &mut DirBuilder) {
    use std::os::unix::fs::DirBuilderExt as _;
    builder.mode(0o700);
}

#[cfg(not(unix))]
fn owner_only_dir(_builder: &mut DirBuilder) {}

/// A new file that only this user can read or write, from the call that creates it.
fn create_new_owner_only(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path)
}

/// Make the rename itself durable. Best effort: a filesystem that cannot sync a folder has still
/// kept the file, and the write has succeeded either way.
#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

/// What the stores' own tests check about the files they write.
#[cfg(test)]
pub(crate) mod testing {
    use std::path::Path;

    /// The file at `path` is readable and writable by its owner only, and the folder it is in is
    /// closed to everyone else too: what every store in this crate must leave behind.
    ///
    /// Only meaningful where there are permission bits to read, so on Windows it checks that the
    /// file exists and nothing more (there the ACL is inherited; see [`super::write`]).
    pub(crate) fn assert_private(path: &Path) {
        assert!(path.is_file(), "{} should be a file", path.display());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{} is mode {mode:o}, not 600", path.display());
        }
    }

    /// The same for the folder a secret lives in, where this crate made it.
    pub(crate) fn assert_private_dir(dir: &Path) {
        assert!(dir.is_dir(), "{} should be a folder", dir.display());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(dir).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{} is mode {mode:o}, not 700", dir.display());
        }
    }

    /// A folder of its own under the system temp folder, removed when dropped.
    pub(crate) struct TempDir(pub(crate) std::path::PathBuf);

    impl TempDir {
        pub(crate) fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "oaiy-secret-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{assert_private, assert_private_dir, TempDir};
    use super::*;

    /// Everything in `dir`, by name.
    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_secret_reads_back_exactly_and_leaves_nothing_else_behind() {
        let dir = TempDir::new("roundtrip");
        let path = dir.0.join("providers.json");
        let secret: Vec<u8> = (0..=255u8).cycle().take(70_000).collect();
        write(&path, &secret).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), secret, "binary content survives, whatever its size");
        assert_eq!(names(&dir.0), ["providers.json"], "no staging file is left beside it");
        assert_private(&path);

        write(&path, "sk-second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "sk-second", "the second write replaces the first");
        assert_eq!(names(&dir.0), ["providers.json"]);
        assert_private(&path);

        write(&path, "").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"", "an empty secret is still written");
    }

    #[test]
    fn a_missing_folder_is_made_private_and_a_path_with_no_file_name_is_refused() {
        let dir = TempDir::new("mkdir");
        let path = dir.0.join("ai").join("deeper").join("providers.json");
        write(&path, "x").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "x");
        assert_private_dir(path.parent().unwrap());
        assert_private_dir(&dir.0.join("ai"));
        assert!(write(Path::new(""), "x").is_err(), "no file name, no file");
        assert!(write(Path::new("/"), "x").is_err());
    }

    /// A folder somebody made already is not ours to change: the data folder can be relocated.
    #[cfg(unix)]
    #[test]
    fn an_existing_folder_keeps_its_permissions() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("existing");
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        write(&dir.0.join("providers.json"), "x").unwrap();
        create_private_dir(&dir.0).unwrap();
        let mode = std::fs::metadata(&dir.0).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "an existing folder is left alone");
        assert_private(&dir.0.join("providers.json"));
    }

    /// The point of the module: at the first moment the file exists it is already private. A plain
    /// `fs::write` fails this under the default umask, and so would writing first and narrowing after.
    #[cfg(unix)]
    #[test]
    fn the_file_is_private_before_the_first_byte_is_written() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("window");
        let path = dir.0.join("endpoint.key");
        let mut looked = false;
        write_with(&path, b"secret", |file| {
            let meta = file.metadata().unwrap();
            assert_eq!(meta.len(), 0, "nothing has been written yet");
            let mode = meta.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "created with mode {mode:o}");
            looked = true;
        })
        .unwrap();
        assert!(looked);
        assert_eq!(std::fs::read(&path).unwrap(), b"secret");
    }

    /// A key file an older build left world-readable is replaced by a private one, not chmod-ed in place.
    #[cfg(unix)]
    #[test]
    fn replacing_a_world_readable_file_leaves_a_private_one() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("loose");
        let path = dir.0.join("providers.json");
        std::fs::write(&path, "old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        write(&path, "new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        assert_private(&path);
    }

    /// The write fails where the rename cannot happen (a folder stands where the file belongs, on
    /// every platform): the error reaches the caller and no copy of the secret is left behind.
    #[test]
    fn a_failed_write_reports_it_and_leaves_no_copy_of_the_secret() {
        let dir = TempDir::new("failed");
        let path = dir.0.join("providers.json");
        std::fs::create_dir_all(path.join("in-the-way")).unwrap();
        assert!(write(&path, "sk-secret").is_err());
        assert_eq!(names(&dir.0), ["providers.json"], "the staging file is removed");
        assert!(path.is_dir(), "and what was in the way is untouched");
    }

    /// A folder nobody can write into stops the write before any copy exists, and the old file is intact.
    #[cfg(unix)]
    #[test]
    fn an_unwritable_folder_leaves_the_old_file_untouched() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = TempDir::new("readonly");
        let path = dir.0.join("providers.json");
        write(&path, "old").unwrap();
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o500)).unwrap();
        let attempt = write(&path, "new");
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o700)).unwrap();
        // A privileged test runner writes anyway; anyone else is refused.
        if attempt.is_err() {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "old");
            assert_eq!(names(&dir.0), ["providers.json"]);
        }
    }

    /// A reader never sees half a file or an empty one while it is being replaced. The replacement
    /// on Windows can briefly refuse a reader that is opening the file at that instant, and that is
    /// not what is being tested, so failed reads are ignored: every read that succeeds must be whole.
    #[test]
    fn a_reader_sees_the_old_secret_or_the_new_one_never_a_mixture() {
        let dir = TempDir::new("atomic");
        let path = dir.0.join("providers.json");
        let a = "A".repeat(256 * 1024);
        let b = "B".repeat(256 * 1024);
        write(&path, &a).unwrap();

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (path, stop, a, b) = (path.clone(), stop.clone(), a.clone(), b.clone());
            std::thread::spawn(move || {
                let mut whole = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        assert!(text == a || text == b, "a torn or empty read: {} bytes", text.len());
                        whole += 1;
                    }
                }
                whole
            })
        };
        for round in 0..60 {
            write(&path, if round % 2 == 0 { &b } else { &a }).unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().unwrap() > 0, "the reader must have read something");
        assert_eq!(names(&dir.0), ["providers.json"]);
    }

    /// Two writers of the same file at once (the GUI and a headless server share a data folder) each
    /// stage a file of their own, so both writes complete and one of them wins whole.
    #[test]
    fn concurrent_writers_do_not_share_a_staging_file() {
        let dir = TempDir::new("concurrent");
        let path = dir.0.join("pairings.json");
        let writers: Vec<_> = (0..8)
            .map(|n| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for round in 0..25 {
                        write(&path, format!("writer {n} round {round}")).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        assert!(std::fs::read_to_string(&path).unwrap().starts_with("writer "));
        assert_eq!(names(&dir.0), ["pairings.json"]);
    }

    #[test]
    fn a_scratch_folder_must_be_new_and_is_private() {
        let dir = TempDir::new("scratch");
        let scratch = dir.0.join("oaiy-flow-run-1");
        create_new_private_dir(&scratch).unwrap();
        assert_private_dir(&scratch);
        assert!(
            create_new_private_dir(&scratch).is_err(),
            "a name that is taken is somebody else's, or a leftover: never reused"
        );
        assert!(
            create_new_private_dir(&dir.0.join("a").join("b")).is_err(),
            "and its parent must already exist"
        );
    }
}
