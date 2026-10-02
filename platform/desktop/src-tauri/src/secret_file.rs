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
//!
//! One store is written elsewhere: the engines studio's configuration (its gateway key and the
//! Hugging Face token). `oaiy-studio` is std-only and this crate depends on it, so its
//! `config::save` does the same in a few lines of its own. A change to how a secret is written
//! here belongs there too.

use std::ffi::OsStr;
use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// How many staging names a write tries before it gives up: leftovers of killed processes are
/// rare, and a folder with this many of them in the way is not one to keep writing secrets into.
const STAGING_NAMES: u64 = 32;

/// Write `contents` to `path`, replacing whatever is there, so that the file is private from the
/// moment it exists.
///
/// - The bytes go to a new file beside `path`, which is created owner-only in the very call that
///   creates it (unix: `open` with mode 0600). There is no moment at which it exists with the
///   default permissions, and the umask can only take permissions away from that.
/// - The staging file is unique to this call, so the GUI and a headless server sharing a data folder,
///   or two threads, cannot trample each other's. It is created with `create_new`, which refuses a
///   name that already exists, even as a symlink somebody planted: such a name is never written
///   to or followed, and the next one is tried (a process killed between creating its staging file
///   and renaming it leaves one behind, and a server in a container is often pid 1 every time).
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
/// There is no mode to set there, so nothing narrows the file: it is NOT private to its owner, and no page or document may say it is. It inherits
/// the access-control list of the folder it is created in. In the default data location that is a folder under the user's profile, which other
/// ordinary accounts cannot read (the administrators and the system can). A data folder the user has moved somewhere else keeps whatever ACL that
/// folder has. Writing ACLs is a later task (with the keystore); this keeps the behaviour the stores had on Windows and adds the atomic replace.
pub fn write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    write_with(path, contents.as_ref(), next_staging_number, |_| {})
}

/// The number in the next staging file's name: this process's, counting up from 0 (so the pid and
/// the number together are what tells two writers' files apart).
fn next_staging_number() -> u64 {
    static STAGED: AtomicU64 = AtomicU64::new(0);
    STAGED.fetch_add(1, Ordering::Relaxed)
}

/// [`write`], with the staging numbers from `next`, and calling `at_creation` with the staging file
/// after it exists and before anything is written to it: the moment a plain `fs::write` would have
/// left it readable by everyone. The tests look at its permissions there, and choose the numbers so
/// that the names they plant are the ones in the way.
fn write_with(
    path: &Path,
    contents: &[u8],
    mut next: impl FnMut() -> u64,
    at_creation: impl FnOnce(&File),
) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{} has no file name", path.display())))?;
    create_private_dir(dir)?;

    let (staging, mut file) = create_staging(dir, name, &mut next)?;
    let staged = (|| {
        at_creation(&file);
        file.write_all(contents)?;
        file.sync_all()
    })();
    drop(file);
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

/// Make the staging file for a write of `name` in `dir`: `.<name>.<pid>-<number>.tmp`, new, owner-only.
///
/// A name that is taken is skipped, not reused and not followed (`create_new` refuses even a
/// symlink), and the next number is tried, up to [`STAGING_NAMES`] times. What holds the name is
/// left alone: it may be somebody else's.
fn create_staging(dir: &Path, name: &OsStr, next: &mut impl FnMut() -> u64) -> io::Result<(PathBuf, File)> {
    for _ in 0..STAGING_NAMES {
        let staging = dir.join(format!(".{}.{}-{}.tmp", name.to_string_lossy(), std::process::id(), next()));
        match create_new_owner_only(&staging) {
            Ok(file) => return Ok((staging, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!("{STAGING_NAMES} staging names for {} are taken in {}", name.to_string_lossy(), dir.display()),
    ))
}

/// `rename`, which replaces `to` if it is there.
///
/// On Windows a replace is refused while another handle is in the middle of replacing the same
/// file or holds it without delete sharing: a second writer of the same file (the GUI and a
/// headless server share a data folder), a search indexer or a virus scanner that has just looked
/// at it. Those clear within moments, so it tries a few times before giving up. It does not retry
/// when a folder stands where the file belongs, which no wait would change.
#[cfg(windows)]
pub(crate) fn rename_over(from: &Path, to: &Path) -> io::Result<()> {
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
pub(crate) fn rename_over(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::rename(from, to)
}

/// What [`read_text`] found at a path.
#[derive(Debug)]
pub enum Text {
    /// There is no file.
    Missing,
    /// The file's text: UTF-8 (a byte order mark taken off) or UTF-16 with its byte order mark.
    Text(String),
    /// The file is there and was read, but its bytes are not text (not UTF-8, and not UTF-16 with a mark). It is not
    /// the store's to write over: see [`keep_aside`].
    Undecodable(String),
    /// The file could not be read (another program holds it, or its permissions say no): what is in it is not known,
    /// so it may be perfectly good and must not be written over.
    Unreadable(io::Error),
}

/// A file of settings or messages as text. Windows PowerShell 5.1's `>` and `Out-File` write UTF-16 with a byte order
/// mark, and some editors write UTF-8 with one: both are read (a store that reads only plain UTF-8 took the owner's
/// own file for garbage, and the next write replaced it).
pub fn read_text(path: &Path) -> Text {
    match std::fs::read(path) {
        Ok(bytes) => match decode_text(&bytes) {
            Ok(text) => Text::Text(text),
            Err(why) => Text::Undecodable(why),
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => Text::Missing,
        Err(e) => Text::Unreadable(e),
    }
}

/// How a store waits for a file that another program holds for a moment (an antivirus scan, an indexer, a backup): it is read again after each of the
/// `start` pauses before the store gives up for now, and then, in memory meanwhile, after each of the `later` ones (the last for ever) until it reads.
/// `window` is how long from the first failure the messages the receptionist takes are kept in memory, to be written when the file can be read.
#[derive(Clone, Debug)]
pub struct Patience {
    pub start: Vec<std::time::Duration>,
    pub later: Vec<std::time::Duration>,
    pub window: std::time::Duration,
}

impl Patience {
    /// What the desktop waits for: under a second at the start (the stores open one after the other, and the desktop is starting: the wait is the
    /// start of the desktop's, however many are busy), then, in memory meanwhile, every 1, 2, 4, 8, 15 and then 30 seconds; messages are kept in memory
    /// for the first fifteen.
    pub fn for_use() -> Self {
        let ms = std::time::Duration::from_millis;
        Self { start: vec![ms(100), ms(200), ms(400)], later: vec![ms(1_000), ms(2_000), ms(4_000), ms(8_000), ms(15_000), ms(30_000)], window: ms(15_000) }
    }
}

impl Default for Patience {
    #[cfg(not(test))]
    fn default() -> Self {
        Self::for_use()
    }

    /// Tests do not wait, do not go back to a file on their own and do not keep messages in memory, unless one says (`Patience { .. }`).
    #[cfg(test)]
    fn default() -> Self {
        let ms = std::time::Duration::from_millis;
        Self { start: vec![ms(1)], later: vec![std::time::Duration::from_secs(3_600)], window: std::time::Duration::ZERO }
    }
}

/// When a file that stayed busy is read again: a store holds one from the first failure until it reads.
#[derive(Debug)]
pub struct Retry {
    later: Vec<std::time::Duration>,
    tries: usize,
    since: std::time::Instant,
    next: std::time::Instant,
    window: std::time::Duration,
}

impl Retry {
    /// The file was busy just now.
    pub fn began(patience: &Patience) -> Self {
        let now = std::time::Instant::now();
        let mut retry = Self { later: patience.later.clone(), tries: 0, since: now, next: now, window: patience.window };
        retry.next = now + retry.pause();
        retry
    }

    fn pause(&self) -> std::time::Duration {
        self.later.get(self.tries).or_else(|| self.later.last()).copied().unwrap_or(std::time::Duration::from_secs(30))
    }

    /// It was read again and was still busy.
    pub fn failed(&mut self) {
        self.tries += 1;
        self.next = std::time::Instant::now() + self.pause();
    }

    /// Whether it is time to read it again.
    pub fn due(&self) -> bool {
        std::time::Instant::now() >= self.next
    }

    /// Whether what is taken meanwhile is still kept in memory (the first moments after it was found busy).
    pub fn in_window(&self) -> bool {
        self.since.elapsed() < self.window
    }
}

/// [`read_text`], but a file that cannot be read (it is held by another program, for a moment) is read again after each of `pauses`: the last answer is
/// the answer.
pub fn read_text_patiently(path: &Path, pauses: &[std::time::Duration]) -> Text {
    let mut text = read_text(path);
    for pause in pauses {
        if !matches!(text, Text::Unreadable(_)) {
            break;
        }
        std::thread::sleep(*pause);
        text = read_text(path);
    }
    text
}

/// `bytes` as text: UTF-16 (little or big endian) when they start with its byte order mark, otherwise UTF-8 with a
/// leading mark taken off. Nothing is guessed and nothing is replaced: a byte sequence that is not valid is an error.
pub fn decode_text(bytes: &[u8]) -> Result<String, String> {
    if let Some(body) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return utf16(body, u16::from_le_bytes);
    }
    if let Some(body) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return utf16(body, u16::from_be_bytes);
    }
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    String::from_utf8(body.to_vec()).map_err(|e| format!("not valid UTF-8 (at byte {})", e.utf8_error().valid_up_to()))
}

fn utf16(body: &[u8], unit: fn([u8; 2]) -> u16) -> Result<String, String> {
    if body.len() % 2 != 0 {
        return Err("UTF-16 with a byte missing".into());
    }
    let units: Vec<u16> = body.chunks_exact(2).map(|pair| unit([pair[0], pair[1]])).collect();
    String::from_utf16(&units).map_err(|_| "not valid UTF-16".into())
}

/// How many names a file is put aside under (`x.corrupt`, `x.corrupt.1`, ...): when they are all taken the oldest copies are let go (see
/// [`make_room_aside`]), so that a bad file is never refused a place for want of one.
const ASIDE_NAMES: u32 = 32;
/// How many of the oldest copies are let go at once when every name is taken (the first, `x.corrupt`, never is): a batch, so that this is not done for each.
const ASIDE_LET_GO: u32 = 8;

fn aside_name(name: &str, n: u32) -> String {
    if n == 0 {
        format!("{name}.corrupt")
    } else {
        format!("{name}.corrupt.{n}")
    }
}

/// Every name is taken: let the oldest copies go, and renumber the rest down so that the numbers still rise with age and the newest is the last. The
/// first copy (`x.corrupt`) is never let go, since it is the earliest evidence of what went wrong, and the newest ones are kept. An error if one
/// that must go cannot be removed.
fn make_room_aside(path: &Path, name: &str) -> io::Result<()> {
    let at = |n: u32| path.with_file_name(aside_name(name, n));
    for n in 1..=ASIDE_LET_GO {
        match std::fs::remove_file(at(n)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    let mut next = 1;
    for n in (ASIDE_LET_GO + 1)..ASIDE_NAMES {
        if std::fs::symlink_metadata(at(n)).is_err() {
            continue;
        }
        if n != next {
            std::fs::rename(at(n), at(next))?;
        }
        next += 1;
    }
    Ok(())
}

/// Put the file at `path` aside as `<name>.corrupt` (`.corrupt.1`, `.corrupt.2` ... when that is taken: an earlier bad
/// file is never replaced by a later one), so that what is in it is kept while the store starts a new file at `path`.
/// Answers where it went. If the file cannot be moved, its bytes are copied there (the original stays, and is safe to
/// replace once a copy exists). When neither works the answer is an error, and the caller must not write to `path`: the
/// file is all that is left of what was in it.
pub fn keep_aside(path: &Path) -> io::Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{} has no file name", path.display())))?
        .to_string_lossy()
        .into_owned();
    let free = || (0..ASIDE_NAMES).map(|n| path.with_file_name(aside_name(&name, n))).find(|aside| std::fs::symlink_metadata(aside).is_err());
    // `rename` replaces what is there on every platform: a name in use is passed over, not replaced.
    // A file another program holds for a moment (a scanner, an indexer) is waited for, as `write` waits for its target.
    let aside = match free() {
        Some(aside) => aside,
        // Every name is taken: the oldest copies are let go, so that this one has a place (or, if they cannot be, it is left where it is).
        None => {
            make_room_aside(path, &name).map_err(|e| io::Error::new(e.kind(), format!("{ASIDE_NAMES} files kept aside as {name}.corrupt are there already, and the oldest could not be let go: {e}")))?;
            free().ok_or_else(|| io::Error::new(io::ErrorKind::AlreadyExists, format!("{ASIDE_NAMES} files kept aside as {name}.corrupt are there already")))?
        }
    };
    match rename_over(path, &aside) {
        Ok(()) => Ok(aside),
        Err(rename_failed) => match std::fs::copy(path, &aside) {
            Ok(_) => Ok(aside),
            Err(_) => Err(rename_failed),
        },
    }
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

/// A new file that only this user can read or write, from the call that creates it (the backup
/// writes its staged copies and its output through this, so a streamed file is private from its
/// first byte too).
pub(crate) fn create_new_owner_only(path: &Path) -> io::Result<File> {
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

    /// A file that cannot be read but can still be replaced: on Windows another program holding it open with only delete
    /// sharing (read is refused, a rename over it is not), elsewhere no read permission (a folder that can still be written
    /// to). Held until dropped. `None` where it cannot be made so (a user who reads everything).
    pub(crate) struct Unreadable {
        #[cfg(windows)]
        _held: std::fs::File,
        #[cfg(unix)]
        path: std::path::PathBuf,
    }

    #[cfg(windows)]
    pub(crate) fn make_unreadable(path: &Path) -> Option<Unreadable> {
        use std::os::windows::fs::OpenOptionsExt as _;
        let held = std::fs::OpenOptions::new().read(true).share_mode(4).open(path).ok()?; // FILE_SHARE_DELETE
        std::fs::read(path).is_err().then_some(Unreadable { _held: held })
    }

    #[cfg(unix)]
    pub(crate) fn make_unreadable(path: &Path) -> Option<Unreadable> {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o000)).ok()?;
        let guard = Unreadable { path: path.to_path_buf() };
        std::fs::read(path).is_err().then_some(guard)
    }

    #[cfg(unix)]
    impl Drop for Unreadable {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(0o600));
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

    /// The desktop does not wait on a busy file for long as it starts (three stores, one after the other, as its local server is built), and goes back
    /// to it, in memory meanwhile, ever less often.
    #[test]
    fn the_desktop_waits_under_a_second_for_a_busy_file_as_it_starts_and_goes_back_to_it_ever_less_often() {
        let secs = std::time::Duration::from_secs;
        let p = Patience::for_use();
        let at_start: std::time::Duration = p.start.iter().sum();
        assert!(!p.start.is_empty() && at_start < secs(1), "the wait at start-up: {at_start:?}");
        assert!(p.later.len() >= 4 && p.later[0] >= secs(1) && p.later.windows(2).all(|w| w[0] <= w[1]), "then ever less often: {:?}", p.later);
        assert!(p.later.last().is_some_and(|last| *last <= secs(30)), "and at least every thirty seconds: {:?}", p.later);
        assert_eq!(p.window, secs(15), "messages are kept in memory for the first fifteen seconds");
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
        write_with(&path, b"secret", next_staging_number, |file| {
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

    /// The name of the staging file this process would make for `endpoint.key` at `number`.
    fn staging_name(number: u64) -> String {
        format!(".endpoint.key.{}-{number}.tmp", std::process::id())
    }

    /// A process killed between creating its staging file and renaming it leaves the file behind,
    /// and the next run of a server that is pid 1 every time asks for the same name. That name is
    /// skipped, whatever holds it (a leftover, or a symlink somebody planted), and the write lands.
    #[test]
    fn a_staging_name_that_is_taken_is_skipped_and_what_holds_it_is_left_alone() {
        let dir = TempDir::new("taken");
        let path = dir.0.join("endpoint.key");
        std::fs::write(dir.0.join(staging_name(100)), "a leftover").unwrap();
        #[cfg(unix)]
        {
            // A planted link to a file the write must not reach.
            let victim = dir.0.join("victim");
            std::fs::write(&victim, "not to be overwritten").unwrap();
            std::os::unix::fs::symlink(&victim, dir.0.join(staging_name(101))).unwrap();
        }
        #[cfg(not(unix))]
        std::fs::write(dir.0.join(staging_name(101)), "another leftover").unwrap();

        let mut numbers = 100..;
        write_with(&path, b"the seed", || numbers.next().unwrap(), |_| {}).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"the seed");
        assert_private(&path);
        assert_eq!(std::fs::read_to_string(dir.0.join(staging_name(100))).unwrap(), "a leftover");
        #[cfg(unix)]
        assert_eq!(std::fs::read_to_string(dir.0.join("victim")).unwrap(), "not to be overwritten");
        let mut expected = vec!["endpoint.key".to_string(), staging_name(100), staging_name(101)];
        #[cfg(unix)]
        expected.push("victim".to_string());
        expected.sort();
        assert_eq!(names(&dir.0), expected, "the two plants stay as they were, and the write left nothing else");
    }

    /// The search for a free name is bounded, and giving up is an error with nothing of the
    /// secret written anywhere.
    #[test]
    fn a_folder_with_every_staging_name_taken_fails_the_write_and_writes_nothing() {
        let dir = TempDir::new("all-taken");
        let path = dir.0.join("endpoint.key");
        for number in 0..STAGING_NAMES {
            std::fs::write(dir.0.join(staging_name(number)), "a leftover").unwrap();
        }
        let mut numbers = 0..;
        let err = write_with(&path, b"the seed", || numbers.next().unwrap(), |_| {}).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists, "{err}");
        assert!(!path.exists(), "the secret was not written");
        assert_eq!(names(&dir.0).len() as u64, STAGING_NAMES, "only the plants are there");
        assert_eq!(numbers.next(), Some(STAGING_NAMES), "and no more names were tried than the bound");
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
    fn text_is_read_as_utf8_with_or_without_a_mark_and_as_utf16_with_one_and_nothing_else_is_guessed() {
        let words = "Ring me back \u{2014} caf\u{e9} \u{1f4de}";
        let le: Vec<u8> = [0xFF, 0xFE].into_iter().chain(words.encode_utf16().flat_map(u16::to_le_bytes)).collect();
        let be: Vec<u8> = [0xFE, 0xFF].into_iter().chain(words.encode_utf16().flat_map(u16::to_be_bytes)).collect();
        let marked: Vec<u8> = [0xEF, 0xBB, 0xBF].into_iter().chain(words.bytes()).collect();
        for (tag, bytes) in [("plain", words.as_bytes().to_vec()), ("utf-16 le", le), ("utf-16 be", be), ("marked utf-8", marked)] {
            assert_eq!(decode_text(&bytes).as_deref(), Ok(words), "{tag}");
        }
        assert_eq!(decode_text(b""), Ok(String::new()));
        assert!(decode_text(&[b'{', 0xC3, 0x28]).is_err(), "bytes that are not UTF-8 are refused, not replaced");
        assert!(decode_text(&[0xFF, 0xFE, 0x7B]).is_err(), "half a UTF-16 unit");
        assert!(decode_text(&[0xFF, 0xFE, 0x00, 0xD8, 0x7B, 0x00]).is_err(), "a lone surrogate");
        assert!(decode_text(&[0xFE, 0xFF, 0xD8, 0x00, 0x00, 0x7B]).is_err(), "and the same the other way round");
    }

    #[test]
    fn read_text_tells_a_missing_file_from_text_from_bytes_and_from_a_file_that_cannot_be_read() {
        let dir = TempDir::new("read-text");
        let path = dir.0.join("ring.json");
        assert!(matches!(read_text(&path), Text::Missing));
        std::fs::write(&path, "{}").unwrap();
        assert!(matches!(read_text(&path), Text::Text(t) if t == "{}"));
        std::fs::write(&path, [0xC3, 0x28]).unwrap();
        assert!(matches!(read_text(&path), Text::Undecodable(_)));
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(matches!(read_text(&path), Text::Unreadable(_)), "a folder where the file belongs is not a missing file");
    }

    #[test]
    fn a_file_put_aside_is_kept_byte_for_byte_and_no_earlier_one_is_replaced() {
        let dir = TempDir::new("aside");
        let path = dir.0.join("messages.json");
        let mut kept = Vec::new();
        for n in 0..3u8 {
            std::fs::write(&path, [0xC3, 0x28, n]).unwrap();
            let aside = keep_aside(&path).unwrap();
            assert!(!path.exists(), "the file has moved");
            kept.push(aside);
        }
        let names: Vec<String> = kept.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["messages.json.corrupt", "messages.json.corrupt.1", "messages.json.corrupt.2"]);
        for (n, aside) in kept.iter().enumerate() {
            assert_eq!(std::fs::read(aside).unwrap(), [0xC3, 0x28, n as u8]);
        }
    }

    /// A file another program holds open (a virus scanner, an indexer) cannot be moved, but it can be read: what is in it
    /// is copied, so it is kept even where the original stays.
    #[cfg(windows)]
    #[test]
    fn a_file_that_cannot_be_moved_is_copied_aside_instead() {
        use std::os::windows::fs::OpenOptionsExt as _;
        let dir = TempDir::new("aside-held");
        let path = dir.0.join("ring.json");
        std::fs::write(&path, [0xC3, 0x28, 0x29]).unwrap();
        let held = std::fs::OpenOptions::new().read(true).share_mode(1).open(&path).unwrap(); // FILE_SHARE_READ: no delete, no rename
        let moved = std::fs::rename(dir.0.join("ring.json"), dir.0.join("elsewhere.json"));
        assert!(moved.is_err(), "the test needs a file that cannot be moved");
        let aside = keep_aside(&path).unwrap();
        assert_eq!(aside.file_name().unwrap(), "ring.json.corrupt");
        assert_eq!(std::fs::read(&aside).unwrap(), [0xC3, 0x28, 0x29], "a copy is kept");
        assert!(path.exists(), "and the original is where it was");
        drop(held);
    }

    /// A scanner holds a file for a moment: it is moved aside when the hold is let go, as a write waits for its target,
    /// and not copied with the original left where it was.
    #[cfg(windows)]
    #[test]
    fn a_file_held_for_a_moment_is_moved_aside_when_it_is_let_go() {
        use std::os::windows::fs::OpenOptionsExt as _;
        let dir = TempDir::new("aside-held-briefly");
        let path = dir.0.join("account.json");
        std::fs::write(&path, [0xC3, 0x28, 0x29]).unwrap();
        let held = std::fs::OpenOptions::new().read(true).share_mode(1).open(&path).unwrap(); // FILE_SHARE_READ: no delete, no rename
        assert!(std::fs::rename(&path, dir.0.join("elsewhere.json")).is_err(), "the test needs a file that cannot be moved");
        let letting_go = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            drop(held);
        });
        let aside = keep_aside(&path).unwrap();
        letting_go.join().unwrap();
        assert_eq!(std::fs::read(&aside).unwrap(), [0xC3, 0x28, 0x29]);
        assert!(!path.exists(), "it was moved, not copied: nothing is left of it where it was");
    }

    #[test]
    fn a_file_with_every_name_to_be_put_aside_under_taken_is_left_where_it_is_when_the_oldest_cannot_be_let_go() {
        let dir = TempDir::new("aside-full");
        let path = dir.0.join("messages.json");
        std::fs::write(&path, "the only copy").unwrap();
        std::fs::create_dir(dir.0.join("messages.json.corrupt")).unwrap();
        // The names are taken by folders that are not empty: nothing can be removed to make room, and nothing is replaced.
        for n in 1..ASIDE_NAMES {
            let taken = dir.0.join(format!("messages.json.corrupt.{n}"));
            std::fs::create_dir(&taken).unwrap();
            std::fs::write(taken.join("in it"), "an earlier one").unwrap();
        }
        let err = keep_aside(&path).unwrap_err();
        assert!(err.to_string().contains("could not be let go"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "the only copy");
        assert_eq!(std::fs::read_to_string(dir.0.join("messages.json.corrupt.1").join("in it")).unwrap(), "an earlier one", "none was replaced");
        assert!(keep_aside(Path::new("/")).is_err(), "no file name, no name to keep it under");
    }

    /// A file that keeps going bad is never refused a place for want of a name: once they are all taken the oldest copies are let go, the first is
    /// kept, the newest is the last, and the numbers still rise with age.
    #[test]
    fn a_file_put_aside_again_and_again_is_never_refused_and_the_first_and_the_newest_copies_are_kept() {
        let dir = TempDir::new("aside-many");
        let path = dir.0.join("messages.json");
        let named = |n: u32| dir.0.join(aside_name("messages.json", n));
        let mut last = PathBuf::new();
        for i in 0..100 {
            std::fs::write(&path, format!("copy {i}")).unwrap();
            last = keep_aside(&path).unwrap_or_else(|e| panic!("copy {i} was refused: {e}"));
            assert!(!path.exists(), "the file has moved");
            if i + 1 <= ASIDE_NAMES as usize {
                assert!((0..=i as u32).all(|n| named(n).exists()), "nothing is let go before every name is taken (copy {i})");
            }
        }
        let kept: Vec<u32> = (0..ASIDE_NAMES).filter(|n| named(*n).exists()).collect();
        assert!(kept.len() <= ASIDE_NAMES as usize && kept.len() >= 20, "{kept:?}");
        assert_eq!(std::fs::read_to_string(named(0)).unwrap(), "copy 0", "the first is the first");
        assert_eq!(std::fs::read_to_string(&last).unwrap(), "copy 99", "and the newest is kept");
        let copies: Vec<u32> = kept.iter().map(|n| std::fs::read_to_string(named(*n)).unwrap().trim_start_matches("copy ").parse().unwrap()).collect();
        assert!(copies.windows(2).all(|w| w[0] < w[1]), "the numbers rise with age: {copies:?}");
        assert_eq!(std::fs::read_to_string(named(*kept.last().unwrap())).unwrap(), "copy 99", "the newest has the highest number");
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
