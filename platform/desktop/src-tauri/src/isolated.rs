//! Opt-in local qualification roots, independent of the normal desktop install.
//!
//! This is storage and process separation, not an OS sandbox for untrusted code.
//! The caller owns the root and must not replace its files while it is running.
//! Startup refuses pre-existing links/reparse points and foreign roots; the lock
//! prevents two cooperating desktop processes from sharing a root. It does not
//! defend against a malicious process with the same user's filesystem rights.

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener},
    path::{Component, Path, PathBuf},
    sync::{Mutex, OnceLock},
};

const MARKER: &str = ".oaiy-isolated.json";
const LOCK: &str = ".oaiy-isolated.lock";
const KIND: &str = "oaiy-desktop-isolated";
const VERSION: u32 = 1;
pub const REFUSAL: &str = "This operation is unavailable in isolated local qualification mode.";

static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Parse arguments after the executable name. Unrelated normal-launch arguments
/// keep their existing behavior; misspelled isolation switches fail closed.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<PathBuf>, String> {
    let mut args = args.into_iter();
    let mut root = None;
    while let Some(arg) = args.next() {
        let value = if arg == "--isolated-root" {
            Some(
                args.next()
                    .ok_or("--isolated-root requires an absolute local directory")?,
            )
        } else if let Some(text) = arg.to_str() {
            if let Some(value) = text.strip_prefix("--isolated-root=") {
                Some(OsString::from(value))
            } else if text.starts_with("--isolated") {
                return Err(format!("unknown isolation option: {text}"));
            } else {
                None
            }
        } else {
            if arg.to_string_lossy().starts_with("--isolated") {
                return Err("isolation arguments must be Unicode".into());
            }
            None
        };
        if let Some(value) = value {
            if root.is_some() {
                return Err("--isolated-root may be specified only once".into());
            }
            let path = PathBuf::from(value);
            validate_absolute(&path)?;
            root = Some(path);
        }
    }
    Ok(root)
}

/// Takes the launch as this process's own (once). Not called `install`: the update guards look for that name, and this
/// installs no package.
pub fn adopt(launch: Launch) -> Result<(), String> {
    LAUNCH
        .set(launch)
        .map_err(|_| "isolated launch is already adopted".into())
}

pub fn current() -> Option<&'static Launch> {
    LAUNCH.get()
}

pub fn active() -> bool {
    current().is_some()
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Marker {
    kind: String,
    version: u32,
    root: String,
    identifier: String,
}

pub struct Launch {
    pub root: PathBuf,
    pub config: PathBuf,
    pub data: PathBuf,
    pub models: PathBuf,
    pub profile: PathBuf,
    pub identifier: String,
    pub port: u16,
    listener: Mutex<Option<TcpListener>>,
    // A held handle, never truncated or deleted: an exclusive lock for the
    // entire process lifetime once Launch is installed in LAUNCH.
    _lock: File,
}

impl Launch {
    pub fn prepare(root: PathBuf) -> Result<Self, String> {
        validate_absolute(&root)?;
        validate_local_volume(&root)?;
        ensure_directory_chain(&root)?;
        let root = canonical_root(&root)?;
        // Resolve again after canonicalization; no ancestor may be a link, and
        // two roots must not nest and consequently share an owned subtree.
        ensure_directory_chain(&root)?;

        let marker_path = root.join(MARKER);
        let marker_exists = match fs::symlink_metadata(&marker_path) {
            Ok(meta) => {
                ordinary_file(&marker_path, &meta)?;
                true
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(format!("cannot inspect isolated root marker: {e}")),
        };
        // A failed initialization can leave only the empty lock file behind.
        // Never adopt a populated directory just because it lacks our marker.
        if !marker_exists {
            for entry in
                fs::read_dir(&root).map_err(|e| format!("cannot inspect isolated root: {e}"))?
            {
                let entry =
                    entry.map_err(|e| format!("cannot inspect isolated root entry: {e}"))?;
                if entry.file_name() != LOCK {
                    return Err(
                        "isolated root must be empty or carry its own valid isolation marker"
                            .into(),
                    );
                }
            }
        }

        let lock_path = root.join(LOCK);
        if let Ok(meta) = fs::symlink_metadata(&lock_path) {
            ordinary_file(&lock_path, &meta)?;
            if meta.len() != 0 {
                return Err("isolated root lock must be an empty regular file".into());
            }
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Excluding FILE_SHARE_DELETE prevents replacing a held lock on
            // Windows. The byte-range lock below excludes another desktop.
            options.share_mode(0x0000_0001 | 0x0000_0002);
        }
        let lock = options
            .open(&lock_path)
            .map_err(|e| format!("cannot open isolated root lock: {e}"))?;
        FileExt::try_lock_exclusive(&lock)
            .map_err(|e| format!("isolated root is already in use or cannot be locked: {e}"))?;
        ordinary_file(
            &lock_path,
            &lock
                .metadata()
                .map_err(|e| format!("cannot inspect root lock: {e}"))?,
        )?;

        let root_key = path_key(&root)?;
        let identifier = identifier_for(&root_key);
        let expected = Marker {
            kind: KIND.into(),
            version: VERSION,
            root: root_key,
            identifier: identifier.clone(),
        };
        if marker_exists {
            let marker = read_marker(&marker_path)?;
            if marker != expected {
                return Err(
                    "isolated root marker is foreign, moved, or has an unsupported version".into(),
                );
            }
            validate_tree(&root)?;
        } else {
            // Check again under the lock so another initialization cannot make
            // this process adopt data it did not validate.
            for entry in
                fs::read_dir(&root).map_err(|e| format!("cannot inspect isolated root: {e}"))?
            {
                if entry.map_err(|e| e.to_string())?.file_name() != LOCK {
                    return Err("isolated root changed during initialization".into());
                }
            }
            let bytes = serde_json::to_vec_pretty(&expected).map_err(|e| e.to_string())?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker_path)
                .map_err(|e| format!("cannot create isolated root marker: {e}"))?;
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(|e| format!("cannot persist isolated root marker: {e}"))?;
        }

        let config = root.join("config");
        let data = root.join("data");
        let models = root.join("models");
        let profile = root.join("webview");
        for dir in [&config, &data, &models, &profile] {
            ensure_child_directory(dir)?;
        }
        let listener = reserve_listener()?;
        let port = listener
            .local_addr()
            .map_err(|e| format!("cannot inspect isolated listener: {e}"))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("cannot prepare isolated listener: {e}"))?;
        Ok(Self {
            root,
            config,
            data,
            models,
            profile,
            identifier,
            port,
            listener: Mutex::new(Some(listener)),
            _lock: lock,
        })
    }

    /// Transfer the already-bound socket exactly once to the HTTP runtime.
    pub fn take_listener(&self) -> Result<TcpListener, String> {
        self.listener
            .lock()
            .map_err(|_| "isolated listener lock is poisoned")?
            .take()
            .ok_or_else(|| "isolated listener has already been taken".into())
    }

    /// Installed before the dashboard's modules. No URL or bearer credential is
    /// accepted from frontend storage: the only value is our reserved port.
    pub fn initialization_script(&self) -> String {
        format!(
            "Object.defineProperty(window, '__OAIY_DESKTOP_LAUNCH__', {{value: Object.freeze({{version: 1, isolated: true, apiPort: {}}}), writable: false, configurable: false, enumerable: false}});",
            self.port,
        )
    }
}

fn validate_absolute(path: &Path) -> Result<(), String> {
    let text = path
        .to_str()
        .ok_or("isolated root must be a Unicode local absolute path")?;
    if !path.is_absolute()
        || text.is_empty()
        || text.chars().any(char::is_control)
        || text.starts_with("//")
        || text.starts_with("\\\\")
    {
        return Err("isolated root must be an ordinary local absolute path; network and device paths are refused".into());
    }
    if path.parent().is_none() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("isolated root must be a dedicated directory without parent traversal".into());
    }
    #[cfg(windows)]
    {
        use std::path::Prefix;
        if !matches!(path.components().next(), Some(Component::Prefix(p)) if matches!(p.kind(), Prefix::Disk(_)))
        {
            return Err("isolated root must use an ordinary absolute local drive path".into());
        }
        for part in path.components().filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        }) {
            let base = part
                .split('.')
                .next()
                .unwrap_or_default()
                .to_ascii_uppercase();
            let reserved = matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                || (base.len() == 4
                    && (base.starts_with("COM") || base.starts_with("LPT"))
                    && matches!(base.as_bytes()[3], b'1'..=b'9'));
            if part.trim() != part
                || part.ends_with('.')
                || part.contains(['<', '>', ':', '"', '|', '?', '*'])
                || reserved
            {
                return Err("isolated root contains a non-ordinary Windows path component".into());
            }
        }
    }
    Ok(())
}

#[cfg(windows)]
fn validate_local_volume(path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    let drive = path
        .components()
        .next()
        .ok_or("isolated root has no drive")?
        .as_os_str();
    let mut wide: Vec<u16> = drive.encode_wide().collect();
    wide.extend(['\\' as u16, 0]);
    // SAFETY: `wide` is a live, NUL-terminated drive root; GetDriveTypeW only
    // reads it and returns a classification. No filesystem or OS setting changes.
    let kind = unsafe { windows_sys::Win32::Storage::FileSystem::GetDriveTypeW(wide.as_ptr()) };
    if matches!(kind, 2 | 3 | 6) {
        Ok(())
    } else {
        Err(
            "isolated root must be on a writable local drive; network/unknown drives are refused"
                .into(),
        )
    }
}

#[cfg(not(windows))]
fn validate_local_volume(_: &Path) -> Result<(), String> {
    // Network mounts cannot be distinguished from local directories through
    // portable std APIs. Until equivalent platform checks are qualified, do
    // not silently offer a weaker isolation guarantee on another OS.
    Err("isolated local qualification currently requires Windows local-drive validation".into())
}

fn is_link(meta: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        meta.file_type().is_symlink() || meta.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

fn ordinary_file(path: &Path, meta: &fs::Metadata) -> Result<(), String> {
    if is_link(meta) || !meta.is_file() {
        return Err(format!(
            "isolated root requires ordinary files, without links/reparse points: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.nlink() != 1 {
            return Err(format!(
                "isolated root refuses hard-linked files: {}",
                path.display()
            ));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        };
        let file = File::open(path)
            .map_err(|e| format!("cannot inspect isolated file {}: {e}", path.display()))?;
        let mut information = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
        // SAFETY: the file handle and output allocation remain live throughout
        // this read-only query. Successful Win32 calls initialize the structure.
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), information.as_mut_ptr()) }
            == 0
        {
            return Err(format!(
                "cannot inspect isolated file links: {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: the successful call immediately above initialized this value.
        let information = unsafe { information.assume_init() };
        if information.nNumberOfLinks != 1 {
            return Err(format!(
                "isolated root refuses hard-linked files: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn ensure_directory_chain(path: &Path) -> Result<(), String> {
    let mut chain = PathBuf::new();
    for component in path.components() {
        chain.push(component.as_os_str());
        // `C:` is a prefix, not an absolute directory until RootDir is added.
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        ensure_child_directory(&chain)?;
        if chain != path
            && chain
                .join(MARKER)
                .try_exists()
                .map_err(|e| format!("cannot inspect isolated ancestor: {e}"))?
        {
            return Err("isolated roots must not be nested inside another isolated root".into());
        }
    }
    Ok(())
}

fn ensure_child_directory(path: &Path) -> Result<(), String> {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path)
                .map_err(|e| format!("cannot create isolated directory {}: {e}", path.display()))?;
            fs::symlink_metadata(path)
                .map_err(|e| format!("cannot inspect isolated directory: {e}"))?
        }
        Err(e) => {
            return Err(format!(
                "cannot inspect isolated directory {}: {e}",
                path.display()
            ))
        }
    };
    if is_link(&meta) || !meta.is_dir() {
        return Err(format!(
            "isolated directory or ancestor is a link/reparse point or not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn canonical_root(path: &Path) -> Result<PathBuf, String> {
    let canonical =
        fs::canonicalize(path).map_err(|e| format!("cannot resolve isolated root: {e}"))?;
    #[cfg(windows)]
    {
        // canonicalize returns the extended-length spelling. Keep a conventional
        // drive path for WebView and the other existing desktop path consumers.
        let text = canonical.to_str().ok_or("isolated root is not Unicode")?;
        let ordinary = PathBuf::from(
            text.strip_prefix("\\\\?\\")
                .ok_or("isolated root resolved to an unexpected device path")?,
        );
        validate_absolute(&ordinary)?;
        Ok(ordinary)
    }
    #[cfg(not(windows))]
    {
        Ok(canonical)
    }
}

fn path_key(path: &Path) -> Result<String, String> {
    let text = path.to_str().ok_or("isolated root is not Unicode")?;
    #[cfg(windows)]
    {
        Ok(text.to_lowercase())
    }
    #[cfg(not(windows))]
    {
        Ok(text.to_owned())
    }
}

fn identifier_for(root: &str) -> String {
    let hash = Sha256::digest(root.as_bytes());
    let digest: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("com.oaiy.isolated.{}", &digest[..48])
}

fn read_marker(path: &Path) -> Result<Marker, String> {
    let file = File::open(path).map_err(|e| format!("cannot open isolated root marker: {e}"))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot inspect isolated root marker: {e}"))?;
    ordinary_file(path, &meta)?;
    if meta.len() > 8192 {
        return Err("isolated root marker is too large".into());
    }
    let mut bytes = Vec::new();
    file.take(8193)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read isolated root marker: {e}"))?;
    if bytes.len() > 8192 {
        return Err("isolated root marker is too large".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid isolated root marker: {e}"))
}

fn validate_tree(root: &Path) -> Result<(), String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("cannot inspect isolated tree: {e}"))? {
            let entry = entry.map_err(|e| format!("cannot inspect isolated entry: {e}"))?;
            let path = entry.path();
            let meta = fs::symlink_metadata(&path)
                .map_err(|e| format!("cannot inspect isolated entry: {e}"))?;
            if is_link(&meta) {
                return Err(format!(
                    "isolated root refuses links/reparse points: {}",
                    path.display()
                ));
            }
            if dir != root && entry.file_name() == MARKER {
                return Err("isolated root contains another isolated root".into());
            }
            if meta.is_dir() {
                pending.push(path);
            } else {
                ordinary_file(&path, &meta)?;
            }
        }
    }
    Ok(())
}

fn reserve_listener() -> Result<TcpListener, String> {
    // A rare OS-selected collision with a normal desktop/dev port is discarded
    // without connecting to it. Never probe, close-and-rebind, or fall back.
    for _ in 0..16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|e| format!("cannot reserve isolated loopback listener: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        if port >= 1024 && port != 17972 && port != 17973 {
            return Ok(listener);
        }
    }
    Err("could not reserve an independent isolated loopback port".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempRoot(PathBuf);
    impl TempRoot {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("oaiy-isolated-test-{}", uuid::Uuid::new_v4())))
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            // Only this test's random, owned root is removed.
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn normal_arguments_do_not_enable_isolation() {
        assert_eq!(parse(args(&[])).unwrap(), None);
        assert_eq!(
            parse(args(&["--hidden", "--other-existing-flag"])).unwrap(),
            None
        );
    }

    #[test]
    fn parses_both_explicit_forms_and_rejects_malformed_isolation() {
        let root = TempRoot::new();
        let text = root.0.to_str().unwrap();
        assert_eq!(
            parse(args(&["--hidden", "--isolated-root", text])).unwrap(),
            Some(root.0.clone())
        );
        assert_eq!(
            parse(args(&[&format!("--isolated-root={text}")])).unwrap(),
            Some(root.0.clone())
        );
        for bad in [
            args(&["--isolated-root"]),
            args(&["--isolated-root", "relative"]),
            args(&["--isolated-root="]),
            args(&["--isolated-rooot", text]),
            args(&["--isolated"]),
            args(&["--isolated_root", text]),
            args(&["--isolated-root", text, "--isolated-root", text]),
            args(&["--isolated-root", "\\\\server\\share\\root"]),
        ] {
            assert!(parse(bad).is_err());
        }
        assert!(validate_absolute(&root.0.join("..").join("outside")).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn root_lifetime_lock_identity_and_listener_are_independent() {
        let a = TempRoot::new();
        let b = TempRoot::new();
        let first = Launch::prepare(a.0.clone()).unwrap();
        let second = Launch::prepare(b.0.clone()).unwrap();
        assert!(Launch::prepare(a.0.clone()).is_err());
        assert_ne!(first.identifier, "com.oaiy.app");
        assert_ne!(first.identifier, second.identifier);
        assert_ne!(first.port, second.port);
        assert!(![17972, 17973].contains(&first.port));
        let socket = first.take_listener().unwrap();
        assert_eq!(socket.local_addr().unwrap().ip(), Ipv4Addr::LOCALHOST);
        assert_eq!(socket.local_addr().unwrap().port(), first.port);
        assert!(TcpListener::bind((Ipv4Addr::LOCALHOST, first.port)).is_err());
        assert!(first.take_listener().is_err());
        let identifier = first.identifier.clone();
        let canonical = first.root.clone();
        drop(socket);
        drop(first);
        let reopened = Launch::prepare(a.0.clone()).unwrap();
        assert_eq!(reopened.identifier, identifier);
        assert_eq!(reopened.root, canonical);
        for dir in [
            &reopened.config,
            &reopened.data,
            &reopened.models,
            &reopened.profile,
        ] {
            assert!(dir.is_dir());
            assert!(dir.starts_with(&reopened.root));
        }
        drop(reopened);
        let differently_cased = PathBuf::from(a.0.to_str().unwrap().to_uppercase());
        let reopened = Launch::prepare(differently_cased).unwrap();
        assert_eq!(reopened.identifier, identifier);
    }

    #[test]
    #[cfg(windows)]
    fn refuses_foreign_populated_roots_and_non_directory_components() {
        let root = TempRoot::new();
        fs::create_dir(&root.0).unwrap();
        fs::write(root.0.join("keep.txt"), "unchanged").unwrap();
        assert!(Launch::prepare(root.0.clone()).is_err());
        assert_eq!(
            fs::read_to_string(root.0.join("keep.txt")).unwrap(),
            "unchanged"
        );
        assert!(!root.0.join(LOCK).exists());
        assert!(Launch::prepare(root.0.join("keep.txt").join("child")).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn rejects_moved_unknown_and_invalid_markers_without_reinitializing() {
        for change in ["version", "root", "unknown"] {
            let root = TempRoot::new();
            drop(Launch::prepare(root.0.clone()).unwrap());
            let path = root.0.join(MARKER);
            let mut marker: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            match change {
                "version" => marker["version"] = 99.into(),
                "root" => marker["root"] = "elsewhere".into(),
                _ => marker["unrecognized"] = true.into(),
            }
            let bytes = serde_json::to_vec(&marker).unwrap();
            fs::write(&path, &bytes).unwrap();
            assert!(Launch::prepare(root.0.clone()).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    #[cfg(windows)]
    fn refuses_nested_roots_and_nonempty_foreign_lock() {
        let root = TempRoot::new();
        drop(Launch::prepare(root.0.clone()).unwrap());
        assert!(Launch::prepare(root.0.join("nested")).is_err());
        fs::write(root.0.join(LOCK), "foreign lock").unwrap();
        assert!(Launch::prepare(root.0.clone()).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn bootstrap_has_only_the_frozen_native_port_contract() {
        let root = TempRoot::new();
        let launch = Launch::prepare(root.0.clone()).unwrap();
        let script = launch.initialization_script();
        assert!(script.contains("Object.defineProperty(window, '__OAIY_DESKTOP_LAUNCH__'"));
        assert!(script.contains(&format!(
            "Object.freeze({{version: 1, isolated: true, apiPort: {}}})",
            launch.port
        )));
        assert!(script.contains("writable: false, configurable: false"));
        assert!(!script.contains("token"));
        assert!(!script.contains(root.0.to_str().unwrap()));
    }

    #[test]
    fn refuses_hard_linked_entries_without_changing_the_outside_file() {
        let root = TempRoot::new();
        let outside = TempRoot::new();
        fs::create_dir(&outside.0).unwrap();
        fs::write(outside.0.join("original"), "unchanged").unwrap();
        fs::create_dir(&root.0).unwrap();
        fs::hard_link(outside.0.join("original"), root.0.join("alias")).unwrap();
        assert!(validate_tree(&root.0).is_err());
        assert_eq!(
            fs::read_to_string(outside.0.join("original")).unwrap(),
            "unchanged"
        );
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_device_network_and_ambiguous_paths() {
        use std::os::windows::ffi::OsStringExt;
        let mut malformed: Vec<u16> = "--isolated-root=C:\\fixture\\".encode_utf16().collect();
        malformed.push(0xd800);
        assert!(parse([OsString::from_wide(&malformed)]).is_err());
        for path in [
            r"\\?\C:\isolated",
            r"\\.\C:\isolated",
            r"\\server\share\isolated",
            r"C:relative",
            r"C:\",
            r"C:\foo.",
            r"C:\foo ",
            r"C:\foo:stream",
            r"C:\CON\root",
        ] {
            assert!(validate_absolute(Path::new(path)).is_err(), "{path}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn refuses_junction_entries_and_ancestors_without_touching_the_target() {
        use std::os::windows::process::CommandExt;
        let root = TempRoot::new();
        let parent = TempRoot::new();
        let outside = TempRoot::new();
        fs::create_dir(&parent.0).unwrap();
        fs::create_dir(&outside.0).unwrap();
        fs::write(outside.0.join("keep"), "unchanged").unwrap();
        drop(Launch::prepare(root.0.clone()).unwrap());
        for link in [root.0.join("data").join("escape"), parent.0.join("escape")] {
            let cmd = PathBuf::from(std::env::var_os("SystemRoot").expect("Windows system root"))
                .join("System32")
                .join("cmd.exe");
            let result = std::process::Command::new(cmd)
                .args(["/D", "/C", "mklink", "/J"])
                .arg(&link)
                .arg(&outside.0)
                .creation_flags(0x0800_0000)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "junction creation failed: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(is_link(&fs::symlink_metadata(&link).unwrap()));
            if link.starts_with(&root.0) {
                assert!(Launch::prepare(root.0.clone()).is_err());
            } else {
                assert!(Launch::prepare(link.join("child")).is_err());
            }
            assert_eq!(
                fs::read_to_string(outside.0.join("keep")).unwrap(),
                "unchanged"
            );
            assert!(!outside.0.join("child").exists());
            fs::remove_dir(&link).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn refuses_symlink_ancestors_and_entries_on_reuse() {
        use std::os::unix::fs::symlink;
        let root = TempRoot::new();
        let outside = TempRoot::new();
        fs::create_dir(&outside.0).unwrap();
        fs::create_dir(&root.0).unwrap();
        symlink(&outside.0, root.0.join("escape")).unwrap();
        assert!(validate_tree(&root.0).is_err());
        assert!(ensure_directory_chain(&root.0.join("escape").join("child")).is_err());
    }

    #[cfg(not(windows))]
    #[test]
    fn unsupported_platform_refuses_before_creating_a_root() {
        let root = TempRoot::new();
        assert!(Launch::prepare(root.0.clone()).is_err());
        assert!(!root.0.exists());
    }
}
