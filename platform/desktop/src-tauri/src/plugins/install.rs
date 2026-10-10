//! Installing and removing plugins.
//!
//! Until this existed, the only way to get a plugin onto the machine was to copy
//! a folder into the plugins directory by hand — and since OAIY's entire
//! connector surface is plugin-supplied, that capped what the product could do.
//!
//! A plugin is installed from a SOURCE on this machine: either a directory
//! containing a `manifest.json`, or a `.zip`/`.tar.gz` of one. Nothing is fetched from
//! the network here — the operator chooses a local path, which keeps this route
//! from becoming a remote-code-install primitive.
//!
//! The staging discipline matters: the source is validated and copied to a
//! temporary directory beside the destination FIRST, and only swapped into place
//! once it is known good. A half-copied plugin directory would otherwise be
//! indistinguishable from a corrupt install.
//!
//! A package that carries a signature is verified while it is staged (see
//! [`super::trust`]): one that does not check out is refused, and whatever was installed
//! stays as it was. A package with no signature installs, whatever the build: what it
//! may do afterwards is the trust policy's to say, and in a release build the person can
//! only trust a package that is already installed.
//!
//! Nothing here changes what a source may be. It is still a path on this machine and
//! never a URL: installing native code from the network is not something this route
//! does, signed or not.

use std::path::{Path, PathBuf};

use super::manifest::PluginManifest;
use super::trust::{PackageTrust, TrustService, TrustState};

/// Cap on an installed plugin, so a runaway archive cannot fill the disk.
const MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
/// Cap on entries, so a zip-bomb-shaped tarball cannot exhaust inodes.
const MAX_ENTRIES: usize = 20_000;

#[derive(Debug)]
pub struct Installed {
    pub id: String,
    pub name: String,
    pub version: String,
    pub dir: PathBuf,
    /// True when this replaced an existing install of the same id.
    pub replaced: bool,
    /// What the package was found to be when it was staged.
    pub trust: PackageTrust,
}

/// Read + validate the manifest at `dir/manifest.json`.
fn read_manifest(dir: &Path) -> Result<PluginManifest, String> {
    let manifest = PluginManifest::load(dir).map_err(|e| e.to_string())?;
    if !valid_plugin_id(&manifest.id) {
        return Err(format!(
            "manifest id {:?} is invalid — use lowercase letters, digits, dash or underscore",
            manifest.id
        ));
    }
    let entry = manifest.resolve_entry(dir).map_err(|e| e.to_string())?;
    if !entry.is_file() {
        return Err(format!("required plugin executable is missing: {}", manifest.entry.command));
    }
    Ok(manifest)
}

/// The id becomes a directory name, so it must not be able to escape the plugins
/// root or collide with a shell/OS special name.
pub(crate) fn valid_plugin_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Reject an archive entry whose path escapes the extraction root.
fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.to_string_lossy().contains(':')
        && !path.to_string_lossy().split(['/', '\\']).any(|part| part == "..")
        && !path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// Recursively copy `src` into `dst`, enforcing the size/entry caps.
fn copy_tree(src: &Path, dst: &Path, budget: &mut u64, entries: &mut usize) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("cannot create {}: {e}", dst.display()))?;
    let read = std::fs::read_dir(src).map_err(|e| format!("cannot read {}: {e}", src.display()))?;
    for entry in read {
        let entry = entry.map_err(|e| format!("cannot read an entry of {}: {e}", src.display()))?;
        *entries += 1;
        if *entries > MAX_ENTRIES {
            return Err("the plugin has too many files".into());
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let meta = std::fs::symlink_metadata(&from)
            .map_err(|e| format!("cannot stat {}: {e}", from.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!("plugin packages cannot contain symbolic links: {}", from.display()));
        }
        if meta.is_dir() {
            copy_tree(&from, &to, budget, entries)?;
        } else if meta.is_file() {
            let len = meta.len();
            *budget = budget
                .checked_sub(len)
                .ok_or_else(|| "the plugin exceeds the size limit".to_string())?;
            std::fs::copy(&from, &to).map_err(|e| format!("cannot copy {}: {e}", from.display()))?;
        }
        // Symlinks and other kinds are skipped deliberately: a link inside a
        // plugin package is a way to reach files outside it.
    }
    Ok(())
}

/// Unpack a `.tar.gz` into `dst`, rejecting entries that escape it.
fn extract_tar_gz(archive: &Path, dst: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive)
        .map_err(|e| format!("cannot open {}: {e}", archive.display()))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    std::fs::create_dir_all(dst).map_err(|e| format!("cannot create {}: {e}", dst.display()))?;
    let mut total: u64 = 0;
    let mut count = 0usize;
    let entries = tar
        .entries()
        .map_err(|e| format!("{} is not a readable tar.gz: {e}", archive.display()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("corrupt archive entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("unreadable archive path: {e}"))?
            .into_owned();
        if !safe_relative(&path) {
            return Err(format!("archive entry {:?} escapes the package", path.display()));
        }
        let kind = entry.header().entry_type();
        if !kind.is_file() && !kind.is_dir() {
            return Err(format!("archive entry {:?} is not a regular file or directory", path.display()));
        }
        count += 1;
        total = total.saturating_add(entry.size());
        if count > MAX_ENTRIES || total > MAX_TOTAL_BYTES {
            return Err("the archive is too large to install".into());
        }
        entry
            .unpack_in(dst)
            .map_err(|e| format!("cannot unpack {:?}: {e}", path.display()))?;
    }
    Ok(())
}

/// Aokie's published artifact is a ZIP. Apply the same containment and resource
/// limits as the tar path, and reject links rather than materializing them.
fn extract_zip(archive: &Path, dst: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("invalid plugin ZIP: {e}"))?;
    if zip.len() > MAX_ENTRIES { return Err("the archive has too many entries".into()); }
    std::fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    let mut total = 0u64;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let relative = entry.enclosed_name().ok_or("ZIP path escapes the package")?;
        if !safe_relative(&relative) || entry.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000) {
            return Err("ZIP entries must be regular files/directories inside the package".into());
        }
        total = total.checked_add(entry.size()).ok_or("the archive is too large")?;
        if total > MAX_TOTAL_BYTES { return Err("the archive is too large to install".into()); }
        let target = dst.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(target).map_err(|e| e.to_string())?;
        } else {
            std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
            let mut out = std::fs::OpenOptions::new().write(true).create_new(true)
                .open(target).map_err(|e| format!("cannot create ZIP entry: {e}"))?;
            let size = entry.size();
            let copied = std::io::copy(&mut std::io::Read::take(&mut entry, size + 1), &mut out)
                .map_err(|e| e.to_string())?;
            if copied != size { return Err("ZIP entry size differs from its declaration".into()); }
            keep_executable(&out, entry.unix_mode())?;
        }
    }
    Ok(())
}

/// A ZIP made on macOS or Linux records each file's mode, and a plugin's program is one of its files: without its
/// executable bit the plugin is installed and cannot be started. Only that bit is taken from the archive (for
/// everyone the file can be read by); nothing else of the mode is, and a ZIP made on Windows records none.
#[cfg(unix)]
fn keep_executable(file: &std::fs::File, mode: Option<u32>) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;
    if mode.is_some_and(|m| m & 0o111 != 0) {
        file.set_permissions(std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot make a ZIP entry executable: {e}"))?;
    }
    Ok(())
}

/// Windows has no executable bit: a program is one by its name.
#[cfg(not(unix))]
fn keep_executable(_file: &std::fs::File, _mode: Option<u32>) -> Result<(), String> {
    Ok(())
}

/// The directory that actually holds the manifest: either `root` itself, or a
/// single top-level folder inside it (the usual shape of a packaged archive).
fn manifest_root(root: &Path) -> PathBuf {
    if root.join("manifest.json").is_file() {
        return root.to_path_buf();
    }
    if let Ok(read) = std::fs::read_dir(root) {
        let dirs: Vec<PathBuf> = read
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        if dirs.len() == 1 && dirs[0].join("manifest.json").is_file() {
            return dirs[0].clone();
        }
    }
    root.to_path_buf()
}

/// Plugin installs running in this process now.
static INSTALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts an install while it runs, however it ends.
struct InstallGuard;

impl InstallGuard {
    fn enter() -> InstallGuard {
        INSTALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        InstallGuard
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        INSTALLS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A plugin is being installed now: an update does not restart the app in the middle of one.
pub fn in_progress() -> bool {
    INSTALLS.load(std::sync::atomic::Ordering::SeqCst) > 0
}

/// Read just the plugin id from a source, WITHOUT installing it.
///
/// The caller needs this before copying: replacing a plugin means removing the
/// old directory, and on Windows a running plugin pins its own executable — so
/// the host has to stop that id first. Only the directory case is cheap enough
/// to peek; for an archive the caller simply skips the pre-stop and gets a clear
/// "stop it first" error if the plugin is running.
pub fn peek_id(source: &Path) -> Result<String, String> {
    if !source.is_dir() {
        return Err("not a directory".into());
    }
    read_manifest(&manifest_root(source)).map(|m| m.id)
}

/// Install a directory, `.zip` or `.tar.gz` into `plugins_root`.
///
/// Returns the installed identity. The caller is responsible for stopping a
/// running instance of the same id first — this function will refuse to replace
/// a directory it cannot remove, which is what a running executable causes on
/// Windows.
///
/// A signed package that does not verify under `trust` is an error, not an install.
pub fn install_from_path(source: &Path, plugins_root: &Path, trust: &TrustService) -> Result<Installed, String> {
    let _installing = InstallGuard::enter();
    if !source.exists() {
        return Err(format!("{} does not exist", source.display()));
    }
    std::fs::create_dir_all(plugins_root)
        .map_err(|e| format!("cannot create the plugins directory: {e}"))?;

    // Stage beside the destination (same volume, so the final swap is a rename).
    let staging = plugins_root.join(format!(".staging-{}", uuid::Uuid::new_v4()));
    // Serialize installs sharing a destination root. Unique staging paths alone
    // do not prevent two replacements from moving each other's live bundle.
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true)
        .open(plugins_root.join(".install.lock")).map_err(|e| e.to_string())?;
    fs2::FileExt::try_lock_exclusive(&lock)
        .map_err(|_| "another plugin installation is in progress".to_string())?;

    let result = (|| -> Result<Installed, String> {
        let mut budget = MAX_TOTAL_BYTES;
        let mut entries = 0usize;
        if source.is_dir() {
            copy_tree(source, &staging, &mut budget, &mut entries)?;
        } else {
            let name = source.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".zip") {
                extract_zip(source, &staging)?;
            } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
                extract_tar_gz(source, &staging)?;
            } else {
                return Err("a plugin package must be a directory, .zip or .tar.gz".into());
            }
        }

        let staged = manifest_root(&staging);
        let manifest = read_manifest(&staged)?;
        // Before anything is moved: a package that fails its signature must not replace
        // a working install, or become one.
        let verdict = trust.assess_staged(&staged, &manifest.id);
        if verdict.state == TrustState::Quarantined {
            return Err(format!(
                "{} was not installed: {}",
                manifest.id,
                verdict.reason.as_deref().unwrap_or("its package failed verification")
            ));
        }
        let dest = plugins_root.join(&manifest.id);
        let replaced = dest.exists();
        let backup = plugins_root.join(format!(".backup-{}-{}", manifest.id, uuid::Uuid::new_v4()));
        if replaced {
            // Preserve every old byte until the validated replacement is live.
            std::fs::rename(&dest, &backup).map_err(|e| {
                format!(
                    "cannot replace the existing {:?} plugin — stop it first ({e})",
                    manifest.id
                )
            })?;
        }
        if let Err(e) = std::fs::rename(&staged, &dest) {
            if replaced {
                std::fs::rename(&backup, &dest).map_err(|rollback| format!(
                    "cannot activate plugin ({e}); cannot restore old bundle ({rollback}); preserved at {}", backup.display()
                ))?;
            }
            return Err(format!("cannot move the plugin into place: {e}; previous bundle preserved"));
        }
        if replaced {
            if let Err(e) = std::fs::remove_dir_all(&backup) {
                log::warn!("plugin installed, but old bundle remains at {}: {e}", backup.display());
            }
        }

        Ok(Installed {
            id: manifest.id.clone(),
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            dir: dest,
            replaced,
            trust: verdict,
        })
    })();

    let _ = std::fs::remove_dir_all(&staging);
    result
}

/// Remove an installed plugin's directory. The caller stops it first.
pub fn uninstall(id: &str, plugins_root: &Path) -> Result<(), String> {
    if !valid_plugin_id(id) {
        return Err(format!("invalid plugin id {id:?}"));
    }
    let dir = plugins_root.join(id);
    if !dir.exists() {
        return Err(format!("no plugin {id:?} is installed"));
    }
    std::fs::remove_dir_all(&dir)
        .map_err(|e| format!("cannot remove the {id:?} plugin — stop it first ({e})"))?;
    // The writable dir lives outside the bundle now, so removing the bundle no
    // longer takes it along. Delete it explicitly: a plugin's state outliving
    // its uninstall surprises people, and Aokie's includes phone pairing keys.
    let data = super::runner::plugin_data_dir(&dir);
    if data.exists() {
        std::fs::remove_dir_all(&data)
            .map_err(|e| format!("removed the {id:?} plugin but not its data at {} ({e})", data.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::trust::tests::{fill, TestKey};
    use crate::plugins::trust::{Publishers, TrustPolicy};

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("oaiy-inst-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The trust of a developer's build, which asks nothing of the plugins these tests
    /// install (they are unsigned).
    fn dev_trust(base: &Path) -> std::sync::Arc<TrustService> {
        trust_with(base, TrustPolicy::developer(), Publishers::default())
    }

    fn trust_with(base: &Path, policy: TrustPolicy, publishers: Publishers) -> std::sync::Arc<TrustService> {
        TrustService::new(policy, publishers, base.join("trusted-plugins.json"))
    }

    fn write_plugin(dir: &Path, id: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            format!(
                r#"{{"schemaVersion":1,"id":"{id}","name":"Test","version":"1.0.0",
                     "pluginApiVersion":1,"entry":{{"kind":"process","command":"x.exe"}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(dir.join("x.exe"), b"binary").unwrap();
    }

    #[test]
    fn an_install_in_progress_is_visible_to_an_update_and_however_it_ends_it_stops_counting() {
        // (Other tests install too, so only what a held guard guarantees is asserted.)
        {
            let _running = InstallGuard::enter();
            assert!(in_progress());
            let _second = InstallGuard::enter();
            drop(_second);
            assert!(in_progress(), "one install ending does not hide another");
        }
        // The real function holds the guard on every path, including an early error.
        let base = tmp("guard");
        let trust = dev_trust(&base);
        assert!(install_from_path(&base.join("missing"), &base.join("plugins"), &trust).is_err());
    }

    #[test]
    fn installs_a_directory_and_reports_its_identity() {
        let base = tmp("dir");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");

        let out = install_from_path(&src, &root, &dev_trust(&base)).unwrap();
        assert_eq!(out.id, "demo");
        assert!(!out.replaced);
        assert!(root.join("demo").join("manifest.json").is_file());
        assert!(root.join("demo").join("x.exe").is_file(), "payload copied too");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn reinstalling_replaces_and_says_so() {
        let base = tmp("replace");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        install_from_path(&src, &root, &dev_trust(&base)).unwrap();
        let out = install_from_path(&src, &root, &dev_trust(&base)).unwrap();
        assert!(out.replaced);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn installs_the_published_zip_shape_and_rejects_zip_traversal() {
        use std::io::Write;
        let base = tmp("zip");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        let archive = base.join("plugin.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        for name in ["manifest.json", "x.exe"] {
            zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(&std::fs::read(src.join(name)).unwrap()).unwrap();
        }
        zip.finish().unwrap();
        assert_eq!(install_from_path(&archive, &root, &dev_trust(&base)).unwrap().id, "demo");
        let original = std::fs::read(root.join("demo/manifest.json")).unwrap();

        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        zip.start_file("../outside.txt", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(b"outside").unwrap();
        zip.finish().unwrap();
        assert!(install_from_path(&archive, &root, &dev_trust(&base)).is_err());
        assert!(!root.join("outside.txt").exists());
        assert_eq!(std::fs::read(root.join("demo/manifest.json")).unwrap(), original);
        let _ = std::fs::remove_dir_all(base);
    }

    /// A plugin for macOS or Linux comes as a ZIP too, and its program has to be one after the install.
    #[cfg(unix)]
    #[test]
    fn a_zip_keeps_its_programs_executable_and_nothing_else_of_their_modes() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt as _;
        let base = tmp("zip-modes");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        let archive = base.join("plugin.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        // The program as a Unix zip records it, and a file whose mode asks for more than a plugin is given
        // (set-user-id, writable by everyone).
        for (name, mode) in [("manifest.json", 0o4666), ("x.exe", 0o755)] {
            zip.start_file(name, zip::write::SimpleFileOptions::default().unix_permissions(mode)).unwrap();
            zip.write_all(&std::fs::read(src.join(name)).unwrap()).unwrap();
        }
        zip.finish().unwrap();
        assert_eq!(install_from_path(&archive, &root, &dev_trust(&base)).unwrap().id, "demo");
        let mode = |name: &str| std::fs::metadata(root.join("demo").join(name)).unwrap().permissions().mode() & 0o7777;
        assert_eq!(mode("x.exe") & 0o111, 0o111, "the program can be run");
        assert_eq!(mode("manifest.json") & 0o111, 0, "a file that is not a program is not made one");
        assert_eq!(mode("manifest.json") & 0o7022, 0, "and no other bit of the archive's mode is taken");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn invalid_replacements_preserve_the_working_bundle() {
        let base = tmp("invalid-replace");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        install_from_path(&src, &root, &dev_trust(&base)).unwrap();
        let original = std::fs::read(root.join("demo/manifest.json")).unwrap();
        let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
        manifest["serviceDefinitions"] = serde_json::json!([{"definitionFile":"definitions/phone.json"}]);
        std::fs::write(src.join("manifest.json"), manifest.to_string()).unwrap();
        let err = install_from_path(&src, &root, &dev_trust(&base)).unwrap_err();
        assert!(err.contains("service definition"), "{err}");
        assert_eq!(std::fs::read(root.join("demo/manifest.json")).unwrap(), original);
        assert_eq!(std::fs::read(root.join("demo/x.exe")).unwrap(), b"binary");
        manifest.as_object_mut().unwrap().remove("serviceDefinitions");
        manifest["pluginApiVersion"] = serde_json::json!(999);
        std::fs::write(src.join("manifest.json"), manifest.to_string()).unwrap();
        assert!(install_from_path(&src, &root, &dev_trust(&base)).is_err());
        assert_eq!(std::fs::read(root.join("demo/manifest.json")).unwrap(), original);
        write_plugin(&src, "demo");
        std::fs::remove_file(src.join("x.exe")).unwrap();
        assert!(install_from_path(&src, &root, &dev_trust(&base)).unwrap_err().contains("executable"));
        assert_eq!(std::fs::read(root.join("demo/manifest.json")).unwrap(), original);
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn refuses_a_source_with_no_manifest() {
        let base = tmp("nomanifest");
        let src = base.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("readme.txt"), b"nothing here").unwrap();
        let err = install_from_path(&src, &base.join("plugins"), &dev_trust(&base)).unwrap_err();
        assert!(err.contains("manifest.json"), "{err}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn refuses_an_id_that_could_escape_the_plugins_root() {
        assert!(!valid_plugin_id("../evil"));
        assert!(!valid_plugin_id("a/b"));
        assert!(!valid_plugin_id(""));
        assert!(valid_plugin_id("aokie"));
        assert!(valid_plugin_id("my_plugin-2"));
    }

    #[test]
    fn archive_entries_must_stay_inside_the_package() {
        assert!(!safe_relative(Path::new("../outside")));
        assert!(!safe_relative(Path::new("/etc/passwd")));
        assert!(safe_relative(Path::new("ui/app.js")));
    }

    #[test]
    fn uninstall_takes_the_plugins_data_with_it() {
        // The data dir moved OUT of the bundle so the package signature can
        // verify, which means deleting the bundle no longer removes it as a
        // side effect. Aokie's state includes phone pairing keys, so this is
        // the property that had to be kept explicitly.
        let base = tmp("uninstall-data");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        install_from_path(&src, &root, &dev_trust(&base)).unwrap();

        let data = super::super::runner::plugin_data_dir(&root.join("demo"));
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("settings.json"), b"pairing-keys").unwrap();

        uninstall("demo", &root).unwrap();
        assert!(!data.exists(), "{} outlived its uninstall", data.display());
    }

    #[test]
    fn uninstall_removes_the_directory_and_refuses_unknown_ids() {
        let base = tmp("uninstall");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        install_from_path(&src, &root, &dev_trust(&base)).unwrap();
        assert!(uninstall("demo", &root).is_ok());
        assert!(!root.join("demo").exists());
        assert!(uninstall("demo", &root).is_err(), "second removal has nothing to do");
        let _ = std::fs::remove_dir_all(&base);
    }

    // --- package trust ------------------------------------------------------

    /// A signed `demo` package in `src`, and a release build's trust that pins its key.
    fn signed_source(base: &Path) -> (PathBuf, TestKey, std::sync::Arc<TrustService>) {
        let src = base.join("src");
        std::fs::create_dir_all(&src).unwrap();
        fill(&src);
        let key = TestKey::generate("test-key-1");
        key.sign(&src, "demo-plugin", "1.0.0");
        let trust = trust_with(base, TrustPolicy::release(), key.pinned_for("Demo Co", &["demo"]));
        (src, key, trust)
    }

    #[test]
    fn a_signed_package_is_verified_as_it_is_installed() {
        let base = tmp("signed");
        let (src, _key, trust) = signed_source(&base);
        let out = install_from_path(&src, &base.join("plugins"), &trust).unwrap();
        assert_eq!(out.trust.state, crate::plugins::trust::TrustState::Verified, "{:?}", out.trust);
        assert_eq!(out.trust.publisher.as_deref(), Some("Demo Co"));
        // What was copied is what was signed, so the installed folder verifies too.
        let installed = trust.assess_fresh(&base.join("plugins").join("demo"), "demo");
        assert_eq!(installed.state, crate::plugins::trust::TrustState::Verified, "{installed:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_zip_of_a_signed_bundle_verifies_once_unpacked() {
        // Aokie publishes a zip with the files at its root and the envelope beside them.
        use std::io::Write;
        let base = tmp("signed-zip");
        let (src, _key, trust) = signed_source(&base);
        let archive = base.join("aokie-plugin-windows-v1.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        for name in ["manifest.json", "demo-plugin.exe", "ui/index.html", "package-manifest.json"] {
            zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(&std::fs::read(src.join(name)).unwrap()).unwrap();
        }
        zip.finish().unwrap();
        let out = install_from_path(&archive, &base.join("plugins"), &trust).unwrap();
        assert_eq!(out.trust.state, crate::plugins::trust::TrustState::Verified, "{:?}", out.trust);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_signed_package_that_fails_is_not_installed_and_the_working_install_stays() {
        let base = tmp("signed-tampered");
        let (src, key, trust) = signed_source(&base);
        let root = base.join("plugins");
        install_from_path(&src, &root, &trust).unwrap();
        let installed = std::fs::read(root.join("demo").join("demo-plugin.exe")).unwrap();

        // The next release arrives with a file changed after it was signed.
        std::fs::write(src.join("demo-plugin.exe"), b"tampered in transit").unwrap();
        let err = install_from_path(&src, &root, &trust).unwrap_err();
        assert!(err.contains("demo was not installed"), "{err}");
        assert!(err.contains("digest mismatch: demo-plugin.exe"), "{err}");
        assert_eq!(std::fs::read(root.join("demo").join("demo-plugin.exe")).unwrap(), installed, "the working install is untouched");

        // A file added to the package that the signature does not list.
        key.sign(&src, "demo-plugin", "1.0.1");
        std::fs::write(src.join("evil.dll"), b"hijack").unwrap();
        let err = install_from_path(&src, &root, &trust).unwrap_err();
        assert!(err.contains("unlisted executable present: evil.dll"), "{err}");

        // Nothing is left behind, staged or backed up.
        let leftovers: Vec<String> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with('.') && n != ".install.lock")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_package_signed_by_a_key_that_is_not_pinned_is_not_installed() {
        let base = tmp("signed-stranger");
        let (src, _key, _trust) = signed_source(&base);
        let trust = trust_with(&base, TrustPolicy::developer(), TestKey::generate("test-key-1").pinned_for("Someone Else", &["demo"]));
        let err = install_from_path(&src, &base.join("plugins"), &trust).unwrap_err();
        assert!(err.contains("signature does not match"), "even a developer build does not install a package whose signature fails: {err}");
        assert!(!base.join("plugins").join("demo").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_unsigned_package_installs_in_any_build_and_says_what_the_build_will_do_with_it() {
        let base = tmp("unsigned");
        let src = base.join("src");
        write_plugin(&src, "demo");

        let release = trust_with(&base, TrustPolicy::release(), Publishers::default());
        let out = install_from_path(&src, &base.join("plugins"), &release).unwrap();
        assert_eq!(out.trust.state, crate::plugins::trust::TrustState::Unsigned, "so the person can trust it, in a release build");

        let out = install_from_path(&src, &base.join("plugins"), &dev_trust(&base)).unwrap();
        assert_eq!(out.trust.state, crate::plugins::trust::TrustState::UnsignedDev);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_url_is_still_not_a_source() {
        // Installing native code from the network is not something this route does, and
        // the trust check changes nothing about that: a signed package has to be on this
        // machine first.
        let base = tmp("url");
        let trust = dev_trust(&base);
        for source in ["https://example.com/plugin.zip", "http://example.com/plugin", "file:///C:/plugin.zip", "//server/share/plugin.zip"] {
            let err = install_from_path(Path::new(source), &base.join("plugins"), &trust).unwrap_err();
            assert!(err.contains("does not exist"), "{source}: {err}");
        }
        assert!(!base.join("plugins").exists(), "nothing was even started");
        let _ = std::fs::remove_dir_all(&base);
    }
}
