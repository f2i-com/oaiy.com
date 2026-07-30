//! Installing and removing plugins.
//!
//! Until this existed, the only way to get a plugin onto the machine was to copy
//! a folder into the plugins directory by hand — and since OAIY's entire
//! connector surface is plugin-supplied, that capped what the product could do.
//!
//! A plugin is installed from a SOURCE on this machine: either a directory
//! containing a `manifest.json`, or a `.tar.gz` of one. Nothing is fetched from
//! the network here — the operator chooses a local path, which keeps this route
//! from becoming a remote-code-install primitive.
//!
//! The staging discipline matters: the source is validated and copied to a
//! temporary directory beside the destination FIRST, and only swapped into place
//! once it is known good. A half-copied plugin directory would otherwise be
//! indistinguishable from a corrupt install.

use std::path::{Path, PathBuf};

use super::manifest::PluginManifest;

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
}

/// Read + validate the manifest at `dir/manifest.json`.
fn read_manifest(dir: &Path) -> Result<PluginManifest, String> {
    let path = dir.join("manifest.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("no readable manifest.json in {}: {e}", dir.display()))?;
    let manifest: PluginManifest =
        serde_json::from_str(&raw).map_err(|e| format!("manifest.json is not valid: {e}"))?;
    if !valid_plugin_id(&manifest.id) {
        return Err(format!(
            "manifest id {:?} is invalid — use lowercase letters, digits, dash or underscore",
            manifest.id
        ));
    }
    Ok(manifest)
}

/// The id becomes a directory name, so it must not be able to escape the plugins
/// root or collide with a shell/OS special name.
fn valid_plugin_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Reject an archive entry whose path escapes the extraction root.
fn safe_relative(path: &Path) -> bool {
    !path.is_absolute()
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
        let meta = entry
            .metadata()
            .map_err(|e| format!("cannot stat {}: {e}", from.display()))?;
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

/// Install the plugin at `source` (a directory or a `.tar.gz`) into `plugins_root`.
///
/// Returns the installed identity. The caller is responsible for stopping a
/// running instance of the same id first — this function will refuse to replace
/// a directory it cannot remove, which is what a running executable causes on
/// Windows.
pub fn install_from_path(source: &Path, plugins_root: &Path) -> Result<Installed, String> {
    if !source.exists() {
        return Err(format!("{} does not exist", source.display()));
    }
    std::fs::create_dir_all(plugins_root)
        .map_err(|e| format!("cannot create the plugins directory: {e}"))?;

    // Stage beside the destination (same volume, so the final swap is a rename).
    let staging = plugins_root.join(format!(".staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    let result = (|| -> Result<Installed, String> {
        let mut budget = MAX_TOTAL_BYTES;
        let mut entries = 0usize;
        if source.is_dir() {
            copy_tree(source, &staging, &mut budget, &mut entries)?;
        } else {
            let name = source.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !(name.ends_with(".tar.gz") || name.ends_with(".tgz")) {
                return Err("a plugin package must be a directory or a .tar.gz".into());
            }
            extract_tar_gz(source, &staging)?;
        }

        let staged = manifest_root(&staging);
        let manifest = read_manifest(&staged)?;
        let dest = plugins_root.join(&manifest.id);
        let replaced = dest.exists();
        if replaced {
            // A running plugin holds its executable open on Windows; surface that
            // as "stop it first" rather than a raw OS error.
            std::fs::remove_dir_all(&dest).map_err(|e| {
                format!(
                    "cannot replace the existing {:?} plugin — stop it first ({e})",
                    manifest.id
                )
            })?;
        }
        std::fs::rename(&staged, &dest)
            .map_err(|e| format!("cannot move the plugin into place: {e}"))?;

        Ok(Installed {
            id: manifest.id.clone(),
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            dir: dest,
            replaced,
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
        .map_err(|e| format!("cannot remove the {id:?} plugin — stop it first ({e})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("oaiy-inst-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
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
    fn installs_a_directory_and_reports_its_identity() {
        let base = tmp("dir");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");

        let out = install_from_path(&src, &root).unwrap();
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
        install_from_path(&src, &root).unwrap();
        let out = install_from_path(&src, &root).unwrap();
        assert!(out.replaced);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn refuses_a_source_with_no_manifest() {
        let base = tmp("nomanifest");
        let src = base.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("readme.txt"), b"nothing here").unwrap();
        let err = install_from_path(&src, &base.join("plugins")).unwrap_err();
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
    fn uninstall_removes_the_directory_and_refuses_unknown_ids() {
        let base = tmp("uninstall");
        let src = base.join("src");
        let root = base.join("plugins");
        write_plugin(&src, "demo");
        install_from_path(&src, &root).unwrap();
        assert!(uninstall("demo", &root).is_ok());
        assert!(!root.join("demo").exists());
        assert!(uninstall("demo", &root).is_err(), "second removal has nothing to do");
        let _ = std::fs::remove_dir_all(&base);
    }
}
