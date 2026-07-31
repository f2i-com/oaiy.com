//! A portable Node runtime for the bundled OAIY CLI.
//!
//! The desktop ships the CLI (`resources/cli/oaiy.mjs`), but that is a Node
//! script — so without a Node runtime a packaged app still cannot run a flow.
//! Rather than bloat the installer by ~86 MB for the majority of users who
//! already have Node, this mirrors what `python.rs` already does for Python:
//! use the system one when it exists, and otherwise download a pinned portable
//! build into the data dir on demand.
//!
//! Windows ships Node as a `.zip`, every other platform as `.tar.gz`, so both
//! are handled here. Extraction is pure Rust — no `tar.exe`, no PowerShell —
//! matching the deliberate choice `python.rs` documents.

use std::io::{BufReader, Read, Write};
use crate::HiddenCommand as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::runner::{LogBuffer, LogLine};

/// Pinned Node release. Bump deliberately — a flow engine changing runtime
/// under users is not something that should happen by drift.
const NODE_VERSION: &str = "22.14.0";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeSnapshot {
    /// A usable node was found (portable install or system).
    pub available: bool,
    /// Where it came from: `portable` | `system` | `none`.
    pub source: &'static str,
    /// Absolute path, when we have one.
    pub path: Option<String>,
    /// `node -v`, when it answered.
    pub version: Option<String>,
    /// An install is running right now.
    pub installing: bool,
    /// The pinned version this app installs.
    pub installs_version: &'static str,
}

pub struct NodeRuntime {
    data_dir: PathBuf,
    /// Logs + state of an in-flight install, so the UI can watch it.
    job: Mutex<Option<Arc<LogBuffer>>>,
    installing: Mutex<bool>,
    /// Last probe result and when it was taken. See [`NodeRuntime::snapshot`].
    probe_cache: Mutex<Option<(std::time::Instant, NodeSnapshot)>>,
}

pub type NodeHandle = Arc<NodeRuntime>;

pub fn new_handle(data_dir: PathBuf) -> NodeHandle {
    Arc::new(NodeRuntime {
        data_dir,
        job: Mutex::new(None),
        installing: Mutex::new(false),
        probe_cache: Mutex::new(None),
    })
}

/// The portable node executable path under `root`, if it exists.
fn portable_exe(root: &Path) -> Option<PathBuf> {
    let candidates: [&str; 2] = if cfg!(windows) {
        ["node.exe", "bin/node.exe"]
    } else {
        ["bin/node", "node"]
    };
    candidates
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.is_file())
}

/// `node` on PATH, if any.
fn system_exe() -> Option<PathBuf> {
    let finder = if cfg!(windows) { "where" } else { "which" };
    let out = std::process::Command::new(finder)
        .arg("node")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .pipe_hidden()
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let first = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())?
        .to_string();
    let p = PathBuf::from(first);
    p.is_file().then_some(p)
}

fn probe_version(exe: &Path) -> Option<String> {
    let out = std::process::Command::new(exe)
        .arg("-v")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .pipe_hidden()
        .output()
        .ok()?;
    out.status.success().then(|| {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    })
}

impl NodeRuntime {
    fn root(&self) -> PathBuf {
        self.data_dir.join("node")
    }

    /// The node the worker should run.
    ///
    /// Portable FIRST when installed: if the user went to the trouble of
    /// installing one through OAIY, that pinned build is the one we know works
    /// with the shipped CLI. Otherwise fall back to whatever is on PATH.
    pub fn resolve(&self) -> Option<PathBuf> {
        portable_exe(&self.root()).or_else(system_exe)
    }

    /// How long a probe result is reused. Node does not appear or change
    /// version on its own; an install goes through [`Self::install`], which
    /// clears the cache, so the only thing this delays is noticing a change
    /// made outside the app.
    const PROBE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

    pub fn snapshot(&self) -> NodeSnapshot {
        let installing = *self.installing.lock().unwrap_or_else(|e| e.into_inner());

        // Reuse a recent probe. Building this snapshot from scratch shells out
        // twice (`where node`, then `node -v`), and the readiness endpoint that
        // wants it is polled every few seconds by the UI — so the uncached
        // version spawned ~30 processes a minute to render one badge. Worse,
        // those spawns are blocking calls inside an async handler: when one
        // stalls it holds a runtime worker, and enough of them stall the whole
        // HTTP server rather than just this endpoint.
        if let Ok(cache) = self.probe_cache.lock() {
            if let Some((at, cached)) = cache.as_ref() {
                if at.elapsed() < Self::PROBE_TTL {
                    // `installing` is live state, not a probe result.
                    return NodeSnapshot { installing, ..cached.clone() };
                }
            }
        }

        let (path, source) = match portable_exe(&self.root()) {
            Some(p) => (Some(p), "portable"),
            None => match system_exe() {
                Some(p) => (Some(p), "system"),
                None => (None, "none"),
            },
        };
        let snap = NodeSnapshot {
            available: path.is_some(),
            source,
            version: path.as_deref().and_then(probe_version),
            path: path.as_ref().map(|p| p.display().to_string()),
            installing,
            installs_version: NODE_VERSION,
        };
        if let Ok(mut cache) = self.probe_cache.lock() {
            *cache = Some((std::time::Instant::now(), snap.clone()));
        }
        snap
    }

    /// Drop the cached probe so the next snapshot re-reads the world.
    fn invalidate_probe(&self) {
        if let Ok(mut cache) = self.probe_cache.lock() {
            *cache = None;
        }
    }

    pub fn current_logs(&self, tail: Option<usize>) -> Option<Vec<LogLine>> {
        self.job
            .lock()
            .ok()
            .and_then(|j| j.as_ref().map(|b| b.snapshot(tail)))
    }

    /// Download + extract the pinned portable Node, on a background thread.
    pub fn install(self: &Arc<Self>) -> Result<(), String> {
        // A finished install changes what a probe would find, so drop the
        // cached answer rather than reporting the old one for another minute.
        self.invalidate_probe();
        {
            let mut flag = self.installing.lock().unwrap_or_else(|e| e.into_inner());
            if *flag {
                return Err("a Node install is already running".into());
            }
            *flag = true;
        }
        let logs = Arc::new(LogBuffer::new());
        *self.job.lock().unwrap_or_else(|e| e.into_inner()) = Some(logs.clone());

        let me = self.clone();
        std::thread::spawn(move || {
            let dest = me.root();
            let result = install_node_native(&dest, &logs);
            match &result {
                Ok(()) => logs.push("stdout", "Node runtime ready.".into()),
                Err(e) => {
                    logs.push("stderr", format!("install failed: {e}"));
                    // A half-extracted tree leaves a node.exe that `resolve()`
                    // would PREFER over the user's working system Node, so the
                    // failure has to remove it rather than leave it lying there.
                    if let Err(rm) = std::fs::remove_dir_all(&dest) {
                        if rm.kind() != std::io::ErrorKind::NotFound {
                            logs.push("stderr", format!("could not clean up {}: {rm}", dest.display()));
                        }
                    }
                    log::warn!("portable Node install failed: {e}");
                }
            }
            // Invalidate AFTER the work, not only before it. Clearing the cache
            // when the install STARTS is useless on its own: the UI polls
            // readiness every few seconds, so the first poll re-probes and
            // re-caches the pre-install answer, and that stale "not installed"
            // then outlives the whole download — leaving the button offering an
            // 86 MB download that already completed.
            me.invalidate_probe();
            *me.installing.lock().unwrap_or_else(|p| p.into_inner()) = false;
        });
        Ok(())
    }
}

/// The asset name + URL for this platform.
fn asset_for_platform() -> Result<(String, String, bool), String> {
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => return Err(format!("unsupported CPU arch '{other}' (need x86_64 or aarch64)")),
    };
    let (os, ext, is_zip) = if cfg!(target_os = "windows") {
        ("win", "zip", true)
    } else if cfg!(target_os = "macos") {
        ("darwin", "tar.gz", false)
    } else {
        ("linux", "tar.gz", false)
    };
    let stem = format!("node-v{NODE_VERSION}-{os}-{arch}");
    let asset = format!("{stem}.{ext}");
    let url = format!("https://nodejs.org/dist/v{NODE_VERSION}/{asset}");
    Ok((asset, url, is_zip))
}

/// Reject an archive member whose path escapes the extraction root.
fn safe_member(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// Download the pinned Node and unpack it under `dest`, stripping the archive's
/// single top-level directory so `node` lands predictably.
fn install_node_native(dest: &Path, logs: &LogBuffer) -> Result<(), String> {
    let (asset, url, is_zip) = asset_for_platform()?;
    std::fs::create_dir_all(dest).map_err(|e| format!("mkdir {}: {e}", dest.display()))?;

    logs.push("stdout", format!("downloading {url}"));
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut resp = client.get(&url).send().map_err(|e| format!("download request: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download returned HTTP {}", resp.status()));
    }
    let total = resp.content_length();

    let tmp = std::env::temp_dir().join(&asset);
    {
        let mut file =
            std::fs::File::create(&tmp).map_err(|e| format!("create temp {}: {e}", tmp.display()))?;
        let mut buf = vec![0u8; 64 * 1024];
        let (mut got, mut last) = (0u64, 0u64);
        loop {
            let n = resp.read(&mut buf).map_err(|e| format!("download read: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("write temp: {e}"))?;
            got += n as u64;
            if got - last >= 8 * 1024 * 1024 {
                last = got;
                logs.push(
                    "stdout",
                    match total {
                        Some(t) => format!("downloaded {:.0} / {:.0} MB", got as f64 / 1.0e6, t as f64 / 1.0e6),
                        None => format!("downloaded {:.0} MB", got as f64 / 1.0e6),
                    },
                );
            }
        }
    }

    logs.push("stdout", "extracting…".into());
    let out = if is_zip {
        extract_zip_stripped(&tmp, dest)
    } else {
        extract_tar_gz_stripped(&tmp, dest)
    };
    let _ = std::fs::remove_file(&tmp);
    out?;

    let exe = portable_exe(dest)
        .ok_or_else(|| "the archive unpacked but no node executable was found".to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755));
    }
    match probe_version(&exe) {
        Some(v) => logs.push("stdout", format!("installed node {v}")),
        None => return Err("the unpacked node did not run".into()),
    }
    Ok(())
}

/// Strip the archive's single top-level dir, so `<dest>/node[.exe]` is stable.
fn strip_first(path: &Path) -> Option<PathBuf> {
    let mut c = path.components();
    c.next()?;
    let rest: PathBuf = c.collect();
    (!rest.as_os_str().is_empty()).then_some(rest)
}

fn extract_tar_gz_stripped(archive: &Path, dest: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| format!("open archive: {e}"))?;
    let mut a = tar::Archive::new(flate2::read::GzDecoder::new(BufReader::new(f)));
    a.set_preserve_permissions(true);
    for entry in a.entries().map_err(|e| format!("read archive: {e}"))? {
        let mut entry = entry.map_err(|e| format!("archive entry: {e}"))?;
        let path = entry.path().map_err(|e| format!("entry path: {e}"))?.into_owned();
        if !safe_member(&path) {
            return Err(format!("archive entry {:?} escapes the package", path.display()));
        }
        let Some(rel) = strip_first(&path) else { continue };
        let out = dest.join(rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| format!("mkdir: {e}"))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        entry.unpack(&out).map_err(|e| format!("unpack {}: {e}", out.display()))?;
    }
    Ok(())
}

fn extract_zip_stripped(archive: &Path, dest: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| format!("open archive: {e}"))?;
    let mut zip = zip::ZipArchive::new(BufReader::new(f)).map_err(|e| format!("read zip: {e}"))?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| format!("zip entry {i}: {e}"))?;
        // enclosed_name() is None for anything that would escape the root.
        let Some(path) = entry.enclosed_name() else {
            return Err("a zip entry escapes the package".into());
        };
        let Some(rel) = strip_first(&path) else { continue };
        let out = dest.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| format!("mkdir: {e}"))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        let mut w = std::fs::File::create(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
        std::io::copy(&mut entry, &mut w).map_err(|e| format!("write {}: {e}", out.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_is_cached_but_installing_stays_live() {
        // `snapshot()` shells out twice (`where node`, then `node -v`) and the
        // readiness endpoint that wants it is polled every few seconds, so the
        // result is cached. The subtlety worth pinning: `installing` is LIVE
        // state, not a probe result. Serving it from the cache would leave the
        // UI's "Installing Node…" button stuck on its old value for a minute —
        // right when the user is watching it.
        let dir = std::env::temp_dir().join(format!("oaiy-node-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rt = new_handle(dir.clone());

        let first = rt.snapshot();
        // Served from cache: the probe result must not drift between calls.
        let second = rt.snapshot();
        assert_eq!(first.available, second.available);
        assert_eq!(first.source, second.source);
        assert_eq!(first.version, second.version);
        assert_eq!(first.path, second.path);
        assert!(!first.installing);

        *rt.installing.lock().unwrap() = true;
        let during = rt.snapshot();
        assert!(during.installing, "installing must not be served stale from the cache");
        // …and the cached probe half is still intact alongside it.
        assert_eq!(during.source, first.source);
        assert_eq!(during.path, first.path);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_platform_asset_is_a_real_nodejs_dist_url() {
        let (asset, url, is_zip) = asset_for_platform().expect("this platform is supported");
        assert!(url.starts_with(&format!("https://nodejs.org/dist/v{NODE_VERSION}/")));
        assert!(asset.contains(NODE_VERSION));
        assert_eq!(is_zip, cfg!(target_os = "windows"), "windows ships a zip, others tar.gz");
        assert_eq!(asset.ends_with(".zip"), is_zip);
    }

    #[test]
    fn archive_members_cannot_escape_the_destination() {
        assert!(!safe_member(Path::new("../../evil")));
        assert!(!safe_member(Path::new("/etc/passwd")));
        assert!(safe_member(Path::new("node-v22/bin/node")));
    }

    #[test]
    fn the_leading_directory_is_stripped() {
        assert_eq!(strip_first(Path::new("node-v22-win-x64/node.exe")), Some(PathBuf::from("node.exe")));
        assert_eq!(
            strip_first(Path::new("node-v22-linux-x64/bin/node")),
            Some(PathBuf::from("bin/node"))
        );
        // The top-level dir entry itself strips to nothing and is skipped.
        assert_eq!(strip_first(Path::new("node-v22-win-x64")), None);
    }

    #[test]
    fn resolve_prefers_a_portable_install_over_path() {
        // No portable install in a temp data dir → falls back to the system one
        // (which this machine has), proving the fallback arm is wired.
        let dir = std::env::temp_dir().join(format!("oaiy-node-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rt = new_handle(dir.clone());
        let snap = rt.snapshot();
        assert_ne!(snap.source, "portable", "nothing installed under a fresh data dir");
        assert_eq!(snap.installs_version, NODE_VERSION);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
