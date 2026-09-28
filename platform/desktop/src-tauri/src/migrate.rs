//! Data-folder migration — copy or move the user's path-independent data
//! from the old data dir to a newly-chosen one, so changing the folder
//! doesn't strand the models someone already downloaded.
//!
//! Runs on a background thread; the UI polls `migration_status` for a
//! progress bar (same poll pattern the downloads UI already uses).
//!
//! We migrate ONLY path-independent content: `models/` (the big user
//! concern — "easily access the models"), `templates/` (custom services +
//! tweaks the user made), and `bin/` (downloaded service binaries like
//! llama.cpp). We deliberately SKIP `venvs/` + `python/` (absolute paths
//! are baked into venv configs, so a copy would be broken — reinstall
//! instead) and `scripts/` (re-seeded from the bundle on every launch).
//!
//! Policy: skip-if-exists at the destination (idempotent re-runs, and we
//! never clobber a file already in the new folder). In Move mode the
//! source file is removed once the destination copy is confirmed present,
//! then emptied source dirs are pruned.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Subdirs that are safe to relocate (see module docs).
pub const MIGRATE_SUBDIRS: &[&str] = &["models", "templates", "bin"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Copy,
    Move,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "copy" => Some(Mode::Copy),
            "move" => Some(Mode::Move),
            _ => None,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Mode::Copy => "copy",
            Mode::Move => "move",
        }
    }
}

/// What a migration would move — surfaced to the UI so it can show
/// "Bring 12 models (43.1 GiB) to the new folder?" and only offer the
/// action when there's actually a pending change with content.
#[derive(Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MigratePlan {
    pub old_dir: String,
    pub new_dir: String,
    pub file_count: u64,
    pub total_bytes: u64,
    /// Which migratable subdirs are present + non-empty in the old dir.
    pub subdirs: Vec<String>,
    /// True when old != new AND there's at least one file to move.
    pub can_migrate: bool,
}

/// Live progress, polled by the UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub struct MigrationProgress {
    pub running: bool,
    pub mode: String,
    pub files_total: u64,
    pub files_done: u64,
    pub bytes_total: u64,
    pub bytes_done: u64,
    pub current: String,
    pub done: bool,
    pub error: Option<String>,
}


/// Shared, poll-able migration state (one at a time).
pub type MigrationHandle = Arc<Mutex<MigrationProgress>>;

fn norm(s: &str) -> String {
    s.trim_end_matches(['/', '\\']).to_lowercase()
}

/// Walk `root` recursively, invoking `f(file_path, size_bytes)` per file.
/// Unreadable dirs are skipped rather than aborting the walk.
fn for_each_file(root: &Path, f: &mut impl FnMut(&Path, u64)) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(entry.path()),
                Ok(ft) if ft.is_file() => {
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    f(&entry.path(), size);
                }
                _ => {}
            }
        }
    }
}

/// Compute what a migration would move.
pub fn plan(old: &Path, new: &Path) -> MigratePlan {
    let mut file_count = 0u64;
    let mut total_bytes = 0u64;
    let mut subdirs = Vec::new();
    for sub in MIGRATE_SUBDIRS {
        let src = old.join(sub);
        if !src.is_dir() {
            continue;
        }
        let mut sub_files = 0u64;
        for_each_file(&src, &mut |_p, size| {
            sub_files += 1;
            total_bytes += size;
        });
        if sub_files > 0 {
            file_count += sub_files;
            subdirs.push((*sub).to_string());
        }
    }
    let changed = norm(&old.to_string_lossy()) != norm(&new.to_string_lossy());
    MigratePlan {
        old_dir: old.display().to_string(),
        new_dir: new.display().to_string(),
        file_count,
        total_bytes,
        subdirs,
        can_migrate: changed && file_count > 0,
    }
}

/// Run the migration synchronously, updating `progress` as it goes. Safe
/// to call on a background thread. A copy error stops the run and is
/// surfaced via `progress.error` — already-copied files remain (and are
/// skipped on a retry).
pub fn run(old: PathBuf, new: PathBuf, mode: Mode, progress: MigrationHandle) {
    // Pre-count so the progress bar is accurate from the first tick.
    let p = plan(&old, &new);
    if let Ok(mut g) = progress.lock() {
        *g = MigrationProgress {
            running: true,
            mode: mode.as_str().to_string(),
            files_total: p.file_count,
            bytes_total: p.total_bytes,
            ..Default::default()
        };
    }

    let result = (|| -> std::io::Result<()> {
        for sub in MIGRATE_SUBDIRS {
            let src_root = old.join(sub);
            if !src_root.is_dir() {
                continue;
            }
            let dst_root = new.join(sub);
            // Snapshot the file list before mutating (Move deletes as we go).
            let mut files: Vec<(PathBuf, u64)> = Vec::new();
            for_each_file(&src_root, &mut |path, size| files.push((path.to_path_buf(), size)));

            for (src, size) in files {
                let rel = src.strip_prefix(&src_root).unwrap_or(&src);
                let dst = dst_root.join(rel);
                if let Ok(mut g) = progress.lock() {
                    g.current = format!("{}/{}", sub, rel.display());
                }
                let dst_existed = dst.exists();
                if !dst_existed {
                    if let Some(parent) = dst.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::copy(&src, &dst)?;
                }
                if mode == Mode::Move {
                    // Delete the source ONLY when this run created the
                    // destination copy itself (i.e. the dest didn't already
                    // exist). On ANY pre-existing-destination collision we
                    // KEEP the source — equal byte-length does NOT mean equal
                    // content, so a same-size-different-bytes dest would
                    // otherwise silently drop the user's data. A "move" must
                    // never delete a source whose content we never copied.
                    if !dst_existed {
                        let _ = std::fs::remove_file(&src);
                    }
                }
                if let Ok(mut g) = progress.lock() {
                    g.files_done += 1;
                    g.bytes_done += size;
                }
            }
            if mode == Mode::Move {
                prune_empty_dirs(&src_root);
            }
        }
        Ok(())
    })();

    if let Ok(mut g) = progress.lock() {
        g.running = false;
        g.done = true;
        g.current = String::new();
        if let Err(e) = result {
            g.error = Some(e.to_string());
        }
    }
}

/// Best-effort removal of now-empty directories under `root` (deepest
/// first). `remove_dir` only succeeds on an empty dir, so this never
/// deletes a file we failed to move.
fn prune_empty_dirs(root: &Path) {
    let mut dirs = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    stack.push(entry.path());
                }
            }
        }
        dirs.push(dir);
    }
    dirs.sort_by_key(|d| std::cmp::Reverse(d.components().count()));
    for d in dirs {
        let _ = std::fs::remove_dir(&d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn plan_counts_only_migratable_subdirs() {
        let base = std::env::temp_dir().join("oaiy-migrate-test-plan");
        let _ = std::fs::remove_dir_all(&base);
        let old = base.join("old");
        write(&old.join("models/a.bin"), b"hello"); // 5
        write(&old.join("models/sub/b.bin"), b"world!!"); // 7
        write(&old.join("templates/svc.json"), b"{}"); // 2
        write(&old.join("venvs/x/pyvenv.cfg"), b"home=C:\\old"); // must NOT migrate
        write(&old.join("scripts/install.ps1"), b"echo hi"); // must NOT migrate

        let pl = plan(&old, &base.join("new"));
        assert!(pl.can_migrate);
        assert_eq!(pl.file_count, 3, "only models + templates files");
        assert_eq!(pl.total_bytes, 14);
        assert!(pl.subdirs.contains(&"models".to_string()));
        assert!(pl.subdirs.contains(&"templates".to_string()));
        assert!(!pl.subdirs.contains(&"venvs".to_string()));
        assert!(!pl.subdirs.contains(&"scripts".to_string()));

        // No change → can't migrate even with content.
        assert!(!plan(&old, &old).can_migrate);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn copy_leaves_source_then_move_removes_it() {
        let base = std::env::temp_dir().join("oaiy-migrate-test-copymove");
        let _ = std::fs::remove_dir_all(&base);
        let old = base.join("old");
        write(&old.join("models/a.bin"), b"hello");
        write(&old.join("models/sub/b.bin"), b"world!!");
        write(&old.join("templates/svc.json"), b"{}");
        write(&old.join("venvs/x/pyvenv.cfg"), b"home=C:\\old");

        // --- Copy ---
        let new = base.join("new");
        let prog = Arc::new(Mutex::new(MigrationProgress::default()));
        run(old.clone(), new.clone(), Mode::Copy, prog.clone());
        {
            let g = prog.lock().unwrap();
            assert!(g.done && !g.running && g.error.is_none());
            assert_eq!(g.files_done, 3);
            assert_eq!(g.bytes_done, 14);
        }
        assert_eq!(std::fs::read(new.join("models/a.bin")).unwrap(), b"hello");
        assert_eq!(std::fs::read(new.join("models/sub/b.bin")).unwrap(), b"world!!");
        assert_eq!(std::fs::read(new.join("templates/svc.json")).unwrap(), b"{}");
        assert!(!new.join("venvs").exists(), "venvs must be skipped");
        assert!(old.join("models/a.bin").exists(), "copy keeps the source");

        // --- Move (into a fresh dir) ---
        let new2 = base.join("new2");
        let prog2 = Arc::new(Mutex::new(MigrationProgress::default()));
        run(old.clone(), new2.clone(), Mode::Move, prog2.clone());
        assert_eq!(std::fs::read(new2.join("models/a.bin")).unwrap(), b"hello");
        assert!(!old.join("models/a.bin").exists(), "move removes the source file");
        assert!(!old.join("models").exists(), "emptied source dir is pruned");
        assert!(old.join("venvs/x/pyvenv.cfg").exists(), "move leaves skipped dirs alone");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn move_keeps_source_on_size_collision() {
        // Move into a dir that already has a same-named but DIFFERENT file:
        // the dest keeps its own copy AND the source is preserved (deleting
        // it would be silent data loss).
        let base = std::env::temp_dir().join("oaiy-migrate-test-movecollide");
        let _ = std::fs::remove_dir_all(&base);
        let old = base.join("old");
        let new = base.join("new");
        write(&old.join("models/m.bin"), b"ORIGINAL-LONG"); // 13 bytes
        write(&new.join("models/m.bin"), b"diff"); // 4 bytes — collision

        let prog = Arc::new(Mutex::new(MigrationProgress::default()));
        run(old.clone(), new.clone(), Mode::Move, prog);

        assert_eq!(std::fs::read(new.join("models/m.bin")).unwrap(), b"diff", "dest kept");
        assert!(
            old.join("models/m.bin").exists(),
            "move must keep the source on a same-name/different-size collision"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn move_keeps_source_on_equal_size_different_bytes() {
        // Move into a dir that already has a same-named file of EQUAL LENGTH
        // but DIFFERENT bytes: skip-if-exists leaves the dest untouched, and
        // the source MUST be preserved — a size-only check would falsely
        // treat the dest as "the same file" and silently delete the source.
        let base = std::env::temp_dir().join("oaiy-migrate-test-eqsizecollide");
        let _ = std::fs::remove_dir_all(&base);
        let old = base.join("old");
        let new = base.join("new");
        write(&old.join("models/m.bin"), b"AAAAA"); // 5 bytes
        write(&new.join("models/m.bin"), b"BBBBB"); // 5 bytes — equal size, different content

        let prog = Arc::new(Mutex::new(MigrationProgress::default()));
        run(old.clone(), new.clone(), Mode::Move, prog);

        assert_eq!(
            std::fs::read(new.join("models/m.bin")).unwrap(),
            b"BBBBB",
            "dest with same size but different bytes is kept untouched"
        );
        assert!(
            old.join("models/m.bin").exists(),
            "move must keep the source when the dest pre-existed (equal size, different bytes)"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn skip_if_exists_does_not_clobber() {
        let base = std::env::temp_dir().join("oaiy-migrate-test-skip");
        let _ = std::fs::remove_dir_all(&base);
        let old = base.join("old");
        let new = base.join("new");
        write(&old.join("models/a.bin"), b"OLD");
        write(&new.join("models/a.bin"), b"NEWER-KEEP"); // already in destination

        let prog = Arc::new(Mutex::new(MigrationProgress::default()));
        run(old.clone(), new.clone(), Mode::Copy, prog.clone());
        assert_eq!(
            std::fs::read(new.join("models/a.bin")).unwrap(),
            b"NEWER-KEEP",
            "existing destination file must not be clobbered"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
