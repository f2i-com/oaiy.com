//! Curated model quick-add catalog.
//!
//! A small JSON file shipped in the binary + seeded to disk so users can
//! extend it. The UI's Models tab renders each entry as a one-click
//! "Download" button that pre-fills the URL/filename/subdir into the
//! existing `Downloads::start()` flow — no special endpoint, just less
//! typing for popular weights.
//!
//! Why on-disk + user-editable instead of fully baked-in?
//!   - The set of "popular models this month" drifts; users want to add
//!     their team's favourites without waiting for a companion release
//!   - One pattern matches services + scripts (seed-on-first-run +
//!     editable) so there's nothing new for the user to learn

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const BUILTIN_CATALOG: &str = include_str!("../../resources/model-catalog.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub url: String,
    pub filename: String,
    #[serde(default)]
    pub subdir: Option<String>,
    /// Pre-known download size in bytes. Used to show "~4.5 GB" in the
    /// UI before the user kicks off a download. `0` or omitted means
    /// unknown.
    #[serde(default)]
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogCategory {
    pub id: String,
    pub name: String,
    pub description: String,
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Catalog {
    pub categories: Vec<CatalogCategory>,
}

/// Load the catalog from the user's data dir, seeding/refreshing the
/// bundled copy as appropriate. Falls back to the bundled copy if the
/// user's copy is missing or malformed (logged) so a bad edit can't
/// break the UI.
///
/// Re-seed policy (so shipping new catalog entries actually reaches
/// people who already launched an older build, WITHOUT clobbering hand
/// edits): we keep a hidden `.model-catalog.seed.json` snapshot of
/// whatever we last wrote. On load:
///   - on-disk missing: write bundled + snapshot
///   - on-disk == snapshot: untouched by user; if bundled changed, refresh both
///   - on-disk != snapshot: user edited; leave it alone
///   - snapshot missing (pre-seed-tracking install): treat the on-disk copy
///     as an old auto-seed and refresh to bundled, creating the snapshot so
///     future edits are detected
pub fn load(data_dir: &Path) -> Catalog {
    let path = data_dir.join("model-catalog.json");
    let seed = data_dir.join(".model-catalog.seed.json");

    let on_disk = std::fs::read_to_string(&path).ok();
    match &on_disk {
        None => {
            // First run — seed both.
            write_pair(&path, &seed);
        }
        Some(current) => {
            match std::fs::read_to_string(&seed).ok() {
                Some(snapshot) => {
                    // Refresh only if the user hasn't touched it AND we
                    // actually have something new to offer.
                    if current.trim() == snapshot.trim() && current.trim() != BUILTIN_CATALOG.trim()
                    {
                        write_pair(&path, &seed);
                    }
                }
                None => {
                    // Pre-seed-tracking install: the on-disk copy is an
                    // old auto-seed (no user-edit tracking existed yet).
                    // Refresh to bundled and start tracking.
                    write_pair(&path, &seed);
                }
            }
        }
    }

    // Read back whatever's now on disk; fall back to bundled on any
    // read/parse failure.
    match std::fs::read_to_string(&path) {
        Ok(s) => match serde_json::from_str::<Catalog>(&s) {
            Ok(c) => return c,
            Err(e) => log::warn!(
                "model-catalog.json at {} is malformed ({e}); using bundled copy",
                path.display()
            ),
        },
        Err(e) => log::warn!("could not read {}: {e}", path.display()),
    }
    serde_json::from_str(BUILTIN_CATALOG).expect("bundled model-catalog.json must parse")
}

/// Write the bundled catalog to `path` and snapshot it to `seed` so a
/// later load can tell "untouched auto-seed" from "user-edited".
fn write_pair(path: &Path, seed: &Path) {
    if let Err(e) = std::fs::write(path, BUILTIN_CATALOG) {
        log::warn!("could not write model-catalog.json at {}: {e}", path.display());
        return;
    }
    if let Err(e) = std::fs::write(seed, BUILTIN_CATALOG) {
        log::warn!("could not write catalog seed snapshot at {}: {e}", seed.display());
    }
}

/// Cached path for diagnostics — surfaced in the snapshot so the UI can
/// show "edit this file to add models".
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSnapshot {
    pub source_path: String,
    pub catalog: Catalog,
}

impl CatalogSnapshot {
    pub fn new(data_dir: &Path) -> Self {
        let path = data_dir.join("model-catalog.json");
        let catalog = load(data_dir);
        Self {
            source_path: path.display().to_string(),
            catalog,
        }
    }
}

/// Helper holder so other code can pass the data dir around for catalog
/// reloads without keeping a copy of the whole `Catalog`.
#[derive(Clone)]
pub struct CatalogHandle {
    data_dir: PathBuf,
}

impl CatalogHandle {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    pub fn snapshot(&self) -> CatalogSnapshot {
        CatalogSnapshot::new(&self.data_dir)
    }
}
