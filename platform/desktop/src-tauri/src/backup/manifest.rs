//! The manifest: the record at the front of every backup, and what it is checked against.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::rules::Excluded;
use super::{container, BackupError, ErrorKind, Limits, Result, MANIFEST_NAME};

/// The manifest's version, and the only one this reads.
pub const VERSION: u32 = 1;

/// The name a backup carries for the program that made it.
pub const APP_NAME: &str = "oaiy";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
}

/// One item of the backup: its name (relative to the data folder, with forward slashes), its size
/// and the SHA-256 of its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Counts {
    pub files: u64,
    pub bytes: u64,
    #[serde(default)]
    pub agent_projects: u64,
    #[serde(default)]
    pub agent_conversations: u64,
    #[serde(default)]
    pub agent_files: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub v: u32,
    /// RFC 3339.
    pub created_at: String,
    pub app: AppInfo,
    pub platform: String,
    pub entries: Vec<Entry>,
    pub excluded: Vec<Excluded>,
    pub counts: Counts,
    pub includes_keys: bool,
    /// What did not go as planned, in plain words (the Agent's storage was not reachable, a file
    /// changed while it was being read).
    pub partial: Vec<String>,
}

impl Manifest {
    /// Check a manifest that came out of a file before anything in it is used: its version, the
    /// names it lists (each safe, none twice, each one a thing a backup may hold) and the sizes.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.v != VERSION {
            return Err(BackupError::new(
                ErrorKind::Unsupported,
                if self.v > VERSION {
                    "This backup was made by a newer OAIY. Update OAIY to open it."
                } else {
                    "This backup is in a format OAIY does not read."
                },
            ));
        }
        if self.app.name != APP_NAME {
            return Err(BackupError::new(ErrorKind::Unsupported, "This is not an OAIY backup."));
        }
        if chrono::DateTime::parse_from_rfc3339(&self.created_at).is_err() {
            return Err(BackupError::new(ErrorKind::Damaged, "This backup's record is damaged."));
        }
        if self.entries.len() + 1 > limits.max_entries {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup holds more items than OAIY will restore."));
        }
        let mut seen = HashSet::new();
        let mut total = 0u64;
        for entry in &self.entries {
            container::check_entry_name(&entry.name, limits)?;
            if entry.name == MANIFEST_NAME {
                return Err(BackupError::new(ErrorKind::Unsafe, "This backup lists its own record as an item."));
            }
            if !seen.insert(entry.name.to_lowercase()) {
                return Err(BackupError::new(ErrorKind::Unsafe, "This backup lists the same item twice."));
            }
            if entry.sha256.len() != 64 || !entry.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err(BackupError::new(ErrorKind::Damaged, "This backup's record is damaged."));
            }
            // What kind of thing it is decides how large it may be: a calendar of 512 MiB is not a calendar.
            if entry.size > limits.entry_cap(&entry.name) {
                return Err(BackupError::new(ErrorKind::TooLarge, "An item in this backup is larger than OAIY will restore."));
            }
            total = total.saturating_add(entry.size);
            // An item the table excludes (a credential, a program) refuses the backup; one it does not know
            // is left for the restore to list as not restored.
            super::rules::category_of_backup_entry(&entry.name)
                .map_err(|why| BackupError::new(ErrorKind::Unsafe, format!("This backup is refused: {why}.")))?;
        }
        if total > limits.max_total_bytes {
            return Err(BackupError::new(ErrorKind::TooLarge, "This backup is larger than OAIY will restore."));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(self).expect("a manifest is always serialisable");
        bytes.push(b'\n');
        bytes
    }
}
