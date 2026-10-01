//! Whether the app is in the middle of something a backup or a restore should not interrupt or copy under.
//!
//! There is ONE decision about that, and it is the updater's: `update::blockers` (a call on OAIY's own
//! line, a call a phone plugin reports or cannot say, an Agent task, a download, an engine's media job,
//! an install, a move of the data folder). A backup asks the source the updater asks
//! (`Updater::activity`, the desktop's probes) and the function the updater asks
//! (`blockers::work_blockers`), so whatever holds an update back holds a backup back, and a new thing
//! the updater learns to look at holds a backup back too. This module keeps no list of sources of its
//! own: it asks, adds the reasons that are the backup's alone (a backup is already being made), and
//! turns the reasons into a refusal in the updater's words.

use crate::update::blockers::{self, Activity, Blocker, Readings};

use super::{state, BackupError, ErrorKind, Result};

/// What does not happen while the app is busy, for the sentences that say so ("It does not ... while it can't tell").
pub const WAITS: &str = "start a backup or a restore";

/// What stands in the way of a backup or a restore: the updater's reasons (same sources, same words) and the backup's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Busy {
    reasons: Vec<Blocker>,
}

impl Busy {
    /// Nothing in the way (a caller that has not looked, or a test).
    pub fn none() -> Self {
        Self::default()
    }

    /// The reasons for what the app is doing, as the updater would work them out (without how long it has been up).
    pub fn from_readings(readings: &Readings) -> Self {
        Self { reasons: blockers::work_blockers(readings, WAITS) }
    }

    /// Look at the app now, asking every source afresh (this is a decision, not a status a window polls), and add the
    /// backup's own reasons. `None`: nothing has said what the app is doing yet, which is still starting up.
    /// This can ask a phone plugin, and so wait a few seconds: not on an async runtime's own thread.
    pub fn look(activity: Option<&dyn Activity>) -> Self {
        let mut busy = match activity {
            Some(activity) => Self::from_readings(&activity.read(true)),
            None => Self { reasons: vec![Blocker { code: "starting", message: "OAIY is still starting up. Try again in a minute.".to_string() }] },
        };
        // A backup is already being made: a second one, or a restart that would end it, waits.
        if state::in_use() {
            busy = busy.and("backupRunning", "A backup is being made.");
        }
        busy
    }

    /// The look could not be made: not the same as quiet, so it refuses.
    pub fn cannot_tell() -> Self {
        Self { reasons: vec![Blocker { code: "unknown", message: "OAIY could not tell what it is doing, so it does not start a backup or a restore. Try again in a moment.".to_string() }] }
    }

    /// One more reason, the backup's own.
    pub fn and(mut self, code: &'static str, message: impl Into<String>) -> Self {
        self.reasons.push(Blocker { code, message: message.into() });
        self
    }

    /// The stable names of the reasons, in the order the updater gives them.
    pub fn codes(&self) -> Vec<&'static str> {
        self.reasons.iter().map(|b| b.code).collect()
    }

    /// What is in the way, in words a person reads.
    pub fn reasons(&self) -> Vec<String> {
        self.reasons.iter().map(|b| b.message.clone()).collect()
    }

    pub fn is_busy(&self) -> bool {
        !self.reasons.is_empty()
    }

    /// `Ok` when nothing is in the way; otherwise a refusal that says what, and that it is worth trying again.
    pub fn refuse_if_busy(&self, what: &str) -> Result<()> {
        if !self.is_busy() {
            return Ok(());
        }
        let mut first = what.chars();
        let what = first.next().map(|c| c.to_uppercase().collect::<String>() + first.as_str()).unwrap_or_default();
        Err(BackupError::new(ErrorKind::Busy, format!("{what} has to wait. {} Try again when it is finished.", self.reasons().join(" "))))
    }
}
