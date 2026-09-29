//! Whether the app is in the middle of something a restart or a backup should not interrupt.
//!
//! Small and self-contained on purpose: a backup refuses to start, and "Restart to finish
//! restoring" refuses, while a phone call is live, an Agent task is running, a download is going,
//! or an engine's media job is working. The numbers come from wherever they are kept (the voice
//! hub, the Agent task hub, the downloads, the engines); this module only turns them into a
//! plain refusal. The updater has the same need; when the two are merged, [`BusySignals`] is the
//! part to share.

use std::sync::OnceLock;

use super::{BackupError, ErrorKind, Result};

/// How many of each kind of thing is going on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BusySignals {
    /// Phone calls that are live.
    pub live_calls: usize,
    /// Tasks flows have given the Agent that it has not answered.
    pub agent_tasks: usize,
    /// Model, engine and other downloads that are running or waiting.
    pub downloads: usize,
    /// Image, video, music and other jobs the engines are working on.
    pub engine_jobs: usize,
    /// Installs of services, Python or Node that are running.
    pub installs: usize,
}

impl BusySignals {
    /// What is going on, in words a person reads after "OAIY is busy: ".
    pub fn reasons(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut add = |n: usize, one: &str, many: &str| {
            if n == 1 {
                out.push(format!("1 {one}"));
            } else if n > 1 {
                out.push(format!("{n} {many}"));
            }
        };
        add(self.live_calls, "phone call is live", "phone calls are live");
        add(self.agent_tasks, "task for the Agent is running", "tasks for the Agent are running");
        add(self.downloads, "download is running", "downloads are running");
        add(self.engine_jobs, "engine job is working", "engine jobs are working");
        add(self.installs, "install is running", "installs are running");
        out
    }

    pub fn is_busy(&self) -> bool {
        self != &Self::default()
    }

    /// `Ok` when nothing is going on; otherwise a refusal that says what, and that it is worth trying again.
    pub fn refuse_if_busy(&self, what: &str) -> Result<()> {
        if !self.is_busy() {
            return Ok(());
        }
        Err(BackupError::new(
            ErrorKind::Busy,
            format!("OAIY is busy ({}), so {what} has to wait. Try again when it is finished.", self.reasons().join("; ")),
        ))
    }
}

type Probe = Box<dyn Fn() -> usize + Send + Sync>;

static LIVE_CALLS: OnceLock<Probe> = OnceLock::new();

/// The voice hub says how many calls are live (set once, when the hub is made).
pub fn register_live_calls(probe: impl Fn() -> usize + Send + Sync + 'static) {
    let _ = LIVE_CALLS.set(Box::new(probe));
}

/// What can be known without the GUI's handles: live calls and Agent tasks. The GUI adds the
/// downloads, installs and engine jobs it can see.
pub fn local_signals() -> BusySignals {
    BusySignals {
        live_calls: LIVE_CALLS.get().map_or(0, |probe| probe()),
        agent_tasks: crate::agent_tasks::pending_count(),
        ..BusySignals::default()
    }
}
