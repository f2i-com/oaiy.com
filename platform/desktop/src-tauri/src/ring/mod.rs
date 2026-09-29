//! Putting a caller through to the owner, and who is rung for it.
//!
//! When a caller asks for a person the receptionist can try to reach the owner:
//! this desktop decides who is rung ([`plan`]), for how long, and whether to ring
//! at all, and the owner's devices ring (a dialog and a notification here, the
//! Companion on this computer or a phone). Nothing is put through unless the owner
//! turned "Transfer calls to me" on ([`settings`]), and whatever happens the
//! receptionist falls back to taking a message.
//!
//! - [`plan`]: the pure policy, with the design's 33 conformance vectors.
//! - [`phrases`]: whether the caller's own words asked for a person.
//! - [`settings`]: `<data>/ring.json`.
//! - [`limits`]: how often callers have been put through this hour.

pub mod limits;
pub mod phrases;
pub mod plan;
pub mod routes;
pub mod settings;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
use serde_json::Value;

pub use plan::{plan, Decision, Inputs, PlanReason, Presence, Reason, RingPlan};
pub use settings::{RingSettings, SettingsError, SettingsStore};

/// What the receptionist may do because of the owner's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Features {
    /// It may try to reach the owner for a caller who asks for a person.
    pub transfer: bool,
    /// It may take a message for the owner.
    pub messages: bool,
}

/// The ring, as this desktop runs it: the owner's settings and the tries so far.
pub struct Ring {
    pub settings: SettingsStore,
    pub attempts: Mutex<limits::Attempts>,
}

impl Ring {
    /// The ring kept in the data folder `dir`.
    pub fn open(dir: &Path) -> Arc<Ring> {
        Arc::new(Ring { settings: SettingsStore::open(dir), attempts: Mutex::new(limits::Attempts::open(dir)) })
    }

    /// A ring that keeps nothing on disk (tests).
    pub fn in_memory(settings: RingSettings) -> Arc<Ring> {
        Arc::new(Ring { settings: SettingsStore::in_memory(settings), attempts: Mutex::new(limits::Attempts::in_memory()) })
    }

    /// What the receptionist may do now.
    pub fn features(&self) -> Features {
        let s = self.settings.get();
        Features { transfer: s.enabled, messages: s.messages_on() }
    }

    /// Change some of the owner's settings (see [`SettingsStore::change`]).
    pub fn change_settings(&self, change: &Value) -> Result<(), SettingsError> {
        self.settings.change(change).map(|_| ())
    }
}

static SHARED: OnceLock<Arc<Ring>> = OnceLock::new();

/// Open this desktop's ring in its data folder. Everything is off until the owner turns it on.
pub fn init(data_dir: &Path) {
    let ring = Ring::open(data_dir);
    let _ = SHARED.set(ring);
}

/// This desktop's ring (none before [`init`]: a desktop that never opened one puts nobody through).
pub fn shared() -> Option<Arc<Ring>> {
    SHARED.get().cloned()
}
