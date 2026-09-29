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
//! - [`host`]: the ring as this desktop runs it, and where a request is allowed or refused.

pub mod contract;
pub mod devices;
pub mod host;
pub mod limits;
pub mod phrases;
pub mod plan;
pub mod presence;
pub mod routes;
pub mod session;
pub mod settings;

#[cfg(test)]
mod session_tests;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use serde::Serialize;

pub use host::{Authorised, CallInfo, CallSource, Clock, DeviceSource, PresenceSource, Ring};
pub use plan::{plan, Decision, Inputs, PlanReason, Presence, Reason, RingPlan};
pub use session::{set_global_notifier, ActiveRing, Action, RingError, RingNotifier, TransferPlugin};
pub use settings::{RingSettings, SettingsError, SettingsStore};

/// What the receptionist may do because of the owner's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Features {
    /// It may try to reach the owner for a caller who asks for a person.
    pub transfer: bool,
    /// It may take a message for the owner.
    pub messages: bool,
}

impl Ring {
    /// The ring kept in the data folder `dir`.
    pub fn open(dir: &Path) -> Arc<Ring> {
        Ring::with(SettingsStore::open(dir), limits::Attempts::open(dir))
    }

    /// A ring that keeps nothing on disk (tests).
    pub fn in_memory(settings: RingSettings) -> Arc<Ring> {
        Ring::with(SettingsStore::in_memory(settings), limits::Attempts::in_memory())
    }

    /// What the receptionist may do now.
    pub fn features(&self) -> Features {
        let s = self.settings.get();
        Features { transfer: s.enabled, messages: s.messages_on() }
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

/// The phone plugin's events that matter to a ring: the call ended (whatever rings for it is over, and a call the
/// owner had is over), and how a request came out (`aokie.call.assistance.resolved`, `data.outcome`).
pub fn apply_plugin_event(ring: &Ring, name: &str, data: &serde_json::Value, correlation: &str) {
    let text = |k: &str| data.get(k).and_then(serde_json::Value::as_str).unwrap_or("");
    match name {
        "aokie.call.ended" => {
            let call = if text("callId").is_empty() { correlation } else { text("callId") };
            if !call.is_empty() {
                ring.call_finished(call);
                ring.call_ended_by_phone(call);
            }
        }
        "aokie.call.assistance.resolved" => {
            use crate::voice::transfer::Outcome;
            let outcome = match text("outcome") {
                "transferred" => Outcome::Accepted,
                "declined" => Outcome::Declined,
                "unavailable" => Outcome::Unavailable,
                "expired" => Outcome::Expired,
                _ => return,
            };
            if !text("requestId").is_empty() {
                ring.resolve(text("requestId"), outcome, "phone");
            }
        }
        _ => {}
    }
}
