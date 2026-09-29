//! The rings going now: a caller is asking for the owner, this desktop rings (a notification and the
//! dashboard's dialog), and the owner may answer there.
//!
//! A ring begins when the phone plugin says the request is out (`oaiy.ring.opened`) for a plan this
//! desktop allowed, and ends the first time anything resolves it: the phone says how it came out
//! (the call's outcome frame, or its `assistance.resolved` event), the owner declines here, the caller
//! hangs up, or its time and a short grace pass with nobody having said. Whatever resolves it first
//! wins and later word about it is ignored, so a ring is never ended twice and the dialog never comes
//! back once it has gone.
//!
//! What the owner can do here is only ever a request. **Accept** asks the phone plugin (through the
//! connector command `call.transfer.respond`, if it has it) to take the call on the owner's behalf: the
//! ring goes on, and is resolved only when the phone says the call was taken. **Decline** and **Take a
//! message instead** end the ring here at once and tell the call, so the caller is offered a message
//! without waiting for the devices; the plugin is asked to withdraw the request too. If a device
//! accepts after that, the takeover is obeyed.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::Duration;

use serde::Serialize;

use super::contract::{OpenedParams, RespondAction};
use super::host::Ring;
use super::plan::RingPlan;
use crate::voice::transfer::Outcome;

/// How long past its time a ring waits to hear how it came out before it is over.
pub const EXPIRY_GRACE: Duration = Duration::from_secs(5);
/// The most rings that ended kept, for the record.
const KEPT: usize = 30;
/// The most words of what the caller said shown in the dialog.
const SAID_SHOWN: usize = 2;

/// What tells the owner a ring has begun: on the GUI, a native notification and the window brought up.
pub trait RingNotifier: Send + Sync {
    /// The owner is being rung for this caller.
    fn ringing(&self, ring: &ActiveRing);
    /// The ring is over, and how it came out.
    fn ended(&self, id: &str, outcome: &str);
    /// Somebody asked for the owner and nobody could be rung, because no device is set up to take a transfer.
    fn noticed(&self, _notice: &Notice) {}
}

/// What asks the phone plugin to take (or drop) a request on the owner's behalf.
pub trait TransferPlugin: Send + Sync {
    /// Ask. An error says why the phone could not be asked (it does not have the command, it is off).
    fn respond(&self, request: &str, action: RespondAction) -> Result<(), String>;
}

static GLOBAL: std::sync::RwLock<Option<Arc<dyn RingNotifier>>> = std::sync::RwLock::new(None);

/// The notifier of this desktop's window (a ring that has none of its own uses it).
pub fn set_global_notifier(notifier: Option<Arc<dyn RingNotifier>>) {
    if let Ok(mut g) = GLOBAL.write() {
        *g = notifier;
    }
}

pub(super) fn global_notifier() -> Option<Arc<dyn RingNotifier>> {
    GLOBAL.read().ok().and_then(|g| g.clone())
}

/// One ring, as the dialog and the API show it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveRing {
    /// The request's id (the phone's).
    pub id: String,
    pub call_id: String,
    pub caller_name: String,
    pub caller_number: String,
    /// What the caller last said: their own words, as this desktop heard them.
    pub said: Vec<String>,
    /// Unix milliseconds: when it began, and when it stops ringing.
    pub started_at: u64,
    pub expires_at: u64,
    /// This desktop's clock now, so the countdown does not trust the window's.
    pub now: u64,
    /// Who else is rung, by name ("this computer", a device).
    pub devices: Vec<String>,
    /// The phone plugin can be asked to take the call for the owner.
    pub can_accept: bool,
    /// What the owner was last told about what they asked here.
    pub note: String,
}

struct Live {
    ring: ActiveRing,
    plan: RingPlan,
}

/// A ring that ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ended {
    pub id: String,
    pub call_id: String,
    pub outcome: &'static str,
    /// Who said: `phone`, `desktop` (the owner, here), `call` (it ended) or `timer`.
    pub source: &'static str,
}

/// What the owner is told when somebody asked for them and no ring could be made for want of a device.
pub const NO_DEVICE_TEXT: &str = "Someone asked for you. No device is set up to take a transfer, so they were offered a message.";
/// The most notices kept, and how long each is shown (milliseconds).
const NOTICES_KEPT: usize = 5;
const NOTICE_MS: u64 = 15 * 60 * 1000;
/// The native notification for one is raised at most this often.
const NOTICE_TOAST_EVERY: Duration = Duration::from_secs(10 * 60);

/// Somebody asked for the owner and nobody could be rung: not a ring, only the owner's word that their setup could not do it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Notice {
    pub id: String,
    pub call_id: String,
    pub caller_name: String,
    pub caller_number: String,
    /// Unix milliseconds.
    pub at: u64,
    pub text: String,
}

#[derive(Default)]
pub struct Sessions {
    live: Vec<Live>,
    ended: VecDeque<Ended>,
    notices: VecDeque<Notice>,
    last_toast: Option<std::time::Instant>,
}

/// What the owner can do with a ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Accept,
    Decline,
    /// Decline, and say the receptionist should take a message.
    Message,
}

impl Action {
    pub fn parse(s: &str) -> Option<Action> {
        match s {
            "accept" => Some(Action::Accept),
            "decline" => Some(Action::Decline),
            "message" => Some(Action::Message),
            _ => None,
        }
    }
}

/// What came of the owner's answer in the dialog.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Responded {
    pub ok: bool,
    pub note: String,
}

/// Why a ring could not be opened or answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingError {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for RingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn error(status: u16, code: &'static str, message: impl Into<String>) -> RingError {
    RingError { status, code, message: message.into() }
}

impl Ring {
    /// The plugin says the request is out: this desktop rings. Only for a plan this desktop allowed for the
    /// call and the plugin asked for; a request for anything else is refused and rings nothing.
    pub fn opened(self: &Arc<Self>, params: &OpenedParams) -> Result<ActiveRing, RingError> {
        let Some(plan) = self.claimed_plan(&params.plan_id, &params.call_id) else {
            return Err(error(409, "unknown_plan", "that plan was not allowed for this call, or has run out: nothing rings"));
        };
        let info = self.call_info(&params.call_id).unwrap_or_default();
        let now = self.now_ms();
        let given = params.expires_at.saturating_mul(1000);
        let longest = now + (u64::from(plan.ring_seconds) + 10) * 1000;
        // The phone's own end for the request, when it is one that makes sense; else the plan's time.
        let expires_at = if given > now && given <= longest { given } else { now + u64::from(plan.ring_seconds) * 1000 };
        let mut devices = Vec::new();
        if plan.desktop_toast {
            devices.push("this computer".to_string());
        }
        for id in plan.targets() {
            devices.push(self.device_label(&id));
        }
        let ring = ActiveRing {
            id: params.request_id.clone(),
            call_id: params.call_id.clone(),
            caller_name: info.name.trim().to_string(),
            caller_number: info.from.trim().to_string(),
            said: info.turns.iter().rev().take(SAID_SHOWN).rev().map(|t| t.chars().take(300).collect()).collect(),
            started_at: now,
            expires_at,
            now,
            devices,
            can_accept: self.plugin().is_some(),
            note: String::new(),
        };
        {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            if sessions.live.iter().any(|l| l.ring.id == ring.id) {
                return Ok(ring);
            }
            sessions.live.push(Live { ring: ring.clone(), plan: plan.clone() });
        }
        if plan.desktop_toast {
            if let Some(notifier) = self.notifier() {
                notifier.ringing(&ring);
            }
        }
        self.watch_expiry(&ring);
        Ok(ring)
    }

    /// The ring is over when its time and a grace have passed with nobody having said how it came out.
    fn watch_expiry(self: &Arc<Self>, ring: &ActiveRing) {
        let (weak, id) = (Arc::downgrade(self), ring.id.clone());
        let wait = Duration::from_millis(ring.expires_at.saturating_sub(self.now_ms())) + *self.expiry_grace.read().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(move || {
            std::thread::sleep(wait);
            if let Some(ring) = Weak::upgrade(&weak) {
                ring.resolve(&id, Outcome::Expired, "timer");
            }
        });
    }

    /// Something says how a ring came out. The first word wins: the ring ends, the notification is closed and the record
    /// kept; whatever comes after, about the same ring, is ignored. Whether this was the word that ended it.
    pub fn resolve(&self, request: &str, outcome: Outcome, source: &'static str) -> bool {
        let live = {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            let Some(at) = sessions.live.iter().position(|l| l.ring.id == request) else { return false };
            let live = sessions.live.remove(at);
            sessions.ended.push_back(Ended { id: live.ring.id.clone(), call_id: live.ring.call_id.clone(), outcome: outcome.as_str(), source });
            while sessions.ended.len() > KEPT {
                sessions.ended.pop_front();
            }
            live
        };
        if live.plan.desktop_toast {
            if let Some(notifier) = self.notifier() {
                notifier.ended(&live.ring.id, outcome.as_str());
            }
        }
        true
    }

    /// The call ended (or is over for the receptionist): whatever still rings for it is over.
    pub fn call_finished(&self, call: &str) {
        let ids: Vec<String> = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).live.iter().filter(|l| l.ring.call_id == call).map(|l| l.ring.id.clone()).collect();
        for id in ids {
            self.resolve(&id, Outcome::Cancelled, "call");
        }
    }

    /// The call says how a request came out (from the phone, or from its clock).
    pub fn outcome_seen(&self, request: &str, outcome: Outcome, source: &'static str) {
        let source = if source == "watchdog" { "timer" } else if source == "desktop" { "desktop" } else { "phone" };
        self.resolve(request, outcome, source);
    }

    /// Somebody asked for the owner and the plan had nobody to ring: told once a call, and shown until dismissed or a while has
    /// passed. The native notification is raised at most every ten minutes, so a caller who rings again and again cannot make
    /// the owner's computer chime for ever; the notice itself is always kept.
    pub(super) fn note_no_device(&self, call: &str, info: &super::host::CallInfo) {
        let now = self.now_ms();
        let notice = {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            sessions.notices.retain(|n| now.saturating_sub(n.at) < NOTICE_MS);
            if sessions.notices.iter().any(|n| n.call_id == call) {
                return;
            }
            let notice = Notice { id: format!("notice_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]), call_id: call.to_string(), caller_name: info.name.trim().to_string(), caller_number: info.from.trim().to_string(), at: now, text: NO_DEVICE_TEXT.to_string() };
            sessions.notices.push_back(notice.clone());
            while sessions.notices.len() > NOTICES_KEPT {
                sessions.notices.pop_front();
            }
            let toast = sessions.last_toast.is_none_or(|at| at.elapsed() >= NOTICE_TOAST_EVERY);
            if toast {
                sessions.last_toast = Some(std::time::Instant::now());
            }
            toast.then_some(notice)
        };
        if let (Some(notice), Some(notifier)) = (notice, self.notifier()) {
            notifier.noticed(&notice);
        }
    }

    /// The notices still shown, oldest first.
    pub fn notices(&self) -> Vec<Notice> {
        let now = self.now_ms();
        self.sessions.lock().unwrap_or_else(|e| e.into_inner()).notices.iter().filter(|n| now.saturating_sub(n.at) < NOTICE_MS).cloned().collect()
    }

    /// The owner has read a notice. Whether there was one.
    pub fn dismiss_notice(&self, id: &str) -> bool {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let before = sessions.notices.len();
        sessions.notices.retain(|n| n.id != id);
        sessions.notices.len() != before
    }

    /// The rings going now, oldest first.
    pub fn active(&self) -> Vec<ActiveRing> {
        let now = self.now_ms();
        self.sessions.lock().unwrap_or_else(|e| e.into_inner()).live.iter().map(|l| ActiveRing { now, ..l.ring.clone() }).collect()
    }

    /// The rings that ended lately, newest last.
    pub fn ended(&self) -> Vec<Ended> {
        self.sessions.lock().unwrap_or_else(|e| e.into_inner()).ended.iter().cloned().collect()
    }

    fn set_note(&self, request: &str, note: &str) {
        if let Some(l) = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).live.iter_mut().find(|l| l.ring.id == request) {
            l.ring.note = note.to_string();
        }
    }

    /// The owner answers a ring in the dialog. See the module docs for what each answer is.
    pub fn respond(&self, request: &str, action: Action) -> Result<Responded, RingError> {
        let Some(call) = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).live.iter().find(|l| l.ring.id == request).map(|l| l.ring.call_id.clone()) else {
            return Err(error(404, "no_ring", "that ring is over"));
        };
        match action {
            Action::Accept => {
                let (ok, note) = match self.plugin() {
                    None => (false, "This computer cannot take the call itself: answer on your Companion (on this computer or your phone).".to_string()),
                    Some(plugin) => match plugin.respond(request, RespondAction::Accept) {
                        Ok(()) => (true, "Asked the Companion to take the call. It is yours when the Companion says so.".to_string()),
                        Err(why) => (false, format!("The phone could not be asked to take the call ({why}). Answer on your Companion.")),
                    },
                };
                self.set_note(request, &note);
                Ok(Responded { ok, note })
            }
            Action::Decline | Action::Message => {
                // The first answer wins: if the phone has already said how it came out, this is late.
                if !self.resolve(request, Outcome::Declined, "desktop") {
                    return Err(error(409, "ring_over", "that ring has just ended"));
                }
                // The receptionist offers the caller a message at once; the plugin is asked to withdraw the request
                // (best effort: a device that accepts before it does is obeyed).
                self.local_outcome(&call, request, Outcome::Declined);
                let withdrawn = self.plugin().is_some_and(|p| p.respond(request, RespondAction::Decline).is_ok());
                let note = match (action, withdrawn) {
                    (Action::Message, true) => "The receptionist will offer to take a message.",
                    (Action::Message, false) => "The receptionist will offer to take a message. Your devices may still ring for a moment.",
                    (_, true) => "Declined. The receptionist will offer to take a message.",
                    (_, false) => "Declined here. The receptionist will offer to take a message; your devices may still ring for a moment.",
                };
                Ok(Responded { ok: true, note: note.to_string() })
            }
        }
    }
}
