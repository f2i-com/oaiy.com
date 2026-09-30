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
//! The owner answers a ring on a Companion, not here: this computer cannot carry the call's audio, and the phone
//! plugin offers the call to the devices the plan names. What the owner can do here is decline (**Decline and take a
//! message**): the phone is asked, on the call's own stream, to withdraw the request (`transfer_cancel`), and its
//! answer decides what happens: `cancelled` and the caller is offered a message, or too late (an owner device had
//! taken it) and nothing is offered while the acceptance goes on. The dialog shows the ring as stopping until then, at
//! most a couple of seconds, and if the phone does not answer the request is over here and a message is offered. If
//! a device accepts after that, the takeover is obeyed. **Not now** is the dialog's own (it puts the box away and
//! rings on).

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::Duration;

use serde::Serialize;

use super::contract::OpenedParams;
use super::host::{Ring, Withdrawal};
use super::plan::RingPlan;
use crate::voice::transfer::{CancelReason, Outcome};

/// How long past its time a ring waits to hear how it came out before it is over.
pub const EXPIRY_GRACE: Duration = Duration::from_secs(5);
/// The most rings that ended kept, for the record.
const KEPT: usize = 200;
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
    /// The owner declined and the phone is being asked to withdraw the request: the dialog waits for its answer.
    pub stopping: bool,
    /// The phone said an owner device had already taken the call when the owner declined: it is being connected, there is nothing left
    /// to decline, and the dialog offers no more to click.
    pub taken: bool,
    /// What the owner was last told about what they asked here.
    pub note: String,
}

struct Live {
    ring: ActiveRing,
    plan: RingPlan,
    /// The plan this ring opened on (which is used up): the plugin saying again that the request is out names it.
    plan_id: String,
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

/// What the owner is told when the phone said an owner device had already taken the call as they declined it.
pub const TAKEN_NOTE: &str = "An owner device took the call just before you declined: it is being connected.";
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
    /// Why nobody could be rung, in a sentence that follows "Someone asked for you."
    pub text: String,
    /// The cause of it, for the page to offer what helps (see [`super::preview::Cause`]).
    pub cause: &'static str,
}

#[derive(Default)]
pub struct Sessions {
    live: Vec<Live>,
    ended: VecDeque<Ended>,
    notices: VecDeque<Notice>,
    last_toast: Option<std::time::Instant>,
}

/// What the owner can do with a ring: end it and have the receptionist take a message. (Taking the call is the Companion's.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Decline it.
    Decline,
    /// Have the receptionist take a message instead.
    Message,
}

impl Action {
    pub fn parse(s: &str) -> Option<Action> {
        match s {
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
        // A request that is going, or is over, is not opened again (a replay, a late duplicate): the ring that is going is the answer, as it is (for
        // the plan it opened on, which is used up), and one that ended stays ended, nobody told again. Nothing is claimed, judged or used up for
        // it: a replay that comes after the caller has said something new must not use those words up as the ask of a ring that is already going.
        {
            let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(live) = sessions.live.iter().find(|l| l.ring.id == params.request_id && l.ring.call_id == params.call_id && l.plan_id == params.plan_id) {
                return Ok(live.ring.clone());
            }
            if sessions.ended.iter().any(|e| e.id == params.request_id) {
                return Err(error(409, "ring_over", "that ring has already ended"));
            }
        }
        if !self.plan_is_claimed(&params.plan_id, &params.call_id) {
            return Err(error(409, "unknown_plan", "that plan was not allowed for this call, or has run out: nothing rings"));
        }
        // The caller hung up while the request was being planned and sent: nobody is rung for a call that is over. The plan is left as it is, so
        // the plugin's refusal of the request (which it must now make) gives the try back.
        if self.call_is_over(&params.call_id) {
            return Err(error(409, "call_ended", "the call ended while the request was being planned: nothing rings"));
        }
        // The call began again since the plan was allowed for it (its session was made anew, or the owner handed the caller back): that request
        // was for another beginning, and is refused as the plugin refuses a call that changed. The plan is left as it is, so its refusal gives
        // the try back.
        if !self.plan_is_current(&params.plan_id, &params.call_id) {
            return Err(error(409, "call_changed", "the call began again since that plan was allowed: nothing rings"));
        }
        // One plan opens one ring: it is used up here, moved out and not copied.
        let Some(taken) = self.take_plan(&params.plan_id, &params.call_id) else {
            return Err(error(409, "unknown_plan", "that plan was not allowed for this call, or has run out: nothing rings"));
        };
        let (plan, judged) = (taken.plan, taken.judged);
        // A request opened a ring: the plugin's consent is there.
        self.note_ring_opened();
        let info = self.call_info(&params.call_id).unwrap_or_default();
        // The ask that this ring is for is acted on now: the next request needs an ask of its own, said after this. What is used up is what the
        // request was judged on, and no more: the caller may have said something since, while the request was being planned and sent, and that is
        // the next request's own. (What the dialog shows is what the caller has said, read just above.)
        self.use_up_asked_turns(&params.call_id, judged);
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
            // What a caller's name is, and what they said, arrive from the phone and the caller: cleaned of control and
            // direction-changing characters before anything shows them.
            caller_name: crate::messages::clean(&info.name, 80),
            caller_number: crate::messages::clean(&info.from, 40),
            said: info.turns.iter().rev().take(SAID_SHOWN).rev().map(|t| crate::messages::clean(t, 300)).collect(),
            started_at: now,
            expires_at,
            now,
            devices,
            stopping: false,
            taken: false,
            note: String::new(),
        };
        let toast = plan.desktop_toast;
        {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            if sessions.live.iter().any(|l| l.ring.id == ring.id) {
                return Ok(ring);
            }
            // The plan moves into the ring it opened: nothing else holds a copy.
            sessions.live.push(Live { ring: ring.clone(), plan, plan_id: params.plan_id.clone() });
        }
        if toast {
            if let Some(notifier) = self.notifier() {
                notifier.ringing(&ring);
            }
        }
        self.watch_expiry(&ring);
        Ok(ring)
    }

    /// The ring is over when its time and a grace have passed with nobody having said how it came out.
    fn watch_expiry(self: &Arc<Self>, ring: &ActiveRing) {
        let (weak, id, call) = (Arc::downgrade(self), ring.id.clone(), ring.call_id.clone());
        let wait = Duration::from_millis(ring.expires_at.saturating_sub(self.now_ms())) + *self.expiry_grace.read().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(move || {
            std::thread::sleep(wait);
            if let Some(ring) = Weak::upgrade(&weak) {
                // Nothing was heard of how it came out: it is over here, and the phone is asked to drop what it may still hold.
                if ring.resolve(&id, Outcome::Expired, "timer") {
                    // Nothing is waited for: the answer, if any, is not read.
                    let _ = ring.cancel_on_call(&call, &id, CancelReason::GaveUp);
                }
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
    pub(super) fn note_no_device(&self, call: &str, info: &super::host::CallInfo, cause: super::preview::Cause) {
        let now = self.now_ms();
        let notice = {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            sessions.notices.retain(|n| now.saturating_sub(n.at) < NOTICE_MS);
            if sessions.notices.iter().any(|n| n.call_id == call) {
                return;
            }
            let notice = Notice { id: format!("notice_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]), call_id: call.to_string(), caller_name: crate::messages::clean(&info.name, 80), caller_number: crate::messages::clean(&info.from, 40), at: now, text: cause.notice_text().to_string(), cause: cause.code() };
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

    /// Set what the owner is told of a ring, and whether it is being stopped. Whether the ring was there and was not already stopping.
    fn set_stopping(&self, request: &str, stopping: bool, note: &str) -> bool {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let Some(l) = sessions.live.iter_mut().find(|l| l.ring.id == request) else { return false };
        let changed = l.ring.stopping != stopping;
        l.ring.stopping = stopping;
        l.ring.note = note.to_string();
        changed
    }

    /// Set the note of a ring that is being stopped, unless the phone has said since that an owner device took the call.
    fn note_stopping(&self, request: &str, note: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = sessions.live.iter_mut().find(|l| l.ring.id == request && l.ring.stopping && !l.ring.taken) {
            l.ring.note = note.to_string();
        }
    }

    /// The phone says it was too late to withdraw the request: an owner device took it. The ring goes on until the phone says so, and
    /// there is nothing left to decline: a second click asks the phone nothing (the contract sends a withdrawal once).
    pub fn cancel_refused(&self, request: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = sessions.live.iter_mut().find(|l| l.ring.id == request) {
            l.ring.stopping = false;
            l.ring.taken = true;
            l.ring.note = TAKEN_NOTE.to_string();
        }
    }

    /// The owner answers a ring in the dialog. See the module docs for what each answer is. What the owner is told is what came of
    /// asking, and never more: not that the phone is being asked before the frame is on the call's stream (or, while the phone has not
    /// yet named the request to the call, that it is waiting to go, which is said).
    pub async fn respond(&self, request: &str, action: Action) -> Result<Responded, RingError> {
        let Some((call, taken)) = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).live.iter().find(|l| l.ring.id == request).map(|l| (l.ring.call_id.clone(), l.ring.taken)) else {
            return Err(error(404, "no_ring", "that ring is over"));
        };
        // An owner device already has the call: there is nothing to decline, and the phone is asked nothing.
        if taken {
            return Ok(Responded { ok: true, note: TAKEN_NOTE.to_string() });
        }
        let reason = match action {
            Action::Decline => CancelReason::OwnerDeclined,
            Action::Message => CancelReason::MessageInstead,
        };
        // A second click while the phone is being asked asks nothing more.
        if !self.set_stopping(request, true, "Asking your Companion to stop ringing. The receptionist will offer the caller a message.") {
            return Ok(Responded { ok: true, note: "Already asking your Companion to stop ringing.".to_string() });
        }
        // The phone is asked, on the call's own stream, and its answer decides what the caller hears: cancelled, and a message
        // is offered; too late, and an owner device has the call. The call says what came of asking.
        let answer = match tokio::time::timeout(Duration::from_secs(3), self.cancel_on_call(&call, request, reason)).await {
            Ok(Ok(answer)) => answer,
            _ => Withdrawal::Unknown,
        };
        let note = match answer {
            Withdrawal::Sent => "Asking your Companion to stop ringing. The receptionist will offer the caller a message.",
            Withdrawal::Queued => "Waiting for the phone to confirm the request, then asking your Companion to stop ringing. The receptionist will offer the caller a message.",
            Withdrawal::Unknown => "Asked; the call has not confirmed that yet. The ring ends when it does.",
            // With no live session to carry the question there is no call to speak to, and with nothing to withdraw there is nothing
            // to wait for: the ring is over here.
            Withdrawal::NoSession => {
                self.resolve(request, Outcome::Declined, "desktop");
                return Ok(Responded { ok: true, note: "Declined. The call is not on this computer any more.".to_string() });
            }
            Withdrawal::Nothing => {
                self.resolve(request, Outcome::Declined, "desktop");
                return Ok(Responded { ok: true, note: "Declined. There was nothing left to stop for this call.".to_string() });
            }
        };
        self.note_stopping(request, note);
        Ok(Responded { ok: true, note: note.to_string() })
    }
}
