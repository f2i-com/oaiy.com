//! The rings going now: what opens one, what ends it (first word wins), what the owner can do in the
//! dialog, and that a ring nobody reports the end of does not ring for ever.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use super::contract::OpenedParams;
use super::plan::{Availability, Device, DeviceKind, Role};
use super::session::Action;
use super::*;
use crate::voice::transfer::{CancelReason, Outcome};

const CALL: &str = "call_1";
const ASKED: &str = "Hello, can I speak to the owner please";

/// The desktop's record of one call, and what the ring asked of it.
struct Calls {
    info: Mutex<Vec<(String, CallInfo)>>,
    /// The phone was asked to withdraw these requests: (call, request, why).
    cancels: Mutex<Vec<(String, String, CancelReason)>>,
    ended: Mutex<Vec<String>>,
    /// The call has a live session that can carry the question to the phone.
    live: std::sync::atomic::AtomicBool,
    /// What the call says of the withdrawal when it has a live session (the frame sent, kept for the phone to name the request, or nothing to withdraw).
    answer: Mutex<Withdrawal>,
    /// The calls the ring said a request was judged on (see `CallSource::consume_turns`), and whether the call then forgets what was said, as the hub does.
    used: Mutex<Vec<String>>,
    consuming: std::sync::atomic::AtomicBool,
    /// The calls that are over (the caller hung up).
    over: Mutex<Vec<String>>,
}

impl Default for Calls {
    fn default() -> Self {
        Self { info: Mutex::default(), cancels: Mutex::default(), ended: Mutex::default(), live: std::sync::atomic::AtomicBool::new(true), answer: Mutex::new(Withdrawal::Sent), used: Mutex::default(), consuming: std::sync::atomic::AtomicBool::new(false), over: Mutex::default() }
    }
}

/// `ring.respond` from a test that is not async: on a runtime of its own.
fn respond(ring: &Ring, request: &str, action: Action) -> Result<crate::ring::session::Responded, RingError> {
    tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(ring.respond(request, action))
}

impl CallSource for Calls {
    fn facts(&self, call: &str) -> Option<CallInfo> {
        self.info.lock().unwrap().iter().find(|(c, _)| c == call).map(|(_, i)| i.clone())
    }
    fn call_ended_by_phone(&self, call: &str) {
        self.ended.lock().unwrap().push(call.to_string());
    }
    fn consume_turns(&self, call: &str) {
        self.used.lock().unwrap().push(call.to_string());
        if self.consuming.load(std::sync::atomic::Ordering::SeqCst) {
            for (c, info) in self.info.lock().unwrap().iter_mut() {
                if c == call {
                    info.turns.clear();
                }
            }
        }
    }
    fn is_over(&self, call: &str) -> bool {
        self.over.lock().unwrap().iter().any(|c| c == call)
    }
    fn cancel_transfer(&self, call: &str, request: &str, reason: CancelReason) -> tokio::sync::oneshot::Receiver<Withdrawal> {
        self.cancels.lock().unwrap().push((call.to_string(), request.to_string(), reason));
        let (reply, answer) = tokio::sync::oneshot::channel();
        let _ = reply.send(if self.live.load(std::sync::atomic::Ordering::SeqCst) { *self.answer.lock().unwrap() } else { Withdrawal::NoSession });
        answer
    }
}

#[derive(Default)]
struct Notified {
    rang: Mutex<Vec<ActiveRing>>,
    ended: Mutex<Vec<(String, String)>>,
    noticed: Mutex<Vec<crate::ring::Notice>>,
}

impl RingNotifier for Notified {
    fn ringing(&self, ring: &ActiveRing) {
        self.rang.lock().unwrap().push(ring.clone());
    }
    fn ended(&self, id: &str, outcome: &str) {
        self.ended.lock().unwrap().push((id.to_string(), outcome.to_string()));
    }
    fn noticed(&self, notice: &crate::ring::Notice) {
        self.noticed.lock().unwrap().push(notice.clone());
    }
}

struct Here(Presence);

impl PresenceSource for Here {
    fn presence(&self) -> Presence {
        self.0
    }
}

struct Phones(Vec<String>);

impl DeviceSource for Phones {
    fn devices(&self, _: &RingSettings) -> Vec<Device> {
        self.0
            .iter()
            .map(|id| Device { id: id.clone(), role: Role::SecondDevice, kind: DeviceKind::Android, call_authority: true, can_take: true, online: true, pushable: false, availability: Availability::Available })
            .collect()
    }
    fn label(&self, id: &str) -> String {
        format!("phone {id}")
    }
}

struct Rig {
    ring: Arc<Ring>,
    calls: Arc<Calls>,
    notified: Arc<Notified>,
}

fn caller(from: &str, name: &str, said: &str) -> CallInfo {
    CallInfo { from: from.into(), name: name.into(), turns: vec!["Hi".into(), said.into()] }
}

fn rig(presence: Presence) -> Rig {
    let ring = Ring::in_memory(RingSettings { enabled: true, ..Default::default() });
    ring.set_presence(Arc::new(Here(presence)));
    ring.set_devices(crate::ring::testing::at_the_pc());
    let calls = Arc::new(Calls::default());
    calls.info.lock().unwrap().push((CALL.into(), caller("+61491570006", "Alex", ASKED)));
    ring.set_calls(calls.clone());
    let notified = Arc::new(Notified::default());
    ring.set_notifier(notified.clone());
    Rig { ring, calls, notified }
}

/// The plan the plugin asked this desktop for, and got.
fn planned(ring: &Arc<Ring>, call: &str) -> String {
    let authorised = ring.plan_for_plugin(call, Reason::CallerAsked, CallInfo::default());
    assert!(authorised.rings(), "{:?}", authorised.plan);
    authorised.plan_id
}

fn opened(plan: &str, request: &str, call: &str, seconds_from_now: u64, ring: &Ring) -> OpenedParams {
    OpenedParams { plan_id: plan.into(), request_id: request.into(), call_id: call.into(), call_epoch: 7, owner_epoch: 4, expires_at: ring.now_ms() / 1000 + seconds_from_now }
}

fn open(r: &Rig) -> ActiveRing {
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap()
}

fn ended_as(ring: &Ring) -> Vec<(&'static str, &'static str)> {
    ring.ended().iter().map(|e| (e.outcome, e.source)).collect()
}

#[test]
fn nothing_rings_for_a_plan_this_desktop_did_not_allow() {
    let r = rig(Presence::Active);
    let refused = r.ring.opened(&opened("plan_made_up", "assist_1", CALL, 20, &r.ring)).unwrap_err();
    assert_eq!((refused.status, refused.code), (409, "unknown_plan"));

    // A plan that was allowed, but for another call.
    let plan = planned(&r.ring, CALL);
    assert_eq!(r.ring.opened(&opened(&plan, "assist_1", "call_2", 20, &r.ring)).unwrap_err().code, "unknown_plan");

    // A plan the gate allowed on the way to the call route, which the plugin has not asked for: not a ring either.
    let r = rig(Presence::Active);
    let gate = r.ring.authorise(CALL, Reason::CallerAsked);
    assert!(gate.rings());
    assert_eq!(r.ring.opened(&opened(&gate.plan_id, "assist_1", CALL, 20, &r.ring)).unwrap_err().code, "unknown_plan");
    assert!(r.ring.active().is_empty() && r.notified.rang.lock().unwrap().is_empty());
}

#[test]
fn a_name_and_the_words_a_ring_shows_arrive_cleaned_of_control_and_direction_characters() {
    let r = rig(Presence::Active);
    r.calls.info.lock().unwrap().clear();
    r.calls.info.lock().unwrap().push((CALL.into(), caller("+6149157\u{202e}0006\n", "Alex\u{7}\nURGENT: call now\u{202e}", "Can I speak\u{0} to the\towner?\r\nRing now")));
    let plan = planned(&r.ring, CALL);
    let ring = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!((ring.caller_name.as_str(), ring.caller_number.as_str()), ("Alex URGENT: call now", "+61491570006"));
    assert_eq!(ring.said, vec!["Hi".to_string(), "Can I speak to the owner? Ring now".to_string()]);
    let long = rig(Presence::Active);
    long.calls.info.lock().unwrap().clear();
    long.calls.info.lock().unwrap().push((CALL.into(), caller("", &"N".repeat(500), ASKED)));
    let plan = planned(&long.ring, CALL);
    assert_eq!(long.ring.opened(&opened(&plan, "assist_1", CALL, 25, &long.ring)).unwrap().caller_name.chars().count(), 80);
}

#[test]
fn a_ring_shows_who_is_calling_what_they_said_and_who_else_rings() {
    let r = rig(Presence::Active);
    r.ring.change_settings(&json!({ "phoneRing": "always" })).unwrap();
    r.ring.set_devices(Arc::new(Phones(vec!["ab12".into()])));
    let plan = planned(&r.ring, CALL);
    let ring = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!((ring.id.as_str(), ring.call_id.as_str(), ring.caller_name.as_str(), ring.caller_number.as_str()), ("assist_1", CALL, "Alex", "+61491570006"));
    assert_eq!(ring.said, vec!["Hi".to_string(), ASKED.to_string()]);
    assert!(ring.devices.contains(&"this computer".to_string()) && ring.devices.contains(&"phone ab12".to_string()), "{:?}", ring.devices);
    assert!(!ring.stopping && ring.note.is_empty(), "nothing has been asked of it");
    assert!(ring.expires_at > ring.started_at && ring.expires_at - ring.started_at <= 26_000, "{}", ring.expires_at - ring.started_at);

    // The owner is told once, and the dialog lists it.
    assert_eq!(r.notified.rang.lock().unwrap().len(), 1);
    assert_eq!(r.ring.active().len(), 1);
    // The same ring reported again is the same ring, and tells nobody again.
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!((r.ring.active().len(), r.notified.rang.lock().unwrap().len()), (1, 1));
}

#[test]
fn a_ring_that_does_not_ring_this_computer_shows_in_no_notification() {
    // The owner is not at the computer: the plan rings the phones only, and this desktop raises nothing.
    let r = rig(Presence::Idle);
    r.ring.set_devices(Arc::new(Phones(vec!["ab12".into()])));
    let plan = planned(&r.ring, CALL);
    let ring = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!(ring.devices, vec!["phone ab12".to_string()]);
    assert!(r.notified.rang.lock().unwrap().is_empty(), "no notification for a ring this computer is not part of");
    r.ring.resolve("assist_1", Outcome::Expired, "timer");
    assert!(r.notified.ended.lock().unwrap().is_empty());

    // Nobody to ring at all: not a ring.
    let r = rig(Presence::Idle);
    let plan = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(!plan.rings() && !plan.plan_id.is_empty());
}

/// The tries counted for `CALL` now.
fn tries(r: &Rig) -> u32 {
    r.ring.attempts.lock().unwrap().counters(CALL, &crate::ring::host::caller_key("+61491570006"), r.ring.clock().unix()).attempts_this_call
}

#[test]
fn a_ring_is_not_opened_for_a_call_that_ended_while_the_request_was_being_planned() {
    let r = rig(Presence::Active);
    let plan = planned(&r.ring, CALL);
    assert_eq!(tries(&r), 1, "the try is counted when the plan is allowed");
    // The caller hangs up after the plan was allowed and before the plugin says the request is out.
    r.calls.over.lock().unwrap().push(CALL.to_string());
    let refused = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap_err();
    assert_eq!((refused.status, refused.code), (409, "call_ended"));
    // A plan this desktop did not allow is refused as it always was, whether or not the call is over.
    assert_eq!(r.ring.opened(&opened("plan_made_up", "assist_1", CALL, 25, &r.ring)).unwrap_err().code, "unknown_plan");
    assert!(r.ring.active().is_empty() && r.notified.rang.lock().unwrap().is_empty(), "nothing rings, and nobody is told");
    assert!(r.calls.used.lock().unwrap().is_empty(), "the ask of a call that is over is not acted on");
    // The plugin then refuses the request itself, as it must: nobody was rung, so the try is given back.
    assert!(r.ring.request_refused(CALL));
    assert_eq!(tries(&r), 0);
    // A call that is over is not planned for either: nothing rings and no try is counted.
    let again = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(!again.rings() && tries(&r) == 0, "{:?}", again.plan);
    assert!(!r.ring.authorise(CALL, Reason::CallerAsked).rings() && tries(&r) == 0, "nor is a request from the call's own gate allowed");
    // A call that begins again (the owner handed the caller back) is a call like any other.
    r.calls.over.lock().unwrap().clear();
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_2", CALL, 25, &r.ring)).unwrap();
    assert_eq!(r.ring.active().len(), 1);
}

#[test]
fn a_plan_opens_one_ring_and_the_same_request_again_is_the_ring_that_is_going_and_uses_nothing_up() {
    let r = rig(Presence::Active);
    r.calls.consuming.store(true, std::sync::atomic::Ordering::SeqCst);
    let plan = planned(&r.ring, CALL);
    let first = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!(r.calls.used.lock().unwrap().len(), 1, "the ask was acted on once");
    // A second request on the same plan is not a ring: the plan was for the one that opened first.
    let second = r.ring.opened(&opened(&plan, "assist_2", CALL, 25, &r.ring)).unwrap_err();
    assert_eq!((second.status, second.code), (409, "unknown_plan"));
    assert_eq!((r.ring.active().len(), r.notified.rang.lock().unwrap().len()), (1, 1), "one ring, one notification");
    assert_eq!(r.calls.used.lock().unwrap().len(), 1, "and nothing more was used up");
    // The caller says something new, and the plugin says again that the first request is out (a replay): the ring that is going, as it is, and
    // the new words are not used up as the ask of a ring that is already going.
    r.calls.info.lock().unwrap()[0].1.turns.push("Actually, could I speak to the manager instead?".into());
    let again = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert_eq!((again.id.as_str(), again.started_at, again.expires_at), (first.id.as_str(), first.started_at, first.expires_at), "the same ring");
    assert_eq!(r.calls.used.lock().unwrap().len(), 1, "a replay uses nothing up");
    assert_eq!(r.calls.info.lock().unwrap()[0].1.turns.len(), 1, "the words they said since are still theirs to ask on");
    assert_eq!(r.notified.rang.lock().unwrap().len(), 1, "nobody is told again");
    // Only the plan it opened on says it again: a plan this desktop did not allow does not, though the request is going.
    assert_eq!(r.ring.opened(&opened("plan_made_up", "assist_1", CALL, 25, &r.ring)).unwrap_err().code, "unknown_plan");
    // A plan is for its own call, and a request that is going on another call is not this one.
    let other = rig(Presence::Active);
    let plan = planned(&other.ring, CALL);
    other.ring.opened(&opened(&plan, "assist_1", CALL, 25, &other.ring)).unwrap();
    assert_eq!(other.ring.opened(&opened(&plan, "assist_1", "call_2", 25, &other.ring)).unwrap_err().code, "unknown_plan");
}

#[test]
fn a_ring_that_ended_stays_ended_when_the_plugin_says_again_that_it_is_out() {
    let r = rig(Presence::Active);
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert!(r.ring.resolve("assist_1", Outcome::Declined, "phone"));
    // A replay, a duplicate, a late report of the same request: no dialog comes back and nobody is told again.
    let again = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap_err();
    assert_eq!((again.status, again.code), (409, "ring_over"));
    assert!(r.ring.active().is_empty());
    assert_eq!(r.notified.rang.lock().unwrap().len(), 1, "the owner was told once");
    assert_eq!(ended_as(&r.ring), vec![("declined", "phone")]);
}

#[test]
fn the_first_word_about_a_ring_ends_it_and_later_words_change_nothing() {
    let r = rig(Presence::Active);
    open(&r);
    assert!(r.ring.resolve("assist_1", Outcome::Accepted, "phone"), "the phone said first");
    assert!(!r.ring.resolve("assist_1", Outcome::Declined, "desktop"), "a decline after it changes nothing");
    assert!(!r.ring.resolve("assist_1", Outcome::Expired, "timer"));
    assert!(r.ring.active().is_empty());
    assert_eq!(r.notified.ended.lock().unwrap().as_slice(), [("assist_1".to_string(), "accepted".to_string())], "the notification is closed once, with how it came out");
    assert_eq!(ended_as(&r.ring), vec![("accepted", "phone")]);
    // The owner's answer that arrives after: the ring is over, and the call is told nothing.
    assert_eq!(respond(&r.ring, "assist_1", Action::Decline).unwrap_err().code, "no_ring");
    assert!(r.calls.cancels.lock().unwrap().is_empty(), "the phone was asked nothing about a ring that was over");
}

#[test]
fn the_owner_takes_a_call_on_a_companion_so_the_dialog_has_no_accept() {
    assert_eq!(Action::parse("accept"), None, "this computer cannot carry the call's audio and has no command to ask the phone to take it");
    assert_eq!((Action::parse("decline"), Action::parse("message")), (Some(Action::Decline), Some(Action::Message)));
    // The phone says a device took it: the ring is over, once.
    let r = rig(Presence::Active);
    open(&r);
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    assert!(r.ring.active().is_empty());
    assert_eq!(ended_as(&r.ring), vec![("accepted", "phone")]);
}

#[test]
fn declining_asks_the_phone_to_withdraw_the_request_and_the_ring_stays_until_the_phone_answers() {
    for (action, why) in [(Action::Decline, CancelReason::OwnerDeclined), (Action::Message, CancelReason::MessageInstead)] {
        let r = rig(Presence::Active);
        open(&r);
        let said = respond(&r.ring, "assist_1", action).unwrap();
        assert!(said.ok && said.note.contains("Asking your Companion to stop ringing") && said.note.contains("offer the caller a message"), "{}", said.note);
        assert_eq!(r.calls.cancels.lock().unwrap().as_slice(), [(CALL.to_string(), "assist_1".to_string(), why)], "{action:?}: the phone is asked, and why");
        // Nothing is decided here: it is stopping, and shown so, until the phone answers.
        let shown = r.ring.active();
        assert_eq!(shown.len(), 1, "{action:?}");
        assert!(shown[0].stopping && shown[0].note == said.note, "{:?}", shown[0]);
        assert!(r.notified.ended.lock().unwrap().is_empty());
        // A second click asks nothing more.
        let again = respond(&r.ring, "assist_1", action).unwrap();
        assert!(again.ok && again.note.contains("Already asking"), "{}", again.note);
        assert_eq!(r.calls.cancels.lock().unwrap().len(), 1, "{action:?}: once");
        // The phone answers that it withdrew it: the ring is over.
        r.ring.outcome_seen("assist_1", Outcome::Cancelled, "phone");
        assert!(r.ring.active().is_empty(), "{action:?}");
        assert_eq!(ended_as(&r.ring), vec![("cancelled", "phone")]);
        assert_eq!(r.notified.ended.lock().unwrap().as_slice(), [("assist_1".to_string(), "cancelled".to_string())]);
    }
}

#[test]
fn too_late_to_withdraw_means_a_device_has_it_the_ring_goes_on_and_the_owner_is_told() {
    let r = rig(Presence::Active);
    open(&r);
    respond(&r.ring, "assist_1", Action::Decline).unwrap();
    r.ring.cancel_refused("assist_1");
    let shown = r.ring.active();
    assert_eq!(shown.len(), 1, "the ring is not over: the acceptance is coming");
    assert!(!shown[0].stopping && shown[0].note.contains("took the call just before you declined"), "{:?}", shown[0]);
    // The acceptance arrives, and is what ends it.
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    assert_eq!(ended_as(&r.ring), vec![("accepted", "phone")]);
}

#[test]
fn what_the_owner_is_told_is_what_came_of_asking_the_phone_and_never_more() {
    // Kept for the phone to name the request to the call: said, and the ring waits, stopping.
    let r = rig(Presence::Active);
    *r.calls.answer.lock().unwrap() = Withdrawal::Queued;
    open(&r);
    let said = respond(&r.ring, "assist_1", Action::Decline).unwrap();
    assert!(said.note.contains("Waiting for the phone to confirm") && !said.note.starts_with("Asking your Companion"), "{}", said.note);
    let shown = r.ring.active();
    assert!(shown.len() == 1 && shown[0].stopping && !shown[0].taken && shown[0].note == said.note, "{:?}", shown[0]);
    // The call did not say in time: not that it was sent.
    let r = rig(Presence::Active);
    *r.calls.answer.lock().unwrap() = Withdrawal::Unknown;
    open(&r);
    let said = respond(&r.ring, "assist_1", Action::Decline).unwrap();
    assert!(said.note.contains("has not confirmed") && r.ring.active()[0].stopping, "{}", said.note);
    // Nothing to withdraw: the ring is over here, and the owner is told there was nothing to stop.
    let r = rig(Presence::Active);
    *r.calls.answer.lock().unwrap() = Withdrawal::Nothing;
    open(&r);
    let said = respond(&r.ring, "assist_1", Action::Message).unwrap();
    assert!(said.ok && said.note.contains("nothing left to stop"), "{}", said.note);
    assert!(r.ring.active().is_empty());
    assert_eq!(ended_as(&r.ring), vec![("declined", "desktop")]);
}

#[test]
fn once_the_phone_says_an_owner_device_has_the_call_a_click_asks_it_nothing_more() {
    let r = rig(Presence::Active);
    open(&r);
    respond(&r.ring, "assist_1", Action::Decline).unwrap();
    r.ring.cancel_refused("assist_1");
    let shown = r.ring.active();
    assert!(shown[0].taken && !shown[0].stopping, "{:?}", shown[0]);
    for action in [Action::Decline, Action::Message, Action::Decline] {
        let said = respond(&r.ring, "assist_1", action).unwrap();
        assert!(said.ok && said.note.contains("took the call just before you declined"), "{}", said.note);
    }
    assert_eq!(r.calls.cancels.lock().unwrap().len(), 1, "the phone was asked once, whatever was clicked after");
    // A note that arrives late does not undo it.
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    assert!(r.ring.active().is_empty());
}

#[test]
fn a_call_with_no_live_session_has_nobody_to_ask_so_the_ring_is_over_here() {
    let r = rig(Presence::Active);
    r.calls.live.store(false, std::sync::atomic::Ordering::SeqCst);
    open(&r);
    let said = respond(&r.ring, "assist_1", Action::Decline).unwrap();
    assert!(said.ok && said.note.contains("not on this computer any more"), "{}", said.note);
    assert!(r.ring.active().is_empty());
    assert_eq!(ended_as(&r.ring), vec![("declined", "desktop")]);
}

#[test]
fn a_decline_that_loses_to_the_phone_reaches_nobody() {
    let r = rig(Presence::Active);
    open(&r);
    // The phone says accepted between the owner's click and the desktop hearing it.
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    let late = respond(&r.ring, "assist_1", Action::Decline).unwrap_err();
    assert_eq!((late.status, late.code), (404, "no_ring"));
    assert!(r.calls.cancels.lock().unwrap().is_empty(), "the phone was not asked to withdraw a call an owner device took");
}

#[test]
fn a_call_that_ends_ends_what_rings_for_it() {
    let r = rig(Presence::Active);
    open(&r);
    r.ring.call_finished("call_2");
    assert_eq!(r.ring.active().len(), 1, "another call's end does not");
    r.ring.call_finished(CALL);
    assert!(r.ring.active().is_empty());
    assert_eq!(ended_as(&r.ring), vec![("cancelled", "call")]);
    assert_eq!(r.notified.ended.lock().unwrap().len(), 1);
}

#[test]
fn a_ring_nobody_reports_the_end_of_is_over_after_its_time_and_a_grace() {
    let r = rig(Presence::Active);
    r.ring.set_expiry_grace(Duration::from_millis(50));
    let plan = planned(&r.ring, CALL);
    // The phone gave the request a second to live.
    r.ring.opened(&opened(&plan, "assist_1", CALL, 1, &r.ring)).unwrap();
    assert_eq!(r.ring.active().len(), 1);
    let until = std::time::Instant::now() + Duration::from_secs(5);
    while !r.ring.active().is_empty() && std::time::Instant::now() < until {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(r.ring.active().is_empty(), "the dialog does not ring for ever");
    assert_eq!(ended_as(&r.ring), vec![("expired", "timer")]);
    assert_eq!(r.notified.ended.lock().unwrap().as_slice(), [("assist_1".to_string(), "expired".to_string())]);
    // Nothing was heard of it, so the phone is asked to drop what it may still hold.
    assert_eq!(r.calls.cancels.lock().unwrap().as_slice(), [(CALL.to_string(), "assist_1".to_string(), CancelReason::GaveUp)]);
}

#[test]
fn a_ring_answered_in_time_is_not_ended_again_by_its_timer() {
    let r = rig(Presence::Active);
    r.ring.set_expiry_grace(Duration::from_millis(30));
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 1, &r.ring)).unwrap();
    assert!(r.ring.resolve("assist_1", Outcome::Declined, "phone"));
    std::thread::sleep(Duration::from_millis(1_300));
    assert_eq!(ended_as(&r.ring), vec![("declined", "phone")], "one record: the timer found nothing to end");
    assert_eq!(r.notified.ended.lock().unwrap().len(), 1);
    assert!(r.calls.cancels.lock().unwrap().is_empty(), "a ring the phone answered is not withdrawn by the timer");
}

#[test]
fn the_time_the_phone_gives_is_believed_only_when_it_makes_sense() {
    // In the past, and hours away: the plan's own time is used (30 seconds for this computer alone).
    for (n, offset) in [(1u64, -50i64), (2, 100_000)] {
        let r = rig(Presence::Active);
        let plan = planned(&r.ring, CALL);
        let mut p = opened(&plan, "assist_1", CALL, 0, &r.ring);
        p.expires_at = u64::try_from(i64::try_from(r.ring.now_ms() / 1000).unwrap() + offset).unwrap();
        let ring = r.ring.opened(&p).unwrap();
        assert!((29_000..=31_000).contains(&(ring.expires_at - ring.started_at)), "{n}: {}", ring.expires_at - ring.started_at);
    }
}

/// An approved phone, and nothing ticked as this computer's Companion: the setup of an owner with one Companion, on a second phone.
fn only_a_phone(r: &Rig) {
    r.ring.set_devices(Arc::new(crate::ring::testing::Devices(vec![crate::ring::testing::android("ph1")])));
}

/// ...whose owner said phones never ring: nothing a call can be offered to.
fn phones_never_ring(r: &Rig) {
    only_a_phone(r);
    r.ring.change_settings(&json!({ "phoneRing": "never" })).unwrap();
}

#[test]
fn a_toast_alone_is_not_a_ring_so_nobody_is_planned_for_and_no_try_is_counted() {
    let r = rig(Presence::Active);
    phones_never_ring(&r);
    // The reference plans a ring for the owner at their computer with only the toast, and this desktop does not: a plugin that
    // offers a transfer only to the devices a plan names would open nothing for it.
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!((plan.plan.decision, plan.plan.reason), (Decision::MessageOnly, PlanReason::NoEndpoint), "{:?}", plan.plan);
    assert_eq!(crate::ring::host::plan_result(&plan)["reason"], "no_endpoint", "the plugin's word for a plan that names nobody; a reason it does not know it turns into plan_unavailable");
    assert!(!plan.rings() && !plan.plan_id.is_empty(), "a plan that rings nobody has an id all the same");
    assert!(!plan.plan.desktop_toast && plan.plan.targets().is_empty());
    // The plugin asking is answered the same way, and nothing was counted for either.
    let asked = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert_eq!((asked.plan.decision, asked.plan.reason), (Decision::MessageOnly, PlanReason::NoEndpoint));
    let counters = r.ring.attempts.lock().unwrap().counters(CALL, "491570006", r.ring.clock().unix());
    assert_eq!((counters.attempts_this_call, counters.global_attempts_last_hour), (0, 0), "no try was spent on a ring that cannot happen");
    assert!(r.ring.active().is_empty() && r.notified.rang.lock().unwrap().is_empty());
    // The same caller can still be put through once a device is set up: nothing was used up.
    r.ring.set_devices(crate::ring::testing::at_the_pc());
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings());
}

#[test]
fn the_owner_at_their_computer_with_only_a_phone_approved_is_rung_on_the_phone_by_default() {
    // The setup of an owner with one Companion, on a second phone, and the settings as they come: "ring when I am away" holds when there is no
    // Companion on this computer to take the call instead, so a caller who asks for the owner rings the phone, whatever the owner is doing.
    for presence in [Presence::Active, Presence::Idle, Presence::Locked, Presence::Off] {
        let r = rig(presence);
        only_a_phone(&r);
        let plan = r.ring.authorise(CALL, Reason::CallerAsked);
        assert!(plan.rings(), "{presence:?}: {:?}", plan.plan);
        assert_eq!(plan.plan.phones, vec!["ph1".to_string()], "{presence:?}");
        assert_eq!(plan.plan.desktop_toast, presence == Presence::Active, "{presence:?}: this computer rings too while they are at it");
        assert!(r.ring.notices().is_empty(), "{presence:?}: nothing to tell the owner");
    }
    // The owner who said phones ring only when they are away and has this computer's Companion set up keeps that: at the computer it rings.
    let r = rig(Presence::Active);
    r.ring.set_devices(Arc::new(crate::ring::testing::Devices(vec![crate::ring::testing::android("ph1"), crate::ring::testing::windows("pc1")])));
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!((plan.plan.phones.clone(), plan.plan.desktop_companions.clone()), (vec![], vec!["pc1".to_string()]));
    // Never means never.
    let r = rig(Presence::Active);
    phones_never_ring(&r);
    assert!(!r.ring.authorise(CALL, Reason::CallerAsked).rings());
    // A phone set to never ring, by device, is left out too.
    let r = rig(Presence::Active);
    only_a_phone(&r);
    r.ring.change_settings(&json!({ "desktopRing": "never" })).unwrap();
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings(), "this computer set never to ring does not stop a phone from ringing");
}

/// What a caller who asks for the owner would get, for `devices` and `settings`, with the owner `presence`.
fn previewed(presence: Presence, devices: Vec<Device>, change: serde_json::Value) -> crate::ring::Preview {
    let r = rig(presence);
    r.ring.set_devices(Arc::new(crate::ring::testing::Devices(devices)));
    r.ring.change_settings(&change).unwrap();
    r.ring.preview()
}

#[test]
fn the_preview_says_what_a_caller_would_get_in_each_state_of_the_owner_and_of_their_devices() {
    use crate::ring::testing::{android, windows};
    let none = json!({});
    // One Companion, on a second phone (the user's own setup): it rings whatever the owner is doing.
    for presence in [Presence::Active, Presence::Idle, Presence::Locked, Presence::Off] {
        let p = previewed(presence, vec![android("ph1")], none.clone());
        assert!(p.enabled && p.rings && p.cause.is_none() && p.devices.iter().any(|d| d.contains("ph1")), "{presence:?}: {p:?}");
        assert!(p.text.starts_with("Right now a caller who asks for you would ring:") && p.text.contains("ph1"), "{presence:?}: {}", p.text);
    }
    // One Companion, the Windows one on this computer: it rings while the owner is at the computer, and only then.
    let p = previewed(Presence::Active, vec![windows("pc1")], none.clone());
    assert!(p.rings && p.devices.iter().any(|d| d == "this computer") && p.devices.iter().any(|d| d.contains("pc1")), "{p:?}");
    for presence in [Presence::Idle, Presence::Locked, Presence::Off] {
        let p = previewed(presence, vec![windows("pc1")], none.clone());
        assert!(!p.rings && p.cause == Some("onlyThisComputers"), "{presence:?}: {p:?}");
        assert!(p.text.contains("which you are not now") && p.text.contains("untick"), "{presence:?}: the way out is said: {}", p.text);
    }
    // A phone the owner ticked as this computer's (the mistake the page's advice made easy): at the computer it rings, away it does not, and
    // the page says so and how to put it right.
    let ticked = vec![windows("ph1")];
    assert!(previewed(Presence::Active, ticked.clone(), none.clone()).rings);
    assert_eq!(previewed(Presence::Idle, ticked, none.clone()).cause, Some("onlyThisComputers"));
    // Both: the computer's Companion while they are at it, the phone when they are away.
    let both = || vec![android("ph1"), windows("pc1")];
    let at = previewed(Presence::Active, both(), none.clone());
    assert!(at.rings && at.devices.iter().any(|d| d.contains("pc1")) && !at.devices.iter().any(|d| d.contains("ph1")), "{at:?}");
    let away = previewed(Presence::Idle, both(), none.clone());
    assert!(away.rings && away.devices.iter().any(|d| d.contains("ph1")) && !away.devices.iter().any(|d| d.contains("pc1")), "{away:?}");
    // None approved, all set never to ring, phones set never to ring.
    for presence in [Presence::Active, Presence::Off] {
        let p = previewed(presence, vec![], none.clone());
        assert!(!p.rings && p.cause == Some("noCompanion") && p.text.contains("No Companion is approved yet"), "{presence:?}: {p:?}");
    }
    let p = previewed(Presence::Active, vec![], json!({ "excludedDevices": ["ph1"] }));
    assert_eq!(p.cause, Some("allNever"));
    let p = previewed(Presence::Active, vec![android("ph1")], json!({ "phoneRing": "never" }));
    assert!(!p.rings && p.cause == Some("phonesOff") && p.text.contains("Set Phones to"), "{p:?}");
    // Quiet hours are a reason of their own, and transfers off says nothing.
    let quiet = json!({ "quietHours": { "enabled": true, "start": "00:00", "end": "23:59", "days": 127, "allowUrgent": false, "allowVip": false } });
    let p = previewed(Presence::Active, vec![android("ph1")], quiet);
    assert!(!p.rings && p.cause.is_none() && p.text.contains("quiet hours"), "{p:?}");
    let p = previewed(Presence::Active, vec![android("ph1")], json!({ "enabled": false }));
    assert!(!p.enabled && !p.rings && p.text.is_empty() && p.plugin.is_none());
    // Nothing is counted for asking.
    let r = rig(Presence::Active);
    only_a_phone(&r);
    r.ring.preview();
    let counters = r.ring.attempts.lock().unwrap().counters(CALL, "491570006", r.ring.clock().unix());
    assert_eq!((counters.attempts_this_call, counters.global_attempts_last_hour), (0, 0));
}

#[test]
fn the_page_is_told_when_the_phone_plugin_does_not_offer_calls_for_transfer_and_why() {
    let r = rig(Presence::Active);
    only_a_phone(&r);
    // Nothing has happened yet, or the calls were offered, or transfers were off: nothing to say.
    assert!(r.ring.preview().plugin.is_none());
    r.ring.note_call(false, false);
    assert!(r.ring.preview().plugin.is_none(), "a call that began with transfers off says nothing about the plugin");
    r.ring.note_call(true, true);
    assert!(r.ring.preview().plugin.is_none());
    // A plugin that never offered a call: too old.
    let old = rig(Presence::Active);
    only_a_phone(&old);
    old.ring.note_call(true, false);
    let line = old.ring.preview().plugin.unwrap();
    assert!(line.contains("does not support transfers yet") && line.contains("offered a message"), "{line}");
    // One that offered before and does not now: its consent for taking calls.
    r.ring.note_call(true, false);
    let line = r.ring.preview().plugin.unwrap();
    assert!(line.contains("consent settings") && !line.contains("does not support"), "{line}");
    // Offered again: it is well.
    r.ring.note_call(true, true);
    assert!(r.ring.preview().plugin.is_none());
    // With no Companion approved, that is the reason it does not offer.
    let none = rig(Presence::Active);
    none.ring.set_devices(Arc::new(crate::ring::testing::Devices(vec![])));
    none.ring.note_call(true, false);
    assert!(none.ring.preview().plugin.unwrap().contains("No Companion is approved"));
    // A request the plugin refused for want of consent says so, until a ring opens.
    r.ring.note_consent_refused();
    assert!(r.ring.preview().plugin.unwrap().contains("consent settings"));
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert!(r.ring.preview().plugin.is_none(), "a ring opened: consent is there");
    // Transfers off: nothing is said at all.
    old.ring.change_settings(&json!({ "enabled": false })).unwrap();
    assert!(old.ring.preview().plugin.is_none());
}

#[tokio::test]
async fn the_preview_is_read_over_http() {
    let r = rig(Presence::Active);
    only_a_phone(&r);
    let base = serve(r.ring.clone()).await;
    let read: serde_json::Value = reqwest::get(format!("{base}/api/ring/preview")).await.unwrap().json().await.unwrap();
    assert_eq!((read["enabled"].clone(), read["rings"].clone(), read["cause"].clone()), (json!(true), json!(true), json!(null)), "{read}");
    assert!(read["devices"].as_array().unwrap().iter().any(|d| d == "this computer"), "{read}");
}

#[test]
fn the_owner_away_still_rings_their_phone_and_only_the_toast_needs_the_computers_companion() {
    let r = rig(Presence::Idle);
    only_a_phone(&r);
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert!(plan.rings(), "{:?}", plan.plan);
    assert_eq!(plan.plan.phones, vec!["ph1".to_string()]);
    assert!(!plan.plan.desktop_toast);
    // At the computer with a phone and the computer's Companion both set up: the Companion on the computer is named.
    let r = rig(Presence::Active);
    r.ring.set_devices(Arc::new(crate::ring::testing::Devices(vec![crate::ring::testing::android("ph1"), crate::ring::testing::windows("pc1")])));
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert!(plan.rings() && plan.plan.desktop_toast);
    assert_eq!((plan.plan.phones.clone(), plan.plan.desktop_companions.clone()), (vec![], vec!["pc1".to_string()]), "a phone rings when the owner is away");
}

#[test]
fn the_owner_is_told_once_a_call_that_nobody_could_be_rung_for_want_of_a_device_and_can_dismiss_it() {
    let r = rig(Presence::Active);
    phones_never_ring(&r);
    r.ring.authorise(CALL, Reason::CallerAsked);
    r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    let notices = r.ring.notices();
    assert_eq!(notices.len(), 1, "once a call, whoever asked");
    let n = &notices[0];
    assert_eq!((n.call_id.as_str(), n.caller_name.as_str(), n.caller_number.as_str(), n.text.as_str(), n.cause), (CALL, "Alex", "+61491570006", crate::ring::Cause::PhonesOff.notice_text(), "phonesOff"));
    // A phone is approved and set to not ring: never told that no device is set up.
    assert!(n.text.contains("Your phone is set to not ring") && n.text.contains("offered a message") && !n.text.contains("No device"), "{}", n.text);
    assert_eq!(r.notified.noticed.lock().unwrap().len(), 1, "and the desktop was asked to tell the owner");
    // A ring is not what this is: nothing is offered to accept.
    assert!(r.ring.active().is_empty());
    // Another call: its own notice, but the native notification is not raised again within ten minutes.
    r.calls.info.lock().unwrap().push(("call_2".into(), caller("+61491570156", "Sam", ASKED)));
    r.ring.authorise("call_2", Reason::CallerAsked);
    assert_eq!(r.ring.notices().len(), 2);
    assert_eq!(r.notified.noticed.lock().unwrap().len(), 1, "the chime is not repeated for a caller who rings again");
    // The owner dismisses one.
    assert!(r.ring.dismiss_notice(&n.id));
    assert!(!r.ring.dismiss_notice(&n.id), "and it is gone");
    assert_eq!(r.ring.notices().len(), 1);
    // Only so many are kept.
    for k in 3..12 {
        let call = format!("call_{k}");
        r.calls.info.lock().unwrap().push((call.clone(), caller("", "Sam", ASKED)));
        r.ring.authorise(&call, Reason::CallerAsked);
    }
    assert_eq!(r.ring.notices().len(), 5);
}

#[test]
fn a_quiet_hour_or_a_disabled_setting_is_not_a_missing_device() {
    // Transfers off: message only for its own reason, and no notice about devices.
    let r = rig(Presence::Active);
    phones_never_ring(&r);
    r.ring.change_settings(&json!({ "enabled": false })).unwrap();
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!(plan.plan.reason, PlanReason::Disabled);
    assert!(r.ring.notices().is_empty());
}

/// The owner allows the receptionist to ask for urgent things, and the caller said one of their phrases.
fn urgent_call(said: &str, initiative: &str) -> Rig {
    let r = rig(Presence::Active);
    r.ring.change_settings(&json!({ "initiative": initiative, "urgentPhrases": ["gas leak"] })).unwrap();
    r.calls.info.lock().unwrap().clear();
    r.calls.info.lock().unwrap().push((CALL.into(), caller("+61491570006", "Alex", said)));
    r
}

#[test]
fn this_desktop_vouches_for_an_urgent_reason_only_when_the_owner_allowed_it_and_their_own_phrase_was_heard() {
    // Allowed, and the caller's own words are the owner's urgent phrase: the plan carries reasonAllowed.
    let r = urgent_call("There is a gas leak in the shop", "on_request_or_urgent");
    let plan = r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default());
    assert!(plan.rings() && plan.reason_allowed, "{:?}", plan.plan);
    assert_eq!(super::host::plan_result(&plan)["reasonAllowed"], json!(true));
    // The gate's plan, which the plugin then asks for, is vouched the same way and the vouching is not lost on the way.
    let r = urgent_call("There is a gas leak in the shop", "on_request_or_urgent");
    let gate = r.ring.authorise(CALL, Reason::Urgent);
    assert!(gate.rings() && gate.reason_allowed);
    let asked = r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default());
    assert_eq!(asked.plan_id, gate.plan_id);
    assert!(asked.reason_allowed, "the plan the plugin is given is the one that was vouched for");
}

#[test]
fn vouching_needs_the_urgent_reason_and_the_owners_permission_and_the_owners_phrase_each_on_its_own() {
    use super::host::vouches_for;
    let says = |t: &str| vec![t.to_string()];
    let allowed = |initiative: &str| {
        let s = RingSettings { urgent_phrases: vec!["gas leak".into()], ..Default::default() };
        RingSettings { initiative: if initiative == "urgent" { super::settings::Initiative::OnRequestOrUrgent } else { s.initiative }, ..s }
    };
    assert!(vouches_for(Reason::Urgent, &allowed("urgent"), &says("a gas leak in the shop")));
    assert!(!vouches_for(Reason::Urgent, &allowed("request"), &says("a gas leak in the shop")), "the owner did not allow it");
    assert!(!vouches_for(Reason::Urgent, &allowed("urgent"), &says("a mow on Tuesday")), "their phrase was not heard");
    assert!(!vouches_for(Reason::Urgent, &RingSettings { initiative: super::settings::Initiative::OnRequestOrUrgent, ..Default::default() }, &says("a gas leak")), "they named no phrase");
    assert!(!vouches_for(Reason::CallerAsked, &allowed("urgent"), &says("a gas leak in the shop")));
    assert!(!vouches_for(Reason::PolicyRule, &allowed("urgent"), &says("a gas leak in the shop")));
}

#[test]
fn an_urgent_reason_nobody_confirmed_is_not_vouched_for() {
    // The owner did not allow the receptionist to ask on its own for urgent things: no ring, and nothing vouched.
    let r = urgent_call("There is a gas leak in the shop", "on_request");
    let plan = r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default());
    assert_eq!((plan.plan.decision, plan.plan.reason, plan.reason_allowed), (Decision::MessageOnly, PlanReason::InitiativeOff, false));
    assert_eq!(super::host::plan_result(&plan)["reasonAllowed"], json!(false));
    // Allowed, but the caller did not say one of the phrases: not urgent, and nothing vouched.
    let r = urgent_call("Can I book a mow please", "on_request_or_urgent");
    let plan = r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default());
    assert_eq!((plan.plan.decision, plan.plan.reason, plan.reason_allowed), (Decision::MessageOnly, PlanReason::NotUrgent, false));
    // The model's own word that it is urgent is not the caller's: a phrase in what the model says is nothing.
    let r = urgent_call("It is urgent, please hurry", "on_request_or_urgent");
    assert!(!r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default()).reason_allowed);
}

#[test]
fn no_other_reason_is_vouched_for_however_urgent_the_caller_sounds() {
    let said = "There is a gas leak, can I speak to the owner please";
    // A caller who asked for a person, with the urgent phrase in it: the reason is caller_asked, and is not vouched.
    let r = urgent_call(said, "on_request_or_urgent");
    let plan = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(plan.rings() && !plan.reason_allowed, "{:?}", plan.plan);
    // The owner's own rule is not something this desktop can see either.
    let r = urgent_call(said, "on_request_or_urgent");
    let plan = r.ring.plan_for_plugin(CALL, Reason::PolicyRule, CallInfo::default());
    assert!(!plan.reason_allowed, "{:?}", plan.plan);
    // A refused plan vouches for nothing.
    let r = urgent_call(said, "on_request_or_urgent");
    r.ring.change_settings(&json!({ "enabled": false })).unwrap();
    assert!(!r.ring.plan_for_plugin(CALL, Reason::Urgent, CallInfo::default()).reason_allowed);
}

/// Another call the desktop heard, from `from`.
fn another_call(r: &Rig, call: &str, from: &str) {
    r.calls.info.lock().unwrap().push((call.into(), caller(from, "Sam", ASKED)));
}

#[test]
fn callers_who_hide_their_number_share_one_small_bucket_of_the_hour_a_call_each_is_not_their_own() {
    let r = rig(Presence::Active);
    // Three calls, three ways of not giving a number: the owner's own limit is three a caller an hour, theirs is two.
    for (call, from) in [("call_a", ""), ("call_b", "Private"), ("call_c", "anonymous")] {
        another_call(&r, call, from);
    }
    assert!(r.ring.authorise("call_a", Reason::CallerAsked).rings());
    assert!(r.ring.authorise("call_b", Reason::CallerAsked).rings());
    let third = r.ring.authorise("call_c", Reason::CallerAsked);
    assert_eq!((third.plan.decision, third.plan.reason), (Decision::Refused, PlanReason::LimitCaller), "{:?}", third.plan);
    // The plugin asking is refused the same way.
    assert_eq!(r.ring.plan_for_plugin("call_c", Reason::CallerAsked, CallInfo::default()).plan.reason, PlanReason::LimitCaller);
    // Callers with a number are not counted against the hidden ones (and hidden calls are not spent from the global hour beyond two).
    another_call(&r, "call_k", "+61491570156");
    assert!(r.ring.authorise("call_k", Reason::CallerAsked).rings(), "a caller who gives a number still gets through");
    let counters = r.ring.attempts.lock().unwrap().counters("call_k", "491570156", r.ring.clock().unix());
    assert_eq!(counters.global_attempts_last_hour, 3);
    // An owner who allows fewer than two a caller keeps that for everyone, hidden included.
    let strict = rig(Presence::Active);
    strict.ring.change_settings(&json!({ "limits": { "perCallerHour": 1 } })).unwrap();
    another_call(&strict, "call_a", "");
    another_call(&strict, "call_b", "Private");
    assert!(strict.ring.authorise("call_a", Reason::CallerAsked).rings());
    assert_eq!(strict.ring.authorise("call_b", Reason::CallerAsked).plan.reason, PlanReason::LimitCaller);
}

#[test]
fn a_number_on_the_vip_list_passes_quiet_hours_but_never_the_limits_a_caller_id_can_be_faked() {
    let late = chrono::DateTime::parse_from_rfc3339("2026-09-30T23:00:00+10:00").unwrap();
    struct At(chrono::DateTime<chrono::FixedOffset>);
    impl Clock for At {
        fn local(&self) -> chrono::DateTime<chrono::FixedOffset> {
            self.0
        }
    }
    let r = rig(Presence::Active);
    r.ring.set_clock(Arc::new(At(late)));
    r.ring.change_settings(&json!({ "vipNumbers": ["0491 570 006"], "quietHours": { "enabled": true } })).unwrap();
    // Quiet hours: a caller who is not a VIP is not rung, and a VIP is.
    another_call(&r, "call_other", "+61491570156");
    assert_eq!(r.ring.authorise("call_other", Reason::CallerAsked).plan.reason, PlanReason::QuietHours);
    for n in 0..3 {
        let call = format!("call_vip{n}");
        another_call(&r, &call, "+61491570006");
        let plan = r.ring.authorise(&call, Reason::CallerAsked);
        assert!(plan.rings(), "try {n}: {:?}", plan.plan);
    }
    // The fourth try in the hour from the same number is refused, VIP or not: the number may be anyone's.
    another_call(&r, "call_vip3", "+61491570006");
    let fourth = r.ring.authorise("call_vip3", Reason::CallerAsked);
    assert_eq!((fourth.plan.decision, fourth.plan.reason), (Decision::Refused, PlanReason::LimitCaller), "{:?}", fourth.plan);
}

#[test]
fn a_try_the_plugin_refused_before_any_ring_is_given_back_with_its_gap_and_one_that_rang_is_not() {
    let r = rig(Presence::Active);
    let counters = |call: &str, key: &str| r.ring.attempts.lock().unwrap().counters(call, key, r.ring.clock().unix());
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings());
    assert_eq!(counters(CALL, "491570006").attempts_this_call, 1);
    // Refused at once: the gap between tries would refuse the next.
    assert_eq!(r.ring.authorise(CALL, Reason::CallerAsked).plan.reason, PlanReason::LimitGap);
    assert!(r.ring.request_refused(CALL), "a try was given back");
    let c = counters(CALL, "491570006");
    assert_eq!((c.attempts_this_call, c.seconds_since_last_attempt, c.caller_attempts_last_hour, c.global_attempts_last_hour), (0, None, 0, 0));
    assert!(!r.ring.request_refused(CALL), "and only once");
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings(), "the same call may ask again at once");
    // The plan that was given back is gone: a plugin cannot open a ring for it.
    let r = rig(Presence::Active);
    let plan = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(plan.rings() && r.ring.request_refused(CALL));
    assert_eq!(r.ring.opened(&opened(&plan.plan_id, "assist_1", CALL, 20, &r.ring)).unwrap_err().code, "unknown_plan");
    // A ring that opened is a try that was used: it is not given back, and another call's tries are its own.
    let r = rig(Presence::Active);
    another_call(&r, "call_2", "+61491570156");
    assert!(r.ring.authorise("call_2", Reason::CallerAsked).rings());
    let plan = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    r.ring.opened(&opened(&plan.plan_id, "assist_1", CALL, 20, &r.ring)).unwrap();
    assert!(!r.ring.request_refused(CALL), "it rang");
    assert_eq!(r.ring.attempts.lock().unwrap().counters(CALL, "491570006", r.ring.clock().unix()).attempts_this_call, 1);
    assert!(r.ring.request_refused("call_2"), "another call's request that never opened is given back");
    assert_eq!(r.ring.attempts.lock().unwrap().counters(CALL, "491570006", r.ring.clock().unix()).attempts_this_call, 1);
}

#[test]
fn a_ring_uses_up_the_ask_it_was_made_on_and_the_next_request_needs_an_ask_of_its_own() {
    struct At(chrono::DateTime<chrono::FixedOffset>);
    impl Clock for At {
        fn local(&self) -> chrono::DateTime<chrono::FixedOffset> {
            self.0
        }
    }
    let r = rig(Presence::Active);
    r.calls.consuming.store(true, std::sync::atomic::Ordering::SeqCst);
    let start = chrono::DateTime::parse_from_rfc3339("2026-09-30T11:00:00+10:00").unwrap();
    r.ring.set_clock(Arc::new(At(start)));
    // A plan allowed is not yet an ask acted on: nothing is used up until a ring opens for it.
    let plan = planned(&r.ring, CALL);
    assert!(r.calls.used.lock().unwrap().is_empty());
    let ring = r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    assert!(ring.said.iter().any(|s| s == ASKED), "the dialog shows what the caller said, which is read before it is used up: {:?}", ring.said);
    assert_eq!(r.calls.used.lock().unwrap().as_slice(), [CALL.to_string()], "the ring said which words it was made on");
    r.ring.resolve("assist_1", Outcome::Declined, "phone");
    // Two minutes on, well past the gap between tries: the caller, rung for and declined, says only that they will leave a message.
    r.ring.set_clock(Arc::new(At(start + chrono::Duration::minutes(2))));
    r.calls.info.lock().unwrap()[0].1.turns.push("No, just take a message please.".into());
    let second = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!((second.plan.decision, second.plan.reason), (Decision::Refused, PlanReason::CallerDidNotAsk), "the ask that was acted on is not an ask again: {:?}", second.plan);
    assert_eq!(r.calls.used.lock().unwrap().len(), 1, "a refusal acts on nothing");
    // Asking again is asking: judged on its own words, and rung. The phone refuses it before it rang, so nothing was acted on and the ask stands.
    r.calls.info.lock().unwrap()[0].1.turns.push("Actually, can I speak to the owner please".into());
    let third = r.ring.authorise(CALL, Reason::CallerAsked);
    assert!(third.rings(), "{:?}", third.plan);
    assert!(r.ring.request_refused(CALL), "a try was given back");
    assert_eq!(r.calls.used.lock().unwrap().len(), 1, "and the ask with it");
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings(), "the same ask, judged again");
}

#[test]
fn a_caller_who_asks_for_the_owner_by_name_rings_only_when_the_desktop_knows_that_name() {
    let r = rig(Presence::Active);
    r.calls.info.lock().unwrap().clear();
    r.calls.info.lock().unwrap().push((CALL.into(), caller("+61491570006", "Alex", "Can I speak to Dave please")));
    // Nothing tells the desktop who the owner is: it is not a request for the owner.
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!((plan.plan.decision, plan.plan.reason), (Decision::Refused, PlanReason::CallerDidNotAsk));
    // The business is named for them (Dave's Lawn Care): it is.
    r.ring.set_names(Arc::new(|| super::phrases::owner_names("Dave's Lawn Care")));
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings());
    // Someone else's name is still not.
    let other = rig(Presence::Active);
    other.ring.set_names(Arc::new(|| super::phrases::owner_names("Dave's Lawn Care")));
    other.calls.info.lock().unwrap().clear();
    other.calls.info.lock().unwrap().push((CALL.into(), caller("+61491570006", "Alex", "Can I speak to Sam please")));
    assert_eq!(other.ring.authorise(CALL, Reason::CallerAsked).plan.reason, PlanReason::CallerDidNotAsk);
}

#[test]
fn a_second_try_on_one_call_straight_after_the_first_is_refused() {
    let r = rig(Presence::Active);
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    r.ring.resolve("assist_1", Outcome::Declined, "phone");
    // A try is counted when a plan is allowed (not when a ring opens), so asking again at once meets the gap between tries.
    let again = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(!again.rings() && !again.plan_id.is_empty(), "{:?}", again.plan);
    assert_eq!(again.plan.reason, PlanReason::LimitGap);
    assert!(r.ring.active().is_empty());
}

#[test]
fn what_the_phone_says_of_a_ring_ends_it() {
    let r = rig(Presence::Active);
    for (event, name) in [("transferred", "accepted"), ("declined", "declined"), ("unavailable", "unavailable"), ("expired", "expired"), ("cancelled", "cancelled")] {
        let r = rig(Presence::Active);
        open(&r);
        let data = json!({ "requestId": "assist_1", "callId": CALL, "outcome": event });
        super::apply_plugin_event(&r.ring, "aokie.call.assistance.resolved", &data, "");
        assert!(r.ring.active().is_empty(), "{event}");
        assert_eq!(ended_as(&r.ring), vec![(name, "phone")], "{event}");
    }
    // An outcome that is not one of these, and one for a ring that is not here, change nothing.
    open(&r);
    super::apply_plugin_event(&r.ring, "aokie.call.assistance.resolved", &json!({ "requestId": "assist_1", "outcome": "ringing" }), "");
    super::apply_plugin_event(&r.ring, "aokie.call.assistance.resolved", &json!({ "requestId": "assist_9", "outcome": "declined" }), "");
    assert_eq!(r.ring.active().len(), 1);
    // The call ending, named by the event or by its correlation, ends the ring and is told to the hub.
    super::apply_plugin_event(&r.ring, "aokie.call.ended", &json!({}), CALL);
    assert!(r.ring.active().is_empty());
    assert_eq!(r.calls.ended.lock().unwrap().as_slice(), [CALL.to_string()]);
    assert_eq!(ended_as(&r.ring), vec![("cancelled", "call")]);
    // Events that are not about rings are ignored.
    super::apply_plugin_event(&r.ring, "aokie.call.started", &json!({ "callId": CALL }), CALL);
    assert_eq!(r.calls.ended.lock().unwrap().len(), 1);
}

async fn serve(ring: Arc<Ring>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, super::routes::router(ring)).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn the_dialog_reads_the_rings_and_answers_them_over_http() {
    let r = rig(Presence::Active);
    let base = serve(r.ring.clone()).await;
    let client = reqwest::Client::new();
    let read = || async { client.get(format!("{base}/api/ring/active")).send().await.unwrap().json::<serde_json::Value>().await.unwrap() };
    assert_eq!(read().await, json!({ "rings": [], "notices": [] }));

    let ring = open(&r);
    let rings = read().await;
    let shown = &rings["rings"][0];
    assert_eq!((shown["id"].as_str(), shown["callerName"].as_str(), shown["callerNumber"].as_str(), shown["stopping"].as_bool()), (Some(ring.id.as_str()), Some("Alex"), Some("+61491570006"), Some(false)), "{shown}");
    assert!(shown.get("canAccept").is_none(), "there is nothing to accept here");
    assert!(shown["expiresAt"].as_u64().unwrap() > shown["now"].as_u64().unwrap(), "the countdown runs on this desktop's clock: {shown}");
    assert_eq!(shown["said"][1], ASKED);

    // Answers this route will not take: an action that is not one, more than was asked for, and a ring that is not there.
    let post = |id: &str, body: serde_json::Value| {
        let url = format!("{base}/api/ring/active/{id}/respond");
        let client = client.clone();
        async move { client.post(url).json(&body).send().await.unwrap() }
    };
    let bad = post("assist_1", json!({ "action": "hang up" })).await;
    assert_eq!(bad.status(), 400);
    assert_eq!(bad.json::<serde_json::Value>().await.unwrap()["error"]["code"], "bad_action");
    assert!(post("assist_1", json!({ "action": "decline", "extra": 1 })).await.status().is_client_error());
    let missing = post("assist_9", json!({ "action": "decline" })).await;
    assert_eq!(missing.status(), 404);
    assert_eq!(missing.json::<serde_json::Value>().await.unwrap()["error"]["code"], "no_ring");
    assert_eq!(read().await["rings"].as_array().unwrap().len(), 1, "none of those touched the ring");
    assert!(r.calls.cancels.lock().unwrap().is_empty());
    // Accepting is not something this computer does.
    let accept = post("assist_1", json!({ "action": "accept" })).await;
    assert_eq!(accept.status(), 400);
    assert_eq!(accept.json::<serde_json::Value>().await.unwrap()["error"]["code"], "bad_action");

    // A message instead: the phone is asked to withdraw it, the dialog shows it stopping, and the phone's answer ends it.
    let message = post("assist_1", json!({ "action": "message" })).await;
    assert_eq!(message.status(), 200);
    let message: serde_json::Value = message.json().await.unwrap();
    assert_eq!(message["ok"], true);
    let stopping = read().await;
    assert_eq!((stopping["rings"][0]["stopping"].clone(), stopping["rings"][0]["note"].clone()), (json!(true), message["note"].clone()));
    assert_eq!(r.calls.cancels.lock().unwrap().as_slice(), [(CALL.to_string(), "assist_1".to_string(), CancelReason::MessageInstead)]);
    r.ring.outcome_seen("assist_1", Outcome::Cancelled, "phone");
    assert_eq!(read().await, json!({ "rings": [], "notices": [] }));
    assert_eq!(post("assist_1", json!({ "action": "decline" })).await.status(), 404);
    assert_eq!(r.calls.cancels.lock().unwrap().len(), 1, "asked once");
}

#[tokio::test]
async fn a_notice_that_nobody_could_be_rung_is_read_and_dismissed_over_http() {
    let r = rig(Presence::Active);
    phones_never_ring(&r);
    let base = serve(r.ring.clone()).await;
    let client = reqwest::Client::new();
    r.ring.authorise(CALL, Reason::CallerAsked);
    let read: serde_json::Value = client.get(format!("{base}/api/ring/active")).send().await.unwrap().json().await.unwrap();
    assert_eq!(read["rings"], json!([]));
    let notice = &read["notices"][0];
    assert_eq!((notice["callerName"].as_str(), notice["callerNumber"].as_str(), notice["callId"].as_str()), (Some("Alex"), Some("+61491570006"), Some(CALL)), "{notice}");
    assert!(notice["text"].as_str().unwrap().contains("Your phone is set to not ring") && notice["cause"] == "phonesOff", "{notice}");
    let id = notice["id"].as_str().unwrap().to_string();
    let gone = client.post(format!("{base}/api/ring/notices/notice_nope/dismiss")).send().await.unwrap();
    assert_eq!(gone.status(), 404);
    let done = client.post(format!("{base}/api/ring/notices/{id}/dismiss")).send().await.unwrap();
    assert_eq!(done.status(), 200);
    let read: serde_json::Value = client.get(format!("{base}/api/ring/active")).send().await.unwrap().json().await.unwrap();
    assert_eq!(read["notices"], json!([]));
}
