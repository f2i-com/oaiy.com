//! The rings going now: what opens one, what ends it (first word wins), what the owner can do in the
//! dialog, and that a ring nobody reports the end of does not ring for ever.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use super::contract::{OpenedParams, RespondAction};
use super::plan::{Availability, Device, DeviceKind, Role};
use super::session::Action;
use super::*;
use crate::voice::transfer::Outcome;

const CALL: &str = "call_1";
const ASKED: &str = "Hello, can I speak to the owner please";

/// The desktop's record of one call, and what the ring told it.
#[derive(Default)]
struct Calls {
    info: Mutex<Vec<(String, CallInfo)>>,
    told: Mutex<Vec<(String, String, Outcome)>>,
    ended: Mutex<Vec<String>>,
}

impl CallSource for Calls {
    fn facts(&self, call: &str) -> Option<CallInfo> {
        self.info.lock().unwrap().iter().find(|(c, _)| c == call).map(|(_, i)| i.clone())
    }
    fn call_ended_by_phone(&self, call: &str) {
        self.ended.lock().unwrap().push(call.to_string());
    }
    fn local_outcome(&self, call: &str, request: &str, outcome: Outcome) {
        self.told.lock().unwrap().push((call.to_string(), request.to_string(), outcome));
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

/// A phone plugin that takes what it is asked, or cannot.
struct Plugin {
    asked: Mutex<Vec<(String, RespondAction)>>,
    answer: Result<(), String>,
}

impl Plugin {
    fn taking() -> Arc<Plugin> {
        Arc::new(Plugin { asked: Mutex::default(), answer: Ok(()) })
    }
}

impl TransferPlugin for Plugin {
    fn respond(&self, request: &str, action: RespondAction) -> Result<(), String> {
        self.asked.lock().unwrap().push((request.to_string(), action));
        self.answer.clone()
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
    authorised.plan_id.unwrap()
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
    assert_eq!(r.ring.opened(&opened(&gate.plan_id.unwrap(), "assist_1", CALL, 20, &r.ring)).unwrap_err().code, "unknown_plan");
    assert!(r.ring.active().is_empty() && r.notified.rang.lock().unwrap().is_empty());
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
    assert!(!ring.can_accept, "there is no plugin to ask");
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
    assert!(!plan.rings() && plan.plan_id.is_none());
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
    assert_eq!(r.ring.respond("assist_1", Action::Decline).unwrap_err().code, "no_ring");
    assert!(r.calls.told.lock().unwrap().is_empty());
}

#[test]
fn accept_asks_the_phone_and_the_ring_goes_on_until_the_phone_says() {
    let r = rig(Presence::Active);
    let plugin = Plugin::taking();
    r.ring.set_plugin(plugin.clone());
    assert!(open(&r).can_accept);
    let said = r.ring.respond("assist_1", Action::Accept).unwrap();
    assert!(said.ok && said.note.contains("Asked the Companion"), "{}", said.note);
    assert_eq!(plugin.asked.lock().unwrap().as_slice(), [("assist_1".to_string(), RespondAction::Accept)]);
    assert_eq!(r.ring.active().len(), 1, "the call is not taken until the phone says so");
    assert_eq!(r.ring.active()[0].note, said.note, "and the dialog keeps what was asked");
    assert!(r.calls.told.lock().unwrap().is_empty(), "the receptionist is told nothing");
    // The phone says it took the call.
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    assert!(r.ring.active().is_empty());
    assert_eq!(ended_as(&r.ring), vec![("accepted", "phone")]);
}

#[test]
fn accept_says_plainly_when_the_phone_cannot_be_asked() {
    // No plugin to ask.
    let r = rig(Presence::Active);
    open(&r);
    let said = r.ring.respond("assist_1", Action::Accept).unwrap();
    assert!(!said.ok && said.note.contains("answer on your Companion"), "{}", said.note);
    assert_eq!(r.ring.active().len(), 1, "it still rings");

    // A plugin that refuses (it has no such command).
    let r = rig(Presence::Active);
    r.ring.set_plugin(Arc::new(Plugin { asked: Mutex::default(), answer: Err("no such command".into()) }));
    open(&r);
    let said = r.ring.respond("assist_1", Action::Accept).unwrap();
    assert!(!said.ok && said.note.contains("no such command") && said.note.contains("Answer on your Companion"), "{}", said.note);
    assert_eq!(r.ring.active().len(), 1, "nothing was claimed, and it still rings");
    assert!(r.calls.told.lock().unwrap().is_empty());
}

#[test]
fn declining_or_a_message_instead_ends_the_ring_here_at_once_and_tells_the_call() {
    for action in [Action::Decline, Action::Message] {
        let r = rig(Presence::Active);
        let plugin = Plugin::taking();
        r.ring.set_plugin(plugin.clone());
        open(&r);
        let said = r.ring.respond("assist_1", action).unwrap();
        assert!(said.ok && said.note.contains("offer to take a message") && !said.note.contains("may still ring"), "{}", said.note);
        assert!(r.ring.active().is_empty(), "{action:?}: gone from the dialog at once");
        assert_eq!(r.calls.told.lock().unwrap().as_slice(), [(CALL.to_string(), "assist_1".to_string(), Outcome::Declined)], "{action:?}: the caller is sent to the message offer");
        assert_eq!(plugin.asked.lock().unwrap().as_slice(), [("assist_1".to_string(), RespondAction::Decline)], "{action:?}: the request is withdrawn");
        assert_eq!(r.notified.ended.lock().unwrap().len(), 1);
        assert_eq!(ended_as(&r.ring), vec![("declined", "desktop")]);
    }

    // With no plugin to withdraw the request the owner is told their devices may still ring, and the caller is still sent on.
    let r = rig(Presence::Active);
    open(&r);
    assert!(r.ring.respond("assist_1", Action::Decline).unwrap().note.contains("may still ring"));
    assert_eq!(r.calls.told.lock().unwrap().len(), 1);
}

#[test]
fn a_decline_that_loses_to_the_phone_reaches_nobody() {
    let r = rig(Presence::Active);
    open(&r);
    // The phone says accepted between the owner's click and the desktop hearing it.
    r.ring.outcome_seen("assist_1", Outcome::Accepted, "phone");
    let late = r.ring.respond("assist_1", Action::Decline).unwrap_err();
    assert_eq!((late.status, late.code), (404, "no_ring"));
    assert!(r.calls.told.lock().unwrap().is_empty(), "the receptionist was not told the owner declined a call they took");
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

/// The owner at their computer, an approved phone, and nothing ticked as this computer's Companion: the default setup.
fn nobody_to_offer_a_call_to(r: &Rig) {
    r.ring.set_devices(Arc::new(crate::ring::testing::Devices(vec![crate::ring::testing::android("ph1")])));
}

#[test]
fn a_toast_alone_is_not_a_ring_so_nobody_is_planned_for_and_no_try_is_counted() {
    let r = rig(Presence::Active);
    nobody_to_offer_a_call_to(&r);
    // The reference plans a ring for the owner at their computer with only the toast, and this desktop does not: a plugin that
    // offers a transfer only to the devices a plan names would open nothing for it.
    let plan = r.ring.authorise(CALL, Reason::CallerAsked);
    assert_eq!((plan.plan.decision, plan.plan.reason), (Decision::MessageOnly, PlanReason::NoDevice), "{:?}", plan.plan);
    assert!(!plan.rings() && plan.plan_id.is_none());
    assert!(!plan.plan.desktop_toast && plan.plan.targets().is_empty());
    // The plugin asking is answered the same way, and nothing was counted for either.
    let asked = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert_eq!((asked.plan.decision, asked.plan.reason), (Decision::MessageOnly, PlanReason::NoDevice));
    let counters = r.ring.attempts.lock().unwrap().counters(CALL, "491570006", r.ring.clock().unix());
    assert_eq!((counters.attempts_this_call, counters.global_attempts_last_hour), (0, 0), "no try was spent on a ring that cannot happen");
    assert!(r.ring.active().is_empty() && r.notified.rang.lock().unwrap().is_empty());
    // The same caller can still be put through once a device is set up: nothing was used up.
    r.ring.set_devices(crate::ring::testing::at_the_pc());
    assert!(r.ring.authorise(CALL, Reason::CallerAsked).rings());
}

#[test]
fn the_owner_away_still_rings_their_phone_and_only_the_toast_needs_the_computers_companion() {
    let r = rig(Presence::Idle);
    nobody_to_offer_a_call_to(&r);
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
    nobody_to_offer_a_call_to(&r);
    r.ring.authorise(CALL, Reason::CallerAsked);
    r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    let notices = r.ring.notices();
    assert_eq!(notices.len(), 1, "once a call, whoever asked");
    let n = &notices[0];
    assert_eq!((n.call_id.as_str(), n.caller_name.as_str(), n.caller_number.as_str(), n.text.as_str()), (CALL, "Alex", "+61491570006", crate::ring::session::NO_DEVICE_TEXT));
    assert!(n.text.contains("No device is set up to take a transfer") && n.text.contains("offered a message"));
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
    nobody_to_offer_a_call_to(&r);
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

#[test]
fn a_second_try_on_one_call_straight_after_the_first_is_refused() {
    let r = rig(Presence::Active);
    let plan = planned(&r.ring, CALL);
    r.ring.opened(&opened(&plan, "assist_1", CALL, 25, &r.ring)).unwrap();
    r.ring.resolve("assist_1", Outcome::Declined, "phone");
    // A try is counted when a plan is allowed (not when a ring opens), so asking again at once meets the gap between tries.
    let again = r.ring.plan_for_plugin(CALL, Reason::CallerAsked, CallInfo::default());
    assert!(!again.rings() && again.plan_id.is_none(), "{:?}", again.plan);
    assert_eq!(again.plan.reason, PlanReason::LimitGap);
    assert!(r.ring.active().is_empty());
}

#[test]
fn what_the_phone_says_of_a_ring_ends_it() {
    let r = rig(Presence::Active);
    for (event, name) in [("transferred", "accepted"), ("declined", "declined"), ("unavailable", "unavailable"), ("expired", "expired")] {
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
    assert_eq!((shown["id"].as_str(), shown["callerName"].as_str(), shown["callerNumber"].as_str(), shown["canAccept"].as_bool()), (Some(ring.id.as_str()), Some("Alex"), Some("+61491570006"), Some(false)), "{shown}");
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
    assert!(r.calls.told.lock().unwrap().is_empty());

    // Accept with nothing to ask is answered, honestly, and the ring goes on.
    let accept = post("assist_1", json!({ "action": "accept" })).await;
    assert_eq!(accept.status(), 200);
    let accept: serde_json::Value = accept.json().await.unwrap();
    assert_eq!(accept["ok"], false);
    assert_eq!(read().await["rings"][0]["note"], accept["note"]);

    // A message instead: the ring ends, the caller is sent on, and asking again finds nothing.
    let message = post("assist_1", json!({ "action": "message" })).await;
    assert_eq!(message.status(), 200);
    assert_eq!(message.json::<serde_json::Value>().await.unwrap()["ok"], true);
    assert_eq!(read().await, json!({ "rings": [], "notices": [] }));
    assert_eq!(r.calls.told.lock().unwrap().as_slice(), [(CALL.to_string(), "assist_1".to_string(), Outcome::Declined)]);
    assert_eq!(post("assist_1", json!({ "action": "decline" })).await.status(), 404);
    assert_eq!(r.calls.told.lock().unwrap().len(), 1, "told once");
}

#[tokio::test]
async fn a_notice_that_nobody_could_be_rung_is_read_and_dismissed_over_http() {
    let r = rig(Presence::Active);
    nobody_to_offer_a_call_to(&r);
    let base = serve(r.ring.clone()).await;
    let client = reqwest::Client::new();
    r.ring.authorise(CALL, Reason::CallerAsked);
    let read: serde_json::Value = client.get(format!("{base}/api/ring/active")).send().await.unwrap().json().await.unwrap();
    assert_eq!(read["rings"], json!([]));
    let notice = &read["notices"][0];
    assert_eq!((notice["callerName"].as_str(), notice["callerNumber"].as_str(), notice["callId"].as_str()), (Some("Alex"), Some("+61491570006"), Some(CALL)), "{notice}");
    assert!(notice["text"].as_str().unwrap().contains("No device is set up"));
    let id = notice["id"].as_str().unwrap().to_string();
    let gone = client.post(format!("{base}/api/ring/notices/notice_nope/dismiss")).send().await.unwrap();
    assert_eq!(gone.status(), 404);
    let done = client.post(format!("{base}/api/ring/notices/{id}/dismiss")).send().await.unwrap();
    assert_eq!(done.status(), 200);
    let read: serde_json::Value = client.get(format!("{base}/api/ring/active")).send().await.unwrap().json().await.unwrap();
    assert_eq!(read["notices"], json!([]));
}
