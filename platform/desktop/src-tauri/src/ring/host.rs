//! The ring as this desktop runs it: the owner's settings, what is known of who could take a
//! call and whether the owner is at the computer, and the one place a request to reach the
//! owner is allowed or refused ([`Ring::authorise`]).
//!
//! A request is judged on what this desktop itself heard of the call (who rang and what they
//! said, from the voice hub), never on what the receptionist's model or a plugin says about it:
//! the caller's own words must have asked for a person ([`super::phrases`]), the owner must have
//! turned transfers on, and the limits must allow another try. A try that is allowed is counted
//! at once and remembered for a short while under a plan id, so the plugin's own question
//! (`oaiy.ring.plan`) about the same request is answered with the same plan and is not counted
//! twice, and a second question about it is a second try.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::limits::Attempts;
use super::plan::{plan, CallFacts, Device, Inputs, Now, Presence, Reason, RingPlan};
use super::settings::{PhoneRing, RingSettings, SettingsError, SettingsStore};
use super::{phrases, Decision, PlanReason};

/// How long a plan that was allowed waits for the plugin to ask for it.
pub const PLAN_TTL: Duration = Duration::from_secs(30);

/// Whether the owner is at the computer (the OS on a desktop, nothing on the headless server).
pub trait PresenceSource: Send + Sync {
    fn presence(&self) -> Presence;
}

/// Who could take a call: the companions the owner approved.
pub trait DeviceSource: Send + Sync {
    fn devices(&self, settings: &RingSettings) -> Vec<Device>;

    /// What to call a device to the owner (its name), by its id.
    fn label(&self, id: &str) -> String {
        id.chars().take(8).collect()
    }
}

/// The clock quiet hours and limits are read by.
pub trait Clock: Send + Sync {
    /// The owner's own clock: its offset is the one their wall clock has now, daylight saving included.
    fn local(&self) -> chrono::DateTime<chrono::FixedOffset>;
    fn unix(&self) -> u64 {
        u64::try_from(self.local().timestamp()).unwrap_or(0)
    }
}

/// What this desktop knows of one call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallInfo {
    /// The number the phone said the call came from ("" when hidden).
    pub from: String,
    pub name: String,
    /// What the caller said, oldest first (not what an earlier request has used up).
    pub turns: Vec<String>,
    /// How many turns the caller had said on the call in all when this was read: the request judged on `turns` uses up that many when a ring
    /// opens on it, and no more (what the caller said while it was being planned and sent is the next request's).
    pub total: u64,
}

/// Where the ring learns of a call: the voice hub.
pub trait CallSource: Send + Sync {
    /// The call, when this desktop knows it (live, in handoff, or ended within ten minutes).
    fn facts(&self, call: &str) -> Option<CallInfo>;
    /// The phone says the call ended (the plugin's own event): the hub ends one that was handed to the owner.
    fn call_ended_by_phone(&self, call: &str);
    /// The phone is asked to withdraw `request` on `call` (the owner declined in the dialog, or it ran out): the call sends
    /// `transfer_cancel` on its stream, waits for the phone's answer and acts on it. What came of the asking is the answer: the frame
    /// is on the wire, or is waiting for the phone to name the request to the call, or there was nothing to withdraw, or no session.
    fn cancel_transfer(&self, call: &str, request: &str, reason: crate::voice::transfer::CancelReason) -> tokio::sync::oneshot::Receiver<Withdrawal>;
    /// A ring opened for `call` on the first `up_to` turns the caller had said (as `facts` gave them: [`CallInfo::total`]): those words are used
    /// up. An ask counts for ONE request, so the next request is judged on what the caller says after it, never on an ask that has been acted on
    /// (a caller who was rung, and who then says "no, just take a message", or who is handed back by the owner and says "thanks, that's all
    /// sorted", has not asked again). What they said after it was read, while the request was planned and sent, is not spent by it. A request
    /// that was refused, or that the phone refused before it rang, has acted on nothing: its ask stands.
    fn consume_turns(&self, _call: &str, _up_to: u64) {}
    /// Which beginning of the call this is: it changes each time the call begins (its session made anew, or handed back by the owner), so a
    /// request planned for one beginning is not opened on another.
    fn generation(&self, _call: &str) -> u64 {
        0
    }
    /// The call has ended (its record is kept a while after: see [`CallSource::facts`]). A request on a call that is over rings nobody: the
    /// caller who hung up while the request was being planned is not one the owner is rung for.
    fn is_over(&self, _call: &str) -> bool {
        false
    }
}

/// What came of asking the phone to withdraw a request (see [`CallSource::cancel_transfer`]): what the owner is told is never more than
/// this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Withdrawal {
    /// The frame is on the call's stream, and its answer is being waited for.
    Sent,
    /// The phone has not yet named the request to the call (its answer to the tool call waits for the line the model spoke to drain):
    /// the frame goes the moment it does, and its answer is waited for from then.
    Queued,
    /// The call has no such request to withdraw (it ended here, or an owner device has it, or this call never asked): nothing was sent.
    Nothing,
    /// The call has no live session to carry it.
    NoSession,
    /// The call did not say in time.
    Unknown,
}

struct Absent;

impl PresenceSource for Absent {
    fn presence(&self) -> Presence {
        Presence::Off
    }
}

impl DeviceSource for Absent {
    fn devices(&self, _: &RingSettings) -> Vec<Device> {
        Vec::new()
    }
}

struct System;

impl Clock for System {
    fn local(&self) -> chrono::DateTime<chrono::FixedOffset> {
        chrono::Local::now().fixed_offset()
    }
}

/// A plan that was allowed, waiting for the plugin to ask for it.
struct Grant {
    plan_id: String,
    call_id: String,
    plan: RingPlan,
    /// This desktop vouched for the reason (see [`Authorised::reason_allowed`]).
    reason_allowed: bool,
    at: Instant,
    claimed: bool,
    /// How many of the caller's turns the request was judged on: what a ring opened on it uses up.
    judged: u64,
    /// Which beginning of the call it was judged for (see [`CallSource::generation`]).
    generation: u64,
}

/// A plan taken for a ring that opens on it.
pub(super) struct Taken {
    pub plan: RingPlan,
    /// See [`Grant::judged`].
    pub judged: u64,
}

/// What a request to reach the owner came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Authorised {
    /// The plan: a ring, or why not.
    pub plan: RingPlan,
    /// The plan's id, always: the plugin refuses a plan without a valid one and would lose the real reason. Only a plan that
    /// rings is kept under it (the plugin's question about the same request is answered with that plan, and `oaiy.ring.opened`
    /// is accepted for it); an id given with a refusal opens nothing.
    pub plan_id: String,
    /// This desktop itself vouches for the reason of the request, so the plugin need not see the caller ask for a person (the
    /// plan's `reasonAllowed`). Only for `urgent`: when the owner allows the receptionist to ask on its own for urgent things
    /// (`initiative`) and this desktop heard, in the caller's own words, one of the owner's urgent phrases. The model's
    /// word for it, and the plugin's, are never enough; for every other reason this is false.
    pub reason_allowed: bool,
    pub caller_number: String,
    pub caller_name: String,
}

impl Authorised {
    pub fn rings(&self) -> bool {
        self.plan.decision == Decision::Ring
    }
}

/// The ring: settings and attempts, and the sources it reads.
pub struct Ring {
    pub settings: SettingsStore,
    pub attempts: Mutex<Attempts>,
    presence: RwLock<Arc<dyn PresenceSource>>,
    devices: RwLock<Arc<dyn DeviceSource>>,
    calls: RwLock<Option<Arc<dyn CallSource>>>,
    clock: RwLock<Arc<dyn Clock>>,
    grants: Mutex<HashMap<String, Grant>>,
    on_features: RwLock<Option<Arc<dyn Fn(super::Features) + Send + Sync>>>,
    /// The names a caller may ask for as the owner (see [`phrases::owner_names`]): what the desktop can tell of them.
    names: RwLock<Arc<dyn Fn() -> Vec<String> + Send + Sync>>,
    /// The rings going now, and the last ones that ended (see `session.rs`).
    pub(super) sessions: Mutex<super::session::Sessions>,
    notifier: RwLock<Option<Arc<dyn super::session::RingNotifier>>>,
    /// How long past its time a ring waits to hear how it came out before it is over.
    pub(super) expiry_grace: RwLock<Duration>,
    /// Whether the phone plugin offers calls for transfer (see `preview.rs`).
    pub(super) offers: super::preview::OfferLog,
}

fn get<T: Clone>(lock: &RwLock<T>) -> T {
    lock.read().unwrap_or_else(|e| e.into_inner()).clone()
}

fn put<T>(lock: &RwLock<T>, value: T) {
    *lock.write().unwrap_or_else(|e| e.into_inner()) = value;
}

impl Ring {
    pub(super) fn with(settings: SettingsStore, attempts: Attempts) -> Arc<Ring> {
        Arc::new(Ring {
            settings,
            attempts: Mutex::new(attempts),
            presence: RwLock::new(Arc::new(Absent)),
            devices: RwLock::new(Arc::new(Absent)),
            calls: RwLock::new(None),
            clock: RwLock::new(Arc::new(System)),
            grants: Mutex::new(HashMap::new()),
            on_features: RwLock::new(None),
            names: RwLock::new(Arc::new(Vec::new)),
            sessions: Mutex::new(super::session::Sessions::default()),
            notifier: RwLock::new(None),
            expiry_grace: RwLock::new(super::session::EXPIRY_GRACE),
            offers: Mutex::default(),
        })
    }

    /// What raises the notification for a ring (the GUI's; a test's).
    pub fn set_notifier(&self, notifier: Arc<dyn super::session::RingNotifier>) {
        put(&self.notifier, Some(notifier));
    }

    /// The notifier in use: this ring's own, else the desktop's.
    pub(super) fn notifier(&self) -> Option<Arc<dyn super::session::RingNotifier>> {
        get(&self.notifier).or_else(super::session::global_notifier)
    }

    /// Ask the phone, through the call, to withdraw `request` (see [`CallSource::cancel_transfer`]): what came of it, when the call says.
    pub fn cancel_on_call(&self, call: &str, request: &str, reason: crate::voice::transfer::CancelReason) -> tokio::sync::oneshot::Receiver<Withdrawal> {
        match get(&self.calls) {
            Some(calls) => calls.cancel_transfer(call, request, reason),
            None => {
                let (reply, answer) = tokio::sync::oneshot::channel();
                let _ = reply.send(Withdrawal::NoSession);
                answer
            }
        }
    }

    /// How long a ring waits past its time to hear how it came out (a test shortens it).
    pub fn set_expiry_grace(&self, grace: Duration) {
        put(&self.expiry_grace, grace);
    }

    /// A device's name to the owner.
    pub(super) fn device_label(&self, id: &str) -> String {
        get(&self.devices).label(id)
    }

    /// The owner's clock in Unix milliseconds.
    pub(super) fn now_ms(&self) -> u64 {
        u64::try_from(self.clock().local().timestamp_millis()).unwrap_or(0)
    }

    /// A plan this desktop allowed and the plugin asked for, taken: what was decided for the call, moved out (not copied) and used up. One plan
    /// opens one ring, for the request that opened first; a second request on it, or the same one again, finds nothing (the same one again
    /// is answered before it gets here: see [`Ring::opened`]). Once a ring has opened for it the try is a ring's, and is never given back.
    pub(super) fn take_plan(&self, plan_id: &str, call: &str) -> Option<Taken> {
        let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
        grants.get(plan_id).filter(|g| g.claimed && g.call_id == call && g.at.elapsed() < PLAN_TTL * 4)?;
        grants.remove(plan_id).map(|g| Taken { plan: g.plan, judged: g.judged })
    }

    /// Whether the plan `plan_id` was judged for the call as it is now: not for an earlier beginning of it (its session made anew, or handed back
    /// by the owner, since). A plan that is not there is not current.
    pub(super) fn plan_is_current(&self, plan_id: &str, call: &str) -> bool {
        let generation = self.call_generation(call);
        self.grants.lock().unwrap_or_else(|e| e.into_inner()).get(plan_id).is_some_and(|g| g.call_id == call && g.generation == generation)
    }

    /// Which beginning of the call this is (see [`CallSource::generation`]).
    fn call_generation(&self, call: &str) -> u64 {
        get(&self.calls).map_or(0, |c| c.generation(call))
    }

    /// Whether the call is over (see [`CallSource::is_over`]).
    pub(super) fn call_is_over(&self, call: &str) -> bool {
        get(&self.calls).is_some_and(|c| c.is_over(call))
    }

    /// The plugin refused a request this desktop allowed (its own checks: consent, a changed call, a plan it could not use)
    /// before any ring opened for it: the try is given back, so a refusal that rang nobody does not start the gap between tries or
    /// use up the caller's hour. A request that opened a ring is never given back (its plan is used up when it opens). Whether a try was.
    pub fn request_refused(&self, call: &str) -> bool {
        let refunded = {
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            let newest = grants.iter().filter(|(_, g)| g.call_id == call).max_by_key(|(_, g)| g.at).map(|(id, _)| id.clone());
            newest.is_some_and(|id| grants.remove(&id).is_some())
        };
        refunded && self.attempts.lock().unwrap_or_else(|e| e.into_inner()).forget_last(call)
    }

    /// Where the owner's name comes from (the business named for them), asked each time a request is judged.
    pub fn set_names(&self, names: Arc<dyn Fn() -> Vec<String> + Send + Sync>) {
        put(&self.names, names);
    }

    pub fn set_presence(&self, source: Arc<dyn PresenceSource>) {
        put(&self.presence, source);
    }

    pub fn set_devices(&self, source: Arc<dyn DeviceSource>) {
        put(&self.devices, source);
    }

    pub fn set_calls(&self, source: Arc<dyn CallSource>) {
        put(&self.calls, Some(source));
    }

    pub fn set_clock(&self, clock: Arc<dyn Clock>) {
        put(&self.clock, clock);
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        get(&self.clock)
    }

    pub fn presence(&self) -> Presence {
        get(&self.presence).presence()
    }

    pub fn devices(&self, settings: &RingSettings) -> Vec<Device> {
        get(&self.devices).devices(settings)
    }

    /// What this desktop knows of `call`.
    pub fn call_info(&self, call: &str) -> Option<CallInfo> {
        get(&self.calls).and_then(|c| c.facts(call))
    }

    /// The first `up_to` words a ring for `call` was made on are used up (see [`CallSource::consume_turns`]).
    pub(super) fn use_up_asked_turns(&self, call: &str, up_to: u64) {
        if let Some(calls) = get(&self.calls) {
            calls.consume_turns(call, up_to);
        }
    }

    /// The phone reports the call ended.
    pub fn call_ended_by_phone(&self, call: &str) {
        if let Some(calls) = get(&self.calls) {
            calls.call_ended_by_phone(call);
        }
    }

    /// Tell `listener` whenever a change to the settings changes what the receptionist may do.
    pub fn set_on_features(&self, listener: Arc<dyn Fn(super::Features) + Send + Sync>) {
        put(&self.on_features, Some(listener));
    }

    /// Change some of the owner's settings (see [`SettingsStore::change`]).
    pub fn change_settings(&self, change: &Value) -> Result<(), SettingsError> {
        let before = self.features();
        self.settings.change(change)?;
        let after = self.features();
        if before != after {
            if let Some(listener) = get(&self.on_features) {
                listener(after);
            }
        }
        Ok(())
    }

    /// The plan for a request on `call`, without counting it, and whether it was made `no_endpoint` because only this computer's toast
    /// would have rung (the owner is then told why nobody rang).
    fn decide(&self, call: &str, reason: Reason, info: &CallInfo, settings: &RingSettings) -> (RingPlan, bool) {
        let now_unix = self.clock().unix();
        let local = self.clock().local();
        let caller_key = caller_key(&info.from);
        let counters = self.attempts.lock().unwrap_or_else(|e| e.into_inner()).counters(call, &caller_key, now_unix);
        let mut effective = settings.clone();
        effective.away = settings.away_at(now_unix);
        // Callers who hide their number share one small bucket of the hour, whatever the owner allows a known caller.
        if caller_key == super::limits::WITHHELD {
            effective.limits.per_caller_hour = effective.limits.per_caller_hour.min(super::limits::WITHHELD_PER_HOUR);
        }
        let inputs = Inputs {
            devices: self.devices(settings),
            presence: self.presence(),
            // There is no relay in this build: a phone is reached by the plugin's own session with it.
            relay_healthy: true,
            call: CallFacts {
                reason,
                caller_asked_confirmed: phrases::caller_asked_for(&info.turns, &get(&self.names)()),
                urgent_confirmed: phrases::urgent(&info.turns, &settings.urgent_phrases),
                // A number on the owner's list passes quiet hours, and nothing more: a caller ID can be faked.
                caller_is_vip: settings.is_vip(&info.from),
                vip_bypasses_limits: false,
            },
            limits: counters,
            now: Now::of(&local),
            settings: effective,
        };
        Ring::plan_for(&inputs)
    }

    /// The plan the policy makes for `inputs`, as this desktop answers with it: what the reference plans, with two things changed that the phone
    /// plugin's contract and the owner's own purpose ask for.
    ///
    /// A phone rings for a caller who asks for the owner whether or not they are at the computer, unless they said phones never ring: the
    /// reference rings phones only when the owner is away, and only names a Companion that is a Windows one, so an owner at their computer with
    /// a phone approved and nothing ticked as this computer's would ring nobody (the plan would name only the toast, which is not a device).
    /// "Ring when I am away" therefore also holds when there is no Companion on this computer to take the call instead; the reference vectors
    /// are the raw policy, and this is what is done with its answer. Then a plan that would ring only the toast is made `no_endpoint`
    /// (see [`name_somebody`]).
    pub(super) fn plan_for(inputs: &Inputs) -> (RingPlan, bool) {
        let first = plan(inputs);
        let nobody = (first.rings() && first.targets().is_empty()) || (first.decision == Decision::MessageOnly && first.reason == PlanReason::NoEndpoint);
        if nobody && inputs.settings.phone_ring == PhoneRing::WhenAway {
            let mut again = inputs.clone();
            again.settings.phone_ring = PhoneRing::Always;
            let with_phones = plan(&again);
            if with_phones.rings() && !with_phones.targets().is_empty() {
                return (with_phones, false);
            }
        }
        name_somebody(first)
    }

    /// Whether a request to reach the owner on `call` is allowed: judged on what this desktop heard
    /// of the call (an unknown call is refused as a request nobody asked). A ring that is allowed
    /// counts as a try now, and is kept for [`PLAN_TTL`] under its plan id.
    pub fn authorise(&self, call: &str, reason: Reason) -> Authorised {
        let settings = self.settings.get();
        let Some(info) = self.call_info(call).filter(|_| !self.call_is_over(call)) else {
            return Authorised { plan: unknown_call(&settings), plan_id: new_plan_id(), reason_allowed: false, caller_number: String::new(), caller_name: String::new() };
        };
        self.judge(call, reason, info, &settings)
    }

    fn judge(&self, call: &str, reason: Reason, info: CallInfo, settings: &RingSettings) -> Authorised {
        let (plan, _) = self.decide(call, reason, &info, settings);
        // Nobody could be rung for want of a device that may ring: the owner is told, and why.
        if plan.reason == PlanReason::NoEndpoint {
            self.note_no_device(call, &info, self.cause(settings));
        }
        let reason_allowed = plan.decision == Decision::Ring && vouches_for(reason, settings, &info.turns);
        let id = new_plan_id();
        let generation = self.call_generation(call);
        if plan.decision == Decision::Ring {
            let now_unix = self.clock().unix();
            self.attempts.lock().unwrap_or_else(|e| e.into_inner()).record(call, &caller_key(&info.from), now_unix);
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            grants.retain(|_, g| g.at.elapsed() < PLAN_TTL);
            grants.insert(id.clone(), Grant { plan_id: id.clone(), call_id: call.to_string(), plan: plan.clone(), reason_allowed, at: Instant::now(), claimed: false, judged: info.total, generation });
        }
        Authorised { plan, plan_id: id, reason_allowed, caller_number: info.from, caller_name: info.name }
    }

    /// The plugin's question `oaiy.ring.plan` about a request on `call`: the plan this desktop already
    /// allowed for it (once), else a fresh judgement, which counts as a try. The call is judged on this
    /// desktop's own record of it; `fallback` (what the plugin says of it) is used only for a call this
    /// desktop has no record of.
    pub fn plan_for_plugin(&self, call: &str, reason: Reason, fallback: CallInfo) -> Authorised {
        let settings = self.settings.get();
        // A call that is over (the caller hung up while the request was on its way) is not planned for: nothing rings, and no try is counted.
        if self.call_is_over(call) {
            return Authorised { plan: unknown_call(&settings), plan_id: new_plan_id(), reason_allowed: false, caller_number: String::new(), caller_name: String::new() };
        }
        let info = self.call_info(call).unwrap_or(fallback);
        // A plan this desktop allowed for an earlier beginning of the call (its session was made anew, or handed back, since) is not the one the
        // plugin asks about: it is let go, its try is given back, and the request is judged afresh on the call as it is.
        let generation = self.call_generation(call);
        let stale = {
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            let stale: Vec<String> = grants.iter().filter(|(_, g)| g.call_id == call && !g.claimed && g.generation != generation).map(|(id, _)| id.clone()).collect();
            stale.into_iter().filter(|id| grants.remove(id).is_some()).count()
        };
        for _ in 0..stale {
            self.attempts.lock().unwrap_or_else(|e| e.into_inner()).forget_last(call);
        }
        {
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            grants.retain(|_, g| g.at.elapsed() < PLAN_TTL);
            if let Some(grant) = grants.values_mut().find(|g| g.call_id == call && !g.claimed) {
                grant.claimed = true;
                return Authorised { plan: grant.plan.clone(), plan_id: grant.plan_id.clone(), reason_allowed: grant.reason_allowed, caller_number: info.from, caller_name: info.name };
            }
        }
        let authorised = self.judge(call, reason, info, &settings);
        if authorised.rings() {
            if let Some(g) = self.grants.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&authorised.plan_id) {
                g.claimed = true;
            }
        }
        authorised
    }

    /// Whether the plan `plan_id` is one this desktop allowed for `call` and the plugin has asked for.
    pub fn plan_is_claimed(&self, plan_id: &str, call: &str) -> bool {
        self.grants.lock().unwrap_or_else(|e| e.into_inner()).get(plan_id).is_some_and(|g| g.claimed && g.call_id == call && g.at.elapsed() < PLAN_TTL * 4)
    }
}

/// `plan` as this desktop answers with it, and whether it was made `no_endpoint` because only this computer's toast would have rung.
///
/// The reference rings this computer's toast on its own (its vector V01: the owner at the PC). But a toast is not somebody a call can
/// be offered to: the phone plugin offers a transfer only to the devices a plan names, so a plan that names none opens nothing (it
/// answers `no_endpoint`), and a try counted for it would be spent for a ring that cannot happen. Decided here, before anything is
/// counted: the caller is offered a message, and the owner is told why nobody rang.
pub(super) fn name_somebody(plan: RingPlan) -> (RingPlan, bool) {
    if plan.rings() && plan.targets().is_empty() {
        return (RingPlan::refuse(PlanReason::NoEndpoint, Decision::MessageOnly), true);
    }
    (plan, false)
}

/// Who a call's limits are counted against: the caller's number (its last nine digits), or, for every hidden, withheld or
/// unparseable number, one shared bucket ([`super::limits::WITHHELD`]).
pub(super) fn caller_key(from: &str) -> String {
    crate::voice::contacts::key(from).unwrap_or_else(|| super::limits::WITHHELD.to_string())
}

/// A new plan id: letters, digits and an underscore, which the plugin takes as a token.
fn new_plan_id() -> String {
    format!("plan_{}", &uuid::Uuid::new_v4().simple().to_string()[..16])
}

/// Whether this desktop vouches for `reason` on a call where the caller said `turns`: for `urgent` only, when the owner
/// allows the receptionist to ask on its own for urgent things and the caller's own words held one of their urgent phrases.
/// It is judged here on its own, whatever the plan says, so the two checks are each tested and neither leans on the other.
pub(super) fn vouches_for(reason: Reason, settings: &RingSettings, turns: &[String]) -> bool {
    reason == Reason::Urgent && settings.initiative == super::settings::Initiative::OnRequestOrUrgent && phrases::urgent(turns, &settings.urgent_phrases)
}

/// The plan for a call this desktop has no record of: nobody it heard asked for anyone.
fn unknown_call(settings: &RingSettings) -> RingPlan {
    let (decision, reason) = if settings.enabled { (Decision::Refused, PlanReason::CallerDidNotAsk) } else { (Decision::MessageOnly, PlanReason::Disabled) };
    RingPlan { decision, reason, ring_seconds: 0, phones: Vec::new(), wake: Vec::new(), desktop_toast: false, desktop_companions: Vec::new() }
}

/// A plan as the plugin reads it (`oaiy.ring.plan` answers with this).
pub fn plan_result(authorised: &Authorised) -> Value {
    let p = &authorised.plan;
    json!({
        "planId": authorised.plan_id,
        "decision": p.decision,
        "reason": p.reason,
        "ringSeconds": p.ring_seconds,
        "phones": p.phones,
        "wake": p.wake,
        "desktopToast": p.desktop_toast,
        "desktopCompanions": p.desktop_companions,
        "reasonAllowed": authorised.reason_allowed,
    })
}
