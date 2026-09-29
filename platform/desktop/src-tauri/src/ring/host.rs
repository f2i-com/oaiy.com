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
use super::settings::{RingSettings, SettingsError, SettingsStore};
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
    /// What the caller said, oldest first.
    pub turns: Vec<String>,
}

/// Where the ring learns of a call: the voice hub.
pub trait CallSource: Send + Sync {
    /// The call, when this desktop knows it (live, in handoff, or ended within ten minutes).
    fn facts(&self, call: &str) -> Option<CallInfo>;
    /// The phone says the call ended (the plugin's own event): the hub ends one that was handed to the owner.
    fn call_ended_by_phone(&self, call: &str);
    /// The owner answered a ring in the dialog (a decline): the call is told, as it is told what the phone says.
    fn local_outcome(&self, call: &str, request: &str, outcome: crate::voice::transfer::Outcome);
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
    at: Instant,
    claimed: bool,
}

/// What a request to reach the owner came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Authorised {
    /// The plan: a ring, or why not.
    pub plan: RingPlan,
    /// Set when a ring is allowed: the plugin's question about this request is answered with it.
    pub plan_id: Option<String>,
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
    /// The rings going now, and the last ones that ended (see `session.rs`).
    pub(super) sessions: Mutex<super::session::Sessions>,
    notifier: RwLock<Option<Arc<dyn super::session::RingNotifier>>>,
    plugin: RwLock<Option<Arc<dyn super::session::TransferPlugin>>>,
    /// How long past its time a ring waits to hear how it came out before it is over.
    pub(super) expiry_grace: RwLock<Duration>,
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
            sessions: Mutex::new(super::session::Sessions::default()),
            notifier: RwLock::new(None),
            plugin: RwLock::new(None),
            expiry_grace: RwLock::new(super::session::EXPIRY_GRACE),
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

    /// What asks the phone plugin to accept or decline a request on the owner's behalf.
    pub fn set_plugin(&self, plugin: Arc<dyn super::session::TransferPlugin>) {
        put(&self.plugin, Some(plugin));
    }

    pub(super) fn plugin(&self) -> Option<Arc<dyn super::session::TransferPlugin>> {
        get(&self.plugin)
    }

    /// Tell the hub what the owner did in the dialog.
    pub(super) fn local_outcome(&self, call: &str, request: &str, outcome: crate::voice::transfer::Outcome) {
        if let Some(calls) = get(&self.calls) {
            calls.local_outcome(call, request, outcome);
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

    /// A plan this desktop allowed and the plugin asked for: what was decided for the call.
    pub(super) fn claimed_plan(&self, plan_id: &str, call: &str) -> Option<RingPlan> {
        let grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
        grants.get(plan_id).filter(|g| g.claimed && g.call_id == call && g.at.elapsed() < PLAN_TTL * 4).map(|g| g.plan.clone())
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

    /// The plan for a request on `call`, without counting it: for showing what would happen.
    fn decide(&self, call: &str, reason: Reason, info: &CallInfo, settings: &RingSettings) -> RingPlan {
        let now_unix = self.clock().unix();
        let local = self.clock().local();
        let caller_key = crate::voice::contacts::key(&info.from).unwrap_or_else(|| format!("call:{call}"));
        let counters = self.attempts.lock().unwrap_or_else(|e| e.into_inner()).counters(call, &caller_key, now_unix);
        let mut effective = settings.clone();
        effective.away = settings.away_at(now_unix);
        let inputs = Inputs {
            devices: self.devices(settings),
            presence: self.presence(),
            // There is no relay in this build: a phone is reached by the plugin's own session with it.
            relay_healthy: true,
            call: CallFacts {
                reason,
                caller_asked_confirmed: phrases::caller_asked(&info.turns),
                urgent_confirmed: phrases::urgent(&info.turns, &settings.urgent_phrases),
                caller_is_vip: settings.is_vip(&info.from),
            },
            limits: counters,
            now: Now::of(&local),
            settings: effective,
        };
        plan(&inputs)
    }

    /// Whether a request to reach the owner on `call` is allowed: judged on what this desktop heard
    /// of the call (an unknown call is refused as a request nobody asked). A ring that is allowed
    /// counts as a try now, and is kept for [`PLAN_TTL`] under its plan id.
    pub fn authorise(&self, call: &str, reason: Reason) -> Authorised {
        let settings = self.settings.get();
        let Some(info) = self.call_info(call) else {
            return Authorised { plan: unknown_call(&settings), plan_id: None, caller_number: String::new(), caller_name: String::new() };
        };
        self.judge(call, reason, info, &settings)
    }

    fn judge(&self, call: &str, reason: Reason, info: CallInfo, settings: &RingSettings) -> Authorised {
        let plan = self.decide(call, reason, &info, settings);
        let mut plan_id = None;
        if plan.decision == Decision::Ring {
            let now_unix = self.clock().unix();
            let caller_key = crate::voice::contacts::key(&info.from).unwrap_or_else(|| format!("call:{call}"));
            self.attempts.lock().unwrap_or_else(|e| e.into_inner()).record(call, &caller_key, now_unix);
            let id = format!("plan_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            grants.retain(|_, g| g.at.elapsed() < PLAN_TTL);
            grants.insert(id.clone(), Grant { plan_id: id.clone(), call_id: call.to_string(), plan: plan.clone(), at: Instant::now(), claimed: false });
            plan_id = Some(id);
        }
        Authorised { plan, plan_id, caller_number: info.from, caller_name: info.name }
    }

    /// The plugin's question `oaiy.ring.plan` about a request on `call`: the plan this desktop already
    /// allowed for it (once), else a fresh judgement, which counts as a try. The call is judged on this
    /// desktop's own record of it; `fallback` (what the plugin says of it) is used only for a call this
    /// desktop has no record of.
    pub fn plan_for_plugin(&self, call: &str, reason: Reason, fallback: CallInfo) -> Authorised {
        let settings = self.settings.get();
        let info = self.call_info(call).unwrap_or(fallback);
        {
            let mut grants = self.grants.lock().unwrap_or_else(|e| e.into_inner());
            grants.retain(|_, g| g.at.elapsed() < PLAN_TTL);
            if let Some(grant) = grants.values_mut().find(|g| g.call_id == call && !g.claimed) {
                grant.claimed = true;
                return Authorised { plan: grant.plan.clone(), plan_id: Some(grant.plan_id.clone()), caller_number: info.from, caller_name: info.name };
            }
        }
        let authorised = self.judge(call, reason, info, &settings);
        if let Some(id) = &authorised.plan_id {
            if let Some(g) = self.grants.lock().unwrap_or_else(|e| e.into_inner()).get_mut(id) {
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

/// The plan for a call this desktop has no record of: nobody it heard asked for anyone.
fn unknown_call(settings: &RingSettings) -> RingPlan {
    let (decision, reason) = if settings.enabled { (Decision::Refused, PlanReason::CallerDidNotAsk) } else { (Decision::MessageOnly, PlanReason::Disabled) };
    RingPlan { decision, reason, ring_seconds: 0, phones: Vec::new(), wake: Vec::new(), desktop_toast: false, desktop_companions: Vec::new() }
}

/// A plan as the plugin reads it (`oaiy.ring.plan` answers with this).
pub fn plan_result(authorised: &Authorised) -> Value {
    let p = &authorised.plan;
    json!({
        "planId": authorised.plan_id.clone().unwrap_or_default(),
        "decision": p.decision,
        "reason": p.reason,
        "ringSeconds": p.ring_seconds,
        "phones": p.phones,
        "wake": p.wake,
        "desktopToast": p.desktop_toast,
        "desktopCompanions": p.desktop_companions,
    })
}
