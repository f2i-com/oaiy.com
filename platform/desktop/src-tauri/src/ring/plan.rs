//! The ring policy: who is rung for a caller who wants a person, for how long, or why nobody is.
//!
//! [`plan`] is a pure function of the owner's settings, whether they are at the
//! computer, the devices that could take the call, the time and the caller's
//! record. It follows the reference in `docs/contracts/transfer/` (the design's
//! Appendix A.8) rule for rule, and the 33 conformance vectors in
//! `tests/vectors.json` are its known answers.
//!
//! The rules, first match wins:
//!
//! 1. The master switch is off: `message_only` / `disabled`.
//! 2. The model said the caller asked, and the caller's own words do not
//!    (the phrase check, [`super::phrases`]): `refused` / `caller_did_not_ask`.
//!    Any other reason needs `initiative` to allow it (`initiative_off`), and
//!    `urgent` needs the caller's words to match a phrase the owner chose
//!    (`not_urgent`).
//! 3. The limits: per call, the gap between tries, per caller per hour (a VIP is
//!    exempt) and overall per hour: `refused` / `limit_*`.
//! 4. Quiet hours: only an urgent request the owner allowed, or a VIP, passes.
//! 5. Who rings: the desktop while the owner is at it, phones (second devices)
//!    when they are away or always, per the settings.
//! 6. Nobody to ring: `message_only` / `all_do_not_disturb` or `no_endpoint`.
//! 7. Otherwise `ring`, for the time set (20 to 90 seconds, 30 at most when only
//!    the desktop rings).
//!
//! `message_only` means the receptionist offers to take a message; `refused`
//! means it does not offer a person at all. Whatever the answer, the fallback is
//! taking a message, never silence.

use serde::{Deserialize, Serialize};

use super::settings::{QuietHours, RingSettings};

/// The shortest a ring lasts, seconds.
pub const RING_MIN: u32 = 20;
/// The longest a ring lasts, seconds.
pub const RING_MAX: u32 = 90;
/// How long a ring lasts unless the owner says, seconds.
pub const RING_DEFAULT: u32 = 40;
/// A ring only the desktop hears lasts this long at most: an owner at the computer answers or does not.
pub const DESKTOP_ONLY_CAP: u32 = 30;

/// Why the model asks for a person.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// The caller asked for the owner, a manager or a person.
    CallerAsked,
    /// The model thinks the matter is urgent (only when the owner allowed it).
    Urgent,
    /// A rule the owner wrote (not offered to the model yet).
    PolicyRule,
}

impl Reason {
    pub fn parse(s: &str) -> Option<Reason> {
        match s {
            "caller_asked" => Some(Reason::CallerAsked),
            "urgent" => Some(Reason::Urgent),
            "policy_rule" => Some(Reason::PolicyRule),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Reason::CallerAsked => "caller_asked",
            Reason::Urgent => "urgent",
            Reason::PolicyRule => "policy_rule",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Ring,
    MessageOnly,
    Refused,
}

/// Why a plan is what it is. `ok` is a ring; the rest are the reasons a tool result carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReason {
    Ok,
    Disabled,
    InitiativeOff,
    NotUrgent,
    CallerDidNotAsk,
    LimitCall,
    LimitGap,
    LimitCaller,
    LimitGlobal,
    QuietHours,
    AllDoNotDisturb,
    /// Nobody a call can be offered to: no device at all, or only this computer's toast, which is a notification and not
    /// somebody (see `Ring::decide`). The phone plugin's reason for a plan that names no device is this one, whether or not it
    /// sets the toast, so it is the one that goes on the wire.
    NoEndpoint,
}

impl PlanReason {
    pub fn as_str(self) -> &'static str {
        match self {
            PlanReason::Ok => "ok",
            PlanReason::Disabled => "disabled",
            PlanReason::InitiativeOff => "initiative_off",
            PlanReason::NotUrgent => "not_urgent",
            PlanReason::CallerDidNotAsk => "caller_did_not_ask",
            PlanReason::LimitCall => "limit_call",
            PlanReason::LimitGap => "limit_gap",
            PlanReason::LimitCaller => "limit_caller",
            PlanReason::LimitGlobal => "limit_global",
            PlanReason::QuietHours => "quiet_hours",
            PlanReason::AllDoNotDisturb => "all_do_not_disturb",
            PlanReason::NoEndpoint => "no_endpoint",
        }
    }
}

/// Whether the owner is at the computer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Presence {
    /// Input within `desktopActiveSeconds`, and the session is unlocked.
    Active,
    /// No input for longer.
    Idle,
    /// The session is locked.
    Locked,
    /// Not known (the headless server, or the window is not running): never a reason to ring the desktop.
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A device that lends the call its microphone and speaker (the default).
    SecondDevice,
    /// The handset carrying the cellular call: never rung.
    GatewayHandset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// The Windows Companion on this computer.
    Windows,
    /// A phone.
    Android,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Available,
    Busy,
    DoNotDisturb,
}

/// A device that could take the call: a companion the owner approved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Device {
    /// Its endpoint key's thumbprint.
    pub id: String,
    pub role: Role,
    pub kind: DeviceKind,
    /// Its key is approved in the plugin's roster.
    pub call_authority: bool,
    /// It may take a transferred call (takeover, resume and assistance grants).
    pub can_take: bool,
    /// It has a live session now.
    pub online: bool,
    /// It can be woken by a push.
    pub pushable: bool,
    pub availability: Availability,
}

/// What is known of the call and its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallFacts {
    pub reason: Reason,
    /// The caller's own words asked for a person (checked here, not taken from the model).
    pub caller_asked_confirmed: bool,
    /// The caller's words match one of the owner's urgent phrases.
    pub urgent_confirmed: bool,
    pub caller_is_vip: bool,
    /// A VIP is exempt from the per-caller hourly limit (the reference). This desktop turns it off: a caller ID can be faked,
    /// so a number on the owner's list is a hint for quiet hours and never a way round the limits. The reference vectors
    /// leave it out, and it is then on.
    #[serde(default = "vip_bypass_by_default")]
    pub vip_bypasses_limits: bool,
}

fn vip_bypass_by_default() -> bool {
    true
}

/// Tries so far, from the attempt ledger ([`super::limits`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Counters {
    pub attempts_this_call: u32,
    /// Seconds since this call's last try (None: none yet).
    pub seconds_since_last_attempt: Option<u64>,
    pub caller_attempts_last_hour: u32,
    pub global_attempts_last_hour: u32,
}

/// The time on the owner's clock: the day (0 Sunday to 6 Saturday) and the minute of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Now {
    pub day: u8,
    pub minutes: u32,
}

impl Now {
    /// The day and minute of `time` as its own clock shows them: whatever offset it
    /// carries (the owner's, daylight saving included) is the wall clock quiet
    /// hours are written in.
    pub fn of<Tz: chrono::TimeZone>(time: &chrono::DateTime<Tz>) -> Now {
        use chrono::{Datelike, Timelike};
        Now { day: time.weekday().num_days_from_sunday() as u8, minutes: time.hour() * 60 + time.minute() }
    }
}

/// Everything [`plan`] reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Inputs {
    pub settings: RingSettings,
    pub presence: Presence,
    pub devices: Vec<Device>,
    /// The relay (the way a phone is reached) answered lately. In this build there is no relay: true.
    pub relay_healthy: bool,
    pub call: CallFacts,
    pub limits: Counters,
    pub now: Now,
}

/// What [`plan`] decides.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RingPlan {
    pub decision: Decision,
    pub reason: PlanReason,
    pub ring_seconds: u32,
    /// Phones to offer the call to, by endpoint thumbprint.
    pub phones: Vec<String>,
    /// Of those, the ones a push may wake.
    pub wake: Vec<String>,
    /// The desktop rings: its notification and dialog.
    pub desktop_toast: bool,
    /// The Windows Companions on this computer to offer the call to.
    pub desktop_companions: Vec<String>,
}

impl RingPlan {
    pub(crate) fn refuse(reason: PlanReason, decision: Decision) -> RingPlan {
        RingPlan { decision, reason, ring_seconds: 0, phones: Vec::new(), wake: Vec::new(), desktop_toast: false, desktop_companions: Vec::new() }
    }

    /// Whether anyone is rung.
    pub fn rings(&self) -> bool {
        self.decision == Decision::Ring
    }

    /// Every device the plugin should offer the call to: the phones and the Windows Companions.
    pub fn targets(&self) -> Vec<String> {
        let mut all = self.phones.clone();
        for id in &self.desktop_companions {
            if !all.contains(id) {
                all.push(id.clone());
            }
        }
        all
    }
}

fn minutes_of(hhmm: &str) -> Option<u32> {
    let (h, m) = hhmm.split_once(':')?;
    let (h, m): (u32, u32) = (h.trim().parse().ok()?, m.trim().parse().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

/// Whether the owner's clock is inside the quiet hours. A window that ends before it starts crosses
/// midnight: the hours after midnight belong to the window that began the evening before, so a day
/// that is off in `days` is off for the evening and not for the small hours that follow it.
pub fn in_quiet_hours(q: &QuietHours, day: u8, minutes: u32) -> bool {
    if !q.enabled {
        return false;
    }
    let (Some(start), Some(end)) = (minutes_of(&q.start), minutes_of(&q.end)) else { return false };
    let day_on = |d: u8| q.days & (1 << (d % 7)) != 0;
    if start == end {
        return false;
    }
    if start < end {
        return day_on(day) && minutes >= start && minutes < end;
    }
    if minutes >= start {
        return day_on(day);
    }
    if minutes < end {
        return day_on((day + 6) % 7);
    }
    false
}

/// Decide who rings. See the module docs for the rules.
pub fn plan(i: &Inputs) -> RingPlan {
    let s = &i.settings;
    if !s.enabled {
        return RingPlan::refuse(PlanReason::Disabled, Decision::MessageOnly);
    }
    if i.call.reason == Reason::CallerAsked && !i.call.caller_asked_confirmed {
        return RingPlan::refuse(PlanReason::CallerDidNotAsk, Decision::Refused);
    }
    if i.call.reason != Reason::CallerAsked {
        if s.initiative != super::settings::Initiative::OnRequestOrUrgent {
            return RingPlan::refuse(PlanReason::InitiativeOff, Decision::MessageOnly);
        }
        if i.call.reason == Reason::Urgent && !i.call.urgent_confirmed {
            return RingPlan::refuse(PlanReason::NotUrgent, Decision::MessageOnly);
        }
    }
    let (l, c) = (&s.limits, &i.limits);
    if c.attempts_this_call >= l.per_call {
        return RingPlan::refuse(PlanReason::LimitCall, Decision::Refused);
    }
    if c.seconds_since_last_attempt.is_some_and(|since| since < l.gap_seconds) {
        return RingPlan::refuse(PlanReason::LimitGap, Decision::Refused);
    }
    if !(i.call.caller_is_vip && i.call.vip_bypasses_limits) && c.caller_attempts_last_hour >= l.per_caller_hour {
        return RingPlan::refuse(PlanReason::LimitCaller, Decision::Refused);
    }
    if c.global_attempts_last_hour >= l.global_hour {
        return RingPlan::refuse(PlanReason::LimitGlobal, Decision::Refused);
    }
    if in_quiet_hours(&s.quiet_hours, i.now.day, i.now.minutes) {
        let pass_urgent = s.quiet_hours.allow_urgent && i.call.reason == Reason::Urgent;
        let pass_vip = s.quiet_hours.allow_vip && i.call.caller_is_vip;
        if !pass_urgent && !pass_vip {
            return RingPlan::refuse(PlanReason::QuietHours, Decision::MessageOnly);
        }
    }

    use super::settings::{Away, DesktopRing, PhoneRing};
    let p = i.presence;
    let away = s.away == Away::On || (s.away == Away::Auto && p != Presence::Active);
    // Every ring to a phone goes through the relay: while it cannot be reached, no phone can be rung.
    let reachable = |d: &Device| i.relay_healthy && (d.online || d.pushable);
    let candidates: Vec<&Device> = i.devices.iter().filter(|d| d.call_authority && d.can_take && d.role == Role::SecondDevice && reachable(d)).collect();
    let available: Vec<&Device> = candidates.iter().copied().filter(|d| d.availability == Availability::Available).collect();
    let phone_wanted = s.phone_ring == PhoneRing::Always || (s.phone_ring == PhoneRing::WhenAway && away);
    let phones: Vec<&Device> = if phone_wanted { available.iter().copied().filter(|d| d.kind != DeviceKind::Windows).collect() } else { Vec::new() };
    let desktop_allowed = s.desktop_ring != DesktopRing::Never && s.away != Away::On && p != Presence::Off && (s.desktop_ring == DesktopRing::Always || p == Presence::Active);
    // Every paired Windows Companion is named, running or not: the toast is what starts one that is not, and the phone plugin offers
    // it the request when it connects inside the ring window. (The reference names only one that is online; a plan for the owner at
    // the computer would then name nobody at the very moment it matters, and the plugin answers a plan that names nobody `no_endpoint`.)
    let desktop_companions: Vec<String> = if desktop_allowed {
        i.devices
            .iter()
            .filter(|d| d.call_authority && d.can_take && d.kind == DeviceKind::Windows && d.availability == Availability::Available)
            .map(|d| d.id.clone())
            .collect()
    } else {
        Vec::new()
    };
    let phone_ids: Vec<String> = phones.iter().map(|d| d.id.clone()).collect();
    if phone_ids.is_empty() && !desktop_allowed && desktop_companions.is_empty() {
        let dnd = phone_wanted && !candidates.is_empty() && available.is_empty();
        return RingPlan::refuse(if dnd { PlanReason::AllDoNotDisturb } else { PlanReason::NoEndpoint }, Decision::MessageOnly);
    }
    let mut ring = s.ring_seconds.clamp(RING_MIN, RING_MAX);
    if phone_ids.is_empty() {
        ring = ring.min(DESKTOP_ONLY_CAP);
    }
    let wake: Vec<String> = phones.iter().filter(|d| d.pushable).map(|d| d.id.clone()).collect();
    RingPlan { decision: Decision::Ring, reason: PlanReason::Ok, ring_seconds: ring, phones: phone_ids, wake, desktop_toast: desktop_allowed, desktop_companions }
}
