//! The ring policy: the design's 33 vectors, and the cases they do not cover
//! (quiet hours across midnight and daylight saving, do-not-disturb, no
//! endpoints, several endpoints, and that adding a restriction never adds a ring).

use serde_json::{json, Value};

use super::plan::{in_quiet_hours, Availability, Decision, Device, DeviceKind, Inputs, Now, PlanReason, Presence, Reason, Role, DESKTOP_ONLY_CAP};
use super::settings::{Away, DesktopRing, PhoneRing, QuietHours};
use super::*;

const VECTORS: &str = include_str!("tests/vectors.json");

/// `patch` laid over `base`: objects member by member, anything else replaced.
fn merge(base: &mut Value, patch: &Value) {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                match b.get_mut(k) {
                    Some(existing) if existing.is_object() && v.is_object() => merge(existing, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (base, patch) => *base = patch.clone(),
    }
}

fn base() -> Inputs {
    let doc: Value = serde_json::from_str(VECTORS).unwrap();
    serde_json::from_value(doc["base"].clone()).unwrap()
}

fn phone(id: &str) -> Device {
    Device { id: id.into(), role: Role::SecondDevice, kind: DeviceKind::Android, call_authority: true, can_take: true, online: false, pushable: true, availability: Availability::Available }
}

fn windows(id: &str) -> Device {
    Device { kind: DeviceKind::Windows, online: true, pushable: false, ..phone(id) }
}

#[test]
fn the_conformance_vectors_give_their_known_answers() {
    let doc: Value = serde_json::from_str(VECTORS).unwrap();
    let vectors = doc["vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 33);
    for v in vectors {
        let mut input = doc["base"].clone();
        merge(&mut input, &v["patch"]);
        let inputs: Inputs = serde_json::from_value(input).unwrap_or_else(|e| panic!("{}: {e}", v["id"]));
        let got = serde_json::to_value(plan(&inputs)).unwrap();
        assert_eq!(got, v["expect"], "{} {}", v["id"], v["case"]);
    }
}

#[test]
fn quiet_hours_across_midnight_belong_to_the_day_they_began() {
    // 22:00 to 06:00, Fridays only (bit 5): Friday night and the small hours after it, Saturday.
    let q = QuietHours { enabled: true, start: "22:00".into(), end: "06:00".into(), days: 1 << 5, allow_urgent: false, allow_vip: true };
    let (fri, sat, thu, sun) = (5, 6, 4, 0);
    assert!(in_quiet_hours(&q, fri, 22 * 60), "Friday 22:00, the window's first minute");
    assert!(in_quiet_hours(&q, fri, 23 * 60 + 59));
    assert!(in_quiet_hours(&q, sat, 0), "Saturday midnight is still Friday's window");
    assert!(in_quiet_hours(&q, sat, 5 * 60 + 59));
    assert!(!in_quiet_hours(&q, sat, 6 * 60), "06:00 is outside");
    assert!(!in_quiet_hours(&q, sat, 22 * 60), "Saturday evening: Saturday is not a day it is on");
    assert!(!in_quiet_hours(&q, fri, 3 * 60), "Friday's small hours belong to Thursday's window, which is off");
    assert!(!in_quiet_hours(&q, thu, 23 * 60), "Thursday night is off");
    assert!(!in_quiet_hours(&q, sun, 3 * 60), "Sunday's small hours follow Saturday, which is off");
    // Every day on: no gap at midnight, none at the day change of the week (Saturday into Sunday).
    let all = QuietHours { days: 127, ..q.clone() };
    for day in 0..7u8 {
        assert!(in_quiet_hours(&all, day, 23 * 60 + 59) && in_quiet_hours(&all, (day + 1) % 7, 0), "day {day} into the next");
        assert!(!in_quiet_hours(&all, day, 12 * 60));
    }
}

#[test]
fn quiet_hours_within_one_day_and_the_odd_cases() {
    let weekdays = QuietHours { enabled: true, start: "09:00".into(), end: "17:00".into(), days: 0b011_1110, allow_urgent: false, allow_vip: true };
    assert!(in_quiet_hours(&weekdays, 1, 9 * 60) && in_quiet_hours(&weekdays, 5, 16 * 60 + 59));
    assert!(!in_quiet_hours(&weekdays, 1, 9 * 60 - 1) && !in_quiet_hours(&weekdays, 1, 17 * 60), "the end is outside");
    assert!(!in_quiet_hours(&weekdays, 0, 12 * 60) && !in_quiet_hours(&weekdays, 6, 12 * 60), "the weekend is off");
    // A window with the same start and end is no window; a disabled one never applies; times that are not times never apply.
    let same = QuietHours { start: "08:00".into(), end: "08:00".into(), ..weekdays.clone() };
    assert!(!in_quiet_hours(&same, 1, 8 * 60));
    assert!(!in_quiet_hours(&QuietHours { enabled: false, ..weekdays.clone() }, 1, 12 * 60));
    assert!(!in_quiet_hours(&QuietHours { start: "late".into(), ..weekdays.clone() }, 1, 12 * 60));
    assert!(!in_quiet_hours(&QuietHours { end: "24:00".into(), ..weekdays }, 1, 12 * 60));
    // A window ending at midnight exactly, and one starting at it.
    let evening = QuietHours { enabled: true, start: "20:00".into(), end: "00:00".into(), days: 127, allow_urgent: false, allow_vip: true };
    assert!(in_quiet_hours(&evening, 2, 23 * 60 + 59) && !in_quiet_hours(&evening, 3, 0), "ends at midnight");
    let small = QuietHours { start: "00:00".into(), end: "05:00".into(), ..evening };
    assert!(in_quiet_hours(&small, 3, 0) && in_quiet_hours(&small, 3, 4 * 60 + 59) && !in_quiet_hours(&small, 3, 5 * 60));
}

#[test]
fn the_owners_clock_is_the_one_the_time_carries_so_zone_and_daylight_saving_move_the_window() {
    use chrono::{DateTime, FixedOffset};
    let q = QuietHours { enabled: true, start: "21:00".into(), end: "07:00".into(), days: 127, allow_urgent: false, allow_vip: true };
    let at = |iso: &str| -> Now { Now::of(&DateTime::parse_from_rfc3339(iso).unwrap()) };
    let quiet = |now: Now| in_quiet_hours(&q, now.day, now.minutes);

    // The same instant, four clocks: Wednesday 30 Sep 2026, 10:30 UTC.
    let instant = DateTime::parse_from_rfc3339("2026-09-30T10:30:00+00:00").unwrap();
    let zones = [(0, false), (10, false), (11, true), (-5, true)];
    for (hours, expect) in zones {
        let local = instant.with_timezone(&FixedOffset::east_opt(hours * 3600).unwrap());
        assert_eq!(quiet(Now::of(&local)), expect, "UTC{hours:+}: {} {}", local.format("%a %H:%M"), expect);
    }
    // Sydney's clocks go forward at 02:00 on Sunday 4 Oct 2026 (+10 to +11): 02:30 that night never happens,
    // and the small hours either side of the gap are one window.
    let late_saturday = at("2026-10-03T23:30:00+10:00");
    assert_eq!((late_saturday.day, late_saturday.minutes), (6, 23 * 60 + 30));
    let gap_edge = DateTime::parse_from_rfc3339("2026-10-04T01:59:00+10:00").unwrap();
    let past_gap = DateTime::parse_from_rfc3339("2026-10-04T03:00:00+11:00").unwrap();
    assert_eq!((past_gap - gap_edge).num_minutes(), 1, "one real minute apart");
    assert_eq!((Now::of(&gap_edge).minutes, Now::of(&past_gap).minutes, Now::of(&past_gap).day), (119, 180, 0), "01:59 is followed by 03:00, on the Sunday");
    assert!(quiet(late_saturday) && quiet(Now::of(&gap_edge)) && quiet(Now::of(&past_gap)));
    // ...and back at 03:00 on Sunday 5 Apr 2026 (+11 to +10): 02:30 happens twice, and both are inside the window.
    let first = DateTime::parse_from_rfc3339("2026-04-05T02:30:00+11:00").unwrap();
    let second = DateTime::parse_from_rfc3339("2026-04-05T02:30:00+10:00").unwrap();
    assert_eq!((second - first).num_minutes(), 60, "an hour apart");
    assert_eq!(Now::of(&first), Now::of(&second), "one wall-clock minute, twice");
    assert!(quiet(Now::of(&first)) && quiet(Now::of(&second)));
    // The window's own edges do not move with the offset: 06:59 in, 07:00 out, in summer and in winter time.
    for offset in ["+10:00", "+11:00"] {
        let inside = at(&format!("2026-10-06T06:59:00{offset}"));
        let outside = at(&format!("2026-10-06T07:00:00{offset}"));
        assert!(quiet(inside) && !quiet(outside), "{offset}");
    }
    // The day is the owner's day, not UTC's: 23:30 Tuesday in Sydney is Tuesday 13:30 UTC, and 01:00 Wednesday in Sydney is still Tuesday in UTC.
    assert_eq!(at("2026-09-30T01:00:00+10:00").day, 3);
    assert_eq!(DateTime::parse_from_rfc3339("2026-09-30T01:00:00+10:00").unwrap().with_timezone(&FixedOffset::east_opt(0).unwrap()).format("%a").to_string(), "Tue");
}

#[test]
fn do_not_disturb_and_no_endpoint_are_told_apart() {
    // Owner away, the only phone on do-not-disturb: that is the reason, not "no endpoint".
    let mut i = base();
    i.presence = Presence::Idle;
    i.devices[0].availability = Availability::DoNotDisturb;
    let p = plan(&i);
    assert_eq!((p.decision, p.reason), (Decision::MessageOnly, PlanReason::AllDoNotDisturb));
    // Busy is unavailable too.
    i.devices[0].availability = Availability::Busy;
    assert_eq!(plan(&i).reason, PlanReason::AllDoNotDisturb);
    // With one other phone free, the free one rings and the one on do-not-disturb does not.
    i.devices = vec![Device { availability: Availability::DoNotDisturb, ..phone("asleep") }, phone("free")];
    let p = plan(&i);
    assert_eq!((p.decision, p.phones.clone(), p.wake.clone()), (Decision::Ring, vec!["free".to_string()], vec!["free".to_string()]));
    // No devices at all, and the owner away: no endpoint.
    i.devices.clear();
    assert_eq!(plan(&i).reason, PlanReason::NoEndpoint);
    // A phone the owner may not be rung on is not asked for do-not-disturb (phones are not wanted while the owner is at the computer).
    let mut at_desk = base();
    at_desk.devices[0].availability = Availability::DoNotDisturb;
    let p = plan(&at_desk);
    assert_eq!((p.decision, p.desktop_toast, p.ring_seconds), (Decision::Ring, true, DESKTOP_ONLY_CAP), "the desktop still rings");
}

#[test]
fn several_endpoints_ring_together_and_only_the_right_ones() {
    let mut i = base();
    i.settings.phone_ring = PhoneRing::Always;
    i.devices = vec![phone("pixel"), Device { online: true, pushable: false, ..phone("tablet") }, windows("win1"), Device { role: Role::GatewayHandset, ..phone("handset") }, Device { can_take: false, ..phone("viewer") }];
    let p = plan(&i);
    assert_eq!(p.decision, Decision::Ring);
    assert_eq!(p.phones, vec!["pixel".to_string(), "tablet".to_string()], "second devices that may take the call, not the handset, not a viewer, not the Windows Companion");
    assert_eq!(p.wake, vec!["pixel".to_string()], "only the pushable one is woken");
    assert_eq!(p.desktop_companions, vec!["win1".to_string()]);
    assert!(p.desktop_toast);
    assert_eq!(p.targets(), vec!["pixel".to_string(), "tablet".to_string(), "win1".to_string()], "the plugin offers the call to all of them");
    assert_eq!(p.ring_seconds, 40);
    // The Windows Companion is never among the phones, and is rung only while the owner is at the desktop.
    i.presence = Presence::Idle;
    let p = plan(&i);
    assert!(p.desktop_companions.is_empty() && !p.desktop_toast);
    assert_eq!(p.phones, vec!["pixel".to_string(), "tablet".to_string()]);
    // Owner at the desktop, phones only when away: the Companion on this computer and the notification ring, and no phone.
    let mut desk = base();
    desk.devices = vec![phone("pixel"), windows("win1")];
    let p = plan(&desk);
    assert_eq!((p.phones.clone(), p.desktop_companions.clone(), p.desktop_toast, p.ring_seconds), (vec![], vec!["win1".to_string()], true, 30));
    // A Windows Companion that is not connected cannot take a call from the desktop.
    desk.devices[1].online = false;
    assert!(plan(&desk).desktop_companions.is_empty());
}

#[test]
fn a_ring_lasts_the_time_set_inside_twenty_to_ninety_seconds_and_thirty_at_most_for_the_desktop_alone() {
    let mut i = base();
    for (set, phone_ring, presence, expect) in [
        (40, PhoneRing::Always, Presence::Active, 40),
        (0, PhoneRing::Always, Presence::Active, 20),
        (19, PhoneRing::Always, Presence::Active, 20),
        (20, PhoneRing::Always, Presence::Active, 20),
        (90, PhoneRing::Always, Presence::Active, 90),
        (91, PhoneRing::Always, Presence::Active, 90),
        (u32::MAX, PhoneRing::Always, Presence::Active, 90),
        (90, PhoneRing::WhenAway, Presence::Active, 30),
        (25, PhoneRing::WhenAway, Presence::Active, 25),
        (90, PhoneRing::Never, Presence::Active, 30),
    ] {
        i.settings.ring_seconds = set;
        i.settings.phone_ring = phone_ring;
        i.presence = presence;
        let p = plan(&i);
        assert_eq!((p.decision, p.ring_seconds), (Decision::Ring, expect), "{set} {phone_ring:?} {presence:?}");
    }
}

#[test]
fn phones_and_the_desktop_follow_their_own_settings_and_presence() {
    let mut i = base();
    i.presence = Presence::Idle;
    i.settings.phone_ring = PhoneRing::Never;
    assert_eq!(plan(&i).reason, PlanReason::NoEndpoint);
    i.settings.desktop_ring = DesktopRing::Always;
    let p = plan(&i);
    assert_eq!((p.decision, p.desktop_toast, p.phones.len()), (Decision::Ring, true, 0), "always: the desktop rings even with the owner idle");
    // Locked, and the headless case (presence unknown): the desktop is not rung on auto, whatever always says about a phone.
    i.settings.desktop_ring = DesktopRing::Auto;
    i.settings.phone_ring = PhoneRing::WhenAway;
    for presence in [Presence::Locked, Presence::Idle] {
        i.presence = presence;
        let p = plan(&i);
        assert_eq!((p.decision, p.desktop_toast, p.phones.clone()), (Decision::Ring, false, vec!["pixel2".to_string()]), "{presence:?}");
    }
    i.presence = Presence::Off;
    i.settings.desktop_ring = DesktopRing::Always;
    assert!(!plan(&i).desktop_toast, "no presence reading (the headless server): the desktop is never rung");
}

#[test]
fn the_away_switch_decides_who_rings() {
    let mut i = base();
    i.settings.away = Away::On;
    let p = plan(&i);
    assert_eq!((p.desktop_toast, p.phones.clone()), (false, vec!["pixel2".to_string()]));
    i.settings.away = Away::Off;
    i.presence = Presence::Idle;
    let p = plan(&i);
    assert_eq!((p.desktop_toast, p.phones.len()), (false, 0), "away off: not away, so no phone, and idle means no desktop ring on auto");
    assert_eq!(p.reason, PlanReason::NoEndpoint);
}

#[test]
fn the_reasons_are_checked_in_the_order_the_policy_gives() {
    let mut i = base();
    // Everything wrong at once: the master switch speaks first, then whether the caller asked, then the limits, then quiet hours.
    i.settings.enabled = false;
    i.call.caller_asked_confirmed = false;
    i.limits.attempts_this_call = 9;
    i.now.minutes = 22 * 60;
    assert_eq!(plan(&i).reason, PlanReason::Disabled);
    i.settings.enabled = true;
    assert_eq!(plan(&i).reason, PlanReason::CallerDidNotAsk);
    i.call.caller_asked_confirmed = true;
    assert_eq!(plan(&i).reason, PlanReason::LimitCall);
    i.limits.attempts_this_call = 0;
    assert_eq!(plan(&i).reason, PlanReason::QuietHours);
    // A refusal is a refusal: the receptionist is told not to offer a person at all; the rest fall back to a message.
    i.call.caller_asked_confirmed = false;
    assert_eq!(plan(&i).decision, Decision::Refused);
    i.call.caller_asked_confirmed = true;
    assert_eq!(plan(&i).decision, Decision::MessageOnly);
}

#[test]
fn the_limits_are_the_owners_own_numbers() {
    let mut i = base();
    i.settings.limits.per_call = 1;
    i.limits.attempts_this_call = 1;
    assert_eq!(plan(&i).reason, PlanReason::LimitCall);
    i.settings.limits.per_call = 3;
    i.limits.seconds_since_last_attempt = Some(100);
    i.settings.limits.gap_seconds = 120;
    assert_eq!(plan(&i).reason, PlanReason::LimitGap);
    i.limits.seconds_since_last_attempt = Some(120);
    assert_eq!(plan(&i).decision, Decision::Ring, "the gap is over exactly at its length");
    i.settings.limits.per_caller_hour = 5;
    i.limits.caller_attempts_last_hour = 4;
    assert_eq!(plan(&i).decision, Decision::Ring);
    i.limits.caller_attempts_last_hour = 5;
    assert_eq!(plan(&i).reason, PlanReason::LimitCaller);
    i.limits.caller_attempts_last_hour = 0;
    i.settings.limits.global_hour = 4;
    i.limits.global_attempts_last_hour = 4;
    assert_eq!(plan(&i).reason, PlanReason::LimitGlobal);
    i.call.caller_is_vip = true;
    assert_eq!(plan(&i).reason, PlanReason::LimitGlobal, "a VIP is exempt from the per-caller limit only");
}

/// Everything the policy depends on, in every combination that matters, and the ways to restrict it further.
fn every_input() -> Vec<Inputs> {
    let mut all = Vec::new();
    let device_sets: Vec<Vec<Device>> = vec![
        vec![],
        vec![phone("pixel2")],
        vec![Device { online: true, pushable: false, ..phone("pixel2") }],
        vec![phone("pixel2"), windows("win1")],
        vec![Device { availability: Availability::Busy, ..phone("pixel2") }],
        vec![Device { can_take: false, ..phone("pixel2") }],
    ];
    for devices in device_sets {
        for presence in [Presence::Active, Presence::Idle, Presence::Locked, Presence::Off] {
            for phone_ring in [PhoneRing::WhenAway, PhoneRing::Always, PhoneRing::Never] {
                for desktop_ring in [DesktopRing::Auto, DesktopRing::Always, DesktopRing::Never] {
                    for away in [Away::Auto, Away::On, Away::Off] {
                        for reason in [Reason::CallerAsked, Reason::Urgent] {
                            let mut i = base();
                            i.devices = devices.clone();
                            i.presence = presence;
                            i.settings.phone_ring = phone_ring;
                            i.settings.desktop_ring = desktop_ring;
                            i.settings.away = away;
                            i.settings.quiet_hours.enabled = false;
                            i.settings.initiative = settings::Initiative::OnRequestOrUrgent;
                            i.call = plan::CallFacts { reason, caller_asked_confirmed: true, urgent_confirmed: true, caller_is_vip: false, vip_bypasses_limits: true };
                            all.push(i);
                        }
                    }
                }
            }
        }
    }
    all
}

#[test]
fn adding_a_restriction_never_turns_a_message_into_a_ring() {
    type Restrict = fn(&mut Inputs);
    let restrictions: [(&str, Restrict); 9] = [
        ("quiet hours", |i| {
            i.settings.quiet_hours = QuietHours { enabled: true, start: "00:00".into(), end: "23:59".into(), days: 127, allow_urgent: false, allow_vip: false };
        }),
        ("all devices on do-not-disturb", |i| i.devices.iter_mut().for_each(|d| d.availability = Availability::DoNotDisturb)),
        ("the relay is down", |i| i.relay_healthy = false),
        ("the master switch is off", |i| i.settings.enabled = false),
        ("the desktop never rings", |i| i.settings.desktop_ring = DesktopRing::Never),
        ("phones never ring", |i| i.settings.phone_ring = PhoneRing::Never),
        ("the per-call limit is reached", |i| i.limits.attempts_this_call = 2),
        ("the caller did not ask", |i| i.call.caller_asked_confirmed = false),
        ("a gap is not over", |i| i.limits.seconds_since_last_attempt = Some(1)),
    ];
    let mut restricted_rings = 0;
    for input in every_input() {
        let before = plan(&input);
        for (what, restrict) in &restrictions {
            let mut narrower = input.clone();
            restrict(&mut narrower);
            let after = plan(&narrower);
            if before.decision != Decision::Ring {
                assert_ne!(after.decision, Decision::Ring, "{what} turned {:?} into a ring: {narrower:?}", before.reason);
            } else if after.decision == Decision::Ring {
                restricted_rings += 1;
                // A ring that survives a restriction rings no more devices than before.
                assert!(after.phones.iter().all(|p| before.phones.contains(p)), "{what}");
                assert!(after.desktop_companions.iter().all(|p| before.desktop_companions.contains(p)), "{what}");
                assert!(!after.desktop_toast || before.desktop_toast, "{what}");
            }
        }
    }
    assert!(restricted_rings > 0, "the grid has rings that survive some restrictions, so the second half of the check ran");
}

#[test]
fn a_plan_is_the_same_whoever_asks_and_however_often() {
    let i = base();
    assert_eq!(plan(&i), plan(&i.clone()), "pure: nothing is kept between plans");
    // Nothing in a plan carries the model's words or the caller's: a plan names devices by thumbprint and nothing else.
    let p = serde_json::to_value(plan(&i)).unwrap();
    assert_eq!(p.as_object().unwrap().keys().cloned().collect::<std::collections::BTreeSet<_>>(), ["decision", "desktopCompanions", "desktopToast", "reason", "ringSeconds", "phones", "wake"].iter().map(|s| s.to_string()).collect());
}

#[test]
fn a_reason_is_named_as_the_wire_names_it() {
    for (reason, name) in [(Reason::CallerAsked, "caller_asked"), (Reason::Urgent, "urgent"), (Reason::PolicyRule, "policy_rule")] {
        assert_eq!(reason.as_str(), name);
        assert_eq!(Reason::parse(name), Some(reason));
        assert_eq!(serde_json::to_value(reason).unwrap(), json!(name));
    }
    assert_eq!(Reason::parse("Caller_Asked"), None);
    assert_eq!(Reason::parse(""), None);
    for reason in [
        PlanReason::Ok,
        PlanReason::Disabled,
        PlanReason::InitiativeOff,
        PlanReason::NotUrgent,
        PlanReason::CallerDidNotAsk,
        PlanReason::LimitCall,
        PlanReason::LimitGap,
        PlanReason::LimitCaller,
        PlanReason::LimitGlobal,
        PlanReason::QuietHours,
        PlanReason::AllDoNotDisturb,
        PlanReason::NoEndpoint,
        PlanReason::NoDevice,
    ] {
        assert_eq!(serde_json::to_value(reason).unwrap(), json!(reason.as_str()));
    }
}
