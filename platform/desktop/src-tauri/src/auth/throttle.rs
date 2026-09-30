//! The login throttle (design 4.7.4): who may try a password, and when. A pure state machine on a [`Clock`],
//! saved to `throttle.json` (under `login`, beside the failed-bearer throttle's state, in the file the guard
//! writes) so that a restart or an out-of-memory kill does not clear a block or slow mode.
//!
//! | Layer | Rule |
//! |---|---|
//! | L1 per address | Keyed by the effective client address (IPv6 by /64). **5 consecutive failures** block that address from `login`, `setup` and `link` for **15 minutes**; each new block within 24 hours doubles it (15 m, 30 m, 1 h, 2 h, 4 h, 8 h, 16 h, capped at 24 h). A success clears the counters. A blocked request is answered `429` with `Retry-After` and no hashing happens. |
//! | L2 global slow mode | When **20 or more failures** (any address, any anonymous route) fall in the last **60 minutes**, slow mode is on until fewer than 20 remain in that window. |
//! | Known-device lane | A login that presents a valid `dev` cookie skips L1 and L2 and their counters and has its own ladder per device id: **5 consecutive failures** block that device's lane for 15 minutes, doubling to 24 hours (35 guesses a day per device at most); a success clears it. It never clears an address's L1 state. |
//!
//! There is no account lockout, deliberately: a hard lockout of a one-owner account lets any stranger lock the
//! owner out. What the owner is protected by instead is that the device lane and the session lane cannot be
//! filled by anonymous traffic (`lanes.rs`).
//!
//! **What counts.** A failure is a wrong password, a wrong setup code or a wrong link code, from an address, on the
//! anonymous lane. A request that was blocked is not a failure (nothing was checked), and a failure on the device
//! lane is only the device's. The last 256 failure times are kept: that is exact for "20 in 60 minutes" (only the
//! twentieth newest matters) and for the alert's "200 in 24 hours".
//!
//! **The alert.** `login.attack` fires when slow mode has been on continuously for 60 minutes, or 200 failures land
//! in 24 hours, at most once an hour while that holds.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::bearer_throttle::block_ms;
use super::clock::Clock;

/// Consecutive failures that start a block, on an address and on a device.
pub const FAILURES_TO_BLOCK: u32 = 5;
/// Failures in [`SLOW_WINDOW_MS`] that turn slow mode on.
pub const SLOW_FAILURES: usize = 20;
pub const SLOW_WINDOW_MS: u64 = 60 * 60_000;
/// The spacing of the starts of anonymous verifications in slow mode.
pub const SLOW_SPACING_MS: u64 = 5_000;
/// A block that ended longer ago than this starts the ladder again.
pub const RESET_MS: u64 = 24 * 3_600_000;
/// The failure times kept.
pub const KEEP_FAILURES: usize = 256;
/// Slow mode on for this long, continuously, is an attack.
pub const ATTACK_SLOW_MS: u64 = 60 * 60_000;
/// This many failures in a day is an attack.
pub const ATTACK_FAILURES: usize = 200;
pub const ATTACK_WINDOW_MS: u64 = 24 * 3_600_000;
/// The alert is not repeated within this time.
pub const ALERT_EVERY_MS: u64 = 3_600_000;
/// Addresses tracked at once: a flood of rotating addresses cannot grow memory without bound.
pub const MAX_TRACKED: usize = 50_000;
/// Addresses (and devices) written to the file: the most that matter, so that the file stays a few hundred
/// kilobytes whatever is tracked.
pub const MAX_SAVED: usize = 2_000;
/// How often the state is written when it changed.
pub const FLUSH_EVERY_MS: u64 = 5_000;

/// One rung of a ladder: the state of one address or one device.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Ladder {
    /// Failures since the last success or the last block began.
    consecutive: u32,
    /// Blocked until this time (zero: not blocked).
    blocked_until: u64,
    /// The rung of the current block (1 is 15 minutes).
    level: u32,
    /// When the last block ended.
    last_block_end: u64,
    last_seen: u64,
}

impl Ladder {
    fn blocked_for(&self, now: u64) -> Option<u64> {
        (self.blocked_until > now).then(|| (self.blocked_until - now).div_ceil(1000).max(1))
    }

    /// One failure. Whether it began a block.
    fn fail(&mut self, now: u64) -> bool {
        self.last_seen = now;
        self.consecutive += 1;
        if self.consecutive < FAILURES_TO_BLOCK {
            return false;
        }
        self.consecutive = 0;
        // Within a day of the last block ending the next is twice as long; after a quiet day, the first rung.
        self.level =
            if self.last_block_end != 0 && now.saturating_sub(self.last_block_end) < RESET_MS {
                self.level + 1
            } else {
                1
            };
        self.blocked_until = now + block_ms(self.level);
        self.last_block_end = self.blocked_until;
        true
    }

    fn succeed(&mut self, now: u64) {
        self.last_seen = now;
        self.consecutive = 0;
        self.level = 0;
        self.blocked_until = 0;
        self.last_block_end = 0;
    }

    /// Nothing left worth remembering.
    fn is_quiet(&self, now: u64) -> bool {
        self.consecutive == 0
            && self.blocked_until <= now
            && (self.last_block_end == 0 || now.saturating_sub(self.last_block_end) >= RESET_MS)
    }
}

/// What a check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Open,
    /// Blocked: seconds until it may try again.
    Blocked {
        retry_after_s: u64,
    },
}

/// What a failure did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Failed {
    /// This failure began a block of the address (or device).
    pub blocked: bool,
    /// The seconds of that block.
    pub block_s: u64,
}

/// Why the alert fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attack {
    /// Slow mode has been on for an hour without a break.
    SlowModeSustained,
    /// 200 failures within 24 hours.
    ManyFailures,
}

impl Attack {
    pub fn name(self) -> &'static str {
        match self {
            Attack::SlowModeSustained => "slow_mode_60_minutes",
            Attack::ManyFailures => "200_failures_in_24_hours",
        }
    }
}

#[derive(Default)]
struct State {
    addresses: HashMap<String, Ladder>,
    devices: HashMap<String, Ladder>,
    /// Failure times, oldest first, at most [`KEEP_FAILURES`].
    failures: VecDeque<u64>,
    /// When slow mode last turned on and has stayed on.
    slow_since: Option<u64>,
    last_alert: u64,
    dirty: bool,
    last_flush: u64,
}

/// The throttle. One per server; every method takes `&self`.
pub struct LoginThrottle {
    state: Mutex<State>,
    clock: Arc<dyn Clock>,
}

impl LoginThrottle {
    pub fn new(clock: Arc<dyn Clock>) -> LoginThrottle {
        LoginThrottle {
            state: Mutex::new(State::default()),
            clock,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---- L1: an address --------------------------------------------------------------------

    /// Whether `key` (the effective client address, IPv6 by /64) may try now.
    pub fn address_gate(&self, key: &str) -> Gate {
        let now = self.clock.now_ms();
        match self
            .lock()
            .addresses
            .get(key)
            .and_then(|l| l.blocked_for(now))
        {
            Some(retry_after_s) => Gate::Blocked { retry_after_s },
            None => Gate::Open,
        }
    }

    /// One failure of the anonymous lane from `key`. Counts towards its block and towards slow mode.
    pub fn address_failed(&self, key: &str) -> Failed {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        if !s.addresses.contains_key(key) && s.addresses.len() >= MAX_TRACKED {
            evict(&mut s.addresses, now);
        }
        let ladder = s.addresses.entry(key.to_string()).or_default();
        let blocked = ladder.fail(now);
        let block_s = if blocked {
            block_ms(ladder.level).div_ceil(1000)
        } else {
            0
        };
        s.failures.push_back(now);
        while s.failures.len() > KEEP_FAILURES {
            s.failures.pop_front();
        }
        refresh_slow(&mut s, now);
        s.dirty = true;
        Failed { blocked, block_s }
    }

    /// A success from `key`: its counters are cleared.
    pub fn address_succeeded(&self, key: &str) {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        if let Some(l) = s.addresses.get_mut(key) {
            l.succeed(now);
            s.dirty = true;
        }
    }

    // ---- L2: slow mode ---------------------------------------------------------------------

    /// Whether slow mode is on: 20 or more failures in the last 60 minutes.
    pub fn is_slow(&self) -> bool {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        refresh_slow(&mut s, now);
        slow_now(&s, now)
    }

    /// How many failures fall in the last 60 minutes (at most the 256 kept).
    pub fn recent_failures(&self) -> usize {
        let now = self.clock.now_ms();
        self.lock()
            .failures
            .iter()
            .filter(|t| now.saturating_sub(**t) < SLOW_WINDOW_MS)
            .count()
    }

    // ---- the known-device lane -------------------------------------------------------------

    pub fn device_gate(&self, id: &str) -> Gate {
        let now = self.clock.now_ms();
        match self.lock().devices.get(id).and_then(|l| l.blocked_for(now)) {
            Some(retry_after_s) => Gate::Blocked { retry_after_s },
            None => Gate::Open,
        }
    }

    /// One failure on a device's lane. It touches neither an address nor slow mode.
    pub fn device_failed(&self, id: &str) -> Failed {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        let ladder = s.devices.entry(id.to_string()).or_default();
        let blocked = ladder.fail(now);
        let block_s = if blocked {
            block_ms(ladder.level).div_ceil(1000)
        } else {
            0
        };
        s.dirty = true;
        Failed { blocked, block_s }
    }

    pub fn device_succeeded(&self, id: &str) {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        if let Some(l) = s.devices.get_mut(id) {
            l.succeed(now);
            s.dirty = true;
        }
    }

    /// A device that is revoked has no ladder any more.
    pub fn forget_device(&self, id: &str) {
        let mut s = self.lock();
        if s.devices.remove(id).is_some() {
            s.dirty = true;
        }
    }

    // ---- the alert -------------------------------------------------------------------------

    /// Whether the condition of the alert holds now.
    pub fn under_attack(&self) -> Option<Attack> {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        refresh_slow(&mut s, now);
        attack_now(&s, now)
    }

    /// The alert, once: `Some` when the condition holds and it has not been raised in the last hour.
    pub fn take_alert(&self) -> Option<Attack> {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        refresh_slow(&mut s, now);
        let attack = attack_now(&s, now)?;
        if s.last_alert != 0 && now.saturating_sub(s.last_alert) < ALERT_EVERY_MS {
            return None;
        }
        s.last_alert = now;
        s.dirty = true;
        Some(attack)
    }

    // ---- what the console shows ------------------------------------------------------------

    /// The addresses blocked now, with the seconds left: `auth status`.
    pub fn blocked_addresses(&self) -> Vec<(String, u64)> {
        let now = self.clock.now_ms();
        let s = self.lock();
        let mut out: Vec<(String, u64)> = s
            .addresses
            .iter()
            .filter_map(|(k, l)| l.blocked_for(now).map(|r| (k.clone(), r)))
            .collect();
        out.sort();
        out
    }

    /// The devices blocked now.
    pub fn blocked_devices(&self) -> Vec<(String, u64)> {
        let now = self.clock.now_ms();
        let s = self.lock();
        let mut out: Vec<(String, u64)> = s
            .devices
            .iter()
            .filter_map(|(k, l)| l.blocked_for(now).map(|r| (k.clone(), r)))
            .collect();
        out.sort();
        out
    }

    /// How many addresses are tracked.
    pub fn tracked(&self) -> usize {
        self.lock().addresses.len()
    }

    // ---- saving ----------------------------------------------------------------------------

    /// Whether the state changed since it was last written, and it is time to write it (at most every 5 s).
    pub fn flush_due(&self) -> bool {
        let now = self.clock.now_ms();
        let s = self.lock();
        s.dirty && now.saturating_sub(s.last_flush) >= FLUSH_EVERY_MS
    }

    /// The state for the `login` key of `throttle.json` (the file is the guard's: it holds this beside the
    /// failed-bearer throttle's state, and the guard is its only writer), and the mark that it was taken.
    pub fn snapshot(&self) -> Value {
        let now = self.clock.now_ms();
        let mut s = self.lock();
        s.dirty = false;
        s.last_flush = now;
        json!({
            "addresses": saved(&s.addresses, now, MAX_SAVED),
            "devices": saved(&s.devices, now, MAX_SAVED),
            "failures": s.failures.iter().filter(|t| now.saturating_sub(**t) < ATTACK_WINDOW_MS).collect::<Vec<_>>(),
            "slow_since_ms": s.slow_since,
            "last_alert_ms": s.last_alert,
        })
    }

    /// Load a saved state (what `snapshot` gave): a block, the failure window and the ladder survive a restart.
    /// What cannot be read is ignored (a damaged file is worth a warning, not a login that will not start), and
    /// what is read is made sane first, as the failed-bearer throttle does: a file is not trusted to name a block
    /// that ends in the year 3000, or a ladder that is a billion failures up.
    pub fn restore(&self, saved: &Value) {
        if !saved.is_object() {
            return;
        }
        let now = self.clock.now_ms();
        let mut s = self.lock();
        if let Some(m) = saved
            .get("addresses")
            .and_then(|v| serde_json::from_value::<HashMap<String, Ladder>>(v.clone()).ok())
        {
            s.addresses = m
                .into_iter()
                .filter_map(|(k, l)| sane(l, now).map(|l| (k, l)))
                .take(MAX_TRACKED)
                .collect();
        }
        if let Some(m) = saved
            .get("devices")
            .and_then(|v| serde_json::from_value::<HashMap<String, Ladder>>(v.clone()).ok())
        {
            s.devices = m
                .into_iter()
                .filter_map(|(k, l)| sane(l, now).map(|l| (k, l)))
                .take(64)
                .collect();
        }
        if let Some(mut times) = saved
            .get("failures")
            .and_then(|v| serde_json::from_value::<Vec<u64>>(v.clone()).ok())
        {
            times.sort_unstable();
            // The newest ones; and a time in the future (the clock went back) is now.
            let skip = times.len().saturating_sub(KEEP_FAILURES);
            s.failures = times.into_iter().skip(skip).map(|t| t.min(now)).collect();
        }
        s.last_alert = saved
            .get("last_alert_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(now);
        s.slow_since = saved
            .get("slow_since_ms")
            .and_then(Value::as_u64)
            .map(|t| t.min(now));
        refresh_slow(&mut s, now);
    }
}

/// What is worth saving of a table: the ladders that are not quiet, and when there are more than `most` of them
/// the ones that matter most, the blocks that end last first (then the most failures, then the newest). The file is
/// the guard's, written whole every few seconds: a flood of rotating addresses that were each blocked (30 000
/// /64s in an hour is what a botnet of a modest size gives) must not make it megabytes long. What is not saved is
/// still held in memory (up to `MAX_TRACKED`); it is only forgotten by a restart.
fn saved(m: &HashMap<String, Ladder>, now: u64, most: usize) -> HashMap<String, Ladder> {
    let mut live: Vec<(&String, &Ladder)> = m.iter().filter(|(_, l)| !l.is_quiet(now)).collect();
    if live.len() > most {
        live.select_nth_unstable_by(most, |(ka, a), (kb, b)| {
            let key = |l: &Ladder| {
                (
                    l.blocked_until.max(now),
                    l.level,
                    l.consecutive,
                    l.last_seen,
                )
            };
            key(b).cmp(&key(a)).then_with(|| ka.cmp(kb))
        });
        live.truncate(most);
    }
    live.into_iter()
        .map(|(k, l)| (k.clone(), l.clone()))
        .collect()
}

/// A ladder read from a file, with what cannot be true cut back; `None` when nothing of it is worth keeping.
fn sane(mut l: Ladder, now: u64) -> Option<Ladder> {
    let ceiling = now.saturating_add(block_ms(u32::MAX));
    l.blocked_until = l.blocked_until.min(ceiling);
    l.last_block_end = l.last_block_end.min(ceiling);
    l.last_seen = l.last_seen.min(now);
    l.level = l.level.min(21);
    l.consecutive = l.consecutive.min(FAILURES_TO_BLOCK - 1);
    (!l.is_quiet(now)).then_some(l)
}

fn slow_now(s: &State, now: u64) -> bool {
    let n = s.failures.len();
    n >= SLOW_FAILURES && now.saturating_sub(s.failures[n - SLOW_FAILURES]) < SLOW_WINDOW_MS
}

/// Keep `slow_since` true to the failures: set when slow mode turns on, cleared when it turns off.
fn refresh_slow(s: &mut State, now: u64) {
    if slow_now(s, now) {
        if s.slow_since.is_none() {
            s.slow_since = Some(now);
            s.dirty = true;
        }
    } else if s.slow_since.is_some() {
        s.slow_since = None;
        s.dirty = true;
    }
}

fn attack_now(s: &State, now: u64) -> Option<Attack> {
    if s.slow_since
        .is_some_and(|since| now.saturating_sub(since) >= ATTACK_SLOW_MS)
    {
        return Some(Attack::SlowModeSustained);
    }
    let in_a_day = s
        .failures
        .iter()
        .filter(|t| now.saturating_sub(**t) < ATTACK_WINDOW_MS)
        .count();
    (in_a_day >= ATTACK_FAILURES).then_some(Attack::ManyFailures)
}

/// Make room: drop the quiet address seen longest ago (never one that is blocked).
fn evict(map: &mut HashMap<String, Ladder>, now: u64) {
    let victim = map
        .iter()
        .filter(|(_, l)| l.blocked_until <= now)
        .min_by_key(|(_, l)| l.last_seen)
        .map(|(k, _)| k.clone());
    if let Some(k) = victim {
        map.remove(&k);
    } else if let Some(k) = map
        .iter()
        .min_by_key(|(_, l)| l.last_seen)
        .map(|(k, _)| k.clone())
    {
        map.remove(&k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::clock::ManualClock;

    const T0: u64 = 1_790_000_000_000;
    const MIN: u64 = 60_000;
    const HOUR: u64 = 60 * MIN;

    fn throttle() -> (LoginThrottle, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(T0));
        (LoginThrottle::new(clock.clone()), clock)
    }

    fn blocked(t: &LoginThrottle, key: &str) -> Option<u64> {
        match t.address_gate(key) {
            Gate::Blocked { retry_after_s } => Some(retry_after_s),
            Gate::Open => None,
        }
    }

    // T9: L1 -------------------------------------------------------------------------------------

    #[test]
    fn five_consecutive_failures_block_an_address_for_fifteen_minutes_and_not_four() {
        let (t, clock) = throttle();
        for i in 1..=4 {
            assert!(!t.address_failed("a").blocked, "failure {i}");
            assert_eq!(t.address_gate("a"), Gate::Open, "failure {i}");
        }
        let fifth = t.address_failed("a");
        assert!(fifth.blocked);
        assert_eq!(fifth.block_s, 15 * 60);
        assert_eq!(blocked(&t, "a"), Some(900));
        // Another address is not affected.
        assert_eq!(t.address_gate("b"), Gate::Open);
        // Seconds are counted down and rounded up.
        clock.advance(MIN + 1);
        assert_eq!(blocked(&t, "a"), Some(840), "839.999 seconds left");
        // The block ends at exactly fifteen minutes.
        clock.set(T0 + 15 * MIN - 1);
        assert!(blocked(&t, "a").is_some());
        clock.set(T0 + 15 * MIN);
        assert_eq!(t.address_gate("a"), Gate::Open);
    }

    #[test]
    fn a_success_clears_the_counters_and_the_ladder() {
        let (t, _clock) = throttle();
        for _ in 0..4 {
            t.address_failed("a");
        }
        t.address_succeeded("a");
        // Four more failures are still not a block: the four before the success were cleared.
        for _ in 0..4 {
            assert!(!t.address_failed("a").blocked);
        }
        assert_eq!(t.address_gate("a"), Gate::Open);
        // A success also clears a block that was running (the owner proved themself from that address).
        assert!(t.address_failed("a").blocked);
        assert!(blocked(&t, "a").is_some());
        t.address_succeeded("a");
        assert_eq!(t.address_gate("a"), Gate::Open);
    }

    #[test]
    fn each_new_block_within_a_day_doubles_up_to_twenty_four_hours() {
        let (t, clock) = throttle();
        let mut seen = Vec::new();
        for _ in 0..10 {
            for i in 0..5 {
                let f = t.address_failed("a");
                assert_eq!(f.blocked, i == 4);
                if f.blocked {
                    seen.push(f.block_s);
                }
            }
            // Wait the block out and try again at once: the next block is the next rung.
            let wait = blocked(&t, "a").unwrap();
            clock.advance(wait * 1000);
            assert_eq!(t.address_gate("a"), Gate::Open);
        }
        let minutes: Vec<u64> = seen.iter().map(|s| s / 60).collect();
        // 15 m, 30 m, 1 h, 2 h, 4 h, 8 h, 16 h, then capped at 24 h.
        assert_eq!(minutes, [15, 30, 60, 120, 240, 480, 960, 1440, 1440, 1440]);
    }

    #[test]
    fn a_quiet_day_starts_the_ladder_again_but_a_day_less_a_moment_does_not() {
        let (t, clock) = throttle();
        for _ in 0..5 {
            t.address_failed("a");
        }
        clock.advance(15 * MIN);
        // 24 hours after the block ended, less a millisecond: still the second rung.
        clock.advance(24 * HOUR - 1);
        for _ in 0..4 {
            t.address_failed("a");
        }
        assert_eq!(t.address_failed("a").block_s, 30 * 60);
        // Now a full quiet day after that block ended: the first rung again.
        clock.advance(30 * MIN);
        clock.advance(24 * HOUR);
        for _ in 0..4 {
            t.address_failed("a");
        }
        assert_eq!(t.address_failed("a").block_s, 15 * 60);
    }

    #[test]
    fn a_blocked_address_is_not_counted_again_because_the_caller_does_not_check_it() {
        // The gate is what stops a blocked request before anything is checked; the failures that a caller records
        // are only those of requests that were checked. The throttle itself never counts a gate.
        let (t, _clock) = throttle();
        for _ in 0..5 {
            t.address_failed("a");
        }
        let before = t.recent_failures();
        for _ in 0..50 {
            let _ = t.address_gate("a");
        }
        assert_eq!(t.recent_failures(), before);
    }

    // T9: L2 -------------------------------------------------------------------------------------

    #[test]
    fn twenty_failures_in_sixty_minutes_turn_slow_mode_on_and_nineteen_do_not() {
        let (t, clock) = throttle();
        for i in 0..19 {
            t.address_failed(&format!("addr-{i}"));
        }
        assert!(!t.is_slow(), "19 failures from 19 addresses");
        t.address_failed("addr-19");
        assert!(t.is_slow());
        // It stays on while 20 remain in the window, and goes off when the oldest of them is an hour old.
        clock.set(T0 + HOUR - 1);
        assert!(t.is_slow());
        clock.set(T0 + HOUR);
        assert!(!t.is_slow(), "fewer than 20 remain");
    }

    #[test]
    fn slow_mode_counts_failures_of_any_address_and_falls_as_the_window_slides() {
        let (t, clock) = throttle();
        // 30 failures, one every 3 minutes from 30 different addresses (the last at 87 minutes): the newest
        // 20 are all inside the last 60 minutes.
        for i in 0..30 {
            t.address_failed(&format!("addr-{i}"));
            if i < 29 {
                clock.advance(3 * MIN);
            }
        }
        assert!(t.is_slow(), "the last 20 fall in the last 60 minutes");
        // Ten more minutes with no failures: 17 of them (those after minute 37) are still inside 60 minutes,
        // and 17 is fewer than 20: slow mode is off.
        clock.advance(10 * MIN);
        assert!(!t.is_slow());
        assert_eq!(t.recent_failures(), 17);
    }

    #[test]
    fn a_failure_of_a_blocked_address_that_was_never_checked_does_not_exist_and_a_device_never_counts(
    ) {
        let (t, _clock) = throttle();
        for i in 0..100 {
            t.device_failed(&format!("dev-{}", i % 4));
        }
        assert!(!t.is_slow(), "the device lane is not in slow mode's count");
        assert_eq!(t.recent_failures(), 0);
        assert_eq!(t.tracked(), 0, "and it makes no address ladder");
    }

    // The device lane ----------------------------------------------------------------------------

    #[test]
    fn a_device_has_its_own_ladder_and_a_success_clears_it_and_it_never_touches_an_address() {
        let (t, clock) = throttle();
        for _ in 0..4 {
            assert!(!t.device_failed("d1").blocked);
        }
        assert!(t.device_failed("d1").blocked);
        assert!(matches!(
            t.device_gate("d1"),
            Gate::Blocked { retry_after_s: 900 }
        ));
        assert_eq!(t.device_gate("d2"), Gate::Open, "another device");
        // Doubling, as for an address.
        clock.advance(15 * MIN);
        for _ in 0..4 {
            t.device_failed("d1");
        }
        assert_eq!(t.device_failed("d1").block_s, 30 * 60);
        clock.advance(30 * MIN);
        t.device_succeeded("d1");
        for _ in 0..4 {
            assert!(!t.device_failed("d1").blocked, "cleared by the success");
        }
        // An address blocked at L1 is not cleared by a device success, and a device is not blocked by it.
        for _ in 0..5 {
            t.address_failed("a");
        }
        t.device_succeeded("d1");
        assert!(blocked(&t, "a").is_some());
        assert_eq!(t.device_gate("d3"), Gate::Open);
        t.forget_device("d1");
        assert_eq!(t.device_gate("d1"), Gate::Open);
    }

    #[test]
    fn a_device_guesses_at_most_thirty_five_times_a_day() {
        // 5 per block and 7 blocks in a day (15 m, 30 m, 1 h, 2 h, 4 h, 8 h, 16 h = 31 h30 of blocks would pass
        // the day; within 24 hours: 15 m + 30 m + 1 h + 2 h + 4 h + 8 h = 15 h 45 m, and the 16 h block starts).
        let (t, clock) = throttle();
        let mut guesses = 0;
        while clock.now_ms() < T0 + 24 * HOUR {
            match t.device_gate("d") {
                Gate::Open => {
                    guesses += 1;
                    t.device_failed("d");
                }
                Gate::Blocked { retry_after_s } => clock.advance(retry_after_s * 1000),
            }
        }
        assert!(guesses <= 35, "{guesses} guesses in a day");
        assert!(guesses >= 30, "and the bound is close: {guesses}");
    }

    // The alert ------------------------------------------------------------------------------------

    #[test]
    fn slow_mode_on_for_an_hour_without_a_break_is_an_attack_and_it_is_raised_once_an_hour() {
        let (t, clock) = throttle();
        let refill = |t: &LoginThrottle, n: usize| {
            for i in 0..n {
                t.address_failed(&format!("addr-{i}"));
            }
        };
        refill(&t, 20);
        assert!(t.is_slow());
        assert_eq!(t.under_attack(), None);
        // Keep slow mode on: 20 fresh failures every 30 minutes.
        for _ in 0..2 {
            clock.advance(30 * MIN);
            refill(&t, 20);
            assert!(t.is_slow());
        }
        // 60 minutes on: raised now, and only once until the next hour.
        assert_eq!(t.under_attack(), Some(Attack::SlowModeSustained));
        assert_eq!(t.take_alert(), Some(Attack::SlowModeSustained));
        assert_eq!(t.take_alert(), None);
        clock.advance(59 * MIN);
        refill(&t, 20);
        assert_eq!(t.take_alert(), None, "not within an hour of the last");
        clock.advance(2 * MIN);
        refill(&t, 20);
        assert_eq!(t.take_alert(), Some(Attack::SlowModeSustained));
    }

    #[test]
    fn a_break_in_slow_mode_restarts_the_hour() {
        let (t, clock) = throttle();
        for i in 0..20 {
            t.address_failed(&format!("addr-{i}"));
        }
        clock.advance(59 * MIN);
        assert!(t.is_slow());
        // It goes off (the window slides past all 20), then on again: the hour restarts.
        clock.advance(2 * MIN);
        assert!(!t.is_slow());
        for i in 0..20 {
            t.address_failed(&format!("again-{i}"));
        }
        clock.advance(59 * MIN);
        assert_eq!(t.under_attack(), None);
        for i in 0..20 {
            t.address_failed(&format!("more-{i}"));
        }
        clock.advance(2 * MIN);
        assert_eq!(t.under_attack(), Some(Attack::SlowModeSustained));
    }

    #[test]
    fn two_hundred_failures_in_twenty_four_hours_are_an_attack_even_without_slow_mode() {
        let (t, clock) = throttle();
        // 199 failures spread so that slow mode is never on (fewer than 20 in any hour: 8 an hour is 192...).
        let mut n = 0;
        while n < 199 {
            t.address_failed(&format!("addr-{n}"));
            n += 1;
            clock.advance(7 * MIN);
        }
        assert!(!t.is_slow());
        assert_eq!(t.under_attack(), None, "199");
        t.address_failed("addr-last");
        assert_eq!(t.under_attack(), Some(Attack::ManyFailures), "200");
        assert_eq!(t.take_alert(), Some(Attack::ManyFailures));
        // A day later they have aged out.
        clock.advance(25 * HOUR);
        assert_eq!(t.under_attack(), None);
    }

    // Saving ---------------------------------------------------------------------------------------

    #[test]
    fn the_state_is_saved_and_loaded_so_a_restart_clears_nothing() {
        let (t, clock) = throttle();
        for _ in 0..5 {
            t.address_failed("blocked-addr");
        }
        for i in 0..20 {
            t.address_failed(&format!("addr-{i}"));
        }
        for _ in 0..5 {
            t.device_failed("blocked-dev");
        }
        // A ladder rung in progress: 3 failures.
        for _ in 0..3 {
            t.address_failed("partial");
        }
        let saved = t.snapshot();
        // The file is JSON text: through the text, as the disk does it.
        let text = serde_json::to_string(&saved).unwrap();
        let reloaded: Value = serde_json::from_str(&text).unwrap();

        clock.advance(2 * MIN);
        let after = LoginThrottle::new(clock.clone());
        after.restore(&reloaded);
        assert!(
            blocked(&after, "blocked-addr").is_some(),
            "the block survived the restart"
        );
        assert!(matches!(
            after.device_gate("blocked-dev"),
            Gate::Blocked { .. }
        ));
        assert!(
            after.is_slow(),
            "and slow mode: 25 failures are in the window"
        );
        assert_eq!(after.recent_failures(), 28);
        // The rung in progress: two more failures make five.
        assert!(!after.address_failed("partial").blocked);
        assert!(after.address_failed("partial").blocked);
        // And the ladder level: a second block of the first address is the second rung.
        clock.advance(15 * MIN);
        for _ in 0..4 {
            after.address_failed("blocked-addr");
        }
        assert_eq!(after.address_failed("blocked-addr").block_s, 30 * 60);
    }

    #[test]
    fn a_damaged_or_foreign_saved_state_is_ignored() {
        let (t, _clock) = throttle();
        for bad in [
            Value::Null,
            json!("text"),
            json!(7),
            json!([1, 2]),
            json!({ "addresses": "no", "devices": [1], "failures": "x" }),
            json!({ "addresses": { "a": "no" }, "devices": { "d": [1] } }),
        ] {
            t.restore(&bad);
            assert_eq!(t.tracked(), 0, "{bad}");
            assert!(!t.is_slow());
        }
    }

    #[test]
    fn a_file_is_not_trusted_to_name_a_block_of_a_thousand_years_or_a_ladder_of_a_billion_failures()
    {
        let (t, clock) = throttle();
        let year_3000 = 32_503_680_000_000u64;
        t.restore(&json!({
            "addresses": {
                "a": { "consecutive": u32::MAX, "blocked_until": year_3000, "level": u32::MAX,
                       "last_block_end": year_3000, "last_seen": year_3000 },
                "c": { "consecutive": 4, "blocked_until": 0, "level": 21,
                       "last_block_end": year_3000, "last_seen": 1 },
                "quiet": { "consecutive": 0, "blocked_until": 0, "level": 3, "last_block_end": 0, "last_seen": 1 }
            },
            "devices": { "d": { "consecutive": 0, "blocked_until": year_3000, "level": 2,
                                "last_block_end": year_3000, "last_seen": year_3000 } },
        }));
        // The block is a day at the most, and the address that had nothing to remember is not kept.
        assert_eq!(t.tracked(), 2);
        match t.address_gate("a") {
            Gate::Blocked { retry_after_s } => {
                assert!(retry_after_s <= 24 * 3600, "{retry_after_s}")
            }
            Gate::Open => panic!("the block is real, but a day at the most"),
        }
        match t.device_gate("d") {
            Gate::Blocked { retry_after_s } => {
                assert!(retry_after_s <= 24 * 3600, "{retry_after_s}")
            }
            Gate::Open => panic!("the device's block is real too"),
        }
        // After that day the address may try again, and the block it earns next is a day at the most.
        clock.advance(25 * HOUR);
        assert_eq!(t.address_gate("a"), Gate::Open);
        let again = t.address_failed("a");
        assert!(again.block_s <= 24 * 3600, "{again:?}");
        // A ladder that a file said ended in the year 3000 ended, at the latest, a day from now: a day after that
        // the next block is the first rung, not the top of a ladder the file made up.
        clock.advance(25 * HOUR);
        let first = t.address_failed("c");
        assert_eq!(first.block_s, 15 * 60, "{first:?}");
    }

    #[test]
    fn a_failure_time_in_the_future_after_a_clock_that_went_back_is_now() {
        let (t, clock) = throttle();
        let future: Vec<u64> = (0..25).map(|i| T0 + 10 * HOUR + i).collect();
        t.restore(&json!({ "v": 1, "failures": future }));
        assert_eq!(t.recent_failures(), 25);
        assert!(t.is_slow());
        // They age from now, not from the future.
        clock.advance(HOUR);
        assert!(!t.is_slow());
    }

    #[test]
    fn saving_is_due_five_seconds_after_the_last_and_only_when_something_changed() {
        let (t, clock) = throttle();
        assert!(!t.flush_due(), "nothing changed");
        t.address_failed("a");
        clock.advance(FLUSH_EVERY_MS - 1);
        // The first write is due at once: nothing was ever written.
        assert!(t.flush_due());
        let _ = t.snapshot();
        assert!(!t.flush_due());
        t.address_failed("a");
        assert!(!t.flush_due(), "changed, but written a moment ago");
        clock.advance(FLUSH_EVERY_MS);
        assert!(t.flush_due());
    }

    // Bounds ---------------------------------------------------------------------------------------

    #[test]
    fn the_failure_window_keeps_the_newest_256_and_the_addresses_are_bounded() {
        let (t, _clock) = throttle();
        for i in 0..1000 {
            t.address_failed(&format!("addr-{i}"));
        }
        assert_eq!(t.recent_failures(), KEEP_FAILURES);
        assert!(t.is_slow());
        assert_eq!(t.tracked(), 1000);
    }

    #[test]
    fn what_is_saved_is_bounded_and_keeps_the_blocks_that_end_last() {
        let (t, clock) = throttle();
        let fail_five = |key: &str| {
            for _ in 0..FAILURES_TO_BLOCK {
                t.address_failed(key);
            }
        };
        // More blocked addresses than the file holds, all blocked for fifteen minutes...
        for i in 0..MAX_SAVED + 500 {
            fail_five(&format!("early-{i:05}"));
        }
        // ...and a few that were blocked a little later: their blocks end last, so they are the ones kept.
        clock.advance(MIN);
        for i in 0..10 {
            fail_five(&format!("late-{i}"));
        }
        assert_eq!(t.tracked(), MAX_SAVED + 510, "memory holds them all");
        let saved = t.snapshot();
        let addresses = saved["addresses"].as_object().unwrap();
        assert_eq!(addresses.len(), MAX_SAVED);
        for i in 0..10 {
            assert!(addresses.contains_key(&format!("late-{i}")), "late-{i}");
        }
        assert!(
            saved.to_string().len() < 400_000,
            "{} bytes for {MAX_SAVED} addresses",
            saved.to_string().len()
        );
        // What was kept is what a restart brings back, and a block is a block.
        let (again, _) = throttle();
        again.restore(&saved);
        assert_eq!(again.tracked(), MAX_SAVED);
        assert!(blocked(&again, "late-3").is_some());
    }

    #[test]
    fn of_addresses_that_matter_equally_the_same_ones_are_kept_whatever_the_order_of_the_table() {
        // Two tables that went through the same failures hold them in an order of their own (a hash map's), and
        // what is saved does not depend on it: the same state is the same file.
        let file = || {
            let (t, _clock) = throttle();
            for i in 0..MAX_SAVED + 500 {
                for _ in 0..FAILURES_TO_BLOCK {
                    t.address_failed(&format!("addr-{i:05}"));
                }
            }
            t.snapshot()["addresses"].clone()
        };
        let first = file();
        assert_eq!(first.as_object().unwrap().len(), MAX_SAVED);
        assert_eq!(first, file());
    }

    #[test]
    fn a_block_that_is_running_is_kept_before_a_ladder_that_only_remembers_blocks_that_ended() {
        let (t, clock) = throttle();
        let fail_five = |key: &str| {
            for _ in 0..FAILURES_TO_BLOCK {
                t.address_failed(key);
            }
        };
        // Addresses on the third rung whose blocks are over (a day is not up, so the ladder is remembered)...
        for wait in [16 * MIN, 31 * MIN, 61 * MIN] {
            for i in 0..MAX_SAVED + 500 {
                fail_five(&format!("over-{i:05}"));
            }
            clock.advance(wait);
        }
        // ...and a few on the first whose block is running now: it is those the file is for.
        for i in 0..10 {
            fail_five(&format!("now-{i}"));
        }
        assert!(blocked(&t, "now-0").is_some());
        assert!(blocked(&t, "over-00000").is_none());
        let saved = t.snapshot();
        let addresses = saved["addresses"].as_object().unwrap();
        assert_eq!(addresses.len(), MAX_SAVED);
        for i in 0..10 {
            assert!(addresses.contains_key(&format!("now-{i}")), "now-{i}");
        }
    }

    #[test]
    fn a_ladder_with_nothing_left_to_remember_is_not_saved() {
        let (t, _clock) = throttle();
        for _ in 0..FAILURES_TO_BLOCK {
            t.address_failed("blocked");
        }
        for key in ["fine-1", "fine-2"] {
            t.address_failed(key);
            t.address_succeeded(key);
        }
        t.address_failed("one-failure");
        let saved = t.snapshot();
        let mut names: Vec<&String> = saved["addresses"].as_object().unwrap().keys().collect();
        names.sort();
        assert_eq!(names, ["blocked", "one-failure"]);
    }

    #[test]
    fn a_table_that_fits_the_bound_is_saved_whole() {
        let (t, _clock) = throttle();
        for i in 0..MAX_SAVED {
            for _ in 0..FAILURES_TO_BLOCK {
                t.address_failed(&format!("addr-{i}"));
            }
        }
        assert_eq!(
            t.snapshot()["addresses"].as_object().unwrap().len(),
            MAX_SAVED
        );
    }

    #[test]
    fn a_full_table_makes_room_from_the_quiet_and_never_from_a_block() {
        let clock = Arc::new(ManualClock::new(T0));
        let mut map: HashMap<String, Ladder> = HashMap::new();
        let blocked_one = Ladder {
            blocked_until: T0 + HOUR,
            last_seen: T0 - 10 * HOUR,
            ..Ladder::default()
        };
        map.insert("blocked".into(), blocked_one);
        for i in 0..5u64 {
            let l = Ladder {
                last_seen: T0 - i * 1000,
                consecutive: 1,
                ..Ladder::default()
            };
            map.insert(format!("quiet-{i}"), l);
        }
        evict(&mut map, clock.now_ms());
        assert!(
            map.contains_key("blocked"),
            "the block was the least recently seen and is kept"
        );
        assert!(
            !map.contains_key("quiet-4"),
            "the quiet one seen longest ago went"
        );
        assert_eq!(map.len(), 5);
        // When everything is blocked one goes anyway: memory is bounded before anything.
        let mut all: HashMap<String, Ladder> = (0..3)
            .map(|i| {
                (
                    format!("b{i}"),
                    Ladder {
                        blocked_until: T0 + HOUR,
                        last_seen: T0 - i,
                        ..Ladder::default()
                    },
                )
            })
            .collect();
        evict(&mut all, T0);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn seen_from_outside_a_block_of_two_addresses_is_listed_with_its_seconds() {
        let (t, _clock) = throttle();
        for _ in 0..5 {
            t.address_failed("b");
            t.address_failed("a");
        }
        t.address_failed("c");
        assert_eq!(
            t.blocked_addresses(),
            vec![("a".to_string(), 900), ("b".to_string(), 900)]
        );
        assert!(t.blocked_devices().is_empty());
    }
}
