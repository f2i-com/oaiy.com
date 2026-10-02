//! Time and randomness behind traits, and the clock offset (README sections 1 and 9.4).
//!
//! **Two clocks, and what each is for.** The *relay clock* is the relay's `X-OAIY-Time`: a client keeps an offset `time - localNowAtReceipt`, sampled when the response is
//! received (a held poll's time is its end), takes the median of the last five samples and slews the offset it applies by at most one second a minute. It is used for what the
//! relay judges (pairing windows, a ring's `expiresAt` on a phone) and for warning the owner of a difference above 60 seconds, and **never** for a signed command or ticket.
//! The *provider clock* is the same estimator fed by the provider's own TLS responses (the `Date` header or a `serverTime` member); an offset older than 150 seconds is
//! `clock_unknown` and one above 60 seconds is `clock_mismatch`, and in either case signed items are refused (9.4). Both are [`OffsetClock`]; the provider's adds a verdict.

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::http::Cancel;

/// The wall clock, a monotonic clock and a sleep, as the platform has them. A test supplies a fake one and the loop's pauses cost no time.
///
/// **The monotonic clock must keep counting while the device is suspended.** The age of an identity proof, the relay's time, the schedule of proofs and the replacement gap of P1 are
/// all measured with it, so a clock that stands still in sleep makes a proof of eight hours ago look a minute old. On Android that is `SystemClock.elapsedRealtime()` (not
/// `uptimeMillis()`, and not `nanoTime()` on a platform where that stops in sleep); on Linux `CLOCK_BOOTTIME` (not `CLOCK_MONOTONIC`); on macOS `mach_continuous_time` (not
/// `mach_absolute_time`); on Windows a counter that includes sleep (`QueryPerformanceCounter` does; `QueryUnbiasedInterruptTime` does not). [`SystemClock`] uses
/// `std::time::Instant`, which is the platform's own counter and is **not** one of these on Linux, Android and macOS: a host there supplies its own `Clock`. As a backstop the
/// client also ages the proof by the wall clock (see [`crate::client::RelayClient`]): a wall clock that has moved forward by more than the proof's life, or back by more than a
/// minute, makes the proof stale whatever the monotonic clock says.
pub trait Clock: Send + Sync {
    /// Seconds since 1970-01-01, which may step (the owner sets the PC's clock).
    fn unix_now(&self) -> i64;
    /// Time since an arbitrary start, which never goes backwards and **does count the time the device spends suspended** (see the trait): what intervals are measured with.
    fn monotonic(&self) -> Duration;
    /// Sleeps for `duration`, or until `cancel` is set. Returns true when it slept the whole time, false when it was cancelled.
    fn sleep(&self, duration: Duration, cancel: &Cancel) -> bool;
}

/// The operating system's clocks: the wall clock, and `std::time::Instant` as the monotonic one. `Instant` includes the time the machine spends suspended on Windows, and on Linux,
/// Android and macOS it does not (the platform's own documentation, not something this crate measured by suspending a machine): there the host passes a [`Clock`] of its own, as
/// the trait says.
#[derive(Debug)]
pub struct SystemClock {
    start: Instant,
}

impl SystemClock {
    /// A clock whose monotonic time starts now.
    pub fn new() -> SystemClock {
        SystemClock { start: Instant::now() }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        SystemClock::new()
    }
}

impl Clock for SystemClock {
    fn unix_now(&self) -> i64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
    }

    fn monotonic(&self) -> Duration {
        self.start.elapsed()
    }

    fn sleep(&self, duration: Duration, cancel: &Cancel) -> bool {
        // Sleep in slices of 50 ms so that a cancel is noticed within a slice, without a thread per sleep.
        let end = Instant::now() + duration;
        loop {
            if cancel.is_cancelled() {
                return false;
            }
            let left = end.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return true;
            }
            std::thread::sleep(left.min(Duration::from_millis(50)));
        }
    }
}

/// Fills `buf` from the operating system's random generator (through `oaiy-crypto`, the one place this workspace asks for it). A failure of the generator is a panic: no
/// protocol value may be made from weaker randomness, and it does not fail on any platform this ships on.
pub(crate) fn os_fill(buf: &mut [u8]) {
    for chunk in buf.chunks_mut(32) {
        let random = oaiy_crypto::zeroize::Secret::<32>::random().expect("the operating system's random generator");
        chunk.copy_from_slice(&random.expose()[..chunk.len()]);
    }
}

/// A source of the jitter draw `u` of the pauses (README P6), and of nothing else. **No secret and no protocol value is drawn from it**: the pairing secret, the offer's nonce
/// and `jti`, the proof nonces, the identity keys and the device ids all come from the operating system's generator inside this crate (`oaiy-crypto`), whatever a host passes
/// here, so that a predictable generator (a seeded one in a test, a poor one on a platform) can only make the pauses predictable.
pub trait Rng: Send {
    /// Fills `buf`.
    fn fill(&mut self, buf: &mut [u8]);

    /// A number from 0 up to but not including 1 (53 random bits).
    fn unit(&mut self) -> f64 {
        let mut b = [0u8; 8];
        self.fill(&mut b);
        (u64::from_le_bytes(b) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The operating system's random generator ([`os_fill`]): what a product passes to [`crate::client::RelayClient::new`] for its jitter.
#[derive(Debug, Default)]
pub struct OsRng;

impl Rng for OsRng {
    fn fill(&mut self, buf: &mut [u8]) {
        os_fill(buf);
    }
}

/// How many samples the median is taken over (README section 1).
pub const SAMPLES: usize = 5;
/// The most the applied offset moves in a minute, in seconds.
pub const MAX_SLEW_PER_MINUTE_S: f64 = 1.0;
/// A difference above this many seconds is warned about (relay clock) or refuses signed items (provider clock).
pub const MISMATCH_S: i64 = 60;
/// The provider offset is `clock_unknown` when its newest sample is older than this many seconds.
pub const STALE_S: u64 = 150;
/// A sample that is more than this many seconds (a day) from the local wall clock is not believed: a relay (or a provider) cannot move the time this crate judges windows by to
/// another day by saying so in its first answer. It is counted as a mismatch (the owner is told the clocks differ), and the local clock is used until a sample is plausible.
pub const MAX_SKEW_S: i64 = 86_400;

/// The offset of a remote clock, estimated and kept **on the monotonic clock**: each sample is the remote time minus the monotonic time at which it arrived, the median of the last
/// five is the estimate, and the value applied moves towards it by at most one second a minute. The remote time now is then the monotonic time plus that value, and does not
/// depend on the local wall clock at all: when the owner (or NTP) steps the PC's clock, the relay's time stays where it was, and a mismatch with the wall clock shows as a
/// mismatch. (The wall clock is read for two things only: to refuse a sample that is not plausible, and to say how far the remote time is from it.)
#[derive(Debug, Clone)]
pub struct OffsetClock {
    /// `remote_time - monotonic seconds` of each of the last [`SAMPLES`] samples.
    samples: VecDeque<f64>,
    applied: f64,
    applied_at: Duration,
    last_sample_at: Option<Duration>,
    /// The offset of the newest sample that was not believed (more than [`MAX_SKEW_S`] from the wall clock), until a plausible one arrives.
    implausible: Option<i64>,
}

impl Default for OffsetClock {
    fn default() -> Self {
        OffsetClock::new()
    }
}

impl OffsetClock {
    /// A clock with no sample.
    pub fn new() -> OffsetClock {
        OffsetClock { samples: VecDeque::with_capacity(SAMPLES), applied: 0.0, applied_at: Duration::ZERO, last_sample_at: None, implausible: None }
    }

    /// Adds the sample `remote_time` that arrived at monotonic time `mono`, when the local wall clock read `local_unix_at_receipt`. The first sample is applied at once; later
    /// ones move the applied value towards the median by at most one second a minute of monotonic time. A sample more than [`MAX_SKEW_S`] from the wall clock is not added.
    pub fn observe(&mut self, remote_time: i64, local_unix_at_receipt: i64, mono: Duration) {
        let skew = remote_time.saturating_sub(local_unix_at_receipt);
        if skew.abs() > MAX_SKEW_S {
            self.implausible = Some(skew);
            return;
        }
        self.implausible = None;
        if self.samples.len() == SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(remote_time as f64 - mono.as_secs_f64());
        let target = self.median();
        if self.last_sample_at.is_none() {
            self.applied = target;
        } else {
            self.advance(mono, target);
        }
        self.applied_at = mono;
        self.last_sample_at = Some(mono);
    }

    fn median(&self) -> f64 {
        let mut v: Vec<f64> = self.samples.iter().copied().collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    }

    fn advance(&mut self, mono: Duration, target: f64) {
        let minutes = mono.saturating_sub(self.applied_at).as_secs_f64() / 60.0;
        let step = (target - self.applied).clamp(-MAX_SLEW_PER_MINUTE_S * minutes, MAX_SLEW_PER_MINUTE_S * minutes);
        self.applied += step;
        self.applied_at = mono;
    }

    /// The value applied at monotonic time `mono`: the remote time is the monotonic time in seconds plus this. `None` without a believed sample.
    fn base(&self, mono: Duration) -> Option<f64> {
        self.last_sample_at?;
        let minutes = mono.saturating_sub(self.applied_at).as_secs_f64() / 60.0;
        let step = (self.median() - self.applied).clamp(-MAX_SLEW_PER_MINUTE_S * minutes, MAX_SLEW_PER_MINUTE_S * minutes);
        Some(self.applied + step)
    }

    /// The remote clock's `now`, whole seconds, from the monotonic time alone (a step of the local wall clock does not move it): `None` without a believed sample.
    pub fn remote_now(&self, mono: Duration) -> Option<i64> {
        self.base(mono).map(|b| (mono.as_secs_f64() + b).round() as i64)
    }

    /// The remote clock minus the local wall clock `local_unix`, in seconds, at monotonic time `mono`: what a warning to the owner reports. A sample that was not believed reports
    /// its own difference.
    pub fn offset(&self, local_unix: i64, mono: Duration) -> Option<f64> {
        match self.base(mono) {
            Some(b) if self.implausible.is_none() => Some(mono.as_secs_f64() + b - local_unix as f64),
            _ => self.implausible.map(|s| s as f64),
        }
    }

    /// True when the remote clock differs from the local wall clock by more than 60 seconds either way (the relay clock's warning; the provider clock's `clock_mismatch`).
    pub fn mismatch(&self, local_unix: i64, mono: Duration) -> bool {
        self.offset(local_unix, mono).is_some_and(|o| o.abs().round() as i64 > MISMATCH_S)
    }

    /// Seconds since the newest sample, or `None` without one.
    pub fn age_s(&self, mono: Duration) -> Option<u64> {
        self.last_sample_at.map(|t| mono.saturating_sub(t).as_secs())
    }

    /// The provider clock's verdict (README 9.4): no sample, or one older than 150 seconds, is `clock_unknown`; an offset above 60 seconds is `clock_mismatch`; otherwise
    /// the provider's time is trustworthy to judge a signed command or ticket.
    pub fn provider_verdict(&self, local_unix: i64, mono: Duration) -> ProviderClock {
        if self.implausible.is_some() {
            return ProviderClock::Mismatch;
        }
        match self.age_s(mono) {
            None => ProviderClock::Unknown,
            Some(age) if age > STALE_S => ProviderClock::Unknown,
            Some(_) if self.mismatch(local_unix, mono) => ProviderClock::Mismatch,
            Some(_) => ProviderClock::Good,
        }
    }
}

/// The verdict on the provider clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderClock {
    /// Fresh and within a minute: signed items may be judged by it.
    Good,
    /// No sample, or none for 150 seconds: signed items are refused and nothing signed is pruned (`clock_unknown`).
    Unknown,
    /// More than a minute off: signed items are refused and nothing signed is pruned (`clock_mismatch`).
    Mismatch,
}
#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn the_first_sample_is_applied_and_the_median_of_five_resists_one_outlier() {
        let mut c = OffsetClock::new();
        assert_eq!(c.remote_now(s(0)), None);
        c.observe(1000, 990, s(0));
        assert_eq!(c.remote_now(s(0)), Some(1000));
        assert_eq!(c.offset(990, s(0)), Some(10.0));
        // One wild sample among good ones does not move the median: the wall clock and the monotonic clock run together, the relay is 10 s ahead, and once it says 500 s.
        for i in 1..=4u64 {
            let wall = 990 + i as i64;
            c.observe(wall + if i == 2 { 500 } else { 10 }, wall, s(i));
        }
        assert_eq!(c.median(), 1000.0);
        assert!((c.offset(994, s(4)).unwrap() - 10.0).abs() < 1.0);
    }

    #[test]
    fn the_median_is_over_the_last_five_samples_and_two_wild_ones_do_not_move_it() {
        let mut c = OffsetClock::new();
        // Offsets 10, 10, 500, 500, 10: the median of five is 10 (the median of the last three would be 500).
        for (i, off) in [10i64, 10, 500, 500, 10].iter().enumerate() {
            c.observe(1000 + i as i64 + off, 1000 + i as i64, s(i as u64));
        }
        assert_eq!(c.median(), 1010.0);
        // A sixth sample pushes the oldest out: 10, 500, 500, 10, 500.
        c.observe(1005 + 500, 1005, s(5));
        assert_eq!(c.median(), 1500.0);
    }

    #[test]
    fn the_applied_offset_slews_by_at_most_a_second_a_minute() {
        let mut c = OffsetClock::new();
        c.observe(1000, 1000, s(0));
        // The relay's clock jumps by 100 seconds and stays there: every sample now says so (the wall clock keeps pace with the monotonic one).
        for i in 1..=5u64 {
            c.observe(1100 + i as i64, 1000 + i as i64, s(i));
        }
        assert_eq!(c.median(), 1100.0);
        let at = |t: u64| c.offset(1000 + t as i64, s(t)).unwrap();
        assert!(at(5) < 1.0, "{}", at(5));
        assert!((0.9..=1.2).contains(&at(65)), "{}", at(65));
        assert!((9.0..=11.0).contains(&at(5 + 600)), "{}", at(5 + 600));
        // and it only ever moves towards the median
        assert!(at(5 + 100_000) <= 100.0);
    }

    #[test]
    fn a_step_of_the_wall_clock_does_not_move_the_remote_time() {
        let mut c = OffsetClock::new();
        c.observe(1000, 1000, s(0));
        assert_eq!(c.remote_now(s(100)), Some(1100));
        // The owner (or NTP) sets the PC's clock forward an hour: the remote time stays where the monotonic clock puts it, and the difference shows as a mismatch.
        assert!(!c.mismatch(1100, s(100)));
        assert!(c.mismatch(1100 + 3600, s(100)));
        assert_eq!(c.remote_now(s(100)), Some(1100));
        // Samples taken after the step (the wall clock is an hour ahead of the remote one now) do not move it either.
        for i in 101..=110u64 {
            c.observe(1000 + i as i64, 1000 + 3600 + i as i64, s(i));
        }
        assert_eq!(c.remote_now(s(110)), Some(1110));
        assert!(c.mismatch(1110 + 3600, s(110)));
    }

    #[test]
    fn a_sample_a_day_or_more_from_the_wall_clock_is_not_believed() {
        let mut c = OffsetClock::new();
        // The relay says it is ten years later: not applied, and the owner is told the clocks differ.
        c.observe(1000 + 10 * 365 * 86_400, 1000, s(0));
        assert_eq!(c.remote_now(s(0)), None);
        assert!(c.mismatch(1000, s(0)));
        assert_eq!(c.provider_verdict(1000, s(0)), ProviderClock::Mismatch);
        // Exactly a day is believed (and is a mismatch, as any difference above a minute is); one second more is not.
        let mut edge = OffsetClock::new();
        edge.observe(1000 + MAX_SKEW_S, 1000, s(0));
        assert!(edge.remote_now(s(0)).is_some() && edge.mismatch(1000, s(0)));
        let mut over = OffsetClock::new();
        over.observe(1000 + MAX_SKEW_S + 1, 1000, s(0));
        assert!(over.remote_now(s(0)).is_none());
        // A plausible sample after it clears it.
        c.observe(1005, 1000, s(1));
        assert_eq!(c.remote_now(s(1)), Some(1005));
        assert!(!c.mismatch(1001, s(1)));
        // And a lying sample after good ones changes nothing but the warning.
        c.observe(1000 + 10 * 365 * 86_400, 1002, s(2));
        assert_eq!(c.remote_now(s(2)), Some(1006));
        assert!(c.mismatch(1002, s(2)));
    }

    #[test]
    fn the_verdicts() {
        let mut c = OffsetClock::new();
        assert_eq!(c.provider_verdict(0, s(0)), ProviderClock::Unknown);
        c.observe(1000, 1000, s(10));
        let at = |c: &OffsetClock, t: u64| c.provider_verdict(1000 + t as i64 - 10, s(t));
        assert_eq!(at(&c, 10), ProviderClock::Good);
        assert_eq!(at(&c, 10 + 150), ProviderClock::Good);
        assert_eq!(at(&c, 10 + 151), ProviderClock::Unknown, "older than 150 seconds");
        let mut off = OffsetClock::new();
        off.observe(1061, 1000, s(0));
        assert_eq!(off.provider_verdict(1001, s(1)), ProviderClock::Mismatch, "61 seconds");
        let mut near = OffsetClock::new();
        near.observe(1060, 1000, s(0));
        assert_eq!(near.provider_verdict(1001, s(1)), ProviderClock::Good, "60 seconds is not above 60");
        assert!(near.remote_now(s(1)).is_some());
    }

    #[test]
    fn the_os_random_source_fills_any_length_and_unit_is_below_one() {
        let mut r = OsRng;
        let mut a = [0u8; 100];
        r.fill(&mut a);
        assert!(a.iter().any(|b| *b != 0));
        for _ in 0..1000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
