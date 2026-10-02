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
pub trait Clock: Send + Sync {
    /// Seconds since 1970-01-01, which may step (the owner sets the PC's clock).
    fn unix_now(&self) -> i64;
    /// Time since an arbitrary start, which never goes backwards: what intervals are measured with.
    fn monotonic(&self) -> Duration;
    /// Sleeps for `duration`, or until `cancel` is set. Returns true when it slept the whole time, false when it was cancelled.
    fn sleep(&self, duration: Duration, cancel: &Cancel) -> bool;
}

/// The operating system's clocks.
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

/// The offset of a remote clock from the local one: median of the last five samples, applied with a slew of at most one second a minute, and kept on the monotonic clock so
/// that a step of the PC's wall clock does not move it.
#[derive(Debug, Clone)]
pub struct OffsetClock {
    samples: VecDeque<i64>,
    applied: f64,
    applied_at: Duration,
    last_sample_at: Option<Duration>,
}

impl Default for OffsetClock {
    fn default() -> Self {
        OffsetClock::new()
    }
}

impl OffsetClock {
    /// A clock with no sample.
    pub fn new() -> OffsetClock {
        OffsetClock { samples: VecDeque::with_capacity(SAMPLES), applied: 0.0, applied_at: Duration::ZERO, last_sample_at: None }
    }

    /// Adds the sample `remote_time - local_unix_at_receipt` taken at monotonic time `mono`. The first sample is applied at once; later ones move the applied offset towards
    /// the median by at most one second a minute of monotonic time.
    pub fn observe(&mut self, remote_time: i64, local_unix_at_receipt: i64, mono: Duration) {
        let sample = remote_time - local_unix_at_receipt;
        if self.samples.len() == SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        let target = self.median() as f64;
        if self.last_sample_at.is_none() {
            self.applied = target;
        } else {
            self.advance(mono, target);
        }
        self.applied_at = mono;
        self.last_sample_at = Some(mono);
    }

    fn median(&self) -> i64 {
        let mut v: Vec<i64> = self.samples.iter().copied().collect();
        v.sort_unstable();
        v[v.len() / 2]
    }

    fn advance(&mut self, mono: Duration, target: f64) {
        let minutes = mono.saturating_sub(self.applied_at).as_secs_f64() / 60.0;
        let step = (target - self.applied).clamp(-MAX_SLEW_PER_MINUTE_S * minutes, MAX_SLEW_PER_MINUTE_S * minutes);
        self.applied += step;
        self.applied_at = mono;
    }

    /// The offset applied at monotonic time `mono`, in seconds, or `None` without a sample.
    pub fn offset(&self, mono: Duration) -> Option<f64> {
        self.last_sample_at?;
        let target = self.median() as f64;
        let minutes = mono.saturating_sub(self.applied_at).as_secs_f64() / 60.0;
        let step = (target - self.applied).clamp(-MAX_SLEW_PER_MINUTE_S * minutes, MAX_SLEW_PER_MINUTE_S * minutes);
        Some(self.applied + step)
    }

    /// The remote clock's `now`, whole seconds, from the local wall clock and the monotonic time: `None` without a sample.
    pub fn remote_now(&self, local_unix: i64, mono: Duration) -> Option<i64> {
        self.offset(mono).map(|o| local_unix + o.round() as i64)
    }

    /// True when the offset is above 60 seconds either way (the relay clock's warning; the provider clock's `clock_mismatch`).
    pub fn mismatch(&self, mono: Duration) -> bool {
        self.offset(mono).is_some_and(|o| o.abs().round() as i64 > MISMATCH_S)
    }

    /// Seconds since the newest sample, or `None` without one.
    pub fn age_s(&self, mono: Duration) -> Option<u64> {
        self.last_sample_at.map(|t| mono.saturating_sub(t).as_secs())
    }

    /// The provider clock's verdict (README 9.4): no sample, or one older than 150 seconds, is `clock_unknown`; an offset above 60 seconds is `clock_mismatch`; otherwise
    /// the provider's time is trustworthy to judge a signed command or ticket.
    pub fn provider_verdict(&self, mono: Duration) -> ProviderClock {
        match self.age_s(mono) {
            None => ProviderClock::Unknown,
            Some(age) if age > STALE_S => ProviderClock::Unknown,
            Some(_) if self.mismatch(mono) => ProviderClock::Mismatch,
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
        assert_eq!(c.offset(s(0)), None);
        c.observe(1000, 990, s(0));
        assert_eq!(c.offset(s(0)), Some(10.0));
        // One wild sample among good ones does not move the median.
        for (i, local) in [1001, 1002, 1003, 1004].iter().enumerate() {
            c.observe(*local + if i == 1 { 500 } else { 10 }, *local, s(i as u64 + 1));
        }
        assert_eq!(c.median(), 10);
        assert!(c.offset(s(5)).unwrap() - 10.0 < 1.0);
    }

    #[test]
    fn the_median_is_over_the_last_five_samples_and_two_wild_ones_do_not_move_it() {
        let mut c = OffsetClock::new();
        // Offsets 10, 10, 500, 500, 10: the median of five is 10 (the median of the last three would be 500).
        for (i, off) in [10i64, 10, 500, 500, 10].iter().enumerate() {
            c.observe(1000 + i as i64 + off, 1000 + i as i64, s(i as u64));
        }
        assert_eq!(c.median(), 10);
        // A sixth sample pushes the oldest out: 10, 500, 500, 10, 500.
        c.observe(1005 + 500, 1005, s(5));
        assert_eq!(c.median(), 500);
    }
    #[test]
    fn the_applied_offset_slews_by_at_most_a_second_a_minute() {
        let mut c = OffsetClock::new();
        c.observe(1000, 1000, s(0));
        // The relay's clock jumps by 100 seconds and stays there: every sample now says so.
        for i in 1..=5u64 {
            c.observe(1100 + i as i64, 1000 + i as i64, s(i));
        }
        assert_eq!(c.median(), 100);
        let after_5_s = c.offset(s(5)).unwrap();
        assert!(after_5_s < 1.0, "{after_5_s}");
        let after_a_minute = c.offset(s(65)).unwrap();
        assert!((0.9..=1.2).contains(&after_a_minute), "{after_a_minute}");
        let after_ten_minutes = c.offset(s(5 + 600)).unwrap();
        assert!((9.0..=11.0).contains(&after_ten_minutes), "{after_ten_minutes}");
        // and it only ever moves towards the median
        assert!(c.offset(s(5 + 100_000)).unwrap() <= 100.0);
    }

    #[test]
    fn the_verdicts() {
        let mut c = OffsetClock::new();
        assert_eq!(c.provider_verdict(s(0)), ProviderClock::Unknown);
        c.observe(1000, 1000, s(10));
        assert_eq!(c.provider_verdict(s(10)), ProviderClock::Good);
        assert_eq!(c.provider_verdict(s(10 + 150)), ProviderClock::Good);
        assert_eq!(c.provider_verdict(s(10 + 151)), ProviderClock::Unknown, "older than 150 seconds");
        let mut off = OffsetClock::new();
        off.observe(1061, 1000, s(0));
        assert_eq!(off.provider_verdict(s(1)), ProviderClock::Mismatch, "61 seconds");
        let mut near = OffsetClock::new();
        near.observe(1060, 1000, s(0));
        assert_eq!(near.provider_verdict(s(1)), ProviderClock::Good, "60 seconds is not above 60");
        assert!(near.remote_now(5000, s(1)).is_some());
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
