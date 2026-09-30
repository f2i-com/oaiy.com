//! Time, as a thing the access model asks for instead of reading: every expiry, idle timeout,
//! throttle window and handoff lifetime is measured on a [`Clock`], so that the tests can move it (a
//! jump forward, a jump back) and nothing waits in real time.

use std::sync::atomic::{AtomicU64, Ordering};

/// Wall-clock time in milliseconds since the epoch.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// The system's clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// A clock the caller sets: for the tests of every part of the access model that has a time in it.
pub struct ManualClock(AtomicU64);

impl ManualClock {
    pub fn new(ms: u64) -> Self {
        ManualClock(AtomicU64::new(ms))
    }

    pub fn set(&self, ms: u64) {
        self.0.store(ms, Ordering::SeqCst);
    }

    pub fn advance(&self, ms: u64) {
        self.0.fetch_add(ms, Ordering::SeqCst);
    }

    pub fn rewind(&self, ms: u64) {
        self.0.fetch_sub(ms, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// A clock that only goes forward, whatever the wall clock does: what a link code's five minutes and an elevation's
/// ten are measured on, so that a jump of the wall clock cannot extend one (design 6, "clock skew").
pub struct MonotonicClock {
    start: std::time::Instant,
}

impl MonotonicClock {
    pub fn new() -> MonotonicClock {
        MonotonicClock {
            start: std::time::Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ms(&self) -> u64 {
        // One is added so that "zero" can never be a moment.
        self.start.elapsed().as_millis() as u64 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manual_clock_moves_only_when_told() {
        let c = ManualClock::new(1_000);
        assert_eq!(c.now_ms(), 1_000);
        c.advance(500);
        assert_eq!(c.now_ms(), 1_500);
        c.rewind(1_000);
        assert_eq!(c.now_ms(), 500);
        c.set(9);
        assert_eq!(c.now_ms(), 9);
    }

    #[test]
    fn the_system_clock_is_after_2024() {
        assert!(SystemClock.now_ms() > 1_704_067_200_000);
    }
}
