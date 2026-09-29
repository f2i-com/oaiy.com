//! How many times callers have been put through lately, so a caller cannot make the owner's
//! devices ring over and over: the tries of the last hour, kept across a restart in
//! `<data>/ring-attempts.json` (owner-only: it holds callers' numbers).
//!
//! A try is counted the moment it is allowed, not when its ring opens, so a request that
//! passes the policy and then never reaches a ring (a plugin that fails between the two, and
//! says nothing) still uses one up: the limit cannot be got around by making the second half
//! fail. When the plugin says it refused the request itself (consent, a changed call, a plan it
//! could not use), the try is given back: nobody was rung, and a refusal must not start the gap
//! between tries or spend the caller's hour.
//!
//! Callers who hide their number (or give one that is not a number) share one bucket of the hour,
//! a small one, so neither hiding nor making numbers up gets a bucket of its own for every call.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::plan::Counters;

/// The attempts' file name in the data folder.
pub const FILE_NAME: &str = "ring-attempts.json";
/// How long a try counts against the hourly limits, seconds.
pub const WINDOW_SECONDS: u64 = 3_600;
/// The caller key of every withheld, hidden or unparseable number: they share ONE bucket, so a caller who hides their
/// number (or makes a new one up each time) cannot have a bucket of their own for every call.
pub const WITHHELD: &str = "withheld";
/// The most tries one hour allows from the withheld bucket, whatever the owner's per-caller limit says.
pub const WITHHELD_PER_HOUR: u32 = 2;
/// The most tries kept (the limits are far below it: this only bounds the file).
const MAX_KEPT: usize = 500;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Attempt {
    /// Unix seconds.
    at: u64,
    call: String,
    /// The caller's number key (the last nine digits), or [`WITHHELD`] for a hidden or unparseable number.
    caller: String,
}

#[derive(Default, Serialize, Deserialize)]
struct File {
    #[serde(default)]
    attempts: Vec<Attempt>,
}

/// The last hour's tries.
pub struct Attempts {
    path: Option<PathBuf>,
    entries: Vec<Attempt>,
}

impl Attempts {
    /// The tries kept in `<dir>/ring-attempts.json` (none when there is no usable file).
    pub fn open(dir: &Path) -> Self {
        let path = dir.join(FILE_NAME);
        let entries = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<File>(&text).ok())
            .map(|f| f.attempts)
            .unwrap_or_default();
        Self { path: Some(path), entries }
    }

    /// Tries held in memory only.
    pub fn in_memory() -> Self {
        Self { path: None, entries: Vec::new() }
    }

    /// The counters for `call` and its `caller` at `now` (Unix seconds).
    pub fn counters(&self, call: &str, caller: &str, now: u64) -> Counters {
        let recent = |a: &&Attempt| now.saturating_sub(a.at) < WINDOW_SECONDS;
        let mine: Vec<&Attempt> = self.entries.iter().filter(recent).filter(|a| a.call == call).collect();
        Counters {
            attempts_this_call: mine.len() as u32,
            seconds_since_last_attempt: mine.iter().map(|a| now.saturating_sub(a.at)).min(),
            caller_attempts_last_hour: self.entries.iter().filter(recent).filter(|a| a.caller == caller).count() as u32,
            global_attempts_last_hour: self.entries.iter().filter(recent).count() as u32,
        }
    }

    /// Give back the latest try counted for `call`: the request it was counted for never came to a ring (the plugin refused it),
    /// so it must not start the gap between tries or use up an hour's allowance. Whether there was one.
    pub fn forget_last(&mut self, call: &str) -> bool {
        let Some(at) = self.entries.iter().rposition(|a| a.call == call) else { return false };
        self.entries.remove(at);
        self.keep();
        true
    }

    fn keep(&self) {
        if let Some(path) = &self.path {
            let body = serde_json::to_string(&File { attempts: self.entries.clone() }).unwrap_or_default();
            if let Err(e) = crate::secret_file::write(path, body) {
                log::warn!("ring: the attempts could not be kept: {e}");
            }
        }
    }

    /// Count a try at `now`, and keep the ledger (tries older than the hour are let go).
    pub fn record(&mut self, call: &str, caller: &str, now: u64) {
        self.entries.retain(|a| now.saturating_sub(a.at) < WINDOW_SECONDS);
        self.entries.push(Attempt { at: now, call: call.to_string(), caller: caller.to_string() });
        if self.entries.len() > MAX_KEPT {
            let extra = self.entries.len() - MAX_KEPT;
            self.entries.drain(..extra);
        }
        self.keep();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::{assert_private, TempDir};

    #[test]
    fn nothing_tried_is_all_zero() {
        let a = Attempts::in_memory();
        assert_eq!(a.counters("call_1", "491570006", 1_000), Counters::default());
    }

    #[test]
    fn tries_count_by_call_by_caller_and_overall_inside_the_hour() {
        let mut a = Attempts::in_memory();
        a.record("call_1", "491570006", 1_000);
        a.record("call_1", "491570006", 1_100);
        a.record("call_2", "491570006", 1_200);
        a.record("call_3", "491570156", 1_300);
        let c = a.counters("call_1", "491570006", 1_400);
        assert_eq!((c.attempts_this_call, c.seconds_since_last_attempt), (2, Some(300)));
        assert_eq!((c.caller_attempts_last_hour, c.global_attempts_last_hour), (3, 4));
        let other = a.counters("call_9", "491570999", 1_400);
        assert_eq!((other.attempts_this_call, other.seconds_since_last_attempt, other.caller_attempts_last_hour, other.global_attempts_last_hour), (0, None, 0, 4));
    }

    #[test]
    fn a_try_given_back_is_forgotten_with_its_gap_and_its_share_of_the_hour() {
        let dir = TempDir::new("ring-attempts-refund");
        let mut a = Attempts::open(&dir.0);
        a.record("call_1", "491570006", 1_000);
        a.record("call_1", "491570006", 1_100);
        a.record("call_2", "491570156", 1_150);
        // The latest try of that call goes, not the first and not another call's.
        assert!(a.forget_last("call_1"));
        let c = a.counters("call_1", "491570006", 1_200);
        assert_eq!((c.attempts_this_call, c.seconds_since_last_attempt, c.caller_attempts_last_hour, c.global_attempts_last_hour), (1, Some(200), 1, 2));
        assert!(a.forget_last("call_1"));
        let c = a.counters("call_1", "491570006", 1_200);
        assert_eq!((c.attempts_this_call, c.seconds_since_last_attempt), (0, None), "no try left, so no gap");
        assert!(!a.forget_last("call_1"), "nothing more to give back");
        assert!(!a.forget_last("call_9"));
        assert_eq!(a.counters("call_2", "491570156", 1_200).global_attempts_last_hour, 1);
        // Kept: a fresh desktop on the same folder reads it back.
        assert_eq!(Attempts::open(&dir.0).counters("call_1", "491570006", 1_200).global_attempts_last_hour, 1);
    }

    #[test]
    fn a_try_older_than_an_hour_no_longer_counts() {
        let mut a = Attempts::in_memory();
        a.record("call_1", "491570006", 1_000);
        assert_eq!(a.counters("call_1", "491570006", 1_000 + WINDOW_SECONDS - 1).global_attempts_last_hour, 1);
        assert_eq!(a.counters("call_1", "491570006", 1_000 + WINDOW_SECONDS), Counters::default());
        // ...and a later try lets it go of the ledger.
        a.record("call_2", "491570156", 1_000 + WINDOW_SECONDS);
        assert_eq!(a.entries.len(), 1);
    }

    #[test]
    fn the_ledger_is_owner_only_and_survives_a_restart() {
        let dir = TempDir::new("ring-attempts");
        let mut a = Attempts::open(&dir.0);
        a.record("call_1", "491570006", 5_000);
        assert_private(&dir.0.join(FILE_NAME));
        let again = Attempts::open(&dir.0);
        let c = again.counters("call_1", "491570006", 5_010);
        assert_eq!((c.attempts_this_call, c.seconds_since_last_attempt, c.caller_attempts_last_hour), (1, Some(10), 1));
    }

    #[test]
    fn a_ledger_that_cannot_be_read_starts_empty_and_the_file_is_bounded() {
        let dir = TempDir::new("ring-attempts-bad");
        std::fs::write(dir.0.join(FILE_NAME), "{nope").unwrap();
        let mut a = Attempts::open(&dir.0);
        assert_eq!(a.counters("c", "k", 1), Counters::default());
        for n in 0..(MAX_KEPT as u64 + 20) {
            a.record("c", "k", 10 + n);
        }
        assert_eq!(a.entries.len(), MAX_KEPT);
    }
}
