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
    /// The file could not be read, or could not be put aside: the tries are counted in memory, and the file is left as it is.
    protected: bool,
}

impl Attempts {
    /// The tries kept in `<dir>/ring-attempts.json` (UTF-8, or UTF-16 with its byte order mark). No file: none. A file
    /// that is not the tries, or is not text, is put aside as `ring-attempts.json.corrupt` (and so on) and the count starts
    /// again; one that cannot be read or put aside is left as it is, and is not written over.
    pub fn open(dir: &Path) -> Self {
        use crate::secret_file::{read_text, Text};
        let path = dir.join(FILE_NAME);
        let mut attempts = Self { path: Some(path.clone()), entries: Vec::new(), protected: false };
        let why = match read_text(&path) {
            Text::Missing => None,
            Text::Text(text) => match serde_json::from_str::<File>(&text) {
                Ok(file) => {
                    attempts.entries = file.attempts;
                    None
                }
                Err(e) => Some(e.to_string()),
            },
            Text::Undecodable(why) => Some(why),
            Text::Unreadable(e) => {
                log::warn!("ring: {} could not be read ({e}); the tries are counted in memory only", path.display());
                attempts.protected = true;
                None
            }
        };
        if let Some(why) = why {
            match crate::secret_file::keep_aside(&path) {
                Ok(aside) => log::warn!("ring: {} is not usable ({why}); it is kept as {}", path.display(), aside.display()),
                Err(e) => {
                    log::warn!("ring: {} is not usable ({why}) and could not be put aside ({e}); the tries are counted in memory only", path.display());
                    attempts.protected = true;
                }
            }
        }
        attempts
    }

    /// Tries held in memory only.
    pub fn in_memory() -> Self {
        Self { path: None, entries: Vec::new(), protected: false }
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
        if let Some(path) = self.path.as_ref().filter(|_| !self.protected) {
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
        let other = a.counters("call_9", "491570157", 1_400);
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
        assert_eq!(std::fs::read(dir.0.join("ring-attempts.json.corrupt")).unwrap(), b"{nope", "the file that would not read is kept, not written over");
    }

    /// The hour's tries are read as another program may have saved them, and a file that is not text is kept, like the settings and the messages.
    #[test]
    fn a_ledger_saved_as_utf16_is_read_and_one_that_is_not_text_is_kept_aside() {
        let dir = TempDir::new("ring-attempts-utf16");
        let mut a = Attempts::open(&dir.0);
        a.record("call_1", "491570006", 5_000);
        let text = std::fs::read_to_string(dir.0.join(FILE_NAME)).unwrap();
        let le: Vec<u8> = [0xFF, 0xFE].into_iter().chain(text.encode_utf16().flat_map(u16::to_le_bytes)).collect();
        std::fs::write(dir.0.join(FILE_NAME), le).unwrap();
        let read = Attempts::open(&dir.0);
        assert_eq!(read.counters("call_1", "491570006", 5_010).global_attempts_last_hour, 1, "the try is counted, not forgotten");
        assert!(!dir.0.join("ring-attempts.json.corrupt").exists());
        let bytes: &[u8] = &[b'{', 0xC3, 0x28];
        std::fs::write(dir.0.join(FILE_NAME), bytes).unwrap();
        let mut bad = Attempts::open(&dir.0);
        assert_eq!(bad.counters("call_1", "491570006", 5_010), Counters::default());
        assert_eq!(std::fs::read(dir.0.join("ring-attempts.json.corrupt")).unwrap(), bytes);
        bad.record("call_2", "491570156", 5_020);
        assert_eq!(std::fs::read(dir.0.join("ring-attempts.json.corrupt")).unwrap(), bytes, "and later tries do not touch it");
    }

    /// A ledger that cannot be read for the moment may be perfectly good: it is not put aside, and the tries since are counted in memory and do not replace it.
    #[test]
    fn a_ledger_that_cannot_be_read_is_left_alone_and_not_written_over() {
        let dir = TempDir::new("ring-attempts-unreadable");
        let mut first = Attempts::open(&dir.0);
        first.record("call_1", "491570006", 5_000);
        let before = std::fs::read(dir.0.join(FILE_NAME)).unwrap();
        let Some(lock) = crate::secret_file::testing::make_unreadable(&dir.0.join(FILE_NAME)) else {
            eprintln!("skipped: this user reads every file");
            return;
        };
        let mut a = Attempts::open(&dir.0);
        assert_eq!(a.counters("call_1", "491570006", 5_010).attempts_this_call, 0, "what is in it is not known");
        a.record("call_2", "491570156", 5_020);
        assert_eq!(a.counters("call_2", "491570156", 5_030).attempts_this_call, 1, "the limits still hold in memory");
        assert!(!dir.0.join("ring-attempts.json.corrupt").exists(), "a file that may be good is not put aside");
        drop(lock);
        assert_eq!(std::fs::read(dir.0.join(FILE_NAME)).unwrap(), before, "and it is as it was");
        assert_eq!(Attempts::open(&dir.0).counters("call_1", "491570006", 5_040).attempts_this_call, 1, "read again once it can be");
    }

    #[test]
    fn a_ledger_that_cannot_be_read_or_put_aside_is_not_written_over_and_still_counts_in_memory() {
        let dir = TempDir::new("ring-attempts-protected");
        let bytes: &[u8] = &[b'{', 0xC3, 0x28];
        std::fs::write(dir.0.join(FILE_NAME), bytes).unwrap();
        std::fs::create_dir_all(dir.0.join("ring-attempts.json.corrupt")).unwrap();
        for n in 1..40 {
            std::fs::create_dir_all(dir.0.join(format!("ring-attempts.json.corrupt.{n}"))).unwrap();
        }
        let mut a = Attempts::open(&dir.0);
        a.record("call_1", "491570006", 5_000);
        assert_eq!(a.counters("call_1", "491570006", 5_010).attempts_this_call, 1, "the limits still hold in memory");
        assert_eq!(std::fs::read(dir.0.join(FILE_NAME)).unwrap(), bytes, "the file is as it was");
    }
}
