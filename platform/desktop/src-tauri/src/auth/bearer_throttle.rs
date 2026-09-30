//! The failed-bearer throttle (design 4.5.6).
//!
//! Guessing a bearer must cost something on every route, not only on login. A request whose
//! `Authorization` names an unknown id or carries a wrong secret (`401 token_invalid`) counts one failure
//! against the client address (IPv6 by /64). Twenty failures in ten minutes block that address from every
//! request that carries `Authorization` for 15 minutes (`429 rate_limited`, `Retry-After`, no lookup),
//! doubling on each new block within 24 hours up to 24 hours: the same ladder as login. Expired and
//! revoked credentials do not count (a stale paired app is not an attacker). It is not applied to
//! loopback peers when the exposure is local (a desktop's own windows and a local script cannot be
//! blocked by it), and a block never applies to requests without `Authorization`, so a public route
//! stays reachable.
//!
//! The state is small and bounded, and can be saved and loaded (`throttle.json` holds it with the login
//! throttle of a later step) so that a restart does not clear a block.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::clock::Clock;

/// Failures inside [`WINDOW_MS`] that start a block.
pub const FAILURES_TO_BLOCK: usize = 20;
pub const WINDOW_MS: u64 = 10 * 60_000;
/// The first block; each new one within [`RESET_MS`] doubles it.
pub const FIRST_BLOCK_MS: u64 = 15 * 60_000;
pub const MAX_BLOCK_MS: u64 = 24 * 3_600_000;
/// A block that ended longer ago than this starts the ladder again.
pub const RESET_MS: u64 = 24 * 3_600_000;
/// Addresses tracked at once; the least recently seen quiet ones make room.
pub const MAX_TRACKED: usize = 50_000;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    /// Times of recent failures, oldest first.
    failures: VecDeque<u64>,
    /// Blocked until this time (zero: not blocked).
    blocked_until: u64,
    /// When the last block ended, and how many have begun within [`RESET_MS`] of the one before ending.
    last_block_end: u64,
    level: u32,
    last_seen: u64,
}

/// The throttle. One per server; every method takes `&self`.
pub struct BearerThrottle {
    state: Mutex<HashMap<String, Entry>>,
    clock: Arc<dyn Clock>,
    /// Changed since it was last taken for saving.
    dirty: AtomicBool,
}

/// The length of the `level`th block (1 is the first).
pub fn block_ms(level: u32) -> u64 {
    FIRST_BLOCK_MS
        .saturating_mul(1u64 << (level.saturating_sub(1)).min(20))
        .min(MAX_BLOCK_MS)
}

impl BearerThrottle {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        BearerThrottle {
            state: Mutex::new(HashMap::new()),
            clock,
            dirty: AtomicBool::new(false),
        }
    }

    /// Whether the state changed since the last call (and forget that it did): what the periodic save asks.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::Relaxed)
    }

    /// The state changed and is not saved (a save that failed).
    pub fn set_dirty(&self) {
        self.dirty.store(true, Ordering::Relaxed);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Entry>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Seconds until `key` may present a bearer again, if it is blocked now.
    pub fn blocked_for(&self, key: &str) -> Option<u64> {
        let now = self.clock.now_ms();
        let state = self.lock();
        let until = state.get(key)?.blocked_until;
        (until > now).then(|| (until - now).div_ceil(1000).max(1))
    }

    /// Count one failed bearer from `key` (an unknown id or a wrong secret). Returns whether this failure
    /// began a block.
    pub fn record_failure(&self, key: &str) -> bool {
        let now = self.clock.now_ms();
        let mut state = self.lock();
        self.set_dirty();
        if !state.contains_key(key) && state.len() >= MAX_TRACKED {
            evict(&mut state, now);
        }
        let e = state.entry(key.to_string()).or_default();
        e.last_seen = now;
        if e.blocked_until > now {
            // Already blocked: nothing more is counted while the block runs.
            return false;
        }
        while e
            .failures
            .front()
            .is_some_and(|t| now.saturating_sub(*t) >= WINDOW_MS)
        {
            e.failures.pop_front();
        }
        e.failures.push_back(now);
        if e.failures.len() < FAILURES_TO_BLOCK {
            return false;
        }
        e.failures.clear();
        // Within a day of the last block ending, the next one is twice as long; after a quiet day, the first rung.
        e.level = if e.last_block_end != 0 && now.saturating_sub(e.last_block_end) < RESET_MS {
            e.level + 1
        } else {
            1
        };
        e.blocked_until = now + block_ms(e.level);
        e.last_block_end = e.blocked_until;
        true
    }

    /// How many addresses are tracked (for the tests and the status).
    pub fn tracked(&self) -> usize {
        self.lock().len()
    }

    /// The state, for saving.
    pub fn snapshot(&self) -> serde_json::Value {
        let now = self.clock.now_ms();
        let state = self.lock();
        let live: HashMap<&String, &Entry> = state
            .iter()
            .filter(|(_, e)| {
                e.blocked_until > now
                    || !e.failures.is_empty()
                    || now.saturating_sub(e.last_block_end) < RESET_MS && e.last_block_end != 0
            })
            .collect();
        serde_json::to_value(live).unwrap_or(serde_json::Value::Null)
    }

    /// Load a saved state (a restart does not clear a block). What cannot be read is ignored, and what is
    /// read is made sane first: a file is not trusted to name a block that ends in the year 3000, or
    /// failures that have not happened yet, and an address with nothing left to remember is not loaded.
    pub fn restore(&self, saved: &serde_json::Value) {
        let Ok(loaded) = serde_json::from_value::<HashMap<String, Entry>>(saved.clone()) else {
            return;
        };
        let now = self.clock.now_ms();
        let mut state = self.lock();
        for (k, v) in loaded.into_iter().take(MAX_TRACKED) {
            if let Some(entry) = sane(v, now) {
                state.insert(k, entry);
            }
        }
    }
}

/// An entry read from a file, with what cannot be true cut back; `None` when nothing of it is worth keeping
/// (no block, no recent failure, and the last block is more than a day over).
fn sane(mut e: Entry, now: u64) -> Option<Entry> {
    let ceiling = now.saturating_add(MAX_BLOCK_MS);
    e.blocked_until = e.blocked_until.min(ceiling);
    e.last_block_end = e.last_block_end.min(ceiling);
    e.last_seen = e.last_seen.min(now);
    e.level = e.level.min(21);
    // Oldest first (a file may say anything), only the recent past, and never a block's worth.
    let mut times: Vec<u64> = e.failures.drain(..).collect();
    times.sort_unstable();
    times.retain(|t| *t <= now && now.saturating_sub(*t) < WINDOW_MS);
    e.failures = times.into();
    while e.failures.len() >= FAILURES_TO_BLOCK {
        e.failures.pop_front();
    }
    let worth_keeping = e.blocked_until > now
        || !e.failures.is_empty()
        || (e.last_block_end != 0 && now.saturating_sub(e.last_block_end) < RESET_MS);
    worth_keeping.then_some(e)
}

/// `<data>/auth/throttle.json`: the throttle's state on disk, so that a restart or a kill does not clear a
/// block (design 4.5.6 and 4.7.4). It holds `{"v":1,"bearer":{...}}` and whatever else a later step (the login
/// throttle) has put in it, kept and written back. A file written by a newer build (another `v`) is left
/// alone: read as empty, never written.
pub struct ThrottleFile {
    path: PathBuf,
    /// Whatever else the file holds: the state of another limiter (the web login's, under `login`), kept as it was
    /// read and replaced when its owner hands over a newer one ([`ThrottleFile::set`]).
    extra: Mutex<Map<String, Value>>,
    writable: bool,
}

/// The `v` of `throttle.json` this build reads and writes.
pub const THROTTLE_FILE_VERSION: u64 = 1;

impl ThrottleFile {
    /// Read `path`: the file, and the saved state of the bearer throttle if there is one. A file that is not
    /// there, or is not JSON, is an empty state (this is a throttle, not a credential: nothing is lost that
    /// cannot be counted again).
    pub fn open(path: &Path) -> (ThrottleFile, Option<Value>) {
        let mut file = ThrottleFile {
            path: path.to_path_buf(),
            extra: Mutex::new(Map::new()),
            writable: true,
        };
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("auth: {} could not be read: {e}", path.display());
                }
                return (file, None);
            }
        };
        let Ok(Value::Object(mut doc)) = serde_json::from_slice::<Value>(&bytes) else {
            log::warn!(
                "auth: {} is not readable: starting with no blocks",
                path.display()
            );
            return (file, None);
        };
        match doc.remove("v").and_then(|v| v.as_u64()) {
            Some(THROTTLE_FILE_VERSION) => {}
            other => {
                log::warn!(
                    "auth: {} has version {other:?}: it was written by another OAIY and is left alone",
                    path.display()
                );
                file.writable = false;
                return (file, None);
            }
        }
        let bearer = doc.remove("bearer");
        file.extra = Mutex::new(doc);
        (file, bearer)
    }

    /// What the file held under `key` when it was read, or was last given: the state of a limiter other than the
    /// bearer throttle (the web login's, under `login`).
    pub fn saved(&self, key: &str) -> Option<Value> {
        self.extra
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
    }

    /// Give the file the state of another limiter, to be written with the next save.
    pub fn set(&self, key: &str, state: Value) {
        self.extra
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_string(), state);
    }

    /// Write the file: atomic and private, with the bearer throttle's `state` and what else it held.
    pub fn save(&self, bearer: Value) -> std::io::Result<()> {
        if !self.writable {
            return Ok(());
        }
        let mut doc = self.extra.lock().unwrap_or_else(|e| e.into_inner()).clone();
        doc.insert("v".into(), Value::from(THROTTLE_FILE_VERSION));
        doc.insert("bearer".into(), bearer);
        let mut text =
            serde_json::to_string_pretty(&Value::Object(doc)).map_err(std::io::Error::other)?;
        text.push('\n');
        crate::secret_file::write(&self.path, text)
    }
}

/// Make room: drop the quiet address seen longest ago (never one that is blocked).
fn evict(state: &mut HashMap<String, Entry>, now: u64) {
    let victim = state
        .iter()
        .filter(|(_, e)| e.blocked_until <= now)
        .min_by_key(|(_, e)| e.last_seen)
        .map(|(k, _)| k.clone());
    if let Some(k) = victim {
        state.remove(&k);
    } else if let Some(k) = state
        .iter()
        .min_by_key(|(_, e)| e.last_seen)
        .map(|(k, _)| k.clone())
    {
        state.remove(&k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::clock::ManualClock;

    const T0: u64 = 1_790_000_000_000;

    fn throttle() -> (BearerThrottle, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(T0));
        (BearerThrottle::new(clock.clone()), clock)
    }

    fn fail(t: &BearerThrottle, key: &str, n: usize) {
        for _ in 0..n {
            t.record_failure(key);
        }
    }

    #[test]
    fn nineteen_failures_in_ten_minutes_block_nothing_and_the_twentieth_blocks_for_fifteen_minutes()
    {
        let (t, _) = throttle();
        fail(&t, "203.0.113.9", FAILURES_TO_BLOCK - 1);
        assert_eq!(t.blocked_for("203.0.113.9"), None);
        assert!(
            t.record_failure("203.0.113.9"),
            "the twentieth begins the block"
        );
        assert_eq!(t.blocked_for("203.0.113.9"), Some(15 * 60));
        // Another address is untouched.
        assert_eq!(t.blocked_for("203.0.113.10"), None);
    }

    #[test]
    fn failures_older_than_ten_minutes_do_not_count() {
        let (t, clock) = throttle();
        fail(&t, "a", 15);
        clock.advance(WINDOW_MS);
        fail(&t, "a", 15);
        assert_eq!(
            t.blocked_for("a"),
            None,
            "15 + 15 across the window edge is not 20 inside it"
        );
        // Exactly inside the window it is.
        let (t, clock) = throttle();
        fail(&t, "b", 10);
        clock.advance(WINDOW_MS - 1);
        fail(&t, "b", 10);
        assert!(t.blocked_for("b").is_some());
    }

    #[test]
    fn a_block_counts_down_and_ends_and_the_next_one_within_a_day_doubles() {
        let (t, clock) = throttle();
        fail(&t, "k", 20);
        assert_eq!(t.blocked_for("k"), Some(900));
        clock.advance(10 * 60_000);
        assert_eq!(t.blocked_for("k"), Some(300));
        clock.advance(5 * 60_000);
        assert_eq!(t.blocked_for("k"), None, "the block ends at 15 minutes");
        // The second block, within 24 hours of the first: 30 minutes.
        fail(&t, "k", 20);
        assert_eq!(t.blocked_for("k"), Some(30 * 60));
        clock.advance(30 * 60_000);
        fail(&t, "k", 20);
        assert_eq!(t.blocked_for("k"), Some(60 * 60), "then 1 hour");
    }

    #[test]
    fn the_ladder_is_15_30_60_120_240_480_960_minutes_then_24_hours() {
        let want: Vec<u64> = [15, 30, 60, 120, 240, 480, 960, 1440, 1440, 1440]
            .iter()
            .map(|m| m * 60_000)
            .collect();
        let got: Vec<u64> = (1..=10).map(block_ms).collect();
        assert_eq!(got, want);
        assert_eq!(block_ms(0), FIRST_BLOCK_MS, "there is no level 0");
        assert_eq!(block_ms(u32::MAX), MAX_BLOCK_MS, "and no overflow");
        // Driven through the throttle on a fake clock: a block that begins as the last one ends is the next rung.
        let (t, clock) = throttle();
        for (level, minutes) in [15u64, 30, 60, 120, 240, 480, 960, 1440, 1440]
            .iter()
            .enumerate()
        {
            fail(&t, "k", 20);
            assert_eq!(
                t.blocked_for("k"),
                Some(minutes * 60),
                "block {}",
                level + 1
            );
            clock.advance(minutes * 60_000);
        }
    }

    #[test]
    fn a_quiet_day_starts_the_ladder_again() {
        let (t, clock) = throttle();
        fail(&t, "k", 20);
        assert_eq!(t.blocked_for("k"), Some(15 * 60));
        clock.advance(FIRST_BLOCK_MS + RESET_MS - 1);
        fail(&t, "k", 20);
        assert_eq!(
            t.blocked_for("k"),
            Some(30 * 60),
            "one millisecond short of a quiet day after the block ended: the next rung"
        );
        clock.advance(block_ms(2) + RESET_MS);
        fail(&t, "k", 20);
        assert_eq!(
            t.blocked_for("k"),
            Some(15 * 60),
            "a quiet day after the block ended: the first rung again"
        );
    }

    #[test]
    fn nothing_is_counted_while_a_block_runs_and_it_does_not_grow_by_being_hammered() {
        let (t, clock) = throttle();
        fail(&t, "k", 20);
        let until = t.blocked_for("k").unwrap();
        for _ in 0..1000 {
            assert!(!t.record_failure("k"));
        }
        assert_eq!(
            t.blocked_for("k"),
            Some(until),
            "a flood inside the block does not lengthen it"
        );
        clock.advance(until * 1000);
        assert_eq!(t.blocked_for("k"), None);
        // And the failures during the block did not count toward the next.
        fail(&t, "k", 19);
        assert_eq!(t.blocked_for("k"), None);
    }

    #[test]
    fn an_ipv6_client_is_one_key_by_its_slash_64() {
        use crate::auth::clientip::bucket_key;
        let (t, _) = throttle();
        let a = bucket_key("2001:db8:0:1::1".parse().unwrap());
        let b = bucket_key("2001:db8:0:1:ffff::9".parse().unwrap());
        assert_eq!(a, b);
        fail(&t, &a, 10);
        fail(&t, &b, 10);
        assert!(
            t.blocked_for(&a).is_some(),
            "the two addresses of one /64 count together"
        );
    }

    #[test]
    fn the_retry_after_rounds_up_and_is_never_zero() {
        let (t, clock) = throttle();
        fail(&t, "k", 20);
        clock.advance(FIRST_BLOCK_MS - 1);
        assert_eq!(t.blocked_for("k"), Some(1));
        clock.advance(1);
        assert_eq!(t.blocked_for("k"), None);
    }

    #[test]
    fn the_state_is_bounded_and_a_blocked_address_is_the_last_to_be_forgotten() {
        let (t, clock) = throttle();
        fail(&t, "blocked", 20);
        for i in 0..(MAX_TRACKED + 100) {
            clock.advance(1);
            t.record_failure(&format!("ip-{i}"));
        }
        assert!(t.tracked() <= MAX_TRACKED);
        assert!(
            t.blocked_for("blocked").is_some(),
            "a flood of new addresses cannot push out a block"
        );
    }

    #[test]
    fn a_restart_does_not_clear_a_block() {
        let (t, clock) = throttle();
        fail(&t, "k", 20);
        fail(&t, "quiet", 3);
        let saved = t.snapshot();
        let restarted = BearerThrottle::new(clock.clone());
        assert_eq!(restarted.blocked_for("k"), None);
        restarted.restore(&saved);
        assert_eq!(restarted.blocked_for("k"), Some(15 * 60));
        // The three failures came along too.
        fail(&restarted, "quiet", 16);
        assert_eq!(restarted.blocked_for("quiet"), None);
        fail(&restarted, "quiet", 1);
        assert!(restarted.blocked_for("quiet").is_some());
        // A saved state that cannot be read is ignored.
        restarted.restore(&serde_json::json!("garbage"));
        restarted.restore(&serde_json::json!({ "x": { "failures": "no" } }));
    }

    #[test]
    fn f9_a_failure_makes_the_state_dirty_until_it_is_taken() {
        let (t, _) = throttle();
        assert!(!t.take_dirty(), "nothing has happened");
        t.record_failure("k");
        assert!(t.take_dirty());
        assert!(!t.take_dirty(), "taken once");
        t.set_dirty();
        assert!(t.take_dirty(), "a save that failed is asked for again");
        // A block that is only read changes nothing.
        fail(&t, "b", 20);
        t.take_dirty();
        assert!(t.blocked_for("b").is_some());
        assert!(!t.take_dirty());
    }

    #[test]
    fn f9_a_file_is_not_trusted_to_name_a_block_of_the_year_3000_or_failures_from_the_future() {
        let (t, clock) = throttle();
        let now = clock.now_ms();
        let day = 24 * 3_600_000u64;
        let entry = |failures: Vec<u64>, blocked_until: u64, last_block_end: u64, level: u32| {
            serde_json::json!({
                "failures": failures, "blocked_until": blocked_until,
                "last_block_end": last_block_end, "level": level, "last_seen": now + 5 * day
            })
        };
        t.restore(&serde_json::json!({
            // A block that ends at the end of time: cut to a day.
            "forever": entry(vec![], u64::MAX, u64::MAX, u32::MAX),
            // Failures that have not happened yet, and ones from an hour ago: only the recent past counts.
            "future": entry(vec![now + 1000, now + 2000], 0, 0, 0),
            "old": entry(vec![now - 3_600_000], 0, 0, 0),
            // Nothing left to remember: not loaded.
            "stale": entry(vec![], now - 2 * day, now - 2 * day, 3),
            // A hundred failures in the last minute: never more than a block's worth are kept.
            "many": entry((0..100).map(|i| now - i * 1000).collect(), 0, 0, 0),
        }));
        assert!(
            t.blocked_for("forever").unwrap() <= day / 1000,
            "{:?}",
            t.blocked_for("forever")
        );
        assert!(t.blocked_for("forever").unwrap() > day / 1000 - 5);
        assert_eq!(t.blocked_for("future"), None);
        assert_eq!(t.blocked_for("old"), None);
        assert_eq!(t.tracked(), 2, "forever and many; the others are not kept");
        // "many" has at most 19 failures, the newest: one more and it is blocked.
        assert_eq!(t.blocked_for("many"), None);
        assert_eq!(
            t.snapshot()["many"]["failures"].as_array().unwrap().len(),
            FAILURES_TO_BLOCK - 1
        );
        t.record_failure("many");
        assert!(t.blocked_for("many").is_some(), "the twentieth blocks");
        // The block it earns is the first rung, not a rung of a made-up level.
        assert_eq!(t.blocked_for("many"), Some(15 * 60));
        // The made-up level of "forever" (the largest number there is) is cut: an hour after its block ends
        // the next one is the top rung, and nothing overflows on the way.
        clock.advance(day + 3_600_000);
        for _ in 0..20 {
            t.record_failure("forever");
        }
        let next = t.blocked_for("forever").unwrap();
        assert!(next <= day / 1000 && next > day / 2000, "{next}");
    }

    fn scratch(tag: &str) -> crate::secret_file::testing::TempDir {
        crate::secret_file::testing::TempDir::new(tag)
    }

    #[test]
    fn f9_the_throttle_file_round_trips_keeps_what_else_it_holds_and_leaves_a_newer_file_alone() {
        let dir = scratch("throttle-file");
        let path = dir.0.join("throttle.json");
        // Not there: an empty state.
        let (file, saved) = ThrottleFile::open(&path);
        assert!(saved.is_none());
        // Saved and read back, with the block in it.
        let (t, clock) = throttle();
        fail(&t, "203.0.113.9", 20);
        file.save(t.snapshot()).unwrap();
        let (again, saved) = ThrottleFile::open(&path);
        let restarted = BearerThrottle::new(clock.clone());
        restarted.restore(&saved.expect("the bearer state"));
        assert_eq!(restarted.blocked_for("203.0.113.9"), Some(15 * 60));
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc["v"], 1);
        // What a later step put in the file is kept when this one writes.
        let mut with_more = doc.clone();
        with_more["login"] = serde_json::json!({ "l1": [1, 2, 3] });
        std::fs::write(&path, with_more.to_string()).unwrap();
        let (file, _) = ThrottleFile::open(&path);
        file.save(t.snapshot()).unwrap();
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc["login"], serde_json::json!({ "l1": [1, 2, 3] }));
        assert!(doc["bearer"].get("203.0.113.9").is_some());
        // The other limiter's state can be read back, and replaced: the next save writes the newer one.
        assert_eq!(
            file.saved("login"),
            Some(serde_json::json!({ "l1": [1, 2, 3] }))
        );
        assert_eq!(file.saved("nothing"), None);
        file.set("login", serde_json::json!({ "l1": [4] }));
        file.save(t.snapshot()).unwrap();
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc["login"], serde_json::json!({ "l1": [4] }));
        assert_eq!(doc["v"], 1);
        drop(again);
        // A file of another version is read as empty and never written.
        let newer = r#"{"v":2,"bearer":{"x":1},"future":true}"#;
        std::fs::write(&path, newer).unwrap();
        let (file, saved) = ThrottleFile::open(&path);
        assert!(saved.is_none());
        file.save(t.snapshot()).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), newer);
        // A file that is not JSON is an empty state, and the next save replaces it.
        std::fs::write(&path, "{ not json").unwrap();
        let (file, saved) = ThrottleFile::open(&path);
        assert!(saved.is_none());
        file.save(t.snapshot()).unwrap();
        assert!(ThrottleFile::open(&path).1.is_some());
        // Not an object, no version, a version that is not a number: all empty, none written over a newer one.
        for text in ["[]", "3", r#"{"bearer":{}}"#, r#"{"v":"1","bearer":{}}"#] {
            std::fs::write(&path, text).unwrap();
            let (_, saved) = ThrottleFile::open(&path);
            assert!(saved.is_none(), "{text}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn f9_the_throttle_file_is_private() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = scratch("throttle-file-mode");
        let path = dir.0.join("throttle.json");
        let (file, _) = ThrottleFile::open(&path);
        file.save(serde_json::json!({})).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
