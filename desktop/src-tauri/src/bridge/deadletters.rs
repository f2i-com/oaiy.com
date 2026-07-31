//! Dead letters: events that arrived and produced no work.
//!
//! At-least-once delivery makes a *lost* event survivable — a well-behaved
//! plugin re-sends anything it has not seen acked. What it does not cover is an
//! event that was accepted, acked, and then failed to become a run. Two paths
//! do exactly that today, and both are silent:
//!
//! 1. **Shed under flood.** The host's event queue is bounded on purpose (an
//!    unbounded one is a memory-exhaustion path). `try_send` drops on a full
//!    queue and writes a line to the plugin's log ring — a ring which, under
//!    the very flood that caused the drop, is itself overwriting. The event is
//!    gone and nothing durable records that it existed.
//!
//! 2. **Dispatched to nothing, operationally.** A binding matched but could not
//!    fire: its condition would not evaluate, a loop guard refused it, or more
//!    than [`MAX_BINDINGS_PER_EVENT`] matched. The event ring holds the reason
//!    for 500 events and forgets it on restart.
//!
//! In both cases something that was supposed to happen did not, and after a
//! restart there is no evidence it was ever supposed to.
//!
//! # What is NOT a dead letter
//!
//! A binding that is disabled, is `manual`, or whose condition evaluated
//! *false* is working correctly — that is the trigger system declining, which is
//! the whole design (see [`super::triggers`]: everything fails towards not
//! running). Recording those would bury the real failures under routine
//! not-firing. An event that matched no binding at all is likewise ordinary:
//! most plugin events are bound by nobody.
//!
//! So a dead letter means: **this event should have produced work, and did
//! not.**
//!
//! # Redrive re-dispatches; it does not replay a decision
//!
//! Redrive feeds the stored envelope back through the ordinary dispatch path
//! against the *current* bindings. That matters for safety: a guard-refused
//! event redriven is offered to the guards again and refused again, rather than
//! being forced through. Fix the binding and redrive succeeds; don't, and it
//! doesn't. The only thing redrive changes is the idempotency key, which gains
//! an attempt suffix — sound precisely because a dead letter is by definition an
//! event that reserved no run, so there is no execution to duplicate.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cap on retained dead letters. Small on purpose: this is a queue a human works
/// through, not an archive. Oldest are evicted first — a dead letter that has
/// been sitting unaddressed past the newest 500 is not going to be actioned.
const MAX_DEAD_LETTERS: usize = 500;

/// Compact the file once it exceeds `max(this, 2 x live entries)` lines. Mirrors
/// the run ledger's journal, for the same reason: replay cost must not grow
/// without bound just because the process has been up a long time.
const COMPACT_MIN_LINES: usize = 512;

/// Why an event produced no work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeadReason {
    /// The host's event queue was full. The event never reached dispatch.
    Shed,
    /// Dispatch ran, and every binding that matched failed to fire for an
    /// operational reason. Carries the dispatcher's own words.
    NotReserved { detail: String },
}

impl DeadReason {
    pub fn message(&self) -> String {
        match self {
            DeadReason::Shed => {
                "the host event queue was full, so this event was dropped before dispatch".into()
            }
            DeadReason::NotReserved { detail } => detail.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeadLetter {
    pub id: String,
    /// Plugin that emitted it.
    pub source: String,
    /// Event name, e.g. `aokie.call.incoming`.
    pub event: String,
    pub reason: DeadReason,
    /// The whole envelope, because redrive has to reconstruct the event exactly.
    pub envelope: Value,
    pub recorded_at_ms: u64,
    /// Redrive attempts so far. Also what makes each retry's idempotency key
    /// unique.
    #[serde(default)]
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_attempt_ms: Option<u64>,
    /// What the most recent redrive did, in the dispatcher's words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_outcome: Option<String>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Durable, bounded dead-letter queue.
///
/// The same append-and-replay JSONL as the ledger, for the same reason: the
/// write rate is low, the set is bounded, and a dead letter that does not
/// survive a restart is barely a dead letter at all — the whole failure this
/// fixes is "nothing recorded that it existed".
#[derive(Default)]
pub struct DeadLetterQueue {
    /// Newest last, matching insertion order.
    items: VecDeque<DeadLetter>,
    seq: u64,
    path: Option<PathBuf>,
    /// Lines currently in the file, including superseded ones. Drives
    /// compaction; see [`DeadLetterQueue::append_one`].
    lines_on_disk: usize,
}

impl DeadLetterQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a durable queue, replaying whatever is on disk.
    ///
    /// A corrupt line is skipped rather than fatal — losing one dead letter to a
    /// torn write is much better than losing the queue.
    pub fn open(path: PathBuf) -> Self {
        let mut q = Self::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            // LAST WRITE PER ID WINS — the same upsert semantics the run
            // ledger's journal replays with, and what makes appending an update
            // (a redrive attempt) sound: the newer line supersedes the older
            // rather than replaying as a second, stale copy of the same entry.
            // Insertion order is preserved so "newest last" still holds.
            let mut order: Vec<String> = Vec::new();
            let mut latest: std::collections::HashMap<String, DeadLetter> =
                std::collections::HashMap::new();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(item) = serde_json::from_str::<DeadLetter>(line) {
                    q.seq = q.seq.max(seq_of(&item.id));
                    if !latest.contains_key(&item.id) {
                        order.push(item.id.clone());
                    }
                    latest.insert(item.id.clone(), item);
                }
            }
            for id in order {
                if let Some(item) = latest.remove(&id) {
                    q.items.push_back(item);
                }
            }
        }
        while q.items.len() > MAX_DEAD_LETTERS {
            q.items.pop_front();
        }
        q.path = Some(path);
        // Replay may have dropped lines (corrupt, superseded, over cap); write
        // back what we actually hold so the file matches memory from here on —
        // which is also what makes `lines_on_disk` an exact count rather than a
        // guess.
        q.persist();
        q
    }

    /// Append ONE record. The common path, and the only one on the hot side of
    /// a flood.
    ///
    /// Recording a shed event used to rewrite and fsync the entire retained
    /// queue — up to [`MAX_DEAD_LETTERS`] entries, each carrying a whole event
    /// envelope. That is O(queue) work per drop, and drops arrive precisely
    /// when the machine is already struggling to keep up, so the cost grew with
    /// the very backlog it was recording. Appending is O(1); the file is
    /// last-wins on replay (see [`Self::open`]), so a later line for the same id
    /// simply supersedes an earlier one, exactly as the run ledger's journal
    /// does.
    fn append_one(&self, item: &DeadLetter) {
        let Some(path) = &self.path else { return };
        let write = || -> std::io::Result<()> {
            let line = serde_json::to_string(item).map_err(std::io::Error::other)?;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_all()
        };
        if let Err(e) = write() {
            log::warn!("cannot append a dead letter to {}: {e}", path.display());
        }
    }

    /// Rewrite the file to exactly what is held now.
    ///
    /// Reserved for the paths that REMOVE something — eviction, dismissal,
    /// clear — because an append-only file cannot express a deletion, and for
    /// the compaction that keeps replay bounded.
    fn persist(&self) {
        let Some(path) = &self.path else { return };
        let tmp = path.with_extension("jsonl.tmp");
        let write = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            for item in &self.items {
                let line = serde_json::to_string(item).map_err(std::io::Error::other)?;
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, path)
        };
        if let Err(e) = write() {
            // A queue that cannot be written must not take the runtime down —
            // it degrades to in-memory, which is what it was before this existed.
            log::warn!("cannot write dead letters to {}: {e}", path.display());
        }
    }

    /// Rewrite and re-baseline the line count. `persist` leaves the file holding
    /// exactly one line per live entry, so anything that calls it must say so —
    /// otherwise the counter drifts up forever and compaction runs on every
    /// write.
    fn persist_and_reset(&mut self) {
        self.persist();
        self.lines_on_disk = self.items.len();
    }

    /// Record an event that produced no work. Returns the stored entry.
    pub fn record(
        &mut self,
        source: &str,
        event: &str,
        reason: DeadReason,
        envelope: Value,
    ) -> DeadLetter {
        self.seq += 1;
        let item = DeadLetter {
            id: format!("dl_{}", self.seq),
            source: source.to_string(),
            event: event.to_string(),
            reason,
            envelope,
            recorded_at_ms: now_ms(),
            attempts: 0,
            last_attempt_ms: None,
            last_outcome: None,
        };
        self.items.push_back(item.clone());
        // Eviction is a deletion, which an append cannot express — so the rare
        // over-cap case falls back to the full rewrite, and everything else
        // takes the O(1) path.
        if self.items.len() > MAX_DEAD_LETTERS {
            while self.items.len() > MAX_DEAD_LETTERS {
                self.items.pop_front();
            }
            self.persist_and_reset();
        } else if self.lines_on_disk >= self.compact_threshold() {
            // Bounded replay: without this, a queue that is written to for a
            // long time accumulates superseded lines forever and `open` gets
            // slower every launch.
            self.persist_and_reset();
        } else {
            self.append_one(&item);
            self.lines_on_disk += 1;
        }
        item
    }

    /// Compact once the file holds meaningfully more lines than live entries.
    /// Two per entry: a dead letter is written once and then updated by a
    /// redrive attempt, so steady state is well under this.
    fn compact_threshold(&self) -> usize {
        COMPACT_MIN_LINES.max(self.items.len() * 2)
    }

    /// Newest first — a queue is worked from the most recent failure back.
    pub fn list(&self, limit: usize) -> Vec<DeadLetter> {
        self.items.iter().rev().take(limit).cloned().collect()
    }

    pub fn get(&self, id: &str) -> Option<DeadLetter> {
        self.items.iter().find(|d| d.id == id).cloned()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Count an attempt and record what it did. Kept even on success, until the
    /// caller drops it — see [`Self::remove`].
    pub fn note_attempt(&mut self, id: &str, outcome: String) -> Option<DeadLetter> {
        let item = self.items.iter_mut().find(|d| d.id == id)?;
        item.attempts += 1;
        item.last_attempt_ms = Some(now_ms());
        item.last_outcome = Some(outcome);
        let snapshot = item.clone();
        // An update, not a deletion: append the new version and let replay's
        // last-wins rule supersede the old line.
        if self.lines_on_disk >= self.compact_threshold() {
            self.persist_and_reset();
        } else {
            self.append_one(&snapshot);
            self.lines_on_disk += 1;
        }
        Some(snapshot)
    }

    /// Drop one — after a redrive that reserved a run, or a human deciding it
    /// does not matter.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.items.len();
        self.items.retain(|d| d.id != id);
        let removed = self.items.len() != before;
        if removed {
            // A deletion cannot be appended, so this is one of the few paths
            // that genuinely needs the full rewrite.
            self.persist_and_reset();
        }
        removed
    }

    /// Empty the queue.
    pub fn clear(&mut self) -> usize {
        let n = self.items.len();
        self.items.clear();
        self.persist_and_reset();
        n
    }

    /// The idempotency key a redrive should use: the original plus the attempt
    /// number, so each retry is a distinct reservation.
    ///
    /// Safe because a dead letter reserved no run — there is no prior execution
    /// under the original key to duplicate. It is deliberately derived rather
    /// than random so that redriving the same entry twice at the same attempt
    /// count (a double-click, a retried HTTP request) still collapses to one
    /// reservation.
    pub fn redrive_key(original: &str, attempt: u32) -> String {
        if original.is_empty() {
            format!("redrive-{attempt}")
        } else {
            format!("{original}#redrive{attempt}")
        }
    }
}

/// `dl_12` -> 12, so a replayed queue keeps issuing unique ids.
fn seq_of(id: &str) -> u64 {
    id.strip_prefix("dl_").and_then(|n| n.parse().ok()).unwrap_or(0)
}

pub type DeadLetterHandle = Arc<Mutex<DeadLetterQueue>>;

pub fn new_handle() -> DeadLetterHandle {
    Arc::new(Mutex::new(DeadLetterQueue::new()))
}

pub fn open_handle(path: PathBuf) -> DeadLetterHandle {
    Arc::new(Mutex::new(DeadLetterQueue::open(path)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope(name: &str) -> Value {
        json!({ "name": name, "idempotencyKey": "k1", "data": { "from": "+61400000000" } })
    }

    struct Dir(PathBuf);
    impl Dir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("oaiy-dlq-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> PathBuf {
            self.0.join("deadletters.jsonl")
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn records_and_lists_newest_first() {
        let mut q = DeadLetterQueue::new();
        q.record("aokie", "call.incoming", DeadReason::Shed, envelope("a"));
        q.record(
            "aokie",
            "call.missed",
            DeadReason::NotReserved { detail: "guard refused".into() },
            envelope("b"),
        );

        let all = q.list(10);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].event, "call.missed");
        assert_ne!(all[0].id, all[1].id);
    }

    #[test]
    fn survives_a_restart_with_the_envelope_intact() {
        let dir = Dir::new("restart");
        {
            let mut q = DeadLetterQueue::open(dir.path());
            q.record("aokie", "call.incoming", DeadReason::Shed, envelope("a"));
        }
        // The entire point: an event shed at 3am is still there in the morning,
        // and complete enough to redrive.
        let q = DeadLetterQueue::open(dir.path());
        assert_eq!(q.len(), 1);
        let item = &q.list(1)[0];
        assert_eq!(item.reason, DeadReason::Shed);
        assert_eq!(item.envelope["data"]["from"], "+61400000000");
    }

    #[test]
    fn ids_stay_unique_across_a_restart() {
        let dir = Dir::new("ids");
        let first = {
            let mut q = DeadLetterQueue::open(dir.path());
            q.record("p", "e", DeadReason::Shed, envelope("a")).id
        };
        let mut q = DeadLetterQueue::open(dir.path());
        let second = q.record("p", "e", DeadReason::Shed, envelope("b")).id;
        // Reusing `dl_1` would make remove() and redrive target the wrong entry.
        assert_ne!(first, second);
    }

    #[test]
    fn a_corrupt_line_does_not_discard_the_queue() {
        let dir = Dir::new("corrupt");
        {
            let mut q = DeadLetterQueue::open(dir.path());
            q.record("p", "good", DeadReason::Shed, envelope("a"));
        }
        let mut text = std::fs::read_to_string(dir.path()).unwrap();
        text.push_str("{not json\n");
        std::fs::write(dir.path(), text).unwrap();

        let q = DeadLetterQueue::open(dir.path());
        assert_eq!(q.len(), 1);
        assert_eq!(q.list(1)[0].event, "good");
    }

    #[test]
    fn the_queue_is_bounded() {
        let mut q = DeadLetterQueue::new();
        for i in 0..MAX_DEAD_LETTERS + 20 {
            q.record("p", &format!("e{i}"), DeadReason::Shed, envelope("x"));
        }
        assert_eq!(q.len(), MAX_DEAD_LETTERS);
        // Oldest evicted, newest kept.
        assert_eq!(q.list(1)[0].event, format!("e{}", MAX_DEAD_LETTERS + 19));
    }

    #[test]
    fn attempts_are_counted_and_persisted() {
        let dir = Dir::new("attempts");
        let id = {
            let mut q = DeadLetterQueue::open(dir.path());
            let id = q.record("p", "e", DeadReason::Shed, envelope("a")).id;
            q.note_attempt(&id, "no binding matched".into());
            q.note_attempt(&id, "reserved run_7".into());
            id
        };
        let q = DeadLetterQueue::open(dir.path());
        let item = q.get(&id).unwrap();
        assert_eq!(item.attempts, 2);
        // The latest outcome, so a human can see redrive is now getting further.
        assert_eq!(item.last_outcome.as_deref(), Some("reserved run_7"));
        assert!(item.last_attempt_ms.is_some());
    }

    #[test]
    fn an_appended_update_supersedes_rather_than_duplicating() {
        // `record` and `note_attempt` APPEND rather than rewriting the whole
        // queue, so the same id legitimately appears on several lines. Replay
        // therefore has to be last-wins; without that, one entry redriven twice
        // would come back after a restart as three separate rows, each offering
        // its own Redrive button for the same event.
        let dir = Dir::new("supersede");
        let id = {
            let mut q = DeadLetterQueue::open(dir.path());
            let id = q.record("p", "e", DeadReason::Shed, envelope("a")).id;
            q.note_attempt(&id, "first try".into());
            q.note_attempt(&id, "second try".into());
            id
        };

        // More lines on disk than entries — proof the appends really happened
        // and this is not testing a rewrite.
        let lines = std::fs::read_to_string(dir.path())
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert!(lines > 1, "expected appended lines, found {lines}");

        let q = DeadLetterQueue::open(dir.path());
        assert_eq!(q.len(), 1, "the same id must not replay as several entries");
        let item = q.get(&id).unwrap();
        assert_eq!(item.attempts, 2);
        assert_eq!(item.last_outcome.as_deref(), Some("second try"));
    }

    #[test]
    fn the_file_is_compacted_instead_of_growing_without_bound() {
        // Appending is O(1) per record, but replay is O(lines) — so the file has
        // to be compacted or every launch gets slower than the last.
        let dir = Dir::new("compact");
        let mut q = DeadLetterQueue::open(dir.path());
        let id = q.record("p", "e", DeadReason::Shed, envelope("a")).id;
        for i in 0..(COMPACT_MIN_LINES + 50) {
            q.note_attempt(&id, format!("attempt {i}"));
        }
        let lines = std::fs::read_to_string(dir.path())
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert!(
            lines <= COMPACT_MIN_LINES,
            "file should have been compacted, holds {lines} lines"
        );
        // …and compaction must not lose the state it was compacting.
        let reopened = DeadLetterQueue::open(dir.path());
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.get(&id).unwrap().attempts as usize, COMPACT_MIN_LINES + 50);
    }

    #[test]
    fn removing_is_durable() {
        let dir = Dir::new("remove");
        let id = {
            let mut q = DeadLetterQueue::open(dir.path());
            let id = q.record("p", "e", DeadReason::Shed, envelope("a")).id;
            assert!(q.remove(&id));
            assert!(!q.remove(&id), "removing twice must not claim success");
            id
        };
        // A resolved dead letter that reappears on restart would have someone
        // redriving the same event every morning.
        let q = DeadLetterQueue::open(dir.path());
        assert!(q.get(&id).is_none());
        assert!(q.is_empty());
    }

    #[test]
    fn a_redrive_key_is_derived_not_random() {
        // Same entry + same attempt = same key, so a double-click collapses to
        // one reservation instead of running the flow twice.
        assert_eq!(
            DeadLetterQueue::redrive_key("k1", 1),
            DeadLetterQueue::redrive_key("k1", 1)
        );
        // Different attempts must differ, or a second redrive would be swallowed
        // as a duplicate of the first.
        assert_ne!(
            DeadLetterQueue::redrive_key("k1", 1),
            DeadLetterQueue::redrive_key("k1", 2)
        );
        // And it must never collide with the original key.
        assert_ne!(DeadLetterQueue::redrive_key("k1", 0), "k1");
        assert!(!DeadLetterQueue::redrive_key("", 1).is_empty());
    }
}
