//! The run ledger: reserve, claim, finalise.
//!
//! This is where the Bridge Protocol's hard guarantees live. Everything else in
//! the bridge is plumbing around these three transitions.
//!
//! ```text
//!                                   ┌──────────► Cancelled
//!                                   │
//!   reserve ──► Queued ──► Running ─┼──────────► Succeeded
//!                   ▲              │
//!                   │              ├──────────► Failed
//!         claim (exactly once)     └──────────► TimedOut
//! ```
//!
//! Three invariants, each of which exists because its absence is expensive:
//!
//! 1. **Reserve is idempotent.** A run is recorded under a unique
//!    `idempotency_key` *before* anything executes. A duplicate request returns
//!    the existing run. Triggers fire from retried webhooks, several browser
//!    tabs, and at-least-once event delivery — without this, "the customer got
//!    three follow-up emails" is a routine Tuesday.
//!
//! 2. **Claim has exactly one winner.** `Queued -> Running` is a compare-and-set
//!    under the same lock that read the status. The loser is told it lost. This
//!    is what makes at-most-once execution a property of the ledger rather than a
//!    hope about how many workers happen to be running.
//!
//! 3. **Terminal states are immutable.** A second finalise is refused. A late
//!    reply from a cancelled run cannot overwrite the outcome a consumer already
//!    read and acted on.
//!
//! Plus the loop guards, which are conformance requirements rather than niceties:
//! flows emit terminal events (`flow.succeeded` and friends) and those events can
//! trigger flows, so without guards a two-binding cycle is an unbounded fork bomb
//! that presents to the user as a hung machine.
//!
//! # Durability
//!
//! In memory, optionally backed by an append-and-replay JOURNAL on disk
//! ([`Ledger::open`]). Every mutation appends the updated record as one JSON
//! line and fsyncs; on startup the journal is replayed (last write per run id
//! wins) and compacted. A dependency-free journal, not SQLite, because the
//! ledger is bounded and small and a bundled C library is the wrong weight for a
//! cross-platform tray app — and the write pattern is exactly the two operations
//! a SQL backend would run: an upsert per mutation, replayed as last-wins.
//!
//! Durability matters most for the STANDALONE consumer. FormLogic re-drives from
//! its own `flow_run_logs`, but a coding agent or an OpenAI-compatible client has
//! no ledger of its own — so a reboot mid-run must not leave the outcome
//! unknowable. On replay, a run left `Running` (its executor died with the app)
//! is finalised `Failed`/`runtime_unavailable`, so a consumer polling gets a
//! terminal state instead of a run stuck `running` forever, and a queued run
//! survives to be claimed.
//!
//! The `fired` loop-guard set is deliberately NOT persisted: its job is covered
//! by the idempotency keys, which ARE persisted (on the run records). A trigger
//! tree interrupted by a restart that re-fires a binding produces a Duplicate,
//! because the run it would create already exists on disk.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Where a run executes. Mirrors `protocol/v1` `runtime`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Desktop,
    Browser,
    Cli,
    Cloud,
}

/// Run lifecycle. Mirrors `protocol/v1/run-result.schema.json`.
///
/// Note `Succeeded`/`Failed` rather than FormLogic's `done`/`error`: the protocol
/// uses the past-tense pair throughout, and the conformance suite refuses `done`
/// so the two vocabularies cannot quietly coexist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    TimedOut,
    Cancelled,
}

impl RunStatus {
    /// Terminal states are immutable once reached.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            RunStatus::Succeeded | RunStatus::Failed | RunStatus::TimedOut | RunStatus::Cancelled
        )
    }
}

/// Closed error taxonomy. Mirrors `protocol/v1/error.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunErrorCode {
    InvalidRequest,
    InvalidFlow,
    FlowNotFound,
    /// The caller's grants do not cover it — a PERMISSION problem.
    CapabilityDenied,
    /// Granted, but not usable right now — an OPERATIONAL problem. Must carry
    /// actionable detail naming what to install or start.
    CapabilityUnavailable,
    ConnectionMissing,
    NodeFailed,
    Timeout,
    Cancelled,
    RuntimeUnavailable,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunError {
    pub code: RunErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    /// Defaulted on deserialize: the schema marks retryable optional, and
    /// requiring it made a caller's finish body fail to PARSE — surfacing as a
    /// 422 where the handler meant to answer 409. Found by the double-finish
    /// probe. Absent means false, the conservative direction.
    #[serde(default)]
    pub retryable: bool,
}

impl RunError {
    pub fn new(code: RunErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            detail: None,
            node_id: None,
            capability: None,
            // Conservative default: a caller that re-drives a non-retryable
            // failure just burns the ledger, so opting IN is the safe direction.
            retryable: false,
        }
    }

    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Who triggered this run, when it was triggered by another run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineageRef {
    /// The run at the top of this tree. Guard scope is per-tree.
    pub root_run_id: Option<String>,
    pub parent_run_id: Option<String>,
    /// The binding that fired. Needed for the self-retrigger and once-per-tree
    /// guards; a run with no binding is a direct invocation.
    pub binding_id: Option<String>,
    pub depth: u32,
}

/// Maximum trigger depth in one run tree.
///
/// 16 is deep enough for any legitimate fan-out chain and shallow enough that a
/// runaway cycle is capped in well under a second.
pub const MAX_LINEAGE_DEPTH: u32 = 16;

/// Cap on remembered `(root, binding, event)` triples, so a long-lived process
/// with millions of runs cannot grow the guard set without bound. Eviction is
/// FIFO by insertion — the oldest tree's guards go first, and a tree that old
/// has already terminated.
const GUARD_CAPACITY: usize = 8192;

/// Cap on retained runs. The ledger held every run forever — each carrying its
/// full input JSON — plus a by_key entry, so an event flood or a plugin looping
/// `flow.run` grew memory without limit until the tray process was OOM-killed,
/// taking every supervised plugin and service with it. Only TERMINAL runs are
/// evicted (oldest first); a queued or running run is never dropped, so nothing
/// in flight is lost. Evicting a terminal run means a much-later retry of its
/// idempotency key could re-execute — an acceptable trade against unbounded
/// growth, and the keys are the caller's to choose, so a live retry loop keeps
/// its recent runs.
const MAX_RUNS: usize = 20_000;

/// What a caller asks for. Mirrors `protocol/v1/run-request.schema.json`.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub caller_product: String,
    pub flow_id: Option<String>,
    pub inline_graph: bool,
    /// The flow's named inputs. Stored on the record because the worker that
    /// CLAIMS the run is not the caller that reserved it — without this, a
    /// claimed run would execute with no inputs and "succeed" on empty data,
    /// which is the quiet-wrong-output failure the protocol forbids.
    pub input: Option<serde_json::Value>,
    /// Wall-clock budget for execution. `None` = the runner's default.
    pub timeout_ms: Option<u64>,
    /// `sync` | `async` | `queued`. The desktop's own worker must NOT claim
    /// `queued` runs — those are reserved for an external claimer (a browser
    /// session, a remote worker). Auto-claiming them would race the very
    /// consumer that asked for reserve-without-execution.
    pub mode: String,
    pub correlation_id: String,
    pub idempotency_key: String,
    pub lineage: LineageRef,
    /// The event name that triggered this, when it came from a binding. Part of
    /// the once-per-tree guard key.
    pub trigger_event: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    pub run_id: String,
    pub status: RunStatus,
    pub caller_product: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_id: Option<String>,
    pub correlation_id: String,
    pub idempotency_key: String,
    /// Named flow inputs, carried so the claimer can execute. See [`RunRequest`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    pub mode: String,
    /// Set when a caller asked a RUNNING run to stop. The run stays `Running` —
    /// cancellation is a request, not a state — but the worker polls this flag
    /// and kills the child when it flips. Without recording it, `cancel` on a
    /// running run was acknowledged (202) and then went nowhere: no field
    /// existed for the runner to observe.
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub cancel_requested: bool,
    /// Convenience for HTTP callers: set only on the Duplicate path by the
    /// routes layer. Lives on the record so the wire shape has one source.
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub idempotent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Runtime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
    /// Stored as epoch millis; SERIALIZED as RFC 3339 under the schema's field
    /// names. The wire is the contract (`run-result.schema.json` says
    /// `startedAt: date-time`), and the first cut leaked the internal
    /// representation — `startedAtMs: 1785…` — which a validating consumer
    /// rejects outright under `additionalProperties: false`.
    #[serde(rename = "reservedAt", with = "iso_ms")]
    pub reserved_at_ms: u64,
    #[serde(
        rename = "startedAt",
        with = "iso_ms_opt",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub started_at_ms: Option<u64>,
    #[serde(
        rename = "finishedAt",
        with = "iso_ms_opt",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub finished_at_ms: Option<u64>,
    pub lineage: LineageRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_event: Option<String>,
}

/// Result of a reservation attempt.
#[derive(Debug)]
pub enum ReserveOutcome {
    /// A new run was recorded. Safe to execute.
    Reserved(RunRecord),
    /// This `idempotency_key` already exists. **Do not execute.** Return this
    /// record to the caller with `idempotent: true`.
    Duplicate(RunRecord),
    /// A loop guard refused the run. Not an error the caller did wrong — it is
    /// the system declining to run in a cycle.
    Refused { reason: String },
}

/// Result of a claim attempt.
#[derive(Debug)]
pub enum ClaimOutcome {
    /// This caller won. It — and only it — must execute the run.
    Claimed(RunRecord),
    /// Someone else already claimed it. HTTP 409.
    AlreadyClaimed { claimed_by: Option<String> },
    /// The run reached a terminal state before anyone claimed it.
    NotClaimable { status: RunStatus },
    Unknown,
}

/// Epoch-millis <-> RFC 3339, so the stored representation stays cheap while
/// the wire matches `format: date-time`.
mod iso_ms {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(ms: &u64, s: S) -> Result<S::Ok, S::Error> {
        match chrono::DateTime::from_timestamp_millis(*ms as i64) {
            Some(dt) => s.serialize_str(&dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
            None => Err(serde::ser::Error::custom("timestamp out of range")),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        let raw = String::deserialize(d)?;
        chrono::DateTime::parse_from_rfc3339(&raw)
            .map(|dt| dt.timestamp_millis() as u64)
            .map_err(serde::de::Error::custom)
    }
}

mod iso_ms_opt {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(ms: &Option<u64>, s: S) -> Result<S::Ok, S::Error> {
        match ms {
            Some(v) => super::iso_ms::serialize(v, s),
            // skip_serializing_if handles None; reaching here means a caller
            // disabled the skip, and null is the honest value.
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
        let raw = Option::<String>::deserialize(d)?;
        match raw {
            None => Ok(None),
            Some(text) => chrono::DateTime::parse_from_rfc3339(&text)
                .map(|dt| Some(dt.timestamp_millis() as u64))
                .map_err(serde::de::Error::custom),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Monotonic-ish run id. Not a UUID: the ledger's uniqueness guarantee comes from
/// `idempotency_key`, and a sortable id makes run history readable.
fn mint_run_id(seq: u64) -> String {
    format!("run_{:013x}_{:x}", now_ms(), seq)
}

/// Compact the journal once it exceeds `max(this, 2 x live runs)` lines. Two,
/// not more: a run appends ~3 lines over its life (reserve/claim/finish), so a
/// higher factor would let the file outgrow the live set faster than compaction
/// reclaims it.
/// long-lived process does not accumulate an unbounded rewrite history.
const COMPACT_MIN_LINES: usize = 512;

/// Append-only durability for the ledger. One JSON line per mutation (the full,
/// updated record); replay is last-write-wins per run id.
struct Journal {
    path: PathBuf,
    file: std::fs::File,
    /// Lines written since the last compaction — the compaction trigger.
    lines: usize,
}

impl Journal {
    fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self { path, file, lines: 0 })
    }

    /// Append one record and flush it to the OS. fsync per write is affordable at
    /// a desktop's run rate (a few writes per run) and keeps the on-disk state
    /// honest — a lost write here is a run a consumer might never learn the fate
    /// of. Errors are surfaced to the caller, which logs; a journal that cannot
    /// be written must not take the in-memory ledger down with it.
    fn append(&mut self, rec: &RunRecord) -> std::io::Result<()> {
        let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.sync_all()?;
        self.lines += 1;
        Ok(())
    }

    /// Rewrite the journal to exactly `records`, atomically (`.tmp` + rename), so
    /// evicted/duplicate/orphan-fixed lines are dropped and a crash mid-compaction
    /// cannot leave a half-written file that replays as corruption.
    fn compact(&mut self, records: &[RunRecord]) -> std::io::Result<()> {
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            for rec in records {
                let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        // Reopen the append handle onto the freshly-rotated file.
        self.file = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        self.lines = records.len();
        Ok(())
    }
}

#[derive(Default)]
pub struct Ledger {
    runs: HashMap<String, RunRecord>,
    /// `idempotency_key` -> `run_id`. The uniqueness gate.
    by_key: HashMap<String, String>,
    /// Run ids in reservation order, for bounded eviction of terminal runs.
    run_order: VecDeque<String>,
    /// `(root_run_id, binding_id, event)` triples already fired in a tree.
    fired: HashSet<(String, String, String)>,
    /// Insertion order over `fired`, for bounded FIFO eviction.
    fired_order: VecDeque<(String, String, String)>,
    seq: u64,
    /// On-disk durability. `None` for an in-memory ledger (tests, the default);
    /// `Some` after [`Ledger::open`].
    journal: Option<Journal>,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a durable ledger backed by a journal at `path`.
    ///
    /// Replays the existing journal (last write per run id wins), fixes up
    /// orphaned runs, then compacts — so the on-disk file after `open` reflects
    /// exactly the loaded, capped, fixed state, dropping the replay history. A
    /// journal that fails to open is a hard error the caller decides on (the
    /// build wiring falls back to an in-memory ledger with a log, so a
    /// filesystem problem degrades durability rather than blocking startup).
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        let mut ledger = Ledger::new();

        // Replay: collect the last record per run id, then insert in reservation
        // order so the cap evicts the oldest terminal runs exactly as at runtime.
        if let Ok(text) = std::fs::read_to_string(&path) {
            let mut latest: HashMap<String, RunRecord> = HashMap::new();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<RunRecord>(line) {
                    // Last line for a run id wins — that is the upsert semantics.
                    Ok(rec) => {
                        latest.insert(rec.run_id.clone(), rec);
                    }
                    // A single corrupt line (a torn final write, say) must not
                    // discard the whole ledger. Skip it and keep the rest.
                    Err(_) => continue,
                }
            }
            let mut records: Vec<RunRecord> = latest.into_values().collect();
            records.sort_by_key(|r| r.reserved_at_ms);
            for mut rec in records {
                // A run left Running means its executor died with the app — the
                // result was never recorded. Finalise it so a consumer polling
                // gets a terminal answer rather than a run stuck Running forever.
                if rec.status == RunStatus::Running {
                    rec.status = RunStatus::Failed;
                    rec.finished_at_ms = Some(now_ms());
                    rec.error = Some(
                        RunError::new(
                            RunErrorCode::RuntimeUnavailable,
                            "OAIY Desktop restarted while this run was executing; its result was not recorded.",
                        )
                        .retryable(),
                    );
                }
                ledger.restore(rec);
            }
            ledger.evict_terminal_over_cap();
        }

        // Attach the journal and rewrite it to the loaded state in one shot, so
        // the replay history + orphan fixes are flattened to a clean baseline.
        let mut journal = Journal::open(path)?;
        let snapshot: Vec<RunRecord> = ledger.runs.values().cloned().collect();
        journal.compact(&snapshot)?;
        ledger.journal = Some(journal);
        Ok(ledger)
    }

    /// Insert a record loaded from the journal, bypassing reserve()'s guards and
    /// dedupe (the record already exists — replaying it must not re-run the loop
    /// guards or mint a new id).
    fn restore(&mut self, rec: RunRecord) {
        self.by_key.insert(rec.idempotency_key.clone(), rec.run_id.clone());
        self.run_order.push_back(rec.run_id.clone());
        self.runs.insert(rec.run_id.clone(), rec);
    }

    /// Append `rec` to the journal (if durable) and compact when it has grown.
    fn persist(&mut self, rec: &RunRecord) {
        let live = self.runs.len();
        let compact_now = match self.journal.as_mut() {
            None => return,
            Some(j) => {
                if let Err(e) = j.append(rec) {
                    eprintln!("[ledger] journal append failed (durability degraded): {e}");
                }
                j.lines > COMPACT_MIN_LINES.max(live.saturating_mul(2))
            }
        };
        if compact_now {
            let snapshot: Vec<RunRecord> = self.runs.values().cloned().collect();
            if let Some(j) = self.journal.as_mut() {
                if let Err(e) = j.compact(&snapshot) {
                    eprintln!("[ledger] journal compaction failed: {e}");
                }
            }
        }
    }

    /// Record a run before executing it.
    ///
    /// The order here is deliberate: guards are evaluated BEFORE the uniqueness
    /// check, so a refused cycle never consumes an idempotency key. Otherwise a
    /// guard refusal would poison that key and a legitimate retry of the same
    /// logical event could never run.
    pub fn reserve(&mut self, req: &RunRequest) -> ReserveOutcome {
        // Duplicates BEFORE guards. A retried triggered request — same key, same
        // lineage, redelivered webhook — used to hit guard 3 first ("this
        // binding already handled this event") and get a Refused, which the HTTP
        // layer maps to 422 "stop retrying". The caller then never learned the
        // original runId, so the run's outcome was unreachable to the very
        // consumer that triggered it. The review caught it: the duplicate-key
        // tests all used empty lineage, which disables guard 3. A retry of an
        // already-reserved request is by definition a Duplicate, whatever its
        // lineage says. Neither ordering consumes the key on refusal, so the
        // guard-refusal-does-not-poison-the-key property is unchanged.
        if let Some(existing_id) = self.by_key.get(&req.idempotency_key) {
            if let Some(rec) = self.runs.get(existing_id) {
                return ReserveOutcome::Duplicate(rec.clone());
            }
        }

        if let Some(reason) = self.guard_violation(req) {
            return ReserveOutcome::Refused { reason };
        }

        self.seq += 1;
        let run_id = mint_run_id(self.seq);
        let record = RunRecord {
            run_id: run_id.clone(),
            status: RunStatus::Queued,
            caller_product: req.caller_product.clone(),
            flow_id: req.flow_id.clone(),
            correlation_id: req.correlation_id.clone(),
            idempotency_key: req.idempotency_key.clone(),
            input: req.input.clone(),
            timeout_ms: req.timeout_ms,
            mode: req.mode.clone(),
            cancel_requested: false,
            runtime: None,
            claimed_by: None,
            output: None,
            error: None,
            idempotent: false,
            reserved_at_ms: now_ms(),
            started_at_ms: None,
            finished_at_ms: None,
            lineage: req.lineage.clone(),
            trigger_event: req.trigger_event.clone(),
        };

        // Note the guard is recorded on RESERVE, not on completion. A run that is
        // reserved and then never claimed still counts as "this binding fired for
        // this event" — otherwise a crash between reserve and claim would let the
        // same event re-fire the same binding forever.
        self.record_fired(req);

        self.by_key
            .insert(req.idempotency_key.clone(), run_id.clone());
        self.run_order.push_back(run_id.clone());
        self.runs.insert(run_id, record.clone());
        self.evict_terminal_over_cap();
        self.persist(&record);
        ReserveOutcome::Reserved(record)
    }

    /// Drop oldest TERMINAL runs until under [`MAX_RUNS`]. A queued/running run is
    /// never evicted — losing an in-flight run would strand its caller — so the
    /// effective ceiling can exceed the cap when many runs are simultaneously
    /// live, which is self-limiting (they terminate) and far better than OOM.
    fn evict_terminal_over_cap(&mut self) {
        if self.runs.len() <= MAX_RUNS {
            return;
        }
        let mut scanned = 0usize;
        let budget = self.run_order.len();
        while self.runs.len() > MAX_RUNS && scanned < budget {
            let Some(id) = self.run_order.pop_front() else { break };
            scanned += 1;
            match self.runs.get(&id) {
                Some(rec) if rec.status.is_terminal() => {
                    let key = rec.idempotency_key.clone();
                    self.runs.remove(&id);
                    // Only clear by_key if it still points at THIS run — a later
                    // reservation could in principle reuse the string.
                    if self.by_key.get(&key).map(String::as_str) == Some(id.as_str()) {
                        self.by_key.remove(&key);
                    }
                }
                // Still live: keep it, and put it at the back so the scan makes
                // progress instead of re-examining it forever.
                Some(_) => self.run_order.push_back(id),
                None => {} // already gone
            }
        }
    }

    /// The three loop guards. `None` means the run may proceed.
    fn guard_violation(&self, req: &RunRequest) -> Option<String> {
        // Guard 1: depth. The blunt backstop that bounds everything else.
        if req.lineage.depth > MAX_LINEAGE_DEPTH {
            return Some(format!(
                "trigger depth {} exceeds the maximum of {MAX_LINEAGE_DEPTH}",
                req.lineage.depth
            ));
        }

        let Some(binding_id) = req.lineage.binding_id.as_deref() else {
            // A direct invocation has no binding, so neither remaining guard
            // applies — they are both about a binding re-firing.
            return None;
        };

        // Guard 2: a run never re-triggers the binding that produced it. This is
        // the tight self-loop — a flow whose success event is handled by the same
        // binding that started it.
        if let Some(parent) = req.lineage.parent_run_id.as_deref() {
            if let Some(parent_rec) = self.runs.get(parent) {
                if parent_rec.lineage.binding_id.as_deref() == Some(binding_id) {
                    return Some(format!(
                        "binding {binding_id} produced run {parent} and cannot handle its own outcome"
                    ));
                }
            }
        }

        // Guard 3: each (tree, binding, event) fires at most once. Catches the
        // longer cycles guard 2 cannot see — A triggers B triggers A — without
        // waiting for the depth cap to stop them 16 runs later.
        if let (Some(root), Some(event)) =
            (req.lineage.root_run_id.as_deref(), req.trigger_event.as_deref())
        {
            let key = (root.to_string(), binding_id.to_string(), event.to_string());
            if self.fired.contains(&key) {
                return Some(format!(
                    "binding {binding_id} already handled {event} in this run tree"
                ));
            }
        }

        None
    }

    fn record_fired(&mut self, req: &RunRequest) {
        let (Some(root), Some(binding), Some(event)) = (
            req.lineage.root_run_id.as_deref(),
            req.lineage.binding_id.as_deref(),
            req.trigger_event.as_deref(),
        ) else {
            return;
        };
        let key = (root.to_string(), binding.to_string(), event.to_string());
        if self.fired.insert(key.clone()) {
            self.fired_order.push_back(key);
            while self.fired_order.len() > GUARD_CAPACITY {
                if let Some(old) = self.fired_order.pop_front() {
                    self.fired.remove(&old);
                }
            }
        }
    }

    /// Transition `Queued -> Running`, exactly once.
    ///
    /// The compare-and-set happens under the caller's `&mut self`, i.e. inside
    /// the same lock that read the status. Splitting the read and the write is
    /// precisely the race this exists to prevent.
    pub fn claim(&mut self, run_id: &str, runtime: Runtime, worker: &str) -> ClaimOutcome {
        let outcome = {
            let Some(rec) = self.runs.get_mut(run_id) else {
                return ClaimOutcome::Unknown;
            };
            match rec.status {
                RunStatus::Queued => {
                    rec.status = RunStatus::Running;
                    rec.runtime = Some(runtime);
                    rec.claimed_by = Some(worker.to_string());
                    rec.started_at_ms = Some(now_ms());
                    ClaimOutcome::Claimed(rec.clone())
                }
                RunStatus::Running => ClaimOutcome::AlreadyClaimed {
                    claimed_by: rec.claimed_by.clone(),
                },
                other => ClaimOutcome::NotClaimable { status: other },
            }
        };
        // Persist only the winning transition — a lost claim changed nothing.
        if let ClaimOutcome::Claimed(rec) = &outcome {
            self.persist(rec);
        }
        outcome
    }

    /// Move a RUNNING run to a terminal state. Refuses anything else: a run that
    /// is already terminal, and a run nobody has claimed.
    ///
    /// `Err` here is not "the run failed" — it is "this finalise was rejected",
    /// which the HTTP layer maps to 409. Conflating the two would let a late
    /// worker reply overwrite an outcome a consumer already acted on.
    pub fn finish(
        &mut self,
        run_id: &str,
        status: RunStatus,
        output: Option<serde_json::Value>,
        error: Option<RunError>,
    ) -> Result<RunRecord, String> {
        if !status.is_terminal() {
            return Err(format!("{status:?} is not a terminal status"));
        }
        // The protocol requires a failure to carry an error. Enforcing it here
        // rather than trusting callers means a `failed` run can never reach a
        // consumer with nothing to display.
        if matches!(status, RunStatus::Failed | RunStatus::TimedOut) && error.is_none() {
            return Err(format!("{status:?} must carry an error"));
        }
        let updated = {
            let Some(rec) = self.runs.get_mut(run_id) else {
                return Err(format!("unknown run {run_id}"));
            };
            if rec.status.is_terminal() {
                return Err(format!(
                    "run {run_id} is already {:?}; terminal states are immutable",
                    rec.status
                ));
            }
            // Only an executing run can end. Refusing a QUEUED one closes a hole
            // the terminal check alone left open: a caller that never claimed the
            // run could journal an outcome for work nothing had run, and because
            // the record was then terminal it dropped out of
            // `claimable_by_worker` — so nothing ever would. `protocol/README.md`
            // draws every terminal edge off `running`; the single exception,
            // `queued -> cancelled`, belongs to `request_cancel` and is
            // implemented there. Cancellation of a running run deliberately
            // leaves the status Running, so the worker can still finalise it.
            if rec.status != RunStatus::Running {
                return Err(format!(
                    "run {run_id} is {:?}, not Running; only a claimed run can be finalised",
                    rec.status
                ));
            }
            rec.status = status;
            rec.output = output;
            rec.error = error;
            rec.finished_at_ms = Some(now_ms());
            rec.clone()
        };
        self.persist(&updated);
        Ok(updated)
    }

    /// Request cancellation.
    ///
    /// A queued run cancels immediately — nothing is executing. A running one is
    /// only *asked*: a node mid-HTTP-call does not stop on command, so the runner
    /// finalises it when it notices. Reporting a running run as already cancelled
    /// would tell the consumer work had stopped when it had not.
    pub fn request_cancel(&mut self, run_id: &str) -> Result<RunStatus, String> {
        let (status, updated) = {
            let Some(rec) = self.runs.get_mut(run_id) else {
                return Err(format!("unknown run {run_id}"));
            };
            match rec.status {
                RunStatus::Queued => {
                    rec.status = RunStatus::Cancelled;
                    rec.finished_at_ms = Some(now_ms());
                    rec.error = Some(RunError::new(
                        RunErrorCode::Cancelled,
                        "Cancelled before it started.",
                    ));
                    (RunStatus::Cancelled, rec.clone())
                }
                RunStatus::Running => {
                    // The cancel REQUEST is durable too: a reboot between the
                    // request and the runner noticing must not lose it, or a
                    // consumer's cancel would silently do nothing after a restart.
                    rec.cancel_requested = true;
                    (RunStatus::Running, rec.clone())
                }
                other => return Err(format!("run {run_id} is already {other:?}")),
            }
        };
        self.persist(&updated);
        Ok(status)
    }

    pub fn get(&self, run_id: &str) -> Option<RunRecord> {
        self.runs.get(run_id).cloned()
    }

    /// Runs the DESKTOP'S OWN worker may claim: queued status, and not
    /// `mode: "queued"` — those are reserved for an external claimer, and
    /// auto-claiming them would race the very consumer that asked for
    /// reserve-without-execution.
    pub fn claimable_by_worker(&self, limit: usize) -> Vec<RunRecord> {
        let mut q: Vec<_> = self
            .runs
            .values()
            .filter(|r| r.status == RunStatus::Queued && r.mode != "queued")
            .cloned()
            .collect();
        q.sort_by_key(|r| r.reserved_at_ms);
        q.truncate(limit);
        q
    }

    /// Claimable runs, oldest first — a worker polls this then claims.
    pub fn queued(&self, limit: usize) -> Vec<RunRecord> {
        let mut q: Vec<_> = self
            .runs
            .values()
            .filter(|r| r.status == RunStatus::Queued)
            .cloned()
            .collect();
        q.sort_by_key(|r| r.reserved_at_ms);
        q.truncate(limit);
        q
    }

    /// History, newest first — what a human needs to answer "why did that fail?"
    ///
    /// Everything the ledger keeps (up to [`MAX_RUNS`], and durable across
    /// restarts) was already there; until this there was simply no way to reach
    /// it without knowing a run id, which meant a failed run was only diagnosable
    /// by whoever happened to be holding its id at the time.
    ///
    /// `statuses` empty means every state. Sorted by reservation time rather than
    /// by `run_order`, which eviction reorders.
    pub fn recent(&self, limit: usize, statuses: &[RunStatus]) -> Vec<RunRecord> {
        let mut q: Vec<_> = self
            .runs
            .values()
            .filter(|r| statuses.is_empty() || statuses.contains(&r.status))
            .cloned()
            .collect();
        q.sort_by(|a, b| b.reserved_at_ms.cmp(&a.reserved_at_ms));
        q.truncate(limit);
        q
    }

    /// Per-status totals over everything retained, so a history view can say how
    /// many runs failed without paging the whole ledger to count them.
    pub fn status_counts(&self) -> HashMap<RunStatus, usize> {
        let mut counts = HashMap::new();
        for rec in self.runs.values() {
            *counts.entry(rec.status).or_insert(0) += 1;
        }
        counts
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// Shared handle. Matches the `services::*Handle` convention in this crate.
pub type LedgerHandle = Arc<Mutex<Ledger>>;

pub fn new_handle() -> LedgerHandle {
    Arc::new(Mutex::new(Ledger::new()))
}

/// A durable ledger handle backed by a journal at `path`, or an in-memory one
/// with a log if the journal cannot be opened — a filesystem problem degrades
/// durability, it does not block the runtime from starting.
pub fn open_handle(path: PathBuf) -> LedgerHandle {
    match Ledger::open(path.clone()) {
        Ok(l) => Arc::new(Mutex::new(l)),
        Err(e) => {
            eprintln!(
                "[ledger] could not open the durable journal at {} ({e}); running in memory (runs will not survive a restart)",
                path.display()
            );
            new_handle()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(key: &str) -> RunRequest {
        RunRequest {
            caller_product: "formlogic".into(),
            flow_id: Some("caller-lookup".into()),
            inline_graph: false,
            input: Some(serde_json::json!({"phone": "+61400000000"})),
            timeout_ms: Some(30_000),
            mode: "async".into(),
            correlation_id: "call_abc".into(),
            idempotency_key: key.into(),
            lineage: LineageRef::default(),
            trigger_event: None,
        }
    }

    // --- durability -------------------------------------------------------

    /// A scratch journal path, unique per test, cleaned on drop.
    struct JournalDir(std::path::PathBuf);
    impl JournalDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("oaiy-ledger-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> std::path::PathBuf {
            self.0.join("ledger.jsonl")
        }
    }
    impl Drop for JournalDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_queued_run_survives_a_reopen() {
        // The durability win: a run reserved before a restart is still there,
        // still Queued, still claimable, after the process comes back.
        let d = JournalDir::new("survive");
        let run_id = {
            let mut l = Ledger::open(d.path()).unwrap();
            match l.reserve(&req("k1")) {
                ReserveOutcome::Reserved(r) => r.run_id,
                o => panic!("{o:?}"),
            }
        };
        let l2 = Ledger::open(d.path()).unwrap();
        let rec = l2.get(&run_id).expect("the run must survive the reopen");
        assert_eq!(rec.status, RunStatus::Queued);
        assert_eq!(rec.idempotency_key, "k1");
        // Its input survived too — a claimer must be able to execute it.
        assert_eq!(rec.input, Some(serde_json::json!({"phone": "+61400000000"})));
    }

    #[test]
    fn idempotency_survives_a_reopen() {
        // A retry of a completed run's key after a restart must NOT re-execute —
        // this is the property a standalone consumer relies on for exactly-once.
        let d = JournalDir::new("idem");
        let original;
        {
            let mut l = Ledger::open(d.path()).unwrap();
            original = match l.reserve(&req("dup")) {
                ReserveOutcome::Reserved(r) => r.run_id,
                o => panic!("{o:?}"),
            };
            l.claim(&original, Runtime::Desktop, "d").ok_claimed();
            l.finish(&original, RunStatus::Succeeded, Some(serde_json::json!({"ok": true})), None)
                .unwrap();
        }
        let mut l2 = Ledger::open(d.path()).unwrap();
        match l2.reserve(&req("dup")) {
            ReserveOutcome::Duplicate(r) => {
                assert_eq!(r.run_id, original);
                assert_eq!(r.status, RunStatus::Succeeded, "the completed outcome survived");
                assert_eq!(r.output, Some(serde_json::json!({"ok": true})));
            }
            o => panic!("a reopened key must dedupe, got {o:?}"),
        }
    }

    #[test]
    fn a_run_left_running_is_finalised_on_reload() {
        // Its executor died with the app. A consumer polling must get a terminal
        // answer, not a run stuck Running forever.
        let d = JournalDir::new("orphan");
        let run_id = {
            let mut l = Ledger::open(d.path()).unwrap();
            let r = match l.reserve(&req("k")) {
                ReserveOutcome::Reserved(r) => r,
                o => panic!("{o:?}"),
            };
            l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
            // Deliberately NOT finished — simulate the process dying here.
            r.run_id
        };
        let l2 = Ledger::open(d.path()).unwrap();
        let rec = l2.get(&run_id).unwrap();
        assert_eq!(rec.status, RunStatus::Failed);
        let err = rec.error.expect("a failed run must carry an error");
        assert_eq!(err.code, RunErrorCode::RuntimeUnavailable);
        assert!(err.retryable, "the consumer may re-drive it");
        assert!(err.message.contains("restarted"), "{}", err.message);
    }

    #[test]
    fn a_terminal_run_reloads_unchanged() {
        let d = JournalDir::new("terminal");
        let run_id = {
            let mut l = Ledger::open(d.path()).unwrap();
            let r = match l.reserve(&req("k")) {
                ReserveOutcome::Reserved(r) => r,
                o => panic!("{o:?}"),
            };
            l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
            l.finish(&r.run_id, RunStatus::Succeeded, Some(serde_json::json!(42)), None).unwrap();
            r.run_id
        };
        let rec = Ledger::open(d.path()).unwrap().get(&run_id).unwrap();
        assert_eq!(rec.status, RunStatus::Succeeded, "a finished run is not re-orphaned");
        assert_eq!(rec.output, Some(serde_json::json!(42)));
    }

    #[test]
    fn a_pending_cancel_survives_a_reopen() {
        // A consumer's cancel of a running run must not silently vanish on a
        // restart before the runner noticed.
        let d = JournalDir::new("cancel");
        let run_id = {
            let mut l = Ledger::open(d.path()).unwrap();
            let r = match l.reserve(&req("k")) {
                ReserveOutcome::Reserved(r) => r,
                o => panic!("{o:?}"),
            };
            l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
            assert_eq!(l.request_cancel(&r.run_id).unwrap(), RunStatus::Running);
            r.run_id
        };
        // On reload the run is orphan-finalised (Failed), but the reopen must not
        // panic and the record is present — the cancel intent was durably written.
        let rec = Ledger::open(d.path()).unwrap().get(&run_id).unwrap();
        assert!(rec.status.is_terminal());
    }

    #[test]
    fn the_journal_is_compacted_and_stays_bounded() {
        // Many mutations must not grow the file without bound: compaction keeps
        // it proportional to the LIVE run set, not the mutation history.
        let d = JournalDir::new("compact");
        {
            let mut l = Ledger::open(d.path()).unwrap();
            // Each run does 3 mutations (reserve/claim/finish) = 3 lines. 600
            // runs = 1800 writes, well past COMPACT_MIN_LINES.
            for i in 0..600 {
                let key = format!("k{i}");
                let r = match l.reserve(&req(&key)) {
                    ReserveOutcome::Reserved(r) => r,
                    o => panic!("{o:?}"),
                };
                l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
                l.finish(&r.run_id, RunStatus::Succeeded, Some(serde_json::json!(i)), None).unwrap();
            }
        }
        // During the run, in-flight compaction kept the file well under the 1800
        // mutations — proving it does not grow with mutation HISTORY.
        let during = std::fs::read_to_string(d.path()).unwrap().lines().count();
        assert!(
            during < 1800,
            "journal grew with mutation history ({during} lines for 1800 mutations)"
        );

        // The strongest bound: a fresh open() compacts to exactly one line per
        // live run — O(live), not O(mutations).
        let live = {
            let l2 = Ledger::open(d.path()).unwrap();
            l2.len()
        };
        let after = std::fs::read_to_string(d.path()).unwrap().lines().count();
        assert_eq!(after, live, "open() must compact to one line per live run");
        assert!(live > 0 && live <= 600);
    }

    #[test]
    fn a_corrupt_line_does_not_discard_the_whole_ledger() {
        // A torn final write (power loss mid-append) must lose only that line.
        let d = JournalDir::new("corrupt");
        let good_id;
        {
            let mut l = Ledger::open(d.path()).unwrap();
            good_id = match l.reserve(&req("good")) {
                ReserveOutcome::Reserved(r) => r.run_id,
                o => panic!("{o:?}"),
            };
        }
        // Append a garbage line, as a half-flushed write would leave.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(d.path()).unwrap();
            f.write_all(b"{ this is not a complete record\n").unwrap();
        }
        let l2 = Ledger::open(d.path()).unwrap();
        assert!(l2.get(&good_id).is_some(), "the good run must survive a corrupt neighbour");
    }

    #[test]
    fn an_in_memory_ledger_has_no_journal() {
        // The default path stays pure — no file IO, so the other tests and the
        // hot path pay nothing.
        let mut l = Ledger::new();
        assert!(l.journal.is_none());
        // And mutations still work without a journal.
        assert!(matches!(l.reserve(&req("k")), ReserveOutcome::Reserved(_)));
    }

    // --- invariant 1: reserve is idempotent -------------------------------

    #[test]
    fn reserve_records_a_queued_run() {
        let mut l = Ledger::new();
        match l.reserve(&req("k1")) {
            ReserveOutcome::Reserved(r) => {
                assert_eq!(r.status, RunStatus::Queued);
                assert!(r.started_at_ms.is_none(), "queued runs have not started");
            }
            other => panic!("expected Reserved, got {other:?}"),
        }
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn the_same_key_never_executes_twice() {
        let mut l = Ledger::new();
        let first = match l.reserve(&req("dup")) {
            ReserveOutcome::Reserved(r) => r,
            other => panic!("expected Reserved, got {other:?}"),
        };
        match l.reserve(&req("dup")) {
            ReserveOutcome::Duplicate(r) => assert_eq!(r.run_id, first.run_id),
            other => panic!("a repeated idempotency key must be a Duplicate, got {other:?}"),
        }
        assert_eq!(l.len(), 1, "a duplicate must not create a second run");
    }

    #[test]
    fn different_keys_are_different_runs() {
        let mut l = Ledger::new();
        l.reserve(&req("a"));
        l.reserve(&req("b"));
        assert_eq!(l.len(), 2);
    }

    // --- invariant 2: exactly one claimer ---------------------------------

    #[test]
    fn only_one_worker_can_claim_a_run() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };

        match l.claim(&r.run_id, Runtime::Desktop, "desk-1") {
            ClaimOutcome::Claimed(rec) => {
                assert_eq!(rec.status, RunStatus::Running);
                assert_eq!(rec.claimed_by.as_deref(), Some("desk-1"));
                assert!(rec.started_at_ms.is_some());
            }
            o => panic!("first claim must win, got {o:?}"),
        }

        match l.claim(&r.run_id, Runtime::Browser, "tab-2") {
            ClaimOutcome::AlreadyClaimed { claimed_by } => {
                assert_eq!(claimed_by.as_deref(), Some("desk-1"), "names the winner");
            }
            o => panic!("second claim must lose, got {o:?}"),
        }
    }

    #[test]
    fn a_terminal_run_is_not_claimable() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
        l.finish(&r.run_id, RunStatus::Succeeded, Some(serde_json::json!({})), None)
            .unwrap();
        match l.claim(&r.run_id, Runtime::Desktop, "d2") {
            ClaimOutcome::NotClaimable { status } => assert_eq!(status, RunStatus::Succeeded),
            o => panic!("expected NotClaimable, got {o:?}"),
        }
    }

    #[test]
    fn claiming_an_unknown_run_is_not_a_panic() {
        let mut l = Ledger::new();
        assert!(matches!(
            l.claim("run_nope", Runtime::Desktop, "d"),
            ClaimOutcome::Unknown
        ));
    }

    // --- invariant 3: terminal states are immutable -----------------------

    #[test]
    fn a_second_finalise_is_refused() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
        l.finish(
            &r.run_id,
            RunStatus::Succeeded,
            Some(serde_json::json!({"ok": true})),
            None,
        )
        .expect("first finalise succeeds");

        let second = l.finish(
            &r.run_id,
            RunStatus::Failed,
            None,
            Some(RunError::new(RunErrorCode::NodeFailed, "late failure")),
        );
        assert!(second.is_err(), "a late reply must not overwrite the outcome");

        let rec = l.get(&r.run_id).unwrap();
        assert_eq!(rec.status, RunStatus::Succeeded);
        assert_eq!(rec.output, Some(serde_json::json!({"ok": true})));
    }

    #[test]
    fn a_failure_must_carry_an_error() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
        assert!(
            l.finish(&r.run_id, RunStatus::Failed, None, None).is_err(),
            "a failed run with nothing to display is useless to a consumer"
        );
        assert!(l
            .finish(&r.run_id, RunStatus::TimedOut, None, None)
            .is_err());
        assert!(l
            .finish(
                &r.run_id,
                RunStatus::Failed,
                None,
                Some(RunError::new(RunErrorCode::Timeout, "took too long"))
            )
            .is_ok());
    }

    #[test]
    fn finish_refuses_a_non_terminal_status() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        assert!(l.finish(&r.run_id, RunStatus::Running, None, None).is_err());
    }

    #[test]
    fn only_a_claimed_run_can_be_finalised() {
        // The write-side half of the single-winner claim. Without it the claim
        // bought nothing: anyone reaching /finish could journal an outcome for a
        // run that had never executed, and the now-terminal record dropped out of
        // claimable_by_worker, so nothing ever would execute it.
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        let why = l
            .finish(
                &r.run_id,
                RunStatus::Succeeded,
                Some(serde_json::json!({"fabricated": true})),
                None,
            )
            .expect_err("a queued run has no result to record");
        assert!(why.contains("not Running"), "{why}");

        // Untouched, and still there for a worker to pick up.
        let rec = l.get(&r.run_id).unwrap();
        assert_eq!(rec.status, RunStatus::Queued);
        assert!(rec.output.is_none(), "no outcome may be invented");
        assert!(rec.finished_at_ms.is_none());
        assert_eq!(l.queued(10).len(), 1, "the run must stay claimable");

        // And once it IS claimed, the same finalise lands.
        l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
        assert!(l
            .finish(&r.run_id, RunStatus::Succeeded, Some(serde_json::json!(1)), None)
            .is_ok());
    }

    // --- cancellation ------------------------------------------------------

    #[test]
    fn cancelling_a_queued_run_is_immediate() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        assert_eq!(l.request_cancel(&r.run_id).unwrap(), RunStatus::Cancelled);
        assert_eq!(l.get(&r.run_id).unwrap().status, RunStatus::Cancelled);
    }

    #[test]
    fn cancelling_a_running_run_is_only_a_request() {
        let mut l = Ledger::new();
        let r = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&r.run_id, Runtime::Desktop, "d").ok_claimed();
        // Still Running: claiming otherwise would tell the consumer the work had
        // stopped when a node may be mid-HTTP-call.
        assert_eq!(l.request_cancel(&r.run_id).unwrap(), RunStatus::Running);
        assert_eq!(l.get(&r.run_id).unwrap().status, RunStatus::Running);
        // Which is also what lets the runner finalise it after killing the child:
        // finish() requires Running, and a cancel request deliberately does not
        // leave that state.
        assert!(l
            .finish(&r.run_id, RunStatus::Cancelled, None, None)
            .is_ok());
    }

    // --- loop guards -------------------------------------------------------

    fn triggered(key: &str, root: &str, parent: &str, binding: &str, event: &str, depth: u32) -> RunRequest {
        RunRequest {
            caller_product: "formlogic".into(),
            flow_id: Some("handler".into()),
            inline_graph: false,
            input: None,
            timeout_ms: None,
            mode: "async".into(),
            correlation_id: "c".into(),
            idempotency_key: key.into(),
            lineage: LineageRef {
                root_run_id: Some(root.into()),
                parent_run_id: Some(parent.into()),
                binding_id: Some(binding.into()),
                depth,
            },
            trigger_event: Some(event.into()),
        }
    }

    #[test]
    fn depth_is_capped() {
        let mut l = Ledger::new();
        let at_cap = triggered("k1", "root", "p", "b1", "flow.succeeded", MAX_LINEAGE_DEPTH);
        assert!(matches!(l.reserve(&at_cap), ReserveOutcome::Reserved(_)));

        let over = triggered("k2", "root", "p", "b2", "flow.succeeded", MAX_LINEAGE_DEPTH + 1);
        match l.reserve(&over) {
            ReserveOutcome::Refused { reason } => assert!(reason.contains("depth")),
            o => panic!("depth {} must be refused, got {o:?}", MAX_LINEAGE_DEPTH + 1),
        }
    }

    #[test]
    fn a_binding_cannot_handle_its_own_outcome() {
        let mut l = Ledger::new();
        // A run produced BY binding b1...
        let parent = match l.reserve(&triggered("p", "root", "root", "b1", "aokie.call.ended", 1)) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        // ...must not be handled by b1 again.
        let child = triggered("c", "root", &parent.run_id, "b1", "flow.succeeded", 2);
        match l.reserve(&child) {
            ReserveOutcome::Refused { reason } => assert!(reason.contains("b1"), "{reason}"),
            o => panic!("a self-handling binding must be refused, got {o:?}"),
        }
    }

    #[test]
    fn a_binding_event_pair_fires_once_per_tree() {
        let mut l = Ledger::new();
        let first = triggered("k1", "root", "root", "b9", "flow.succeeded", 1);
        assert!(matches!(l.reserve(&first), ReserveOutcome::Reserved(_)));

        // Same tree, same binding, same event, DIFFERENT idempotency key — the
        // longer A->B->A cycle that guard 2 cannot see.
        let again = triggered("k2", "root", "other", "b9", "flow.succeeded", 2);
        match l.reserve(&again) {
            ReserveOutcome::Refused { reason } => assert!(reason.contains("already handled")),
            o => panic!("expected a once-per-tree refusal, got {o:?}"),
        }

        // A different tree is unaffected.
        let other_tree = triggered("k3", "root2", "root2", "b9", "flow.succeeded", 1);
        assert!(matches!(l.reserve(&other_tree), ReserveOutcome::Reserved(_)));
    }

    #[test]
    fn a_direct_invocation_is_never_guarded() {
        let mut l = Ledger::new();
        // No binding, so the binding guards must not apply however many times
        // an operator presses Run.
        for i in 0..5 {
            assert!(matches!(
                l.reserve(&req(&format!("manual-{i}"))),
                ReserveOutcome::Reserved(_)
            ));
        }
        assert_eq!(l.len(), 5);
    }

    #[test]
    fn a_guard_refusal_does_not_consume_the_idempotency_key() {
        let mut l = Ledger::new();
        let over = triggered("shared-key", "root", "p", "b", "e", MAX_LINEAGE_DEPTH + 1);
        assert!(matches!(l.reserve(&over), ReserveOutcome::Refused { .. }));

        // The same key must still be usable — otherwise one guard refusal would
        // permanently poison a legitimate event's key.
        let ok = RunRequest {
            lineage: LineageRef::default(),
            trigger_event: None,
            ..triggered("shared-key", "root", "p", "b", "e", 0)
        };
        assert!(matches!(l.reserve(&ok), ReserveOutcome::Reserved(_)));
    }

    #[test]
    fn the_guard_set_is_bounded() {
        let mut l = Ledger::new();
        for i in 0..(GUARD_CAPACITY + 50) {
            let r = triggered(&format!("k{i}"), &format!("root{i}"), "p", "b", "e", 1);
            l.reserve(&r);
        }
        assert!(
            l.fired.len() <= GUARD_CAPACITY,
            "guard set grew to {}, past the {GUARD_CAPACITY} cap",
            l.fired.len()
        );
        assert_eq!(l.fired.len(), l.fired_order.len(), "index and order agree");
    }

    // --- queue ordering ----------------------------------------------------

    #[test]
    fn queued_returns_claimable_runs_only() {
        let mut l = Ledger::new();
        let a = match l.reserve(&req("a")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.reserve(&req("b"));
        l.claim(&a.run_id, Runtime::Desktop, "d").ok_claimed();

        let q = l.queued(10);
        assert_eq!(q.len(), 1, "a claimed run is no longer claimable");
        assert_eq!(q[0].idempotency_key, "b");
    }

    // --- serialisation matches the protocol's vocabulary -------------------

    #[test]
    fn the_wire_shape_matches_the_schema_not_the_storage() {
        // run-result.schema.json says `startedAt: date-time` under
        // additionalProperties:false. The first cut leaked the storage
        // representation — startedAtMs: 1785… — so every response failed
        // validation at a conforming consumer. The conformance suite now
        // refuses the Ms spelling; this is the Rust side of the same pin.
        let mut l = Ledger::new();
        let rec = match l.reserve(&req("k")) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&rec.run_id, Runtime::Desktop, "dev-1").ok_claimed();
        l.finish(
            &rec.run_id,
            RunStatus::Failed,
            None,
            Some(RunError {
                code: RunErrorCode::NodeFailed,
                message: "boom".into(),
                detail: None,
                node_id: Some("n1".into()),
                capability: None,
                retryable: true,
            }),
        )
        .unwrap();

        let v = serde_json::to_value(l.get(&rec.run_id).unwrap()).unwrap();
        let obj = v.as_object().unwrap();

        for leaked in ["reservedAtMs", "startedAtMs", "finishedAtMs", "node_id"] {
            assert!(
                !v.to_string().contains(leaked),
                "{leaked} leaked into the wire: {v}"
            );
        }
        for (field, _) in [("reservedAt", 0), ("startedAt", 0), ("finishedAt", 0)] {
            let s = obj
                .get(field)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("{field} missing or not a string: {v}"));
            assert!(
                s.contains('T') && s.ends_with('Z'),
                "{field} must be RFC 3339 UTC: {s}"
            );
        }
        assert_eq!(v["error"]["nodeId"], "n1", "RunError must be camelCase: {v}");

        // And it round-trips, so a stored record can be re-read.
        let back: RunRecord = serde_json::from_value(v).unwrap();
        assert_eq!(back.run_id, rec.run_id);
        assert!(back.started_at_ms.is_some());
    }

    #[test]
    fn a_retried_triggered_request_is_a_duplicate_not_a_guard_refusal() {
        // The review's finding: guards ran before the idempotency lookup, so a
        // redelivered webhook carrying lineage hit guard 3 ("already handled")
        // and got a 422 "stop retrying" — losing the caller its only route to
        // the original runId. The duplicate-key tests all used empty lineage,
        // which disables guard 3, so they never noticed.
        let mut l = Ledger::new();
        let first = triggered("same-key", "root", "root", "b1", "flow.succeeded", 1);
        let original = match l.reserve(&first) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };

        // The identical request, redelivered.
        let retry = triggered("same-key", "root", "root", "b1", "flow.succeeded", 1);
        match l.reserve(&retry) {
            ReserveOutcome::Duplicate(r) => assert_eq!(r.run_id, original.run_id),
            o => panic!("a retry with lineage must be a Duplicate, got {o:?}"),
        }
    }

    #[test]
    fn status_serialises_as_the_protocol_spells_it() {
        let j = |s: RunStatus| serde_json::to_string(&s).unwrap();
        assert_eq!(j(RunStatus::Succeeded), "\"succeeded\"");
        assert_eq!(j(RunStatus::TimedOut), "\"timed_out\"");
        // NOT FormLogic's "done"/"error" — the conformance suite refuses those,
        // so the two vocabularies cannot quietly coexist.
        assert_ne!(j(RunStatus::Succeeded), "\"done\"");
    }

    #[test]
    fn error_codes_serialise_as_the_protocol_spells_them() {
        let j = |c: RunErrorCode| serde_json::to_string(&c).unwrap();
        assert_eq!(j(RunErrorCode::CapabilityUnavailable), "\"capability_unavailable\"");
        assert_eq!(j(RunErrorCode::CapabilityDenied), "\"capability_denied\"");
        assert_eq!(j(RunErrorCode::NodeFailed), "\"node_failed\"");
    }

    #[test]
    fn runtime_serialises_lowercase() {
        assert_eq!(serde_json::to_string(&Runtime::Desktop).unwrap(), "\"desktop\"");
    }

    // --- history ----------------------------------------------------------

    /// Reserve, claim and finish a run in one go, so the history tests read as
    /// the sequence of outcomes they are about.
    fn finished(l: &mut Ledger, key: &str, status: RunStatus) -> String {
        let r = match l.reserve(&req(key)) {
            ReserveOutcome::Reserved(r) => r,
            o => panic!("{o:?}"),
        };
        l.claim(&r.run_id, Runtime::Desktop, "w").ok_claimed();
        let err = (!matches!(status, RunStatus::Succeeded | RunStatus::Cancelled))
            .then(|| RunError::new(RunErrorCode::NodeFailed, format!("{key} blew up")));
        l.finish(&r.run_id, status, None, err).unwrap();
        r.run_id
    }

    #[test]
    fn recent_returns_newest_first() {
        let mut l = Ledger::new();
        // reserved_at_ms can tie inside one millisecond, so stamp them apart —
        // the ordering claim is about time, not about insertion luck.
        let ids: Vec<String> = ["a", "b", "c"]
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let id = finished(&mut l, k, RunStatus::Succeeded);
                l.runs.get_mut(&id).unwrap().reserved_at_ms = 1_000 + i as u64;
                id
            })
            .collect();

        let seen: Vec<String> = l.recent(10, &[]).into_iter().map(|r| r.run_id).collect();
        assert_eq!(seen, vec![ids[2].clone(), ids[1].clone(), ids[0].clone()]);
    }

    #[test]
    fn recent_filters_to_the_statuses_asked_for() {
        let mut l = Ledger::new();
        finished(&mut l, "ok", RunStatus::Succeeded);
        let bad = finished(&mut l, "bad", RunStatus::Failed);
        let slow = finished(&mut l, "slow", RunStatus::TimedOut);

        let failures = l.recent(10, &[RunStatus::Failed, RunStatus::TimedOut]);
        let ids: HashSet<&str> = failures.iter().map(|r| r.run_id.as_str()).collect();
        assert_eq!(ids, HashSet::from([bad.as_str(), slow.as_str()]));
        // The whole point of the view: the failure carries its own reason, so
        // nobody has to go and look the run up by id to find out what happened.
        assert!(failures.iter().all(|r| r.error.is_some()));
    }

    #[test]
    fn recent_respects_its_limit() {
        let mut l = Ledger::new();
        for i in 0..5 {
            finished(&mut l, &format!("k{i}"), RunStatus::Succeeded);
        }
        assert_eq!(l.recent(2, &[]).len(), 2);
        assert_eq!(l.recent(0, &[]).len(), 0);
    }

    #[test]
    fn status_counts_cover_every_state_present() {
        let mut l = Ledger::new();
        finished(&mut l, "a", RunStatus::Succeeded);
        finished(&mut l, "b", RunStatus::Succeeded);
        finished(&mut l, "c", RunStatus::Failed);
        // …and one left queued, to prove live runs are counted too.
        l.reserve(&req("d"));

        let counts = l.status_counts();
        assert_eq!(counts.get(&RunStatus::Succeeded), Some(&2));
        assert_eq!(counts.get(&RunStatus::Failed), Some(&1));
        assert_eq!(counts.get(&RunStatus::Queued), Some(&1));
        // Absent rather than zero — a caller reads this with `unwrap_or(0)`.
        assert_eq!(counts.get(&RunStatus::Cancelled), None);
    }

    #[test]
    fn history_survives_a_reopen() {
        let dir = JournalDir::new("history");
        {
            let mut l = Ledger::open(dir.path()).unwrap();
            finished(&mut l, "gone-wrong", RunStatus::Failed);
        }
        // The whole value of a history view is that it outlives the process that
        // produced it — a failure you can only read before the next restart is
        // not a history.
        let l = Ledger::open(dir.path()).unwrap();
        let failures = l.recent(10, &[RunStatus::Failed]);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].error.as_ref().unwrap().message.contains("blew up"));
    }

    /// Test helper: assert a claim won, so the tests above stay about their own
    /// subject rather than re-matching on ClaimOutcome each time.
    trait ClaimAssert {
        fn ok_claimed(self);
    }
    impl ClaimAssert for ClaimOutcome {
        fn ok_claimed(self) {
            match self {
                ClaimOutcome::Claimed(_) => {}
                other => panic!("expected the claim to succeed, got {other:?}"),
            }
        }
    }
}
