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
//! In-memory for now, behind [`LedgerHandle`]. The API is deliberately shaped so
//! the store can move to SQLite without callers changing: every mutation is a
//! single method taking a key, returning a typed outcome, and the guards are
//! evaluated inside the lock rather than by the caller. The unique-key
//! reservation and the compare-and-set claim are exactly the two operations a
//! SQL backend implements as `INSERT … ON CONFLICT` and
//! `UPDATE … WHERE status = 'queued'`.
//!
//! A restart currently loses queued runs. That is honest for a tray app whose
//! consumers re-drive from their own durable side, and it is the reason
//! `reserve` is keyed on a caller-supplied string rather than something we mint.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct RunError {
    pub code: RunErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cancel_requested: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Runtime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
    pub reserved_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Default)]
pub struct Ledger {
    runs: HashMap<String, RunRecord>,
    /// `idempotency_key` -> `run_id`. The uniqueness gate.
    by_key: HashMap<String, String>,
    /// `(root_run_id, binding_id, event)` triples already fired in a tree.
    fired: HashSet<(String, String, String)>,
    /// Insertion order over `fired`, for bounded FIFO eviction.
    fired_order: VecDeque<(String, String, String)>,
    seq: u64,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a run before executing it.
    ///
    /// The order here is deliberate: guards are evaluated BEFORE the uniqueness
    /// check, so a refused cycle never consumes an idempotency key. Otherwise a
    /// guard refusal would poison that key and a legitimate retry of the same
    /// logical event could never run.
    pub fn reserve(&mut self, req: &RunRequest) -> ReserveOutcome {
        if let Some(reason) = self.guard_violation(req) {
            return ReserveOutcome::Refused { reason };
        }

        if let Some(existing_id) = self.by_key.get(&req.idempotency_key) {
            if let Some(rec) = self.runs.get(existing_id) {
                return ReserveOutcome::Duplicate(rec.clone());
            }
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
        self.runs.insert(run_id, record.clone());
        ReserveOutcome::Reserved(record)
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
    }

    /// Move a run to a terminal state. Refuses if it is already terminal.
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
        let Some(rec) = self.runs.get_mut(run_id) else {
            return Err(format!("unknown run {run_id}"));
        };
        if rec.status.is_terminal() {
            return Err(format!(
                "run {run_id} is already {:?}; terminal states are immutable",
                rec.status
            ));
        }
        rec.status = status;
        rec.output = output;
        rec.error = error;
        rec.finished_at_ms = Some(now_ms());
        Ok(rec.clone())
    }

    /// Request cancellation.
    ///
    /// A queued run cancels immediately — nothing is executing. A running one is
    /// only *asked*: a node mid-HTTP-call does not stop on command, so the runner
    /// finalises it when it notices. Reporting a running run as already cancelled
    /// would tell the consumer work had stopped when it had not.
    pub fn request_cancel(&mut self, run_id: &str) -> Result<RunStatus, String> {
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
                Ok(RunStatus::Cancelled)
            }
            RunStatus::Running => {
                rec.cancel_requested = true;
                Ok(RunStatus::Running)
            }
            other => Err(format!("run {run_id} is already {other:?}")),
        }
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
