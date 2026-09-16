//! One warm script host for the whole desktop: `oaiy script --serve`, kept
//! running, spoken to over NDJSON.
//!
//! ```text
//!   condition / app-logic lane ──► ScriptHost::global().evaluate(request)
//!                                     │  (one batch at a time)
//!                                     ▼
//!                       node oaiy.mjs script --serve      (spawned on first use)
//!                         stdin  ◄── {"op":"batch","id":"b1","request":…}
//!                         stdout ──► {"op":"result","id":"b1","result":…}
//!                         stderr ──► drained, logged (the worker's streams)
//! ```
//!
//! # Why a warm child and not a spawn per evaluation
//!
//! A binding condition or an app-logic entry is a few lines of user code that
//! runs in microseconds; `oaiy run` spawns Node, loads and compiles an ~8 MB
//! wasm engine and tears it all down for one answer — seconds of work around
//! microseconds of it, on every plugin event. `oaiy script --serve` (PR5) keeps
//! the engine loaded in one worker thread and answers batches over its stdio,
//! so the desktop pays the start-up once and every evaluation after that is a
//! line each way. The engine, the envelope, the watchdog and the recycle policy
//! all live in the CLI; this file is process supervision and a wire protocol.
//!
//! # One host, process-wide (decision 6a-1)
//!
//! Four owners with no common handle need it — the plugin event thread, the
//! bridge routes, the heartbeat thread and (later) the claim lane — so it is a
//! process-global, `ScriptHost::global()`, the way `worker.rs` keeps its probe
//! cache. The child is started lazily, on the first request: an event with no
//! conditions and no scripts never spawns anything.
//!
//! # Protocol discipline
//!
//! The wire is `cli/src/script.ts`'s, one JSON object per line each way, and
//! every rule there is applied here:
//!
//!   * every line this side sends carries a string `id` of its own making;
//!     replies are demultiplexed by `op` + `id` and a reply is delivered only to
//!     the waiter of the SAME kind (a `pong` never satisfies a batch, a
//!     `result` never satisfies a ping), so a ping answered mid-batch — which
//!     the CLI does deliberately — interleaves without disturbing anything;
//!   * a `result`, `pong` or `error` whose `id` matches nothing is logged and
//!     dropped: nobody asked, nobody is told;
//!   * an `error` with no `id` is the child rejecting a line — a bug on THIS
//!     side — logged, the stream continues, the child is kept;
//!   * an `error` with code `engine_unavailable` means the child is about to
//!     exit 1: the host is marked unavailable at once with the child's reason;
//!   * a line that is not protocol JSON at all is a bug in the child (stdout
//!     is for protocol lines only). One is logged; [`MAX_UNPARSEABLE_LINES`] in
//!     one child's life retire it — enough that a stray diagnostic does not cost
//!     a warm engine, few enough that a child streaming garbage is cut off
//!     within a few kilobytes rather than filling the log;
//!   * the child's stderr is drained from the moment it starts (a worker thread
//!     logs its recycle notices there, and an undrained pipe blocks it).
//!
//! # Deadlines
//!
//! The child watches each job for `budgetMs ?? 1000` plus 1500 ms of grace and
//! answers `timeout` instead of hanging; the desktop's own deadline for a batch
//! is the same figure summed over the jobs — Σ(`budgetMs` ?? 1000 ms) + 1500 ms
//! per job, never less than one grace, capped at [`MAX_BATCH_DEADLINE`] (the
//! app-logic lane's 60 s) — so under normal conditions the child's watchdog
//! fires first and names the job, and only a child that has stopped answering
//! altogether meets this one. When it does, the child is killed and replaced
//! on the next request without a backoff: the desktop ended it, nothing about
//! the install has to change first.
//!
//! # Restart policy
//!
//! A child that ends on its own — exits, closes stdout, reports
//! `engine_unavailable`, or writes [`MAX_UNPARSEABLE_LINES`] lines of garbage —
//! marks the host `Unavailable` with the reason, and the next request after
//! [`RESTART_BACKOFF`] (5 s) spawns a replacement; requests inside the backoff
//! are refused with the reason and how long to wait. Not a loop of retries: a
//! broken install would otherwise be re-spawned on every plugin event, and
//! each spawn is a Node process loading an engine. A child this side killed
//! for a deadline carries no backoff (above). A refused probe (`worker.rs`
//! decides that, with its own 60 s memory of a refusal) or a CLI that speaks no
//! `script` protocol is `Unavailable` the same way.
//!
//! # What the child may not see
//!
//! A flow run is told this desktop's API URL and given its bearer token because
//! its connector nodes call back in. A leaf script has no nodes and calls
//! nobody, so the child gets NEITHER — `OAIY_SERVER_URL` and
//! `OAIY_SERVER_TOKEN` are removed from its environment explicitly, on top of
//! `worker::harden_child`'s deny-list and `OAIY_ZIPP_ASSET_DIR`. The headless
//! server has the real token in its own environment; "not set" is not enough.
//!
//! # Before the child starts
//!
//! The CLI must pass PR3's engine probe (`capabilities --json`: ZIPP, ready)
//! AND report `protocols.script == 1`. Once the child is up, its first `pong`
//! must name the same engine the probe did (`wasmSha256`) — the identity in
//! [`HealthSnapshot::engine`] is the probe's, confirmed by the child, and no
//! literal is written here. A request larger than [`MAX_REQUEST_BYTES`] is
//! refused before anything is spawned or written: the biggest legitimate batch
//! (a dozen app-logic scripts at the engine's 64 KiB dynamic-source ceiling,
//! each with an event and an app's storage as context) is well under it, and a
//! larger one is a caller shipping something a leaf script should not be
//! given — refusing it locally costs nothing and never puts a multi-megabyte
//! line through a pipe the child reads with `readline`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::worker::{self, CliInvocation, EngineIdentity, EngineProbe};

/// A request larger than this is refused locally, unsent (see the module doc).
pub const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
/// A reply line longer than this is treated as garbage rather than parsed (and
/// counts towards [`MAX_UNPARSEABLE_LINES`]). Well above anything the child can
/// legitimately send: its own host-value conversion limit is smaller again, so a
/// line this long is a child that has lost the protocol, not a large result.
pub const MAX_REPLY_LINE_BYTES: usize = 16 * 1024 * 1024;
/// Lines of non-protocol output from one child before it is retired.
pub const MAX_UNPARSEABLE_LINES: u32 = 8;
/// How long after a child ended on its own before a replacement is tried.
pub const RESTART_BACKOFF: Duration = Duration::from_secs(5);
/// The child's own default for a job that names no `budgetMs`
/// (`SCRIPT_DEFAULT_BUDGET_MS` in `cli/src/zipp/script-host.ts`).
pub const DEFAULT_JOB_BUDGET_MS: u64 = 1000;
/// The child's watchdog grace over a job's budget (`SCRIPT_WATCHDOG_GRACE_MS`).
pub const JOB_GRACE_MS: u64 = 1500;
/// Ceiling on one batch's deadline: the app-logic lane's own run timeout.
pub const MAX_BATCH_DEADLINE: Duration = Duration::from_secs(60);
/// How long a freshly spawned child has to answer its first ping. It loads and
/// compiles the engine before reading a line, the same work as the probe.
pub const START_DEADLINE: Duration = worker::PROBE_DEADLINE;
/// How long `shutdown` waits for the child to exit on its own before killing it.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);
/// Poll interval while waiting for a child to exit.
const EXIT_POLL: Duration = Duration::from_millis(25);

// ---------------------------------------------------------------------------
// The public shape
// ---------------------------------------------------------------------------

/// Whether the host can serve right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    /// No child has proven itself yet: none has been asked for, or one is
    /// being started. Not a failure — the heartbeat treats it as "not
    /// unavailable".
    Starting,
    /// A child is up and has answered.
    Ready,
    /// The last attempt failed for `reason`; a replacement is tried on the
    /// next request once the backoff (if any) has passed.
    Unavailable { reason: String },
}

/// [`ScriptHost::health`]: the state, when it last changed, and what is running.
#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub health: Health,
    /// When `health` last changed — for a heartbeat that rate-limits its beats.
    pub since: Instant,
    /// The engine the running child serves: the probe's identity, confirmed by
    /// the child's first `pong`. `None` until a child has answered.
    pub engine: Option<EngineIdentity>,
    /// The child's `pong.instance` when last seen — its own count of engine
    /// workers, up by one at every recycle. 0 until a child has answered.
    pub instance: u64,
    /// How many children this host has spawned in this process.
    pub children: u64,
}

/// Why a batch was not answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// The serialised request exceeds [`MAX_REQUEST_BYTES`]; nothing was spawned or sent.
    RequestTooLarge { bytes: usize, cap: usize },
    /// No child can serve right now. `retry_after` is set while a backoff is in force.
    Unavailable { reason: String, retry_after: Option<Duration> },
    /// The CLI refused the whole request (`invalid_request`) — the requester's
    /// fault, deterministic, nothing ran.
    Refused { message: String },
    /// The child answered this batch with an `error` line (its own fault, e.g.
    /// `invalid_line`), or with a result this side cannot read. The child is kept.
    Failed { code: String, message: String },
    /// The batch deadline passed with no answer: the child was killed and is
    /// replaced on the next request.
    TimedOut { deadline: Duration },
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostError::RequestTooLarge { bytes, cap } => {
                write!(f, "the script request is {bytes} bytes; the host sends at most {cap}")
            }
            HostError::Unavailable { reason, retry_after: Some(d) } => {
                write!(f, "the script host is unavailable ({reason}); retry in {}ms", d.as_millis())
            }
            HostError::Unavailable { reason, retry_after: None } => {
                write!(f, "the script host is unavailable ({reason})")
            }
            HostError::Refused { message } => write!(f, "the script request was refused: {message}"),
            HostError::Failed { code, message } => write!(f, "the script host failed the batch ({code}): {message}"),
            HostError::TimedOut { deadline } => {
                write!(f, "the script host did not answer within {}ms", deadline.as_millis())
            }
        }
    }
}

/// One job's answer, as `protocol/v1/script-result.schema.json` describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct JobResult {
    pub id: String,
    /// `Ok(value)` for `ok: true` (`Null` when the job returned nothing),
    /// `Err` with the CLI's `errorKind` and `error` otherwise.
    pub outcome: Result<Value, JobError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobError {
    /// `guest`, `resource`, `timeout`, `host`, `prepare`, `source` or `unsupported`.
    pub kind: String,
    pub message: String,
}

/// A served batch: the CLI's `{v, engine, results}` read into shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ScriptResponse {
    /// The identity the CLI put on the response, verbatim.
    pub engine: Value,
    pub results: Vec<JobResult>,
}

impl ScriptResponse {
    /// The result for job `id`, if the CLI answered it.
    pub fn result(&self, id: &str) -> Option<&JobResult> {
        self.results.iter().find(|r| r.id == id)
    }
}

/// The request envelope for `jobs`, protocol v1. The caller builds each job as
/// the schema says (`id`, `mode`, `source`, `globals`, `entry`, `args`,
/// `budgetMs`, …); nothing here inspects them.
pub fn batch_request(jobs: Vec<Value>) -> Value {
    json!({ "v": 1, "jobs": jobs })
}

/// The desktop's deadline for `request` (see the module doc): Σ over the jobs of
/// (`budgetMs` ?? [`DEFAULT_JOB_BUDGET_MS`]) + [`JOB_GRACE_MS`], at least one
/// grace, at most [`MAX_BATCH_DEADLINE`].
pub fn batch_deadline(request: &Value) -> Duration {
    let sum_ms: u64 = request
        .get("jobs")
        .and_then(Value::as_array)
        .map(|jobs| {
            jobs.iter()
                .map(|j| j.get("budgetMs").and_then(Value::as_u64).unwrap_or(DEFAULT_JOB_BUDGET_MS) + JOB_GRACE_MS)
                .fold(0u64, u64::saturating_add)
        })
        .unwrap_or(0);
    Duration::from_millis(sum_ms.max(JOB_GRACE_MS)).min(MAX_BATCH_DEADLINE)
}

// ---------------------------------------------------------------------------
// The demultiplexer: pure, so every routing rule is pinned without a process
// ---------------------------------------------------------------------------

/// What a waiter is waiting for. A reply is delivered only to a waiter of its
/// own kind, so a `pong` can never be mistaken for a batch's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitKind {
    Batch,
    Ping,
}

/// A reply routed to a waiter.
#[derive(Debug, Clone, PartialEq)]
enum Reply {
    /// `result` — the `result` field, the CLI's response or refusal verbatim.
    Result(Value),
    /// `error` addressed to this id.
    Error { code: String, message: String },
    /// `pong` — the whole line.
    Pong(Value),
}

/// What the reader did with one stdout line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Routed {
    /// Handed to the waiter with that id.
    Delivered,
    /// A well-formed reply nobody is waiting for: logged, dropped.
    Orphan,
    /// `error` with no id — the child rejected a line this side wrote.
    Unaddressed { code: String, message: String },
    /// `error` `engine_unavailable`: the child is exiting.
    EngineUnavailable { message: String },
    /// Not a protocol line at all.
    Garbage,
}

#[derive(Default)]
struct Waiters {
    map: HashMap<String, (WaitKind, SyncSender<Reply>)>,
}

impl Waiters {
    fn register(&mut self, id: String, kind: WaitKind) -> Receiver<Reply> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.map.insert(id, (kind, tx));
        rx
    }

    fn unregister(&mut self, id: &str) {
        self.map.remove(id);
    }

    /// Drop every waiter: each receiver sees `Disconnected`.
    fn fail_all(&mut self) {
        self.map.clear();
    }

    /// Route one line. The `id` must be a JSON STRING equal to a registered
    /// id — this side sends only strings, so `1` never matches `"1"`.
    fn route(&mut self, line: &str) -> Routed {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            return Routed::Garbage;
        };
        if !v.is_object() {
            return Routed::Garbage;
        }
        let id = v.get("id").and_then(Value::as_str).map(str::to_string);
        let deliver = |me: &mut Self, kind: WaitKind, reply: Reply| -> Routed {
            let Some(id) = id.as_deref() else {
                return Routed::Orphan;
            };
            match me.map.get(id) {
                Some((k, _)) if *k == kind => {
                    let (_, tx) = me.map.remove(id).expect("just seen");
                    // A full or closed channel means the waiter already gave
                    // up (deadline): the reply is late, and dropped.
                    let _ = tx.try_send(reply);
                    Routed::Delivered
                }
                _ => Routed::Orphan,
            }
        };
        match v.get("op").and_then(Value::as_str) {
            Some("result") => {
                let result = v.get("result").cloned().unwrap_or(Value::Null);
                deliver(self, WaitKind::Batch, Reply::Result(result))
            }
            Some("pong") => deliver(self, WaitKind::Ping, Reply::Pong(v.clone())),
            Some("error") => {
                let code = v
                    .pointer("/error/code")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let message = v
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("(no message)")
                    .to_string();
                if code == "engine_unavailable" {
                    return Routed::EngineUnavailable { message };
                }
                if id.is_none() {
                    return Routed::Unaddressed { code, message };
                }
                let addressed = id.as_deref().and_then(|i| self.map.get(i)).map(|(k, _)| *k);
                match addressed {
                    Some(kind) => deliver(self, kind, Reply::Error { code, message }),
                    None => Routed::Orphan,
                }
            }
            _ => Routed::Garbage,
        }
    }
}

/// Read the CLI's `result` field into a [`ScriptResponse`].
fn parse_result(result: Value) -> Result<ScriptResponse, HostError> {
    if let Some(err) = result.get("error") {
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| err.get("code").and_then(Value::as_str))
            .unwrap_or("the request was refused without a reason")
            .to_string();
        return Err(HostError::Refused { message });
    }
    let malformed = |what: &str| HostError::Failed {
        code: "malformed_result".to_string(),
        message: format!("the script host's result {what}"),
    };
    let Some(items) = result.get("results").and_then(Value::as_array) else {
        return Err(malformed("carries no results array"));
    };
    let mut results = Vec::with_capacity(items.len());
    for item in items {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("has a result without an id"))?
            .to_string();
        let outcome = match item.get("ok").and_then(Value::as_bool) {
            Some(true) => Ok(item.get("value").cloned().unwrap_or(Value::Null)),
            Some(false) => Err(JobError {
                kind: item
                    .get("errorKind")
                    .and_then(Value::as_str)
                    .unwrap_or("host")
                    .to_string(),
                message: item
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("(no message)")
                    .to_string(),
            }),
            None => return Err(malformed(&format!("says neither ok nor not for job {id:?}"))),
        };
        results.push(JobResult { id, outcome });
    }
    Ok(ScriptResponse {
        engine: result.get("engine").cloned().unwrap_or(Value::Null),
        results,
    })
}

// ---------------------------------------------------------------------------
// The child
// ---------------------------------------------------------------------------

/// One running `script --serve` process and everything attached to it.
struct Live {
    /// Which spawn this is (1-based), so a retirement for one child can never
    /// take out its successor.
    generation: u64,
    process: Mutex<Child>,
    /// `None` once closed (shutdown sends EOF as well as the op).
    stdin: Mutex<Option<ChildStdin>>,
    waiters: Mutex<Waiters>,
    /// Non-protocol lines seen so far.
    garbage: AtomicU32,
    /// Ids handed out.
    seq: AtomicU64,
    /// Set once the child is retired, by whoever retires it.
    gone: AtomicBool,
    /// Rolling tail of the child's stderr, for the reason when it dies.
    stderr_tail: Arc<Mutex<String>>,
    /// Ready when `gone` is set at exit, for a bounded wait.
    exit_code: Mutex<Option<i32>>,
}

impl Live {
    fn next_id(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.seq.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn register(&self, kind: WaitKind, prefix: &str) -> (String, Receiver<Reply>) {
        let id = self.next_id(prefix);
        let rx = self.waiters.lock().unwrap_or_else(|e| e.into_inner()).register(id.clone(), kind);
        (id, rx)
    }

    fn unregister(&self, id: &str) {
        self.waiters.lock().unwrap_or_else(|e| e.into_inner()).unregister(id);
    }

    /// Write one protocol line. A failure means the child is gone.
    fn write_line(&self, line: &str) -> Result<(), String> {
        let mut guard = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        let Some(stdin) = guard.as_mut() else {
            return Err("its stdin is closed".to_string());
        };
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|e| format!("its stdin could not be written: {e}"))
    }

    /// End the process: close stdin, kill, reap, fail every waiter.
    fn terminate(&self) {
        self.gone.store(true, Ordering::Release);
        drop(self.stdin.lock().unwrap_or_else(|e| e.into_inner()).take());
        if let Ok(mut child) = self.process.lock() {
            let _ = child.kill();
            if let Ok(status) = child.wait() {
                *self.exit_code.lock().unwrap_or_else(|e| e.into_inner()) = status.code();
            }
        }
        self.waiters.lock().unwrap_or_else(|e| e.into_inner()).fail_all();
    }

    /// Wait up to `grace` for the child to exit by itself; `None` if it did not.
    fn wait_exit(&self, grace: Duration) -> Option<i32> {
        let deadline = Instant::now() + grace;
        loop {
            if let Ok(mut child) = self.process.lock() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let code = status.code();
                        *self.exit_code.lock().unwrap_or_else(|e| e.into_inner()) = code;
                        return code.or(Some(-1));
                    }
                    Ok(None) => {}
                    Err(_) => return None,
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(EXIT_POLL);
        }
    }

    fn stderr_tail(&self) -> String {
        worker::tail_of(&self.stderr_tail.lock().map(|g| g.clone()).unwrap_or_default(), 600)
    }
}

/// Where the child's command comes from.
enum CliSource {
    /// The desktop's own resolution (`OAIY_CLI`, the bundled copy, PATH) and
    /// the managed Node runtime, re-read at every spawn.
    Resolved,
    /// A fixed CLI and Node, for tests that hand in a stub.
    Fixed { cli: CliInvocation, node_exe: Option<PathBuf> },
}

struct State {
    child: Option<Arc<Live>>,
    health: Health,
    since: Instant,
    engine: Option<EngineIdentity>,
    instance: u64,
    children: u64,
    /// Earliest moment a replacement may be spawned, while unavailable.
    retry_at: Option<Instant>,
}

struct Inner {
    source: CliSource,
    node: Mutex<Option<crate::services::node_runtime::NodeHandle>>,
    backoff: Duration,
    /// Serialises `evaluate` end to end (spawn, write, wait). Two callers writing
    /// at once would have the second's deadline running while the first's batch
    /// executes — a false kill of a healthy child. Pings do not take it.
    batch: Mutex<()>,
    state: Mutex<State>,
}

/// The warm script host. See the module doc.
pub struct ScriptHost {
    inner: Arc<Inner>,
}

static GLOBAL: OnceLock<ScriptHost> = OnceLock::new();

impl ScriptHost {
    fn new(source: CliSource, backoff: Duration) -> Self {
        ScriptHost {
            inner: Arc::new(Inner {
                source,
                node: Mutex::new(None),
                backoff,
                batch: Mutex::new(()),
                state: Mutex::new(State {
                    child: None,
                    health: Health::Starting,
                    since: Instant::now(),
                    engine: None,
                    instance: 0,
                    children: 0,
                    retry_at: None,
                }),
            }),
        }
    }

    /// The process-wide host, resolving the CLI the way a flow run does.
    pub fn global() -> &'static ScriptHost {
        GLOBAL.get_or_init(|| ScriptHost::new(CliSource::Resolved, RESTART_BACKOFF))
    }

    /// A host bound to one CLI and Node — the injection the stub-CLI tests use,
    /// since `OAIY_CLI` is process-global. Same backoff as the global host.
    pub fn with_cli(cli: CliInvocation, node_exe: Option<PathBuf>) -> Self {
        ScriptHost::with_cli_and_backoff(cli, node_exe, RESTART_BACKOFF)
    }

    /// [`with_cli`](Self::with_cli) with another restart backoff, so a test can
    /// prove the backoff without waiting one out.
    pub(crate) fn with_cli_and_backoff(
        cli: CliInvocation,
        node_exe: Option<PathBuf>,
        backoff: Duration,
    ) -> Self {
        ScriptHost::new(CliSource::Fixed { cli, node_exe }, backoff)
    }

    /// The Node runtime the bundled CLI runs under (the resolved source only):
    /// a packaged install cannot assume `node` on PATH. Same handle the flow
    /// worker and the app-logic lane are given.
    pub fn set_node_runtime(&self, node: Option<crate::services::node_runtime::NodeHandle>) {
        *self.inner.node.lock().unwrap_or_else(|e| e.into_inner()) = node;
    }

    /// The state right now. Never spawns, never blocks on the child.
    pub fn health(&self) -> HealthSnapshot {
        let s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        HealthSnapshot {
            health: s.health.clone(),
            since: s.since,
            engine: s.engine.clone(),
            instance: s.instance,
            children: s.children,
        }
    }

    /// Serve `request` (a `script-request` document, see [`batch_request`])
    /// within [`batch_deadline`].
    pub fn evaluate(&self, request: &Value) -> Result<ScriptResponse, HostError> {
        self.evaluate_within(request, batch_deadline(request))
    }

    /// [`evaluate`](Self::evaluate) with the caller's own deadline.
    ///
    /// Blocking, and meant for a plain thread — the plugin event thread, the
    /// flow runner, `spawn_blocking` from a route — never for a tokio task.
    pub fn evaluate_within(&self, request: &Value, deadline: Duration) -> Result<ScriptResponse, HostError> {
        let bytes = serde_json::to_vec(request).map(|b| b.len()).unwrap_or(usize::MAX);
        if bytes > MAX_REQUEST_BYTES {
            return Err(HostError::RequestTooLarge { bytes, cap: MAX_REQUEST_BYTES });
        }
        let _one_at_a_time = self.inner.batch.lock().unwrap_or_else(|e| e.into_inner());
        let live = self.ensure_child()?;
        let (id, rx) = live.register(WaitKind::Batch, "b");
        let line = json!({ "op": "batch", "id": id, "request": request }).to_string();
        if let Err(why) = live.write_line(&line) {
            live.unregister(&id);
            self.retire(&live, format!("the script host could not be sent a batch: {why}"), true);
            return Err(self.unavailable());
        }
        match rx.recv_timeout(deadline) {
            Ok(Reply::Result(result)) => parse_result(result),
            Ok(Reply::Error { code, message }) => Err(HostError::Failed { code, message }),
            Ok(Reply::Pong(_)) => Err(HostError::Failed {
                code: "protocol".to_string(),
                message: "a pong was routed to a batch".to_string(),
            }),
            Err(RecvTimeoutError::Timeout) => {
                live.unregister(&id);
                self.retire(
                    &live,
                    format!("the script host did not answer a batch within {}ms and was replaced", deadline.as_millis()),
                    false,
                );
                Err(HostError::TimedOut { deadline })
            }
            // The reader retired the child (exit, engine_unavailable, garbage)
            // and recorded why.
            Err(RecvTimeoutError::Disconnected) => Err(self.unavailable()),
        }
    }

    /// Ask the running child for its `pong`. `None` when there is no child or it
    /// does not answer in time (which retires it).
    pub fn ping(&self, timeout: Duration) -> Option<Value> {
        let live = self.inner.state.lock().unwrap_or_else(|e| e.into_inner()).child.clone()?;
        match self.ping_child(&live, timeout) {
            Ok(pong) => Some(pong),
            Err(reason) => {
                self.retire(&live, reason, true);
                None
            }
        }
    }

    /// End the child: `{op:"shutdown"}` and EOF, [`SHUTDOWN_GRACE`] to exit on
    /// its own, then a kill. Bounded. A later request starts a new child.
    pub fn shutdown(&self) {
        let live = {
            let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            let live = s.child.take();
            if live.is_some() {
                s.health = Health::Starting;
                s.since = Instant::now();
                s.retry_at = None;
            }
            live
        };
        let Some(live) = live else {
            return;
        };
        let started = Instant::now();
        let asked = live.write_line(&json!({ "op": "shutdown" }).to_string()).is_ok();
        drop(live.stdin.lock().unwrap_or_else(|e| e.into_inner()).take());
        let exited = if asked { live.wait_exit(SHUTDOWN_GRACE) } else { None };
        match exited {
            Some(code) => log::info!(
                "the script host exited {code} on shutdown after {:?}",
                started.elapsed()
            ),
            None => log::warn!(
                "the script host did not exit within {}ms of shutdown; killed",
                SHUTDOWN_GRACE.as_millis()
            ),
        }
        live.terminate();
    }

    // -- internals --------------------------------------------------------------

    fn unavailable(&self) -> HostError {
        let s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let reason = match &s.health {
            Health::Unavailable { reason } => reason.clone(),
            other => format!("the script host is {other:?}"),
        };
        let retry_after = s
            .retry_at
            .and_then(|at| at.checked_duration_since(Instant::now()))
            .filter(|d| !d.is_zero());
        HostError::Unavailable { reason, retry_after }
    }

    /// Record a failure. `backoff` — whether the child ended on its own (wait
    /// before a replacement) or this side ended it (replace on the next call).
    fn mark_unavailable(&self, reason: String, backoff: bool) {
        let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        log::warn!("script host unavailable: {reason}");
        s.health = Health::Unavailable { reason };
        s.since = Instant::now();
        s.retry_at = backoff.then(|| Instant::now() + self.inner.backoff);
    }

    /// Retire `live` — if it is still the current child — for `reason`.
    fn retire(&self, live: &Arc<Live>, reason: String, backoff: bool) {
        let current = {
            let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            match &s.child {
                Some(c) if c.generation == live.generation => s.child.take().is_some(),
                _ => false,
            }
        };
        if current {
            self.mark_unavailable(reason, backoff);
        }
        live.terminate();
    }

    /// The current child, or a new one. Called under the batch lock.
    fn ensure_child(&self) -> Result<Arc<Live>, HostError> {
        let generation = {
            let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(live) = &s.child {
                if !live.gone.load(Ordering::Acquire) {
                    return Ok(live.clone());
                }
            }
            if let Some(at) = s.retry_at {
                if Instant::now() < at {
                    drop(s);
                    return Err(self.unavailable());
                }
            }
            s.health = Health::Starting;
            s.since = Instant::now();
            s.retry_at = None;
            s.children += 1;
            s.children
        };
        match self.spawn(generation) {
            Ok((live, identity, instance)) => {
                let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
                s.child = Some(live.clone());
                s.health = Health::Ready;
                s.since = Instant::now();
                s.engine = Some(identity);
                s.instance = instance;
                Ok(live)
            }
            Err(reason) => {
                self.mark_unavailable(reason, true);
                Err(self.unavailable())
            }
        }
    }

    fn command_source(&self) -> Result<(CliInvocation, Option<PathBuf>), String> {
        match &self.inner.source {
            CliSource::Fixed { cli, node_exe } => Ok((cli.clone(), node_exe.clone())),
            CliSource::Resolved => {
                let cli = worker::cli_status().ok_or_else(|| {
                    "the OAIY CLI is not available on this machine (install it on PATH, or set OAIY_CLI)"
                        .to_string()
                })?;
                let node_exe = self
                    .inner
                    .node
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .and_then(|n| n.resolve());
                Ok((cli, node_exe))
            }
        }
    }

    /// Probe, check the protocol, spawn, drain, ping. Errors are the reason.
    fn spawn(&self, generation: u64) -> Result<(Arc<Live>, EngineIdentity, u64), String> {
        let (cli, node_exe) = self.command_source()?;
        let identity = match worker::engine_probe(&cli, node_exe.as_deref()) {
            EngineProbe::Ready(id) => id,
            EngineProbe::Unavailable { reason } => {
                return Err(format!(
                    "the OAIY CLI at {} does not run user logic on ZIPP: {reason}. {}",
                    cli.path().display(),
                    worker::engine_fix_hint()
                ))
            }
        };
        match identity.protocols.get("script") {
            Some(1) => {}
            other => {
                return Err(format!(
                    "the OAIY CLI at {} speaks script protocol {}, and this desktop speaks protocol 1 (its protocols: {})",
                    cli.path().display(),
                    other.map_or("(none)".to_string(), |n| n.to_string()),
                    serde_json::to_string(&identity.protocols).unwrap_or_default()
                ))
            }
        }

        let mut cmd = worker::cli_command(&cli, node_exe.as_deref());
        cmd.arg("script").arg("--serve");
        worker::harden_child(&mut cmd);
        // `harden_child` gives a run a null stdin; this child is driven through it.
        cmd.stdin(Stdio::piped());
        // A leaf script calls nobody: no API URL, and never this process's
        // bearer — which the headless server carries in its own environment.
        cmd.env_remove("OAIY_SERVER_URL");
        cmd.env_remove("OAIY_SERVER_TOKEN");
        let started = Instant::now();
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("the OAIY CLI could not be started as a script host: {e}"))?;
        crate::services::job_object::adopt(child.id());

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        if let Some(err) = child.stderr.take() {
            let sink = stderr_tail.clone();
            thread::spawn(move || worker::drain_capped(err, &sink));
        }
        let live = Arc::new(Live {
            generation,
            process: Mutex::new(child),
            stdin: Mutex::new(stdin),
            waiters: Mutex::new(Waiters::default()),
            garbage: AtomicU32::new(0),
            seq: AtomicU64::new(0),
            gone: AtomicBool::new(false),
            stderr_tail,
            exit_code: Mutex::new(None),
        });
        if let Some(out) = stdout {
            let reader_live = live.clone();
            let reader_host = ScriptHost { inner: self.inner.clone() };
            thread::spawn(move || reader_host.read_loop(out, reader_live));
        }

        // The first pong is the readiness signal — the CLI loads the engine
        // before it reads a line — and the check that this child serves the
        // engine the probe saw.
        let pong = match self.ping_child(&live, START_DEADLINE) {
            Ok(pong) => pong,
            Err(reason) => {
                live.terminate();
                return Err(format!("the script host did not come up: {reason}"));
            }
        };
        let child_sha = pong.pointer("/engine/wasmSha256").and_then(Value::as_str).unwrap_or("");
        if child_sha != identity.wasm_sha256 {
            live.terminate();
            return Err(format!(
                "the script host serves an engine ({}) other than the one the CLI reported ({})",
                if child_sha.is_empty() { "unnamed" } else { child_sha },
                identity.wasm_sha256
            ));
        }
        let instance = pong.get("instance").and_then(Value::as_u64).unwrap_or(0);
        log::info!(
            "script host #{generation} up in {:?}: ZIPP {} ({}), instance {instance}",
            started.elapsed(),
            identity.release,
            &identity.wasm_sha256[..identity.wasm_sha256.len().min(12)]
        );
        Ok((live, identity, instance))
    }

    fn ping_child(&self, live: &Arc<Live>, timeout: Duration) -> Result<Value, String> {
        let (id, rx) = live.register(WaitKind::Ping, "p");
        if let Err(why) = live.write_line(&json!({ "op": "ping", "id": id }).to_string()) {
            live.unregister(&id);
            return Err(format!("the script host could not be pinged: {why}"));
        }
        match rx.recv_timeout(timeout) {
            Ok(Reply::Pong(pong)) => {
                if let Some(n) = pong.get("instance").and_then(Value::as_u64) {
                    let mut s = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
                    if s.child.as_ref().is_some_and(|c| c.generation == live.generation) {
                        s.instance = n;
                    }
                }
                Ok(pong)
            }
            Ok(Reply::Error { code, message }) => Err(format!("the script host answered a ping with {code}: {message}")),
            Ok(Reply::Result(_)) => Err("a result was routed to a ping".to_string()),
            Err(RecvTimeoutError::Timeout) => {
                live.unregister(&id);
                Err(format!("the script host did not answer a ping within {}ms", timeout.as_millis()))
            }
            Err(RecvTimeoutError::Disconnected) => Err(match &self.health().health {
                Health::Unavailable { reason } => reason.clone(),
                _ => "the script host ended".to_string(),
            }),
        }
    }

    /// The reader thread: stdout lines to waiters, until EOF.
    fn read_loop(&self, out: std::process::ChildStdout, live: Arc<Live>) {
        let mut reader = BufReader::new(out);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let text = line.trim_end_matches(['\r', '\n']);
            if text.trim().is_empty() {
                continue;
            }
            let routed = if text.len() > MAX_REPLY_LINE_BYTES {
                Routed::Garbage
            } else {
                live.waiters.lock().unwrap_or_else(|e| e.into_inner()).route(text)
            };
            match routed {
                Routed::Delivered => {}
                Routed::Orphan => log::warn!(
                    "script host: a reply nobody was waiting for was dropped: {}",
                    worker::tail_of(text, 200)
                ),
                Routed::Unaddressed { code, message } => log::error!(
                    "script host rejected a line this desktop wrote ({code}): {message}"
                ),
                Routed::EngineUnavailable { message } => {
                    self.retire(&live, format!("the OAIY CLI has no engine to serve scripts with: {message}"), true);
                    break;
                }
                Routed::Garbage => {
                    let n = live.garbage.fetch_add(1, Ordering::Relaxed) + 1;
                    log::error!(
                        "script host wrote a line that is not protocol JSON ({n}/{MAX_UNPARSEABLE_LINES}): {}",
                        worker::tail_of(text, 200)
                    );
                    if n >= MAX_UNPARSEABLE_LINES {
                        self.retire(
                            &live,
                            format!(
                                "the script host wrote {n} lines that are not protocol JSON and was replaced"
                            ),
                            true,
                        );
                        break;
                    }
                }
            }
        }
        // EOF (or a retirement above). If nobody retired it yet, the child ended
        // on its own: reap it and say how.
        if !live.gone.load(Ordering::Acquire) {
            let code = live.wait_exit(Duration::from_secs(2));
            let tail = live.stderr_tail();
            let reason = format!(
                "the script host exited{}{}",
                code.map_or(String::new(), |c| format!(" with code {c}")),
                if tail.is_empty() { String::new() } else { format!(": {tail}") }
            );
            self.retire(&live, reason, true);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // --- the demultiplexer, without a process --------------------------------

    #[test]
    fn a_result_line_reaches_the_waiter_with_its_id_and_no_other() {
        let mut w = Waiters::default();
        let rx1 = w.register("b1".into(), WaitKind::Batch);
        let rx2 = w.register("b2".into(), WaitKind::Batch);
        // A swapped id: b2's result must not land on b1.
        assert_eq!(
            w.route(r#"{"op":"result","id":"b2","result":{"v":1,"results":[]}}"#),
            Routed::Delivered
        );
        assert!(matches!(rx1.try_recv(), Err(mpsc::TryRecvError::Empty)), "b1 is still waiting");
        assert!(matches!(rx2.try_recv(), Ok(Reply::Result(_))));
        // An id nobody registered is dropped, and b1 is still waiting.
        assert_eq!(w.route(r#"{"op":"result","id":"b9","result":null}"#), Routed::Orphan);
        assert!(matches!(rx1.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(w.route(r#"{"op":"result","id":"b1","result":{"v":1,"results":[]}}"#), Routed::Delivered);
        assert!(matches!(rx1.try_recv(), Ok(Reply::Result(_))));
        // Delivered once: a second answer for the same id is an orphan.
        assert_eq!(w.route(r#"{"op":"result","id":"b1","result":null}"#), Routed::Orphan);
    }

    #[test]
    fn a_numeric_id_never_matches_a_string_waiter() {
        let mut w = Waiters::default();
        let rx = w.register("1".into(), WaitKind::Batch);
        assert_eq!(w.route(r#"{"op":"result","id":1,"result":null}"#), Routed::Orphan);
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }

    #[test]
    fn a_pong_between_a_batch_and_its_result_does_not_disturb_the_batch() {
        let mut w = Waiters::default();
        let batch = w.register("b1".into(), WaitKind::Batch);
        let ping = w.register("p1".into(), WaitKind::Ping);
        assert_eq!(w.route(r#"{"op":"pong","id":"p1","engine":{},"instance":1,"jobs":0}"#), Routed::Delivered);
        assert!(matches!(batch.try_recv(), Err(mpsc::TryRecvError::Empty)), "the batch is untouched");
        assert!(matches!(ping.try_recv(), Ok(Reply::Pong(_))));
        // A pong addressed to the batch's id is the wrong kind: dropped.
        assert_eq!(w.route(r#"{"op":"pong","id":"b1","engine":{},"instance":1,"jobs":0}"#), Routed::Orphan);
        // An unsolicited pong (no id) is dropped too.
        assert_eq!(w.route(r#"{"op":"pong","engine":{},"instance":1,"jobs":0}"#), Routed::Orphan);
        assert!(matches!(batch.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(w.route(r#"{"op":"result","id":"b1","result":{"v":1,"results":[]}}"#), Routed::Delivered);
        assert!(matches!(batch.try_recv(), Ok(Reply::Result(_))));
    }

    #[test]
    fn an_error_line_with_an_id_fails_that_waiter_only() {
        let mut w = Waiters::default();
        let a = w.register("b1".into(), WaitKind::Batch);
        let b = w.register("b2".into(), WaitKind::Batch);
        assert_eq!(
            w.route(r#"{"op":"error","id":"b1","error":{"code":"invalid_line","message":"a batch needs a \"request\""}}"#),
            Routed::Delivered
        );
        match a.try_recv() {
            Ok(Reply::Error { code, message }) => {
                assert_eq!(code, "invalid_line");
                assert!(message.contains("request"));
            }
            other => panic!("expected the error, got {other:?}"),
        }
        assert!(matches!(b.try_recv(), Err(mpsc::TryRecvError::Empty)), "b2 is untouched");
    }

    #[test]
    fn an_error_line_without_an_id_is_logged_and_the_stream_continues() {
        // `invalid_line` with `id: null` is the child rejecting a line this
        // side wrote — a bug here, not a reason to lose the child or a batch.
        let mut w = Waiters::default();
        let rx = w.register("b1".into(), WaitKind::Batch);
        assert_eq!(
            w.route(r#"{"op":"error","id":null,"error":{"code":"invalid_line","message":"not JSON: x"}}"#),
            Routed::Unaddressed { code: "invalid_line".into(), message: "not JSON: x".into() }
        );
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(w.map.len(), 1, "the waiter is still registered");
        // An error for an id nobody registered is an orphan.
        assert_eq!(
            w.route(r#"{"op":"error","id":"zz","error":{"code":"invalid_line","message":"?"}}"#),
            Routed::Orphan
        );
    }

    #[test]
    fn engine_unavailable_is_the_end_of_the_child_whatever_its_id() {
        let mut w = Waiters::default();
        let _rx = w.register("b1".into(), WaitKind::Batch);
        assert_eq!(
            w.route(r#"{"op":"error","id":null,"error":{"code":"engine_unavailable","message":"ZIPP engine unavailable: no wasm"}}"#),
            Routed::EngineUnavailable { message: "ZIPP engine unavailable: no wasm".into() }
        );
    }

    #[test]
    fn a_line_that_is_not_protocol_json_is_garbage() {
        let mut w = Waiters::default();
        let rx = w.register("b1".into(), WaitKind::Batch);
        for line in ["not json", "[1,2]", "42", r#"{"no":"op"}"#, r#"{"op":"dance","id":"b1"}"#] {
            assert_eq!(w.route(line), Routed::Garbage, "{line}");
        }
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)), "the batch is untouched");
    }

    // --- the deadline and the result reader -----------------------------------

    #[test]
    fn batch_deadline_sums_budgets_with_grace_and_caps_at_60s() {
        let d = |jobs: Value| batch_deadline(&json!({ "v": 1, "jobs": jobs }));
        // One job, no budget: 1000 + 1500.
        assert_eq!(d(json!([{ "id": "a" }])), Duration::from_millis(2500));
        // Two jobs, one with a budget: (250 + 1500) + (1000 + 1500).
        assert_eq!(d(json!([{ "id": "a", "budgetMs": 250 }, { "id": "b" }])), Duration::from_millis(4250));
        // Empty: one grace, never zero.
        assert_eq!(d(json!([])), Duration::from_millis(JOB_GRACE_MS));
        assert_eq!(batch_deadline(&json!({})), Duration::from_millis(JOB_GRACE_MS));
        // Four Python jobs at 20 s each would be 86 s: capped.
        let four: Vec<Value> = (0..4).map(|i| json!({ "id": i.to_string(), "budgetMs": 20_000 })).collect();
        assert_eq!(d(Value::Array(four)), MAX_BATCH_DEADLINE);
    }

    #[test]
    fn a_result_is_read_into_shape_and_a_refusal_is_refused() {
        let r = parse_result(json!({
            "v": 1, "engine": { "name": "zipp" },
            "results": [
                { "id": "a", "ok": true, "value": 42 },
                { "id": "b", "ok": true },
                { "id": "c", "ok": false, "errorKind": "guest", "error": "boom" }
            ]
        }))
        .unwrap();
        assert_eq!(r.engine, json!({ "name": "zipp" }));
        assert_eq!(r.result("a").unwrap().outcome, Ok(json!(42)));
        assert_eq!(r.result("b").unwrap().outcome, Ok(Value::Null));
        assert_eq!(
            r.result("c").unwrap().outcome,
            Err(JobError { kind: "guest".into(), message: "boom".into() })
        );
        assert!(r.result("d").is_none());
        assert_eq!(
            parse_result(json!({ "v": 1, "error": { "code": "invalid_request", "message": "job x: no source" } })),
            Err(HostError::Refused { message: "job x: no source".into() })
        );
        assert!(matches!(
            parse_result(json!({ "v": 1, "results": [{ "id": "a" }] })),
            Err(HostError::Failed { code, .. }) if code == "malformed_result"
        ));
        assert!(matches!(parse_result(json!({ "v": 1 })), Err(HostError::Failed { .. })));
    }

    #[test]
    fn an_oversized_request_is_refused_before_anything_is_spawned() {
        // A CLI path that does not exist: had the host tried to spawn, the
        // error would name it and health would be Unavailable.
        let host = ScriptHost::with_cli(
            CliInvocation::Node { script: PathBuf::from("Z:/nowhere/oaiy.mjs") },
            None,
        );
        let big = batch_request(vec![json!({ "id": "a", "mode": "program", "source": "x".repeat(MAX_REQUEST_BYTES) })]);
        match host.evaluate(&big) {
            Err(HostError::RequestTooLarge { bytes, cap }) => {
                assert!(bytes > MAX_REQUEST_BYTES);
                assert_eq!(cap, MAX_REQUEST_BYTES);
            }
            other => panic!("expected RequestTooLarge, got {other:?}"),
        }
        let h = host.health();
        assert_eq!(h.health, Health::Starting, "nothing was attempted");
        assert_eq!(h.children, 0);
    }

    // --- a stub `--serve` child through the real spawn path --------------------

    /// A Node on PATH, or the stub-CLI skip/fail policy `worker.rs` uses.
    fn node_on_path() -> Option<PathBuf> {
        let finder = if cfg!(windows) { "where" } else { "which" };
        let found = std::process::Command::new(finder)
            .arg("node")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty())
                    .map(PathBuf::from)
            });
        if found.is_none() && std::env::var_os("CI").is_some() {
            panic!("CI must have node on PATH for the stub script-host tests");
        }
        found
    }

    /// Shaped like `oaiy capabilities --json`, fixture values.
    fn stub_capabilities() -> Value {
        json!({
            "version": "0.2.0",
            "protocols": { "run": 1, "script": 1, "profile": 1 },
            "engine": {
                "name": "zipp", "status": "ready",
                "release": "vS.T.U", "version": "S.T.U", "revision": "feedbeef",
                "wasmSha256": "cd".repeat(32), "languages": ["javascript"]
            },
            "run": { "languages": ["javascript"], "defaultInstructionSteps": 50, "maxInstructionSteps": 2000 },
            "script": { "languages": ["javascript"], "defaultBudgetMs": 1000, "maxBudgetMs": 60000 }
        })
    }

    /// A CLI stand-in whose `script --serve` speaks the protocol and whose
    /// behaviour is chosen per job by its `source`:
    ///   * `crash` — exit 3 before answering;
    ///   * `hang` — never answer this batch;
    ///   * `unavailable` — an `engine_unavailable` line, then exit 1;
    ///   * `garbage:N` — N non-JSON lines, then the answer;
    ///   * `orphans` — a result for an id nobody sent and an unsolicited pong first;
    ///   * `env` — the value is what the child sees of the secrets;
    ///   * `clock:MS` — sleep MS, value `{start, end}`;
    ///   * anything else — the value is the source, verbatim.
    /// `ignore_shutdown`: keep running after `{op:"shutdown"}` (and EOF).
    fn stub_serve(tag: &str, caps: &Value, ignore_shutdown: bool) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("oaiy-stub-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("stub.mjs");
        let engine = {
            let mut e = caps["engine"].clone();
            e.as_object_mut().unwrap().remove("status");
            e
        };
        std::fs::write(
            &script,
            format!(
                r#"import readline from 'node:readline';
const a = process.argv.slice(2);
const CAPS = {caps};
const ENGINE = {engine};
const IGNORE_SHUTDOWN = {ignore};
if (a[0] === 'capabilities') {{ process.stdout.write(JSON.stringify(CAPS)); process.exit(0); }}
if (a[0] !== 'script' || a[1] !== '--serve') process.exit(2);
const write = (o) => process.stdout.write(JSON.stringify(o) + '\n');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let jobs = 0;
const rl = readline.createInterface({{ input: process.stdin }});
let queue = Promise.resolve();
rl.on('line', (line) => {{
  if (line.trim() === '') return;
  let m;
  try {{ m = JSON.parse(line); }} catch {{ write({{ op: 'error', id: null, error: {{ code: 'invalid_line', message: 'not JSON' }} }}); return; }}
  if (m.op === 'ping') {{ const p = {{ op: 'pong', engine: ENGINE, instance: 1, jobs }}; if (m.id !== undefined) p.id = m.id; write(p); return; }}
  if (m.op === 'shutdown') {{ if (IGNORE_SHUTDOWN) {{ setInterval(() => {{}}, 1000); return; }} rl.close(); queue.then(() => process.exit(0)); return; }}
  if (m.op !== 'batch') {{ write({{ op: 'error', id: m.id ?? null, error: {{ code: 'invalid_line', message: 'unknown op' }} }}); return; }}
  queue = queue.then(async () => {{
    const results = [];
    for (const job of m.request.jobs) {{
      const src = String(job.source ?? '');
      jobs += 1;
      if (src === 'crash') process.exit(3);
      if (src === 'hang') {{ await new Promise(() => {{}}); }}
      if (src === 'unavailable') {{ write({{ op: 'error', id: null, error: {{ code: 'engine_unavailable', message: 'ZIPP engine unavailable: stub says so' }} }}); process.exit(1); }}
      if (src.startsWith('garbage:')) {{ const n = Number(src.slice(8)); for (let i = 0; i < n; i++) process.stdout.write('not protocol ' + i + '\n'); }}
      if (src === 'orphans') {{ write({{ op: 'result', id: 'nobody', result: {{ v: 1, results: [] }} }}); write({{ op: 'pong', engine: ENGINE, instance: 1, jobs }}); }}
      if (src === 'env') {{
        results.push({{ id: job.id, ok: true, value: {{
          token: process.env.OAIY_SERVER_TOKEN ?? null, url: process.env.OAIY_SERVER_URL ?? null,
          leak: process.env.OPENAI_API_KEY ?? null, asset: process.env.OAIY_ZIPP_ASSET_DIR ?? null,
          path: process.env.PATH !== undefined || process.env.Path !== undefined,
        }} }});
        continue;
      }}
      if (src.startsWith('clock:')) {{ const start = Date.now(); await sleep(Number(src.slice(6))); results.push({{ id: job.id, ok: true, value: {{ start, end: Date.now() }} }}); continue; }}
      results.push({{ id: job.id, ok: true, value: src }});
    }}
    write({{ op: 'result', id: m.id, result: {{ v: 1, engine: ENGINE, results }} }});
  }});
}});
rl.on('close', () => {{ if (!IGNORE_SHUTDOWN) queue.then(() => process.exit(0)); }});
"#,
                caps = caps,
                engine = engine,
                ignore = if ignore_shutdown { "true" } else { "false" },
            ),
        )
        .unwrap();
        (dir, script)
    }

    fn host_for(script: PathBuf, node: &Path) -> ScriptHost {
        ScriptHost::with_cli(CliInvocation::Node { script }, Some(node.to_path_buf()))
    }

    fn host_for_backoff(script: PathBuf, node: &Path, backoff: Duration) -> ScriptHost {
        ScriptHost::with_cli_and_backoff(CliInvocation::Node { script }, Some(node.to_path_buf()), backoff)
    }

    fn job(id: &str, source: &str) -> Value {
        json!({ "id": id, "mode": "program", "source": source })
    }

    fn value_of(r: &Result<ScriptResponse, HostError>, id: &str) -> Value {
        match r {
            Ok(resp) => resp.result(id).unwrap_or_else(|| panic!("no result {id}: {resp:?}")).outcome.clone().unwrap(),
            Err(e) => panic!("expected a response, got {e:?}"),
        }
    }

    #[test]
    fn a_stub_serve_child_answers_batches_end_to_end_and_health_follows() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("echo", &stub_capabilities(), false);
        let host = host_for(script, &node);
        assert_eq!(host.health().health, Health::Starting, "nothing spawned yet");

        let r = host.evaluate(&batch_request(vec![job("a", "one"), job("b", "two")]));
        assert_eq!(value_of(&r, "a"), json!("one"));
        assert_eq!(value_of(&r, "b"), json!("two"));
        let h = host.health();
        assert_eq!(h.health, Health::Ready);
        assert_eq!(h.children, 1);
        assert_eq!(h.instance, 1, "from the child's pong");
        let engine = h.engine.expect("the probe's identity");
        assert_eq!(engine.wasm_sha256, "cd".repeat(32));
        assert_eq!(engine.protocols.get("script"), Some(&1));

        // A second batch reuses the child; orphan lines in between are dropped.
        let r = host.evaluate(&batch_request(vec![job("c", "orphans"), job("d", "three")]));
        assert_eq!(value_of(&r, "d"), json!("three"));
        assert_eq!(host.health().children, 1, "same child");

        // A ping interleaves with nothing in flight and reports the child's count.
        let pong = host.ping(Duration::from_secs(5)).expect("a pong");
        assert_eq!(pong["instance"], json!(1));
        assert_eq!(pong["jobs"], json!(4));

        host.shutdown();
        assert_eq!(host.health().health, Health::Starting, "no child, nothing wrong");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_child_never_sees_the_server_token_or_url() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        // The headless server carries its real token in its own environment;
        // the leaf-script child must not inherit it. Set the lot in THIS
        // process, then read what the child sees — and take them back out on
        // the way out, assertion or not, so no other test in this binary
        // inherits them.
        struct Planted(&'static [&'static str]);
        impl Drop for Planted {
            fn drop(&mut self) {
                for name in self.0 {
                    std::env::remove_var(name);
                }
            }
        }
        let _planted = Planted(&[
            "OAIY_SERVER_TOKEN",
            "OAIY_SERVER_URL",
            "OPENAI_API_KEY",
            "OAIY_ZIPP_ASSET_DIR",
        ]);
        std::env::set_var("OAIY_SERVER_TOKEN", "must-not-leak");
        std::env::set_var("OAIY_SERVER_URL", "http://127.0.0.1:1");
        std::env::set_var("OPENAI_API_KEY", "must-not-leak-either");
        std::env::set_var("OAIY_ZIPP_ASSET_DIR", "C:/somewhere/else");
        let (dir, script) = stub_serve("env", &stub_capabilities(), false);
        let host = host_for(script, &node);
        let r = host.evaluate(&batch_request(vec![job("e", "env")]));
        let seen = value_of(&r, "e");
        assert_eq!(seen["token"], Value::Null, "OAIY_SERVER_TOKEN reached the child: {seen}");
        assert_eq!(seen["url"], Value::Null, "OAIY_SERVER_URL reached the child: {seen}");
        assert_eq!(seen["leak"], Value::Null, "a NEVER_FORWARD key reached the child: {seen}");
        assert_eq!(seen["asset"], Value::Null, "OAIY_ZIPP_ASSET_DIR reached the child: {seen}");
        assert_eq!(seen["path"], json!(true), "the child still has a PATH");
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn health_flips_on_child_exit_and_back_after_the_backoff() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("crash", &stub_capabilities(), false);
        let host = host_for_backoff(script, &node, Duration::from_millis(400));
        assert_eq!(host.health().health, Health::Starting);
        let r = host.evaluate(&batch_request(vec![job("ok", "fine")]));
        assert_eq!(value_of(&r, "ok"), json!("fine"));
        let ready_at = host.health();
        assert_eq!(ready_at.health, Health::Ready);

        // The child dies mid-batch: the batch fails, health says so, and the
        // flip is timestamped.
        let r = host.evaluate(&batch_request(vec![job("x", "crash")]));
        match r {
            Err(HostError::Unavailable { reason, .. }) => {
                assert!(reason.contains("exited"), "{reason}");
                assert!(reason.contains("code 3"), "{reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        let down = host.health();
        assert!(matches!(down.health, Health::Unavailable { .. }), "{down:?}");
        assert!(down.since >= ready_at.since);
        assert_eq!(down.children, 1);

        // Inside the backoff: refused with how long to wait, nothing spawned.
        match host.evaluate(&batch_request(vec![job("y", "again")])) {
            Err(HostError::Unavailable { retry_after: Some(d), .. }) => assert!(d <= Duration::from_millis(400)),
            other => panic!("expected a backoff, got {other:?}"),
        }
        assert_eq!(host.health().children, 1, "no spawn inside the backoff");

        // After it: a replacement, and Ready again.
        thread::sleep(Duration::from_millis(450));
        let r = host.evaluate(&batch_request(vec![job("z", "back")]));
        assert_eq!(value_of(&r, "z"), json!("back"));
        let up = host.health();
        assert_eq!(up.health, Health::Ready);
        assert_eq!(up.children, 2);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cli_without_the_script_protocol_is_refused_before_any_child_is_spawned() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let mut caps = stub_capabilities();
        caps["protocols"] = json!({ "run": 1, "profile": 1 });
        let (dir, script) = stub_serve("noscript", &caps, false);
        let host = host_for(script.clone(), &node);
        match host.evaluate(&batch_request(vec![job("a", "one")])) {
            Err(HostError::Unavailable { reason, retry_after }) => {
                assert!(reason.contains("script protocol (none)"), "{reason}");
                assert!(reason.contains("\"run\":1"), "names what it does speak: {reason}");
                assert!(reason.contains(&script.to_string_lossy().to_string()), "names the CLI: {reason}");
                assert!(retry_after.is_some(), "a broken install is not re-probed on every event");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert!(matches!(host.health().health, Health::Unavailable { .. }));
        // The refusal costs no engine: the child was never started, so a
        // shutdown finds nothing to end.
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_engine_unavailable_line_marks_the_host_unavailable_with_the_childs_reason() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("unavail", &stub_capabilities(), false);
        let host = host_for(script, &node);
        match host.evaluate(&batch_request(vec![job("a", "unavailable")])) {
            Err(HostError::Unavailable { reason, retry_after }) => {
                assert!(reason.contains("stub says so"), "{reason}");
                assert!(retry_after.is_some());
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_on_stdout_is_tolerated_up_to_the_cap_and_retires_the_child_at_it() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("garbage", &stub_capabilities(), false);
        let host = host_for_backoff(script, &node, Duration::ZERO);
        // One short of the cap: logged, the answer still arrives, the child is kept.
        let under = format!("garbage:{}", MAX_UNPARSEABLE_LINES - 1);
        let r = host.evaluate(&batch_request(vec![job("a", &under)]));
        assert_eq!(value_of(&r, "a"), json!(under));
        assert_eq!(host.health().health, Health::Ready);
        // One more line reaches the cap: retired, the batch fails.
        match host.evaluate(&batch_request(vec![job("b", "garbage:1")])) {
            Err(HostError::Unavailable { reason, .. }) => {
                assert!(reason.contains("not protocol JSON"), "{reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert_eq!(host.health().children, 1);
        // And the next call gets a fresh child (backoff zero here).
        let r = host.evaluate(&batch_request(vec![job("c", "fresh")]));
        assert_eq!(value_of(&r, "c"), json!("fresh"));
        assert_eq!(host.health().children, 2);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_batch_past_its_deadline_kills_the_child_and_the_next_call_gets_a_new_one() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("hang", &stub_capabilities(), false);
        let host = host_for(script, &node);
        let started = Instant::now();
        let deadline = Duration::from_millis(600);
        match host.evaluate_within(&batch_request(vec![job("a", "hang")]), deadline) {
            Err(HostError::TimedOut { deadline: d }) => assert_eq!(d, deadline),
            other => panic!("expected TimedOut, got {other:?}"),
        }
        assert!(started.elapsed() < Duration::from_secs(10), "bounded: {:?}", started.elapsed());
        assert!(matches!(host.health().health, Health::Unavailable { .. }));
        // No backoff for a child THIS side ended: the next call works at once.
        let r = host.evaluate(&batch_request(vec![job("b", "after")]));
        assert_eq!(value_of(&r, "b"), json!("after"));
        assert_eq!(host.health().children, 2);
        assert_eq!(host.health().health, Health::Ready);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shutdown_is_bounded_even_when_the_child_ignores_it() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        // A cooperative child exits on its own, well inside the grace.
        let (dir, script) = stub_serve("bye", &stub_capabilities(), false);
        let host = host_for(script, &node);
        let r = host.evaluate(&batch_request(vec![job("a", "x")]));
        assert_eq!(value_of(&r, "a"), json!("x"));
        let live = host.inner.state.lock().unwrap().child.clone().expect("a child");
        let t = Instant::now();
        host.shutdown();
        assert!(t.elapsed() < SHUTDOWN_GRACE, "it exited on its own: {:?}", t.elapsed());
        assert_eq!(*live.exit_code.lock().unwrap(), Some(0), "exit 0 is the ack");
        let _ = std::fs::remove_dir_all(&dir);

        // One that ignores shutdown and EOF is killed at the grace.
        let (dir, script) = stub_serve("deaf", &stub_capabilities(), true);
        let host = host_for(script, &node);
        let r = host.evaluate(&batch_request(vec![job("a", "x")]));
        assert_eq!(value_of(&r, "a"), json!("x"));
        let live = host.inner.state.lock().unwrap().child.clone().expect("a child");
        let t = Instant::now();
        host.shutdown();
        let took = t.elapsed();
        assert!(took >= SHUTDOWN_GRACE, "waited the grace: {took:?}");
        assert!(took < SHUTDOWN_GRACE + Duration::from_secs(5), "then killed: {took:?}");
        assert!(live.gone.load(Ordering::Acquire));
        assert_ne!(*live.exit_code.lock().unwrap(), Some(0), "killed, not exited");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluate_from_two_threads_serialises_the_batches() {
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_serve("clock", &stub_capabilities(), false);
        let host = Arc::new(host_for(script, &node));
        let spans: Vec<(i64, i64)> = thread::scope(|s| {
            let handles: Vec<_> = (0..3)
                .map(|i| {
                    let host = host.clone();
                    s.spawn(move || {
                        let id = format!("t{i}");
                        let r = host.evaluate(&batch_request(vec![job(&id, "clock:150")]));
                        let v = value_of(&r, &id);
                        (v["start"].as_i64().unwrap(), v["end"].as_i64().unwrap())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut sorted = spans.clone();
        sorted.sort();
        for w in sorted.windows(2) {
            assert!(w[1].0 >= w[0].1, "batches overlapped in the child: {spans:?}");
        }
        assert_eq!(host.health().children, 1, "one child served all three");
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- the real staged CLI ---------------------------------------------------

    /// Where `desktop/scripts/sync-cli.mjs` stages the CLI and its engine.
    fn staged_cli_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("cli")
    }

    /// The bundled CLI, if this checkout has one staged, and a Node to run it —
    /// skipped on a dev box without either, a failure under CI (which stages both
    /// before `cargo test`), the policy `app_logic.rs` set.
    fn staged_runner() -> Option<(PathBuf, PathBuf)> {
        let cli = staged_cli_dir().join("oaiy.mjs");
        let found = if cli.is_file() { node_on_path().map(|node| (node, cli)) } else { None };
        if found.is_none() && std::env::var_os("CI").is_some() {
            panic!("CI must stage the CLI (node desktop/scripts/sync-cli.mjs) and have node on PATH before cargo test");
        }
        found
    }

    #[test]
    fn the_real_script_host_answers_a_javascript_job_and_a_python_project() {
        let Some((node, cli)) = staged_runner() else {
            eprintln!("no staged CLI or no node — skipping");
            return;
        };
        let host = host_for(cli, &node);
        let started = Instant::now();
        let request = batch_request(vec![
            json!({ "id": "js", "mode": "program", "source": "6 * 7" }),
            json!({ "id": "body", "mode": "body", "source": "return { n: [1, 2, 3].map((x) => x * 2) }" }),
            json!({
                "id": "py", "language": "python", "mode": "python-project",
                "files": {
                    "main.py": "import lib\n\ndef run(ctx):\n    return {\"v\": lib.twice(ctx[\"n\"]), \"ctx\": ctx}\n",
                    "lib.py": "def twice(n):\n    return n * 2\n"
                },
                "entry": "main", "call": "run", "args": [{ "n": 21 }], "budgetMs": 20_000
            }),
            json!({ "id": "throws", "mode": "program", "source": "throw new Error('nope')" }),
        ]);
        let r = host.evaluate(&request);
        eprintln!("real script host: first batch in {:?}", started.elapsed());
        assert_eq!(value_of(&r, "js"), json!(42));
        assert_eq!(value_of(&r, "body"), json!({ "n": [2, 4, 6] }));
        assert_eq!(value_of(&r, "py"), json!({ "v": 42, "ctx": { "n": 21 } }));
        let resp = r.as_ref().unwrap();
        match &resp.result("throws").unwrap().outcome {
            Err(JobError { kind, message }) => {
                assert_eq!(kind, "guest");
                assert!(message.contains("nope"), "{message}");
            }
            other => panic!("a throwing job is a guest error, got {other:?}"),
        }
        // The response's engine is the one the probe reported — from the CLI,
        // never from a literal here.
        let h = host.health();
        assert_eq!(h.health, Health::Ready);
        let engine = h.engine.expect("identity");
        assert_eq!(engine.name, "zipp");
        assert_eq!(resp.engine["wasmSha256"], json!(engine.wasm_sha256));
        assert_eq!(resp.engine["release"], json!(engine.release));
        assert!(h.instance >= 1);

        // Warm: the second batch is a line each way.
        let t = Instant::now();
        let r = host.evaluate(&batch_request(vec![json!({ "id": "again", "mode": "program", "source": "'warm'" })]));
        assert_eq!(value_of(&r, "again"), json!("warm"));
        eprintln!("real script host: warm batch in {:?}", t.elapsed());
        assert_eq!(host.health().children, 1);

        // A malformed request is the CLI's refusal, deterministic, nothing run.
        match host.evaluate(&batch_request(vec![json!({ "id": "bad", "mode": "program" })])) {
            Err(HostError::Refused { message }) => assert!(!message.is_empty()),
            other => panic!("expected the refusal, got {other:?}"),
        }
        assert_eq!(host.health().health, Health::Ready, "a refusal costs no child");

        let t = Instant::now();
        host.shutdown();
        assert!(t.elapsed() < SHUTDOWN_GRACE, "the real child exits on shutdown: {:?}", t.elapsed());
    }
}
