//! The desktop flow worker: claim queued runs, execute them through the CLI,
//! finalise the ledger.
//!
//! ```text
//!   ledger.claimable_by_worker() ──► claim ──► resolve flow file
//!                                               ──► spawn `oaiy run <flow> --inputs …`
//!                                                     ──► parse output / watch cancel
//!                                                           ──► ledger.finish(...)
//! ```
//!
//! # There is no engine here — that is the point
//!
//! The worker resolves a flow file and hands it to the `oaiy` CLI, which runs
//! the same `oaiy-core` the browser uses. The desktop never parses the graph:
//! parsing it is the first step towards reimplementing it, and a second engine
//! kept "in sync" by a parity test is exactly the FormLogic failure mode this
//! architecture exists to avoid (see `bridge/mod.rs`).
//!
//! # Flow storage
//!
//! Flows live as files under `<data>/flows/<id>.json`, pushed there over HTTP
//! (`PUT /api/bridge/flows/:id`). The id is validated against a strict charset
//! before touching the filesystem — it becomes a path component, and a permissive
//! id would make `PUT /api/bridge/flows/..%2F..%2Fevil` a file write anywhere.
//!
//! # CLI resolution
//!
//! In order: the `OAIY_CLI` env var (a path to `oaiy.mjs`/`oaiy.js` run via
//! node, or a native binary), then the CLI bundle SHIPPED with the app beside
//! the executable, then `oaiy` on `PATH`. The bundled step is what lets an
//! installed OAIY run flows with nothing else to install; an explicit env var
//! still wins so an operator's own build is never silently overridden. No CLI at
//! all is a **typed, actionable failure** on each run — `runtime_unavailable`,
//! naming the fixes — never a silent stall of the queue.

use crate::HiddenCommand as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use super::ledger::{ClaimOutcome, LedgerHandle, RunError, RunErrorCode, RunRecord, RunStatus};

/// How often the worker polls for claimable runs. In-process, so this can be
/// tight without any network cost; 500ms keeps a triggered flow feeling
/// immediate without busy-spinning.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Default wall-clock budget when the run specifies none.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
/// How often a running child is checked for exit / cancellation.
const CHILD_POLL: Duration = Duration::from_millis(250);
/// Cap on captured child output kept for error reporting.
const MAX_CAPTURE: usize = 16 * 1024;

/// Where flow files live and how ids map to paths.
pub struct FlowStore {
    dir: PathBuf,
}

impl FlowStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Validate an id before it becomes a path component.
    ///
    /// Strict charset, not sanitisation: rejecting `../evil` outright is
    /// verifiable, while "cleaning" it invites the next bypass.
    pub fn valid_id(id: &str) -> bool {
        !id.is_empty()
            && id.len() <= 128
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }

    pub fn path_of(&self, id: &str) -> Option<PathBuf> {
        Self::valid_id(id).then(|| self.dir.join(format!("{id}.json")))
    }

    /// Store a flow document. The desktop deliberately does NOT validate the
    /// graph — the CLI owns the schema, and validating here would be the first
    /// step towards a second engine. It checks only that the body is JSON, so a
    /// corrupted upload fails at PUT time rather than at run time.
    pub fn put(&self, id: &str, body: &str) -> Result<PathBuf, String> {
        let path = self
            .path_of(id)
            .ok_or_else(|| format!("invalid flow id {id:?}: use letters, digits, - and _"))?;
        serde_json::from_str::<Value>(body).map_err(|e| format!("body is not valid JSON: {e}"))?;
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, body).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        Ok(path)
    }

    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let path = self
            .path_of(id)
            .ok_or_else(|| format!("invalid flow id {id:?}"))?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    }

    /// List stored flows as (id, best-effort name).
    pub fn list(&self) -> Vec<(String, Option<String>)> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out: Vec<(String, Option<String>)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let id = name.strip_suffix(".json")?.to_string();
                if !Self::valid_id(&id) {
                    return None;
                }
                // Best-effort display name — never a parse requirement.
                let title = std::fs::read_to_string(e.path())
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .and_then(|v| {
                        v.get("name")
                            .or_else(|| v.get("title"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    });
                Some((id, title))
            })
            .collect();
        out.sort();
        out
    }
}

/// How the worker invokes the CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliInvocation {
    /// `node <script> run …` — `OAIY_CLI` pointed at the .mjs entry.
    Node { script: PathBuf },
    /// `<binary> run …`.
    Binary { path: PathBuf },
}

/// Candidate locations for the CLI bundle that ships INSIDE the app.
///
/// Tauri lays resources out beside the executable (`<install>/resources/…` on
/// Windows/Linux, `…/Contents/Resources/…` in a macOS bundle), and `cargo run`
/// leaves them under the target dir — so this probes the handful of places the
/// same file legitimately lands rather than depending on a Tauri API, which
/// keeps `worker.rs` usable from the headless binary that has no AppHandle.
fn bundled_cli_script() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let rel = [
        "resources/cli/oaiy.mjs",
        "../resources/cli/oaiy.mjs",
        "../Resources/cli/oaiy.mjs",
        // `cargo run` / `tauri dev`: resources are not copied, so fall back to
        // the source tree next to the crate.
        "../../resources/cli/oaiy.mjs",
    ];
    // NOT canonicalized: on Windows that yields a verbatim UNC path
    // (`\\?\C:\…`), which node fails to resolve as a main module — it reported
    // `EISDIR: illegal operation on a directory, lstat 'C:'`. `current_exe()` is
    // already absolute, so joining is enough.
    rel.iter().map(|r| dir.join(r)).find(|p| p.is_file())
}

/// Resolve how to invoke the OAIY CLI.
///
/// Order matters: an explicit `OAIY_CLI` wins (an operator pointing at a build
/// must not be silently overridden by what we ship), then the copy bundled with
/// the app, then whatever is on PATH. The bundled step is why an installed OAIY
/// can run flows without the user installing anything else — before it, a
/// packaged app resolved nothing and every run failed `runtime_unavailable`.
pub fn resolve_cli(
    env_value: Option<&str>,
    bundled: impl Fn() -> Option<PathBuf>,
    path_lookup: impl Fn(&str) -> Option<PathBuf>,
) -> Option<CliInvocation> {
    if let Some(raw) = env_value.map(str::trim).filter(|s| !s.is_empty()) {
        let p = PathBuf::from(raw);
        let is_js = p
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("mjs") || e.eq_ignore_ascii_case("js"));
        return Some(if is_js {
            CliInvocation::Node { script: p }
        } else {
            CliInvocation::Binary { path: p }
        });
    }
    if let Some(script) = bundled() {
        return Some(CliInvocation::Node { script });
    }
    path_lookup("oaiy").map(|path| CliInvocation::Binary { path })
}

/// `where`/`which` lookup for the default path.
fn lookup_on_path(name: &str) -> Option<PathBuf> {
    let finder = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(finder)
        .arg(name)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .pipe_hidden()
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(PathBuf::from)
}

/// How the CLI resolves right now, for the readiness endpoint.
///
/// Uses the SAME resolution a run does, so status cannot claim the runtime is
/// ready while an actual run fails with `runtime_unavailable` (or the reverse).
pub fn cli_status() -> Option<CliInvocation> {
    /// How long a resolution is reused. The CLI is bundled with the app or sits
    /// on PATH; neither changes while the process runs, short of the user
    /// installing one — which a minute of staleness covers.
    const TTL: std::time::Duration = std::time::Duration::from_secs(60);
    static CACHE: std::sync::Mutex<Option<(std::time::Instant, Option<CliInvocation>)>> =
        std::sync::Mutex::new(None);

    // Resolving falls through to `where oaiy`, i.e. a child process — and the
    // readiness endpoint that calls this is polled every few seconds by the UI.
    // Uncached, that spawned a process per poll to answer a question whose
    // answer does not change, inside an async handler where a blocking spawn
    // occupies a runtime worker.
    if let Ok(cache) = CACHE.lock() {
        if let Some((at, cached)) = cache.as_ref() {
            if at.elapsed() < TTL {
                return cached.clone();
            }
        }
    }
    let resolved = resolve_cli(
        std::env::var("OAIY_CLI").ok().as_deref(),
        bundled_cli_script,
        lookup_on_path,
    );
    if let Ok(mut cache) = CACHE.lock() {
        *cache = Some((std::time::Instant::now(), resolved.clone()));
    }
    resolved
}

pub struct Worker {
    ledger: LedgerHandle,
    flows: Arc<FlowStore>,
    device_label: String,
    stop: Arc<AtomicBool>,
    /// Resolves the Node runtime per run, so installing one mid-session is
    /// picked up without restarting the app.
    node: Option<crate::services::node_runtime::NodeHandle>,
}

impl Worker {
    /// Start the claim loop on a background thread. Returns a stop flag the app
    /// flips on exit.
    pub fn start(
        ledger: LedgerHandle,
        flows: Arc<FlowStore>,
        device_label: String,
        node: Option<crate::services::node_runtime::NodeHandle>,
    ) -> Arc<AtomicBool> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            ledger,
            flows,
            device_label,
            stop: stop.clone(),
            node,
        };
        thread::spawn(move || worker.run_loop());
        stop
    }

    fn run_loop(&self) {
        while !self.stop.load(Ordering::Relaxed) {
            let claimable = match self.ledger.lock() {
                Ok(l) => l.claimable_by_worker(3),
                Err(_) => Vec::new(),
            };
            for run in claimable {
                if self.stop.load(Ordering::Relaxed) {
                    return;
                }
                // Claim under the lock; execute outside it. Losing the claim is
                // normal — another worker or a browser got there first.
                let claimed = match self.ledger.lock() {
                    Ok(mut l) => l.claim(
                        &run.run_id,
                        super::ledger::Runtime::Desktop,
                        &self.device_label,
                    ),
                    Err(_) => continue,
                };
                if let ClaimOutcome::Claimed(rec) = claimed {
                    let outcome = self.execute(&rec);
                    let (status, output, error) = outcome;
                    if let Ok(mut l) = self.ledger.lock() {
                        let _ = l.finish(&rec.run_id, status, output, error);
                    }
                }
            }
            thread::sleep(POLL_INTERVAL);
        }
    }

    /// Execute one claimed run. Returns (status, output, error) for finish().
    fn execute(&self, run: &RunRecord) -> (RunStatus, Option<Value>, Option<RunError>) {
        let Some(flow_id) = run.flow_id.as_deref() else {
            return fail(
                RunErrorCode::InvalidFlow,
                "this run has no flowId; inline graphs are not executable on the desktop yet",
            );
        };
        let Some(flow_path) = self.flows.path_of(flow_id) else {
            return fail(
                RunErrorCode::InvalidRequest,
                &format!("flow id {flow_id:?} is not a valid identifier"),
            );
        };
        if !flow_path.is_file() {
            let known: Vec<String> = self.flows.list().into_iter().map(|(id, _)| id).collect();
            return (
                RunStatus::Failed,
                None,
                Some(
                    RunError::new(
                        RunErrorCode::FlowNotFound,
                        format!("no flow named {flow_id:?} is stored on this desktop"),
                    )
                    .with_detail(format!(
                        "Push it with PUT /api/bridge/flows/{flow_id}. Stored flows: {}",
                        if known.is_empty() { "(none)".into() } else { known.join(", ") }
                    )),
                ),
            );
        }

        // Inputs travel by file, not argv: values can be large, can contain
        // quoting hazards, and argv is visible to every process lister on the
        // machine — inputs may hold user data.
        let scratch = std::env::temp_dir().join(format!("oaiy-run-{}", run.run_id));
        if let Err(e) = std::fs::create_dir_all(&scratch) {
            return fail(RunErrorCode::Internal, &format!("cannot create scratch dir: {e}"));
        }
        let inputs_path = scratch.join("inputs.json");
        let out_path = scratch.join("result.json");
        let inputs_body = run
            .input
            .clone()
            .unwrap_or_else(|| Value::Object(Default::default()));
        if let Err(e) = std::fs::write(&inputs_path, inputs_body.to_string()) {
            return fail(RunErrorCode::Internal, &format!("cannot write inputs: {e}"));
        }

        let timeout = run
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT);

        let outcome = run_flow_cli(
            CliRequest {
                flow_path: &flow_path,
                inputs_path: &inputs_path,
                out_path: &out_path,
                // A locally stored flow talks to nothing on anyone's behalf.
                connector_path: None,
                timeout,
                node: self.node.as_ref(),
            },
            // Cancellation: the flag the cancel endpoint sets. Observed from
            // here because the worker is the only thing that can stop the work.
            &|| {
                self.ledger
                    .lock()
                    .ok()
                    .and_then(|l| l.get(&run.run_id))
                    .map(|r| r.cancel_requested)
                    .unwrap_or(false)
            },
        );

        let _ = std::fs::remove_dir_all(&scratch);
        match outcome {
            CliOutcome::Succeeded(v) => (RunStatus::Succeeded, Some(v), None),
            CliOutcome::Unreadable(why) => fail(RunErrorCode::Internal, &why),
            CliOutcome::Failed { exit_code, detail } => (
                RunStatus::Failed,
                None,
                Some(
                    RunError::new(
                        RunErrorCode::NodeFailed,
                        format!("the flow failed (CLI exit {exit_code})"),
                    )
                    .with_detail(detail)
                    .retryable(),
                ),
            ),
            CliOutcome::TimedOut => (
                RunStatus::TimedOut,
                None,
                Some(RunError::new(
                    RunErrorCode::Timeout,
                    format!(
                        "the flow exceeded its {}s budget and was killed",
                        timeout.as_secs()
                    ),
                )),
            ),
            CliOutcome::Cancelled => (
                RunStatus::Cancelled,
                None,
                Some(RunError::new(
                    RunErrorCode::Cancelled,
                    "cancelled while running; side effects already performed stay performed",
                )),
            ),
            CliOutcome::Unavailable {
                message,
                detail,
                retryable,
            } => {
                let mut e = RunError::new(RunErrorCode::RuntimeUnavailable, message);
                if let Some(d) = detail {
                    e = e.with_detail(d);
                }
                if retryable {
                    e = e.retryable();
                }
                (RunStatus::Failed, None, Some(e))
            }
        }
    }
}

/// One CLI invocation, by file: which graph, which inputs, where the result goes.
///
/// Everything travels as a path rather than argv because values can be large,
/// can contain quoting hazards, and argv is visible to every process lister on
/// the machine — inputs and connector credentials are exactly what must not be.
pub struct CliRequest<'a> {
    pub flow_path: &'a Path,
    pub inputs_path: &'a Path,
    pub out_path: &'a Path,
    /// Connector config handed over with `--connector`, when the graph is a
    /// linked provider's and its nodes must be able to reach that provider.
    /// `None` for a flow stored on this desktop, which speaks for nobody.
    pub connector_path: Option<&'a Path>,
    pub timeout: Duration,
    pub node: Option<&'a crate::services::node_runtime::NodeHandle>,
}

/// How a CLI invocation ended.
///
/// A closed set with no "unknown": every arm has to become a reported outcome,
/// because a run that started and was never answered is worse than one that
/// never started.
#[derive(Debug)]
pub enum CliOutcome {
    Succeeded(Value),
    /// Exit 0 but the result file is missing or is not JSON. Ours, not the
    /// flow's — hence separate from `Failed`.
    Unreadable(String),
    /// The CLI ran and reported failure. `detail` is the useful tail of stderr.
    Failed { exit_code: i32, detail: String },
    TimedOut,
    Cancelled,
    /// There is no CLI, or it would not start. Actionable, never silent.
    Unavailable {
        message: String,
        detail: Option<String>,
        /// True when installing something makes this work — as opposed to a
        /// launch that failed on its own terms, which retrying will repeat.
        retryable: bool,
    },
}

/// Spawn the bundled CLI on one flow and wait for it.
///
/// The single place this crate starts a flow engine: the desktop worker uses it
/// for a locally stored flow, and the link's flow runner for a graph claimed
/// from the provider. Two copies of this would drift on the parts that are easy
/// to get wrong — draining both pipes, adopting the child into the job object,
/// dropping credentials from its environment.
pub fn run_flow_cli(req: CliRequest, cancelled: &dyn Fn() -> bool) -> CliOutcome {
    let Some(cli) = resolve_cli(
        std::env::var("OAIY_CLI").ok().as_deref(),
        bundled_cli_script,
        lookup_on_path,
    ) else {
        return CliOutcome::Unavailable {
            message: "the OAIY CLI is not available on this machine".into(),
            detail: Some(
                "Install the `oaiy` CLI so it is on PATH, or set OAIY_CLI to the path of \
                 cli/bin/oaiy.mjs. The desktop runs flows through the CLI so desktop and \
                 browser execution share one engine."
                    .into(),
            ),
            retryable: true,
        };
    };

    let timeout_secs = req.timeout.as_secs().max(1).to_string();
    let mut cmd = match &cli {
        CliInvocation::Node { script } => {
            // Prefer a resolved Node (portable install, else PATH) over the
            // bare name: a packaged app cannot assume `node` is on PATH.
            let exe = req
                .node
                .and_then(|n| n.resolve())
                .unwrap_or_else(|| PathBuf::from("node"));
            let mut c = Command::new(exe);
            c.arg(script);
            c
        }
        CliInvocation::Binary { path } => Command::new(path),
    };
    cmd.arg("run")
        .arg(req.flow_path)
        .arg("--inputs")
        .arg(req.inputs_path)
        .arg("-o")
        .arg(req.out_path)
        .arg("--timeout")
        .arg(&timeout_secs);
    if let Some(connector) = req.connector_path {
        cmd.arg("--connector").arg(connector);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Drop known credential env vars before the CLI inherits the rest.
    //
    // A flow is untrusted code — writable over HTTP, claimable from a linked
    // provider, and deliberately never graph-validated here — so the same
    // reasoning that gives plugins an allow-listed environment applies. The CLI
    // legitimately needs far more than a plugin (it IS the engine: PATH, HOME,
    // node's own vars), so this is a deny-list of the sensitive names rather
    // than an allow-list, and it matters because the engine's getSecret() reads
    // process.env by name BEFORE its own store — an inherited AWS/OpenAI key
    // would be directly addressable from inside a flow. A flow that genuinely
    // needs a cloud key should carry it as a constant, not inherit it
    // ambiently. Reuses the plugin host's list so the two paths cannot drift.
    for name in crate::plugins::runner::NEVER_FORWARD {
        cmd.env_remove(name);
    }
    // Tell the child where THIS desktop's API is.
    //
    // A connector node that chats, calls a plugin connector or touches a
    // service comes back to us over loopback, and the child had no way to know
    // the port — it fell back to a default that happens to be right only while
    // nobody changes it. On a desktop started with a different port those
    // operations would quietly address whatever else is listening there.
    cmd.env(
        "OAIY_SERVER_URL",
        format!("http://127.0.0.1:{}", crate::DESKTOP_PORT),
    );
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return CliOutcome::Unavailable {
                message: format!("could not launch the OAIY CLI ({cli:?}): {e}"),
                detail: None,
                retryable: false,
            }
        }
    };
    // A flow run makes HTTP calls and writes files. Orphaned by a forced exit it
    // keeps doing both, with no ledger entry left to record it and no timeout
    // left to stop it.
    crate::services::job_object::adopt(child.id());

    // Drain BOTH pipes on their own threads, from the start. The first cut read
    // stderr only after exit and stdout never — so a CLI that logged more than
    // one OS pipe buffer (~4-64 KB; routine for a verbose Node process) blocked
    // in write(), could never exit, and was killed at the deadline as a false
    // `timed_out`. The review traced the whole chain, and this crate documents
    // the identical hazard for plugin stderr.
    let captured_err = Arc::new(std::sync::Mutex::new(String::new()));
    if let Some(err) = child.stderr.take() {
        let sink = captured_err.clone();
        thread::spawn(move || drain_capped(err, &sink));
    }
    let captured_out = Arc::new(std::sync::Mutex::new(String::new()));
    if let Some(out) = child.stdout.take() {
        let sink = captured_out.clone();
        thread::spawn(move || drain_capped(out, &sink));
    }

    // Grace over the CLI's own timeout so the CLI gets to time out FIRST and
    // report which node was stuck — killing from out here loses that.
    let deadline = Instant::now() + req.timeout + Duration::from_secs(10);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(_) => break None,
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        if cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return CliOutcome::Cancelled;
        }
        thread::sleep(CHILD_POLL);
    };

    let capture = captured_err.lock().map(|g| g.clone()).unwrap_or_default();

    match status {
        Some(st) if st.success() => match std::fs::read_to_string(req.out_path) {
            Ok(body) => match serde_json::from_str::<Value>(&body) {
                Ok(v) => CliOutcome::Succeeded(v),
                Err(e) => CliOutcome::Unreadable(format!(
                    "the CLI reported success but its output is not JSON: {e}"
                )),
            },
            Err(_) => CliOutcome::Unreadable(
                "the CLI reported success but wrote no result file".to_string(),
            ),
        },
        Some(st) => CliOutcome::Failed {
            exit_code: st.code().unwrap_or(-1),
            detail: tail_of(&capture, 1500),
        },
        None => CliOutcome::TimedOut,
    }
}

/// Read a pipe to EOF, keeping at most [`MAX_CAPTURE`] bytes.
///
/// The read must continue past the cap — stopping would refill the pipe and
/// recreate the deadlock the cap exists to report on. Late bytes overwrite
/// nothing; the buffer simply stops growing, and `tail_of` later keeps the end.
fn drain_capped<R: std::io::Read>(mut reader: R, sink: &std::sync::Mutex<String>) {
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(mut s) = sink.lock() {
                    if s.len() < MAX_CAPTURE {
                        let room = MAX_CAPTURE - s.len();
                        let chunk = String::from_utf8_lossy(&buf[..n]);
                        if chunk.len() <= room {
                            s.push_str(&chunk);
                        } else {
                            let mut end = room;
                            while end > 0 && !chunk.is_char_boundary(end) {
                                end -= 1;
                            }
                            s.push_str(&chunk[..end]);
                        }
                    }
                }
            }
        }
    }
}

fn fail(code: RunErrorCode, msg: &str) -> (RunStatus, Option<Value>, Option<RunError>) {
    (RunStatus::Failed, None, Some(RunError::new(code, msg)))
}

/// Last `n` chars of captured output, so the error carries the useful end of a
/// stack trace rather than its preamble.
fn tail_of(s: &str, n: usize) -> String {
    let t = s.trim();
    if t.len() <= n {
        t.to_string()
    } else {
        let start = t.len() - n;
        // Don't split a UTF-8 char.
        let start = (start..t.len()).find(|i| t.is_char_boundary(*i)).unwrap_or(start);
        format!("…{}", &t[start..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CLI resolution order ---------------------------------------------

    #[test]
    fn an_explicit_oaiy_cli_wins_over_everything_else() {
        // An operator pointing at their own build must never be silently
        // overridden by the copy we ship.
        let r = resolve_cli(Some("C:/dev/oaiy-web/cli/bin/oaiy.mjs"), || None, |_| {
            Some(PathBuf::from("C:/on/path/oaiy.exe"))
        });
        match r {
            Some(CliInvocation::Node { script }) => {
                assert!(script.to_string_lossy().contains("cli/bin/oaiy.mjs"));
            }
            other => panic!("expected the explicit .mjs, got {other:?}"),
        }
    }

    #[test]
    fn a_non_js_oaiy_cli_is_treated_as_a_binary() {
        match resolve_cli(Some("/usr/local/bin/oaiy"), || None, |_| None) {
            Some(CliInvocation::Binary { path }) => assert!(path.ends_with("oaiy")),
            other => panic!("expected a binary, got {other:?}"),
        }
    }

    #[test]
    fn blank_or_absent_env_does_not_count_as_an_override() {
        // Whitespace must not resolve to a nonsense empty path — it has to fall
        // through to the bundled/PATH lookup like an unset variable.
        for env in [Some("   "), Some(""), None] {
            let r = resolve_cli(env, || None, |_| Some(PathBuf::from("/found/on/path/oaiy")));
            // Either the bundled script (when this checkout has one staged) or
            // the PATH hit — never an empty Node script path.
            match r {
                Some(CliInvocation::Node { script }) => assert!(!script.as_os_str().is_empty()),
                Some(CliInvocation::Binary { path }) => assert!(!path.as_os_str().is_empty()),
                None => panic!("expected a resolution for env {env:?}"),
            }
        }
    }

    #[test]
    fn the_bundled_cli_is_used_before_path_but_after_an_explicit_env() {
        let bundled = || Some(PathBuf::from("C:/app/resources/cli/oaiy.mjs"));
        let on_path = |_: &str| Some(PathBuf::from("C:/on/path/oaiy.exe"));

        // No env → the copy we ship wins over PATH, which is what makes an
        // installed app work without the user installing anything.
        match resolve_cli(None, bundled, on_path) {
            Some(CliInvocation::Node { script }) => {
                assert!(script.to_string_lossy().contains("resources/cli"))
            }
            other => panic!("expected the bundled script, got {other:?}"),
        }
        // An explicit env var still beats the bundle.
        match resolve_cli(Some("C:/dev/oaiy.mjs"), bundled, on_path) {
            Some(CliInvocation::Node { script }) => {
                assert!(script.to_string_lossy().contains("dev"))
            }
            other => panic!("expected the explicit path, got {other:?}"),
        }
        // No env and no bundle → PATH.
        match resolve_cli(None, || None, on_path) {
            Some(CliInvocation::Binary { path }) => {
                assert!(path.to_string_lossy().contains("on/path"))
            }
            other => panic!("expected the PATH binary, got {other:?}"),
        }
        // Nothing anywhere → the honest None the readiness endpoint reports.
        assert_eq!(resolve_cli(None, || None, |_| None), None);
    }

    // --- flow ids become path components ----------------------------------

    #[test]
    fn flow_ids_are_a_strict_charset() {
        for good in ["caller-lookup", "flow_1", "ABC123"] {
            assert!(FlowStore::valid_id(good), "{good}");
        }
        for bad in [
            "", "../evil", "..\\evil", "a/b", "a\\b", "a.json", "a b", "a%2Fb",
            &"x".repeat(200),
        ] {
            assert!(!FlowStore::valid_id(bad), "{bad:?} must be refused");
        }
    }

    #[test]
    fn a_bad_id_never_touches_the_filesystem() {
        let store = FlowStore::new(std::env::temp_dir().join("oaiy-flowstore-nowhere"));
        assert!(store.path_of("../../escape").is_none());
        assert!(store.put("../../escape", "{}").is_err());
    }

    #[test]
    fn put_requires_json_but_not_a_schema() {
        // The CLI owns the schema; the store only refuses corruption. Validating
        // the graph here would be the first step towards a second engine.
        let dir = std::env::temp_dir().join(format!("oaiy-flowstore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FlowStore::new(dir.clone());
        assert!(store.put("f1", "{ not json").is_err());
        assert!(store.put("f1", r#"{"anything": "goes", "name": "Demo"}"#).is_ok());
        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0, "f1");
        assert_eq!(listed[0].1.as_deref(), Some("Demo"));
        assert!(store.delete("f1").unwrap());
        assert!(!store.delete("f1").unwrap(), "second delete reports absent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- CLI resolution ----------------------------------------------------

    #[test]
    fn oaiy_cli_env_wins_and_js_goes_through_node() {
        let r = resolve_cli(Some(r"C:\repo\cli\bin\oaiy.mjs"), || None, |_| {
            panic!("PATH must not be consulted when the env var is set")
        });
        assert_eq!(
            r,
            Some(CliInvocation::Node {
                script: PathBuf::from(r"C:\repo\cli\bin\oaiy.mjs")
            })
        );
        let r = resolve_cli(Some(r"C:\tools\oaiy.exe"), || None, |_| unreachable!());
        assert!(matches!(r, Some(CliInvocation::Binary { .. })));
    }

    #[test]
    fn a_blank_env_var_falls_through_to_path() {
        let r = resolve_cli(Some("   "), || None, |name| {
            assert_eq!(name, "oaiy");
            Some(PathBuf::from("/usr/local/bin/oaiy"))
        });
        assert!(matches!(r, Some(CliInvocation::Binary { .. })));
    }

    #[test]
    fn no_cli_anywhere_is_none_not_a_guess() {
        assert_eq!(resolve_cli(None, || None, |_| None), None);
    }

    // --- output capture ----------------------------------------------------

    #[test]
    fn the_secret_deny_list_covers_the_known_credentials() {
        // The CLI child drops these before inheriting the rest. Pin the list so a
        // rename or a new provider key is a conscious decision, and assert the
        // engine-relevant ones are present.
        let deny = crate::plugins::runner::NEVER_FORWARD;
        for expected in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OAIY_HF_TOKEN", "AWS_SECRET_ACCESS_KEY"] {
            assert!(deny.contains(&expected), "{expected} must be denied to the CLI child");
        }
    }

    #[test]
    fn tail_keeps_the_end_of_a_stack_trace() {
        let long = format!("{}THE ACTUAL ERROR", "preamble ".repeat(500));
        let t = tail_of(&long, 100);
        assert!(t.contains("THE ACTUAL ERROR"), "{t}");
        assert!(t.starts_with('…'));
        assert!(tail_of("short", 100) == "short");
    }
}
