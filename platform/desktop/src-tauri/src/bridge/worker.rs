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
//! node, or a native binary), then `oaiy` on `PATH`. No CLI is a **typed,
//! actionable failure** on each run — `runtime_unavailable`, naming both fixes —
//! never a silent stall of the queue.

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

/// Resolve the CLI. Pure over its inputs so the policy is testable.
pub fn resolve_cli(env_value: Option<&str>, path_lookup: impl Fn(&str) -> Option<PathBuf>) -> Option<CliInvocation> {
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
    path_lookup("oaiy").map(|path| CliInvocation::Binary { path })
}

/// `where`/`which` lookup for the default path.
fn lookup_on_path(name: &str) -> Option<PathBuf> {
    let finder = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(finder)
        .arg(name)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
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

pub struct Worker {
    ledger: LedgerHandle,
    flows: Arc<FlowStore>,
    device_label: String,
    stop: Arc<AtomicBool>,
}

impl Worker {
    /// Start the claim loop on a background thread. Returns a stop flag the app
    /// flips on exit.
    pub fn start(
        ledger: LedgerHandle,
        flows: Arc<FlowStore>,
        device_label: String,
    ) -> Arc<AtomicBool> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            ledger,
            flows,
            device_label,
            stop: stop.clone(),
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

        let Some(cli) = resolve_cli(std::env::var("OAIY_CLI").ok().as_deref(), lookup_on_path)
        else {
            return (
                RunStatus::Failed,
                None,
                Some(
                    RunError::new(
                        RunErrorCode::RuntimeUnavailable,
                        "the OAIY CLI is not available on this machine",
                    )
                    .with_detail(
                        "Install the `oaiy` CLI so it is on PATH, or set OAIY_CLI to the path of \
                         cli/bin/oaiy.mjs. The desktop runs flows through the CLI so desktop and \
                         browser execution share one engine.",
                    )
                    .retryable(),
                ),
            );
        };

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
        let timeout_secs = timeout.as_secs().max(1).to_string();

        let mut cmd = match &cli {
            CliInvocation::Node { script } => {
                let mut c = Command::new("node");
                c.arg(script);
                c
            }
            CliInvocation::Binary { path } => Command::new(path),
        };
        cmd.arg("run")
            .arg(&flow_path)
            .arg("--inputs")
            .arg(&inputs_path)
            .arg("-o")
            .arg(&out_path)
            .arg("--timeout")
            .arg(&timeout_secs)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return fail(
                    RunErrorCode::RuntimeUnavailable,
                    &format!("could not launch the OAIY CLI ({cli:?}): {e}"),
                )
            }
        };

        // Grace over the CLI's own timeout so the CLI gets to time out FIRST and
        // report which node was stuck — killing from out here loses that.
        let deadline = Instant::now() + timeout + Duration::from_secs(10);
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
            // Cancellation: the flag the cancel endpoint sets. Observed here
            // because the worker is the only thing that can actually stop work.
            let cancelled = self
                .ledger
                .lock()
                .ok()
                .and_then(|l| l.get(&run.run_id))
                .map(|r| r.cancel_requested)
                .unwrap_or(false);
            if cancelled {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_dir_all(&scratch);
                return (
                    RunStatus::Cancelled,
                    None,
                    Some(RunError::new(
                        RunErrorCode::Cancelled,
                        "cancelled while running; side effects already performed stay performed",
                    )),
                );
            }
            thread::sleep(CHILD_POLL);
        };

        let mut capture = String::new();
        if let Some(mut err) = child.stderr.take() {
            use std::io::Read;
            let _ = err.by_ref().take(MAX_CAPTURE as u64).read_to_string(&mut capture);
        }

        let result = match status {
            Some(st) if st.success() => match std::fs::read_to_string(&out_path) {
                Ok(body) => match serde_json::from_str::<Value>(&body) {
                    Ok(v) => (RunStatus::Succeeded, Some(v), None),
                    Err(e) => fail(
                        RunErrorCode::Internal,
                        &format!("the CLI reported success but its output is not JSON: {e}"),
                    ),
                },
                Err(_) => fail(
                    RunErrorCode::Internal,
                    "the CLI reported success but wrote no result file",
                ),
            },
            Some(st) => (
                RunStatus::Failed,
                None,
                Some(
                    RunError::new(
                        RunErrorCode::NodeFailed,
                        format!("the flow failed (CLI exit {})", st.code().unwrap_or(-1)),
                    )
                    .with_detail(tail_of(&capture, 1500))
                    .retryable(),
                ),
            ),
            None => (
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
        };

        let _ = std::fs::remove_dir_all(&scratch);
        result
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
        let r = resolve_cli(Some(r"C:\repo\cli\bin\oaiy.mjs"), |_| {
            panic!("PATH must not be consulted when the env var is set")
        });
        assert_eq!(
            r,
            Some(CliInvocation::Node {
                script: PathBuf::from(r"C:\repo\cli\bin\oaiy.mjs")
            })
        );
        let r = resolve_cli(Some(r"C:\tools\oaiy.exe"), |_| unreachable!());
        assert!(matches!(r, Some(CliInvocation::Binary { .. })));
    }

    #[test]
    fn a_blank_env_var_falls_through_to_path() {
        let r = resolve_cli(Some("   "), |name| {
            assert_eq!(name, "oaiy");
            Some(PathBuf::from("/usr/local/bin/oaiy"))
        });
        assert!(matches!(r, Some(CliInvocation::Binary { .. })));
    }

    #[test]
    fn no_cli_anywhere_is_none_not_a_guess() {
        assert_eq!(resolve_cli(None, |_| None), None);
    }

    // --- output capture ----------------------------------------------------

    #[test]
    fn tail_keeps_the_end_of_a_stack_trace() {
        let long = format!("{}THE ACTUAL ERROR", "preamble ".repeat(500));
        let t = tail_of(&long, 100);
        assert!(t.contains("THE ACTUAL ERROR"), "{t}");
        assert!(t.starts_with('…'));
        assert!(tail_of("short", 100) == "short");
    }
}
