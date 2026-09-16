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
//!
//! # The runner has to be ZIPP
//!
//! The CLI runs every flow on the ZIPP VM, and this desktop refuses one that
//! does not. Before the first run through a resolved CLI (and again after the
//! file changes, or a minute after a refusal) it asks `capabilities --json` and
//! requires `engine.name == "zipp"` with `engine.status == "ready"`; an
//! `OAIY_CLI` override pointing at some other build, or a shipped copy staged
//! without its engine, is `runtime_unavailable` naming the fix — never a run
//! on whatever JavaScript the host happened to have. Every payload is then read
//! the same way: one that does not say `engine: "zipp"` is not a result, and
//! `errorCode` `engine_unavailable`/`timeout` are the host's verdicts, not the
//! flow's. Nothing here names a release, a digest or a revision: identity comes
//! from the CLI's own report.

use crate::HiddenCommand as _;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

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

impl CliInvocation {
    /// The file this resolves to, whichever way it is started.
    pub fn path(&self) -> &Path {
        match self {
            CliInvocation::Node { script } => script,
            CliInvocation::Binary { path } => path,
        }
    }
}

// ---------------------------------------------------------------------------
// The engine probe: is this CLI a ZIPP runner?
// ---------------------------------------------------------------------------

/// What the CLI says it is, from `capabilities --json`.
///
/// Every value is copied from that report. None is written here: the release
/// and the digest are whatever the staged `SOURCE.json` recorded when the CLI
/// was built, and a literal in this file would be a second, drifting source.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineIdentity {
    pub name: String,
    pub release: String,
    pub version: String,
    pub revision: String,
    pub wasm_sha256: String,
    /// What THIS CLI runs (`run.languages`) — not what the engine bundle could.
    /// Kept for the heartbeat and for a second language later; nothing reads
    /// it yet, and nothing here must advertise a language no host runs.
    pub run_languages: Vec<String>,
    /// The CLI's `protocols` map, every integer entry of it (`run`, `script`,
    /// `profile`, …), so a consumer that needs a protocol other than `run` —
    /// the script host needs `script` — can require it from the same report
    /// instead of probing again. Non-integer values are dropped; unknown keys
    /// are kept, not refused.
    pub protocols: std::collections::BTreeMap<String, u64>,
}

/// The answer to "can this CLI run a flow on ZIPP right now?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineProbe {
    Ready(EngineIdentity),
    /// It cannot, and `reason` says why in the CLI's own words (or which check
    /// of the report failed).
    Unavailable { reason: String },
}

impl EngineProbe {
    pub fn is_ready(&self) -> bool {
        matches!(self, EngineProbe::Ready(_))
    }
}

/// A lower-case hex sha256, or not a digest at all.
///
/// The same rule `desktop/scripts/sync-cli-lib.mjs` already enforces on the
/// staging side (`/^[0-9a-f]{64}$/`), so both ends of the chain agree on what a
/// digest is rather than one end trusting whatever the other wrote.
fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The head of a digest, for a log line.
///
/// `get`, not `[..12]`: a byte range bounded by `min` bounds the LENGTH, not a
/// UTF-8 character boundary, so a digest with a multi-byte character in it
/// panicked the thread that took the probe — the bridge's run loop, or the
/// warm-up, which then left readiness stuck (see [`ClearOnDrop`]). Logging is
/// never worth a panic, whatever [`is_hex64`] has already refused upstream.
fn short_digest(sha: &str) -> &str {
    sha.get(..12).unwrap_or(sha)
}

/// Read a `capabilities --json` report. Pure, so the shape is pinned without a
/// process.
///
/// Requires `protocols.run == 1` (the payload shape this file reads),
/// `engine.name == "zipp"`, `engine.status == "ready"` and an
/// `engine.wasmSha256` that IS a sha256. An unavailable engine's
/// `engine.reason` is the error, verbatim: it is the CLI's own account of what
/// is wrong with its staged artifact, which is what the user needs.
///
/// The digest is checked here and nowhere else, so everything downstream — the
/// heartbeat, the script host's identity match, the log line — has 64 hex
/// characters or has nothing.
pub fn parse_capabilities(v: &Value) -> Result<EngineIdentity, String> {
    let run_protocol = v.pointer("/protocols/run").and_then(Value::as_u64);
    if run_protocol != Some(1) {
        return Err(format!(
            "the CLI speaks run protocol {}, and this desktop reads protocol 1",
            run_protocol.map_or("(none)".to_string(), |n| n.to_string())
        ));
    }
    let engine = v
        .get("engine")
        .filter(|e| e.is_object())
        .ok_or_else(|| "the CLI's capabilities report names no engine".to_string())?;
    let text = |key: &str| -> Option<String> {
        engine.get(key).and_then(Value::as_str).map(str::to_string)
    };
    let name = text("name").unwrap_or_default();
    if name != "zipp" {
        return Err(format!(
            "the CLI runs user logic on {}, not on ZIPP",
            if name.is_empty() { "an unnamed engine".to_string() } else { format!("{name:?}") }
        ));
    }
    match text("status").as_deref() {
        Some("ready") => {}
        Some(status) => {
            return Err(text("reason").unwrap_or_else(|| {
                format!("the CLI reports its ZIPP engine as {status:?} without saying why")
            }))
        }
        None => return Err("the CLI's capabilities report gives its engine no status".to_string()),
    }
    let wasm_sha256 = text("wasmSha256").unwrap_or_default();
    if !is_hex64(&wasm_sha256) {
        return Err(format!(
            "the CLI names its engine digest {}, which is not a sha256",
            if wasm_sha256.is_empty() {
                "nowhere".to_string()
            } else {
                format!("{wasm_sha256:?}")
            }
        ));
    }
    let run_languages = v
        .pointer("/run/languages")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    let protocols = v
        .get("protocols")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, n)| n.as_u64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default();
    Ok(EngineIdentity {
        name,
        release: text("release").unwrap_or_default(),
        version: text("version").unwrap_or_default(),
        revision: text("revision").unwrap_or_default(),
        wasm_sha256,
        run_languages,
        protocols,
    })
}

/// How long `capabilities --json` may take. It hashes and compiles the staged
/// wasm (~8 MB) — a second or two — and never runs a flow.
pub(crate) const PROBE_DEADLINE: Duration = Duration::from_secs(15);
/// How long a refusal is remembered before the CLI is asked again, so a fixed
/// install recovers without a restart. A `Ready` answer is kept until the CLI
/// file itself changes, or until a RUN finds the engine gone
/// ([`remember_run_refusal`]) — after which this TTL governs its recovery too.
const PROBE_TTL: Duration = Duration::from_secs(60);

/// Start `<cli> …` the way a run does: through the resolved Node when the CLI
/// is a script, directly when it is a binary.
pub(crate) fn cli_command(cli: &CliInvocation, node_exe: Option<&Path>) -> Command {
    match cli {
        CliInvocation::Node { script } => {
            // Prefer a resolved Node (portable install, else PATH) over the
            // bare name: a packaged app cannot assume `node` is on PATH.
            let mut c = Command::new(node_exe.map_or_else(|| PathBuf::from("node"), Path::to_path_buf));
            c.arg(script);
            c
        }
        CliInvocation::Binary { path } => Command::new(path),
    }
}

/// The environment and window discipline every CLI child gets — a flow run and
/// the probe alike, from one place so the two cannot drift.
///
/// Drop known credential env vars before the CLI inherits the rest.
///
/// A flow is untrusted code — writable over HTTP, claimable from a linked
/// provider, and deliberately never graph-validated here — so the same
/// reasoning that gives plugins an allow-listed environment applies. The CLI
/// legitimately needs far more than a plugin (it IS the engine: PATH, HOME,
/// node's own vars), so this is a deny-list of the sensitive names rather
/// than an allow-list, and it matters because the engine's getSecret() reads
/// process.env by name BEFORE its own store — an inherited AWS/OpenAI key
/// would be directly addressable from inside a flow. A flow that genuinely
/// needs a cloud key should carry it as a constant, not inherit it
/// ambiently. Reuses the plugin host's list so the two paths cannot drift.
///
/// `OAIY_ZIPP_ASSET_DIR` goes too: it points the CLI at an engine folder other
/// than the one staged beside it. The CLI's digest check makes a wrong folder
/// inert, but the desktop ships exactly one engine and a child of this process
/// should not be told to look for another.
pub(crate) fn harden_child(cmd: &mut Command) {
    for name in crate::plugins::runner::NEVER_FORWARD {
        cmd.env_remove(name);
    }
    cmd.env_remove("OAIY_ZIPP_ASSET_DIR");
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
}

/// Ask a resolved CLI what it is. One process, `capabilities --json`, no flow,
/// no server URL or token (it needs neither), same hardening as a run.
///
/// The report is read from stdout REGARDLESS of the exit code: the CLI exits 1
/// for an unavailable engine but still prints the report, and its
/// `engine.reason` is the one line the user needs. Only an unreadable stdout
/// falls back to the exit code and the tail of stderr.
pub fn probe_cli_capabilities(cli: &CliInvocation, node_exe: Option<&Path>) -> EngineProbe {
    let mut cmd = cli_command(cli, node_exe);
    cmd.arg("capabilities").arg("--json");
    harden_child(&mut cmd);
    // A run is told where this desktop's API is and given its credential; the
    // probe runs no flow and talks to nobody, so it gets neither — and must not
    // inherit the headless server's real bearer token from the environment.
    cmd.env_remove("OAIY_SERVER_URL");
    cmd.env_remove("OAIY_SERVER_TOKEN");
    let started = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return EngineProbe::Unavailable {
                reason: format!("the OAIY CLI could not be started to ask what it runs: {e}"),
            }
        }
    };
    crate::services::job_object::adopt(child.id());
    let (out, err) = drain_child(&mut child);
    let deadline = started + PROBE_DEADLINE;
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
        thread::sleep(CHILD_POLL);
    };
    let stdout = out.lock().map(|g| g.clone()).unwrap_or_default();
    let stderr = err.lock().map(|g| g.clone()).unwrap_or_default();
    let Some(status) = status else {
        return EngineProbe::Unavailable {
            reason: format!(
                "the OAIY CLI did not answer `capabilities --json` within {}s",
                PROBE_DEADLINE.as_secs()
            ),
        };
    };
    log::debug!(
        "probed {} for its engine in {:?} (exit {})",
        cli.path().display(),
        started.elapsed(),
        status.code().unwrap_or(-1)
    );
    match serde_json::from_str::<Value>(&stdout) {
        Ok(report) => match parse_capabilities(&report) {
            Ok(identity) => EngineProbe::Ready(identity),
            Err(reason) => EngineProbe::Unavailable { reason },
        },
        Err(_) => EngineProbe::Unavailable {
            reason: format!(
                "the OAIY CLI answered `capabilities --json` with something other than a report (exit {}){}",
                status.code().unwrap_or(-1),
                match tail_of(&stderr, 600) {
                    t if t.is_empty() => String::new(),
                    t => format!(": {t}"),
                }
            ),
        },
    }
}

/// Start draining both pipes of a child, from the moment it starts.
///
/// The first cut of the flow runner read stderr only after exit and stdout
/// never — so a CLI that logged more than one OS pipe buffer (~4-64 KB; routine
/// for a verbose Node process) blocked in write(), could never exit, and was
/// killed at the deadline as a false `timed_out`.
fn drain_child(child: &mut std::process::Child) -> (Arc<Mutex<String>>, Arc<Mutex<String>>) {
    let captured_out = Arc::new(Mutex::new(String::new()));
    if let Some(out) = child.stdout.take() {
        let sink = captured_out.clone();
        thread::spawn(move || drain_capped(out, &sink));
    }
    let captured_err = Arc::new(Mutex::new(String::new()));
    if let Some(err) = child.stderr.take() {
        let sink = captured_err.clone();
        thread::spawn(move || drain_capped(err, &sink));
    }
    (captured_out, captured_err)
}

/// What a cached probe answer is FOR: the same file, unchanged, run the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeKey {
    cli: CliInvocation,
    modified: Option<SystemTime>,
    node_exe: Option<PathBuf>,
}

impl ProbeKey {
    fn of(cli: &CliInvocation, node_exe: Option<&Path>) -> Self {
        ProbeKey {
            cli: cli.clone(),
            modified: std::fs::metadata(cli.path()).and_then(|m| m.modified()).ok(),
            node_exe: node_exe.map(Path::to_path_buf),
        }
    }
}

#[derive(Debug, Clone)]
struct ProbeEntry {
    key: ProbeKey,
    at: Instant,
    probe: EngineProbe,
}

static PROBE: Mutex<Option<ProbeEntry>> = Mutex::new(None);
/// Set while a background warm-up is probing, so the readiness endpoint starts
/// at most one.
static PROBING: AtomicBool = AtomicBool::new(false);

/// Clears an [`AtomicBool`] however the scope ends, a panic included.
///
/// [`warm_engine_probe`] used to clear [`PROBING`] on the line after the probe
/// returned. Nothing else ever clears it, so a panic on that thread — and one
/// was reachable, see [`short_digest`] — left it set for the life of the
/// process: the readiness endpoint would never start another warm-up and would
/// answer `unknown` until the desktop was restarted.
struct ClearOnDrop<'a>(&'a AtomicBool);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Whether a remembered answer still stands for `key` at `now`.
///
/// `Ready` is reused for as long as the key matches — the CLI is a file that
/// does not change while the process runs, short of a reinstall, which
/// changes its mtime and so the key. A refusal is reused for [`PROBE_TTL`]
/// only, so an install fixed underneath a running desktop recovers on its own.
///
/// A `Ready` entry can also be REPLACED by a refusal without the key changing:
/// [`remember_run_refusal`] does that when a run reports the engine gone. The
/// key is the CLI file, and the engine is not the CLI file.
fn reusable(entry: Option<&ProbeEntry>, key: &ProbeKey, now: Instant) -> Option<EngineProbe> {
    let entry = entry?;
    if entry.key != *key {
        return None;
    }
    match &entry.probe {
        EngineProbe::Ready(_) => Some(entry.probe.clone()),
        EngineProbe::Unavailable { .. } if now.duration_since(entry.at) < PROBE_TTL => {
            Some(entry.probe.clone())
        }
        EngineProbe::Unavailable { .. } => None,
    }
}

/// Replace whatever is remembered with what a RUN just found.
///
/// Pure, so the rule is pinned without the process-global cache or a child
/// process. `now` becomes the entry's age, so [`PROBE_TTL`] governs recovery
/// from here exactly as it does for a probe's own refusal.
fn remember_run_refusal(
    cache: &mut Option<ProbeEntry>,
    key: ProbeKey,
    reason: String,
    now: Instant,
) {
    *cache = Some(ProbeEntry {
        key,
        at: now,
        probe: EngineProbe::Unavailable { reason },
    });
}

/// Record, against the process-global cache, that a run of `cli` found the
/// engine unavailable.
///
/// A run is the strongest evidence there is about an engine: it just tried to
/// use it. Until this existed, a `Ready` probe stood for the life of the
/// process — the cache key is the CLI FILE's mtime, and the engine is not the
/// CLI file. Delete, quarantine or half-replace the staged wasm under an
/// `oaiy.mjs` whose mtime never moved and every run failed
/// `runtime_unavailable` while `/api/bridge/status` kept answering
/// `ready: true` and Overview stayed silent — the exact lie the readiness
/// comment in `routes.rs` says it prevents. Same for an `OAIY_CLI` override
/// whose engine folder breaks after the first probe.
///
/// Only a run's own answer comes here. A refusal the PROBE made is already
/// remembered, and re-stamping it would keep pushing its TTL out and stop a
/// fixed install from recovering.
fn note_run_found_engine_unavailable(
    cli: &CliInvocation,
    node_exe: Option<&Path>,
    reason: String,
) {
    let key = ProbeKey::of(cli, node_exe);
    if let Ok(mut cache) = PROBE.lock() {
        remember_run_refusal(&mut cache, key, reason, Instant::now());
    }
}

/// The probe for `cli`, remembered or taken now. The lock is never held across
/// the child process: a slow probe must not block the readiness endpoint's
/// cache read.
pub fn engine_probe(cli: &CliInvocation, node_exe: Option<&Path>) -> EngineProbe {
    let key = ProbeKey::of(cli, node_exe);
    if let Ok(cache) = PROBE.lock() {
        if let Some(p) = reusable(cache.as_ref(), &key, Instant::now()) {
            return p;
        }
    }
    let probe = probe_cli_capabilities(cli, node_exe);
    match &probe {
        EngineProbe::Ready(id) => log::info!(
            "the OAIY CLI at {} runs user logic on ZIPP {} ({})",
            cli.path().display(),
            id.release,
            short_digest(&id.wasm_sha256)
        ),
        EngineProbe::Unavailable { reason } => log::warn!(
            "the OAIY CLI at {} does not run user logic on ZIPP: {reason}",
            cli.path().display()
        ),
    }
    if let Ok(mut cache) = PROBE.lock() {
        *cache = Some(ProbeEntry {
            key,
            at: Instant::now(),
            probe: probe.clone(),
        });
    }
    probe
}

/// The remembered probe for `cli`, if there is one that still stands. Never
/// spawns: this is what the readiness endpoint reads, on every UI poll, inside
/// an async handler.
pub fn cached_engine_probe(cli: &CliInvocation, node_exe: Option<&Path>) -> Option<EngineProbe> {
    let key = ProbeKey::of(cli, node_exe);
    PROBE
        .lock()
        .ok()
        .and_then(|cache| reusable(cache.as_ref(), &key, Instant::now()))
}

/// Fill the cache on a thread of its own, at most one at a time. For the
/// readiness endpoint's cold start: it reports `unknown` now and `ready` (or
/// the reason) on its next poll.
pub fn warm_engine_probe(cli: CliInvocation, node_exe: Option<PathBuf>) {
    if PROBING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    thread::spawn(move || {
        // Cleared by the guard, not by the next line: a probe that panics must
        // not take readiness down with it for the life of the process.
        let _clear = ClearOnDrop(&PROBING);
        let _ = engine_probe(&cli, node_exe.as_deref());
    });
}

/// What to do about a CLI that is not a ZIPP runner — the two situations need
/// different fixes, and the message has to name the right one.
pub(crate) fn engine_fix_hint() -> &'static str {
    let overridden = std::env::var("OAIY_CLI")
        .ok()
        .is_some_and(|v| !v.trim().is_empty());
    if overridden {
        "OAIY_CLI is set: remove the override so the CLI shipped with OAIY Desktop is used, \
         or point it at a build whose `capabilities --json` reports engine \"zipp\" as ready."
    } else {
        "Reinstall OAIY Desktop: the CLI it ships was staged without its ZIPP engine, or the \
         engine files beside it have been altered."
    }
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
        // Ask the CLI what it runs before the first claim, off the UI's path:
        // the readiness endpoint reads the cached answer, and a fresh start
        // should say "ready" (or why not) before anyone queues a run.
        if let Some(cli) = cli_status() {
            let node_exe = self.node.as_ref().and_then(|n| n.resolve());
            let _ = engine_probe(&cli, node_exe.as_deref());
        }
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
                // The engine's own default: a flow stored here was written
                // against no provider's policy, and the desktop sets none.
                instruction_budget: None,
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
        bridge_outcome(outcome, timeout)
    }
}

/// How the bridge lane reports what the CLI did. Pure, so every arm is pinned
/// without a ledger or a process.
///
/// A payload IS the flow's own report, and its `success: false` is a failed
/// run. This lane used to report any parseable payload as `succeeded` — a flow
/// that said "I failed, here is why" was recorded done, with the reason left
/// in an output nobody reads. The link's flow runner had fixed this for its
/// lane; the bridge never had.
fn bridge_outcome(
    outcome: CliOutcome,
    timeout: Duration,
) -> (RunStatus, Option<Value>, Option<RunError>) {
    match outcome {
        CliOutcome::Succeeded(v) if v.get("success") == Some(&Value::Bool(false)) => (
            RunStatus::Failed,
            None,
            Some(RunError::new(RunErrorCode::NodeFailed, flow_failure_message(&v))),
        ),
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

/// The flow's own account of its failure, from a `success: false` payload.
fn flow_failure_message(v: &Value) -> String {
    v.get("error")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| match v.get("status").and_then(Value::as_str) {
            Some(s) => format!("the flow reported {s:?} without saying why"),
            None => "the flow reported failure without saying why".to_string(),
        })
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
    /// ZIPP instruction steps per script entry (`--instruction-budget`), the
    /// provider's policy from its descriptor. `None` leaves the engine's own
    /// default; the CLI refuses a value outside the engine's range, which is
    /// why the descriptor checks it first.
    pub instruction_budget: Option<u64>,
    pub node: Option<&'a crate::services::node_runtime::NodeHandle>,
}

/// The CLI's argv for one run, so what is passed is pinned by a test.
pub fn cli_args(req: &CliRequest) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "run".into(),
        req.flow_path.into(),
        "--inputs".into(),
        req.inputs_path.into(),
        "-o".into(),
        req.out_path.into(),
        "--timeout".into(),
        req.timeout.as_secs().max(1).to_string().into(),
    ];
    if let Some(steps) = req.instruction_budget {
        args.push("--instruction-budget".into());
        args.push(steps.to_string().into());
    }
    if let Some(connector) = req.connector_path {
        args.push("--connector".into());
        args.push(connector.into());
    }
    args
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
    let node_exe = req.node.and_then(|n| n.resolve());
    run_flow_cli_with(&cli, node_exe.as_deref(), req, cancelled)
}

/// [`run_flow_cli`] with the CLI and Node already chosen.
///
/// Split out so a test can hand in a stub CLI: `OAIY_CLI` is process-global,
/// and under `cargo test` `current_exe()` is `target/debug/deps/…`, from where
/// the bundled lookup never reaches `src-tauri/resources`.
///
/// The engine probe happens here, once per CLI file (cached), before anything
/// is spawned — so all three lanes that arrive here (the bridge, the link's
/// flow runner, app logic) refuse the same non-ZIPP runner the same way.
pub fn run_flow_cli_with(
    cli: &CliInvocation,
    node_exe: Option<&Path>,
    req: CliRequest,
    cancelled: &dyn Fn() -> bool,
) -> CliOutcome {
    if let EngineProbe::Unavailable { reason } = engine_probe(cli, node_exe) {
        return CliOutcome::Unavailable {
            message: format!(
                "the OAIY CLI at {} does not run user logic on ZIPP",
                cli.path().display()
            ),
            detail: Some(format!("{reason}. {}", engine_fix_hint())),
            // Not retryable: re-driving the same run would meet the same CLI.
            // The fix is a reinstall or dropping the override, after which the
            // NEXT claim asks the CLI again (at most once a minute) — nothing
            // re-queues this one.
            retryable: false,
        };
    }

    let mut cmd = cli_command(cli, node_exe);
    cmd.args(cli_args(&req));
    harden_child(&mut cmd);
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
    // And who it is when it gets there. Privileged routes — the AI gateway
    // among them — fail closed on a missing Origin, and a spawned CLI has none,
    // so without this every chat node in a relayed flow was refused with
    // "origin not allowed" and failed the whole run. The child is this
    // process's own, so it carries this process's credential.
    let internal = crate::internal_token();
    if internal.is_empty() {
        cmd.env_remove("OAIY_SERVER_TOKEN");
    } else {
        cmd.env("OAIY_SERVER_TOKEN", internal);
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

    // Drain BOTH pipes on their own threads, from the start — see
    // `drain_child` for the deadlock this prevents. The review traced the whole
    // chain, and this crate documents the identical hazard for plugin stderr.
    let (_captured_out, captured_err) = drain_child(&mut child);

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

    let Some(st) = status else {
        return CliOutcome::TimedOut;
    };

    // The RESULT FILE is the flow's own report, and it outranks the exit code.
    //
    // The runtime can die on the way out — a libuv teardown assertion on
    // Windows (`UV_HANDLE_CLOSING`, exit 0xC0000409) fires AFTER the flow has
    // finished and written its result. Judging by exit code alone threw that
    // work away and reported a failure for a run that had completed, with a
    // "detail" that was whatever the crash happened to leave in the pipe.
    //
    // Reading the file first is also the honest order: what the flow says it
    // did is better evidence than how its host process happened to exit.
    let reported = std::fs::read_to_string(req.out_path)
        .ok()
        .and_then(|body| serde_json::from_str::<Value>(&body).ok());
    if let Some(v) = reported {
        if !st.success() && !matches!(v.get("success"), Some(Value::Bool(false))) {
            log::warn!(
                "the OAIY CLI exited {} after writing a result; honouring the result",
                st.code().unwrap_or(-1)
            );
        }
        let outcome = classify_result(v);
        // What the run found outranks what the probe remembered. Every
        // `Unavailable` arm of `classify_result` means the CLI is not the
        // runner the probe said it was — a missing engine, or a payload from
        // some other one — so readiness stops saying `ready` for it instead of
        // saying so until the process restarts.
        //
        // The MESSAGE alone is the reason: for `engine_unavailable` it is the
        // CLI's own account of what is wrong with its artifact, and `detail` is
        // `engine_fix_hint()`, which every reader of this reason already
        // appends for itself.
        if let CliOutcome::Unavailable { message, .. } = &outcome {
            note_run_found_engine_unavailable(cli, node_exe, message.clone());
        }
        return outcome;
    }

    if st.success() {
        // Exit 0 and nothing to show for it is ours, not the flow's.
        return CliOutcome::Unreadable(
            "the CLI reported success but wrote no readable result file".to_string(),
        );
    }
    CliOutcome::Failed {
        exit_code: st.code().unwrap_or(-1),
        detail: tail_of(&capture, 1500),
    }
}

/// Read the CLI's result payload as what it says it is. Pure, so every arm is
/// pinned without a process.
///
/// The CLI writes its payload even when the flow never ran: with
/// `errorCode: "engine_unavailable"` when its ZIPP artifact is missing or
/// altered, and `errorCode: "timeout"` when its own `--timeout` fired (which it
/// does before this desktop's deadline, by design, so the CLI can name the
/// stuck node). Read as a plain result, both were `Succeeded` — the bridge lane
/// reported a missing engine as a run that worked. So:
///
///   * a payload that does not say `engine: "zipp"` is not a result from the
///     runner this desktop ships; an older or foreign CLI wrote it;
///   * `engine_unavailable` is the runtime failing, not the flow — reported as
///     such, with the CLI's own reason, and never retried from here;
///   * `timeout` is the same outcome as this desktop killing the child, so all
///     three lanes report `timeout` rather than `node_failed`;
///   * anything else is the flow's own report, `success` and all, for the lane
///     to read.
pub fn classify_result(v: Value) -> CliOutcome {
    match v.get("engine").and_then(Value::as_str) {
        Some("zipp") => {}
        Some(other) => {
            return CliOutcome::Unavailable {
                message: format!(
                    "the OAIY CLI answered with an engine that is not ZIPP ({other:?})"
                ),
                detail: Some(engine_fix_hint().to_string()),
                retryable: false,
            }
        }
        None => {
            return CliOutcome::Unavailable {
                message: "the OAIY CLI answered without naming the engine the flow ran on, \
                          so it is not the runner this desktop ships"
                    .to_string(),
                detail: Some(engine_fix_hint().to_string()),
                retryable: false,
            }
        }
    }
    match v.get("errorCode").and_then(Value::as_str) {
        Some("engine_unavailable") => CliOutcome::Unavailable {
            message: v
                .get("error")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .unwrap_or("the OAIY CLI could not start its ZIPP engine")
                .to_string(),
            // Never retried from here: the same CLI would write the same
            // payload. The user's fix is named in `detail`.
            detail: Some(engine_fix_hint().to_string()),
            retryable: false,
        },
        Some("timeout") => CliOutcome::TimedOut,
        _ => CliOutcome::Succeeded(v),
    }
}

/// Read a pipe to EOF, keeping the LAST [`MAX_CAPTURE`] bytes.
///
/// The read must continue past the cap — stopping would refill the pipe and
/// recreate the deadlock the cap exists to report on.
///
/// The last bytes, not the first, and that is the whole point. This used to
/// stop appending once full, so what survived was the START of the output. The
/// engine announces every module it registers before a flow runs, which fills
/// the cap on its own — so every failure was reported as a wall of
/// registration notices and the actual error, printed last, was thrown away.
/// Runs failed with no readable reason for it.
pub(crate) fn drain_capped<R: std::io::Read>(mut reader: R, sink: &std::sync::Mutex<String>) {
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(mut s) = sink.lock() {
                    s.push_str(&String::from_utf8_lossy(&buf[..n]));
                    // Drop from the front, on a char boundary, so the buffer
                    // holds a rolling window ending at the newest output.
                    if s.len() > MAX_CAPTURE {
                        let mut cut = s.len() - MAX_CAPTURE;
                        while cut < s.len() && !s.is_char_boundary(cut) {
                            cut += 1;
                        }
                        s.drain(..cut);
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
pub(crate) fn tail_of(s: &str, n: usize) -> String {
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
    use serde_json::json;

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
    fn a_flood_of_output_keeps_the_error_at_the_end_not_the_preamble() {
        // The bug this pins, seen live: the engine announces every module it
        // registers before a flow runs, which filled the capture on its own.
        // What survived was that preamble, so every failed run was reported as
        // a wall of registration notices with the real error — printed last —
        // discarded. Runs failed with no readable reason.
        let sink = std::sync::Mutex::new(String::new());
        let noise = "[OAIYRuntime] Registered module 'Browser' with methods: …\n".repeat(4000);
        let output = format!("{noise}Error: the thing that actually went wrong\n");
        assert!(output.len() > MAX_CAPTURE * 2, "the flood must exceed the cap");
        drain_capped(std::io::Cursor::new(output.into_bytes()), &sink);

        let kept = sink.lock().unwrap().clone();
        assert!(kept.len() <= MAX_CAPTURE, "the cap still holds: {}", kept.len());
        assert!(
            kept.contains("the thing that actually went wrong"),
            "the END must survive, not the preamble"
        );
        assert!(tail_of(&kept, 1500).contains("the thing that actually went wrong"));
    }

    #[test]
    fn a_rolling_capture_never_splits_a_character() {
        // The window slides by bytes; a multi-byte char straddling the cut
        // would corrupt the buffer and could panic on the next boundary op.
        let sink = std::sync::Mutex::new(String::new());
        let body = "é".repeat(MAX_CAPTURE); // 2 bytes each: guarantees straddling
        drain_capped(std::io::Cursor::new(body.into_bytes()), &sink);
        let kept = sink.lock().unwrap().clone();
        assert!(kept.len() <= MAX_CAPTURE);
        assert!(kept.chars().all(|c| c == 'é'), "no replacement chars");
    }

    #[test]
    fn tail_keeps_the_end_of_a_stack_trace() {
        let long = format!("{}THE ACTUAL ERROR", "preamble ".repeat(500));
        let t = tail_of(&long, 100);
        assert!(t.contains("THE ACTUAL ERROR"), "{t}");
        assert!(t.starts_with('…'));
        assert!(tail_of("short", 100) == "short");
    }

    // --- the runner has to be ZIPP: the capabilities report -----------------

    fn good_capabilities() -> Value {
        // Shaped like `oaiy capabilities --json`; the values are a fixture's,
        // not any release's.
        json!({
            "version": "0.2.0",
            "protocols": { "run": 1, "script": 1, "profile": 1 },
            "engine": {
                "name": "zipp", "status": "ready",
                "release": "vF.I.X", "version": "F.I.X", "revision": "0123abcd",
                "wasmSha256": "ab".repeat(32), "languages": ["javascript", "python"]
            },
            "run": { "languages": ["javascript"], "defaultInstructionSteps": 50, "maxInstructionSteps": 2000 }
        })
    }

    #[test]
    fn the_protocols_map_is_kept_whole_so_other_consumers_can_require_their_own() {
        // `run` is what this file needs; the script host needs `script`. Both
        // come from the one report, integers only, unknown keys kept.
        let id = parse_capabilities(&good_capabilities()).unwrap();
        assert_eq!(id.protocols.get("run"), Some(&1));
        assert_eq!(id.protocols.get("script"), Some(&1));
        assert_eq!(id.protocols.get("profile"), Some(&1));
        let mut v = good_capabilities();
        v["protocols"] = json!({ "run": 1, "script": "one", "later": 3 });
        let id = parse_capabilities(&v).unwrap();
        assert_eq!(id.protocols.get("script"), None, "a non-integer version is no version");
        assert_eq!(id.protocols.get("later"), Some(&3), "an unknown protocol is carried, not refused");
        assert_eq!(id.protocols.len(), 2);
    }

    #[test]
    fn a_good_capabilities_report_is_a_ready_engine_with_what_this_host_runs() {
        let id = parse_capabilities(&good_capabilities()).unwrap();
        assert_eq!(id.name, "zipp");
        assert_eq!(id.release, "vF.I.X");
        assert_eq!(id.revision, "0123abcd");
        assert_eq!(id.wasm_sha256, "ab".repeat(32));
        // `run.languages`, not `engine.languages`: the bundle can run Python,
        // this CLI does not, and nothing may advertise what no host runs.
        assert_eq!(id.run_languages, vec!["javascript".to_string()]);
    }

    #[test]
    fn a_digest_that_is_not_a_sha256_is_not_a_capabilities_report() {
        // Item 2: `wasm_sha256` was copied out of the report with no check at
        // all, and then byte-sliced for a log line. `.min(12)` bounds the
        // LENGTH, not a UTF-8 character boundary, so a multi-byte digest
        // panicked whichever thread took the probe. It is refused here instead,
        // by the same rule `desktop/scripts/sync-cli-lib.mjs` already applies
        // on the staging side.
        for (label, sha) in [
            ("a multi-byte digest — the one that panicked", "\u{e9}".repeat(32)),
            ("upper case", "AB".repeat(32)),
            ("too short", "ab".repeat(31)),
            ("too long", "ab".repeat(33)),
            ("not hex", "zz".repeat(32)),
            ("empty", String::new()),
        ] {
            let mut v = good_capabilities();
            v["engine"]["wasmSha256"] = json!(sha);
            let e = match parse_capabilities(&v) {
                Err(e) => e,
                Ok(id) => panic!("{label}: {sha:?} must not pass as a digest, got {id:?}"),
            };
            assert!(e.contains("not a sha256"), "{label}: {e}");
        }
        let mut v = good_capabilities();
        v["engine"]["wasmSha256"] = Value::Null;
        assert!(parse_capabilities(&v).is_err(), "no digest is not a digest");
    }

    #[test]
    fn a_log_line_never_panics_on_a_digest_whatever_is_in_it() {
        // Belt as well as braces: `parse_capabilities` refuses these now, and
        // the head taken for the log line still must not split a character.
        assert_eq!(short_digest(&"ab".repeat(32)), "abababababab");
        assert_eq!(short_digest("abc"), "abc");
        assert_eq!(short_digest(""), "");
        // Byte 12 lands INSIDE the sixth two-byte character here (1 + 2×5 = 11),
        // which is what `&s[..s.len().min(12)]` panics on and `get` does not.
        // A digest of `"é".repeat(32)` ALONE would not catch it — 12 is a
        // boundary there, so the old slice would look sound while still being a
        // byte range.
        let multibyte = format!("a{}", "é".repeat(32));
        assert!(!multibyte.is_char_boundary(12), "the fixture must straddle a character");
        // `get` answers None for a range that is not a boundary, so the whole
        // string is logged: long, and never a panic, which is the point.
        assert_eq!(short_digest(&multibyte), multibyte);
    }

    #[test]
    fn a_panicking_probe_still_lets_the_next_warm_up_start() {
        // Item 8: `PROBING` was cleared on the line AFTER the probe returned,
        // so a panic inside it (item 2's was reachable) left the flag set for
        // the life of the process — nothing else clears it, and the readiness
        // endpoint would answer `unknown` for ever. Tested against a LOCAL
        // flag, never the process-global one: cargo runs these in parallel.
        //
        // The panic's own message reaching stderr is expected. The hook is not
        // swapped for a quiet one, because it is global and a test running
        // beside this one would lose its own.
        let flag = AtomicBool::new(true);
        let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _clear = ClearOnDrop(&flag);
            panic!("the probe panicked on a digest");
        }));
        assert!(died.is_err(), "the probe must actually have panicked");
        assert!(
            !flag.load(Ordering::Acquire),
            "a panic past the guard left PROBING set: readiness would stay unknown until restart"
        );
        // And the ordinary path still clears it.
        flag.store(true, Ordering::Release);
        {
            let _clear = ClearOnDrop(&flag);
        }
        assert!(!flag.load(Ordering::Acquire), "the guard clears it on an ordinary return too");
    }

    #[test]
    fn a_run_that_finds_the_engine_gone_replaces_the_ready_probe() {
        // Item 1: a `Ready` entry has no TTL and `classify_result` never
        // touched the cache, so ONE good probe made readiness say `ready:true`
        // for the life of the process. Delete the staged wasm while `oaiy.mjs`
        // keeps its mtime and the key still matches: every run failed
        // `runtime_unavailable` and `/api/bridge/status` said nothing.
        let key = ProbeKey {
            cli: CliInvocation::Node { script: PathBuf::from("oaiy.mjs") },
            modified: None,
            node_exe: None,
        };
        let now = Instant::now();
        let mut cache = Some(ProbeEntry {
            key: key.clone(),
            at: now,
            probe: EngineProbe::Ready(parse_capabilities(&good_capabilities()).unwrap()),
        });
        assert!(
            reusable(cache.as_ref(), &key, now + PROBE_TTL * 10).unwrap().is_ready(),
            "a Ready probe stands while the CLI file is unchanged — that part is right"
        );

        // A run comes back with the CLI's own `engine_unavailable` payload.
        let outcome = classify_result(json!({
            "success": false,
            "engine": "zipp",
            "errorCode": "engine_unavailable",
            "error": "zipp/zipp_wasm_bg.wasm is missing",
        }));
        let CliOutcome::Unavailable { message, .. } = &outcome else {
            panic!("engine_unavailable must be Unavailable, got {outcome:?}")
        };
        remember_run_refusal(&mut cache, key.clone(), message.clone(), now);

        match reusable(cache.as_ref(), &key, now) {
            Some(EngineProbe::Unavailable { reason }) => {
                assert!(reason.contains("zipp_wasm_bg.wasm is missing"), "{reason}")
            }
            other => panic!("readiness must stop saying ready, got {other:?}"),
        }
        // And the existing TTL governs recovery, so an install fixed underneath
        // a running desktop still comes back on its own.
        assert!(
            reusable(cache.as_ref(), &key, now + PROBE_TTL).is_none(),
            "after the TTL the CLI is asked again"
        );
    }

    #[test]
    fn an_engine_that_is_not_zipp_is_refused_by_name() {
        let mut v = good_capabilities();
        v["engine"]["name"] = json!("v8");
        let e = parse_capabilities(&v).unwrap_err();
        assert!(e.contains("\"v8\""), "{e}");
        assert!(e.contains("not on ZIPP"), "{e}");
        v["engine"]["name"] = Value::Null;
        assert!(parse_capabilities(&v).is_err(), "an unnamed engine is not ZIPP either");
    }

    #[test]
    fn an_unavailable_engine_is_refused_with_the_clis_own_reason() {
        let mut v = good_capabilities();
        v["engine"]["status"] = json!("unavailable");
        v["engine"]["reason"] = json!("ZIPP engine unavailable: zipp/zipp_wasm_bg.wasm has sha256 dead…, not beef…");
        let e = parse_capabilities(&v).unwrap_err();
        assert!(e.contains("sha256 dead"), "the CLI's reason is the error: {e}");
        // No reason given: still refused, still says what is known.
        v["engine"].as_object_mut().unwrap().remove("reason");
        let e = parse_capabilities(&v).unwrap_err();
        assert!(e.contains("\"unavailable\""), "{e}");
    }

    #[test]
    fn a_run_protocol_this_desktop_does_not_read_is_refused() {
        let mut v = good_capabilities();
        v["protocols"]["run"] = json!(2);
        let e = parse_capabilities(&v).unwrap_err();
        assert!(e.contains("protocol 2"), "{e}");
        v.as_object_mut().unwrap().remove("protocols");
        assert!(parse_capabilities(&v).is_err());
    }

    // --- the cache: reuse rules --------------------------------------------

    #[test]
    fn a_ready_probe_is_kept_while_the_cli_file_is_the_same_and_a_refusal_expires() {
        let key = ProbeKey {
            cli: CliInvocation::Node { script: PathBuf::from("C:/app/resources/cli/oaiy.mjs") },
            modified: Some(SystemTime::UNIX_EPOCH),
            node_exe: None,
        };
        let now = Instant::now();
        let ready = ProbeEntry {
            key: key.clone(),
            at: now - PROBE_TTL * 10,
            probe: EngineProbe::Ready(parse_capabilities(&good_capabilities()).unwrap()),
        };
        // Ready: reused however old, while the key matches…
        assert!(reusable(Some(&ready), &key, now).is_some_and(|p| p.is_ready()));
        // …but not for a CLI file that has since changed (mtime), nor a
        // different file, nor a different Node.
        let mut touched = key.clone();
        touched.modified = Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1));
        assert!(reusable(Some(&ready), &touched, now).is_none());
        let mut other = key.clone();
        other.cli = CliInvocation::Binary { path: PathBuf::from("C:/tools/oaiy.exe") };
        assert!(reusable(Some(&ready), &other, now).is_none());
        let mut node = key.clone();
        node.node_exe = Some(PathBuf::from("C:/node/node.exe"));
        assert!(reusable(Some(&ready), &node, now).is_none());

        // A refusal is remembered for the TTL, then asked again — so a fixed
        // install recovers without a restart.
        let refused = ProbeEntry {
            key: key.clone(),
            at: now,
            probe: EngineProbe::Unavailable { reason: "no engine".into() },
        };
        assert!(reusable(Some(&refused), &key, now + PROBE_TTL / 2).is_some());
        assert!(reusable(Some(&refused), &key, now + PROBE_TTL).is_none());
        assert!(reusable(None, &key, now).is_none());
    }

    // --- the payload: what the CLI wrote, read as what it says -------------

    #[test]
    fn a_timed_out_payload_is_a_timeout_not_a_result() {
        // The CLI's own --timeout fires before this desktop's deadline (by
        // design, so it can name the stuck node) and it WRITES the payload.
        let out = classify_result(json!({
            "success": false, "status": "aborted", "errorCode": "timeout", "engine": "zipp",
            "error": "flow timed out after 2s"
        }));
        assert!(matches!(out, CliOutcome::TimedOut), "{out:?}");
    }

    #[test]
    fn an_engine_unavailable_payload_is_the_runtime_failing_not_a_result() {
        // The bug that made this PR urgent: the CLI writes this payload when
        // its ZIPP artifact is missing or altered, and the bridge lane reported
        // it `succeeded`.
        let out = classify_result(json!({
            "success": false, "status": "failed", "jobId": "", "results": {},
            "error": "ZIPP engine unavailable: zipp/zipp_wasm_bg.wasm is missing",
            "errorCode": "engine_unavailable", "engine": "zipp"
        }));
        match out {
            CliOutcome::Unavailable { message, detail, retryable } => {
                assert!(message.contains("ZIPP engine unavailable"), "the CLI's reason: {message}");
                assert!(detail.is_some_and(|d| d.contains("Reinstall") || d.contains("OAIY_CLI")));
                assert!(!retryable, "a missing engine is not fixed by re-driving the run");
            }
            other => panic!("engine_unavailable must be Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn a_payload_from_an_engine_that_is_not_zipp_is_refused() {
        let out = classify_result(json!({ "success": true, "output": 1, "engine": "v8" }));
        match out {
            CliOutcome::Unavailable { message, .. } => assert!(message.contains("\"v8\""), "{message}"),
            other => panic!("a v8 payload must be Unavailable, got {other:?}"),
        }
        // A payload that names no engine at all is from a CLI older than this
        // convention — and so not the runner this desktop ships.
        let out = classify_result(json!({ "success": true, "output": 1 }));
        assert!(matches!(out, CliOutcome::Unavailable { .. }), "{out:?}");
    }

    #[test]
    fn a_zipp_payload_is_the_flows_own_report_success_or_not() {
        let ok = classify_result(json!({ "success": true, "output": {"a": 1}, "engine": "zipp" }));
        match ok {
            CliOutcome::Succeeded(v) => assert_eq!(v["output"]["a"], 1),
            other => panic!("{other:?}"),
        }
        // A flow's OWN failure is still its report, for the lane to read.
        let failed = classify_result(json!({ "success": false, "status": "failed", "error": "boom", "engine": "zipp" }));
        assert!(matches!(failed, CliOutcome::Succeeded(_)), "{failed:?}");
    }

    // --- the bridge lane: what a payload becomes in the ledger -------------

    #[test]
    fn the_bridge_lane_reports_a_flows_own_failure_as_failed_not_succeeded() {
        // Pre-existing bug, fixed here: any parseable payload was `succeeded`.
        let (status, output, error) = bridge_outcome(
            CliOutcome::Succeeded(json!({ "success": false, "status": "failed", "error": "node x threw", "engine": "zipp" })),
            Duration::from_secs(5),
        );
        assert_eq!(status, RunStatus::Failed);
        assert!(output.is_none());
        let e = error.expect("a failed run carries its error");
        assert_eq!(e.code, RunErrorCode::NodeFailed);
        assert!(e.message.contains("node x threw"), "{}", e.message);

        let (status, output, error) = bridge_outcome(
            CliOutcome::Succeeded(json!({ "success": true, "output": 7, "engine": "zipp" })),
            Duration::from_secs(5),
        );
        assert_eq!(status, RunStatus::Succeeded);
        assert_eq!(output.unwrap()["output"], 7);
        assert!(error.is_none());
    }

    #[test]
    fn the_bridge_lane_reports_an_unavailable_runtime_and_a_timeout_by_code() {
        let (status, _, error) = bridge_outcome(
            CliOutcome::Unavailable { message: "no ZIPP".into(), detail: Some("fix".into()), retryable: false },
            Duration::from_secs(5),
        );
        assert_eq!(status, RunStatus::Failed);
        assert_eq!(error.unwrap().code, RunErrorCode::RuntimeUnavailable);
        let (status, _, error) = bridge_outcome(CliOutcome::TimedOut, Duration::from_secs(5));
        assert_eq!(status, RunStatus::TimedOut);
        assert_eq!(error.unwrap().code, RunErrorCode::Timeout);
    }

    // --- argv: the budget per lane ------------------------------------------

    #[test]
    fn the_instruction_budget_is_passed_when_a_provider_sets_one_and_omitted_otherwise() {
        let flow = PathBuf::from("g.json");
        let inputs = PathBuf::from("i.json");
        let out = PathBuf::from("r.json");
        let connector = PathBuf::from("c.json");
        fn req<'a>(
            paths: &'a (PathBuf, PathBuf, PathBuf),
            budget: Option<u64>,
            connector: Option<&'a Path>,
        ) -> CliRequest<'a> {
            CliRequest {
                flow_path: &paths.0,
                inputs_path: &paths.1,
                out_path: &paths.2,
                connector_path: connector,
                timeout: Duration::from_secs(300),
                instruction_budget: budget,
                node: None,
            }
        }
        let paths = (flow, inputs, out);
        let strs = |args: Vec<OsString>| -> Vec<String> {
            args.into_iter().map(|a| a.to_string_lossy().into_owned()).collect()
        };
        // The FormLogic lanes: the provider's 200M from its descriptor.
        let a = strs(cli_args(&req(&paths, Some(200_000_000), Some(&connector))));
        assert_eq!(
            a,
            ["run", "g.json", "--inputs", "i.json", "-o", "r.json", "--timeout", "300",
             "--instruction-budget", "200000000", "--connector", "c.json"]
        );
        // The bridge: no policy, so the engine's own default — no flag at all.
        let b = strs(cli_args(&req(&paths, None, None)));
        assert!(!b.iter().any(|s| s.contains("instruction-budget")), "{b:?}");
        assert_eq!(b, ["run", "g.json", "--inputs", "i.json", "-o", "r.json", "--timeout", "300"]);
    }

    // --- a stub CLI through the real spawn path ------------------------------

    /// A Node on PATH, or the same skip/fail policy as the app-logic e2e:
    /// skipped on a box without one, a failure under CI (which stages both).
    fn node_on_path() -> Option<PathBuf> {
        let finder = if cfg!(windows) { "where" } else { "which" };
        let found = Command::new(finder)
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
            panic!("CI must have node on PATH for the stub-CLI tests");
        }
        found
    }

    /// Write a CLI stand-in: `capabilities --json` prints `caps`; `run … -o F`
    /// writes `payload` to F and exits 0.
    fn stub_cli(tag: &str, caps: &Value, payload: &Value) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("oaiy-stub-cli-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("stub.mjs");
        std::fs::write(
            &script,
            format!(
                "import fs from 'node:fs';\n\
                 const a = process.argv.slice(2);\n\
                 if (a[0] === 'capabilities') {{ process.stdout.write({caps}); process.exit(0); }}\n\
                 if (a[0] === 'run') {{ fs.writeFileSync(a[a.indexOf('-o') + 1], {payload}); process.exit(0); }}\n\
                 process.exit(2);\n",
                caps = serde_json::to_string(&caps.to_string()).unwrap(),
                payload = serde_json::to_string(&payload.to_string()).unwrap(),
            ),
        )
        .unwrap();
        (dir, script)
    }

    fn run_stub(dir: &Path, script: PathBuf, node: &Path) -> CliOutcome {
        let flow = dir.join("flow.json");
        let inputs = dir.join("inputs.json");
        let out = dir.join("result.json");
        std::fs::write(&flow, "{}").unwrap();
        std::fs::write(&inputs, "{}").unwrap();
        run_flow_cli_with(
            &CliInvocation::Node { script },
            Some(node),
            CliRequest {
                flow_path: &flow,
                inputs_path: &inputs,
                out_path: &out,
                connector_path: None,
                timeout: Duration::from_secs(20),
                instruction_budget: None,
                node: None,
            },
            &|| false,
        )
    }

    #[test]
    fn a_cli_whose_engine_is_not_zipp_is_refused_before_any_flow_is_run() {
        // An `OAIY_CLI` override pointing at some other build: the probe
        // refuses it, the flow is never handed over, and the failure names the
        // fix. The stub's `run` would happily "succeed" — the assertion on the
        // result file not existing is what proves it never ran.
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let mut caps = good_capabilities();
        caps["engine"]["name"] = json!("v8");
        let (dir, script) = stub_cli("v8", &caps, &json!({ "success": true, "output": 1, "engine": "v8" }));
        let outcome = run_stub(&dir, script.clone(), &node);
        match outcome {
            CliOutcome::Unavailable { message, detail, retryable } => {
                assert!(message.contains("does not run user logic on ZIPP"), "{message}");
                assert!(message.contains(&script.to_string_lossy().to_string()), "names the CLI: {message}");
                let d = detail.expect("the reason and the fix");
                assert!(d.contains("\"v8\""), "the probe's reason: {d}");
                assert!(d.contains("OAIY_CLI") || d.contains("Reinstall"), "the fix: {d}");
                assert!(!retryable, "a bad OAIY_CLI override is not retryable");
            }
            other => panic!("a v8 CLI must be refused, got {other:?}"),
        }
        assert!(!dir.join("result.json").exists(), "the flow must never have been handed to the stub");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_that_reports_engine_unavailable_stops_readiness_saying_ready() {
        // Item 1, through the real process boundary: the stub's
        // `capabilities --json` is a READY ZIPP engine, so the probe caches
        // `Ready` — and then its `run` writes the payload a CLI writes when its
        // staged wasm has gone. Before this, that cached `Ready` stood for the
        // life of the process and `/api/bridge/status` kept answering
        // `ready: true` over an install where every run failed.
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_cli(
            "gone",
            &good_capabilities(),
            &json!({
                "success": false,
                "status": "failed",
                "engine": "zipp",
                "errorCode": "engine_unavailable",
                "error": "zipp/zipp_wasm_bg.wasm is missing",
            }),
        );
        let cli = CliInvocation::Node { script: script.clone() };

        // `PROBE` is one process-global slot and cargo runs tests in parallel,
        // so another test's probe can evict this entry between the run and the
        // read. That shows up as `None` — a key mismatch — never as a stale
        // `Ready`, so retrying settles it while the assertion that matters
        // holds on every attempt.
        let mut seen = None;
        for _ in 0..3 {
            let outcome = run_stub(&dir, script.clone(), &node);
            match &outcome {
                CliOutcome::Unavailable { message, retryable, .. } => {
                    assert!(message.contains("zipp_wasm_bg.wasm is missing"), "{message}");
                    assert!(!*retryable, "the same CLI would write the same payload");
                }
                other => panic!("an engine_unavailable payload must be Unavailable, got {other:?}"),
            }
            match cached_engine_probe(&cli, Some(&node)) {
                Some(EngineProbe::Ready(id)) => panic!(
                    "readiness still says ready after a run found the engine gone: {id:?}"
                ),
                Some(p @ EngineProbe::Unavailable { .. }) => {
                    seen = Some(p);
                    break;
                }
                // Evicted by a parallel test's probe; ask again.
                None => continue,
            }
        }
        match seen {
            Some(EngineProbe::Unavailable { reason }) => {
                assert!(reason.contains("zipp_wasm_bg.wasm is missing"), "{reason}")
            }
            other => panic!("the run's own reason must be what readiness reports, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cli_that_claims_zipp_but_answers_with_another_engine_is_refused_at_the_payload() {
        // The second gate, through the real process boundary: the probe
        // passes, the payload does not say `engine: "zipp"`, and the run is
        // not a result.
        let Some(node) = node_on_path() else {
            eprintln!("no node on PATH — skipping");
            return;
        };
        let (dir, script) = stub_cli(
            "liar",
            &good_capabilities(),
            &json!({ "success": true, "status": "completed", "output": 1, "engine": "v8" }),
        );
        let outcome = run_stub(&dir, script, &node);
        match outcome {
            CliOutcome::Unavailable { message, .. } => {
                assert!(message.contains("not ZIPP"), "{message}");
                assert!(message.contains("\"v8\""), "{message}");
            }
            other => panic!("a v8 payload must be refused, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
