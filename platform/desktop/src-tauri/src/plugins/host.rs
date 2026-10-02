//! The plugin host: live processes, supervision, and the event pipeline.
//!
//! This is where the pieces built so far become one system:
//!
//! ```text
//!   registry (state + gate)          triggers (bindings)      ledger (runs)
//!        ▲                                  ▲                     ▲
//!        │ state changes                    │ dispatch            │ reserve
//!        │                                  │                     │
//!   PluginHost ──── start/stop ────► PluginProcess (per plugin)
//!        │                                  │
//!        │  health/crash supervisor         │ event.emit (validated)
//!        └──── one thread, all plugins ◄────┴──► event thread → ring + triggers → ack
//! ```
//!
//! # Lock discipline
//!
//! The registry lock is never held across a plugin RPC. A connector call can
//! legitimately take seconds (dialling a phone), and holding the registry lock
//! that long would freeze `/api/plugins`, health transitions and every other
//! gate check behind one slow command. So: gate under the lock, copy what the
//! call needs, drop the lock, then do the RPC.
//!
//! # Events go through a channel, not straight to dispatch
//!
//! The `EventSink` closure runs on the plugin's **reader thread**. Dispatching
//! from there would take the ledger and bindings locks on a thread that must
//! stay responsive (it also routes RPC replies), and a slow dispatch would
//! starve every in-flight request of its answers. So the sink only sends into a
//! channel; a dedicated event thread does the heavy work — ring buffer, trigger
//! dispatch, and the `event.ack` that lets the plugin stop re-delivering.
//!
//! The rule covers the FAILURE path too, which is where it was quietly lost
//! once: recording a shed event durably is a whole-file rewrite plus an fsync,
//! and doing it in the sink put a disk sync per dropped event in front of every
//! RPC reply the reader still had to route — under a flood, i.e. exactly when
//! the plugin can least afford it. So sheds get their own channel and their own
//! thread as well. Nothing on the reader thread may wait on a lock or the disk.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::process::{CallError, PluginProcess, SpawnOptions};
use super::registry::{GateRefusal, PluginRegistryHandle, PluginState};
use super::runner::{restart_delay, should_restart, HealthTracker, HealthVerdict, HEALTH_INTERVAL};
use super::trust::LaunchRefusal;
use crate::bridge::deadletters::{DeadLetterHandle, DeadReason};
use crate::bridge::ledger::{LedgerHandle, LineageRef, ReserveOutcome, RunRequest, RunStatus};
use crate::bridge::triggers::{dispatch, DispatchOutcome, Event, SkipReason, TriggerBinding};
use crate::link::outbox::account_key;
use crate::services::runner::LogBuffer;

/// Does this skip mean something went WRONG, or the trigger system working?
///
/// The distinction is the whole value of the dead-letter queue. Disabled,
/// manual, and condition-false are the design doing its job (see
/// [`crate::bridge::triggers`]: everything fails towards not running), and
/// recording them would bury the real failures under routine not-firing.
/// Duplicate likewise means the run already exists — the opposite of lost work.
///
/// What remains are the cases where an author or operator meant work to happen
/// and it did not: a condition that will not parse, a guard refusal, and a
/// fan-out past the per-event ceiling.
fn is_operational_failure(reason: &SkipReason) -> bool {
    match reason {
        SkipReason::ConditionUnevaluatable { .. }
        | SkipReason::Guard(_)
        | SkipReason::TooManyBindings => true,
        SkipReason::Disabled
        | SkipReason::ManualMode
        | SkipReason::ConditionFalse
        | SkipReason::Duplicate => false,
    }
}

/// What one dispatch did. `reserved` is a fact the dispatcher already knows;
/// the alternative was for callers to re-derive it by grepping `outcomes` for a
/// substring of a human-readable message, which silently breaks the moment
/// anyone rewords it.
struct Dispatched {
    /// Per-binding, for the event ring. Human-readable — not for branching on.
    outcomes: Vec<String>,
    /// Set when the event should have produced work and did not.
    dead: Option<DeadReason>,
    /// Did any binding actually reserve a run?
    reserved: bool,
}

/// Should this dispatch be dead-lettered?
///
/// Only when NOTHING was reserved and at least one binding failed for an
/// operational reason. A binding that fired makes the event handled — the
/// others declining is ordinary fan-out, not lost work.
fn dead_reason_for(results: &[DispatchOutcome]) -> Option<DeadReason> {
    if results
        .iter()
        .any(|o| matches!(o, DispatchOutcome::Reserved { .. }))
    {
        return None;
    }
    let failures: Vec<String> = results
        .iter()
        .filter_map(|o| match o {
            DispatchOutcome::Skipped { binding_id, reason } if is_operational_failure(reason) => {
                Some(format!("{binding_id}: {}", reason.message()))
            }
            _ => None,
        })
        .collect();
    (!failures.is_empty()).then(|| DeadReason::NotReserved { detail: failures.join("; ") })
}

/// Default deadline for a forwarded connector command.
pub const CONNECTOR_TIMEOUT: Duration = Duration::from_secs(30);
/// Events kept for `GET /api/bridge/events` polling.
const EVENT_RING_CAPACITY: usize = 500;
/// How long a plugin must stay up before its crash-restart budget is refilled.
///
/// The rest of the supervision policy lives in [`super::runner`]; this one is
/// here because the host is the only thing that observes uptime. Several health
/// intervals long, so "it came up" and "it stayed up" cannot be the same event —
/// that conflation is what made [`super::runner::MAX_RESTART_ATTEMPTS`]
/// unreachable.
const STABLE_UPTIME: Duration = Duration::from_secs(60);

/// One received event, as the polling endpoint returns it.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceivedEvent {
    /// Monotonic sequence number — a poller passes the last one it saw back as
    /// `since`. Sequence, not timestamp: two events in the same millisecond must
    /// not be skippable.
    pub seq: u64,
    pub received_at_ms: u64,
    pub envelope: Value,
    /// What the trigger dispatcher did with it, binding by binding. Empty when
    /// no binding matched — which is itself the answer to "why didn't my flow
    /// run".
    pub outcomes: Vec<String>,
}

/// A free-text log field, JSON-quoted, so a condition containing a quote or a
/// newline cannot break one log line into two.
fn json_str(s: &str) -> String {
    Value::String(s.to_string()).to_string()
}

/// The line one `$event` rewrite leaves behind.
///
/// A function rather than a `log::warn!` in place, so the format is pinned by a
/// test: this line is the only record of an edit to a file the USER owns, and it
/// has to carry enough to reconstruct exactly what changed — which file, which
/// binding, the text before and the text after.
fn migration_line(
    path: &std::path::Path,
    r: &crate::bridge::triggers::ConditionRewrite,
) -> String {
    format!(
        "trigger-condition-migrated: file={} binding={} before={} after={}",
        json_str(&path.display().to_string()),
        json_str(&r.binding_id),
        json_str(&r.before),
        json_str(&r.after),
    )
}

/// Persistent trigger bindings, JSON on disk.
///
/// A file rather than a database because the write rate is human (someone edits
/// a binding), the read rate is per-event, and the whole set is cached in
/// memory. Written atomically (`.tmp` + rename) so a crash mid-write cannot
/// leave half a JSON file that silently disables every trigger on next boot.
pub struct TriggerStore {
    path: PathBuf,
    bindings: Vec<TriggerBinding>,
    /// The file's ORIGINAL bytes, kept only while a `$event` migration has been
    /// applied in memory and its `.bak` could not be written. See [`migrate`].
    ///
    /// [`migrate`]: TriggerStore::migrate
    pending_backup: Option<String>,
}

impl TriggerStore {
    pub fn load(path: PathBuf) -> Self {
        let mut original = None;
        let bindings = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let parsed = Self::parse(&path, &text);
                original = Some(text);
                parsed
            }
            // No file at all IS the first boot. Anything else is a file that
            // holds the user's bindings and could not be read — a lock from a
            // backup agent or AV, bad UTF-8 — and must not be mistaken for
            // "this workspace has no triggers", because the next `upsert` would
            // make that true on disk.
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("[triggers] cannot read {}: {e}", path.display());
                    Self::quarantine(&path);
                }
                Vec::new()
            }
        };
        let mut store = Self { path, bindings, pending_backup: None };
        if let Some(original) = original {
            store.migrate(original);
        }
        store
    }

    /// Rewrite bare `event` / `$event` condition operands, once.
    ///
    /// The Rust grammar read `$event` alone as `event.data`; ZIPP reads it as
    /// the whole envelope, which is truthy even when `data` is not. Two readings
    /// of one token IS the bug, so there is no compatibility window: the stored
    /// text is rewritten a single time, the original is kept beside the file as
    /// `.bak`, every rewrite is logged with what it was and what it became, and
    /// the old reading survives nowhere.
    ///
    /// Nothing to change means nothing happens — no `.bak`, no write — so a file
    /// already in the new form is left byte-identical, whatever its formatting.
    ///
    /// The original is never lost. The `.bak` is written and renamed into place
    /// BEFORE the rewrite, and the rewrite itself is `persist`'s `.tmp`+rename,
    /// so the file on disk is always wholly the old one or wholly the new one.
    /// If the `.bak` cannot be written, the rewrite is not attempted at all: the
    /// migrated bindings are used IN MEMORY — they are the safe reading, and the
    /// migration is idempotent, so the next boot simply tries again — and the
    /// original text is held for `persist` to back up before it writes anything.
    fn migrate(&mut self, original: String) {
        let rewrites = crate::bridge::triggers::migrate_bindings(&mut self.bindings);
        if rewrites.is_empty() {
            return;
        }
        for r in &rewrites {
            log::warn!("{}", migration_line(&self.path, r));
        }
        match Self::write_backup(&self.path, &original) {
            Ok(bak) => {
                log::warn!(
                    "{} trigger condition(s) were rewritten; the original is at {}",
                    rewrites.len(),
                    bak.display()
                );
                if let Err(e) = self.persist() {
                    log::error!(
                        "the rewritten trigger conditions could not be saved ({e}); they are in force for this run and will be rewritten again on the next boot"
                    );
                }
            }
            Err(e) => {
                log::error!(
                    "the original trigger bindings could not be copied aside ({e}), so {} rewritten condition(s) were NOT saved; they are in force for this run only",
                    rewrites.len()
                );
                self.pending_backup = Some(original);
            }
        }
    }

    /// Copy `original` to `<path>.bak`, atomically.
    fn write_backup(path: &std::path::Path, original: &str) -> Result<PathBuf, String> {
        let bak = path.with_extension("json.bak");
        let tmp = path.with_extension("json.bak.tmp");
        std::fs::write(&tmp, original).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &bak).map_err(|e| format!("{}: {e}", bak.display()))?;
        Ok(bak)
    }

    /// Deserialize entry by entry, keeping everything that loads.
    ///
    /// Whole-file `from_str::<Vec<TriggerBinding>>` made ONE malformed binding
    /// — a hand edit, a field a newer build wrote — deserialize to nothing:
    /// every automation stopped firing, `GET /api/bridge/triggers` returned the
    /// same empty list a workspace with no triggers returns, and the first
    /// `upsert` afterwards rewrote the file from that empty Vec, destroying the
    /// other bindings for good. The ledger and the dead-letter queue both skip
    /// only the bad entry for exactly this reason.
    fn parse(path: &std::path::Path, text: &str) -> Vec<TriggerBinding> {
        let rows: Vec<Value> = match serde_json::from_str(text) {
            Ok(rows) => rows,
            Err(e) => {
                // Not a list at all, so there is nothing to salvage per entry —
                // salvage the file instead.
                eprintln!("[triggers] {} is not a list of bindings ({e})", path.display());
                Self::quarantine(path);
                return Vec::new();
            }
        };
        let total = rows.len();
        let loaded: Vec<TriggerBinding> = rows
            .into_iter()
            .filter_map(|row| match serde_json::from_value::<TriggerBinding>(row.clone()) {
                Ok(b) => Some(b),
                Err(e) => {
                    // Loud, because the symptom otherwise is a flow that stopped
                    // running and a UI that says nothing is wrong.
                    log::warn!("skipping a trigger binding that will not load ({e}): {row}");
                    None
                }
            })
            .collect();

        // A well-formed list where NOTHING survived is not "the user has no
        // bindings" — it is a shape this build cannot read. The realistic cause
        // is our own doing: add a required field to `TriggerBinding` and every
        // existing row fails at once. Returning empty is then indistinguishable
        // from a fresh install, and the first `upsert` writes that emptiness
        // over the only copy. Quarantine so the file survives to be recovered.
        if total > 0 && loaded.is_empty() {
            log::warn!(
                "none of the {total} bindings in {} could be loaded; keeping the file aside",
                path.display()
            );
            Self::quarantine(path);
        }
        loaded
    }

    /// Move a bindings file we could not load out of the way.
    ///
    /// Renamed rather than left in place, because `persist` rewrites the whole
    /// file from memory: leaving it means the first `upsert` after a failed load
    /// silently overwrites the only copy the user has. A `.corrupt` file is
    /// something they can hand back to us; an overwritten one is not.
    fn quarantine(path: &std::path::Path) {
        let aside = path.with_extension("json.corrupt");
        match std::fs::rename(path, &aside) {
            Ok(()) => eprintln!("[triggers] kept the original at {}", aside.display()),
            Err(e) => eprintln!("[triggers] could not preserve {}: {e}", path.display()),
        }
    }

    pub fn list(&self) -> &[TriggerBinding] {
        &self.bindings
    }

    pub fn upsert(&mut self, binding: TriggerBinding) -> Result<(), String> {
        if binding.id.trim().is_empty() {
            return Err("binding id must not be empty".into());
        }
        // A binding is only ever wrong in one direction: it does not fire, and
        // nothing says why. Each of these produces exactly that — a row that
        // looks correct in the list and is incapable of ever doing anything —
        // so they are refused where the mistake is made rather than discovered
        // later from a dead letter.
        if binding.event.trim().is_empty() {
            return Err("binding event must not be empty — it would match nothing".into());
        }
        if binding.flow_id.trim().is_empty() {
            return Err("binding flowId must not be empty — there would be nothing to run".into());
        }
        // The CONDITION is not checked here any more. It is JavaScript, and the
        // only thing that can honestly say whether it parses is the engine that
        // will run it — which is a process, and this runs under the trigger
        // store's lock. `routes::upsert_trigger` asks ZIPP before it takes that
        // lock; a condition that reaches this point unchecked still cannot fire,
        // because dispatch fails closed on anything it cannot decide.
        match self.bindings.iter_mut().find(|b| b.id == binding.id) {
            Some(existing) => *existing = binding,
            None => self.bindings.push(binding),
        }
        self.persist()
    }

    pub fn remove(&mut self, id: &str) -> Result<bool, String> {
        let before = self.bindings.len();
        self.bindings.retain(|b| b.id != id);
        let removed = self.bindings.len() != before;
        if removed {
            self.persist()?;
        }
        Ok(removed)
    }

    fn persist(&self) -> Result<(), String> {
        // A migration ran but its `.bak` did not. Writing now would overwrite the
        // only copy of what the user actually wrote, so retry the backup and
        // refuse rather than lose it. Loud and recoverable beats silent.
        if let Some(original) = self.pending_backup.as_deref() {
            Self::write_backup(&self.path, original).map_err(|e| {
                format!("the original trigger bindings still cannot be copied aside ({e}), so this write would destroy them")
            })?;
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&self.bindings).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, body).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &self.path).map_err(|e| e.to_string())
    }
}

pub type TriggerStoreHandle = Arc<Mutex<TriggerStore>>;

struct EventRing {
    seq: AtomicU64,
    ring: Mutex<VecDeque<ReceivedEvent>>,
}

/// Everything mutable about plugin processes, under ONE lock.
///
/// One table rather than separate maps because the review confirmed four
/// distinct races between them: two concurrent start()s both passing an empty
/// running-map check (check-then-act across a seconds-long handshake), a stop()
/// arriving mid-handshake and reporting success while the plugin came up
/// anyway, a pending restart resurrecting a manually-stopped plugin, and the
/// supervisor's stale snapshot misreading a graceful stop as a crash. Every one
/// of them is an atomicity problem between "who is running", "who is starting"
/// and "who is scheduled to restart" — so those three live behind one mutex and
/// every transition is a single critical section.
#[derive(Default)]
struct ProcTable {
    running: HashMap<String, Arc<PluginProcess>>,
    /// Ids with a start in flight (spawn + handshake take seconds, outside the
    /// lock). A second start() sees the id here and returns idempotently
    /// instead of spawning a rival process onto the same hardware.
    starting: std::collections::HashSet<String>,
    /// Ids whose in-flight start should be abandoned: stop() arrived during the
    /// handshake. The finishing start() kills the process instead of
    /// registering it.
    stop_during_start: std::collections::HashSet<String>,
    /// Children that have been spawned but not yet handshaken.
    ///
    /// `stop_during_start` alone is only a REQUEST — the start thread reads it
    /// when its handshake returns, up to HANDSHAKE_TIMEOUT after the child was
    /// spawned. On app exit there is no such time: `stop_all` returned, the
    /// process exited, and the child survived as an orphan still holding the
    /// dongle it opened during init. So the handle is published here from the
    /// instant the child exists, and `stop_all` can actually stop it.
    starting_procs: HashMap<String, Arc<PluginProcess>>,
    /// Crash-restart due times. In the table — NOT supervisor-local — so a
    /// manual stop() can cancel one before it fires.
    restarts: HashMap<String, Instant>,
    /// When each running plugin was registered, for [`STABLE_UPTIME`]. Entries
    /// are consumed by the refill, so a plugin pays for the registry lock once
    /// per life rather than once per supervisor tick.
    started_at: HashMap<String, Instant>,
    /// Log rings, kept past the process that produced them.
    ///
    /// Written on spawn and NOT removed when the plugin leaves `running`: the
    /// stderr a plugin writes on its way out ("no dongle at COM3", a traceback)
    /// is the only evidence of why it crashed, and the Logs button is offered in
    /// every state. Reading these from `running` meant the panel said "No output
    /// yet." within one supervisor tick of the exit — precisely when there was
    /// something to read. services/registry keeps its runner and installer after
    /// exit for the same reason and the same LogBuffer type. Replaced on the
    /// next spawn, so a restarted plugin does not show its previous life.
    log_rings: HashMap<String, LogBuffer>,
    /// The roster each broker plugin was handed at its last `plugin.init`, which is the roster it holds (it learns
    /// it only then, and refuses an admission that differs from it). Replaced, or taken away when the plugin was
    /// started with none, by each start; see `handle_companion_admission`.
    rosters: HashMap<String, crate::companion::RosterSnapshot>,
    /// Set by `stop_all`: the app is exiting. Nothing may spawn a child after
    /// it — the autostart loop and a due crash-restart would otherwise start a
    /// plugin with nobody left to stop it.
    shutting_down: bool,
}

pub struct PluginHost {
    pub registry: PluginRegistryHandle,
    pub ledger: LedgerHandle,
    pub triggers: TriggerStoreHandle,
    /// Events that arrived and produced no work. See [`crate::bridge::deadletters`].
    pub dead: DeadLetterHandle,
    procs: Mutex<ProcTable>,
    events: EventRing,
    /// Bounded: a plugin can emit events faster than the single event thread
    /// dispatches them (each dispatch takes the ledger lock). An unbounded
    /// channel let a flooding plugin grow this queue without limit — the review's
    /// memory-exhaustion path. `try_send` drops on a full queue and logs it,
    /// which is the right failure: better to shed events under a flood, with a
    /// record, than to run the machine out of memory. At-least-once delivery
    /// means a dropped event is re-sent by a well-behaved plugin anyway.
    event_tx: SyncSender<(String, Value)>,
    /// Events the queue above refused, on their way to a durable record.
    ///
    /// A second channel rather than doing the work in the sink: recording a shed
    /// rewrites and fsyncs the whole dead-letter queue, and the sink runs on the
    /// plugin's reader thread — the thread this module promises never to block,
    /// during the flood that caused the drop. And not the event thread either,
    /// which is by definition swamped whenever anything is being shed.
    shed_tx: SyncSender<(String, Value)>,
    desktop_version: String,
    dev_mode: bool,
    /// Set after construction by whichever binary wired an HTTP surface.
    ///
    /// Optional because the host is also built in tests and by tools with no
    /// companion support at all; those must keep working, and answering
    /// `companion.admission` with an honest "not configured" beats making every
    /// caller supply a broker it will never use.
    companion: Mutex<Option<CompanionBroker>>,
    /// The linked account, for fanning events out to its flows. Set after
    /// construction like the companion broker, and for the same reason: the
    /// host exists before the link does, and in tests there is no link at all.
    link: Mutex<Option<crate::link::LinkHandle>>,
    /// Events on their way to the linked account, kept on disk until they get
    /// there (see [`crate::link::outbox`]). Made with the link.
    outbox: std::sync::OnceLock<Arc<crate::link::outbox::Outbox>>,
    /// The account's trigger bindings, cached between events.
    flow_bindings: crate::link::flows::FlowBindings,
    /// The account's apps and their logic scripts, cached between events.
    app_logic: crate::link::app_logic::Catalog,
    /// What runs this event's conditions and app-logic scripts.
    ///
    /// The process-wide warm script host, unless a test has substituted its own
    /// — the engine is the one thing on the event thread that reaches a child
    /// process, and a unit test must be able to answer without one.
    scripts: Mutex<std::sync::Arc<dyn crate::bridge::script_host::ScriptBatch>>,
    /// The ring a phone plugin's `oaiy.ring.*` requests and call events reach: this one, else the desktop's own
    /// ([`crate::ring::shared`]). A test gives its own, so it does not share a process-wide one with the others.
    ring: Mutex<Option<Arc<crate::ring::Ring>>>,
}

/// A screen's capability belongs to the exact process the host started from a
/// verified package. Holding the Arc prevents an old process's identity being
/// reused after stop/restart while a completion is still in flight.
pub(crate) struct ScreenCapabilityLease {
    process: Arc<PluginProcess>,
}

/// What the host needs to answer `companion.admission`: this desktop's own
/// device trust, and the upstream that turns it into a gateway admission.
#[derive(Clone)]
pub struct CompanionBroker {
    pub companion: crate::companion::routes::CompanionHandle,
    pub upstream: crate::companion::upstream::UpstreamHandle,
}

impl PluginHost {
    pub(crate) fn screen_capability(&self, id: &str, capability: &str) -> Result<ScreenCapabilityLease, (&'static str, &'static str)> {
        {
            let reg = self.registry.lock().map_err(|_| ("capability_unavailable", "The plugin registry is unavailable."))?;
            let rec = reg.get(id).ok_or(("capability_unavailable", "The plugin is not installed."))?;
            if rec.user_disabled || !rec.state.accepts_commands() || !rec.trust.as_ref().is_some_and(|trust|
                matches!(trust.state, crate::plugins::trust::TrustState::Verified | crate::plugins::trust::TrustState::TrustedLocal)) {
                return Err(("capability_unavailable", "Trust and start the plugin in Plugins before using AI completion."));
            }
            // Adding a spending capability must not retroactively grant it to
            // an older broad host wildcard. This screen contract requires its
            // literal name in the package the owner reviewed.
            if !reg.grants(id, capability) || !rec.manifest.as_ref().is_some_and(|manifest|
                manifest.capabilities.iter().any(|declared| declared == capability)) {
                return Err(("capability_denied", "The plugin does not declare the required host capability."));
            }
        }
        let process = self.procs.lock().map_err(|_| ("capability_unavailable", "The plugin process is unavailable."))?
            .running.get(id).cloned().ok_or(("capability_unavailable", "The plugin process is not running."))?;
        if process.check_exited().is_some() {
            return Err(("capability_unavailable", "The plugin process has exited."));
        }
        Ok(ScreenCapabilityLease { process })
    }

    pub(crate) fn holds_screen_capability(&self, id: &str, capability: &str, lease: &ScreenCapabilityLease) -> bool {
        self.screen_capability(id, capability).is_ok_and(|current| Arc::ptr_eq(&current.process, &lease.process))
    }

    /// Build the host, start its background threads (events, outbox, shed,
    /// supervisor) and start the plugins that start at boot.
    pub fn new(
        registry: PluginRegistryHandle,
        ledger: LedgerHandle,
        triggers: TriggerStoreHandle,
        dead: DeadLetterHandle,
        desktop_version: String,
        dev_mode: bool,
    ) -> Arc<Self> {
        let host = Self::assemble(registry, ledger, triggers, dead, desktop_version, dev_mode);
        host.spawn_autostart();
        host
    }

    /// The host and its background threads, without the boot autostart.
    ///
    /// Apart from `new` only for the tests of what starting a plugin does, which put a
    /// plugin on disk after the host exists: a boot autostart that scanned a moment late
    /// would start it (or not) on its own, and race the start the test is about.
    pub(crate) fn assemble(
        registry: PluginRegistryHandle,
        ledger: LedgerHandle,
        triggers: TriggerStoreHandle,
        dead: DeadLetterHandle,
        desktop_version: String,
        dev_mode: bool,
    ) -> Arc<Self> {
        let (event_tx, event_rx) = sync_channel::<(String, Value)>(1024);
        // Smaller than the event queue: this only ever holds what that queue
        // already refused, and each slot is a whole envelope.
        let (shed_tx, shed_rx) = sync_channel::<(String, Value)>(256);
        let host = Arc::new(Self {
            registry,
            ledger,
            triggers,
            dead,
            procs: Mutex::new(ProcTable::default()),
            events: EventRing {
                seq: AtomicU64::new(0),
                ring: Mutex::new(VecDeque::with_capacity(EVENT_RING_CAPACITY)),
            },
            event_tx,
            shed_tx,
            desktop_version,
            dev_mode,
            companion: Mutex::new(None),
            link: Mutex::new(None),
            outbox: std::sync::OnceLock::new(),
            flow_bindings: crate::link::flows::FlowBindings::new(),
            app_logic: crate::link::app_logic::Catalog::new(),
            scripts: Mutex::new(std::sync::Arc::new(crate::bridge::script_host::GlobalHost)),
            ring: Mutex::new(None),
        });

        // Event thread: ring + trigger dispatch + ack.
        {
            let host = Arc::downgrade(&host);
            thread::spawn(move || {
                while let Ok((plugin_id, envelope)) = event_rx.recv() {
                    let Some(host) = host.upgrade() else { break };
                    host.process_event(&plugin_id, envelope);
                }
            });
        }

        // Outbox thread: sends the events kept for the linked account, oldest
        // first, and waits while FormLogic cannot be reached. Off the event
        // thread, so nothing local waits on FormLogic.
        {
            let host = Arc::downgrade(&host);
            thread::spawn(move || loop {
                let Some(h) = host.upgrade() else { break };
                let Some(outbox) = h.outbox.get().cloned() else {
                    drop(h);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                };
                let link = h.link.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let account = link.as_ref().and_then(|l| l.account()).map(|a| account_key(&a));
                let heard = if outbox.retrying() {
                    // The outbox spaces its own tries; the caches' own waits
                    // after a failure would only add to them.
                    h.flow_bindings.retry_now();
                    h.app_logic.retry_now();
                    link.as_ref().and_then(|l| l.status().last_heartbeat_at)
                } else {
                    None
                };
                outbox.send_due(
                    account.as_deref(),
                    heard,
                    &|plugin, envelope| h.deliver_to_account(plugin, envelope),
                    &|plugin, envelope, why| {
                        let name = envelope.get("name").and_then(Value::as_str).unwrap_or("(unnamed)").to_string();
                        h.record_dead(plugin, &name, DeadReason::NotDelivered { detail: why }, envelope.clone());
                    },
                );
                drop(h);
                outbox.wait_for_work();
            });
        }

        // Shed thread: the durable record of a dropped event, and the log-ring
        // notice beside it. Both take a lock and one of them fsyncs; neither may
        // happen on the reader thread that dropped the event.
        {
            let host = Arc::downgrade(&host);
            thread::spawn(move || {
                while let Ok((plugin_id, envelope)) = shed_rx.recv() {
                    let Some(host) = host.upgrade() else { break };
                    host.logs_drop_notice(&plugin_id);
                    host.record_shed(&plugin_id, envelope);
                }
            });
        }

        // Supervisor thread: health probes, crash detection, bounded restarts.
        {
            let host = Arc::downgrade(&host);
            thread::spawn(move || {
                let mut trackers: HashMap<String, HealthTracker> = HashMap::new();
                loop {
                    thread::sleep(HEALTH_INTERVAL);
                    let Some(host) = host.upgrade() else { break };
                    host.supervise(&mut trackers);
                }
            });
        }

        host
    }

    /// Start every plugin that should be running at boot.
    ///
    /// On its own thread, because starting is slow — a plugin that loads speech
    /// models takes seconds, and boot must not wait for it. Without this a
    /// plugin was only ever running if someone opened the app and clicked Start,
    /// which for something like a phone bridge means it quietly answers nothing
    /// until a human remembers it exists. `autostart_ids` already knew which
    /// ones qualify (loadable, not user-disabled); nothing called it.
    fn spawn_autostart(self: &Arc<Self>) {
        let host = Arc::downgrade(self);
        thread::spawn(move || {
            let Some(host) = host.upgrade() else { return };
            let ids = match host.registry.lock() {
                Ok(mut reg) => {
                    // The registry is populated lazily; without a scan a fresh
                    // process has no plugins to autostart at all.
                    reg.scan();
                    reg.autostart_ids()
                }
                Err(_) => return,
            };
            for id in ids {
                // Quitting during boot is ordinary — this loop is slow by
                // design. Without the check it kept spawning children after
                // stop_all had already run and had nothing left to stop them
                // with. `start` refuses too; stopping here keeps a quit from
                // logging one failure per remaining plugin.
                if host.is_shutting_down() {
                    return;
                }
                // Sequential: two plugins loading model weights at once on a
                // laptop is worse than one after the other, and boot is not
                // waiting on this thread anyway.
                if let Err(e) = host.start(&id) {
                    // Not fatal. The supervisor and the UI both surface plugin
                    // state, and a plugin that will not start at boot will not
                    // start on a click either — the reason is what matters, and
                    // `start` has already recorded it on the record.
                    eprintln!("[plugins] autostart {id}: {e}");
                }
            }
        });
    }

    /// Start a plugin: spawn, handshake, mark `Running`.
    pub fn start(self: &Arc<Self>, id: &str) -> Result<(), String> {
        // Before anything that could spawn. Every caller of `start` outlives
        // `stop_all` — the autostart loop, a due crash-restart, a POST
        // /api/plugins/:id/start already sitting in a blocking task — and a
        // child spawned after shutdown has nobody left to stop it: it survives
        // the app as an orphan holding whatever hardware it opened.
        if self.is_shutting_down() {
            return Err(format!("{id} was not started: OAIY Desktop is shutting down"));
        }
        // Copy what spawn needs out of the registry, then release the lock —
        // spawning and handshaking take seconds.
        let (dir, trust) = {
            let mut reg = self.registry.lock().map_err(|_| "registry lock poisoned")?;
            // Scan first: "drop a folder in plugins/, then POST start" is the
            // documented install flow, and without this it failed with "no plugin
            // named X" until some OTHER endpoint happened to trigger a scan.
            // Same call-order bug as capability discovery, found the same way —
            // by driving the API in the documented order. `scan` preserves live
            // state, so rescanning here cannot disturb running plugins.
            reg.scan();
            let rec = reg
                .get(id)
                .ok_or_else(|| format!("no plugin named {id:?} is installed"))?;
            if rec.user_disabled {
                return Err(format!(
                    "{id} is turned off. Enable it in OAIY Desktop → Plugins first."
                ));
            }
            // The manifest it is started from is not this one: that comes from the bytes
            // the launch verifies (below). A plugin held back by its package is not turned
            // away here on the scan's word either: a scan may only have looked at the
            // folder's sizes and times, and the launch check, which reads the bytes, is the
            // one that decides (it gives the person the same verdict when it agrees).
            // Only a plugin with no manifest for any other reason stops here.
            if rec.manifest.is_none() && !rec.refused_by_trust() {
                return Err(format!(
                    "{id} cannot start: {}",
                    rec.reason.clone().unwrap_or_else(|| "its manifest is invalid".into())
                ));
            }
            (rec.dir.clone(), reg.trust())
        };

        // Claim the start ATOMICALLY. "Check the map, then spawn" is a
        // check-then-act race across a seconds-long handshake: two concurrent
        // starts both pass an empty-map check and spawn rival processes onto the
        // same hardware (the review traced it end to end). Inserting into
        // `starting` under the same lock that reads `running` makes the second
        // caller's outcome deterministic: idempotent success.
        {
            let mut t = self.procs.lock().map_err(|_| "process table poisoned")?;
            if let Some(existing) = t.running.get(id) {
                if existing.check_exited().is_none() {
                    return Ok(());
                }
                t.running.remove(id);
            }
            if !t.starting.insert(id.to_string()) {
                // A start is already in flight; joining it is what the caller
                // wanted anyway.
                return Ok(());
            }
            // A manual start supersedes any scheduled crash-restart.
            t.restarts.remove(id);
            t.stop_during_start.remove(id);
        }
        // From here on, every return path must clear the `starting` claim.
        let claim = StartClaim { host: self, id: id.to_string() };

        // Verify the package again, from its bytes, immediately before the launch.
        // The scan a moment ago (and the one at boot, and the one on install) only ever
        // shows a verdict: a file swapped since then must not slip through, and a folder
        // that scanned as fine may not be any longer. The permit this yields is what
        // `PluginProcess::spawn` requires, so this is the one way a plugin starts, and it
        // carries the manifest parsed from the very bytes that were hashed: what is
        // started is what was verified, not a later read of the file.
        let permit = match trust.authorize_launch(&dir, id) {
            Ok(permit) => permit,
            Err(LaunchRefusal::Untrusted(verdict)) => {
                let why = verdict.reason.clone().unwrap_or_else(|| "its package is not trusted".into());
                log::warn!("plugin {id} was not started: {why}");
                if let Ok(mut reg) = self.registry.lock() {
                    reg.withhold(id, *verdict);
                }
                return Err(format!("{id} was not started: {why}"));
            }
            Err(LaunchRefusal::Manifest(e)) => {
                // It scanned as loadable a moment ago and does not now: the next scan
                // lists it with the reason, as it does any plugin whose manifest is broken.
                let why = e.reason();
                log::warn!("plugin {id} was not started: {why}");
                return Err(format!("{id} cannot start: {why}"));
            }
        };
        // The record follows what is about to run: what its gate, its events and its
        // capabilities allow (and the companion seed below is handed on) is what this
        // process was started from.
        if let Ok(mut reg) = self.registry.lock() {
            reg.adopt_launch(id, &permit);
        }
        let plugin_api_version = permit.manifest().plugin_api_version;

        self.set_state(id, PluginState::Starting, Some("Launching…".into()));

        let host_for_events = Arc::downgrade(self);
        let plugin_for_events = id.to_string();
        let host_for_requests = Arc::downgrade(self);
        let plugin_for_requests = id.to_string();

        let spawn_result = PluginProcess::spawn(
            SpawnOptions {
                desktop_version: self.desktop_version.clone(),
                dev_mode: self.dev_mode,
                permit,
                events: Arc::new(move |_name, envelope| {
                    if let Some(host) = host_for_events.upgrade() {
                        // try_send, not send: this closure runs on the reader
                        // thread (see the module docs), and a bounded queue must
                        // never block it — a blocked reader stops routing RPC
                        // replies too. A full queue means the event thread is
                        // swamped; shed with a log rather than grow without bound.
                        if let Err(e) =
                            host.event_tx.try_send((plugin_for_events.clone(), envelope))
                        {
                            let (_, envelope) = match e {
                                std::sync::mpsc::TrySendError::Full(v) => v,
                                std::sync::mpsc::TrySendError::Disconnected(v) => v,
                            };
                            // The log ring is itself overwriting under the flood
                            // that caused this, so the durable record is what
                            // actually survives — but writing it is the shed
                            // thread's job, not this thread's. See `note_shed`.
                            host.note_shed(&plugin_for_events, envelope);
                        }
                    }
                }),
                requests: Arc::new(move |method, params| {
                    match host_for_requests.upgrade() {
                        Some(host) => host.handle_plugin_request(&plugin_for_requests, method, params),
                        None => Err((
                            "runtime_unavailable".into(),
                            "the host is shutting down".into(),
                        )),
                    }
                }),
            },
        );

        let process = match spawn_result {
            Ok(p) => Arc::new(p),
            Err(e) => {
                self.set_state(id, PluginState::Crashed, Some(e.clone()));
                return Err(e);
            }
        };
        // A child exists from here on, so this is the critical section that has
        // to settle its ownership — and it settles the shutdown race by being
        // the FIRST one after the spawn. Either `stop_all` got here first, in
        // which case we see the flag and kill the child ourselves, or we did, in
        // which case it is in `starting_procs` and `stop_all` will stop it. The
        // check at the top of `start` only saves the work; this is what shrinks
        // the orphan window from up to HANDSHAKE_TIMEOUT (10s of dongle
        // enumeration and model loading) to the thread-scheduling gap between
        // `spawn` returning and this lock being taken. Not zero — if the process
        // exits inside that gap the child is orphaned exactly as before — but
        // small enough that the realistic case, quitting during a slow start,
        // is covered.
        let abandon: Option<&str> = match self.procs.lock() {
            Ok(mut t) if !t.shutting_down => {
                // Published BEFORE the handshake, which can take
                // HANDSHAKE_TIMEOUT — far longer than app exit is willing to
                // wait, and `stop_during_start` is not read until afterwards.
                t.starting_procs.insert(id.to_string(), process.clone());
                // And the log ring, so the output of a start that never
                // completes — the ModuleNotFoundError, the missing DLL — is
                // still readable once `start` has returned. That is the
                // commonest plugin failure there is.
                t.log_rings.insert(id.to_string(), process.logs.clone());
                None
            }
            Ok(_) => Some("OAIY Desktop is shutting down"),
            // Nothing can hold a handle to this child, so the only alternative
            // to killing it is leaking it.
            Err(_) => Some("the process table is unusable"),
        };
        if let Some(why) = abandon {
            process.kill();
            let reason = format!("{id} was not started: {why}");
            self.set_state(id, PluginState::Stopped, Some(reason.clone()));
            return Err(reason);
        }

        // Handshake, with the process killed on failure — a plugin that cannot
        // answer plugin.init is not going to answer anything else, and leaving
        // it alive would hold its hardware while reporting Crashed.
        // A broker plugin signs as this desktop's companion endpoint, so it is
        // handed that identity at init. Every other plugin gets None, and so
        // does a broker with no approved device — see `private_bootstrap`.
        let companion_bootstrap = self.init_bootstrap(id, plugin_api_version);
        match process.init(
            plugin_api_version,
            &super::runner::plugin_data_dir(&dir),
            self.dev_mode,
            companion_bootstrap,
        ) {
            Ok(_) => {}
            Err(e) => {
                let reason = format!("did not complete the init handshake: {e}");
                process.kill();
                self.set_state(id, PluginState::Crashed, Some(reason.clone()));
                return Err(reason);
            }
        }

        {
            let mut t = self.procs.lock().map_err(|_| "process table poisoned")?;
            // A stop() that arrived during the handshake wins: registering the
            // process now would leave it Running seconds after the user was told
            // the stop succeeded. Kill it instead — it never served anything.
            if t.stop_during_start.remove(id) {
                drop(t);
                process.kill();
                drop(claim);
                self.set_state(id, PluginState::Stopped, Some("Stopped while starting.".into()));
                return Err(format!("{id} was stopped while it was starting"));
            }
            t.running.insert(id.to_string(), process);
            // The clock the restart budget is refilled from. NOT refilled here:
            // see `refill_restart_budget_if_stable`.
            t.started_at.insert(id.to_string(), Instant::now());
        }
        drop(claim);
        self.set_state(id, PluginState::Running, None);
        Ok(())
    }

    /// The private bootstrap `plugin.init` hands a broker plugin (its endpoint identity and its roster of phones),
    /// or `None` for a plugin that is not a broker and for a broker with nobody approved. The roster in it is the
    /// roster the plugin holds until it is started again: it accepts an admission only if it equals that one, so that
    /// is what `companion.admission` presents for it. It is kept here, in the same step as the bootstrap is made,
    /// and BEFORE the handshake, the last moment the plugin can ask; a plugin started with none holds none, and what
    /// an earlier run of it held is let go.
    fn init_bootstrap(&self, id: &str, plugin_api_version: u32) -> Option<Value> {
        let (bootstrap, roster) = match self
            .companion
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|_| self.registry.lock().map(|reg| reg.grants(id, "oaiy.companion.admission")).unwrap_or(false))
            .and_then(|b| b.companion.identity_for(id).ok())
            .and_then(|identity| identity.private_bootstrap_with_roster(plugin_api_version as u16))
        {
            Some((bootstrap, roster)) => (Some(bootstrap), Some(roster)),
            None => (None, None),
        };
        if let Ok(mut t) = self.procs.lock() {
            match roster {
                Some(roster) => t.rosters.insert(id.to_string(), roster),
                None => t.rosters.remove(id),
            };
        }
        bootstrap
    }
    /// Stop a plugin gracefully. Also cancels a scheduled crash-restart and
    /// overrides a start that is still mid-handshake — a stop that can be
    /// outraced by its own plugin coming up is not a stop.
    pub fn stop(&self, id: &str) -> Result<(), String> {
        let (process, start_in_flight) = {
            let mut t = self.procs.lock().map_err(|_| "process table poisoned")?;
            // Cancel any pending crash-restart: the user said stop, and a timer
            // resurrecting the plugin afterwards reads as the stop not working.
            t.restarts.remove(id);
            let in_flight = t.starting.contains(id);
            if in_flight {
                // The start will observe this when its handshake completes and
                // kill the process instead of registering it.
                t.stop_during_start.insert(id.to_string());
            }
            t.started_at.remove(id);
            (t.running.remove(id), in_flight)
        };
        match process {
            Some(p) => {
                p.shutdown();
                self.set_state(id, PluginState::Stopped, Some("Stopped by request.".into()));
                Ok(())
            }
            None if start_in_flight => {
                self.set_state(
                    id,
                    PluginState::Stopped,
                    Some("Stop requested while starting; the start will be abandoned.".into()),
                );
                Ok(())
            }
            None => {
                // Not running is what "stop" wants; align the recorded state
                // rather than erroring on an already-satisfied request.
                self.set_state(id, PluginState::Stopped, Some("Was not running.".into()));
                Ok(())
            }
        }
    }

    /// The ids of the plugins running now (an update notes them before it stops everything).
    pub fn running_ids(&self) -> Vec<String> {
        let table = self.procs.lock().unwrap_or_else(|e| e.into_inner());
        table.running.keys().cloned().collect()
    }

    /// Let plugins start again after [`Self::stop_all`]: for an update that stopped everything and then could
    /// not install, so the app carries on as it was. (Quitting never calls it: the process is going away.)
    pub fn resume(&self) {
        if let Ok(mut table) = self.procs.lock() {
            table.shutting_down = false;
        }
    }

    /// Stop everything. Called on app exit so no child outlives the host.
    pub fn stop_all(&self) {
        let (drained, in_flight): (Vec<(String, Arc<PluginProcess>)>, Vec<(String, Arc<PluginProcess>)>) =
            match self.procs.lock() {
                Ok(mut t) => {
                    // Nothing new may spawn, no restart may fire, and any
                    // in-flight start must be abandoned when it completes.
                    t.shutting_down = true;
                    t.restarts.clear();
                    let starting: Vec<String> = t.starting.iter().cloned().collect();
                    for id in starting {
                        t.stop_during_start.insert(id);
                    }
                    t.started_at.clear();
                    (t.running.drain().collect(), t.starting_procs.drain().collect())
                }
                Err(_) => return,
            };
        for (id, p) in drained {
            p.shutdown();
            self.set_state(&id, PluginState::Stopped, Some("OAIY Desktop is exiting.".into()));
        }
        // Children that were spawned but had not finished handshaking. Their
        // start thread will honour `stop_during_start` when the handshake
        // returns — but that is up to HANDSHAKE_TIMEOUT away, and this process
        // exits as soon as we return. Without stopping them here, quitting
        // during a slow start (the boot autostart window, or Start-then-quit)
        // left a child with no owner, still holding its dongle, until the user
        // found it or replugged the device.
        for (id, p) in in_flight {
            // `kill`, not `shutdown`. A graceful shutdown waits SHUTDOWN_GRACE
            // for the child to answer `plugin.shutdown` — but a child that is
            // still handshaking is by definition not reading stdin yet (it is
            // doing the slow work, enumerating a dongle or loading models, that
            // created this window at all), so the full grace always elapses. On
            // the exact path this covers — quit during boot autostart — that
            // added ~5s per plugin to a Quit the user is watching. There is also
            // nothing to be graceful ABOUT: the plugin has been handed no work,
            // so it has no state to flush.
            p.kill();
            self.set_state(&id, PluginState::Stopped, Some("OAIY Desktop is exiting.".into()));
        }
    }

    /// Recent log lines for a plugin, running or not.
    ///
    /// The retained ring, not the live process: see [`ProcTable::log_rings`].
    /// `None` means this plugin has never been spawned in this session, which is
    /// a different thing from having produced no output and is why the caller is
    /// given the distinction.
    pub fn logs(&self, id: &str, tail: Option<usize>) -> Option<Vec<crate::services::runner::LogLine>> {
        self.procs
            .lock()
            .ok()?
            .log_rings
            .get(id)
            .map(|logs| logs.snapshot(tail))
    }

    /// Forward a connector command through the capability gate to the plugin.
    pub fn forward_connector(
        &self,
        connector_id: &str,
        command: &str,
        payload: Option<Value>,
        idempotency_key: Option<&str>,
        timeout: Duration,
    ) -> Result<Value, ForwardError> {
        // Gate under the registry lock; copy the plugin id out; release. The RPC
        // below can take seconds and must not hold the lock.
        let plugin_id = {
            let reg = self
                .registry
                .lock()
                .map_err(|_| ForwardError::Internal("registry lock poisoned".into()))?;
            let rec = reg
                .gate(connector_id, command, idempotency_key)
                .map_err(ForwardError::Refused)?;
            rec.id.clone()
        };

        let process = self
            .procs
            .lock()
            .map_err(|_| ForwardError::Internal("process table poisoned".into()))?
            .running
            .get(&plugin_id)
            .cloned();
        let Some(process) = process else {
            // The registry said Running/Unhealthy but no live process exists —
            // a crash the supervisor has not observed yet. Refuse honestly.
            return Err(ForwardError::NotRunning { plugin_id });
        };

        let mut params = json!({
            "connectorId": connector_id,
            "command": command,
        });
        if let Some(p) = payload {
            params["payload"] = p;
        }
        if let Some(k) = idempotency_key {
            params["requestId"] = json!(k);
        }

        process
            .request("connector.request", params, timeout)
            .map_err(ForwardError::Call)
    }

    /// Events since `since` (exclusive), for polling consumers.
    pub fn events_since(&self, since: u64, limit: usize) -> Vec<ReceivedEvent> {
        match self.events.ring.lock() {
            Ok(ring) => ring
                .iter()
                .filter(|e| e.seq > since)
                .take(limit)
                .cloned()
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    // --- internals ---------------------------------------------------------

    /// Send one kept event to the linked account: its flows, then its apps'
    /// logic. Runs on the outbox thread (see `crate::link::outbox`), so a
    /// FormLogic that cannot be reached holds up nothing on this machine.
    fn deliver_to_account(&self, plugin_id: &str, envelope: &Value) -> crate::link::outbox::Delivery {
        use crate::link::outbox::Delivery;
        let event = Event {
            name: envelope.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
            source: plugin_id.to_string(),
            correlation_id: envelope.get("correlationId").and_then(Value::as_str).unwrap_or("").to_string(),
            idempotency_key: envelope.get("idempotencyKey").and_then(Value::as_str).unwrap_or("").to_string(),
            data: envelope.get("data").cloned().unwrap_or(Value::Null),
            origin_run: None,
        };
        let flows = self.fan_out_to_linked_flows(&event, envelope);
        let logic = self.fan_out_to_app_logic(&event, envelope);
        match (flows, logic) {
            (Delivery::Later(a), _) | (_, Delivery::Later(a)) => Delivery::Later(a),
            (Delivery::Refused(a), Delivery::Refused(b)) => Delivery::Refused(format!("{a}; {b}")),
            (Delivery::Refused(a), _) | (_, Delivery::Refused(a)) => Delivery::Refused(a),
            _ => Delivery::Done,
        }
    }

    /// Keep an event for the linked account, when there is one. Err when it
    /// could not be written, so the plugin is not told it arrived.
    fn queue_for_account(&self, plugin_id: &str, envelope: &Value) -> Result<(), String> {
        let link = self.link.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(account) = link.as_ref().and_then(|l| l.account()) else {
            // Not linked: this desktop is all there is.
            return Ok(());
        };
        let Some(outbox) = self.outbox.get() else { return Ok(()) };
        match outbox.enqueue(plugin_id, &account_key(&account), envelope) {
            Ok(()) => Ok(()),
            Err(crate::link::outbox::Enqueue::Full) => {
                let name = envelope.get("name").and_then(Value::as_str).unwrap_or("(unnamed)").to_string();
                self.record_dead(
                    plugin_id,
                    &name,
                    DeadReason::NotDelivered { detail: "not sent: too many events were already waiting for the linked account".into() },
                    envelope.clone(),
                );
                Ok(())
            }
            Err(crate::link::outbox::Enqueue::NotWritten(e)) => Err(e),
        }
    }

    /// Reserve a run on the linked account for every binding this event fires.
    ///
    /// The local dispatch has already happened. Each outcome is logged with the
    /// binding it belongs to, because a trigger that silently does nothing is
    /// the hardest kind of automation bug to find. `Later` when a run could not
    /// be reserved for want of FormLogic (reserving again is harmless: each run
    /// is keyed by its binding and the event).
    fn fan_out_to_linked_flows(&self, event: &crate::bridge::triggers::Event, envelope: &Value) -> crate::link::outbox::Delivery {
        use crate::link::outbox::Delivery;
        let link = {
            let guard = self.link.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(l) => l.clone(),
                None => return Delivery::Done,
            }
        };
        let Some(account) = link.account() else {
            return Delivery::Done;
        };
        let Some(spec) = crate::link::descriptor::find(link.data_dir(), &account.connector_id)
            .and_then(|d| d.flows)
        else {
            return Delivery::Done;
        };
        let bindings = match self.flow_bindings.load(&account, &spec) {
            Ok(b) => b,
            Err(e) => {
                // Not "no bindings": the event waits until they can be read.
                return Delivery::Later(format!("the account's flow triggers could not be read: {e}"));
            }
        };
        // The envelope, not the internal event: conditions are authored against
        // `event.data.*` as it appears on the wire. Nothing is locked here
        // either — this lane never held the ledger — so the batch simply runs.
        // The provider's prelude, resolved ONCE for this lane. Its conditions
        // are the provider's, written against the provider's own helpers, so a
        // prelude this desktop has not got skips every one of them rather than
        // deciding them in half a language.
        let held = crate::link::script_profile::resolve(&account, link.data_dir());
        let selection = crate::link::flows::select(
            self.script_evaluator().as_ref(),
            &bindings,
            &event.name,
            &event.source,
            envelope,
            held.prelude(),
        );
        for (binding, reason) in &selection.skipped {
            log::info!(
                "flow binding {} did not fire for {}: {}",
                binding.id,
                event.name,
                reason.message()
            );
        }
        let (mut later, mut refused) = (None, Vec::new());
        for binding in selection.fire {
            match crate::link::flows::reserve(
                &account,
                &spec,
                binding,
                &event.name,
                &event.correlation_id,
                &event.idempotency_key,
                envelope,
            ) {
                Ok(Some(run_id)) => {
                    log::info!("flow {} queued run {run_id} for {}", binding.id, event.name)
                }
                // Already reserved by an earlier delivery — the idempotency
                // gate doing its job, not a failure.
                Ok(None) => log::debug!("flow binding {} already had this event", binding.id),
                Err(e) if e.later => {
                    later.get_or_insert(format!("flow trigger {}: {e}", binding.id));
                }
                Err(e) => {
                    log::warn!("flow binding {} could not reserve a run: {e}", binding.id);
                    refused.push(format!("flow trigger {} could not start its flow: {e}", binding.id));
                }
            }
        }
        match (later, refused.is_empty()) {
            (Some(why), _) => crate::link::outbox::Delivery::Later(why),
            (None, false) => crate::link::outbox::Delivery::Refused(refused.join("; ")),
            (None, true) => crate::link::outbox::Delivery::Done,
        }
    }

    /// Run the linked account's app LOGIC SCRIPTS against this event.
    ///
    /// A third mechanism, separate from the local trigger dispatch above and
    /// from the flows lane beside it: the apps installed on the account carry
    /// small scripts that turn an event into record writes, and this desktop
    /// implements none of that itself. Without this call the flows fire and the
    /// transcript stays empty.
    ///
    /// Best effort and non-fatal, on the same terms as the flow fan-out. It is
    /// also the most expensive thing on this thread — every script of every app
    /// runs on the warm script host, in the third and last batch this event
    /// sends — so an account whose apps carry no event scripts must and does
    /// cost nothing here, not even a batch.
    fn fan_out_to_app_logic(&self, event: &crate::bridge::triggers::Event, envelope: &Value) -> crate::link::outbox::Delivery {
        use crate::link::outbox::Delivery;
        let link = {
            let guard = self.link.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(l) => l.clone(),
                None => return Delivery::Done,
            }
        };
        let Some(account) = link.account() else {
            return Delivery::Done;
        };
        let Some(spec) = crate::link::descriptor::find(link.data_dir(), &account.connector_id)
            .and_then(|d| d.app_logic)
        else {
            return Delivery::Done;
        };
        let apps = match self.app_logic.load(&account, &spec) {
            Ok(a) => a,
            Err(e) => {
                // Not "no apps": the event waits until they can be read.
                return Delivery::Later(format!("the account's app logic could not be read: {e}"));
            }
        };
        let held = crate::link::script_profile::resolve(&account, link.data_dir());
        let storage = crate::link::app_logic::StorageStore::open(link.data_dir());
        let connector = |connector_id: &str,
                         command: &str,
                         payload: Option<Value>,
                         key: &str|
         -> Result<Value, String> {
            self.forward_connector(connector_id, command, payload, Some(key), CONNECTOR_TIMEOUT)
                .map_err(|e| match e {
                    ForwardError::Refused(refusal) => {
                        refusal.message()
                    }
                    ForwardError::NotRunning { plugin_id } => {
                        format!("the {plugin_id} plugin is not running on this desktop")
                    }
                    ForwardError::Call(err) => {
                        format!("the {connector_id} plugin did not answer {command:?}: {err}")
                    }
                    ForwardError::Internal(message) => message,
                })
        };
        let mut later = None;
        for outcome in crate::link::app_logic::handle_event(
            &account,
            &spec,
            &apps,
            &storage,
            envelope,
            self.script_evaluator().as_ref(),
            held.prelude(),
            &connector,
        ) {
            // Every outcome, always. A script that quietly records nothing is
            // the hardest kind of failure to find — the install looks fine and
            // the console is simply empty.
            match &outcome.detail {
                Ok(what) => log::info!(
                    "app logic {}/{} {}: {what}",
                    outcome.app,
                    outcome.script,
                    outcome.effect
                ),
                // A write FormLogic was not there for: the event is sent again
                // later, and said once there, not here each time.
                Err(why) if outcome.transient => {
                    later.get_or_insert(format!("app logic {}/{}: {why}", outcome.app, outcome.script));
                }
                Err(why) => log::warn!(
                    "app logic {}/{} {} failed for {}: {why}",
                    outcome.app,
                    outcome.script,
                    outcome.effect,
                    event.name
                ),
            }
        }
        later.map_or(Delivery::Done, Delivery::Later)
    }

    /// The event thread's work: ring, triggers, ack.
    fn process_event(&self, plugin_id: &str, envelope: Value) {
        let name = envelope
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let idempotency_key = envelope
            .get("idempotencyKey")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let event = Event {
            name: name.clone(),
            source: plugin_id.to_string(),
            correlation_id: envelope
                .get("correlationId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            idempotency_key: idempotency_key.clone(),
            data: envelope.get("data").cloned().unwrap_or(Value::Null),
            origin_run: None,
        };

        // A call ended, or a request to reach the owner came out: whatever rings for it is over.
        if matches!(name.as_str(), "aokie.call.ended" | "aokie.call.assistance.resolved") {
            if let Some(ring) = self.ring() {
                crate::ring::apply_plugin_event(&ring, &name, &event.data, &event.correlation_id);
            }
        }

        // Recorded in the calendar only while a plugin provides it (turned off, it is not written to).
        if name == "aokie.appointment.requested" && crate::calendar::available() {
            if let Some(cal) = crate::calendar::shared() {
                cal.record_request(&event.data);
            }
        }

        let dispatched = self.dispatch_event(&event);
        if let Some(reason) = dispatched.dead {
            self.record_dead(&event.source, &event.name, reason, envelope.clone());
        }
        let outcomes = dispatched.outcomes;

        // …and the SAME event to the linked account: its own flows (the flows a
        // user built live in the provider's web app) and its apps' logic
        // scripts (which write the records). Both need FormLogic, so the event
        // is kept on disk and sent by the outbox thread: nothing here waits on
        // FormLogic, and nothing is lost while it cannot be reached. Kept
        // before the ack, for the same reason the local dispatch is; if it
        // could not be kept, no ack, and the plugin sends it again.
        let kept = self.queue_for_account(plugin_id, &envelope);
        if let Err(e) = &kept {
            log::warn!("{name} could not be kept for the linked account ({e}); the plugin will send it again");
        }

        let seq = self.events.seq.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut ring) = self.events.ring.lock() {
            if ring.len() >= EVENT_RING_CAPACITY {
                ring.pop_front();
            }
            ring.push_back(ReceivedEvent {
                seq,
                received_at_ms: now_ms(),
                envelope,
                outcomes,
            });
        }

        // Ack AFTER the dispatch outcome is recorded: the ack is the plugin's
        // permission to stop re-delivering, and an event acked before its runs
        // were reserved would be lost entirely if we crashed in between. The
        // ledger's idempotency keys make the redelivery harmless.
        if !idempotency_key.is_empty() && kept.is_ok() {
            let process = self
                .procs
                .lock()
                .ok()
                .and_then(|t| t.running.get(plugin_id).cloned());
            if let Some(p) = process {
                let _ = p.ack_event(&idempotency_key);
            }
        }
    }

    /// Decide conditions with `evaluator` instead of the process-wide host.
    ///
    /// For tests. Nothing in the product calls it: the default IS the host.
    pub fn set_script_evaluator(
        &self,
        evaluator: std::sync::Arc<dyn crate::bridge::script_host::ScriptBatch>,
    ) {
        *self.scripts.lock().unwrap_or_else(|e| e.into_inner()) = evaluator;
    }

    /// What decides conditions right now. The save-time check on
    /// `POST /api/bridge/triggers` reaches the same evaluator through here, so a
    /// test that substitutes one substitutes it for every lane at once.
    pub fn script_evaluator(&self) -> std::sync::Arc<dyn crate::bridge::script_host::ScriptBatch> {
        self.scripts.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The linked provider's prelude, if this desktop is linked at all: what
    /// is already held here, as this desktop's own triggers do not wait on the
    /// provider (the heartbeat keeps the copy fresh).
    ///
    /// Unlinked is [`Held::NotRequired`] rather than `Missing`: a desktop with
    /// no account has no provider to be missing a prelude FROM, and its own
    /// trigger bindings are none the worse for it.
    fn linked_prelude(&self) -> crate::link::script_profile::Held {
        let link = {
            let guard = self.link.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(l) => l.clone(),
                None => return crate::link::script_profile::Held::NotRequired,
            }
        };
        match link.account() {
            Some(account) => crate::link::script_profile::resolve_held(&account, link.data_dir()),
            None => crate::link::script_profile::Held::NotRequired,
        }
    }

    /// Let a binary supply the companion broker once its HTTP surface exists.
    pub fn set_companion_broker(&self, broker: CompanionBroker) {
        *self.companion.lock().unwrap_or_else(|e| e.into_inner()) = Some(broker);
    }

    /// Give the host the linked account, so plugin events can reach the flows
    /// the user built there. Until this is set, events stay local — which is
    /// the correct behaviour for an unlinked desktop.
    pub fn set_link(&self, link: crate::link::LinkHandle) {
        // Where events wait for the account, with what an earlier run left there.
        self.outbox.get_or_init(|| crate::link::outbox::Outbox::open(link.data_dir())).make_current();
        *self.link.lock().unwrap_or_else(|e| e.into_inner()) = Some(link);
        // A fresh link may belong to a different account entirely; keeping the
        // previous account's bindings would fire the wrong flows, and keeping
        // its logic scripts would write the previous account's records.
        self.flow_bindings.invalidate();
        self.app_logic.invalidate();
    }

    /// Broker a companion admission for the plugin that hosts the WebRTC peer.
    ///
    /// The plugin sends its own view of the endpoint binding. The host ignores
    /// it and rebuilds the roster from its own identity store, because the
    /// plugin is the process a phone talks to — exactly the process that must
    /// not get to choose which phones are trusted.
    fn handle_companion_admission(
        &self,
        plugin_id: &str,
        params: Value,
    ) -> Result<Value, (String, String)> {
        let granted = self
            .registry
            .lock()
            .map(|reg| reg.grants(plugin_id, "oaiy.companion.admission"))
            .unwrap_or(false);
        if !granted {
            return Err((
                "capability_denied".into(),
                format!("{plugin_id} does not declare the oaiy.companion.admission capability"),
            ));
        }

        let broker = self
            .companion
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .ok_or_else(|| {
                (
                    "unavailable".to_string(),
                    "this build has no companion broker".to_string(),
                )
            })?;

        let identity = broker
            .companion
            .identity_for(plugin_id)
            .map_err(|(_, message)| ("capability_denied".to_string(), message))?;
        let status = identity.status();

        // OUR roster, not the plugin's.
        if status.approved_mobiles.is_empty() {
            return Err((
                "not_paired".into(),
                "no Companion device has been approved on this desktop yet".into(),
            ));
        }
        // Sorted: the issuer needs the thumbprints strictly ascending and checks that what it echoes is what was
        // sent, and the plugin compares the echo with a sorted list. The roster is kept in the order phones were
        // approved, so two phones or more failed here at random.
        let live = crate::companion::RosterSnapshot::of(&status).ok_or_else(|| {
            (
                "unavailable".to_string(),
                "the desktop endpoint identity is unavailable".to_string(),
            )
        })?;
        // What the plugin holds, not what the roster is now. The plugin learns its roster once, at `plugin.init`, and
        // refuses an admission that differs from it, so a phone approved after that used to break the admissions of every
        // phone until the plugin was started again. The roster it was handed is presented while every phone in it is
        // still approved; a phone approved since waits for the next start with the plugin, which does not know it either.
        // A phone revoked since is NOT presented again: the roster as it is now goes to the issuer, which the plugin
        // refuses, as it always has, until it is started again.
        let held = self.procs.lock().ok().and_then(|t| t.rosters.get(plugin_id).cloned());
        let roster = match held {
            Some(held) if held.still_holds_in(&live) => held,
            _ => live,
        };
        let binding = serde_json::json!({
            "endpointPublicKey": roster.endpoint_key,
            "holderKeyThumbprint": roster.endpoint_key.thumbprint,
            "approvedPeerKeyThumbprints": roster.thumbprints,
            "peerRosterRevision": roster.revision,
            "peerRosterHash": roster.hash,
        });

        let config = broker.upstream.get().ok_or_else(|| {
            (
                "unavailable".to_string(),
                "no Companion relay is configured on this desktop".to_string(),
            )
        })?;
        // The plugin may name its own app; otherwise the configured default.
        // Refusing when neither exists beats guessing: the app id is what scopes
        // the admission on the issuer.
        let app_id = params
            .get("appId")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| config.app_id.clone())
            .ok_or_else(|| {
                (
                    "invalid_request".to_string(),
                    "no appId was supplied and the relay config sets no default".to_string(),
                )
            })?;
        let display_name = params.get("displayName").and_then(Value::as_str);

        crate::companion::upstream::broker(&config, &app_id, plugin_id, display_name, &binding)
            .map_err(|message| ("upstream_error".to_string(), message))
    }

    /// Give this host its own ring (a test's); a host without one uses the desktop's.
    pub fn set_ring(&self, ring: Arc<crate::ring::Ring>) {
        if let Ok(mut r) = self.ring.lock() {
            *r = Some(ring);
        }
    }

    /// The ring the phone plugin's requests and events reach.
    fn ring(&self) -> Option<Arc<crate::ring::Ring>> {
        self.ring.lock().ok().and_then(|r| r.clone()).or_else(crate::ring::shared)
    }

    /// The phone plugin asks who may be rung for a caller who wants the owner (`oaiy.ring.plan`), and tells the
    /// desktop the request is out (`oaiy.ring.opened`). Allowed for a plugin that holds `oaiy.companion.admission`
    /// (already trusted with the device roster). The contract is `docs/contracts/transfer/`: the call is judged on
    /// this desktop's own record of it, a try that is allowed is counted when it is allowed, and nothing rings for a
    /// plan this desktop did not allow for the call.
    pub(crate) fn handle_ring_request(
        &self,
        plugin_id: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, (String, String)> {
        let granted = self
            .registry
            .lock()
            .map(|reg| reg.grants(plugin_id, "oaiy.companion.admission"))
            .unwrap_or(false);
        if !granted {
            return Err((
                "capability_denied".into(),
                format!("{plugin_id} does not declare the oaiy.companion.admission capability"),
            ));
        }
        let Some(ring) = self.ring() else {
            return Err(("unavailable".into(), "this build has no ring".into()));
        };
        Self::ring_request(&ring, method, params)
    }

    /// [`Self::handle_ring_request`] once the plugin is known to hold the capability, for the ring `ring`.
    pub(crate) fn ring_request(ring: &Arc<crate::ring::Ring>, method: &str, params: Value) -> Result<Value, (String, String)> {
        let text = |k: &str| params.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let number = |k: &str| params.get(k).and_then(Value::as_u64);
        let bad = |why: &str| ("invalid_request".to_string(), why.to_string());
        match method {
            "oaiy.ring.plan" => {
                let call = text("callId");
                if call.is_empty() || call.len() > 256 {
                    return Err(bad("oaiy.ring.plan needs a callId"));
                }
                let reason = crate::ring::Reason::parse(&text("reason")).ok_or_else(|| bad("reason is caller_asked, urgent or policy_rule"))?;
                let turns: Vec<String> = params
                    .get("recentCallerTurns")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                    .unwrap_or_default();
                let fallback = crate::ring::CallInfo { from: text("callerNumber"), name: String::new(), turns: crate::ring::phrases::recent(&turns), ..Default::default() };
                Ok(crate::ring::host::plan_result(&ring.plan_for_plugin(&call, reason, fallback)))
            }
            _ => {
                let (Some(call_epoch), Some(owner_epoch), Some(expires_at)) = (number("callEpoch"), number("ownerEpoch"), number("expiresAt")) else {
                    return Err(bad("oaiy.ring.opened needs callEpoch, ownerEpoch and expiresAt"));
                };
                let opened = crate::ring::contract::OpenedParams {
                    plan_id: text("planId"),
                    request_id: text("requestId"),
                    call_id: text("callId"),
                    call_epoch,
                    owner_epoch,
                    expires_at,
                };
                if opened.plan_id.is_empty() || opened.request_id.is_empty() || opened.call_id.is_empty() {
                    return Err(bad("oaiy.ring.opened needs planId, requestId and callId"));
                }
                ring.opened(&opened).map(|_| json!({ "ok": true })).map_err(|e| (e.code.to_string(), e.message))
            }
        }
    }

    /// Answer a plugin-initiated request (`flow.run`, `companion.admission`, `oaiy.ring.plan`, `oaiy.ring.opened`).
    fn handle_plugin_request(
        &self,
        plugin_id: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, (String, String)> {
        if crate::isolated::active() {
            return Err(("isolated_capability_unavailable".into(), crate::isolated_policy::REFUSAL.into()));
        }
        if method == "companion.admission" {
            return self.handle_companion_admission(plugin_id, params);
        }
        if method == "oaiy.ring.plan" || method == "oaiy.ring.opened" {
            return self.handle_ring_request(plugin_id, method, params);
        }
        if method != "flow.run" {
            return Err((
                "invalid_request".into(),
                format!(
                    "unknown method {method:?}; this host answers flow.run, companion.admission, oaiy.ring.plan and oaiy.ring.opened"
                ),
            ));
        }

        // The capability gate. A plugin that can start arbitrary flows reaches
        // every capability those flows hold, so this is checked per call even
        // though the manifest was validated at load.
        let granted = self
            .registry
            .lock()
            .map(|reg| reg.grants(plugin_id, "oaiy.flow.run"))
            .unwrap_or(false);
        if !granted {
            return Err((
                "capability_denied".into(),
                format!("{plugin_id} does not declare the oaiy.flow.run capability"),
            ));
        }

        let flow_id = params
            .get("flowId")
            .or_else(|| params.get("flowSlug"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(flow_id) = flow_id else {
            return Err(("invalid_request".into(), "flow.run needs a flowId".into()));
        };

        // The phone's business lookup: answered by the calendar (while a plugin
        // provides it), unless the person stored a flow of that name to answer it
        // their way. With the calendar off it is a flow like any other.
        if flow_id == "business-lookup" && crate::calendar::available() {
            if let Some(cal) = crate::calendar::shared().filter(|c| !c.flow_answers(&flow_id)) {
                let input = params.get("input").cloned().unwrap_or(Value::Null);
                let text = |k: &str| input.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                let digest = cal.lookup(&text("question"), &text("from"), crate::calendar::local_now());
                return Ok(json!({
                    "runId": format!("calendar_{}", now_ms()),
                    "status": "done",
                    "result": { "digest": digest },
                }));
            }
        }

        let correlation = params
            .get("correlationId")
            .and_then(Value::as_str)
            .unwrap_or("plugin")
            .to_string();
        // A plugin that supplies no idempotency key gets a unique one: it asked
        // for no dedupe, and inventing a stable key on its behalf would silently
        // collapse distinct requests.
        let idempotency_key = params
            .get("idempotencyKey")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("plugin:{plugin_id}:{}:{}", now_ms(), rand_suffix()));

        let req = RunRequest {
            caller_product: format!("plugin:{plugin_id}"),
            flow_id: Some(flow_id),
            inline_graph: false,
            input: params.get("input").cloned(),
            timeout_ms: params.get("timeoutMs").and_then(Value::as_u64),
            mode: "async".into(),
            correlation_id: correlation,
            idempotency_key,
            lineage: LineageRef::default(),
            trigger_event: None,
        };

        let mut ledger = self
            .ledger
            .lock()
            .map_err(|_| ("internal".to_string(), "ledger lock poisoned".to_string()))?;
        let run = match ledger.reserve(&req) {
            ReserveOutcome::Reserved(run) | ReserveOutcome::Duplicate(run) => run,
            ReserveOutcome::Refused { reason } => {
                return Err(("invalid_request".into(), format!("refused: {reason}")))
            }
        };
        drop(ledger);
        // A plugin asks with a budget and reads `result` (Aokie's lookups, within
        // seconds during a call): wait for the run, up to that budget, on this
        // request's own thread. Past it, answer as before: accepted, still running.
        let deadline = Instant::now() + std::time::Duration::from_millis(req.timeout_ms.unwrap_or(0).min(60_000));
        let mut current = run;
        while !current.status.is_terminal() && Instant::now() < deadline {
            thread::sleep(std::time::Duration::from_millis(100));
            match self.ledger.lock().ok().and_then(|l| l.get(&current.run_id)) {
                Some(r) => current = r,
                None => break,
            }
        }
        let output = current.output.clone().unwrap_or(Value::Null);
        // The flow's own output (the engine wraps it beside how it ran); a text answer as `answer`.
        let own = output.get("output").cloned().unwrap_or(output);
        let result = match own {
            Value::String(s) => json!({ "answer": s }),
            Value::Null => Value::Null,
            other => other,
        };
        Ok(json!({
            "runId": current.run_id,
            "status": if current.status == RunStatus::Succeeded { json!("done") } else { json!(current.status) },
            "result": result,
            "error": current.error,
        }))
    }

    /// One supervisor tick: probe, detect exits, schedule bounded restarts.
    fn supervise(self: &Arc<Self>, trackers: &mut HashMap<String, HealthTracker>) {
        // Fire due restarts. They live in the table so stop() can cancel them;
        // remove-then-start keeps each one at-most-once.
        let now = Instant::now();
        let due: Vec<String> = match self.procs.lock() {
            Ok(mut t) => {
                let due: Vec<String> = t
                    .restarts
                    .iter()
                    .filter(|(_, at)| **at <= now)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in &due {
                    t.restarts.remove(id);
                }
                due
            }
            Err(_) => Vec::new(),
        };
        for id in due {
            // start() re-checks user_disabled and manifest validity; a plugin
            // stopped or disabled while the restart was pending was already
            // cancelled out of the map by stop().
            let _ = self.start(&id);
        }

        let snapshot: Vec<(String, Arc<PluginProcess>)> = match self.procs.lock() {
            Ok(t) => t.running.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Err(_) => return,
        };

        for (id, process) in snapshot {
            // Exit first: probing a dead process would just add a 5s timeout to
            // what check_exited answers instantly.
            if let Some(code) = process.check_exited() {
                // The snapshot is stale by up to a whole tick, and stop() may
                // have removed — or a new start() replaced — this entry while we
                // were probing another plugin. Only treat the exit as a crash if
                // the table still holds THIS process (`Arc::ptr_eq`); otherwise
                // the exit was a graceful stop already accounted for, and
                // "crashing" it would overwrite Stopped, burn a restart attempt,
                // and resurrect a plugin the user shut down.
                let still_ours = match self.procs.lock() {
                    Ok(mut t) => match t.running.get(&id) {
                        Some(current) if Arc::ptr_eq(current, &process) => {
                            t.running.remove(&id);
                            t.started_at.remove(&id);
                            true
                        }
                        _ => false,
                    },
                    Err(_) => false,
                };
                if !still_ours {
                    trackers.remove(&id);
                    continue;
                }
                trackers.remove(&id);
                let reason = match code {
                    Some(c) => format!("Exited with code {c}."),
                    None => "Exited (killed or crashed with no code).".into(),
                };
                let attempts = self
                    .registry
                    .lock()
                    .map(|mut reg| reg.note_restart(&id))
                    .unwrap_or(u32::MAX);
                if should_restart(attempts.saturating_sub(1)) {
                    let delay = restart_delay(attempts);
                    self.set_state(
                        &id,
                        PluginState::Crashed,
                        Some(format!("{reason} Restarting in {}s (attempt {attempts}).", delay.as_secs())),
                    );
                    if let Ok(mut t) = self.procs.lock() {
                        t.restarts.insert(id.clone(), Instant::now() + delay);
                    }
                } else {
                    self.set_state(
                        &id,
                        PluginState::Crashed,
                        Some(format!(
                            "{reason} Restarted {} times without staying up; start it manually once the cause is fixed.",
                            attempts.saturating_sub(1)
                        )),
                    );
                }
                continue;
            }

            // Alive, and alive is what the restart budget is bought with.
            self.refill_restart_budget_if_stable(&id);

            let health = process.health();
            // A stop/restart may have replaced this process during the probe.
            // Keep ownership stable while publishing its result so the old
            // process cannot mark a fresh one healthy or undo a manual stop.
            let processes = match self.procs.lock() {
                Ok(processes) => processes,
                Err(_) => continue,
            };
            if !processes.running.get(&id).is_some_and(|current| Arc::ptr_eq(current, &process)) {
                trackers.remove(&id);
                continue;
            }
            if let Ok(mut registry) = self.registry.lock() {
                registry.note_health(&id, health.as_ref().map_err(|error| error.to_string()));
            }
            let tracker = trackers.entry(id.clone()).or_default();
            match health {
                Ok(v) if v.get("status").and_then(Value::as_str) == Some("ok") => {
                    if tracker.record_ok() == HealthVerdict::Ok {
                        // Only lift Unhealthy → Running when it was unhealthy;
                        // set_state on Running clears the reason either way.
                        let was_unhealthy = self
                            .registry
                            .lock()
                            .ok()
                            .and_then(|reg| reg.get(&id).map(|r| r.state == PluginState::Unhealthy))
                            .unwrap_or(false);
                        if was_unhealthy {
                            self.set_state(&id, PluginState::Running, None);
                        }
                    }
                }
                Ok(v) => {
                    // The plugin answered "degraded"/"error": alive, honest, and
                    // telling us something is wrong. Not a miss.
                    let detail = v
                        .get("detail")
                        .and_then(Value::as_str)
                        .unwrap_or("the plugin reports itself degraded")
                        .to_string();
                    if let HealthVerdict::Unhealthy { detail, .. } =
                        tracker.record_self_reported_degraded(detail)
                    {
                        self.set_state(&id, PluginState::Unhealthy, Some(detail));
                    }
                }
                Err(e) => {
                    if let HealthVerdict::Unhealthy { detail, .. } =
                        tracker.record_miss(e.to_string())
                    {
                        self.set_state(&id, PluginState::Unhealthy, Some(detail));
                    }
                }
            }
        }
    }

    /// Run the trigger dispatcher over one event.
    ///
    /// Returns the per-binding outcomes for the event ring, and — when the event
    /// should have produced work but did not — the reason to dead-letter it.
    fn dispatch_event(&self, event: &Event) -> Dispatched {
        if crate::isolated::active() {
            return Dispatched {
                outcomes: vec![crate::isolated_policy::REFUSAL.into()],
                dead: None,
                reserved: false,
            };
        }
        let bindings: Vec<TriggerBinding> = match self.triggers.lock() {
            Ok(t) => t.list().to_vec(),
            Err(_) => Vec::new(),
        };

        // Conditions FIRST, holding NOTHING. `evaluate_conditions` takes the
        // script host's process-wide batch lock and, on the first conditioned
        // event after boot, spawns the child that serves it. Under the ledger
        // lock that would queue every plugin event in the process behind a
        // script host that is starting — and the ledger lock is also what the
        // HTTP surface takes to reserve, claim and finalise runs. So the
        // engine is asked out here, and `dispatch` is handed the answers.
        //
        // An event whose bindings carry no conditions sends no batch and starts
        // no child: `conditions::decide` returns early on an empty job list.
        // This desktop's OWN bindings, so the prelude is taken when there is
        // one and its absence is not a refusal: nobody else decides these.
        let held = self.linked_prelude();
        let verdicts = crate::bridge::triggers::evaluate_conditions(
            self.script_evaluator().as_ref(),
            &bindings,
            event,
            held.prelude_or_bare(),
        );

        let results = match self.ledger.lock() {
            Ok(mut ledger) => dispatch(&mut ledger, &bindings, event, &verdicts),
            // Not a skip: dispatch never ran. Nothing can have been reserved, so
            // this is always a dead letter.
            Err(_) => {
                return Dispatched {
                    outcomes: vec!["ledger lock poisoned; nothing dispatched".into()],
                    dead: Some(DeadReason::NotReserved {
                        detail: "the run ledger was unavailable, so nothing was dispatched".into(),
                    }),
                    reserved: false,
                }
            }
        };

        let dead = dead_reason_for(&results);
        let reserved = results
            .iter()
            .any(|o| matches!(o, DispatchOutcome::Reserved { .. }));
        let outcomes: Vec<String> = results
            .into_iter()
            .map(|o| match o {
                DispatchOutcome::Reserved { binding_id, run } => {
                    format!("{binding_id}: reserved {}", run.run_id)
                }
                DispatchOutcome::Skipped { binding_id, reason } => {
                    format!("{binding_id}: skipped — {}", reason.message())
                }
            })
            .collect();
        Dispatched { outcomes, dead, reserved }
    }

    /// Is the app on its way out? See [`ProcTable::shutting_down`].
    fn is_shutting_down(&self) -> bool {
        self.procs.lock().map(|t| t.shutting_down).unwrap_or(false)
    }

    /// Refill a plugin's crash-restart budget once it has actually STAYED up.
    ///
    /// This used to happen the moment `plugin.init` returned, which made
    /// [`super::runner::MAX_RESTART_ATTEMPTS`] unreachable: a plugin that
    /// handshakes and then dies — Aokie answers `plugin.init` before it touches
    /// the radio, so an unplugged dongle looks exactly like this — zeroed its
    /// own counter on every cycle. `should_restart` was therefore always true,
    /// `restart_delay` never left 1s, and the terminal "start it manually once
    /// the cause is fixed" state was dead code. The user got a child process
    /// every ~11 seconds indefinitely, each one grabbing and releasing the
    /// hardware, and was never told what to fix.
    ///
    /// Returns whether the budget was refilled.
    fn refill_restart_budget_if_stable(&self, id: &str) -> bool {
        let stable = match self.procs.lock() {
            Ok(mut t) => {
                let stable = t
                    .started_at
                    .get(id)
                    .is_some_and(|at| at.elapsed() >= STABLE_UPTIME);
                if stable {
                    // Once per life: the entry IS the outstanding claim, so
                    // dropping it keeps every later tick off the registry lock.
                    t.started_at.remove(id);
                }
                stable
            }
            Err(_) => false,
        };
        if stable {
            if let Ok(mut reg) = self.registry.lock() {
                reg.reset_restarts(id);
            }
        }
        stable
    }

    /// Hand a shed event to the shed thread. Never blocks — that is the point.
    ///
    /// The caller is the `EventSink` closure, i.e. the plugin's reader thread,
    /// which also routes every RPC reply (see the module docs). Recording used
    /// to happen inline there: a full dead-letter rewrite plus an `fsync` per
    /// dropped event, and a `procs` lock for the log notice, in front of the
    /// `plugin.health` and `connector.request` replies the same thread still had
    /// to deliver. A flood therefore timed out live connector calls and could
    /// mark a perfectly alive plugin `Unhealthy` — the durable-record feature
    /// undoing the responsiveness the channel exists to protect.
    fn note_shed(&self, plugin_id: &str, envelope: Value) {
        // A full shed channel means the recorder is behind as well. The event is
        // then lost with no record, which is bad — and still the right end of
        // the line, because the only alternative is blocking the reader thread.
        let _ = self.shed_tx.try_send((plugin_id.to_string(), envelope));
    }

    /// Record an event dropped before it reached dispatch.
    fn record_shed(&self, plugin_id: &str, envelope: Value) {
        let name = envelope
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("(unnamed)")
            .to_string();
        self.record_dead(plugin_id, &name, DeadReason::Shed, envelope);
    }

    fn record_dead(&self, source: &str, event: &str, reason: DeadReason, envelope: Value) {
        if let Ok(mut q) = self.dead.lock() {
            q.record(source, event, reason, envelope);
        }
    }

    /// Re-dispatch a stored dead letter against the CURRENT bindings.
    ///
    /// Deliberately the ordinary path, guards and all: a guard-refused event is
    /// offered to the guards again and refused again rather than forced through.
    /// The only change is the idempotency key, which gains an attempt suffix —
    /// sound because a dead letter reserved no run, so there is nothing to
    /// double-execute.
    ///
    /// Returns the dispatch outcomes, and whether any binding actually reserved.
    pub fn redrive(&self, id: &str) -> Option<(Vec<String>, bool)> {
        let item = self.dead.lock().ok()?.get(id)?;
        if matches!(item.reason, DeadReason::NotDelivered { .. }) {
            // It never reached the linked account: it goes there again, through
            // the outbox, as the account linked now.
            let queued = self.link.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|l| l.account()).is_some();
            if !queued {
                return Some((vec!["not sent: this desktop is not linked to an account".into()], false));
            }
            return Some(match self.queue_for_account(&item.source, &item.envelope) {
                Ok(()) => {
                    if let Ok(mut q) = self.dead.lock() {
                        q.remove(id);
                    }
                    (vec!["kept to send to the linked account".into()], true)
                }
                Err(e) => (vec![format!("could not be kept to send: {e}")], false),
            });
        }
        let envelope = &item.envelope;
        let original_key = envelope
            .get("idempotencyKey")
            .and_then(Value::as_str)
            .unwrap_or("");

        let event = Event {
            name: item.event.clone(),
            source: item.source.clone(),
            correlation_id: envelope
                .get("correlationId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            idempotency_key: crate::bridge::DeadLetterQueue::redrive_key(
                original_key,
                item.attempts + 1,
            ),
            data: envelope.get("data").cloned().unwrap_or(Value::Null),
            origin_run: None,
        };

        // The dead reason is deliberately dropped: this entry already exists, and
        // recording a second one for the same event would grow the queue every
        // time someone retries a redrive that keeps failing.
        let Dispatched { outcomes, reserved, .. } = self.dispatch_event(&event);

        if let Ok(mut q) = self.dead.lock() {
            if reserved {
                // Done: leaving it would have someone redriving the same event
                // every morning.
                q.remove(id);
            } else {
                q.note_attempt(id, outcomes.join("; "));
            }
        }
        Some((outcomes, reserved))
    }

    /// Note a dropped event on the plugin's own log ring, so a flood is
    /// diagnosable rather than silent. Best-effort and rate-oblivious — under a
    /// real flood this line itself is shed by the ring's own cap.
    fn logs_drop_notice(&self, plugin_id: &str) {
        if let Ok(t) = self.procs.lock() {
            if let Some(p) = t.running.get(plugin_id) {
                p.logs.push("stderr", "[event dropped] host event queue full".into());
            }
        }
    }

    fn set_state(&self, id: &str, state: PluginState, reason: Option<String>) {
        if let Ok(mut reg) = self.registry.lock() {
            reg.set_state(id, state, reason);
        }
        // After the lock is let go: the modules name their provider's state.
        crate::modules::poke();
    }
}

/// Panic-safe release of a `starting` claim.
///
/// A guard rather than manual removal at each return, because start() has five
/// exit paths and a forgotten one would leave the id permanently "starting" —
/// every later start() would return Ok while doing nothing, an unstartable
/// plugin that reports success.
struct StartClaim<'a> {
    host: &'a PluginHost,
    id: String,
}

impl Drop for StartClaim<'_> {
    fn drop(&mut self) {
        if let Ok(mut t) = self.host.procs.lock() {
            t.starting.remove(&self.id);
            // A stop intent that never met its start (e.g. the spawn failed
            // before the completion check) must not linger and abandon the NEXT
            // legitimate start.
            t.stop_during_start.remove(&self.id);
            // The in-flight handle only covers the window between spawn and
            // registration in `running`; past that, `running` owns the process.
            t.starting_procs.remove(&self.id);
        }
    }
}

/// Why a forwarded connector command failed, mapped for the HTTP layer.
#[derive(Debug)]
pub enum ForwardError {
    /// The gate said no. Carries the registry's typed refusal.
    Refused(GateRefusal),
    /// The registry thinks it is running but no process exists.
    NotRunning { plugin_id: String },
    /// The RPC itself failed.
    Call(CallError),
    Internal(String),
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A short non-cryptographic suffix for minted idempotency keys. Uniqueness
/// within a process lifetime is all that is needed — the key exists to NOT
/// collide, unlike a caller-supplied key which exists to collide on purpose.
fn rand_suffix() -> u64 {
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::ledger::{RunRecord, RunStatus};
    use crate::bridge::triggers::MAX_BINDINGS_PER_EVENT;

    fn reserved(binding: &str) -> DispatchOutcome {
        DispatchOutcome::Reserved {
            binding_id: binding.into(),
            run: RunRecord {
                run_id: "run_1".into(),
                status: RunStatus::Queued,
                caller_product: "aokie".into(),
                flow_id: Some("answer".into()),
                correlation_id: "c".into(),
                idempotency_key: "k".into(),
                input: None,
                timeout_ms: None,
                mode: "async".into(),
                cancel_requested: false,
                idempotent: false,
                runtime: None,
                claimed_by: None,
                output: None,
                error: None,
                reserved_at_ms: 0,
                started_at_ms: None,
                finished_at_ms: None,
                lineage: Default::default(),
                trigger_event: None,
            },
        }
    }

    fn skipped(binding: &str, reason: SkipReason) -> DispatchOutcome {
        DispatchOutcome::Skipped { binding_id: binding.into(), reason }
    }

    #[test]
    fn the_trigger_system_declining_is_not_a_dead_letter() {
        // Every one of these is the design working as documented: triggers fail
        // towards not running. Recording them would bury the real failures under
        // routine not-firing, and the queue would be useless within a day.
        for reason in [
            SkipReason::Disabled,
            SkipReason::ManualMode,
            SkipReason::ConditionFalse,
            SkipReason::Duplicate,
        ] {
            assert_eq!(
                dead_reason_for(&[skipped("b", reason.clone())]),
                None,
                "{reason:?} must not dead-letter"
            );
        }
    }

    #[test]
    fn work_that_was_meant_to_happen_and_did_not_is_a_dead_letter() {
        for reason in [
            SkipReason::ConditionUnevaluatable {
                expression: "event.data.x ==== 1".into(),
                why: "unparseable".into(),
            },
            SkipReason::Guard("depth 17 exceeds the maximum".into()),
            SkipReason::TooManyBindings,
        ] {
            let dead = dead_reason_for(&[skipped("b", reason.clone())]);
            assert!(dead.is_some(), "{reason:?} must dead-letter");
            // The reason travels with it — a dead letter you cannot diagnose is
            // just a mystery with a timestamp.
            assert!(dead.unwrap().message().contains("b: "));
        }
    }

    #[test]
    fn an_event_that_reserved_anything_is_handled() {
        // Fan-out: one binding fired, another was disabled, a third could not
        // evaluate. The event produced work, so nothing was lost.
        let dead = dead_reason_for(&[
            reserved("fires"),
            skipped("off", SkipReason::Disabled),
            skipped(
                "broken",
                SkipReason::ConditionUnevaluatable {
                    expression: "???".into(),
                    why: "unparseable".into(),
                },
            ),
        ]);
        assert_eq!(dead, None);
    }

    #[test]
    fn an_event_nobody_bound_is_not_a_dead_letter() {
        // Most plugin events are bound by nobody. Treating that as a failure
        // would grow the queue without anything being wrong.
        assert_eq!(dead_reason_for(&[]), None);
    }

    #[test]
    fn several_failures_are_reported_together() {
        let dead = dead_reason_for(&[
            skipped("a", SkipReason::Guard("cycle".into())),
            skipped("b", SkipReason::TooManyBindings),
        ])
        .expect("both bindings failed");
        let msg = dead.message();
        assert!(msg.contains("a: "), "{msg}");
        assert!(msg.contains("b: "), "{msg}");
        assert!(msg.contains(&MAX_BINDINGS_PER_EVENT.to_string()), "{msg}");
    }
    // --- the wiring, not just the classifier --------------------------------
    //
    // The unit tests above prove the RULE; these prove the plumbing that applies
    // it. Worth the extra setup: the two previous features in this area both
    // passed their unit tests while being broken end to end, and a dead-letter
    // queue that is never written to is worse than none — it reads as "nothing
    // was lost".

    use crate::bridge::deadletters::DeadLetterQueue;
    use crate::bridge::triggers::BindingMode;

    struct Sandbox(PathBuf);
    impl Sandbox {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::AtomicU32;
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!("oaiy-host-{tag}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn binding(id: &str, condition: Option<&str>) -> TriggerBinding {
        TriggerBinding {
            id: id.into(),
            event: "aokie.call.incoming".into(),
            flow_id: "answer".into(),
            mode: BindingMode::Async,
            enabled: true,
            condition: condition.map(str::to_string),
            input_map: Default::default(),
            sort_order: 0,
        }
    }

    fn envelope() -> Value {
        json!({
            "name": "aokie.call.incoming",
            "idempotencyKey": "evt-1",
            "correlationId": "call_1",
            "data": { "from": "+61400000000" }
        })
    }

    /// A host over temp dirs, with `bindings` installed.
    ///
    /// Written straight to the bindings file rather than through `upsert`,
    /// because that is the path a binding hand-edited into triggers.json, or
    /// saved by an older build, actually takes — and it is the path the `$event`
    /// migration runs on.
    ///
    /// Conditions are decided by the process-wide script host unless a test
    /// substitutes one (`host_evaluating`). A binding with no condition asks it
    /// nothing, which is why most of the tests here need no evaluator at all.
    fn host_with(tag: &str, bindings: Vec<TriggerBinding>) -> (Sandbox, Arc<PluginHost>) {
        let sb = Sandbox::new(tag);
        let path = sb.0.join("triggers.json");
        std::fs::write(&path, serde_json::to_string(&bindings).unwrap()).unwrap();
        let triggers: TriggerStoreHandle = Arc::new(Mutex::new(TriggerStore::load(path)));
        let host = PluginHost::new(
            crate::plugins::registry::new_handle(sb.0.join("plugins")),
            crate::bridge::ledger::new_handle(),
            triggers,
            crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );
        (sb, host)
    }

    /// Install a plugin manifest so the registry grants the named capabilities.
    fn install_plugin(sb: &Sandbox, id: &str, capabilities: &[&str]) {
        let dir = sb.0.join("plugins").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = serde_json::json!({
            "schemaVersion": 3,
            "id": id,
            "name": id,
            "version": "0.1.0",
            "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe" },
            "capabilities": capabilities,
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    }

    fn broker_for(sb: &Sandbox, host: &Arc<PluginHost>) -> crate::companion::routes::CompanionHandle {
        let companion =
            crate::companion::new_handle(sb.0.clone(), host.registry.clone());
        let upstream = crate::companion::upstream::UpstreamStore::open(sb.0.join("relay.json"));
        host.set_companion_broker(CompanionBroker {
            companion: companion.clone(),
            upstream,
        });
        companion
    }

    #[test]
    fn the_phones_business_lookup_is_answered_by_the_calendar() {
        // Aokie's lookup_business_data runs `business-lookup` and reads
        // `{status:"done", result:{digest}}`; with no flow of that name stored,
        // the calendar answers, at once.
        let (sb, host) = host_with("lookup", vec![]);
        install_plugin(&sb, "aokie", &["flow.run"]);
        host.registry.lock().unwrap().scan();
        crate::calendar::init(&sb.0);
        let _calendar = crate::modules::test_gate::enable(&[crate::modules::PHONE, crate::modules::CALENDAR]);
        let out = host
            .handle_plugin_request("aokie", "flow.run", serde_json::json!({"flowSlug": "business-lookup", "input": {"question": "Any times this week?", "from": "+61400000000"}, "timeoutMs": 6000}))
            .unwrap();
        assert_eq!(out["status"], "done");
        let digest = out["result"]["digest"].as_str().unwrap();
        assert!(digest.contains("Opening hours:") && digest.contains("Free times"), "{digest}");
    }

    #[test]
    fn with_the_calendar_off_the_business_lookup_is_a_flow_like_any_other() {
        let (sb, host) = host_with("lookup-off", vec![]);
        install_plugin(&sb, "aokie", &["flow.run"]);
        host.registry.lock().unwrap().scan();
        crate::calendar::init(&sb.0);
        let _off = crate::modules::test_gate::enable(&[]);
        let out = host
            .handle_plugin_request("aokie", "flow.run", serde_json::json!({"flowSlug": "business-lookup", "input": {"question": "Any times this week?"}, "timeoutMs": 300}))
            .unwrap();
        // Not the calendar's answer: a run of the flow, which no worker takes here.
        assert_eq!(out["status"], "queued");
        assert!(out["result"].get("digest").is_none());
    }

    #[test]
    fn another_flow_is_waited_for_only_within_its_budget() {
        // No worker runs here, so the run stays queued: the answer comes when
        // the plugin's budget is spent, saying so, rather than never.
        let (sb, host) = host_with("flowwait", vec![]);
        install_plugin(&sb, "aokie", &["flow.run"]);
        host.registry.lock().unwrap().scan();
        let started = Instant::now();
        let out = host
            .handle_plugin_request("aokie", "flow.run", serde_json::json!({"flowId": "manager-action-plan", "input": {}, "timeoutMs": 300}))
            .unwrap();
        assert!(started.elapsed() >= Duration::from_millis(250) && started.elapsed() < Duration::from_secs(5));
        assert_eq!(out["status"], "queued");
        assert!(out["runId"].as_str().is_some_and(|id| !id.is_empty()));
    }

    #[test]
    fn admission_is_refused_to_a_plugin_that_did_not_declare_the_capability() {
        // The capability is the whole authorisation for reaching the roster.
        // A plugin that can broker admissions decides which phones take live
        // call audio, so this is checked per call, not once at load.
        let (sb, host) = host_with("adm-nocap", vec![]);
        install_plugin(&sb, "aokie", &["flow.run"]);
        host.registry.lock().unwrap().scan();
        broker_for(&sb, &host);

        let (code, message) = host
            .handle_plugin_request("aokie", "companion.admission", serde_json::json!({}))
            .unwrap_err();
        assert_eq!(code, "capability_denied");
        assert!(message.contains("oaiy.companion.admission"), "{message}");
    }

    #[test]
    fn admission_is_refused_before_any_device_has_been_approved() {
        // An admission with an empty roster admits nobody, so asking the issuer
        // for one would spend a round trip to obtain a token that cannot carry
        // a call. Saying so plainly is what tells the user to pair a phone.
        let (sb, host) = host_with("adm-nopair", vec![]);
        install_plugin(&sb, "aokie", &["companion.admission"]);
        host.registry.lock().unwrap().scan();
        broker_for(&sb, &host);

        let (code, message) = host
            .handle_plugin_request("aokie", "companion.admission", serde_json::json!({}))
            .unwrap_err();
        assert_eq!(code, "not_paired");
        assert!(message.contains("approved"), "{message}");
    }

    /// An issuer as the relay's and FormLogic's are: it refuses a roster whose thumbprints are not strictly ascending
    /// or whose hash is not the protocol's for them, and otherwise echoes what it was asked to bind.
    fn strict_issuer() -> crate::link::testkit::Provider {
        use crate::link::testkit::{Provider, Reply};
        Provider::start(|req| {
            let body: Value = serde_json::from_str(&req.body).unwrap_or(Value::Null);
            let thumbprints: Vec<String> = body["approvedPeerKeyThumbprints"]
                .as_array()
                .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            if thumbprints.is_empty() || !thumbprints.windows(2).all(|w| w[0] < w[1]) {
                return Reply::status(400, r#"{"error":"approvedPeerKeyThumbprints must be strictly ascending"}"#);
            }
            let revision = body["peerRosterRevision"].as_u64().unwrap_or(0);
            if body["peerRosterHash"] != json!(crate::companion::peer_roster_hash(revision, &thumbprints)) {
                return Reply::status(400, r#"{"error":"peerRosterHash is not the hash of that roster"}"#);
            }
            let mut echoed = body.clone();
            echoed["accessToken"] = json!("admission-token");
            Reply::ok(&echoed.to_string())
        })
    }

    /// A host with `aokie` installed, a broker whose issuer is [`strict_issuer`], and the identity phones are approved on.
    fn admitting_host(tag: &str) -> (Sandbox, Arc<PluginHost>, crate::companion::EndpointIdentityHandle, crate::link::testkit::Provider) {
        let (sb, host) = host_with(tag, vec![]);
        install_plugin(&sb, "aokie", &["companion.admission"]);
        host.registry.lock().unwrap().scan();
        let companion = broker_for(&sb, &host);
        let issuer = strict_issuer();
        let broker = host.companion.lock().unwrap().clone().unwrap();
        broker
            .upstream
            .set(crate::companion::upstream::UpstreamConfig { base_url: issuer.base.clone(), token: "flk_issuer".into(), app_id: Some("app_1".into()) })
            .unwrap();
        let identity = companion.identity_for("aokie").unwrap();
        (sb, host, identity, issuer)
    }

    /// The roster the issuer was last asked to bind, as `(thumbprints, revision, hash)`.
    fn asked_to_bind(issuer: &crate::link::testkit::Provider) -> (Vec<String>, u64, String) {
        let last = issuer.requests().pop().expect("the issuer was asked");
        let body: Value = serde_json::from_str(&last.body).unwrap();
        let thumbprints = body["approvedPeerKeyThumbprints"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
        (thumbprints, body["peerRosterRevision"].as_u64().unwrap(), body["peerRosterHash"].as_str().unwrap().to_string())
    }

    #[test]
    fn two_phones_or_more_are_admitted_with_a_sorted_roster_however_they_were_approved() {
        // The host used to send the roster in the order phones were approved. The relay and FormLogic's issuer need
        // it strictly ascending and the plugin compares the echo with a sorted list, so with two phones or more the
        // admission failed unless the thumbprints happened to be in order (one time in n!).
        use crate::companion::identity::testing::phone_key;
        let (a, b) = (phone_key(1), phone_key(2));
        // The phone approved later has the smaller thumbprint: the roster is kept in the wrong order.
        let (first, second) = if a.thumbprint > b.thumbprint { (a, b) } else { (b, a) };
        let (_sb, host, identity, issuer) = admitting_host("adm-two");
        identity.approve_for_tests(&first);
        identity.approve_for_tests(&second);
        let kept: Vec<String> = identity.status().approved_mobiles.iter().map(|m| m.endpoint_key.thumbprint.clone()).collect();
        assert_eq!(kept, [first.thumbprint.clone(), second.thumbprint.clone()], "kept in the order approved, which is not sorted");

        let admission = host.handle_plugin_request("aokie", "companion.admission", json!({})).expect("the issuer accepts what it is sent");
        assert_eq!(admission["accessToken"], "admission-token");
        let (thumbprints, revision, hash) = asked_to_bind(&issuer);
        assert_eq!(thumbprints, [second.thumbprint.clone(), first.thumbprint.clone()], "sorted");
        assert_eq!(revision, 2);
        assert_eq!(hash, crate::companion::peer_roster_hash(2, &thumbprints));

        // Up to the relay's sixteen, in any order.
        for n in [3usize, 5, 16] {
            let (_sb, host, identity, issuer) = admitting_host("adm-many");
            for seed in (1..=n as u8).rev() {
                identity.approve_for_tests(&phone_key(seed));
            }
            let kept: Vec<String> = identity.status().approved_mobiles.iter().map(|m| m.endpoint_key.thumbprint.clone()).collect();
            assert_eq!(kept.len(), n);
            assert!(kept.windows(2).any(|w| w[0] > w[1]), "{n} phones: kept out of order");
            host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap_or_else(|e| panic!("{n} phones: {e:?}"));
            let (thumbprints, _, _) = asked_to_bind(&issuer);
            assert_eq!(thumbprints.len(), n);
            assert!(thumbprints.windows(2).all(|w| w[0] < w[1]), "{thumbprints:?}");
        }
    }

    /// What `start` records for a broker plugin at its `plugin.init`, for a test that has no process to start.
    fn plugin_initialised(host: &PluginHost, identity: &crate::companion::EndpointIdentityHandle) {
        let (_, roster) = identity.private_bootstrap_with_roster(1).expect("a roster to hand over");
        host.procs.lock().unwrap().rosters.insert("aokie".into(), roster);
    }

    #[test]
    fn a_running_plugin_is_presented_the_roster_it_was_handed_until_it_is_started_again() {
        use crate::companion::identity::testing::phone_key;
        let (a, b, c) = (phone_key(1), phone_key(2), phone_key(3));
        let (_sb, host, identity, issuer) = admitting_host("adm-held");
        let sorted = |keys: &[&crate::companion::EndpointPublicKey]| -> Vec<String> {
            let mut thumbprints: Vec<String> = keys.iter().map(|k| k.thumbprint.clone()).collect();
            thumbprints.sort();
            thumbprints
        };
        let admit = |what: &str| {
            host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            asked_to_bind(&issuer)
        };

        // Two phones, the plugin starts and is handed them.
        identity.approve_for_tests(&b);
        identity.approve_for_tests(&a);
        plugin_initialised(&host, &identity);
        let (handed, revision, hash) = admit("two phones");
        assert_eq!(handed, sorted(&[&a, &b]));
        assert_eq!(revision, 2);

        // A third is approved. The plugin does not know it, and goes on being admitted for the two it does: the
        // issuer is asked for the roster the plugin was handed, to the byte, and not for a roster of three that the
        // plugin would refuse.
        identity.approve_for_tests(&c);
        assert_eq!(identity.status().approved_mobiles.len(), 3);
        assert_eq!(admit("a phone approved after init"), (handed.clone(), revision, hash.clone()));

        // A phone is revoked. The roster the plugin holds names it, so that roster is not presented: the issuer is
        // asked for the roster as it is now (which the plugin refuses, and fails closed until it is started again),
        // never for one that admits the revoked phone.
        identity.revoke(&a.thumbprint).unwrap();
        let (now, now_revision, now_hash) = admit("a phone revoked after init");
        assert_eq!(now, sorted(&[&b, &c]));
        assert!(!now.contains(&a.thumbprint), "the revoked phone is not in what the issuer is asked to admit");
        assert_ne!((now_revision, now_hash), (revision, hash));

        // The plugin is started again: it is handed the roster as it is, and that is what it is presented, whatever
        // is approved or revoked after.
        plugin_initialised(&host, &identity);
        let (restarted, restarted_revision, restarted_hash) = admit("after a restart");
        assert_eq!(restarted, sorted(&[&b, &c]));
        let d = phone_key(5);
        identity.approve_for_tests(&d);
        assert_eq!(admit("a fourth phone approved after the restart"), (restarted, restarted_revision, restarted_hash));

        // Nobody approved is nobody admitted, whatever the plugin was handed before; and a plugin that was started with
        // nobody holds nothing, so what it is presented is the roster as it is.
        for k in [&b, &c, &d] {
            identity.revoke(&k.thumbprint).unwrap();
        }
        let (code, _) = host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap_err();
        assert_eq!(code, "not_paired", "an empty roster is still not paired");
        host.procs.lock().unwrap().rosters.remove("aokie");
        identity.approve_for_tests(&phone_key(6));
        assert_eq!(admit("one phone and a plugin that holds none").0.len(), 1, "no held roster: the roster as it is");
    }

    #[test]
    fn what_start_hands_a_plugin_at_init_is_the_roster_it_is_presented_with_no_process_to_start() {
        // `start` makes the bootstrap for the handshake with `init_bootstrap`, which also keeps the roster in it. This is
        // that step on its own, so a platform with no Node to run the stand-in plugin below checks it too.
        use crate::companion::identity::testing::phone_key;
        let (a, b, c, d) = (phone_key(1), phone_key(2), phone_key(3), phone_key(4));
        let (_sb, host, identity, issuer) = admitting_host("adm-init");
        let asked = || {
            host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap();
            asked_to_bind(&issuer)
        };
        let held = |host: &PluginHost| host.procs.lock().unwrap().rosters.get("aokie").cloned();

        // Nobody approved: no bootstrap, and nothing held.
        assert!(host.init_bootstrap("aokie", 1).is_none() && held(&host).is_none());

        identity.approve_for_tests(&b);
        identity.approve_for_tests(&a);
        let handed = host.init_bootstrap("aokie", 1).expect("a bootstrap once phones are approved");
        let roster = &handed["approvedMobileRoster"];
        let (thumbprints, revision, hash) = asked();
        assert_eq!((revision, hash.as_str()), (roster["revision"].as_u64().unwrap(), roster["rosterHash"].as_str().unwrap()), "what it was handed is what is presented");
        assert_eq!(held(&host).map(|r| r.thumbprints), Some(thumbprints.clone()));

        // A phone approved, and one revoked, since: the held roster stays while it can, and goes to the live one when it cannot.
        identity.approve_for_tests(&c);
        assert_eq!(asked(), (thumbprints.clone(), revision, hash.clone()));
        identity.revoke(&a.thumbprint).unwrap();
        assert!(!asked().0.contains(&a.thumbprint));

        // Handed again, it holds the roster as it is now.
        let again = host.init_bootstrap("aokie", 1).unwrap();
        assert_ne!(again["approvedMobileRoster"]["rosterHash"], roster["rosterHash"]);
        let (now, now_revision, _) = asked();
        assert_eq!(now_revision, again["approvedMobileRoster"]["revision"].as_u64().unwrap());
        identity.approve_for_tests(&d);
        assert_eq!(asked().0, now, "a phone approved after the second init is not in what it holds");

        // Started with nobody approved, a plugin holds nothing, and what an earlier run held is let go.
        for k in [&b, &c, &d] {
            identity.revoke(&k.thumbprint).unwrap();
        }
        assert!(host.init_bootstrap("aokie", 1).is_none());
        assert!(held(&host).is_none(), "the roster of the earlier run is not kept");
        // A plugin that is not a broker is handed nothing and holds nothing.
        assert!(host.init_bootstrap("some-other-plugin", 1).is_none());
        assert!(host.procs.lock().unwrap().rosters.get("some-other-plugin").is_none());
    }

    #[test]
    fn a_new_key_is_never_presented_a_roster_of_the_old_one() {
        // `rotate` mints a new key and clears the roster: every phone has to pair again, and a plugin that holds
        // the old key and the old phones must not have them admitted under it.
        use crate::companion::identity::testing::phone_key;
        let (_sb, host, identity, issuer) = admitting_host("adm-rotated");
        identity.approve_for_tests(&phone_key(1));
        plugin_initialised(&host, &identity);
        identity.rotate().unwrap();
        identity.approve_for_tests(&phone_key(1));
        host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap();
        let sent: Value = serde_json::from_str(&issuer.requests()[0].body).unwrap();
        let new_key = identity.status().endpoint_key.unwrap();
        assert_eq!(sent["endpointPublicKey"]["publicKey"], new_key.public_key.as_str(), "the key it is presented under is the key it has");
    }

    #[test]
    fn a_host_with_no_broker_says_so_instead_of_panicking() {
        // The host is also built by tests and tools that never wire companion
        // support; those must keep answering, honestly.
        let (sb, host) = host_with("adm-nobroker", vec![]);
        install_plugin(&sb, "aokie", &["companion.admission"]);
        host.registry.lock().unwrap().scan();

        let (code, _) = host
            .handle_plugin_request("aokie", "companion.admission", serde_json::json!({}))
            .unwrap_err();
        assert_eq!(code, "unavailable");
    }

    #[test]
    fn a_ring_request_is_refused_to_a_plugin_that_did_not_declare_the_capability() {
        let (sb, host) = host_with("ring-nocap", vec![]);
        install_plugin(&sb, "aokie", &["flow.run"]);
        host.registry.lock().unwrap().scan();
        for method in ["oaiy.ring.plan", "oaiy.ring.opened"] {
            let (code, message) = host.handle_plugin_request("aokie", method, serde_json::json!({"callId": "call_1"})).unwrap_err();
            assert_eq!(code, "capability_denied", "{method}");
            assert!(message.contains("oaiy.companion.admission"), "{message}");
        }
    }

    #[test]
    fn the_phone_plugins_shipped_manifest_names_the_capability_bare_and_is_answered_the_ring_requests_all_the_same() {
        // Aokie's manifest declares `companion.admission`, with no `oaiy.` in front (its manifest is in its own repository, on its own release
        // cycle). This host reads that as `oaiy.companion.admission` when it loads the manifest, so a plugin that names it the way Aokie does
        // is answered: were it refused `capability_denied`, the plugin would call every ring plan unavailable and nobody would ever be rung.
        let (sb, host) = host_with("ring-bare", vec![]);
        install_plugin(&sb, "aokie", &["companion.admission"]);
        host.registry.lock().unwrap().scan();
        host.set_ring(a_ring());
        let plan = host
            .handle_plugin_request("aokie", "oaiy.ring.plan", serde_json::json!({"callId": "call_1", "reason": "caller_asked", "recentCallerTurns": ["Can I speak to the owner please"]}))
            .expect("a plugin that declares the bare capability is answered");
        assert_eq!(plan["decision"], "ring", "{plan}");
        let (code, _) = host
            .handle_plugin_request("aokie", "oaiy.ring.opened", serde_json::json!({"planId": "plan_made_up", "requestId": "assist_1", "callId": "call_1", "callEpoch": 1, "ownerEpoch": 1, "expiresAt": 1_789_000_040u64}))
            .unwrap_err();
        assert_ne!(code, "capability_denied", "answered, and refused for what it is: a plan this desktop never made");
    }

    /// The desktop's record of one call, for the ring's questions.
    struct OneCall;

    impl crate::ring::CallSource for OneCall {
        fn facts(&self, call: &str) -> Option<crate::ring::CallInfo> {
            let asked = vec!["Can I speak to the owner please".to_string()];
            match call {
                "call_1" => Some(crate::ring::CallInfo { from: "+61491570006".into(), name: "Alex".into(), turns: asked, ..Default::default() }),
                "call_2" => Some(crate::ring::CallInfo { from: "+61491570156".into(), name: "Sam".into(), turns: asked, ..Default::default() }),
                "call_3" => Some(crate::ring::CallInfo { from: "+61491570157".into(), name: "Kim".into(), turns: vec!["There is a gas leak at the shop".to_string()], ..Default::default() }),
                _ => None,
            }
        }
        fn call_ended_by_phone(&self, _: &str) {}
        fn cancel_transfer(&self, _: &str, _: &str, _: crate::voice::transfer::CancelReason) -> tokio::sync::oneshot::Receiver<crate::ring::Withdrawal> {
            let (reply, answer) = tokio::sync::oneshot::channel();
            let _ = reply.send(crate::ring::Withdrawal::Sent);
            answer
        }
    }

    struct Here;

    impl crate::ring::PresenceSource for Here {
        fn presence(&self) -> crate::ring::Presence {
            crate::ring::Presence::Active
        }
    }

    fn a_ring() -> Arc<crate::ring::Ring> {
        let ring = crate::ring::Ring::in_memory(crate::ring::RingSettings { enabled: true, ..Default::default() });
        ring.set_calls(Arc::new(OneCall));
        ring.set_presence(Arc::new(Here));
        ring.set_devices(crate::ring::testing::at_the_pc());
        ring
    }

    #[test]
    fn the_plugin_asks_who_may_be_rung_and_says_the_request_is_out() {
        let ring = a_ring();
        let asked = |ring: &Arc<crate::ring::Ring>, call: &str| PluginHost::ring_request(ring, "oaiy.ring.plan", json!({"callId": call, "reason": "caller_asked", "recentCallerTurns": ["Can I speak to the owner please"]}));
        let plan = asked(&ring, "call_1").unwrap();
        assert_eq!((plan["decision"].as_str(), plan["reason"].as_str(), plan["desktopToast"].as_bool()), (Some("ring"), Some("ok"), Some(true)), "{plan}");
        assert_eq!(plan["reasonAllowed"], json!(false), "a caller who asked for a person needs no vouching");
        let plan_id = plan["planId"].as_str().unwrap().to_string();
        assert!(!plan_id.is_empty());

        // The request is out: the desktop rings, for that plan and that call only.
        let opened = |plan: &str, call: &str| PluginHost::ring_request(&ring, "oaiy.ring.opened", json!({"planId": plan, "requestId": "assist_1", "callId": call, "callEpoch": 1, "ownerEpoch": 1, "expiresAt": ring.clock().unix() + 25}));
        assert_eq!(opened("plan_made_up", "call_1").unwrap_err().0, "unknown_plan");
        assert_eq!(opened(&plan_id, "call_2").unwrap_err().0, "unknown_plan");
        assert!(ring.active().is_empty(), "nothing rang for either");
        assert_eq!(opened(&plan_id, "call_1").unwrap(), json!({"ok": true}));
        assert_eq!(ring.active().len(), 1);
        assert_eq!(ring.active()[0].caller_number, "+61491570006");

        // A second question straight away is a second try, and the gap between tries refuses it.
        let again = asked(&ring, "call_1").unwrap();
        assert_eq!((again["decision"].as_str(), again["reason"].as_str()), (Some("refused"), Some("limit_gap")), "{again}");
        // A refusal has a plan id like any plan (the plugin loses the real reason without one), but nothing can be opened with it.
        let refused_id = crate::ring::testing::plan_id_of(&again);
        assert_ne!(refused_id, plan_id);
        assert_eq!(opened(&refused_id, "call_1").unwrap_err().0, "unknown_plan");
    }

    #[test]
    fn an_urgent_plan_says_this_desktop_vouches_for_the_reason_only_when_it_heard_the_owners_phrase() {
        let ring = a_ring();
        let ask = |call: &str, reason: &str| PluginHost::ring_request(&ring, "oaiy.ring.plan", json!({"callId": call, "reason": reason, "recentCallerTurns": []})).unwrap();
        // Not allowed by the owner: nothing rings, nothing is vouched.
        let plan = ask("call_3", "urgent");
        assert_eq!((plan["decision"].as_str(), plan["reason"].as_str(), plan["reasonAllowed"].as_bool()), (Some("message_only"), Some("initiative_off"), Some(false)), "{plan}");
        // Allowed, with the owner's own phrase: the plan rings and says so.
        ring.change_settings(&json!({"initiative": "on_request_or_urgent", "urgentPhrases": ["gas leak"]})).unwrap();
        let plan = ask("call_3", "urgent");
        assert_eq!((plan["decision"].as_str(), plan["reasonAllowed"].as_bool()), (Some("ring"), Some(true)), "{plan}");
        // A call that did not say it is not urgent, whatever the plugin says of it in the request.
        let plan = PluginHost::ring_request(&ring, "oaiy.ring.plan", json!({"callId": "call_1", "reason": "urgent", "recentCallerTurns": ["There is a gas leak"]})).unwrap();
        assert_eq!((plan["decision"].as_str(), plan["reason"].as_str(), plan["reasonAllowed"].as_bool()), (Some("message_only"), Some("not_urgent"), Some(false)), "{plan}");
    }

    #[test]
    fn a_ring_request_the_desktop_cannot_make_sense_of_is_refused_and_rings_nothing() {
        let ring = a_ring();
        let code = |method: &str, params: Value| PluginHost::ring_request(&ring, method, params).unwrap_err().0;
        assert_eq!(code("oaiy.ring.plan", json!({})), "invalid_request", "no call");
        assert_eq!(code("oaiy.ring.plan", json!({"callId": "call_1", "reason": "because"})), "invalid_request");
        assert_eq!(code("oaiy.ring.opened", json!({"planId": "p", "requestId": "r", "callId": "call_1"})), "invalid_request", "no epochs or time");
        assert_eq!(code("oaiy.ring.opened", json!({"planId": "", "requestId": "r", "callId": "call_1", "callEpoch": 1, "ownerEpoch": 1, "expiresAt": 1})), "invalid_request");
        // A call this desktop never heard (the plugin's own): its word for what the caller said is used, and is checked
        // the same way, so a request the caller did not make rings nobody.
        let plan = |turns: Value| PluginHost::ring_request(&ring, "oaiy.ring.plan", json!({"callId": "call_9", "reason": "caller_asked", "callerNumber": "+61491570156", "recentCallerTurns": turns})).unwrap();
        let not_asked = plan(json!(["What are your opening hours?"]));
        assert_eq!((not_asked["decision"].as_str(), not_asked["reason"].as_str()), (Some("refused"), Some("caller_did_not_ask")), "{not_asked}");
        crate::ring::testing::plan_id_of(&not_asked);
        assert_eq!(plan(json!(["Put me through to the owner"]))["decision"], "ring");
    }

    #[test]
    fn an_unknown_method_names_both_methods_the_host_answers() {
        // The message is a plugin author's only clue when they misspell one.
        let (_sb, host) = host_with("adm-unknown", vec![]);
        let (code, message) = host
            .handle_plugin_request("aokie", "companion.admit", serde_json::json!({}))
            .unwrap_err();
        assert_eq!(code, "invalid_request");
        assert!(message.contains("flow.run"), "{message}");
        assert!(message.contains("companion.admission"), "{message}");
    }

    #[test]
    fn an_event_whose_binding_cannot_evaluate_is_dead_lettered() {
        // `====` is not JavaScript, so ZIPP refuses to compile it and the
        // binding is refused rather than guessed. The author meant this to fire;
        // without a dead letter they would never learn it didn't.
        let refusing = Arc::new(crate::bridge::conditions::testing::FakeHost::by_source(|_| {
            Err(crate::bridge::conditions::testing::guest("SyntaxError: unexpected token '='"))
        }));
        let (_sb, host) = host_evaluating("broken", vec![binding("b1", Some("$event.data.from ==== 1"))], refusing);
        host.process_event("aokie", envelope());

        let dead = host.dead.lock().unwrap().list(10);
        assert_eq!(dead.len(), 1, "an unevaluatable condition must dead-letter");
        assert_eq!(dead[0].event, "aokie.call.incoming");
        assert_eq!(dead[0].source, "aokie");
        assert!(dead[0].reason.message().contains("could not be evaluated"));
        // The envelope has to survive intact or redrive cannot reconstruct it.
        assert_eq!(dead[0].envelope["data"]["from"], "+61400000000");
    }

    #[test]
    fn an_event_that_fires_normally_leaves_the_queue_empty() {
        let (_sb, host) = host_with("fine", vec![binding("b1", None)]);
        host.process_event("aokie", envelope());

        assert_eq!(host.ledger.lock().unwrap().len(), 1, "the binding should have fired");
        // A queue that fills up with successes is a queue nobody reads.
        assert!(host.dead.lock().unwrap().is_empty());
    }

    #[test]
    fn a_disabled_binding_does_not_dead_letter() {
        let mut b = binding("b1", None);
        b.enabled = false;
        let (_sb, host) = host_with("disabled", vec![b]);
        host.process_event("aokie", envelope());
        assert!(host.dead.lock().unwrap().is_empty());
    }

    #[test]
    fn redrive_reserves_a_run_once_the_binding_is_fixed_and_clears_the_entry() {
        let refusing = Arc::new(crate::bridge::conditions::testing::FakeHost::by_source(|_| {
            Err(crate::bridge::conditions::testing::guest("SyntaxError: unexpected token '='"))
        }));
        let (_sb, host) = host_evaluating("redrive", vec![binding("b1", Some("$event.data.from ==== 1"))], refusing);
        host.process_event("aokie", envelope());
        let id = host.dead.lock().unwrap().list(1)[0].id.clone();

        // Redriving while it is still broken must NOT claim success, and must
        // leave the entry in place with the attempt recorded.
        let (_, reserved) = host.redrive(&id).expect("the entry exists");
        assert!(!reserved);
        let still = host.dead.lock().unwrap().get(&id).expect("still queued");
        assert_eq!(still.attempts, 1);
        assert!(still.last_outcome.is_some());
        assert_eq!(host.ledger.lock().unwrap().len(), 0, "nothing should have run");

        // Fix the condition, redrive again: now it fires.
        host.triggers.lock().unwrap().upsert(binding("b1", None)).unwrap();
        let (outcomes, reserved) = host.redrive(&id).expect("the entry exists");
        assert!(reserved, "{outcomes:?}");
        assert_eq!(host.ledger.lock().unwrap().len(), 1);
        // Resolved entries go, or someone redrives the same event every morning.
        assert!(host.dead.lock().unwrap().get(&id).is_none());
    }

    #[test]
    fn redrive_reads_the_dispatchers_fact_not_its_prose() {
        // Regression: redrive decided success by grepping the human-readable
        // outcome for ": reserved ". Reword that message — a prefix, different
        // punctuation, a translation — and redrive stops recognising success:
        // it keeps the entry forever, counts endless attempts, and fires a NEW
        // run on every retry because each gets a fresh idempotency key.
        //
        // So assert the two travel together. If `reserved` is ever re-derived
        // from `outcomes` again, a wording change breaks this test instead of
        // quietly duplicating runs in production.
        let (_sb, host) = host_with("prose", vec![binding("b1", None)]);
        let event = Event {
            name: "aokie.call.incoming".into(),
            source: "aokie".into(),
            correlation_id: "c".into(),
            idempotency_key: "k-prose".into(),
            data: json!({ "from": "+61400000000" }),
            origin_run: None,
        };

        let d = host.dispatch_event(&event);
        assert!(d.reserved, "the binding should have fired");
        assert_eq!(host.ledger.lock().unwrap().len(), 1);

        // Nothing reserved the second time: same key, so the ledger dedupes it.
        let again = host.dispatch_event(&event);
        assert!(!again.reserved, "a duplicate reserves nothing");
        assert_eq!(host.ledger.lock().unwrap().len(), 1, "and creates no second run");
    }

    #[test]
    fn a_redriven_event_does_not_collide_with_its_own_original_key() {
        // The original reservation never happened (that is why it is a dead
        // letter), but the key must still differ or a later legitimate event
        // carrying the same key would be swallowed as a duplicate of the redrive.
        let (_sb, host) = host_with("keys", vec![binding("b1", Some("$event.data.from ==== 1"))]);
        host.process_event("aokie", envelope());
        let id = host.dead.lock().unwrap().list(1)[0].id.clone();

        host.triggers.lock().unwrap().upsert(binding("b1", None)).unwrap();
        host.redrive(&id).expect("the entry exists");

        let run = &host.ledger.lock().unwrap().recent(1, &[])[0];
        assert_ne!(run.idempotency_key, "evt-1");
        assert!(run.idempotency_key.contains("evt-1"), "{}", run.idempotency_key);
    }

    #[test]
    fn a_host_starts_its_autostart_plugins_without_anyone_clicking_start() {
        // `autostart_ids()` existed, was tested, and had NO callers — so a
        // plugin only ran if a human opened the app and pressed Start. For
        // something like a phone bridge that means it quietly answers nothing
        // until someone remembers it exists.
        //
        // The manifest here is valid but its entry command is a stub that
        // cannot execute, so the start ATTEMPT is what we observe: the record
        // leaves `Installed` on its own. If the `spawn_autostart()` call is ever
        // removed, this stays `Installed` forever and the test fails.
        let sb = Sandbox::new("autostart");
        let plugins = sb.0.join("plugins");
        let dir = plugins.join("probe");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string(&json!({
                "schemaVersion": 3,
                "id": "probe",
                "name": "probe plugin",
                "version": "0.1.0",
                "pluginApiVersion": 1,
                "entry": { "kind": "process", "command": "plugin.exe" },
                "capabilities": ["flow.run"],
                "connectors": [],
                "events": []
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("plugin.exe"), b"not a real executable").unwrap();

        let host = PluginHost::new(
            crate::plugins::registry::new_handle(plugins),
            crate::bridge::ledger::new_handle(),
            Arc::new(Mutex::new(TriggerStore::load(sb.0.join("triggers.json")))),
            crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );

        // Autostart runs on its own thread so boot is not blocked by it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let moved = loop {
            let state = host
                .registry
                .lock()
                .ok()
                .and_then(|r| r.get("probe").map(|rec| rec.state));
            if matches!(state, Some(s) if s != PluginState::Installed) {
                break true;
            }
            if std::time::Instant::now() > deadline {
                break false;
            }
            thread::sleep(std::time::Duration::from_millis(100));
        };
        assert!(moved, "the host never attempted to start the plugin");
    }

    #[test]
    fn a_dead_letter_from_a_previous_run_is_still_redrivable() {
        // The queue is durable precisely so an event shed overnight can be
        // actioned in the morning — by a process that never saw it arrive.
        let sb = Sandbox::new("durable");
        let dl_path = sb.0.join("deadletters.jsonl");
        let id = {
            let mut q = DeadLetterQueue::open(dl_path.clone());
            q.record(
                "aokie",
                "aokie.call.incoming",
                crate::bridge::DeadReason::Shed,
                envelope(),
            )
            .id
        };

        let triggers: TriggerStoreHandle =
            Arc::new(Mutex::new(TriggerStore::load(sb.0.join("triggers.json"))));
        triggers.lock().unwrap().upsert(binding("b1", None)).unwrap();
        let host = PluginHost::new(
            crate::plugins::registry::new_handle(sb.0.join("plugins")),
            crate::bridge::ledger::new_handle(),
            triggers,
            crate::bridge::deadletters::open_handle(dl_path),
            "0.0.0-test".into(),
            true,
        );

        let (outcomes, reserved) = host.redrive(&id).expect("loaded from disk");
        assert!(reserved, "{outcomes:?}");
        assert_eq!(host.ledger.lock().unwrap().len(), 1);
    }

    // --- loading bindings: partial is not the same as none ------------------

    #[test]
    fn one_unloadable_binding_does_not_take_the_others_with_it() {
        // The whole-file parse turned a single bad row into ZERO bindings: every
        // automation stopped firing, the API returned the same empty list an
        // untriggered workspace returns, and the next upsert rewrote the file
        // from that empty Vec — the good bindings gone for good.
        let sb = Sandbox::new("partial");
        let path = sb.0.join("triggers.json");
        let first = serde_json::to_value(binding("keep-me", None)).unwrap();
        let last = serde_json::to_value(binding("keep-me-too", None)).unwrap();
        std::fs::write(
            &path,
            // The middle row is what a hand edit — or a field a newer build
            // made required — looks like arriving here: valid JSON, not a
            // binding.
            serde_json::to_string(&json!([first, { "id": "half-written" }, last])).unwrap(),
        )
        .unwrap();

        let store = TriggerStore::load(path);
        let ids: Vec<&str> = store.list().iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids, ["keep-me", "keep-me-too"]);
    }

    #[test]
    fn a_bindings_file_that_will_not_load_is_kept_rather_than_overwritten() {
        let sb = Sandbox::new("corrupt");
        let path = sb.0.join("triggers.json");
        // A torn write: the array never closes, so nothing is recoverable
        // entry by entry.
        std::fs::write(&path, r#"[{"id":"b1","event":"a","flowId":"f","mode":"async""#).unwrap();

        let mut store = TriggerStore::load(path.clone());
        assert!(store.list().is_empty(), "an unparseable file loads nothing");
        // Starting empty is survivable. Silently overwriting the user's only
        // copy on the next edit is not — persist() writes the whole file.
        store.upsert(binding("new", None)).unwrap();
        let kept = std::fs::read_to_string(sb.0.join("triggers.json.corrupt"))
            .expect("the original must still exist somewhere");
        assert!(kept.contains("b1"), "{kept}");

        let reloaded = TriggerStore::load(path);
        assert_eq!(reloaded.list().len(), 1);
        assert_eq!(reloaded.list()[0].id, "new");
    }

    // --- the reader thread must never pay for a shed ------------------------

    #[test]
    fn recording_a_shed_does_not_block_the_thread_that_dropped_the_event() {
        // The sink runs on the plugin's reader thread, which also routes every
        // RPC reply. Recording used to be inline there: a whole-queue rewrite
        // plus an fsync per dropped event, in front of the health and connector
        // replies the same thread still had to deliver.
        let (_sb, host) = host_with("shed", vec![]);
        let held = host.dead.lock().unwrap();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let handoff = host.clone();
        thread::spawn(move || {
            handoff.note_shed("aokie", envelope());
            let _ = done_tx.send(());
        });
        assert!(
            done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
            "the shed handoff waited on the dead-letter queue"
        );
        drop(held);

        // And it is still recorded — durably, which is the point of the feature.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let q = host.dead.lock().unwrap();
            if let Some(item) = q.list(1).first() {
                assert_eq!(item.reason, DeadReason::Shed);
                // Intact, or the entry cannot be redriven.
                assert_eq!(item.envelope["data"]["from"], "+61400000000");
                break;
            }
            drop(q);
            assert!(Instant::now() < deadline, "the shed was never recorded");
            thread::sleep(Duration::from_millis(20));
        }
    }

    // --- shutdown ------------------------------------------------------------

    #[test]
    fn nothing_starts_once_the_host_is_shutting_down() {
        // stop_all runs on RunEvent::Exit — the last thing before the process
        // goes away. A child spawned after it has nobody left to stop it and
        // survives as an orphan holding whatever hardware it opened; the
        // autostart loop's next iteration is exactly that case.
        let (_sb, host) = host_with("shutdown", vec![]);
        host.stop_all();
        let err = host.start("aokie").unwrap_err();
        assert!(err.contains("shutting down"), "{err}");
    }

    #[test]
    fn a_host_stopped_for_an_update_that_did_not_install_can_start_plugins_again() {
        // An update stops everything, and if the installer cannot start, the app carries on: its plugins
        // (Aokie holds the phone dongle) have to be startable again, which stop_all's "shutting down" forbids.
        let (_sb, host) = host_with("resume", vec![]);
        assert!(host.running_ids().is_empty());
        host.stop_all();
        assert!(host.start("aokie").unwrap_err().contains("shutting down"));
        host.resume();
        let err = host.start("aokie").unwrap_err();
        assert!(!err.contains("shutting down"), "started plugins are refused no longer: {err}");
    }

    // --- the restart bound ---------------------------------------------------

    use crate::plugins::registry::PluginRecord;

    /// A registry record for `id` with `attempts` crash-restarts already spent.
    ///
    /// Inserted directly rather than scanned off disk: the bound is arithmetic
    /// over one field, and a manifest would only add a spawn that cannot
    /// succeed anyway.
    fn record_with_restarts(host: &PluginHost, id: &str, attempts: u32) {
        host.registry.lock().unwrap().insert(PluginRecord {
            id: id.into(),
            state: PluginState::Running,
            reason: None,
            dir: PathBuf::from("."),
            manifest: None,
            legacy_capabilities: Vec::new(),
            unknown_capabilities: Vec::new(),
            user_disabled: false,
            restart_attempts: attempts,
            last_health: None,
            last_health_at: None,
            last_health_error: None,
            trust: None,
        });
    }

    #[test]
    fn the_restart_budget_is_refilled_only_once_a_plugin_has_stayed_up() {
        // Refilling at handshake time made MAX_RESTART_ATTEMPTS unreachable: a
        // plugin that answers plugin.init and then dies — Aokie answers before
        // it touches the radio, so an unplugged dongle looks exactly like this —
        // zeroed its own counter every cycle, so should_restart was always true,
        // the backoff never left 1s, and the terminal "start it manually" state
        // was dead code. One child process every ~11 seconds, forever.
        let (_sb, host) = host_with("stable", vec![]);
        record_with_restarts(&host, "aokie", 3);
        let attempts = |host: &Arc<PluginHost>| {
            host.registry.lock().unwrap().get("aokie").unwrap().restart_attempts
        };

        host.procs
            .lock()
            .unwrap()
            .started_at
            .insert("aokie".into(), Instant::now());
        assert!(!host.refill_restart_budget_if_stable("aokie"));
        assert_eq!(attempts(&host), 3, "coming up is not staying up");

        // A machine that booted seconds ago has no Instant this far back.
        let Some(long_ago) = Instant::now().checked_sub(STABLE_UPTIME + Duration::from_secs(1))
        else {
            return;
        };
        host.procs
            .lock()
            .unwrap()
            .started_at
            .insert("aokie".into(), long_ago);
        assert!(host.refill_restart_budget_if_stable("aokie"));
        assert_eq!(attempts(&host), 0, "it stayed up, so it earns a fresh budget");
        // Once per life, or every later tick takes the registry lock for nothing.
        assert!(!host.refill_restart_budget_if_stable("aokie"));
    }

    // --- logs outlive the process --------------------------------------------

    #[test]
    fn a_crashed_plugins_logs_are_still_readable() {
        // The state one supervisor tick after a crash: the process is out of
        // `running` and its stderr is the only evidence of why it died. Reading
        // `running` meant the Logs button — which the panel offers in every
        // state — said "No output yet." precisely when there was output.
        let (_sb, host) = host_with("logs", vec![]);
        let ring = LogBuffer::new();
        ring.push("stderr", "no dongle at COM3".into());
        host.procs
            .lock()
            .unwrap()
            .log_rings
            .insert("aokie".into(), ring);

        let lines = host.logs("aokie", None).expect("a crashed plugin still has logs");
        assert!(lines.iter().any(|l| l.text.contains("no dongle")), "{lines:?}");

        // The load-bearing half: NOTHING may drop the ring on the way out. A
        // future `log_rings.remove(id)` in stop() or the supervisor's exit path
        // would restore the exact bug — the Logs button saying "No output yet."
        // at the one moment there is output worth reading — while the assertion
        // above still passed.
        let _ = host.stop("aokie");
        assert!(
            host.logs("aokie", None).is_some_and(|l| l.iter().any(|x| x.text.contains("no dongle"))),
            "stop() must not discard the retained ring"
        );
        host.stop_all();
        assert!(
            host.logs("aokie", None).is_some_and(|l| l.iter().any(|x| x.text.contains("no dongle"))),
            "shutdown must not discard the retained ring either"
        );

        // A plugin that never ran has nothing, which is a different answer.
        assert!(host.logs("never-spawned", None).is_none());
    }
    // --- conditions: the verdict boundary and the `$event` migration -------

    use crate::bridge::conditions::testing::{guest, FakeHost};

    /// `host_with`, plus the evaluator that decides its conditions.
    fn host_evaluating(
        tag: &str,
        bindings: Vec<TriggerBinding>,
        evaluator: Arc<dyn crate::bridge::script_host::ScriptBatch>,
    ) -> (Sandbox, Arc<PluginHost>) {
        let (sb, host) = host_with(tag, bindings);
        host.set_script_evaluator(evaluator);
        (sb, host)
    }

    #[test]
    fn conditions_are_decided_before_the_ledger_is_locked() {
        // The property this whole restructure exists for. `dispatch` runs under
        // the ledger lock; the script host takes a process-wide lock of its own
        // and may spawn a child. Evaluating from inside the ledger lock would
        // put every plugin event in the process — and every HTTP reserve, claim
        // and finalise — behind an engine that is starting.
        //
        // The proof: an evaluator that takes the ledger lock itself. Outside the
        // lock it succeeds; inside it, a std::sync::Mutex deadlocks. Driven on
        // its own thread with a deadline, so a regression FAILS rather than
        // hanging the suite forever.
        let (_sb, host) = host_with("boundary", vec![binding("b1", Some("event.data.from"))]);
        let ledger = host.ledger.clone();
        host.set_script_evaluator(Arc::new(FakeHost::raw(move |_| {
            // If this runs under the ledger lock, this line never returns.
            let held = ledger.lock().is_ok();
            assert!(held, "the ledger was not available to the evaluator");
            Ok(crate::bridge::script_host::ScriptResponse {
                engine: serde_json::json!({}),
                results: vec![crate::bridge::script_host::JobResult {
                    id: "c0".into(),
                    outcome: Ok(serde_json::json!(true)),
                }],
            })
        })));

        let (tx, rx) = std::sync::mpsc::channel();
        let worker = Arc::clone(&host);
        std::thread::spawn(move || {
            worker.process_event("aokie", envelope());
            let _ = tx.send(());
        });
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("the dispatch deadlocked: conditions are being evaluated under the ledger lock");
        assert_eq!(host.ledger.lock().unwrap().len(), 1, "and the binding still fired");
    }

    #[test]
    fn an_event_whose_bindings_carry_no_conditions_never_reaches_the_engine() {
        // Load-bearing: the script host is spawned lazily, on the first job. A
        // workspace whose triggers have no conditions must never pay for a
        // child process — and most of them do not.
        let fake = Arc::new(FakeHost::always(serde_json::json!(true)));
        let (_sb, host) = host_evaluating("noengine", vec![binding("b1", None)], fake.clone());
        host.process_event("aokie", envelope());
        assert_eq!(host.ledger.lock().unwrap().len(), 1, "the binding fired");
        assert_eq!(fake.calls(), 0, "and nothing asked the engine anything");
    }

    #[test]
    fn zipp_decides_the_dispatch_not_the_rust_grammar() {
        // `$event.data.from` is a condition the Rust grammar reads as TRUE for
        // this envelope. ZIPP says false, and ZIPP is what dispatch obeys.
        let fake = Arc::new(FakeHost::always(serde_json::json!(false)));
        let (_sb, host) = host_evaluating("zippwins", vec![binding("b1", Some("$event.data.from"))], fake);
        host.process_event("aokie", envelope());
        assert_eq!(host.ledger.lock().unwrap().len(), 0, "ZIPP said no");
    }

    // --- the `$event` migration, on the store ------------------------------

    fn stored(path: &std::path::Path, text: &str) {
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn loading_rewrites_a_bare_event_once_and_keeps_the_original_beside_it() {
        let sb = Sandbox::new("migrate");
        let path = sb.0.join("triggers.json");
        let original = r#"[{"id":"b1","event":"e","flowId":"f","mode":"async","condition":"event && event.data.x === 1"}]"#;
        stored(&path, original);

        let store = TriggerStore::load(path.clone());
        assert_eq!(
            store.list()[0].condition.as_deref(),
            Some("event.data && event.data.x === 1"),
            "the stored text is rewritten once, and the old reading survives nowhere"
        );

        // The original is beside the file, byte for byte, and it round-trips:
        // restoring the `.bak` gives back exactly what the user wrote.
        let bak = path.with_extension("json.bak");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), original);
        std::fs::copy(&bak, &path).unwrap();
        let restored = std::fs::read_to_string(&path).unwrap();
        assert_eq!(restored, original);
        let reparsed: Vec<TriggerBinding> = serde_json::from_str(&restored).unwrap();
        assert_eq!(reparsed[0].condition.as_deref(), Some("event && event.data.x === 1"));

        // And the rewritten set is what is on disk now (re-copy it back first).
        std::fs::write(&path, serde_json::to_string(store.list()).unwrap()).unwrap();
        let again = TriggerStore::load(path.clone());
        assert_eq!(again.list()[0].condition.as_deref(), Some("event.data && event.data.x === 1"));
    }

    #[test]
    fn migrating_twice_changes_nothing_the_second_time() {
        let sb = Sandbox::new("migrate-idem");
        let path = sb.0.join("triggers.json");
        stored(&path, r#"[{"id":"b1","event":"e","flowId":"f","mode":"async","condition":"$event"}]"#);

        let first = TriggerStore::load(path.clone());
        assert_eq!(first.list()[0].condition.as_deref(), Some("$event.data"));
        let after_first = std::fs::read_to_string(&path).unwrap();
        drop(first);

        // A second boot must be a no-op: no further `.data`, and not one byte
        // of the file moved.
        let second = TriggerStore::load(path.clone());
        assert_eq!(second.list()[0].condition.as_deref(), Some("$event.data"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), after_first);
    }

    #[test]
    fn a_file_already_in_the_new_form_is_left_byte_for_byte() {
        // Deliberately NOT the formatting `persist` writes: compact, odd
        // spacing, a trailing newline. If anything rewrites the file, the bytes
        // move even though the meaning does not.
        let sb = Sandbox::new("migrate-noop");
        let path = sb.0.join("triggers.json");
        let text = "[ {\"id\":\"b1\",\"event\":\"e\",\"flowId\":\"f\",\"mode\":\"async\",\"condition\":\"$event.data.x === 1\"} ]\n";
        stored(&path, text);

        let store = TriggerStore::load(path.clone());
        assert_eq!(store.list().len(), 1);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "nothing changed, so nothing was written");
        assert!(
            !path.with_extension("json.bak").exists(),
            "and no backup was made for a migration that did not happen"
        );
    }

    #[test]
    fn a_rewrite_that_cannot_be_backed_up_is_not_written_at_all() {
        // The original must survive a failed migration. A directory in the way
        // of the backup's temp file makes the write fail on every platform.
        let sb = Sandbox::new("migrate-readonly");
        let path = sb.0.join("triggers.json");
        let original = r#"[{"id":"b1","event":"e","flowId":"f","mode":"async","condition":"$event"}]"#;
        stored(&path, original);
        std::fs::create_dir_all(path.with_extension("json.bak.tmp")).unwrap();

        let mut store = TriggerStore::load(path.clone());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "the user's file is untouched — not half-written, not emptied"
        );
        assert_eq!(
            store.list()[0].condition.as_deref(),
            Some("$event.data"),
            "the safe reading is in force for this run; the migration is idempotent, so the next boot retries"
        );

        // And a later write must not quietly destroy the original either: it
        // retries the backup first, and refuses rather than overwrite.
        let err = store.upsert(binding("b2", None)).unwrap_err();
        assert!(err.contains("copied aside"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);

        // Clear the obstruction and the same write succeeds, backup and all.
        std::fs::remove_dir_all(path.with_extension("json.bak.tmp")).unwrap();
        store.upsert(binding("b3", None)).unwrap();
        assert_eq!(std::fs::read_to_string(path.with_extension("json.bak")).unwrap(), original);
    }

    #[test]
    fn every_rewrite_is_logged_with_the_text_before_and_after() {
        // The only record of an edit to a file the user owns. It must be enough
        // to reconstruct the change without the `.bak` — and greppable, with the
        // free text quoted so a condition containing a newline stays one line.
        let line = migration_line(
            std::path::Path::new("C:/data/triggers.json"),
            &crate::bridge::triggers::ConditionRewrite {
                binding_id: "b1".into(),
                before: "event && event.data.x === 'a\nb'".into(),
                after: "event.data && event.data.x === 'a\nb'".into(),
            },
        );
        assert_eq!(
            line,
            r#"trigger-condition-migrated: file="C:/data/triggers.json" binding="b1" before="event && event.data.x === 'a\nb'" after="event.data && event.data.x === 'a\nb'""#
        );
        assert!(!line.contains('\n'), "one rewrite is one line: {line}");
    }

    #[test]
    fn an_input_map_selector_is_not_migrated() {
        // `$event` in an inputMap is resolved in Rust and still means the data.
        let sb = Sandbox::new("migrate-inputmap");
        let path = sb.0.join("triggers.json");
        stored(
            &path,
            r#"[{"id":"b1","event":"e","flowId":"f","mode":"async","inputMap":{"all":"$event"}}]"#,
        );
        let store = TriggerStore::load(path.clone());
        assert_eq!(store.list()[0].input_map["all"], "$event");
        assert!(!path.with_extension("json.bak").exists(), "nothing to migrate");
    }

    #[test]
    fn an_unparseable_condition_can_still_be_saved_and_still_cannot_fire() {
        // `upsert` no longer runs a grammar: only the engine can say whether
        // JavaScript parses, and the route asks it before this lock is taken.
        // A binding that reaches the store unchecked must still be inert.
        let (_sb, host) = host_with("upsert-unchecked", vec![]);
        host.triggers
            .lock()
            .unwrap()
            .upsert(binding("b1", Some("$event.data.from ==== 1")))
            .expect("the store saves what the route already checked");

        let fake = Arc::new(FakeHost::by_source(|_| Err(guest("SyntaxError: unexpected token '='"))));
        host.set_script_evaluator(fake);
        host.process_event("aokie", envelope());
        assert_eq!(host.ledger.lock().unwrap().len(), 0, "a condition ZIPP refuses never fires");
        assert_eq!(host.dead.lock().unwrap().list(10).len(), 1, "and it is dead-lettered");
    }

    // --- events for the linked account, with FormLogic away and back ----------

    /// A FormLogic on `port` with one flow trigger on `aokie.call.ended`: every
    /// run it is asked to reserve lands in the returned list.
    fn formlogic_on(port: u16) -> Arc<Mutex<Vec<Value>>> {
        use std::io::{Read, Write};
        let reserved = Arc::new(Mutex::new(Vec::new()));
        let seen = reserved.clone();
        let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("the port is free again");
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                let (head, body) = loop {
                    let Ok(n) = stream.read(&mut buf) else { break (String::new(), String::new()) };
                    if n == 0 {
                        break (String::new(), String::new());
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len: usize = text[..end].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length: ").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
                        if raw.len() >= end + 4 + len {
                            break (text[..end].to_string(), String::from_utf8_lossy(&raw[end + 4..end + 4 + len]).to_string());
                        }
                    }
                };
                let line = head.lines().next().unwrap_or("").to_string();
                let reply = if line.starts_with("GET /api/v1/flow-bindings") {
                    json!({"bindings": [{"id": "b1", "event": "aokie.call.ended", "flow": "call-summary", "enabled": true}]})
                } else if line.starts_with("POST /api/v1/flow-runs") {
                    seen.lock().unwrap().push(serde_json::from_str::<Value>(&body).unwrap_or(Value::Null));
                    json!({"created": true, "run": {"runId": "run-1"}})
                } else if line.starts_with("GET /api/v1/app-logic") {
                    json!({"apps": []})
                } else {
                    json!({})
                };
                let text = reply.to_string();
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
            }
        });
        reserved
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting until {what}");
            thread::sleep(Duration::from_millis(100));
        }
    }

    #[test]
    fn an_event_while_formlogic_is_away_is_kept_and_reaches_its_flow_when_it_is_back() {
        let (sb, host) = host_with("outbox", vec![]);
        // A port FormLogic is not on yet.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        std::fs::create_dir_all(sb.0.join("link")).unwrap();
        std::fs::write(
            sb.0.join("link").join("account.json"),
            json!({"connectorId": "formlogic", "baseUrl": format!("http://127.0.0.1:{port}"), "credential": "flk_test", "accountId": "conn_1", "linkedAt": "2026-09-29T00:00:00Z"}).to_string(),
        )
        .unwrap();
        host.set_link(crate::link::open_handle(sb.0.clone()));

        let started = Instant::now();
        host.process_event("aokie", json!({"name": "aokie.call.ended", "idempotencyKey": "evt-9", "correlationId": "call_9", "data": {"callId": "call_9"}}));
        assert!(started.elapsed() < Duration::from_secs(1), "the event thread does not wait for FormLogic");
        let outbox = host.outbox.get().unwrap().clone();
        wait_until("the first try has failed", || outbox.status().last_error.is_some());
        assert_eq!(outbox.status().waiting, 1, "kept, not lost");
        assert!(host.dead.lock().unwrap().list(10).is_empty(), "and not given up on");

        // FormLogic is back.
        let reserved = formlogic_on(port);
        wait_until("the kept event is sent", || outbox.status().waiting == 0);
        let runs = reserved.lock().unwrap().clone();
        assert_eq!(runs.len(), 1, "one run, not one per try: {runs:?}");
        assert_eq!(runs[0]["bindingId"], "b1");
        assert!(runs[0]["idempotencyKey"].as_str().unwrap().contains("evt-9"), "keyed by the event, so a retry is harmless");
        assert!(outbox.status().last_error.is_none());
    }

    #[test]
    fn an_event_formlogic_refuses_is_dead_lettered_and_sent_again_on_redrive() {
        let (sb, host) = host_with("outbox-redrive", vec![]);
        let outbox = crate::link::outbox::Outbox::open(&sb.0);
        let _ = host.outbox.set(outbox.clone());
        host.record_dead("aokie", "aokie.call.ended", DeadReason::NotDelivered { detail: "HTTP 400: no such flow".into() }, json!({"name": "aokie.call.ended", "idempotencyKey": "evt-1"}));
        let id = host.dead.lock().unwrap().list(10)[0].id.clone();

        // Unlinked: it stays where it is.
        let (said, done) = host.redrive(&id).unwrap();
        assert!(!done && said[0].contains("not linked"), "{said:?}");

        std::fs::create_dir_all(sb.0.join("link")).unwrap();
        std::fs::write(
            sb.0.join("link").join("account.json"),
            json!({"connectorId": "formlogic", "baseUrl": "http://127.0.0.1:9", "credential": "flk_test", "accountId": "conn_1", "linkedAt": "2026-09-29T00:00:00Z"}).to_string(),
        )
        .unwrap();
        host.set_link(crate::link::open_handle(sb.0.clone()));
        let (_, done) = host.redrive(&id).unwrap();
        assert!(done);
        assert!(host.dead.lock().unwrap().list(10).is_empty());
        assert_eq!(outbox.status().waiting, 1, "back in line for the account");
    }

    // ---- package trust -------------------------------------------------------
    //
    // The registry and trust tests prove the rules; these prove that a plugin is or
    // is not started because of them. A stub executable that cannot run stands in
    // where the test is about a refusal (a start that was attempted would end
    // `Crashed`, "cannot launch"; a refused one ends `Disabled` and was never
    // spawned), and a real child (Node behind a `.cmd` shim, as process.rs does) where
    // the test is about a plugin that does start.

    use crate::plugins::registry::PluginRegistry;
    use crate::plugins::trust::tests::TestKey;
    use crate::plugins::trust::{Publishers, TrustPolicy, TrustService, TrustState};

    /// A host whose registry verifies packages under `policy`, with no boot autostart:
    /// the test puts a plugin on disk after the host exists, and starts it itself.
    fn trusting_host(tag: &str, policy: TrustPolicy, publishers: Publishers) -> (Sandbox, Arc<PluginHost>, Arc<TrustService>) {
        let sb = Sandbox::new(tag);
        let plugins = sb.0.join("plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        let trust = TrustService::new(policy, publishers, plugins.join("trusted-plugins.json"));
        let registry = Arc::new(Mutex::new(PluginRegistry::with_trust(plugins, trust.clone())));
        let host = PluginHost::assemble(
            registry,
            crate::bridge::ledger::new_handle(),
            Arc::new(Mutex::new(TriggerStore::load(sb.0.join("triggers.json")))),
            crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );
        (sb, host, trust)
    }

    fn plugin_manifest(id: &str, command: &str) -> String {
        json!({
            "schemaVersion": 3, "id": id, "name": format!("{id} plugin"), "version": "0.1.0",
            "pluginApiVersion": 1, "entry": { "kind": "process", "command": command },
            "capabilities": [], "connectors": [], "events": [],
        })
        .to_string()
    }

    /// A plugin folder whose executable cannot run.
    fn stub_plugin(sb: &Sandbox, id: &str) -> PathBuf {
        let dir = sb.0.join("plugins").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("manifest.json"), plugin_manifest(id, "plugin.exe")).unwrap();
        std::fs::write(dir.join("plugin.exe"), b"not a real executable").unwrap();
        dir
    }

    fn trust_of(host: &PluginHost, id: &str) -> Option<TrustState> {
        host.registry.lock().unwrap().get(id).and_then(|r| r.trust.as_ref().map(|t| t.state))
    }

    fn state_of(host: &PluginHost, id: &str) -> PluginState {
        host.registry.lock().unwrap().get(id).unwrap().state
    }

    /// `start` returns at once when another start of the same plugin is in flight (the
    /// boot autostart may be), so a test that wants it running waits for it.
    #[cfg(windows)]
    fn wait_running(host: &PluginHost, id: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while state_of(host, id) != PluginState::Running {
            assert!(std::time::Instant::now() < deadline, "{id} never came up: {:?}", host.registry.lock().unwrap().get(id).map(|r| r.reason.clone()));
            thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    #[test]
    fn an_unsigned_plugin_is_not_started_in_a_release_build() {
        let (sb, host, _trust) = trusting_host("rel-unsigned", TrustPolicy::release(), Publishers::default());
        stub_plugin(&sb, "probe");

        let err = host.start("probe").unwrap_err();
        assert!(err.contains("Not signed"), "{err}");
        assert_eq!(state_of(&host, "probe"), PluginState::Disabled, "refused, not crashed");
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::Unsigned));
        assert!(host.logs("probe", None).is_none(), "no process was ever spawned");
    }

    #[test]
    fn a_signed_plugin_that_fails_its_signature_is_not_started_in_any_build() {
        for policy in [TrustPolicy::release(), TrustPolicy::developer()] {
            let key = TestKey::generate("fl-test-2026a");
            let (sb, host, _trust) = trusting_host("signed-bad", policy, key.pinned_for("Probe Co", &["probe"]));
            let dir = stub_plugin(&sb, "probe");
            key.sign(&dir, "probe-plugin", "1.0.0");
            std::fs::write(dir.join("plugin.exe"), b"tampered after signing").unwrap();

            let err = host.start("probe").unwrap_err();
            assert!(err.contains("Quarantined: digest mismatch: plugin.exe"), "developer={}: {err}", policy.is_developer());
            assert_eq!(state_of(&host, "probe"), PluginState::Disabled);
            assert_eq!(trust_of(&host, "probe"), Some(TrustState::Quarantined));
            assert!(host.logs("probe", None).is_none(), "no process was ever spawned");
        }
    }

    #[test]
    fn a_file_swapped_between_the_scan_and_the_launch_does_not_slip_through() {
        let key = TestKey::generate("fl-test-2026a");
        let (sb, host, _trust) = trusting_host("swap", TrustPolicy::release(), key.pinned_for("Probe Co", &["probe"]));
        let dir = stub_plugin(&sb, "probe");
        key.sign(&dir, "probe-plugin", "1.0.0");

        // The scan says verified, and the plugin is loadable.
        host.registry.lock().unwrap().scan();
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::Verified));
        assert!(host.registry.lock().unwrap().get("probe").unwrap().is_loadable());

        // The executable is replaced by one of the same length, its modified time put
        // back: the folder looks exactly as the scan left it.
        let exe = dir.join("plugin.exe");
        let before = std::fs::metadata(&exe).unwrap().modified().unwrap();
        std::fs::write(&exe, vec![b'!'; b"not a real executable".len()]).unwrap();
        std::fs::File::options().write(true).open(&exe).unwrap().set_modified(before).unwrap();

        // start() scans first (and is told the same thing), then verifies again from the
        // bytes, which is what catches it.
        let err = host.start("probe").unwrap_err();
        assert!(err.contains("probe was not started") && err.contains("digest mismatch: plugin.exe"), "{err}");
        assert_eq!(state_of(&host, "probe"), PluginState::Disabled);
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::Quarantined));
        assert!(!host.registry.lock().unwrap().get("probe").unwrap().is_loadable(), "and the listing agrees from then on");
        assert!(host.logs("probe", None).is_none(), "the swapped executable never ran");

        // A crash-restart goes through the same door.
        assert!(host.start("probe").is_err());
    }

    /// A plugin that answers the handshake, in a real child process. `None` where there
    /// is no Node to run it (a missing toolchain is not a defect in the code under test).
    #[cfg(windows)]
    fn node_plugin(sb: &Sandbox, id: &str) -> Option<PathBuf> {
        let has_node = std::process::Command::new("node")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !has_node {
            return None;
        }
        let dir = sb.0.join("plugins").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("plugin.mjs"),
            r#"
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    if (msg.method === "plugin.init") send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } });
    else if (msg.method === "plugin.health") send({ jsonrpc: "2.0", id: msg.id, result: { status: "ok" } });
    else if (msg.method === "plugin.shutdown") process.exit(0);
    else if (msg.id !== undefined) send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown method" } });
  }
});
"#,
        )
        .unwrap();
        std::fs::write(dir.join("plugin.cmd"), "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n").unwrap();
        std::fs::write(dir.join("manifest.json"), plugin_manifest(id, "plugin.cmd")).unwrap();
        Some(dir)
    }

    /// A stand-in for the phone plugin, in a real child process: on `test.ring` it asks this host who may be rung
    /// and says the request is out (`oaiy.ring.plan`, `oaiy.ring.opened`) and answers with what it was told; on
    /// `test.resolve` it says, by its own event, how a request came out (as when a Companion takes the call, or the
    /// plugin withdraws it). It has no command for the owner's dialog: the owner answers on a Companion. `None` where
    /// there is no Node.
    #[cfg(windows)]
    fn ring_plugin(sb: &Sandbox) -> Option<PathBuf> {
        let has_node = std::process::Command::new("node").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
        if !has_node {
            return None;
        }
        let dir = sb.0.join("plugins").join("aokie");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("plugin.mjs"),
            r#"
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
const waiting = new Map();
let next = 500;
const ask = (method, params) => new Promise((resolve) => { const id = next++; waiting.set(id, resolve); send({ jsonrpc: "2.0", id, method, params }); });
async function handle(msg) {
  if (msg.method === undefined && msg.id !== undefined && waiting.has(msg.id)) { waiting.get(msg.id)(msg); waiting.delete(msg.id); return; }
  if (msg.method === "plugin.init") send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } });
  else if (msg.method === "plugin.health") send({ jsonrpc: "2.0", id: msg.id, result: { status: "ok" } });
  else if (msg.method === "plugin.shutdown") process.exit(0);
  else if (msg.method === "connector.request") {
    const { command, payload } = msg.params;
    if (command === "test.ring") {
      const plan = await ask("oaiy.ring.plan", payload.plan);
      let opened = null;
      if (plan.result && plan.result.decision === "ring") {
        opened = await ask("oaiy.ring.opened", { planId: plan.result.planId, requestId: payload.requestId, callId: payload.plan.callId, callEpoch: 1, ownerEpoch: 1, expiresAt: payload.expiresAt });
      }
      send({ jsonrpc: "2.0", id: msg.id, result: { ok: true, plan, opened } });
    } else if (command === "test.resolve") {
      send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } });
      send({ jsonrpc: "2.0", method: "event.emit", params: { event: { schemaVersion: 1, source: "aokie", name: "aokie.call.assistance.resolved", correlationId: payload.requestId, idempotencyKey: "aokie:" + payload.requestId + ":resolved", occurredAt: "2026-09-30T00:00:00Z", data: { requestId: payload.requestId, outcome: payload.outcome } } } });
    } else send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown command" } });
  } else if (msg.id !== undefined) send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown method" } });
}
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    handle(msg);
  }
});
"#,
        )
        .unwrap();
        std::fs::write(dir.join("plugin.cmd"), "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n").unwrap();
        let manifest = json!({
            "schemaVersion": 3, "id": "aokie", "name": "aokie plugin", "version": "0.1.0",
            "pluginApiVersion": 1, "entry": { "kind": "process", "command": "plugin.cmd" },
            "capabilities": ["oaiy.companion.admission"],
            "connectors": [{ "id": "aokie", "commands": ["test.ring", "test.resolve"] }],
            "events": ["aokie.call.assistance.resolved"],
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        Some(dir)
    }

    #[cfg(windows)]
    #[test]
    fn a_phone_plugin_process_rings_the_owner_and_what_the_plugin_says_ends_the_ring() {
        let (sb, host, _trust) = trusting_host("ring-process", TrustPolicy::developer(), Publishers::default());
        let Some(_dir) = ring_plugin(&sb) else { return };
        let ring = a_ring();
        host.set_ring(ring.clone());
        host.start("aokie").expect("the stand-in starts");
        wait_running(&host, "aokie");

        // The plugin's own process asks who may be rung and says the request is out: this desktop rings.
        let expires = ring.clock().unix() + 25;
        let ask = |call: &str, request: &str, key: &str| {
            host.forward_connector("aokie", "test.ring", Some(json!({"plan": {"callId": call, "reason": "caller_asked", "recentCallerTurns": []}, "requestId": request, "expiresAt": expires})), Some(key), Duration::from_secs(10)).expect("the plugin answered")
        };
        let says = |request: &str, outcome: &str, key: &str| {
            host.forward_connector("aokie", "test.resolve", Some(json!({"requestId": request, "outcome": outcome})), Some(key), Duration::from_secs(10)).expect("the plugin answered")
        };
        let first = ask("call_1", "assist_1", "k1");
        assert_eq!(first["plan"]["result"]["decision"], "ring", "{first}");
        assert_eq!(first["opened"]["result"], json!({"ok": true}), "{first}");
        assert_eq!(ring.active().len(), 1);
        assert_eq!(ring.active()[0].caller_name, "Alex");

        // An owner device takes the call: the plugin says the request was transferred (its own event, over the event thread),
        // and the ring is over, once.
        says("assist_1", "transferred", "k2");
        wait_until("the plugin's word never reached the ring", || ring.active().is_empty());
        assert_eq!(ring.ended().iter().map(|e| (e.id.as_str(), e.outcome, e.source)).collect::<Vec<_>>(), vec![("assist_1", "accepted", "phone")]);

        // Another caller: the plugin withdraws the request (as it does when this desktop asks it to), and the ring is over.
        let second = ask("call_2", "assist_2", "k3");
        assert_eq!(second["opened"]["result"], json!({"ok": true}), "{second}");
        assert_eq!(ring.active().len(), 1);
        says("assist_2", "cancelled", "k4");
        wait_until("the withdrawal never reached the ring", || ring.active().is_empty());
        assert_eq!(ring.ended().iter().map(|e| (e.id.as_str(), e.outcome, e.source)).collect::<Vec<_>>(), vec![("assist_1", "accepted", "phone"), ("assist_2", "cancelled", "phone")]);

        // A plan the plugin never asked for rings nothing, even from a plugin that holds the capability.
        let stray = host.handle_ring_request("aokie", "oaiy.ring.opened", json!({"planId": "plan_made_up", "requestId": "assist_3", "callId": "call_1", "callEpoch": 1, "ownerEpoch": 1, "expiresAt": expires})).unwrap_err();
        assert_eq!(stray.0, "unknown_plan");
        host.stop("aokie").unwrap();
    }

    /// A phone plugin that answers the handshake in a real child process and writes what `plugin.init` sent it to
    /// `init.json` in its data folder, so that a test can see the roster it was handed. `None` where there is no Node.
    #[cfg(windows)]
    fn admission_plugin(sb: &Sandbox) -> Option<PathBuf> {
        let has_node = std::process::Command::new("node").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
        if !has_node {
            return None;
        }
        let dir = sb.0.join("plugins").join("aokie");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("plugin.mjs"),
            r#"
import fs from "node:fs";
import path from "node:path";
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    if (msg.method === "plugin.init") {
      fs.mkdirSync(msg.params.dataDir, { recursive: true });
      fs.writeFileSync(path.join(msg.params.dataDir, "init.json"), JSON.stringify(msg.params));
      send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } });
    }
    else if (msg.method === "plugin.health") send({ jsonrpc: "2.0", id: msg.id, result: { status: "ok" } });
    else if (msg.method === "plugin.shutdown") process.exit(0);
    else if (msg.id !== undefined) send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown method" } });
  }
});
"#,
        )
        .unwrap();
        std::fs::write(dir.join("plugin.cmd"), "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n").unwrap();
        let manifest = json!({
            "schemaVersion": 3, "id": "aokie", "name": "aokie plugin", "version": "0.1.0",
            "pluginApiVersion": 1, "entry": { "kind": "process", "command": "plugin.cmd" },
            "capabilities": ["oaiy.companion.admission"], "connectors": [], "events": [],
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        Some(dir)
    }

    #[cfg(windows)]
    #[test]
    fn a_started_plugin_is_presented_the_roster_its_init_handed_it_and_a_new_start_takes_the_new_one() {
        // The whole path with a real process: the roster in the `plugin.init` the plugin received is the roster the issuer is
        // asked to bind while it runs, whatever is approved or revoked meanwhile, and starting it again takes the roster as it is.
        use crate::companion::identity::testing::phone_key;
        let (sb, host, _trust) = trusting_host("adm-process", TrustPolicy::developer(), Publishers::default());
        let Some(dir) = admission_plugin(&sb) else { return };
        host.registry.lock().unwrap().scan();
        let companion = broker_for(&sb, &host);
        let issuer = strict_issuer();
        host.companion
            .lock()
            .unwrap()
            .clone()
            .unwrap()
            .upstream
            .set(crate::companion::upstream::UpstreamConfig { base_url: issuer.base.clone(), token: "flk_issuer".into(), app_id: Some("app_1".into()) })
            .unwrap();
        let identity = companion.identity_for("aokie").expect("the plugin declares the capability");
        let (a, b, c) = (phone_key(1), phone_key(2), phone_key(3));
        // Two phones, approved in the order that is not the sorted one.
        let (first, second) = if a.thumbprint > b.thumbprint { (&a, &b) } else { (&b, &a) };
        identity.approve_for_tests(first);
        identity.approve_for_tests(second);

        // The roster in the last `plugin.init` the plugin received.
        let handed = || -> Value {
            let written = std::fs::read_to_string(crate::plugins::runner::plugin_data_dir(&dir).join("init.json")).expect("the plugin wrote what it was sent");
            serde_json::from_str::<Value>(&written).unwrap()["privateBootstrap"]["approvedMobileRoster"].clone()
        };
        let admit = || {
            host.handle_plugin_request("aokie", "companion.admission", json!({})).unwrap_or_else(|e| panic!("{e:?}"));
            asked_to_bind(&issuer)
        };

        host.start("aokie").expect("the stand-in starts");
        wait_running(&host, "aokie");
        let init = handed();
        let presented = admit();
        assert_eq!(presented.1, init["revision"].as_u64().unwrap());
        assert_eq!(presented.2, init["rosterHash"].as_str().unwrap(), "what the plugin was handed is what is presented for it");
        let mut handed_keys: Vec<String> = init["keys"].as_array().unwrap().iter().map(|k| k["thumbprint"].as_str().unwrap().to_string()).collect();
        handed_keys.sort();
        assert_eq!(presented.0, handed_keys);

        // A phone approved while it runs: it is not in what the plugin holds, and what is presented does not change.
        identity.approve_for_tests(&c);
        assert_eq!(admit(), presented);
        // A phone revoked while it runs: the roster the plugin holds names it, so it is not presented again.
        identity.revoke(&a.thumbprint).unwrap();
        let after_revoke = admit();
        assert!(!after_revoke.0.contains(&a.thumbprint) && after_revoke != presented, "{after_revoke:?}");

        // Started again, it is handed the roster as it is, and that is what it is presented.
        host.stop("aokie").unwrap();
        host.start("aokie").expect("and starts again");
        wait_running(&host, "aokie");
        let again = handed();
        assert_ne!(again["rosterHash"], init["rosterHash"]);
        let presented = admit();
        assert_eq!((presented.1, presented.2.as_str()), (again["revision"].as_u64().unwrap(), again["rosterHash"].as_str().unwrap()));
        let mut expected = vec![b.thumbprint.clone(), c.thumbprint.clone()];
        expected.sort();
        assert_eq!(presented.0, expected);
        host.stop("aokie").unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn an_unsigned_plugin_still_starts_in_a_developer_build_as_it_always_has() {
        // The owner's own flow: `tauri dev`, a local build of the plugin with no
        // signature. It starts, and it reports that it is unsigned and allowed.
        let (sb, host, _trust) = trusting_host("dev-unsigned", TrustPolicy::developer(), Publishers::default());
        let Some(_dir) = node_plugin(&sb, "probe") else { return };

        host.start("probe").expect("a developer build starts an unsigned plugin");
        wait_running(&host, "probe");
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::UnsignedDev));
        host.stop("probe").unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_verified_plugin_starts_and_says_who_signed_it() {
        let key = TestKey::generate("fl-test-2026a");
        let (sb, host, _trust) = trusting_host("verified", TrustPolicy::release(), key.pinned_for("Probe Co", &["probe"]));
        let Some(dir) = node_plugin(&sb, "probe") else { return };
        key.sign(&dir, "probe-plugin", "1.0.0");

        host.start("probe").expect("a verified plugin starts in a release build");
        wait_running(&host, "probe");
        let reg = host.registry.lock().unwrap();
        let trust = reg.get("probe").unwrap().trust.clone().unwrap();
        drop(reg);
        assert_eq!(trust.state, TrustState::Verified);
        assert_eq!(trust.publisher.as_deref(), Some("Probe Co"));
        host.stop("probe").unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_plugin_the_person_trusted_starts_in_a_release_build_until_it_changes() {
        let (sb, host, trust) = trusting_host("trusted", TrustPolicy::release(), Publishers::default());
        let Some(dir) = node_plugin(&sb, "probe") else { return };

        assert!(host.start("probe").is_err(), "unsigned, and nobody has trusted it");
        trust.trust_local(&dir, "probe").unwrap();
        host.start("probe").expect("trusted, so it starts");
        wait_running(&host, "probe");
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::TrustedLocal));
        host.stop("probe").unwrap();

        // A changed file is a package nobody trusted.
        let script = dir.join("plugin.mjs");
        let mut text = std::fs::read_to_string(&script).unwrap();
        text.push_str("\n// changed\n");
        std::fs::write(&script, text).unwrap();
        let err = host.start("probe").unwrap_err();
        assert!(err.contains("changed since you trusted it"), "{err}");
        assert_eq!(state_of(&host, "probe"), PluginState::Disabled);
    }

    #[cfg(windows)]
    #[test]
    fn screen_ai_capability_requires_trust_declaration_and_the_same_live_process() {
        let (sb,host,trust) = trusting_host("screen-ai",TrustPolicy::release(),Publishers::default());
        let Some(dir) = node_plugin(&sb,"probe") else { panic!("Node is required for this real-process qualification test"); };
        assert!(host.screen_capability("probe","oaiy.ai.complete").is_err());
        trust.trust_local(&dir,"probe").unwrap(); host.start("probe").unwrap(); wait_running(&host,"probe");
        assert_eq!(host.screen_capability("probe","oaiy.ai.complete").err().unwrap().0,"capability_denied");
        host.stop("probe").unwrap();
        let path = dir.join("manifest.json");
        let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest["capabilities"] = json!(["oaiy.*"]); std::fs::write(&path,manifest.to_string()).unwrap();
        trust.trust_local(&dir,"probe").unwrap(); host.start("probe").unwrap(); wait_running(&host,"probe");
        assert_eq!(host.screen_capability("probe","oaiy.ai.complete").err().unwrap().0,"capability_denied","a historical wildcard cannot grant spending");
        host.stop("probe").unwrap();
        manifest["capabilities"] = json!(["oaiy.ai.complete"]); std::fs::write(path,manifest.to_string()).unwrap();
        trust.trust_local(&dir,"probe").unwrap(); host.start("probe").unwrap(); wait_running(&host,"probe");
        let lease = host.screen_capability("probe","oaiy.ai.complete").unwrap();
        assert!(host.holds_screen_capability("probe","oaiy.ai.complete",&lease));
        host.stop("probe").unwrap(); assert!(!host.holds_screen_capability("probe","oaiy.ai.complete",&lease));
        host.start("probe").unwrap(); wait_running(&host,"probe");
        assert!(!host.holds_screen_capability("probe","oaiy.ai.complete",&lease),"restart cannot revive the old lease");
        let current = host.screen_capability("probe","oaiy.ai.complete").unwrap();
        assert!(host.holds_screen_capability("probe","oaiy.ai.complete",&current)); host.stop("probe").unwrap();
    }

    /// Rewrite a file with other bytes of the same length and put its modified time back.
    #[cfg(windows)]
    fn swap_keeping_size_and_time(path: &std::path::Path, bytes: &[u8]) {
        let before = std::fs::metadata(path).unwrap().modified().unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), bytes.len() as u64);
        std::fs::write(path, bytes).unwrap();
        std::fs::File::options().write(true).open(path).unwrap().set_modified(before).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_manifest_swapped_behind_the_scan_does_not_choose_what_is_started() {
        // manifest.json decides which file is started. Swapped for another of the same
        // length with its time put back, the folder looks as the last scan left it, and a
        // scan that trusted the stat cache would have built the plugin from the swapped
        // file; a launch that read it again would have started what the signature does
        // not name. Neither does: the scan and the launch are about the signed bytes.
        let key = TestKey::generate("fl-test-2026a");
        let (sb, host, _trust) = trusting_host("swap-manifest", TrustPolicy::release(), key.pinned_for("Probe Co", &["probe"]));
        let Some(dir) = node_plugin(&sb, "probe") else { return };
        std::fs::write(dir.join("xother.cmd"), "@echo off\r\necho other> \"%~dp0..\\marker.txt\"\r\n").unwrap();
        key.sign(&dir, "probe-plugin", "1.0.0");
        let marker = sb.0.join("plugins").join("marker.txt");

        host.registry.lock().unwrap().scan();
        assert_eq!(trust_of(&host, "probe"), Some(TrustState::Verified));

        let signed_manifest = std::fs::read(dir.join("manifest.json")).unwrap();
        let swapped = String::from_utf8(signed_manifest.clone()).unwrap().replace("plugin.cmd", "xother.cmd");
        swap_keeping_size_and_time(&dir.join("manifest.json"), swapped.as_bytes());
        let err = host.start("probe").unwrap_err();
        assert!(err.contains("probe was not started") && err.contains("digest mismatch: manifest.json"), "{err}");
        assert_eq!(state_of(&host, "probe"), PluginState::Disabled);
        assert!(!marker.exists(), "the file the swapped manifest names never ran");

        // Put right, it starts, from the entry the signature names.
        swap_keeping_size_and_time(&dir.join("manifest.json"), &signed_manifest);
        host.start("probe").expect("the signed manifest is back");
        wait_running(&host, "probe");
        let entry = host.registry.lock().unwrap().get("probe").unwrap().manifest.as_ref().unwrap().entry.command.clone();
        assert_eq!(entry, "plugin.cmd");
        assert!(!marker.exists());
        host.stop("probe").unwrap();
    }
}
