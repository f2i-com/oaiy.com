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
use crate::bridge::deadletters::{DeadLetterHandle, DeadReason};
use crate::bridge::ledger::{LedgerHandle, LineageRef, ReserveOutcome, RunRequest};
use crate::bridge::triggers::{dispatch, DispatchOutcome, Event, SkipReason, TriggerBinding};
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

/// Persistent trigger bindings, JSON on disk.
///
/// A file rather than a database because the write rate is human (someone edits
/// a binding), the read rate is per-event, and the whole set is cached in
/// memory. Written atomically (`.tmp` + rename) so a crash mid-write cannot
/// leave half a JSON file that silently disables every trigger on next boot.
pub struct TriggerStore {
    path: PathBuf,
    bindings: Vec<TriggerBinding>,
}

impl TriggerStore {
    pub fn load(path: PathBuf) -> Self {
        let bindings = match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&path, &text),
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
        Self { path, bindings }
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
        if let Some(expr) = binding.condition.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
            crate::bridge::triggers::condition_is_evaluatable(expr)
                .map_err(|why| format!("condition cannot be evaluated ({why}): {expr}"))?;
        }
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
    /// The account's trigger bindings, cached between events.
    flow_bindings: crate::link::flows::FlowBindings,
}

/// What the host needs to answer `companion.admission`: this desktop's own
/// device trust, and the upstream that turns it into a gateway admission.
#[derive(Clone)]
pub struct CompanionBroker {
    pub companion: crate::companion::routes::CompanionHandle,
    pub upstream: crate::companion::upstream::UpstreamHandle,
}

impl PluginHost {
    /// Build the host and start its three background threads (events, shed,
/// supervisor).
    pub fn new(
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
            flow_bindings: crate::link::flows::FlowBindings::new(),
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

        host.spawn_autostart();
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
        let (manifest, dir) = {
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
            let m = rec.manifest.clone().ok_or_else(|| {
                format!(
                    "{id} cannot start: {}",
                    rec.reason.clone().unwrap_or_else(|| "its manifest is invalid".into())
                )
            })?;
            (m, rec.dir.clone())
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

        self.set_state(id, PluginState::Starting, Some("Launching…".into()));

        let host_for_events = Arc::downgrade(self);
        let plugin_for_events = id.to_string();
        let host_for_requests = Arc::downgrade(self);
        let plugin_for_requests = id.to_string();

        let spawn_result = PluginProcess::spawn(
            &manifest,
            &dir,
            SpawnOptions {
                desktop_version: self.desktop_version.clone(),
                dev_mode: self.dev_mode,
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
        let companion_bootstrap = self
            .companion
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|_| {
                self.registry
                    .lock()
                    .map(|reg| reg.grants(id, "oaiy.companion.admission"))
                    .unwrap_or(false)
            })
            .and_then(|b| b.companion.identity_for(id).ok())
            .and_then(|identity| identity.private_bootstrap(manifest.plugin_api_version as u16));
        match process.init(
            manifest.plugin_api_version,
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

    /// Reserve a run on the linked account for every binding this event fires.
    ///
    /// Best effort and non-fatal: the local dispatch has already happened, and
    /// a provider that is unreachable must not stop this desktop's own
    /// automation. Each outcome is logged with the binding it belongs to,
    /// because a trigger that silently does nothing is the hardest kind of
    /// automation bug to find.
    fn fan_out_to_linked_flows(&self, event: &crate::bridge::triggers::Event, envelope: &Value) {
        let link = {
            let guard = self.link.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(l) => l.clone(),
                None => return,
            }
        };
        let Some(account) = link.account() else {
            return;
        };
        let Some(spec) = crate::link::descriptor::find(link.data_dir(), &account.connector_id)
            .and_then(|d| d.flows)
        else {
            return;
        };
        let bindings = match self.flow_bindings.load(&account, &spec) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("flow bindings for {}: {e}", event.name);
                return;
            }
        };
        let (fire, skipped) = crate::link::flows::select(&bindings, &event.name);
        for (binding, reason) in skipped {
            log::info!(
                "flow binding {} did not fire for {}: {}",
                binding.id,
                event.name,
                reason.message()
            );
        }
        for binding in fire {
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
                Err(e) => log::warn!("flow binding {} could not reserve a run: {e}", binding.id),
            }
        }
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

        let dispatched = self.dispatch_event(&event);
        if let Some(reason) = dispatched.dead {
            self.record_dead(&event.source, &event.name, reason, envelope.clone());
        }
        let outcomes = dispatched.outcomes;

        // …and the SAME event to the linked account's own flows. The flows a
        // user actually built live in the provider's web app, so without this
        // the event was matched only against local bindings, found nothing, and
        // the flow they wrote never ran. Before the ack, for the same reason
        // the local dispatch is.
        self.fan_out_to_linked_flows(&event, &envelope);

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
        if !idempotency_key.is_empty() {
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

    /// Let a binary supply the companion broker once its HTTP surface exists.
    pub fn set_companion_broker(&self, broker: CompanionBroker) {
        *self.companion.lock().unwrap_or_else(|e| e.into_inner()) = Some(broker);
    }

    /// Give the host the linked account, so plugin events can reach the flows
    /// the user built there. Until this is set, events stay local — which is
    /// the correct behaviour for an unlinked desktop.
    pub fn set_link(&self, link: crate::link::LinkHandle) {
        *self.link.lock().unwrap_or_else(|e| e.into_inner()) = Some(link);
        // A fresh link may belong to a different account entirely; keeping the
        // previous account's bindings would fire the wrong flows.
        self.flow_bindings.invalidate();
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
        let approved: Vec<String> = status
            .approved_mobiles
            .iter()
            .map(|m| m.endpoint_key.thumbprint.clone())
            .collect();
        if approved.is_empty() {
            return Err((
                "not_paired".into(),
                "no Companion device has been approved on this desktop yet".into(),
            ));
        }
        let endpoint_key = status.endpoint_key.clone().ok_or_else(|| {
            (
                "unavailable".to_string(),
                "the desktop endpoint identity is unavailable".to_string(),
            )
        })?;
        let binding = serde_json::json!({
            "endpointPublicKey": endpoint_key,
            "holderKeyThumbprint": endpoint_key.thumbprint,
            "approvedPeerKeyThumbprints": approved,
            "peerRosterRevision": status.roster_revision,
            "peerRosterHash": status.roster_hash,
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

    /// Answer a plugin-initiated request (`flow.run`, `companion.admission`).
    fn handle_plugin_request(
        &self,
        plugin_id: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, (String, String)> {
        if method == "companion.admission" {
            return self.handle_companion_admission(plugin_id, params);
        }
        if method != "flow.run" {
            return Err((
                "invalid_request".into(),
                format!(
                    "unknown method {method:?}; this host answers flow.run and companion.admission"
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
        match ledger.reserve(&req) {
            ReserveOutcome::Reserved(run) | ReserveOutcome::Duplicate(run) => Ok(json!({
                "runId": run.run_id,
                "status": run.status,
            })),
            ReserveOutcome::Refused { reason } => {
                Err(("invalid_request".into(), format!("refused: {reason}")))
            }
        }
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

            let tracker = trackers.entry(id.clone()).or_default();
            match process.health() {
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
        let bindings: Vec<TriggerBinding> = match self.triggers.lock() {
            Ok(t) => t.list().to_vec(),
            Err(_) => Vec::new(),
        };

        let results = match self.ledger.lock() {
            Ok(mut ledger) => dispatch(&mut ledger, &bindings, event),
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
    /// because `upsert` now REFUSES a binding whose condition cannot be parsed —
    /// and the dead-letter tests below need exactly that state. Which is not a
    /// contrivance: a binding saved before that validation existed, or one
    /// hand-edited into triggers.json, arrives through this same load path. It
    /// is precisely why the dead-letter handling still has to work.
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
        // `====` is not an operator the restricted evaluator understands, so the
        // binding is refused rather than guessed. The author meant this to fire;
        // without a dead letter they would never learn it didn't.
        let (_sb, host) = host_with("broken", vec![binding("b1", Some("$event.data.from ==== 1"))]);
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
        let (_sb, host) = host_with("redrive", vec![binding("b1", Some("$event.data.from ==== 1"))]);
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
}
