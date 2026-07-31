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
        let bindings = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, bindings }
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
    /// Crash-restart due times. In the table — NOT supervisor-local — so a
    /// manual stop() can cancel one before it fires.
    restarts: HashMap<String, Instant>,
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
    desktop_version: String,
    dev_mode: bool,
}

impl PluginHost {
    /// Build the host and start its two background threads (events, supervisor).
    pub fn new(
        registry: PluginRegistryHandle,
        ledger: LedgerHandle,
        triggers: TriggerStoreHandle,
        dead: DeadLetterHandle,
        desktop_version: String,
        dev_mode: bool,
    ) -> Arc<Self> {
        let (event_tx, event_rx) = sync_channel::<(String, Value)>(1024);
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
            desktop_version,
            dev_mode,
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
                            host.logs_drop_notice(&plugin_for_events);
                            // The log ring is itself overwriting under the flood
                            // that caused this, so the durable record is what
                            // actually survives.
                            let (_, envelope) = match e {
                                std::sync::mpsc::TrySendError::Full(v) => v,
                                std::sync::mpsc::TrySendError::Disconnected(v) => v,
                            };
                            host.record_shed(&plugin_for_events, envelope);
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

        // Handshake, with the process killed on failure — a plugin that cannot
        // answer plugin.init is not going to answer anything else, and leaving
        // it alive would hold its hardware while reporting Crashed.
        match process.init(manifest.plugin_api_version, &super::runner::plugin_data_dir(&dir), self.dev_mode) {
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
        }
        drop(claim);
        if let Ok(mut reg) = self.registry.lock() {
            reg.reset_restarts(id);
        }
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
        let drained: Vec<(String, Arc<PluginProcess>)> = match self.procs.lock() {
            Ok(mut t) => {
                // No restarts may fire during shutdown, and any in-flight start
                // must be abandoned when it completes.
                t.restarts.clear();
                let starting: Vec<String> = t.starting.iter().cloned().collect();
                for id in starting {
                    t.stop_during_start.insert(id);
                }
                t.running.drain().collect()
            }
            Err(_) => return,
        };
        for (id, p) in drained {
            p.shutdown();
            self.set_state(&id, PluginState::Stopped, Some("OAIY Desktop is exiting.".into()));
        }
    }

    /// Recent log lines for a plugin, running or not.
    pub fn logs(&self, id: &str, tail: Option<usize>) -> Option<Vec<crate::services::runner::LogLine>> {
        self.procs
            .lock()
            .ok()?
            .running
            .get(id)
            .map(|p| p.logs.snapshot(tail))
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

    /// Answer a plugin-initiated request (`flow.run`).
    fn handle_plugin_request(
        &self,
        plugin_id: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, (String, String)> {
        if method != "flow.run" {
            return Err((
                "invalid_request".into(),
                format!("unknown method {method:?}; this host answers only flow.run"),
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
}
