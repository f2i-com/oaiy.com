//! Running the linked account's APP LOGIC SCRIPTS against this desktop's events.
//!
//! The third mechanism that hangs off a plugin event, and the one that actually
//! fills a record. `flows.rs` reserves a run on the account; the local trigger
//! dispatcher fires this desktop's own bindings; and this module executes the
//! small scripts an app carries with itself, which turn one event into a list of
//! EFFECTS — a record written, a key remembered, a connector command sent.
//!
//! ```text
//!   plugin event ──► GET {path}  (apps, their forms, their scripts)
//!                      └─► one logic_block per script ──► `oaiy run` ──► effects
//!                                                            └─► POST/PUT records
//!                                                            └─► storage marker
//!                                                            └─► connector command
//! ```
//!
//! # Four rules everything here follows
//!
//! **There is no second script engine.** The scripts are JavaScript, and this
//! crate does not embed a JS runtime to run them. They are wrapped into a
//! one-node-per-script graph and handed to the same bundled CLI that executes
//! flows — the identical reasoning as `bridge/worker.rs`: a second engine kept
//! "in sync" is the failure this architecture exists to avoid.
//!
//! **The provider is data.** Effect TYPE names, the field names inside an
//! effect, and every path come from [`AppLogicSpec`]. Nothing below knows a
//! provider's name or any of its vocabulary, and nothing here should learn one.
//!
//! **Storage is the dedup, so it is written like it matters.** A script marks an
//! event handled by storing a key. Losing that file replays every transcript
//! line the account has ever recorded, so it is written temp-and-rename and
//! flushed before this function returns — never at some later convenient moment.
//!
//! **Nothing fails quietly.** An effect type the descriptor does not map, a form
//! key that resolves to no form, a write the provider refuses — each becomes an
//! [`Outcome`] naming what and why. A silent skip here looks exactly like a
//! working install that simply records nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::descriptor::{AppLogicCatalogue, AppLogicOperation, AppLogicSpec};
use super::LinkedAccount;
use crate::bridge::worker::{run_flow_cli, CliOutcome, CliRequest};

/// How long a fetched script set is reused.
///
/// Longer than the flow bindings' minute because a script is part of an app's
/// published version rather than something a user toggles, and the document is
/// large — it carries every script of every installed app. Short enough that
/// reinstalling an app is picked up while somebody is still watching.
const CATALOG_TTL: Duration = Duration::from_secs(120);

/// Wall-clock budget for one event's scripts.
///
/// They are pure transforms with no I/O of their own, so this is a guard against
/// a runaway loop rather than a work budget. The CLI gets the same number and
/// times out first, so the error names the script that stuck.
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

/// Keys one app's storage keeps before the oldest are dropped.
///
/// Unbounded, this file grows by one key per transcript line for the life of the
/// install. Bounded, a redelivery older than the window replays — which is the
/// lesser harm, and is the direction that keeps the file readable.
const MAX_STORAGE_KEYS: usize = 20_000;

/// How long the network legs wait.
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a failed catalogue fetch is left alone before it is tried again.
///
/// This runs on the event thread, so a provider that is down would otherwise
/// cost EVERY event a full [`HTTP_TIMEOUT`] — and the events that pile up behind
/// it are dropped, not delayed.
const FETCH_RETRY_AFTER: Duration = Duration::from_secs(30);

// --- what the provider publishes -----------------------------------------

/// One app as the provider listed it, read through the descriptor.
///
/// Kept as the raw document rather than parsed into named fields on purpose:
/// every name in it — the block the scripts live in, the key a form is known
/// by — is the PROVIDER's, and a `#[serde(rename_all)]` struct would spell each
/// one in this file just as surely as a string literal would. Read through
/// [`AppLogicCatalogue`], a second provider is a descriptor away; read through
/// a struct, it is a release away.
#[derive(Debug, Clone)]
pub struct AppEntry(Value);

/// One script, once the catalogue names have been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptRef<'a> {
    pub id: &'a str,
    /// The JavaScript itself, defining `run(ctx)`.
    pub source: &'a str,
}

impl AppEntry {
    /// A stable id for this app, for the storage file and the logs.
    fn key<'a>(&'a self, c: &AppLogicCatalogue) -> Option<&'a str> {
        let app = self.0.get(&c.app_block)?;
        app.get(&c.app_id)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or_else(|| {
                app.get(&c.app_slug)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
            })
    }

    /// The scripts that run on an event, in publication order.
    fn event_scripts<'a>(&'a self, c: &AppLogicCatalogue, hook: &str) -> Vec<ScriptRef<'a>> {
        self.0
            .get(&c.logic_block)
            .and_then(|l| l.get(&c.scripts))
            .and_then(Value::as_array)
            .map(|scripts| {
                scripts
                    .iter()
                    .filter(|s| s.get(&c.script_hook).and_then(Value::as_str) == Some(hook))
                    .filter_map(|s| {
                        let source = s.get(&c.script_source).and_then(Value::as_str)?;
                        if source.trim().is_empty() {
                            return None;
                        }
                        Some(ScriptRef {
                            id: s.get(&c.script_id).and_then(Value::as_str).unwrap_or("").trim(),
                            source,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// This app's forms, as (portable key, the account's own id) pairs. Either
    /// side blank is not a form a script can name, so it is not offered.
    fn forms<'a>(&'a self, c: &AppLogicCatalogue) -> Vec<(&'a str, &'a str)> {
        self.0
            .get(&c.forms)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter_map(|f| {
                let key = f.get(&c.form_key).and_then(Value::as_str)?.trim();
                let id = f.get(&c.form_id).and_then(Value::as_str)?.trim();
                (!key.is_empty() && !id.is_empty()).then_some((key, id))
            })
            .collect()
    }
}

// --- what this module reports --------------------------------------------

/// One thing that was attempted, and how it went.
///
/// Returned rather than logged in place so the caller decides the level and so
/// tests can assert on the sequence — the point of the lane is the ORDER of the
/// writes as much as the writes themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub app: String,
    pub script: String,
    /// The effect type as the script named it, or a lane-level label when the
    /// failure happened before any effect was reached.
    pub effect: String,
    pub detail: Result<String, String>,
}

impl Outcome {
    fn err(app: &str, script: &str, effect: &str, detail: impl Into<String>) -> Self {
        Self {
            app: app.to_string(),
            script: script.to_string(),
            effect: effect.to_string(),
            detail: Err(detail.into()),
        }
    }
}

/// Perform a connector command on this desktop. Supplied by the caller so this
/// module needs no knowledge of the plugin host.
pub type ConnectorFn<'a> =
    dyn Fn(&str, &str, Option<Value>, &str) -> Result<Value, String> + 'a;

// --- the script catalogue, cached ----------------------------------------

/// The account's apps and their scripts, cached between events.
///
/// A failed fetch returns the LAST known set rather than an empty one: treating
/// "could not ask" as "this account has no logic" would silently stop recording
/// the moment the network hiccuped, and the symptom — an empty transcript — does
/// not point at the network at all.
pub struct Catalog {
    inner: Mutex<CatalogState>,
}

#[derive(Default)]
struct CatalogState {
    fetched: Option<(Instant, Vec<AppEntry>)>,
    /// When a failed fetch may be tried again. Without it a provider that is
    /// down costs every single event a full HTTP timeout ON THE EVENT THREAD,
    /// which backs the queue up until events are shed — and a shed event is one
    /// whose records are never written at all.
    retry_at: Option<Instant>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(CatalogState::default()),
        }
    }

    pub fn load(
        &self,
        account: &LinkedAccount,
        spec: &AppLogicSpec,
    ) -> Result<Vec<AppEntry>, String> {
        {
            let state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((at, apps)) = state.fetched.as_ref() {
                if at.elapsed() < CATALOG_TTL {
                    return Ok(apps.clone());
                }
                // Stale, but the last attempt failed recently enough that a
                // fresh one would only cost this event another timeout.
                if state.retry_at.is_some_and(|t| Instant::now() < t) {
                    log::warn!(
                        "app logic: still cannot refresh the account's scripts; running the last \
                         known set of {} app(s), which is {}s old",
                        apps.len(),
                        at.elapsed().as_secs()
                    );
                    return Ok(apps.clone());
                }
            } else if state.retry_at.is_some_and(|t| Instant::now() < t) {
                return Err(
                    "the account's app list could not be read, and there is no earlier one to \
                     fall back on"
                        .to_string(),
                );
            }
        }
        match fetch(account, spec) {
            Ok(apps) => {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                state.fetched = Some((Instant::now(), apps.clone()));
                state.retry_at = None;
                Ok(apps)
            }
            Err(e) => {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                state.retry_at = Some(Instant::now() + FETCH_RETRY_AFTER);
                match state.fetched.as_ref() {
                    // Never silently: running yesterday's scripts because the
                    // network hiccuped is the right call, but an install doing
                    // it for a week with nothing in the log is not.
                    Some((at, apps)) => {
                        log::warn!(
                            "app logic: could not refresh the account's scripts ({e}); running \
                             the last known set of {} app(s), fetched {}s ago",
                            apps.len(),
                            at.elapsed().as_secs()
                        );
                        Ok(apps.clone())
                    }
                    None => Err(e),
                }
            }
        }
    }

    /// Drop the cache, so the next event refetches. A fresh link may be a
    /// different account entirely, whose scripts would write the wrong records.
    pub fn invalidate(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = CatalogState::default();
    }
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("could not build the app-logic client: {e}"))
}

fn fetch(account: &LinkedAccount, spec: &AppLogicSpec) -> Result<Vec<AppEntry>, String> {
    let resp = client()?
        .get(super::oauth::join(&account.base_url, &spec.path))
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the app-logic lane: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().unwrap_or(Value::Null);
        let message = body
            .get("message")
            .or_else(|| body.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the app list");
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "this desktop's key may not read the account's app logic ({message}) — link again"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    let reply: Value = resp
        .json()
        .map_err(|e| format!("the app-logic lane returned an unreadable app list: {e}"))?;
    // A missing array is NOT an account with no apps: it is a reply this
    // descriptor does not describe, and treating the two the same would leave
    // an install recording nothing with nothing to show for it.
    let apps = reply.get(&spec.catalogue.apps).ok_or_else(|| {
        format!(
            "the app-logic lane's reply carries no {:?}, so this connector's descriptor does not \
             describe it",
            spec.catalogue.apps
        )
    })?;
    let apps = apps.as_array().ok_or_else(|| {
        format!(
            "the app-logic lane returned {:?} as a {} rather than a list of apps",
            spec.catalogue.apps,
            type_name(apps)
        )
    })?;
    Ok(apps.iter().cloned().map(AppEntry).collect())
}

// --- the storage a script dedups on ---------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AppStorage {
    /// Insertion order, so the OLDEST key is the one dropped when the cap is
    /// reached. A plain map has no order, and evicting an arbitrary key would
    /// replay an arbitrary event.
    #[serde(default)]
    order: Vec<String>,
    #[serde(default)]
    values: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StorageFile {
    #[serde(default)]
    apps: BTreeMap<String, AppStorage>,
}

/// Serialises every write in this process, however many [`StorageStore`]s were
/// opened.
///
/// The store is opened fresh on each event, so a lock owned by an instance would
/// serialise nothing at all — the two callers would hold two different locks
/// over one file and lose each other's markers.
static STORAGE_GUARD: Mutex<()> = Mutex::new(());

/// What the scripts remember, on disk, keyed by app.
///
/// Written temp-and-rename because a half-written file is not a lost marker but
/// a lost FILE: serde refuses it, the whole map reads as empty, and every event
/// the account has ever handled is handled again.
pub struct StorageStore {
    path: PathBuf,
}

/// What was on disk, and whether it can be trusted.
enum Stored {
    /// The file as read — or an empty one, because there is none yet.
    Read(StorageFile),
    /// The file could not be READ: a lock, a permission, a failing disk. The
    /// markers themselves are almost certainly fine, so this must never be
    /// treated as "empty" — writing an empty map over them would replay every
    /// event the account has ever handled.
    Unreadable(String),
}

impl StorageStore {
    /// `<data>/link/app-logic-storage.json`, beside the stored link.
    pub fn open(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join("link").join("app-logic-storage.json"),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file, distinguishing "there is none" from "it would not read".
    ///
    /// The distinction is the whole point. `read().ok().unwrap_or_default()`
    /// turns a transient read failure into an empty map, and the write that
    /// follows makes it permanent.
    fn read(&self) -> Stored {
        match std::fs::read_to_string(&self.path) {
            Ok(raw) => match serde_json::from_str::<StorageFile>(&raw) {
                Ok(file) => Stored::Read(file),
                // Corrupt, not absent. Deliberately NOT started again from
                // empty: the only copy of the markers is in this file, and
                // overwriting it would silently rewrite the whole transcript on
                // the next delivery. It is left exactly where it is, and said
                // out loud on every event until somebody looks.
                Err(e) => Stored::Unreadable(format!(
                    "{} is not readable JSON ({e}); it holds the markers that stop already \
                     recorded events from being recorded again, so it is left untouched — move \
                     it aside to start over, accepting that everything still in flight replays",
                    self.path.display()
                )),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Stored::Read(StorageFile::default())
            }
            Err(e) => Stored::Unreadable(format!("could not read {}: {e}", self.path.display())),
        }
    }

    /// What one app's scripts can see.
    ///
    /// An error rather than an empty map when the file will not read: the
    /// scripts dedup on these keys, so handing them "you have seen nothing"
    /// makes every one of them act again.
    pub fn values(&self, app: &str) -> Result<Map<String, Value>, String> {
        let _held = STORAGE_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        match self.read() {
            Stored::Read(f) => Ok(f.apps.get(app).map(|a| a.values.clone()).unwrap_or_default()),
            Stored::Unreadable(why) => Err(why),
        }
    }

    /// Store a batch of keys for one app, durably, before returning.
    fn apply(&self, app: &str, sets: &[(String, Value)]) -> Result<(), String> {
        if sets.is_empty() {
            return Ok(());
        }
        let _held = STORAGE_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = match self.read() {
            Stored::Read(f) => f,
            // Read-modify-write over a file we could not read would delete
            // every marker in it.
            Stored::Unreadable(why) => return Err(why),
        };
        let entry = file.apps.entry(app.to_string()).or_default();
        for (key, value) in sets {
            if !entry.values.contains_key(key) {
                entry.order.push(key.clone());
            }
            entry.values.insert(key.clone(), value.clone());
        }
        while entry.order.len() > MAX_STORAGE_KEYS {
            let oldest = entry.order.remove(0);
            entry.values.remove(&oldest);
        }
        let body = serde_json::to_string(&file)
            .map_err(|e| format!("could not encode the app-logic storage: {e}"))?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
        }
        // Temp-and-rename: a torn write here costs every marker in the file.
        //
        // The temp name carries the process id and a counter because the desktop
        // and the headless server can be pointed at one data directory: a shared
        // `.tmp` would let two writers interleave into one file and then rename
        // the mixture into place, which is precisely the corruption this dance
        // exists to prevent.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tmp = self.path.with_extension(format!(
            "json.tmp-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        // Written AND flushed before the rename. `fs::write` leaves the bytes in
        // the page cache, so a crash can land the rename with an empty file —
        // the module doc promises this is durable, so it has to actually be.
        let flush = (|| -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()
        })();
        if let Err(e) = flush {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("could not write {}: {e}", tmp.display()));
        }
        if let Err(e) = std::fs::rename(&tmp, &self.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("could not replace {}: {e}", self.path.display()));
        }
        Ok(())
    }
}

// --- running the scripts ---------------------------------------------------

/// Every script of every app, run against one event, with its effects performed.
///
/// Returns what happened, most useful first for a log: one [`Outcome`] per
/// effect attempted plus one for anything that went wrong before the effects
/// were reached. Never an error — this lane is best effort by construction, and
/// the caller has already dispatched the event locally.
pub fn handle_event(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    apps: &[AppEntry],
    storage: &StorageStore,
    envelope: &Value,
    node: Option<&crate::services::node_runtime::NodeHandle>,
    connector: &ConnectorFn,
) -> Vec<Outcome> {
    let mut out = Vec::new();
    let event_key = envelope
        .get("idempotencyKey")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut ran: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();

    for entry in apps {
        let Some(app) = entry.key(&spec.catalogue) else {
            // Without an id there is nowhere to keep this app's markers, so its
            // scripts would re-handle every event forever.
            out.push(Outcome::err(
                "(unnamed app)",
                "-",
                "lane",
                "the provider listed an app with no id or slug, so its scripts have nowhere \
                 to remember what they have already handled and were not run",
            ));
            continue;
        };
        let scripts = entry.event_scripts(&spec.catalogue, &spec.event_hook);
        if scripts.is_empty() {
            continue;
        }
        // Two entries for one app would run its scripts twice against the SAME
        // snapshot of the markers — neither run can see the other's — so both
        // would write the record and the dedup could not help.
        if !ran.insert(app) {
            out.push(Outcome::err(
                app,
                "-",
                "lane",
                "the provider listed this app more than once; the second copy's scripts were \
                 not run, because both copies would see the same markers and write the same \
                 record twice",
            ));
            continue;
        }

        let seen = match storage.values(app) {
            Ok(v) => v,
            // Not "start from empty": every script dedups on these keys, so
            // running them without would re-record everything still in the
            // window. Loud, and nothing is attempted.
            Err(e) => {
                out.push(Outcome::err(
                    app,
                    "-",
                    "storage",
                    format!(
                        "{e} — the scripts were not run, because without their markers they \
                         would write every event again"
                    ),
                ));
                continue;
            }
        };
        let ctx = json!({
            "event": envelope,
            "storage": seen,
        });
        let results = match run_scripts(&scripts, &ctx, node) {
            Ok(r) => r,
            Err(e) => {
                out.push(Outcome::err(app, "-", "lane", e));
                continue;
            }
        };

        for (script, result) in scripts.iter().zip(results) {
            let name = if script.id.is_empty() {
                "(unnamed script)"
            } else {
                script.id
            };
            let value = match result {
                Ok(v) => v,
                Err(e) => {
                    out.push(Outcome::err(app, name, "script", e));
                    continue;
                }
            };
            let effects = match effects_of(spec, &value) {
                Ok(list) => list,
                Err(e) => {
                    out.push(Outcome::err(app, name, "script", e));
                    continue;
                }
            };
            if effects.is_empty() {
                continue;
            }
            out.extend(apply_effects(
                account, spec, entry, app, name, effects, storage, &event_key, connector,
            ));
        }
    }
    out
}

/// The effects a script asked for, out of whatever it returned.
///
/// A script that asked for nothing is ordinary and says so with an empty list.
/// Everything else is a complaint: `.get()` on a string is `None`, so a script
/// that returned the wrong SHAPE entirely would otherwise be indistinguishable
/// from one that decided this event was not for it — and that is the difference
/// between a quiet install and a broken one.
fn effects_of<'a>(spec: &AppLogicSpec, value: &'a Value) -> Result<&'a [Value], String> {
    if value.is_null() {
        return Ok(&[]);
    }
    let Some(reply) = value.as_object() else {
        return Err(format!(
            "the script replied with a {} rather than an object, so nothing it asked for could \
             be read",
            type_name(value)
        ));
    };
    match reply.get(&spec.fields.effects_list) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(list)) => Ok(list),
        Some(other) => Err(format!(
            "the script listed its effects as a {} rather than a list, so none of them were \
             performed",
            type_name(other)
        )),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

/// Wrap every script into one graph and hand it to the bundled CLI.
///
/// One node per script, chained, so the CLI runs them all in a defined order in
/// ONE process — a process per script would put a Node start-up between every
/// line of a transcript.
///
/// Returns one result per script, in the same order, each either the value its
/// `run(ctx)` produced or the reason it threw. A script that fails is its own
/// failure and must not take its neighbours with it, so the throw is caught
/// INSIDE the generated node rather than out here.
fn run_scripts(
    scripts: &[ScriptRef],
    ctx: &Value,
    node: Option<&crate::services::node_runtime::NodeHandle>,
) -> Result<Vec<Result<Value, String>>, String> {
    let graph = build_graph(scripts, ctx)?;

    // Everything travels by file: the context holds whatever the event carried,
    // and argv is visible to every process lister on the machine.
    // Process id and a counter, not a timestamp: two runs in the same
    // millisecond would share a directory, and the second would read the
    // first's result file as its own.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let scratch = std::env::temp_dir().join(format!(
        "oaiy-app-logic-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)
        .map_err(|e| format!("could not create a working directory for the scripts: {e}"))?;
    let result = run_graph_in(&scratch, &graph, scripts.len(), node);
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

fn node_id(i: usize) -> String {
    format!("s{i}")
}

/// The graph: one code node per script, chained.
///
/// Chained rather than left as loose roots because an unconnected node is not
/// obviously a node the runner has to visit, and a script that is never visited
/// records nothing while reporting nothing either.
fn build_graph(scripts: &[ScriptRef], ctx: &Value) -> Result<Value, String> {
    let mut nodes = Vec::with_capacity(scripts.len());
    let mut edges = Vec::new();
    for (i, script) in scripts.iter().enumerate() {
        let code = script_node_code(script.source, ctx)?;
        nodes.push(json!({
            "id": node_id(i),
            "type": "logic_block",
            "position": { "x": (i as i64) * 240, "y": 0 },
            "data": { "code": code },
        }));
        if i > 0 {
            edges.push(json!({ "source": node_id(i - 1), "target": node_id(i) }));
        }
    }
    Ok(json!({ "nodes": nodes, "edges": edges }))
}

fn run_graph_in(
    scratch: &Path,
    graph: &Value,
    count: usize,
    node: Option<&crate::services::node_runtime::NodeHandle>,
) -> Result<Vec<Result<Value, String>>, String> {
    let graph_path = scratch.join("graph.json");
    let inputs_path = scratch.join("inputs.json");
    let out_path = scratch.join("result.json");
    std::fs::write(&graph_path, graph.to_string())
        .map_err(|e| format!("could not write {}: {e}", graph_path.display()))?;
    // The scripts read their context from inside their own node, so the run has
    // no inputs — but the CLI still expects a file where one is named.
    std::fs::write(&inputs_path, "{}")
        .map_err(|e| format!("could not write {}: {e}", inputs_path.display()))?;

    let outcome = run_flow_cli(
        CliRequest {
            flow_path: &graph_path,
            inputs_path: &inputs_path,
            out_path: &out_path,
            // These scripts speak to nobody: every effect they ask for is
            // performed out here, by this module, against the linked account.
            connector_path: None,
            timeout: RUN_TIMEOUT,
            node,
        },
        // Nothing cancels an event's scripts; the budget does.
        &|| false,
    );

    let report = match outcome {
        CliOutcome::Succeeded(v) => v,
        CliOutcome::Unreadable(why) => return Err(why),
        CliOutcome::Failed { exit_code, detail } => {
            return Err(if detail.is_empty() {
                format!("the scripts could not be run (CLI exit {exit_code})")
            } else {
                format!("the scripts could not be run (CLI exit {exit_code}): {detail}")
            })
        }
        CliOutcome::TimedOut => {
            return Err(format!(
                "the scripts exceeded their {}s budget and were killed",
                RUN_TIMEOUT.as_secs()
            ))
        }
        CliOutcome::Cancelled => return Err("the script run was cancelled".to_string()),
        CliOutcome::Unavailable {
            message, detail, ..
        } => {
            return Err(match detail {
                Some(d) => format!("{message}. {d}"),
                None => message,
            })
        }
    };

    parse_report(&report, count)
}

/// One result per script, from what the runner wrote.
///
/// Separate from the spawning so the shape the runner reports can be pinned
/// against the real thing without a process, and so a script that threw is told
/// apart from a script the runner never reached — those need different fixes.
fn parse_report(report: &Value, count: usize) -> Result<Vec<Result<Value, String>>, String> {
    if report.get("success").and_then(Value::as_bool) == Some(false) {
        let why = report
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("no reason given");
        return Err(format!("the scripts could not be run: {why}"));
    }

    let results = report.get("results").cloned().unwrap_or(Value::Null);
    Ok((0..count)
        .map(|i| {
            let slot = match results.get(node_id(i)) {
                None | Some(Value::Null) => {
                    return Err(
                        "the runner produced no result for this script, so nothing it asked \
                         for was performed"
                            .to_string(),
                    )
                }
                // A JSON string, deliberately — see `script_node_code` for why
                // an object would be silently rewritten on the way out.
                Some(Value::String(s)) => serde_json::from_str::<Value>(s).map_err(|e| {
                    format!("the script's answer was not readable JSON ({e}): {s}")
                })?,
                Some(other) => {
                    return Err(format!(
                        "the runner answered with a {} rather than the script's own report, so \
                         nothing it asked for was performed",
                        type_name(other)
                    ))
                }
            };
            if slot.get("ok").and_then(Value::as_bool) == Some(true) {
                Ok(slot.get("value").cloned().unwrap_or(Value::Null))
            } else {
                Err(format!(
                    "the script threw: {}",
                    slot.get("error").and_then(Value::as_str).unwrap_or("no message")
                ))
            }
        })
        .collect())
}

/// The code for one script's node.
///
/// # Why the source is encoded rather than pasted in
///
/// The runner's code-block compiler rewrites the text of a block line by line,
/// turning every `return X;` it can see into an assignment plus a `break`. That
/// is fine for a block someone typed into a node; it is fatal for a script that
/// defines a FUNCTION, because the rewritten `break` lands inside a function
/// body and the whole run dies at compile with "Illegal break statement" — the
/// first thing this lane hit against the real scripts.
///
/// So the source never appears as source. It is percent-encoded — every byte,
/// so the result is `%` and uppercase hex and nothing else — and decoded at run
/// time inside a `Function`. That also keeps the node's code free of `;` and of
/// newlines, which is what makes the compiler treat it as a single expression
/// and simply assign its value, instead of taking one of the rewriting paths.
///
/// The three checks at the end are not decoration: each one is a way the code
/// would silently take a different compiler branch and produce nothing.
///
/// # Why the answer comes back as a string
///
/// The runner "unboxes" a node's result: any object of one or two keys carrying
/// one of a handful of value-ish names — `text`, `result`, `url`, `output` — is
/// replaced by that key's contents, all the way down. That is helpful for a
/// node returning `{ result: … }` and catastrophic here: a record of two fields
/// one of which is called `text` arrives as the bare text, and the write that
/// follows is missing every other field. Caught against a real transcript
/// script, whose record is exactly that shape.
///
/// A string is not an object, so it is passed through untouched. The effects
/// therefore travel back as JSON text and are parsed out here.
fn script_node_code(source: &str, ctx: &Value) -> Result<String, String> {
    // The try/catch is INSIDE the encoded body, where the compiler cannot see
    // it, so a script that throws reports itself rather than failing the run.
    let body = format!(
        "try {{\n{source}\n;return JSON.stringify({{ ok: true, value: run(ctx) }});\n}} \
         catch (e) {{ return JSON.stringify({{ ok: false, error: String((e && e.message) || e) }}); }}"
    );
    let code = format!(
        "(new Function(\"ctx\",decodeURIComponent(\"{}\")))(JSON.parse(decodeURIComponent(\"{}\")))",
        percent_encode(&body),
        percent_encode(&ctx.to_string())
    );
    for (what, bad) in [("a semicolon", code.contains(';')), ("a newline", code.contains('\n'))] {
        if bad {
            return Err(format!(
                "the generated script node contains {what}, which would make the runner \
                 rewrite it instead of evaluating it"
            ));
        }
    }
    if contains_word(&code, "return") {
        return Err(
            "the generated script node contains a bare `return`, which the runner would \
             rewrite into an illegal break"
                .to_string(),
        );
    }
    Ok(code)
}

/// Every byte as `%XX`, uppercase.
///
/// Deliberately not "encode what has to be encoded": encoding EVERYTHING is what
/// guarantees the output is drawn from `%0-9A-F` alone, and therefore that it
/// can contain neither a semicolon, nor a newline, nor the lowercase word the
/// compiler rewrites on.
fn percent_encode(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.as_bytes() {
        out.push('%');
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

/// Whether `needle` appears in `haystack` at word boundaries.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$';
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let before_ok = start == 0 || !is_word(bytes[start - 1] as char);
        let after_ok = end == haystack.len() || !is_word(bytes[end] as char);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

// --- performing the effects ------------------------------------------------

/// One script's effects, in order.
///
/// Order is the contract, not a convenience: the scripts write the record FIRST
/// and the "already handled" marker AFTER, precisely so a write that fails is
/// not marked handled. This honours that — once a record write fails, the
/// markers that follow it in the SAME script are held back, so the next
/// delivery of the event tries again.
#[allow(clippy::too_many_arguments)]
fn apply_effects(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    entry: &AppEntry,
    app: &str,
    script: &str,
    effects: &[Value],
    storage: &StorageStore,
    event_key: &str,
    connector: &ConnectorFn,
) -> Vec<Outcome> {
    let f = &spec.fields;
    let mut out = Vec::new();
    let mut record_write_failed = false;

    for (index, effect) in effects.iter().enumerate() {
        let kind = effect
            .get(&f.kind)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let Some(operation) = spec
            .effects
            .iter()
            .find(|e| e.effect_type == kind)
            .map(|e| e.operation)
        else {
            let known: Vec<&str> = spec.effects.iter().map(|e| e.effect_type.as_str()).collect();
            out.push(Outcome::err(
                app,
                script,
                &kind,
                format!(
                    "the connector maps no such effect, so it was not performed; it maps {}",
                    known.join(", ")
                ),
            ));
            continue;
        };

        let result = match operation {
            AppLogicOperation::SubmitRecord => {
                submit(account, spec, entry, effect).map(|id| match id {
                    Some(id) => format!("created record {id}"),
                    None => "created a record".to_string(),
                })
            }
            AppLogicOperation::UpdateRecord => update(account, spec, entry, effect),
            AppLogicOperation::SetStorage => {
                let key = effect
                    .get(&f.storage_key)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if key.is_empty() {
                    Err("the effect names no key to store under".to_string())
                } else if record_write_failed {
                    // Deliberately not stored. The script's own order says a
                    // failed write must not be marked handled, and storing it
                    // anyway would lose the record permanently.
                    Err(format!(
                        "held back {key:?} because a record write earlier in this script \
                         failed; the event will be handled again"
                    ))
                } else {
                    // A marker with no value would read as absent on the next
                    // delivery, so an unstated one means "yes, seen".
                    let value = effect.get(&f.storage_value).cloned().unwrap_or(json!(true));
                    // On disk NOW, not batched to the end of the script. The
                    // window between a record that has been written and the
                    // marker that says so is exactly the window in which a
                    // crash duplicates the record, and the effects in between
                    // can each be a request with a thirty-second timeout.
                    storage
                        .apply(app, std::slice::from_ref(&(key.clone(), value)))
                        .map(|()| format!("remembered {key:?}"))
                        .map_err(|e| {
                            format!(
                                "{e} — {key:?} was not saved, so this event will be handled again"
                            )
                        })
                }
            }
            AppLogicOperation::Notify => {
                match effect.get(&f.message).and_then(Value::as_str) {
                    Some(m) if !m.trim().is_empty() => {
                        let level = effect
                            .get(&f.level)
                            .and_then(Value::as_str)
                            .unwrap_or("info");
                        Ok(format!("[{level}] {m}"))
                    }
                    _ => Err("the effect names no message to show".to_string()),
                }
            }
            AppLogicOperation::ConnectorRequest => {
                connector_request(spec, effect, event_key, script, index, connector)
            }
        };

        if matches!(
            operation,
            AppLogicOperation::SubmitRecord | AppLogicOperation::UpdateRecord
        ) && result.is_err()
        {
            record_write_failed = true;
        }
        out.push(Outcome {
            app: app.to_string(),
            script: script.to_string(),
            effect: kind,
            detail: result,
        });
    }
    out
}

/// The form this effect writes to.
///
/// A script names a form by the app's own portable key; the account's real id is
/// on the app's form list. Resolving to nothing is loud: it means the app is
/// installed differently from the version that published the script, and every
/// write it asks for would otherwise vanish.
fn form_id<'a>(spec: &AppLogicSpec, entry: &'a AppEntry, effect: &Value) -> Result<&'a str, String> {
    let key = effect
        .get(&spec.fields.form_key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if key.is_empty() {
        return Err("the effect names no form".to_string());
    }
    let forms = entry.forms(&spec.catalogue);
    forms
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, id)| *id)
        .ok_or_else(|| {
            let known: Vec<&str> = forms.iter().map(|(k, _)| *k).collect();
            format!(
                "this app has no form keyed {key:?}, so the write was not attempted; it has {}",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            )
        })
}

/// The record's values, as the effect carries them.
fn record<'a>(spec: &AppLogicSpec, effect: &'a Value) -> Result<&'a Value, String> {
    match effect.get(&spec.fields.record) {
        Some(v) if v.is_object() => Ok(v),
        Some(other) => Err(format!(
            "the effect carries its record as a {} rather than an object",
            type_name(other)
        )),
        None => Err("the effect carries no record".to_string()),
    }
}

fn submit(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    entry: &AppEntry,
    effect: &Value,
) -> Result<Option<String>, String> {
    let form = form_id(spec, entry, effect)?;
    let values = record(spec, effect)?;
    let url = super::oauth::join(
        &account.base_url,
        &spec.submit_path.replace("{formId}", form),
    );
    let resp = client()?
        .post(&url)
        .bearer_auth(&account.credential)
        .json(&json!({ &spec.fields.record: values }))
        .send()
        .map_err(|e| format!("could not reach the provider to create the record: {e}"))?;
    let status = resp.status();
    let payload: Value = resp.json().unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(format!(
            "the provider refused the record: HTTP {}: {}",
            status.as_u16(),
            refusal(&payload)
        ));
    }
    Ok(created_id(spec, &payload))
}

/// The id of a record the provider just created, wherever it chose to put it.
fn created_id(spec: &AppLogicSpec, payload: &Value) -> Option<String> {
    let direct = payload.get(&spec.fields.id).and_then(Value::as_str);
    let nested = payload
        .as_object()
        .into_iter()
        .flat_map(|o| o.values())
        .filter_map(|v| v.get(&spec.fields.id))
        .find_map(Value::as_str);
    direct.or(nested).map(str::to_string)
}

fn refusal(payload: &Value) -> String {
    payload
        .get("message")
        .or_else(|| payload.get("error"))
        .and_then(Value::as_str)
        .unwrap_or("no reason given")
        .to_string()
}

/// Update a record the effect names, either by id or by one of its own fields.
fn update(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    entry: &AppEntry,
    effect: &Value,
) -> Result<String, String> {
    let f = &spec.fields;
    let form = form_id(spec, entry, effect)?;
    let values = record(spec, effect)?;

    let named = effect
        .get(&f.record_id)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let target = match named {
        Some(id) => Some(id),
        None => {
            let block = effect.get(&f.match_block).ok_or_else(|| {
                "the effect names neither a record nor a field to find one by".to_string()
            })?;
            let field = block
                .get(&f.match_field)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "the effect's match names no field".to_string())?;
            let wanted = block
                .get(&f.match_value)
                .ok_or_else(|| "the effect's match names no value".to_string())?;
            // A blank one is a script whose key did not build — and because a
            // blank compares equal to an absent field, it would quietly match
            // the first record that never had one and overwrite THAT.
            if scalar_text(wanted).trim().is_empty() {
                return Err(format!(
                    "the effect's match wants {field:?} to equal nothing, which would find the \
                     first record that has no {field:?} at all and overwrite it"
                ));
            }
            find_record(account, spec, form, field, wanted)?
        }
    };

    let Some(id) = target else {
        // Not found. Upsert makes it, otherwise this is a correction for a
        // record that was never written — a clean no-op, said out loud rather
        // than swallowed.
        if effect.get(&f.upsert).and_then(Value::as_bool) == Some(true) {
            let created = submit(account, spec, entry, effect)?;
            return Ok(match created {
                Some(id) => format!("no record matched, so one was created ({id})"),
                None => "no record matched, so one was created".to_string(),
            });
        }
        return Ok("no record matched, and the effect does not create one — nothing changed"
            .to_string());
    };

    let url = super::oauth::join(
        &account.base_url,
        &spec
            .update_path
            .replace("{formId}", form)
            .replace("{id}", &id),
    );
    let resp = client()?
        .put(&url)
        .bearer_auth(&account.credential)
        .json(&json!({ &f.record: values }))
        .send()
        .map_err(|e| format!("could not reach the provider to update the record: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let payload: Value = resp.json().unwrap_or(Value::Null);
        return Err(format!(
            "the provider refused the update to {id}: HTTP {}: {}",
            status.as_u16(),
            refusal(&payload)
        ));
    }
    Ok(format!("updated record {id}"))
}

/// Find a record by one of its own fields.
///
/// The listing does not filter, so this reads the newest page and looks. That
/// is a real bound: a record older than the page cannot be found, and an effect
/// that creates on miss would then create a second one. Saying so when the page
/// was full is the difference between a duplicate somebody can explain and a
/// duplicate nobody can.
fn find_record(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    form: &str,
    field: &str,
    wanted: &Value,
) -> Result<Option<String>, String> {
    let path = spec.list_path.replace("{formId}", form);
    let sep = if path.contains('?') { '&' } else { '?' };
    let url = format!(
        "{}{sep}limit={}",
        super::oauth::join(&account.base_url, &path),
        spec.match_scan_limit
    );
    let resp = client()?
        .get(&url)
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the provider to find the record: {e}"))?;
    let status = resp.status();
    let payload: Value = resp.json().unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(format!(
            "the provider refused the record listing: HTTP {}: {}",
            status.as_u16(),
            refusal(&payload)
        ));
    }
    let items = payload
        .get(&spec.fields.items)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let found = items.iter().find_map(|row| {
        let held = row.get(&spec.fields.record)?.get(field)?;
        same_value(held, wanted).then(|| row.get(&spec.fields.id))?
    });
    if let Some(id) = found.and_then(Value::as_str) {
        return Ok(Some(id.to_string()));
    }
    if items.len() as u32 >= spec.match_scan_limit {
        log::info!(
            "app-logic: no record has {field}={} in the newest {} of this form, which is the \
             whole page — an older one would not have been seen",
            scalar_text(wanted),
            spec.match_scan_limit
        );
    }
    Ok(None)
}

/// Whether a stored value and a wanted one are the same.
///
/// Compared as text, because a script that builds a key by concatenation hands
/// over a string where the record may hold a number, and a strict comparison
/// would then miss every row and create a duplicate for each one.
fn same_value(held: &Value, wanted: &Value) -> bool {
    held == wanted || scalar_text(held) == scalar_text(wanted)
}

fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Ask a plugin connector on this desktop to do something.
///
/// The key is derived from the EVENT's own idempotency key and the effect's
/// place in its script — stable across redelivery, which is the entire point: a
/// plugin refuses a side-effecting command without one, and a fresh key on every
/// delivery would defeat the journal that stops the second delivery from
/// sending the message twice. An event with no key of its own gets no command at
/// all rather than a made-up one.
fn connector_request(
    spec: &AppLogicSpec,
    effect: &Value,
    event_key: &str,
    script: &str,
    index: usize,
    connector: &ConnectorFn,
) -> Result<String, String> {
    let f = &spec.fields;
    let id = effect
        .get(&f.connector_id)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "the effect names no connector".to_string())?;
    let command = effect
        .get(&f.command)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "the effect names no command".to_string())?;
    if event_key.trim().is_empty() {
        return Err(format!(
            "the event carries no idempotency key, so {command:?} could not be sent with one \
             that survives a redelivery; it was not sent"
        ));
    }
    let key = idempotency_key(event_key, script, index);
    let payload = effect.get(&f.payload).cloned().filter(|p| !p.is_null());
    connector(id, command, payload, &key).map(|_| format!("{id} performed {command}"))
}

/// The idempotency key for a connector command an effect asked for.
///
/// Derived only from things that are the same on a redelivery — the event's key,
/// which script asked, and where in that script. A generated one would be new
/// each time and would defeat the journal it is meant to key.
fn idempotency_key(event_key: &str, script: &str, index: usize) -> String {
    format!("app-logic-{event_key}-{script}-{index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AppLogicSpec {
        super::super::descriptor::find(std::path::Path::new("/nonexistent"), "formlogic")
            .unwrap()
            .app_logic
            .expect("the shipped connector must declare an app-logic lane")
    }

    fn account(base: String) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: base,
            credential: "flk_secret".into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("oaiy-test".into()),
        }
    }

    fn entry(forms: Value, scripts: Value) -> AppEntry {
        AppEntry(json!({
            "app": { "id": "app-1", "slug": "receptionist" },
            "customLogic": { "version": 1, "runtime": "quickjs", "scripts": scripts },
            "forms": forms,
        }))
    }

    fn store(tag: &str) -> (StorageStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "oaiy-app-logic-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("link")).unwrap();
        (StorageStore::open(&dir), dir)
    }

    fn no_connector() -> impl Fn(&str, &str, Option<Value>, &str) -> Result<Value, String> {
        |_, _, _, _| Err("no plugin host in this test".to_string())
    }

    // --- the descriptor is the whole vocabulary ----------------------------

    #[test]
    fn no_provider_name_or_effect_type_is_written_into_this_module() {
        // The single most important constraint, asserted against the source
        // rather than trusted. Test fixtures may name a provider; the executing
        // code may not — an effect type or a field name spelled here is a
        // second provider nobody can add without a release.
        let source = include_str!("app_logic.rs");
        let code_only = source.split("#[cfg(test)]\nmod tests").next().unwrap();
        for forbidden in [
            "formlogic",
            "FormLogic",
            "aokie",
            "submitResponse",
            "updateResponse",
            "answers",
            "formKey",
            "responseId",
            // Both spellings of each catalogue name. A `#[serde(rename_all =
            // "camelCase")]` struct puts `packFormId` on the wire while the
            // source says `pack_form_id`, so checking only the wire spelling
            // let the provider's whole vocabulary in through the back door —
            // which is exactly how it got in the first time.
            "packFormId",
            "pack_form_id",
            "customLogic",
            "custom_logic",
        ] {
            assert!(
                !code_only.contains(forbidden),
                "{forbidden:?} must not appear outside the tests"
            );
        }
    }

    #[test]
    fn the_shipped_connector_maps_every_operation_this_module_performs() {
        // A closed set is only safe if it is also COMPLETE: an operation the
        // module implements and the connector never names is dead code, and one
        // the connector names and the module cannot perform is a script whose
        // effects silently do nothing.
        let s = spec();
        let ops: Vec<AppLogicOperation> = s.effects.iter().map(|e| e.operation).collect();
        for wanted in [
            AppLogicOperation::SubmitRecord,
            AppLogicOperation::UpdateRecord,
            AppLogicOperation::SetStorage,
            AppLogicOperation::Notify,
            AppLogicOperation::ConnectorRequest,
        ] {
            assert!(ops.contains(&wanted), "{wanted:?} is not mapped");
        }
        assert!(s.submit_path.contains("{formId}"));
        assert!(s.update_path.contains("{formId}") && s.update_path.contains("{id}"));
        assert!(!s.event_hook.trim().is_empty());
    }

    // --- the generated node ------------------------------------------------

    #[test]
    fn a_script_that_defines_a_function_survives_the_code_block_compiler() {
        // The failure this whole encoding exists for. The compiler rewrites
        // every `return X;` it can SEE into an assignment plus `break`, so a
        // script defining `function run(ctx) { … return {}; }` compiled to a
        // `break` inside a function body and the run died outright with
        // "Illegal break statement" — nothing ran, for any app.
        let code = script_node_code(
            "function run(ctx) { if (!ctx.event) return {}; return { effects: [] }; }",
            &json!({ "event": { "name": "x" }, "storage": {} }),
        )
        .unwrap();
        assert!(!contains_word(&code, "return"), "{code}");
        assert!(!code.contains(';'), "a semicolon sends the compiler down another branch");
        assert!(!code.contains('\n'));
        // …and the payload really is only the safe alphabet.
        let blobs: Vec<&str> = code.split('"').filter(|s| s.starts_with('%')).collect();
        assert_eq!(blobs.len(), 2, "the source and the context, both encoded");
        for blob in blobs {
            assert!(
                blob.chars().all(|c| c == '%' || c.is_ascii_digit() || ('A'..='F').contains(&c)),
                "an unencoded byte leaked into {blob}"
            );
        }
    }

    #[test]
    fn a_context_full_of_hazards_still_produces_one_expression() {
        // Real transcript text carries semicolons, quotes, newlines, backslashes
        // and non-ASCII. Any one of them pasted in raw would either break the
        // string or push the compiler onto a rewriting path.
        let code = script_node_code(
            "function run(ctx) { return { effects: [] }; }",
            &json!({
                "event": { "data": { "text": "he said \"stop\"; then\nleft — 90% sure\\done" } },
                "storage": { "seen-\u{1f600}": 1 },
            }),
        )
        .unwrap();
        assert!(!code.contains(';') && !code.contains('\n'));
        assert!(!contains_word(&code, "return"));
    }

    #[test]
    fn word_matching_does_not_fire_on_a_substring() {
        // The guard must not reject a legitimate node, and must not miss a real
        // one. `returned` is not `return`; `x=return` is.
        assert!(contains_word("a return b", "return"));
        assert!(contains_word("return", "return"));
        assert!(contains_word("}return{", "return"));
        assert!(!contains_word("returned", "return"));
        assert!(!contains_word("prereturn", "return"));
        assert!(!contains_word("no_return_here", "return"));
    }

    // --- effects: the closed set and the loud refusals ---------------------

    #[test]
    fn an_effect_the_connector_does_not_map_is_refused_by_name() {
        // Never a silent skip: an unmapped effect looks exactly like a working
        // install that records nothing.
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (storage, dir) = store("unmapped");
        let out = apply_effects(
            &account("http://provider.invalid".into()),
            &s,
            &e,
            "app-1",
            "script-1",
            &[json!({ "type": "records.obliterate" })],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert_eq!(out.len(), 1);
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("maps no such effect"), "{why}");
        assert!(why.contains("storage.set"), "it must say what IS mapped: {why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_form_key_that_matches_nothing_is_refused_before_any_request() {
        // The app is installed differently from the version that published the
        // script. Every write it asks for would otherwise vanish.
        let s = spec();
        // Note the blank key: a real account has one, and a blank effect key
        // must not silently resolve to it.
        let e = entry(
            json!([{ "id": "form-1", "packFormId": "calls" }, { "id": "form-x", "packFormId": "" }]),
            json!([]),
        );
        let (storage, dir) = store("noform");
        let out = apply_effects(
            &account("http://provider.invalid".into()),
            &s,
            &e,
            "app-1",
            "script-1",
            &[
                json!({ "type": "formlogic.submitResponse", "formKey": "ledger", "answers": {} }),
                json!({ "type": "formlogic.submitResponse", "answers": {} }),
            ],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert!(out[0].detail.clone().unwrap_err().contains("no form keyed"));
        assert!(out[0].detail.clone().unwrap_err().contains("calls"));
        assert!(out[1].detail.clone().unwrap_err().contains("names no form"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_marker_is_held_back_when_the_write_before_it_failed() {
        // The scripts write the record first and the marker after, exactly so a
        // failed write is not marked handled. Storing it anyway loses the record
        // for good — no redelivery would ever write it.
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (storage, dir) = store("heldback");
        let out = apply_effects(
            // Nothing listening: the write fails at the connection.
            &account("http://127.0.0.1:1".into()),
            &s,
            &e,
            "app-1",
            "script-1",
            &[
                json!({ "type": "formlogic.submitResponse", "formKey": "calls", "answers": { "a": 1 } }),
                json!({ "type": "storage.set", "key": "seen-1", "value": 1 }),
            ],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert!(out[0].detail.is_err(), "the write must fail in this test");
        let why = out[1].detail.clone().unwrap_err();
        assert!(why.contains("held back"), "{why}");
        assert!(
            storage.values("app-1").unwrap().is_empty(),
            "nothing may be remembered when the write it belongs to failed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_marker_is_on_disk_before_the_call_returns() {
        // Losing this file replays every transcript line the account has. It is
        // written now, not at some later convenient moment.
        let s = spec();
        let e = entry(json!([]), json!([]));
        let (storage, dir) = store("marker");
        let out = apply_effects(
            &account("http://provider.invalid".into()),
            &s,
            &e,
            "app-1",
            "script-1",
            &[
                json!({ "type": "storage.set", "key": "seen-1", "value": 1 }),
                // No value: an unstated marker still has to read as "seen" next
                // time, or the dedup silently stops working.
                json!({ "type": "storage.set", "key": "seen-2" }),
                json!({ "type": "storage.set", "key": "  " }),
            ],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert!(out[0].detail.is_ok() && out[1].detail.is_ok());
        assert!(out[2].detail.clone().unwrap_err().contains("names no key"));

        let raw = std::fs::read_to_string(storage.path()).expect("the file must exist already");
        assert!(raw.contains("seen-1"), "{raw}");
        let values = storage.values("app-1").unwrap();
        assert_eq!(values.get("seen-1"), Some(&json!(1)));
        assert_eq!(values.get("seen-2"), Some(&json!(true)));
        // A different app must not see it — the file is keyed by app.
        assert!(storage.values("app-2").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn storage_is_bounded_and_drops_the_oldest_first() {
        // Unbounded, this grows by one key per transcript line forever. Evicting
        // an arbitrary key instead of the oldest would replay an arbitrary event.
        let (storage, dir) = store("bounded");
        let sets: Vec<(String, Value)> = (0..MAX_STORAGE_KEYS + 5)
            .map(|i| (format!("k{i}"), json!(i)))
            .collect();
        storage.apply("app-1", &sets).unwrap();
        let values = storage.values("app-1").unwrap();
        assert_eq!(values.len(), MAX_STORAGE_KEYS);
        assert!(values.get("k0").is_none(), "the oldest must be the one dropped");
        assert!(values.get(&format!("k{}", MAX_STORAGE_KEYS + 4)).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_storage_file_is_empty_but_an_unreadable_one_is_never_overwritten() {
        // The distinction the whole dedup rests on. "Absent" means a fresh
        // install; "would not read" means the markers are still in there and
        // this process cannot see them. Treating the second as the first —
        // which `read().ok().unwrap_or_default()` does — writes an empty map
        // over the only copy, and every event still inside the window is
        // recorded a second time.
        let (storage, dir) = store("torn");
        assert!(storage.values("app-1").unwrap().is_empty(), "absent is empty");

        storage.apply("app-1", &[("k".into(), json!(1))]).unwrap();
        let good = std::fs::read_to_string(storage.path()).unwrap();

        std::fs::write(storage.path(), "{ not json").unwrap();
        let why = storage.values("app-1").unwrap_err();
        assert!(why.contains("not readable JSON"), "{why}");
        let refused = storage.apply("app-1", &[("k2".into(), json!(2))]).unwrap_err();
        assert!(refused.contains("not readable JSON"), "{refused}");
        assert_eq!(
            std::fs::read_to_string(storage.path()).unwrap(),
            "{ not json",
            "the file must be left exactly as found, for a human to salvage"
        );

        // Put it back and the lane resumes with everything still in place.
        std::fs::write(storage.path(), good).unwrap();
        storage.apply("app-1", &[("k2".into(), json!(2))]).unwrap();
        let values = storage.values("app-1").unwrap();
        assert_eq!(values.get("k"), Some(&json!(1)), "the earlier marker survived");
        assert_eq!(values.get("k2"), Some(&json!(2)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_writers_on_one_file_lose_nothing_and_leave_nothing_behind() {
        // The store is opened FRESH on every event, so a lock owned by an
        // instance serialises nothing: two writers would read the same file,
        // add their own key, and write back — the second silently deleting the
        // first's marker, which is one transcript line recorded twice. A shared
        // temp name is the other half of the same bug: two writers streaming
        // into one temp file rename the mixture into place.
        let (_, dir) = store("concurrent");
        std::thread::scope(|s| {
            for w in 0..6 {
                let dir = dir.clone();
                s.spawn(move || {
                    let store = StorageStore::open(&dir);
                    for i in 0..15 {
                        store
                            .apply("app-1", &[(format!("w{w}-k{i}"), json!(i))])
                            .expect("every write must land");
                    }
                });
            }
        });
        let store = StorageStore::open(&dir);
        let values = store.values("app-1").expect("the file must still be readable");
        assert_eq!(values.len(), 6 * 15, "no writer's markers may be lost");

        let strays: Vec<String> = std::fs::read_dir(dir.join("link"))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(strays.is_empty(), "no temp file may be left behind: {strays:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- idempotency -------------------------------------------------------

    #[test]
    fn a_connector_command_keys_on_the_events_own_key_so_a_retry_cannot_act_twice() {
        // A plugin refuses a side-effecting command with no key, and a FRESH key
        // on every delivery would defeat the journal — the second delivery would
        // send the message again.
        let a = idempotency_key("evt-7", "sms-reply", 2);
        assert_eq!(a, idempotency_key("evt-7", "sms-reply", 2));
        assert!(a.contains("evt-7"));
        assert_ne!(a, idempotency_key("evt-8", "sms-reply", 2));
        assert_ne!(a, idempotency_key("evt-7", "sms-reply", 3));
        assert_ne!(a, idempotency_key("evt-7", "other", 2));
        assert!(!a.trim().is_empty());
    }

    #[test]
    fn an_event_with_no_key_of_its_own_sends_no_command_at_all() {
        // Rather than inventing one. A made-up key is a key that changes on
        // every redelivery, which is worse than refusing: it would dial twice.
        let s = spec();
        let sent = std::sync::Mutex::new(Vec::<String>::new());
        let connector = |_: &str, cmd: &str, _: Option<Value>, key: &str| {
            sent.lock().unwrap().push(format!("{cmd}/{key}"));
            Ok(json!({ "ok": true }))
        };
        let refused = connector_request(
            &s,
            &json!({ "type": "connector.request", "connectorId": "aokie", "command": "sms.send" }),
            "   ",
            "script-1",
            0,
            &connector,
        )
        .unwrap_err();
        assert!(refused.contains("idempotency key"), "{refused}");
        assert!(sent.lock().unwrap().is_empty(), "nothing may be sent");

        // With a key it goes, carrying one derived from the event.
        connector_request(
            &s,
            &json!({ "type": "connector.request", "connectorId": "aokie", "command": "sms.send", "payload": { "to": "+1" } }),
            "evt-9",
            "script-1",
            0,
            &connector,
        )
        .unwrap();
        assert!(sent.lock().unwrap()[0].contains("evt-9"));
    }

    // --- the sequence against a stub provider ------------------------------

    /// A stub provider that answers a listing, a create and an update.
    ///
    /// Hand-rolled HTTP like the relay's and the flow runner's tests. The
    /// assertions are about the sequence and the bodies, not about a server.
    fn stub(
        listing: &'static str,
        write_status: &'static str,
    ) -> (String, std::sync::mpsc::Receiver<(String, String)>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { return };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(head_end) = text.find("\r\n\r\n") {
                        let len: usize = text
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length: ")
                                    .or_else(|| l.strip_prefix("Content-Length: "))
                            })
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        if raw.len() >= head_end + 4 + len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&raw).to_string();
                let line = text.lines().next().unwrap_or_default().to_string();
                let (status, reply) = if line.starts_with("GET ") {
                    ("200 OK", listing.to_string())
                } else {
                    (write_status, r#"{"id":"new-1"}"#.to_string())
                };
                let _ = tx.send((line, text));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.flush();
            }
        });
        (format!("http://127.0.0.1:{port}"), rx)
    }

    const ONE_ROW: &str =
        r#"{"responses":[{"id":"resp-1","answers":{"call_id":"c1","turn_key":"c1:3"}}]}"#;
    const NO_ROWS: &str = r#"{"responses":[]}"#;

    #[test]
    fn a_create_goes_to_the_forms_own_id_and_carries_the_record() {
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (base, rx) = stub(NO_ROWS, "201 Created");
        let (storage, dir) = store("create");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[json!({ "type": "formlogic.submitResponse", "formKey": "calls", "answers": { "call_id": "c9" } })],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert!(out[0].detail.is_ok(), "{:?}", out[0].detail);

        let (line, body) = rx.recv().unwrap();
        // The PACK key never reaches the wire; the account's form id does.
        assert!(line.starts_with("POST /api/v1/forms/form-1/responses "), "{line}");
        assert!(!line.contains("calls"), "the portable key must be resolved: {line}");
        assert!(body.contains("flk_secret"), "the credential travels as a bearer");
        assert!(!line.contains("flk_secret"), "…and never in the URL: {line}");
        assert!(body.contains("\"call_id\":\"c9\""), "{body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_update_matched_by_a_field_reads_the_listing_then_writes_that_record() {
        let s = spec();
        let e = entry(
            json!([{ "id": "form-2", "packFormId": "transcript-turns" }]),
            json!([]),
        );
        let (base, rx) = stub(ONE_ROW, "200 OK");
        let (storage, dir) = store("match");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[json!({
                "type": "formlogic.updateResponse",
                "formKey": "transcript-turns",
                "match": { "field": "turn_key", "value": "c1:3" },
                "answers": { "text": "corrected" },
            })],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert_eq!(out[0].detail.clone().unwrap(), "updated record resp-1");

        let (list, _) = rx.recv().unwrap();
        assert!(list.starts_with("GET /api/v1/forms/form-2/responses?limit="), "{list}");
        let (put, body) = rx.recv().unwrap();
        assert!(put.starts_with("PUT /api/v1/forms/form-2/responses/resp-1 "), "{put}");
        assert!(body.contains("corrected"), "{body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_match_that_finds_nothing_creates_only_when_the_effect_says_so() {
        // Both directions are wrong in their own way: upserting when the script
        // did not ask mints a duplicate row for every correction, and refusing
        // to upsert when it did asked loses a call whose first write never landed.
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (base, rx) = stub(NO_ROWS, "201 Created");
        let (storage, dir) = store("upsert");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[
                json!({
                    "type": "formlogic.updateResponse", "formKey": "calls",
                    "match": { "field": "call_id", "value": "c9" },
                    "answers": { "status": "answered" },
                }),
                json!({
                    "type": "formlogic.updateResponse", "formKey": "calls", "upsert": true,
                    "match": { "field": "call_id", "value": "c9" },
                    "answers": { "call_id": "c9", "status": "completed" },
                }),
            ],
            &storage,
            "evt-1",
            &no_connector(),
        );
        assert!(out[0].detail.clone().unwrap().contains("nothing changed"));
        assert!(out[1].detail.clone().unwrap().contains("was created"));

        let mut writes = 0;
        while let Ok((line, _)) = rx.recv_timeout(Duration::from_millis(400)) {
            if !line.starts_with("GET ") {
                writes += 1;
            }
        }
        assert_eq!(writes, 1, "only the upserting effect may write");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_match_on_nothing_overwrites_nothing() {
        // A key that failed to build arrives as "". Blank compares equal to a
        // field that was never set, so the update would land on the first
        // unrelated record that happens to be missing it — a corruption with
        // a perfectly successful-looking outcome.
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (base, rx) = stub(ONE_ROW, "200 OK");
        let (storage, dir) = store("blankmatch");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[json!({
                "type": "formlogic.updateResponse", "formKey": "calls", "upsert": true,
                "match": { "field": "call_id", "value": "" },
                "answers": { "status": "answered" },
            })],
            &storage,
            "evt-1",
            &no_connector(),
        );
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("equal nothing"), "{why}");
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "nothing may be read or written on a match that cannot mean anything"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_refused_write_is_reported_with_the_providers_own_reason() {
        let s = spec();
        let e = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let (base, _rx) = stub(NO_ROWS, "422 Unprocessable Entity");
        let (storage, dir) = store("refused");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[json!({ "type": "formlogic.submitResponse", "formKey": "calls", "answers": {} })],
            &storage,
            "evt-1",
            &no_connector(),
        );
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("422"), "the code is the diagnosis: {why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_match_compares_a_number_and_its_text_as_the_same_value() {
        // A script builds its key by concatenation and hands over a string where
        // the record holds a number. A strict comparison would miss every row
        // and — with upsert — mint a duplicate for each one.
        assert!(same_value(&json!(3), &json!("3")));
        assert!(same_value(&json!("c1:3"), &json!("c1:3")));
        assert!(!same_value(&json!("c1:3"), &json!("c1:4")));
        assert!(same_value(&json!(null), &json!("")));
    }

    // --- the wire shape the provider actually sends ------------------------

    #[test]
    fn the_catalogue_parses_from_the_providers_actual_wire_shape() {
        // Captured live. Scripts are a LIST under customLogic, each with its own
        // hook — and a hook other than the event one must not run on an event,
        // or this desktop performs work at a moment nobody asked for.
        let raw = json!({
            "apps": [{
                "app": { "id": "e2e19da6", "slug": "aokie-receptionist-78a80a", "name": "Reception" },
                "customLogic": {
                    "version": 1,
                    "runtime": "quickjs",
                    "strictPermissions": true,
                    "connector": { "manifest": {}, "demoDriver": "// …" },
                    "permissions": ["formlogic.responses.write"],
                    "scripts": [
                        { "id": "on-call", "hook": "onConnectorEvent", "runtime": "quickjs",
                          "description": "…", "source": "function run(ctx) { return {}; }" },
                        { "id": "on-install", "hook": "onInstall", "runtime": "quickjs",
                          "source": "function run(ctx) { return {}; }" },
                        { "id": "blank", "hook": "onConnectorEvent", "source": "   " }
                    ]
                },
                "forms": [
                    { "id": "fde1f7f6", "name": "Calls", "displayName": "Calls", "packFormId": "calls" },
                    { "id": "5bda9321", "name": "Untitled", "packFormId": "" }
                ],
                "connectors": ["aokie"]
            }]
        });
        let c = &spec().catalogue;
        let apps = raw["apps"].as_array().unwrap().clone();
        let e = AppEntry(apps[0].clone());
        assert_eq!(e.key(c), Some("e2e19da6"));
        let scripts = e.event_scripts(c, "onConnectorEvent");
        assert_eq!(scripts.len(), 1, "only the event hook, and not the empty one");
        assert_eq!(scripts[0].id, "on-call");
        // The blank key is not a form a script can name — offering it would let
        // an effect that names no form at all resolve to a real one.
        assert_eq!(e.forms(c), vec![("calls", "fde1f7f6")]);

        // An app with no logic at all is ordinary, not a failure.
        let bare = AppEntry(json!({ "app": { "id": "x" } }));
        assert!(bare.event_scripts(c, "onConnectorEvent").is_empty());
        assert_eq!(bare.key(c), Some("x"));
        // …and the slug stands in when there is no id.
        assert_eq!(
            AppEntry(json!({ "app": { "id": "  ", "slug": "s" } })).key(c),
            Some("s")
        );
    }

    #[test]
    fn a_reply_this_descriptor_does_not_describe_is_a_failure_not_an_empty_account() {
        // The two look identical from here and could not be more different: one
        // is an account with nothing installed, the other is a lane that will
        // never record anything for anybody. Only a shape check tells them
        // apart, and without it the symptom is an empty transcript and a
        // console with nothing in it.
        let s = spec();
        let (base, _rx) = stub(r#"{"data":[]}"#, "200 OK");
        let why = fetch(&account(base), &s).unwrap_err();
        assert!(why.contains("does not describe it"), "{why}");

        let (base, _rx) = stub(r#"{"apps":[]}"#, "200 OK");
        assert!(fetch(&account(base), &s).unwrap().is_empty(), "an empty account is fine");
    }

    #[test]
    fn an_app_with_no_id_is_refused_rather_than_run_without_a_memory() {
        // Its markers would have nowhere to live, so every event it handled
        // would be handled again on the next delivery — forever.
        let s = spec();
        let apps = vec![AppEntry(json!({
            "app": {},
            "customLogic": { "scripts": [
                { "id": "s", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("noid");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            None,
            &no_connector(),
        );
        assert_eq!(out.len(), 1);
        assert!(out[0].detail.clone().unwrap_err().contains("nowhere to remember"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_report_tells_a_script_that_threw_from_one_the_runner_never_reached() {
        // Two different fixes: the first is the script author's, the second is
        // this desktop's. Flattening them would send everyone to the wrong one.
        let report = json!({
            "success": true,
            "results": {
                "s0": r#"{"ok":true,"value":{"effects":[]}}"#,
                "s1": r#"{"ok":false,"error":"d is not defined"}"#,
                "s3": { "ok": true, "value": {} },
            }
        });
        let parsed = parse_report(&report, 4).unwrap();
        assert_eq!(parsed[0].clone().unwrap()["effects"], json!([]));
        assert!(parsed[1].clone().unwrap_err().contains("d is not defined"));
        assert!(parsed[2].clone().unwrap_err().contains("no result"));
        // An OBJECT is the shape the runner produces after it has rewritten the
        // answer, which is exactly the corruption the string exists to dodge —
        // so it is refused rather than half-read.
        assert!(parsed[3].clone().unwrap_err().contains("rather than the script's own report"));

        // A run that failed outright is one failure, not N.
        let dead = json!({ "success": false, "error": "Illegal break statement" });
        assert!(parse_report(&dead, 2)
            .unwrap_err()
            .contains("Illegal break statement"));
    }

    // --- against the real runner -------------------------------------------

    /// The bundled CLI, if this checkout has one staged, and a Node to run it.
    ///
    /// Skipped rather than failed when either is missing: the test below is
    /// about whether the generated graph SURVIVES the real compiler, and a box
    /// without Node cannot answer that either way.
    fn staged_runner() -> Option<(PathBuf, PathBuf)> {
        let cli = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("cli")
            .join("oaiy.mjs");
        if !cli.is_file() {
            return None;
        }
        let finder = if cfg!(windows) { "where" } else { "which" };
        let out = std::process::Command::new(finder).arg("node").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let node = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(PathBuf::from)?;
        Some((node, cli))
    }

    #[test]
    fn the_real_scripts_run_in_the_real_runner_and_return_their_effects() {
        // The test that matters. Everything else here is about shapes; this is
        // the one that would have caught "Illegal break statement", which broke
        // EVERY script of EVERY app and produced an empty transcript with a
        // successful-looking desktop.
        //
        // The scripts below are the shipped ones, trimmed to the parts that
        // decide: a hook that must not fire on the wrong event, a dedup that
        // must read the storage it was handed, and a throw that must not take
        // its neighbours down.
        let Some((node, cli)) = staged_runner() else {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        };
        let raw = json!([
            { "id": "turn", "hook": "onConnectorEvent", "source":
              "function run(ctx) {\n  var ev = ctx.event || {};\n  var d = ev.data || {};\n  if (ev.name !== 'aokie.call.turn.final') return {};\n  var key = 'seen-' + String(ev.idempotencyKey);\n  if (ctx.storage && ctx.storage[key]) return {};\n  var text = String(d.text || '').trim();\n  if (text === '') return {};\n  return { effects: [\n    { type: 'formlogic.submitResponse', formKey: 'transcript-turns', answers: { text: text, turn_key: String(d.callId) + ':' + Number(d.turn || 0) } },\n    { type: 'storage.set', key: key, value: 1 }\n  ] };\n}" },
            { "id": "other-event", "hook": "onConnectorEvent", "source":
              "function run(ctx) {\n  var ev = ctx.event || {};\n  if (ev.name !== 'aokie.sms.received') return {};\n  return { effects: [{ type: 'ui.toast', level: 'info', message: 'New SMS' }] };\n}" },
            { "id": "throws", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return nothing.at.all; }" },
            { "id": "after-the-throw", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { effects: [{ type: 'ui.toast', message: 'still here' }] }; }" },
            { "id": "unboxable", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { effects: [{ type: 'formlogic.submitResponse', formKey: 'f', answers: { text: 'body', url: 'https://example.test' } }] }; }" }
        ]);
        // Through the same reader the lane uses, so this really is the graph an
        // event would build.
        let app = AppEntry(json!({ "app": { "id": "app-1" }, "customLogic": { "scripts": raw } }));
        let s = spec();
        let refs = app.event_scripts(&s.catalogue, &s.event_hook);
        assert_eq!(refs.len(), 5, "every script here runs on the event hook");

        let run = |ctx: Value| -> Vec<Result<Value, String>> {
            let dir = std::env::temp_dir().join(format!(
                "oaiy-app-logic-e2e-{}-{}",
                std::process::id(),
                ctx.to_string().len()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let graph = build_graph(&refs, &ctx).unwrap();
            std::fs::write(dir.join("graph.json"), graph.to_string()).unwrap();
            std::fs::write(dir.join("inputs.json"), "{}").unwrap();
            let status = std::process::Command::new(&node)
                .arg(&cli)
                .arg("run")
                .arg(dir.join("graph.json"))
                .arg("--inputs")
                .arg(dir.join("inputs.json"))
                .arg("-o")
                .arg(dir.join("result.json"))
                .arg("--timeout")
                .arg("60")
                .arg("--quiet")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("the runner must start");
            let report: Value = std::fs::read_to_string(dir.join("result.json"))
                .ok()
                .and_then(|b| serde_json::from_str(&b).ok())
                .unwrap_or_else(|| panic!("the runner wrote no readable result (exit {status})"));
            let parsed = parse_report(&report, refs.len()).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            parsed
        };

        let fresh = run(json!({
            "event": {
                "name": "aokie.call.turn.final",
                "idempotencyKey": "evt-1",
                "data": { "callId": "c1", "turn": 3, "text": "hello; \"there\" — ok\\done" },
            },
            "storage": {},
        }));
        let effects = fresh[0].clone().unwrap();
        assert_eq!(effects["effects"][0]["formKey"], "transcript-turns");
        // The hazardous text survives the encoding intact — semicolons, quotes,
        // a backslash and a multi-byte dash, all of which would otherwise end
        // the string or send the compiler down a rewriting path.
        assert_eq!(effects["effects"][0]["answers"]["text"], "hello; \"there\" — ok\\done");
        assert_eq!(effects["effects"][0]["answers"]["turn_key"], "c1:3");
        assert_eq!(effects["effects"][1]["key"], "seen-evt-1");
        // A script for a different event returns nothing rather than firing.
        assert_eq!(fresh[1].clone().unwrap(), json!({}));
        // A script that throws reports itself…
        assert!(fresh[2].clone().unwrap_err().contains("threw"));
        // …and does not take the next one with it, which is why the catch lives
        // inside the node instead of around the run.
        assert_eq!(fresh[3].clone().unwrap()["effects"][0]["message"], "still here");
        // The unboxing hazard, pinned. A record of two fields one of which is
        // called `text` used to come back as the bare text — the write that
        // followed would have been missing every other field, and nothing
        // anywhere would have said so.
        let boxed = fresh[4].clone().unwrap();
        assert_eq!(boxed["effects"][0]["answers"]["text"], "body");
        assert_eq!(boxed["effects"][0]["answers"]["url"], "https://example.test");

        // The dedup: the SAME event with the marker already stored does nothing
        // at all. Without this the storage file would be decoration and every
        // redelivery would write the transcript line again.
        let seen = run(json!({
            "event": {
                "name": "aokie.call.turn.final",
                "idempotencyKey": "evt-1",
                "data": { "callId": "c1", "turn": 3, "text": "hello" },
            },
            "storage": { "seen-evt-1": 1 },
        }));
        assert_eq!(seen[0].clone().unwrap(), json!({}), "a stored marker must suppress the write");
    }

    #[test]
    fn a_script_whose_reply_is_the_wrong_shape_is_complained_about_not_skipped() {
        // `.get("effects")` on a string is `None`, which reads exactly like a
        // script that decided this event was not for it. One is ordinary and
        // one means the app records nothing at all, so they cannot look alike.
        let s = spec();
        assert!(effects_of(&s, &json!({})).unwrap().is_empty());
        assert!(effects_of(&s, &Value::Null).unwrap().is_empty());
        assert!(effects_of(&s, &json!({ "effects": null })).unwrap().is_empty());
        assert_eq!(effects_of(&s, &json!({ "effects": [{ "type": "x" }] })).unwrap().len(), 1);

        let why = effects_of(&s, &json!("done")).unwrap_err();
        assert!(why.contains("rather than an object"), "{why}");
        let why = effects_of(&s, &json!({ "effects": { "type": "x" } })).unwrap_err();
        assert!(why.contains("rather than a list"), "{why}");
    }

    #[test]
    fn an_app_listed_twice_does_not_get_its_scripts_run_twice() {
        // Both copies are handed the SAME snapshot of the markers — neither can
        // see what the other stored — so a dedup that works perfectly would
        // still write the record twice for one event.
        let s = spec();
        let one = json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "s", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
            ] },
            "forms": []
        });
        let apps = vec![AppEntry(one.clone()), AppEntry(one)];
        let (storage, dir) = store("twice");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            // No Node: the first copy fails at the runner, which is fine — the
            // assertion is about the SECOND one never getting that far.
            None,
            &no_connector(),
        );
        let duplicate = out
            .iter()
            .find(|o| o.detail.as_ref().err().is_some_and(|e| e.contains("more than once")));
        assert!(duplicate.is_some(), "{out:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn markers_that_cannot_be_read_stop_the_scripts_instead_of_replaying_them() {
        // The scripts dedup on these keys. Handing them "you have seen nothing"
        // because the file would not read makes every one of them act again —
        // the entire recorded history, rewritten, on one unlucky read.
        let s = spec();
        let apps = vec![AppEntry(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "s", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("unreadable");
        std::fs::write(storage.path(), "{ not json").unwrap();
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            None,
            &no_connector(),
        );
        assert_eq!(out.len(), 1);
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("were not run"), "{why}");
        assert_eq!(
            std::fs::read_to_string(storage.path()).unwrap(),
            "{ not json",
            "and the file is still there to be salvaged"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_app_with_no_event_scripts_costs_nothing() {
        // No runner is started at all — this runs on every plugin event, and an
        // account whose apps carry no event logic must not pay a process for it.
        let s = spec();
        let apps = vec![AppEntry(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [{ "id": "s", "hook": "onInstall", "source": "x" }] },
            "forms": []
        }))];
        let (storage, dir) = store("noscripts");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            None,
            &no_connector(),
        );
        assert!(out.is_empty());
        assert!(!storage.path().exists(), "nothing was written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
