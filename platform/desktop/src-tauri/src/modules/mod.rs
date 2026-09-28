//! Modules: the parts of OAIY that are there only while a plugin provides them.
//!
//! The phone (calls and texts, the Front desk's agents) and the calendar (the
//! phone receptionist's diary) come with a plugin: without one, OAIY shows no
//! Front desk, no phone and no calendar. Everything else is core.
//!
//! ```text
//!   plugin manifests ──► claims()  ─┐
//!   plugin registry  ──► resolve() ─┴─► Snapshot {revision, modules, warnings}
//!                                          ├── is_enabled("phone")   (routes, leases, the plugin host)
//!                                          ├── GET /api/modules       (ETag = revision)
//!                                          └── GET /api/modules/events (server-sent, on each change)
//! ```
//!
//! # When a module is on
//!
//! A module is enabled while an installed plugin claims it, that plugin's
//! manifest loads, and the person has not turned the plugin off. Whether the
//! plugin is running does not matter: a crashed or stopped provider keeps its
//! modules on (its `provider.state` says how it is), so a restart does not
//! make the Front desk and the calendar flicker away and back. A turned-off
//! plugin provides nothing, and the module says why.
//!
//! A plugin claims modules in its manifest's `modules` section
//! (`{"provides": ["phone", "calendar"]}`, or just the list). Aokie predates the
//! section: a plugin with the id `aokie` and no section provides the phone and
//! the calendar (`declared: false`). A phone claim is honoured only when one of
//! the plugin's connectors declares the commands OAIY uses (`phone.status`,
//! `sms.send`, `settings.get`, `settings.set`, `call.dial`). When two plugins
//! provide one module, the lowest plugin id is used, with a warning.
//!
//! # Nothing is deleted
//!
//! A module turned off keeps its data: the calendar (`calendar/`), the callers'
//! names (`callers.json`), the voices, the plugin's own data, and in the Agent
//! app the Front desk's files and conversations. Its routes answer
//! `module_disabled` (409) until it is back.

pub mod routes;

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{watch, Notify};

use crate::plugins::{PluginManifest, PluginRecord, PluginRegistry, PluginRegistryHandle, PluginState};

pub const PHONE: &str = "phone";
pub const CALENDAR: &str = "calendar";

/// The plugin that provided the phone and the calendar before manifests could say so.
pub const LEGACY_PROVIDER: &str = "aokie";

/// How often the modules are worked out again without being asked.
const REFRESH_EVERY: Duration = Duration::from_secs(5);

/// A module this desktop knows.
#[derive(Debug)]
pub struct ModuleDef {
    pub id: &'static str,
    /// As a heading: "Phone".
    pub name: &'static str,
    /// In a sentence: "the phone".
    pub noun: &'static str,
    /// Connector commands a provider's connector must declare.
    pub uses: &'static [&'static str],
    /// Leases only a page using this module takes; let go when it is turned off.
    pub leases: &'static [&'static str],
    /// What it keeps in the data folder (kept when it is off).
    pub store: &'static [&'static str],
}

pub const BUILTIN: &[ModuleDef] = &[
    ModuleDef {
        id: PHONE,
        name: "Phone",
        noun: "the phone",
        uses: &["phone.status", "sms.send", "settings.get", "settings.set", "call.dial"],
        leases: &["answer-calls", "answer-texts"],
        store: &["callers.json", "voices/"],
    },
    ModuleDef { id: CALENDAR, name: "Calendar", noun: "the calendar", uses: &[], leases: &[], store: &["calendar/"] },
];

pub fn def(id: &str) -> Option<&'static ModuleDef> {
    BUILTIN.iter().find(|d| d.id == id)
}

/// The module a lease belongs to (`answer-calls` and `answer-texts`: the phone). Core leases have none.
pub fn lease_module(lease: &str) -> Option<&'static ModuleDef> {
    BUILTIN.iter().find(|d| d.leases.contains(&lease))
}

// ---- What a plugin claims ------------------------------------------------------

/// One module a plugin says it provides.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub module: &'static str,
    /// The connector that serves it (the phone's).
    pub connector: Option<String>,
    /// Why the claim is not honoured; `None` when it is.
    pub refused: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Claims {
    pub claims: Vec<Claim>,
    /// The manifest has a `modules` section (false: the legacy rule, or nothing).
    pub declared: bool,
    /// Module ids it names that this desktop does not know.
    pub unknown: Vec<String>,
}

impl Claims {
    pub fn get(&self, module: &str) -> Option<&Claim> {
        self.claims.iter().find(|c| c.module == module)
    }

    /// Does the plugin provide `module` (claimed, and the claim honoured)?
    pub fn provides(&self, module: &str) -> bool {
        self.get(module).is_some_and(|c| c.refused.is_none())
    }
}

/// The module ids a `modules` section names: `{"provides": [...]}`, or the list
/// itself, of ids or `{"id": ...}` objects.
fn named_modules(section: &Value) -> Vec<String> {
    let list = match section {
        Value::Array(a) => a.as_slice(),
        Value::Object(o) => o.get("provides").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]),
        _ => &[],
    };
    let mut out: Vec<String> = Vec::new();
    for item in list {
        let id = item.as_str().or_else(|| item.get("id").and_then(Value::as_str)).map(str::trim).unwrap_or("");
        if !id.is_empty() && !out.iter().any(|x| x == id) {
            out.push(id.to_string());
        }
    }
    out
}

/// What `manifest` claims to provide, each claim checked against what the module needs.
pub fn claims(manifest: &PluginManifest) -> Claims {
    let (declared, named) = match manifest.extra.get("modules") {
        Some(section) => (true, named_modules(section)),
        None if manifest.id == LEGACY_PROVIDER => (false, vec![PHONE.to_string(), CALENDAR.to_string()]),
        None => (false, Vec::new()),
    };
    let mut out = Claims { declared, ..Claims::default() };
    for id in named {
        let Some(def) = def(&id) else {
            out.unknown.push(id);
            continue;
        };
        out.claims.push(check_claim(manifest, def));
    }
    out
}

/// A claim is honoured when one of the plugin's connectors declares every command the module uses.
fn check_claim(manifest: &PluginManifest, def: &'static ModuleDef) -> Claim {
    if def.uses.is_empty() {
        return Claim { module: def.id, connector: None, refused: None };
    }
    let missing = |c: &crate::plugins::manifest::ConnectorDecl| -> Vec<&'static str> {
        def.uses.iter().copied().filter(|cmd| !c.commands.iter().any(|x| x == cmd)).collect()
    };
    if let Some(c) = manifest.connectors.iter().find(|c| missing(c).is_empty()) {
        return Claim { module: def.id, connector: Some(c.id.clone()), refused: None };
    }
    let refused = match manifest.connectors.iter().min_by_key(|c| missing(c).len()) {
        None => format!("{} claims {}, but it declares no connector (it needs {}).", manifest.name, def.noun, def.uses.join(", ")),
        Some(c) => format!(
            "{} claims {}, but its connector \"{}\" does not declare {}.",
            manifest.name,
            def.noun,
            c.id,
            missing(c).join(", ")
        ),
    };
    Claim { module: def.id, connector: None, refused: Some(refused) }
}

// ---- Which modules are on ------------------------------------------------------

/// The plugin providing a module (or, for one that is off, the plugin that would).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Provider {
    pub plugin_id: String,
    pub name: String,
    pub state: PluginState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector: Option<String>,
    /// It said so in its manifest (false: Aokie's legacy rule).
    pub declared: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Module {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub builtin: bool,
    /// Why it is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub provider: Option<Provider>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub leases: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub uses: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub store: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Resolved {
    pub modules: Vec<Module>,
    pub warnings: Vec<String>,
}

/// A plugin that claims a module, as `resolve` weighs it.
struct Candidate<'a> {
    record: &'a PluginRecord,
    /// `None`: the manifest could not be loaded (an `aokie` whose manifest is broken).
    claim: Option<Claim>,
    declared: bool,
}

impl Candidate<'_> {
    fn eligible(&self) -> bool {
        self.record.manifest.is_some() && !self.record.user_disabled && self.claim.as_ref().is_some_and(|c| c.refused.is_none())
    }

    fn provider(&self) -> Provider {
        Provider {
            plugin_id: self.record.id.clone(),
            name: self.record.manifest.as_ref().map(|m| m.name.clone()).unwrap_or_else(|| self.record.id.clone()),
            state: self.record.state,
            connector: self.claim.as_ref().and_then(|c| c.connector.clone()),
            declared: self.declared,
        }
    }

    /// Why this plugin does not provide the module.
    fn why_not(&self, def: &ModuleDef) -> String {
        let name = self.provider().name;
        if self.record.manifest.is_none() {
            let why = self.record.reason.clone().unwrap_or_else(|| "its manifest could not be read".into());
            return format!("{name} could not be loaded, so it does not provide {}: {why}", def.noun);
        }
        if self.record.user_disabled {
            return format!("{name} is turned off in Plugins.");
        }
        self.claim.as_ref().and_then(|c| c.refused.clone()).unwrap_or_else(|| format!("{name} does not provide {}.", def.noun))
    }
}

/// Which modules are on, from the plugin registry's records. Pure.
pub fn resolve(records: &[PluginRecord]) -> Resolved {
    let mut records: Vec<&PluginRecord> = records.iter().collect();
    // The lowest plugin id wins a tie, so the order is part of the answer.
    records.sort_by(|a, b| a.id.cmp(&b.id));
    let claimed: Vec<(&PluginRecord, Option<Claims>)> = records
        .iter()
        .map(|r| (*r, r.manifest.as_ref().map(claims)))
        .collect();

    let mut warnings = Vec::new();
    for (rec, c) in &claimed {
        for unknown in c.iter().flat_map(|c| c.unknown.iter()) {
            warnings.push(format!("{} names a module this version of OAIY does not know: \"{unknown}\".", rec.id));
        }
    }

    let mut modules = Vec::new();
    for def in BUILTIN {
        let candidates: Vec<Candidate> = claimed
            .iter()
            .filter_map(|(rec, c)| match c {
                Some(c) => c.get(def.id).map(|claim| Candidate { record: rec, claim: Some(claim.clone()), declared: c.declared }),
                None if rec.id == LEGACY_PROVIDER => Some(Candidate { record: rec, claim: None, declared: false }),
                None => None,
            })
            .collect();
        let eligible: Vec<&Candidate> = candidates.iter().filter(|c| c.eligible()).collect();
        let (enabled, reason, provider) = match eligible.first() {
            Some(winner) => {
                if eligible.len() > 1 {
                    let others: Vec<&str> = eligible[1..].iter().map(|c| c.record.id.as_str()).collect();
                    warnings.push(format!(
                        "{} and {} {} provide {}: {} is used (the lowest plugin id).",
                        winner.record.id,
                        others.join(", "),
                        if others.len() == 1 { "both" } else { "all" },
                        def.noun,
                        winner.record.id
                    ));
                }
                for c in candidates.iter().filter(|c| !c.eligible() && c.claim.as_ref().is_some_and(|x| x.refused.is_some())) {
                    warnings.push(c.why_not(def));
                }
                (true, None, Some(winner.provider()))
            }
            None => match candidates.first() {
                Some(c) => (false, Some(c.why_not(def)), Some(c.provider())),
                None => (false, Some(format!("No installed plugin provides {}.", def.noun)), None),
            },
        };
        modules.push(Module {
            id: def.id.to_string(),
            name: def.name.to_string(),
            enabled,
            builtin: true,
            reason,
            provider,
            leases: def.leases.iter().map(|s| s.to_string()).collect(),
            uses: def.uses.iter().map(|s| s.to_string()).collect(),
            store: def.store.iter().map(|s| s.to_string()).collect(),
        });
    }
    Resolved { modules, warnings }
}

// ---- The snapshot, and its revision ---------------------------------------------

/// The modules as `/api/modules` shows them.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// Moves when (and only when) anything below changes.
    pub revision: u64,
    pub modules: Vec<Module>,
    /// What modules add to the app and the dashboard: empty for now.
    pub contributions: serde_json::Map<String, Value>,
    pub warnings: Vec<String>,
}

impl Snapshot {
    fn empty() -> Self {
        Self { revision: 0, modules: Vec::new(), contributions: serde_json::Map::new(), warnings: Vec::new() }
    }

    pub fn module(&self, id: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.id == id)
    }

    pub fn is_enabled(&self, id: &str) -> bool {
        self.module(id).is_some_and(|m| m.enabled)
    }
}

/// Keeps the snapshot, and moves its revision only when what it says changes.
pub struct Publisher {
    current: Arc<Snapshot>,
}

impl Default for Publisher {
    fn default() -> Self {
        Self { current: Arc::new(Snapshot::empty()) }
    }
}

impl Publisher {
    pub fn current(&self) -> Arc<Snapshot> {
        self.current.clone()
    }

    /// Take `resolved`: the snapshot now, and whether it changed.
    pub fn update(&mut self, resolved: Resolved) -> (Arc<Snapshot>, bool) {
        if self.current.revision > 0 && self.current.modules == resolved.modules && self.current.warnings == resolved.warnings {
            return (self.current.clone(), false);
        }
        self.current = Arc::new(Snapshot {
            revision: self.current.revision + 1,
            modules: resolved.modules,
            contributions: serde_json::Map::new(),
            warnings: resolved.warnings,
        });
        (self.current.clone(), true)
    }
}

/// The modules that were on in `before` and are not in `after`.
pub fn went_off(before: &Snapshot, after: &Snapshot) -> Vec<&'static ModuleDef> {
    BUILTIN.iter().filter(|d| before.is_enabled(d.id) && !after.is_enabled(d.id)).collect()
}

// ---- The desktop's modules -------------------------------------------------------

struct Global {
    publisher: Mutex<Publisher>,
    current: RwLock<Arc<Snapshot>>,
    changes: watch::Sender<Arc<Snapshot>>,
    poke: Notify,
    /// Tells this run's revisions from an earlier run's (in the ETag).
    boot: u64,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();
static WATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn global() -> &'static Global {
    GLOBAL.get_or_init(|| {
        let empty = Arc::new(Snapshot::empty());
        let boot = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Global {
            publisher: Mutex::new(Publisher::default()),
            current: RwLock::new(empty.clone()),
            changes: watch::channel(empty).0,
            poke: Notify::new(),
            boot,
        }
    })
}

/// The modules now (before the first look, none: everything is off).
pub fn snapshot() -> Arc<Snapshot> {
    global().current.read().map(|g| g.clone()).unwrap_or_else(|e| e.into_inner().clone())
}

/// The snapshot now, then each change.
pub fn subscribe() -> watch::Receiver<Arc<Snapshot>> {
    global().changes.subscribe()
}

/// The ETag of a snapshot: its revision, in this run.
pub fn etag(snapshot: &Snapshot) -> String {
    format!("\"{}-{:x}\"", snapshot.revision, global().boot)
}

/// Is `id` on now?
pub fn is_enabled(id: &str) -> bool {
    #[cfg(test)]
    if let Some(on) = test_gate::get(id) {
        return on;
    }
    snapshot().is_enabled(id)
}

/// Work the modules out again soon (a plugin changed state).
pub fn poke() {
    global().poke.notify_one();
}

/// Work the modules out from `registry` now, and publish them if they changed.
pub fn refresh(registry: &PluginRegistry) {
    publish(resolve(&registry.list()));
}

fn refresh_handle(plugins: &PluginRegistryHandle) {
    match plugins.lock() {
        Ok(reg) => refresh(&reg),
        Err(e) => refresh(&e.into_inner()),
    }
}

fn publish(resolved: Resolved) {
    let g = global();
    let mut publisher = g.publisher.lock().unwrap_or_else(|e| e.into_inner());
    let before = publisher.current();
    let (now, changed) = publisher.update(resolved);
    if !changed {
        return;
    }
    if let Ok(mut current) = g.current.write() {
        *current = now.clone();
    }
    g.changes.send_replace(now.clone());
    drop(publisher);
    for m in &now.modules {
        let was = before.module(&m.id).map(|b| b.enabled);
        if was != Some(m.enabled) {
            match (&m.reason, m.enabled) {
                (_, true) => log::info!("module {} is on (from {})", m.id, m.provider.as_ref().map(|p| p.plugin_id.as_str()).unwrap_or("?")),
                (Some(r), false) => log::info!("module {} is off: {r}", m.id),
                (None, false) => log::info!("module {} is off", m.id),
            }
        }
    }
    // A module turned off lets go of its leases: no page goes on answering the phone.
    for def in went_off(&before, &now) {
        crate::bridge::leases::drop_names(def.leases);
    }
}

/// Work the modules out now, then again every few seconds and whenever poked.
/// Called once, as the API starts (later calls only refresh).
pub fn start(plugins: PluginRegistryHandle) {
    // The registry fills lazily (the plugins' autostart scans it on its own
    // thread): scanned here first, so the first snapshot does not say the
    // phone is off for the few seconds before that.
    match plugins.lock() {
        Ok(mut reg) => {
            reg.scan();
            refresh(&reg);
        }
        Err(e) => refresh(&e.into_inner()),
    }
    if WATCHING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = global().poke.notified() => {}
                _ = tokio::time::sleep(REFRESH_EVERY) => {}
            }
            refresh_handle(&plugins);
        }
    });
}

/// `409 {"error": {"code": "module_disabled", …}}`: a route of a module that is off.
pub fn disabled_response(id: &str) -> axum::response::Response {
    (StatusCode::CONFLICT, Json(json!({"error": {"code": "module_disabled", "message": disabled_message(id)}}))).into_response()
}

pub fn disabled_message(id: &str) -> String {
    let noun = def(id).map(|d| d.noun).unwrap_or(id);
    let mut noun_cap = noun.to_string();
    if let Some(first) = noun_cap.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    match snapshot().module(id).and_then(|m| m.reason.clone()) {
        Some(reason) => format!("{noun_cap} is off: {reason}"),
        None => format!("{noun_cap} is off: no installed plugin provides it."),
    }
}

/// The ids of the modules a manifest provides, for a quick look (the gateway token).
pub fn provided_by(manifest: &PluginManifest) -> BTreeSet<&'static str> {
    claims(manifest).claims.iter().filter(|c| c.refused.is_none()).map(|c| c.module).collect()
}

/// Tests say which modules are on, on their own thread, without the desktop's registry.
#[cfg(test)]
pub mod test_gate {
    use std::cell::RefCell;

    thread_local! {
        static ON: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }

    pub(super) fn get(id: &str) -> Option<bool> {
        ON.with(|on| on.borrow().as_ref().map(|list| list.iter().any(|x| x == id)))
    }

    /// While the guard lives, exactly `on` are enabled on this thread.
    pub struct Guard(Option<Vec<String>>);

    pub fn enable(on: &[&str]) -> Guard {
        let before = ON.with(|x| x.replace(Some(on.iter().map(|s| s.to_string()).collect())));
        Guard(before)
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let before = self.0.take();
            ON.with(|x| *x.borrow_mut() = before);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn manifest(body: Value) -> PluginManifest {
        serde_json::from_value(body).expect("a manifest")
    }

    fn aokie_manifest() -> Value {
        json!({
            "schemaVersion": 3, "id": "aokie", "name": "Aokie Phone Bridge", "version": "1.0.0", "pluginApiVersion": 1,
            "entry": {"kind": "process", "command": "aokie.exe"},
            "connectors": [{"id": "aokie", "commands": ["phone.status", "call.dial", "call.answer", "sms.send", "settings.get", "settings.set"]}],
        })
    }

    fn record(m: Value, state: PluginState, user_disabled: bool) -> PluginRecord {
        let m = manifest(m);
        PluginRecord {
            id: m.id.clone(),
            state,
            reason: None,
            dir: PathBuf::from(format!("/plugins/{}", m.id)),
            manifest: Some(m),
            legacy_capabilities: Vec::new(),
            unknown_capabilities: Vec::new(),
            user_disabled,
            restart_attempts: 0,
            last_health: None,
            last_health_at: None,
            last_health_error: None,
        }
    }

    fn module<'a>(r: &'a Resolved, id: &str) -> &'a Module {
        r.modules.iter().find(|m| m.id == id).expect("a builtin module")
    }

    #[test]
    fn a_legacy_aokie_provides_the_phone_and_the_calendar() {
        let r = resolve(&[record(aokie_manifest(), PluginState::Running, false)]);
        for id in [PHONE, CALENDAR] {
            let m = module(&r, id);
            assert!(m.enabled, "{id}: {m:?}");
            assert!(m.reason.is_none());
            let p = m.provider.as_ref().unwrap();
            assert_eq!(p.plugin_id, "aokie");
            assert!(!p.declared, "Aokie has no modules section: the legacy rule");
        }
        assert_eq!(module(&r, PHONE).provider.as_ref().unwrap().connector.as_deref(), Some("aokie"));
        assert_eq!(module(&r, PHONE).leases, vec!["answer-calls", "answer-texts"]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        // Another plugin with no section provides nothing.
        let mut other = aokie_manifest();
        other["id"] = json!("weather");
        assert!(claims(&manifest(other)).claims.is_empty());
    }

    #[test]
    fn nothing_installed_means_no_phone_and_no_calendar() {
        let r = resolve(&[]);
        for id in [PHONE, CALENDAR] {
            let m = module(&r, id);
            assert!(!m.enabled);
            assert!(m.provider.is_none());
            assert!(m.reason.as_deref().unwrap().contains("No installed plugin"));
        }
    }

    #[test]
    fn a_turned_off_plugin_provides_nothing_and_says_why() {
        // Turned off while it ran: the registry keeps a running record Running, so it is the opt-out that counts.
        for state in [PluginState::Disabled, PluginState::Running] {
            let r = resolve(&[record(aokie_manifest(), state, true)]);
            for id in [PHONE, CALENDAR] {
                let m = module(&r, id);
                assert!(!m.enabled, "{id} with {state:?}");
                assert!(m.reason.as_deref().unwrap().contains("turned off in Plugins"), "{:?}", m.reason);
                assert_eq!(m.provider.as_ref().unwrap().plugin_id, "aokie");
            }
        }
    }

    #[test]
    fn a_broken_manifest_provides_nothing_and_says_why() {
        let broken = PluginRecord {
            id: "aokie".into(),
            state: PluginState::Disabled,
            reason: Some("manifest.json is not valid JSON: expected value at line 1".into()),
            dir: PathBuf::from("/plugins/aokie"),
            manifest: None,
            legacy_capabilities: Vec::new(),
            unknown_capabilities: Vec::new(),
            user_disabled: false,
            restart_attempts: 0,
            last_health: None,
            last_health_at: None,
            last_health_error: None,
        };
        let r = resolve(&[broken]);
        for id in [PHONE, CALENDAR] {
            let m = module(&r, id);
            assert!(!m.enabled);
            let reason = m.reason.as_deref().unwrap();
            assert!(reason.contains("could not be loaded") && reason.contains("not valid JSON"), "{reason}");
        }
    }

    #[test]
    fn a_crashed_or_stopped_provider_keeps_its_modules() {
        for state in [PluginState::Crashed, PluginState::Stopped, PluginState::Installed, PluginState::Unhealthy, PluginState::Starting] {
            let r = resolve(&[record(aokie_manifest(), state, false)]);
            let m = module(&r, PHONE);
            assert!(m.enabled, "{state:?}");
            assert_eq!(m.provider.as_ref().unwrap().state, state, "the provider's state says how it is");
            assert!(module(&r, CALENDAR).enabled);
        }
    }

    #[test]
    fn a_phone_claim_without_the_commands_it_needs_is_refused() {
        let mut m = aokie_manifest();
        m["id"] = json!("cheapphone");
        m["name"] = json!("Cheap Phone");
        m["modules"] = json!({"provides": ["phone", "calendar"]});
        m["connectors"] = json!([{"id": "cheapphone", "commands": ["phone.status", "sms.send"]}]);
        let c = claims(&manifest(m.clone()));
        assert!(c.declared);
        assert!(!c.provides(PHONE));
        assert!(c.provides(CALENDAR), "the calendar needs no commands");
        let r = resolve(&[record(m, PluginState::Running, false)]);
        let phone = module(&r, PHONE);
        assert!(!phone.enabled);
        let reason = phone.reason.as_deref().unwrap();
        assert!(reason.contains("settings.get") && reason.contains("call.dial") && !reason.contains("sms.send"), "{reason}");
        assert!(module(&r, CALENDAR).enabled);
        assert_eq!(module(&r, CALENDAR).provider.as_ref().unwrap().plugin_id, "cheapphone");
    }

    #[test]
    fn a_declared_section_is_what_counts_even_for_aokie() {
        let mut m = aokie_manifest();
        m["modules"] = json!(["calendar", {"id": "fax"}]);
        let c = claims(&manifest(m.clone()));
        assert!(c.declared);
        assert!(!c.provides(PHONE));
        assert!(c.provides(CALENDAR));
        assert_eq!(c.unknown, vec!["fax"]);
        let r = resolve(&[record(m, PluginState::Running, false)]);
        assert!(!module(&r, PHONE).enabled);
        assert!(module(&r, CALENDAR).provider.as_ref().unwrap().declared);
        assert!(r.warnings.iter().any(|w| w.contains("\"fax\"")), "{:?}", r.warnings);
    }

    #[test]
    fn two_providers_of_one_module_pick_the_lowest_id_and_warn() {
        let mut zed = aokie_manifest();
        zed["id"] = json!("zed-phone");
        zed["name"] = json!("Zed Phone");
        zed["modules"] = json!({"provides": ["phone"]});
        let r = resolve(&[record(zed.clone(), PluginState::Running, false), record(aokie_manifest(), PluginState::Stopped, false)]);
        let phone = module(&r, PHONE);
        assert!(phone.enabled);
        assert_eq!(phone.provider.as_ref().unwrap().plugin_id, "aokie");
        assert!(r.warnings.iter().any(|w| w.contains("aokie") && w.contains("zed-phone")), "{:?}", r.warnings);
        // With Aokie turned off, the other one is used, and there is nothing to warn about.
        let r = resolve(&[record(zed, PluginState::Running, false), record(aokie_manifest(), PluginState::Disabled, true)]);
        assert_eq!(module(&r, PHONE).provider.as_ref().unwrap().plugin_id, "zed-phone");
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn the_revision_moves_only_on_change() {
        let mut p = Publisher::default();
        let running = || resolve(&[record(aokie_manifest(), PluginState::Running, false)]);
        let (first, changed) = p.update(running());
        assert!(changed);
        assert_eq!(first.revision, 1);
        let (same, changed) = p.update(running());
        assert!(!changed);
        assert_eq!(same.revision, 1);
        // A crash changes the provider's state (not whether the phone is on): a new revision.
        let (crashed, changed) = p.update(resolve(&[record(aokie_manifest(), PluginState::Crashed, false)]));
        assert!(changed);
        assert_eq!(crashed.revision, 2);
        assert!(crashed.is_enabled(PHONE));
        assert!(went_off(&first, &crashed).is_empty());
        let (off, _) = p.update(resolve(&[record(aokie_manifest(), PluginState::Disabled, true)]));
        assert_eq!(off.revision, 3);
        let gone: Vec<&str> = went_off(&crashed, &off).iter().map(|d| d.id).collect();
        assert_eq!(gone, vec![PHONE, CALENDAR]);
    }

    #[test]
    fn the_snapshot_reads_as_the_api_shows_it() {
        let mut p = Publisher::default();
        let (snap, _) = p.update(resolve(&[record(aokie_manifest(), PluginState::Running, false)]));
        let v = serde_json::to_value(&*snap).unwrap();
        assert_eq!(v["revision"], 1);
        assert_eq!(v["contributions"], json!({}));
        assert_eq!(v["warnings"], json!([]));
        let phone = &v["modules"][0];
        assert_eq!(phone["id"], "phone");
        assert_eq!(phone["enabled"], true);
        assert_eq!(phone["builtin"], true);
        assert_eq!(phone["provider"], json!({"pluginId": "aokie", "name": "Aokie Phone Bridge", "state": "running", "connector": "aokie", "declared": false}));
        assert!(phone.get("reason").is_none());
        assert!(!v.to_string().contains("/plugins/aokie"), "no plugin directory in the snapshot");
    }

    #[test]
    fn only_the_phone_provider_is_named_for_the_gateway_token() {
        assert!(provided_by(&manifest(aokie_manifest())).contains(PHONE));
        let mut calendar_only = aokie_manifest();
        calendar_only["id"] = json!("diary");
        calendar_only["modules"] = json!({"provides": ["calendar"]});
        assert!(!provided_by(&manifest(calendar_only)).contains(PHONE));
    }

    #[test]
    fn leases_belong_to_their_module() {
        assert_eq!(lease_module("answer-calls").map(|d| d.id), Some(PHONE));
        assert_eq!(lease_module("answer-texts").map(|d| d.id), Some(PHONE));
        assert!(lease_module("answer-tasks").is_none(), "flows' tasks are core");
    }
}
