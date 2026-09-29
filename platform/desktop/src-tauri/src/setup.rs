//! The setup wizard's own record, `<data>/setup.json`, and its routes.
//!
//! The record keeps only what cannot be worked out live: how far the
//! first-run wizard got (the step it is on, the steps skipped, the plugins
//! chosen, whether it is finished), and for each plugin the setup version last
//! finished, the steps recorded done or skipped, and the capabilities the
//! person accepted. Whether a step is *done* is otherwise worked out live by
//! the dashboard (a service installed, a model chosen in Engines, a plugin's
//! own `done` check), so the record can never claim something that is no
//! longer true.
//!
//!   GET  /api/setup                              → the record
//!   PUT  /api/setup {firstRun}                   → the first-run part, replaced
//!   GET  /api/setup/catalog                      → the plugins OAIY knows how to install
//!   GET  /api/setup/plugins/:id                  → its declared setup, capabilities, record
//!   POST /api/setup/plugins/:id/steps/:step {status: done|skipped|todo}
//!   POST /api/setup/plugins/:id/finish           → its setup version, recorded as finished
//!   POST /api/setup/plugins/:id/check/:step[?check=when] → {passed, detail}
//!
//! Reading is a restricted read and every change is privileged (see `http.rs`):
//! accepting a plugin's capabilities is a trust act, and a check runs one of
//! the plugin's commands.
//!
//! # A desktop already in use never gets a surprise wizard
//!
//! When `setup.json` is first made, a desktop that already shows signs of use
//! (a ready AI source, ChatGPT signed in, a language model chosen in Engines,
//! a plugin installed) is recorded as finished, with the reason in
//! `firstRun.migrated`. A record that cannot be read is set aside and treated
//! the same way.
//!
//! # A plugin's `setup` section
//!
//! The manifest parser's typed section (schemaVersion 4, validated at load),
//! reached through [`declared`]. The parser checks each `done` and `when`
//! check; [`judge`] evaluates one against its command's answer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::plugins::manifest::{Check, Condition, SetupDecl, SetupStep, StepKind};
use crate::plugins::registry::{PluginRecord, PluginRegistryHandle};
use crate::plugins::{CallError, ForwardError, PluginHost};

/// The record's file, in the data folder.
pub const FILE: &str = "setup.json";

/// How long a `done`/`when` check may take: an answer after this is "not yet".
pub const CHECK_TIMEOUT: Duration = Duration::from_secs(5);

/// No list in the record grows past this, and no name in it past `MAX_TEXT`.
const MAX_LIST: usize = 64;
const MAX_TEXT: usize = 64;

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupState {
    #[serde(default)]
    pub first_run: FirstRun,
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginSetup>,
}

/// The first-run wizard: resumable and skippable, so where it is is kept here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FirstRun {
    #[serde(default)]
    pub finished: bool,
    /// The step it is on (a step id of the dashboard's), to come back to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    #[serde(default)]
    pub skipped: Vec<String>,
    /// The plugins the person ticked to install and set up.
    #[serde(default)]
    pub chosen_plugins: Vec<String>,
    /// Why a desktop already in use was recorded as finished when this record
    /// was first made. Written by the desktop only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated: Option<String>,
}

/// One plugin's setup, as far as it cannot be worked out live.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginSetup {
    /// The `setup.version` last finished (0: never).
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub done: Vec<String>,
    #[serde(default)]
    pub skipped: Vec<String>,
    /// The resolved capabilities the person accepted, as they were then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissions_accepted: Option<Vec<String>>,
}

/// What a step is recorded as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Done,
    Skipped,
    /// Neither: the record forgets the step.
    Todo,
}

/// `<data>/setup.json`, in memory and on disk.
pub struct Store {
    path: PathBuf,
    state: SetupState,
}

impl Store {
    /// Open the record in `data_dir`, making it when there is none. `in_use`
    /// is asked only then: `Some(reason)` when the desktop is already in use,
    /// which records the first-run wizard as finished.
    pub fn open(data_dir: &Path, in_use: impl FnOnce() -> Option<String>) -> Store {
        let path = data_dir.join(FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<SetupState>(&text) {
                Ok(state) => Store { path, state },
                Err(e) => {
                    // Kept aside for a person to look at, never silently lost;
                    // and a desktop with a record at all has been set up before.
                    let aside = path.with_extension("json.unreadable");
                    let _ = std::fs::rename(&path, &aside);
                    log::warn!("setup: {} could not be read ({e}); kept as {}", path.display(), aside.display());
                    let store = Store { path, state: finished_because(format!("setup.json could not be read ({e})")) };
                    store.save_or_log();
                    store
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let state = match in_use() {
                    Some(reason) => finished_because(reason),
                    None => SetupState::default(),
                };
                let store = Store { path, state };
                store.save_or_log();
                store
            }
            Err(e) => {
                // There, but not readable now: behave as a desktop in use, and
                // leave the file alone.
                log::warn!("setup: {} could not be read: {e}", path.display());
                Store { path, state: finished_because(format!("setup.json could not be read ({e})")) }
            }
        }
    }

    pub fn state(&self) -> &SetupState {
        &self.state
    }

    fn save_or_log(&self) {
        if let Err(e) = write_atomic(&self.path, &self.state) {
            log::warn!("setup: {e}");
        }
    }

    /// Change the record and save it; nothing changes unless it is saved.
    fn change(&mut self, f: impl FnOnce(&mut SetupState) -> Result<(), String>) -> Result<&SetupState, String> {
        let mut next = self.state.clone();
        f(&mut next)?;
        write_atomic(&self.path, &next)?;
        self.state = next;
        Ok(&self.state)
    }

    /// Replace the first-run part. `migrated` is the desktop's to write, so it is kept.
    pub fn set_first_run(&mut self, first_run: FirstRun) -> Result<&SetupState, String> {
        let first_run = clean_first_run(first_run)?;
        self.change(|s| {
            let migrated = s.first_run.migrated.take();
            s.first_run = FirstRun { migrated, ..first_run };
            Ok(())
        })
    }

    /// Record `step` of `plugin` as done, skipped, or neither. `accepted` goes
    /// with the permissions step: the capabilities accepted.
    pub fn mark_step(&mut self, plugin: &str, step: &str, status: StepStatus, accepted: Option<Vec<String>>) -> Result<&SetupState, String> {
        if !valid_plugin_id(plugin) {
            return Err(format!("{plugin:?} is not a plugin id"));
        }
        if !valid_step_id(step) {
            return Err(format!("{step:?} is not a step id"));
        }
        self.change(|s| {
            let entry = s.plugins.entry(plugin.to_string()).or_default();
            entry.done.retain(|d| d != step);
            entry.skipped.retain(|d| d != step);
            match status {
                StepStatus::Done => entry.done.push(step.to_string()),
                StepStatus::Skipped => entry.skipped.push(step.to_string()),
                StepStatus::Todo => {}
            }
            if entry.done.len() > MAX_LIST || entry.skipped.len() > MAX_LIST {
                return Err("too many steps".into());
            }
            if step == PERMISSIONS {
                entry.permissions_accepted = if status == StepStatus::Done { accepted } else { None };
            }
            Ok(())
        })
    }

    /// Record `plugin`'s setup `version` as finished.
    pub fn finish_plugin(&mut self, plugin: &str, version: u32) -> Result<&SetupState, String> {
        if !valid_plugin_id(plugin) {
            return Err(format!("{plugin:?} is not a plugin id"));
        }
        self.change(|s| {
            s.plugins.entry(plugin.to_string()).or_default().version = version;
            Ok(())
        })
    }
}

fn finished_because(reason: String) -> SetupState {
    SetupState { first_run: FirstRun { finished: true, migrated: Some(reason), ..FirstRun::default() }, plugins: BTreeMap::new() }
}

/// Write `state` to `path` whole or not at all: a temporary file beside it, then a rename.
fn write_atomic(path: &Path, state: &SetupState) -> Result<(), String> {
    let text = serde_json::to_string_pretty(state).map_err(|e| e.to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4().simple()));
    std::fs::write(&tmp, text).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("could not save {}: {e}", path.display())
    })
}

fn clean_list(list: Vec<String>, what: &str, valid: fn(&str) -> bool) -> Result<Vec<String>, String> {
    if list.len() > MAX_LIST {
        return Err(format!("{what}: at most {MAX_LIST}"));
    }
    let mut out: Vec<String> = Vec::new();
    for item in list {
        if !valid(&item) {
            return Err(format!("{what}: {item:?} is not a valid name"));
        }
        if !out.contains(&item) {
            out.push(item);
        }
    }
    Ok(out)
}

fn clean_first_run(f: FirstRun) -> Result<FirstRun, String> {
    if let Some(p) = &f.position {
        if !valid_step_id(p) {
            return Err(format!("position: {p:?} is not a step id"));
        }
    }
    Ok(FirstRun {
        finished: f.finished,
        position: f.position,
        skipped: clean_list(f.skipped, "skipped", valid_step_id)?,
        chosen_plugins: clean_list(f.chosen_plugins, "chosenPlugins", valid_plugin_id)?,
        migrated: None,
    })
}

/// A plugin id, as `plugins/install.rs` accepts one.
pub fn valid_plugin_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_TEXT
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// A step id (`^[a-z][a-z0-9-]{0,39}$`), or one of the dashboard's first-run
/// step ids, which may carry a plugin id after a colon (`plugin:aokie`).
pub fn valid_step_id(id: &str) -> bool {
    let (head, tail) = id.split_once(':').unwrap_or((id, ""));
    let head_ok = head.len() <= 40
        && head.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && head.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    head_ok && (tail.is_empty() && !id.ends_with(':') || valid_plugin_id(tail))
}

// ---------------------------------------------------------------------------
// Is this desktop already in use? (asked once, when the record is first made)
// ---------------------------------------------------------------------------

/// The first sign that this desktop is already set up and used, if there is one.
pub fn in_use_signal(data_dir: &Path, plugins_root: &Path, providers: &[crate::ai::providers::AiProviderPublic]) -> Option<String> {
    if providers.iter().any(|p| p.enabled && (p.has_key || p.allow_local)) {
        return Some("an AI provider was already set up".into());
    }
    // The ChatGPT connector keeps its sign-in in its own CODEX_HOME (ai/codex.rs).
    if data_dir.join("ai").join("codex-home").join("auth.json").is_file() {
        return Some("ChatGPT was already signed in".into());
    }
    // The engines' configuration (engines.rs): made with no model on a fresh
    // install, so only a model in it says anything.
    if let Ok(text) = std::fs::read_to_string(data_dir.join("engines").join("oaiy-studio.json")) {
        if let Ok(cfg) = serde_json::from_str::<Value>(&text) {
            let llm = cfg.get("llm");
            let chosen = llm.and_then(|l| l.get("default_model")).and_then(Value::as_str).is_some_and(|m| !m.trim().is_empty());
            let listed = llm.and_then(|l| l.get("models")).and_then(Value::as_array).is_some_and(|m| !m.is_empty());
            if chosen || listed {
                return Some("a language model was already chosen in Engines".into());
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(plugins_root) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with('.') && entry.path().join("manifest.json").is_file() {
                return Some(format!("a plugin ({name}) was already installed"));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// A plugin's declared setup
// ---------------------------------------------------------------------------

/// The step the host always puts first.
pub const PERMISSIONS: &str = "permissions";

/// The `setup` section of `record`'s manifest (schemaVersion 4), if it
/// declares one: the manifest parser's typed section, validated at load (a
/// check's command declared and not journalled, a screen the plugin ships),
/// with its version (default 1) and title (default "Set up <name>") filled
/// in and host steps this OAIY cannot run already left out.
pub fn declared(record: &PluginRecord) -> Option<&SetupDecl> {
    record.manifest.as_ref()?.setup.as_ref()
}

/// Step `id` of a declared setup.
pub fn find_step<'a>(setup: &'a SetupDecl, id: &str) -> Option<&'a SetupStep> {
    setup.steps.iter().find(|s| s.id == id)
}

/// `record`'s capabilities with wildcards expanded: what the permissions step shows and records.
pub fn resolved_capabilities(record: &PluginRecord) -> Vec<String> {
    record.manifest.as_ref().map(|m| m.resolved_capabilities().into_iter().collect()).unwrap_or_default()
}

/// Whether `accepted` covers every capability in `current`.
pub fn accepted_covers(accepted: Option<&Vec<String>>, current: &[String]) -> bool {
    accepted.is_some_and(|a| current.iter().all(|c| a.contains(c)))
}

// ---------------------------------------------------------------------------
// Checks (`done` and `when`): the manifest parser validates them; this
// evaluates them against the command's answer.
// ---------------------------------------------------------------------------

/// A check's conditions: its `all` list, or the one it carries inline.
pub fn conditions(check: &Check) -> Vec<&Condition> {
    match &check.all {
        Some(all) => all.iter().collect(),
        None => vec![&check.condition],
    }
}

/// The value at a dot-separated `path` (a number steps into a list).
pub fn lookup<'a>(data: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(data, |at, key| match at {
        Value::Object(o) => o.get(key),
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => None,
    })
}

fn show(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 60 { format!("{}…", s.chars().take(60).collect::<String>()) } else { s }
}

/// Why `c` fails on `data`, or `None` when it holds. A missing path equals null.
fn failure(c: &Condition, data: &Value) -> Option<String> {
    let found = lookup(data, &c.path);
    let value = found.unwrap_or(&Value::Null);
    let is = || if found.is_none() { "missing".to_string() } else { show(value) };
    if let Some(want) = &c.equals {
        return (value != want).then(|| format!("{} is {}, not {}", c.path, is(), show(want)));
    }
    if let Some(want) = c.present {
        let there = !value.is_null();
        return (there != want).then(|| if want { format!("{} is missing", c.path) } else { format!("{} is {}", c.path, show(value)) });
    }
    if let Some(list) = &c.one_of {
        return (!list.contains(value)).then(|| format!("{} is {}, not one of {}", c.path, is(), show(&Value::Array(list.clone()))));
    }
    if let Some(list) = &c.not_in {
        return list.contains(value).then(|| format!("{} is {}", c.path, is()));
    }
    // No test at all (the parser refuses one): it cannot pass.
    Some(format!("the condition on {} has no test", c.path))
}

/// Whether `data` (the command's answer) passes every condition, and a sentence saying so.
pub fn judge(check: &Check, data: &Value) -> (bool, String) {
    for c in conditions(check) {
        if let Some(why) = failure(c, data) {
            return (false, format!("{}: {why}", check.command));
        }
    }
    (true, format!("{}: every condition holds", check.command))
}

/// A plugin's answer with the SDK envelope (`{ok, data}`) taken off, as the
/// dashboard's plugin screens do; `ok: false` is a failure.
pub fn unwrap_reply(reply: Value) -> Result<Value, String> {
    match reply {
        Value::Object(mut o) => {
            if o.get("ok") == Some(&Value::Bool(false)) {
                let why = match o.get("error") {
                    Some(Value::String(s)) => s.clone(),
                    Some(e) => e.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| e.to_string()),
                    None => "the plugin could not answer".into(),
                };
                return Err(why);
            }
            match o.remove("data") {
                Some(data) => Ok(data),
                None => Ok(Value::Object(o)),
            }
        }
        other => Ok(other),
    }
}

/// Which check of a step: `done` (the default) or `when`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Done,
    When,
}

/// The outcome of a check.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Outcome {
    pub passed: bool,
    pub detail: String,
}

/// How to run `step`'s check of `record`: the connector to send it to, and
/// the check. `Err` is the outcome without sending anything: no such check,
/// or a command that is undeclared or journalled. (The parser refuses such a
/// manifest at load; this holds the line again, as a check must never have a
/// side effect.)
pub fn plan_check(record: &PluginRecord, step_id: &str, which: Which) -> Result<(String, Check), Outcome> {
    let not = |detail: String| Outcome { passed: false, detail };
    let manifest = record.manifest.as_ref().ok_or_else(|| not(format!("{} has no manifest that loads", record.id)))?;
    let setup = declared(record).ok_or_else(|| not(format!("{} declares no setup", record.id)))?;
    let s = find_step(setup, step_id).ok_or_else(|| not(format!("{} has no setup step {step_id:?}", record.id)))?;
    let check = match which {
        // No `when`: the step is always shown.
        Which::When => s.when.clone().ok_or_else(|| Outcome { passed: true, detail: "the step is always shown".into() })?,
        Which::Done => match &s.kind {
            StepKind::Screen { done: Some(done), .. } => done.clone(),
            _ => return Err(not("the step has no done check: it is done when its screen says so".into())),
        },
    };
    let connector = manifest
        .connector_for(&check.command)
        .map(|c| c.id.clone())
        .ok_or_else(|| not(format!("{} is not one of the plugin's commands", check.command)))?;
    if manifest.is_journalled(&check.command) {
        return Err(not(format!("{} has side effects (it is journalled), so it is never sent as a check", check.command)));
    }
    Ok((connector, check))
}

fn describe_forward_error(e: ForwardError) -> String {
    match e {
        ForwardError::Refused(r) => r.message(),
        ForwardError::NotRunning { plugin_id } => format!("{plugin_id} is not running"),
        ForwardError::Call(CallError::Timeout { .. }) => format!("no answer within {} s", CHECK_TIMEOUT.as_secs()),
        ForwardError::Call(CallError::Plugin { message, .. }) => message,
        ForwardError::Call(e) => e.to_string(),
        ForwardError::Internal(m) => m,
    }
}

/// The outcome of `check` from the plugin's reply (or why there is none).
pub fn outcome(check: &Check, reply: Result<Value, String>) -> Outcome {
    match reply.and_then(unwrap_reply) {
        Ok(data) => {
            let (passed, detail) = judge(check, &data);
            Outcome { passed, detail }
        }
        Err(why) => Outcome { passed: false, detail: format!("{}: {why}", check.command) },
    }
}

/// Run `step`'s check of plugin `id`, through the same gated path as the
/// command route (`PluginHost::forward_connector`), with no payload and no
/// idempotency key (so the gate refuses anything journalled), for at most
/// [`CHECK_TIMEOUT`]. An error or no answer is "not passed".
pub async fn run_check(plugins: &PluginRegistryHandle, host: &Arc<PluginHost>, id: &str, step: &str, which: Which) -> Outcome {
    let planned = {
        let Ok(reg) = plugins.lock() else {
            return Outcome { passed: false, detail: "the plugin registry is unavailable".into() };
        };
        match reg.get(id) {
            Some(record) => plan_check(record, step, which),
            None => Err(Outcome { passed: false, detail: format!("{id} is not installed") }),
        }
    };
    let (connector, check) = match planned {
        Ok(p) => p,
        Err(o) => return o,
    };
    let (host, command) = (host.clone(), check.command.clone());
    let call = tokio::task::spawn_blocking(move || host.forward_connector(&connector, &command, None, None, CHECK_TIMEOUT));
    let reply = match tokio::time::timeout(CHECK_TIMEOUT + Duration::from_secs(1), call).await {
        Ok(Ok(Ok(v))) => Ok(v),
        Ok(Ok(Err(e))) => Err(describe_forward_error(e)),
        Ok(Err(join)) => Err(format!("the check did not run: {join}")),
        Err(_) => Err(format!("no answer within {} s", CHECK_TIMEOUT.as_secs())),
    };
    outcome(&check, reply)
}

// ---------------------------------------------------------------------------
// The plugin catalog
// ---------------------------------------------------------------------------

const CATALOG: &str = include_str!("../resources/plugin-catalog.json");

/// Expand a catalog folder template: `{exeDir}` (beside this program),
/// `{sources}` (`OAIY_PLUGIN_SOURCES`), `{repo}` (the repository a build from
/// it came from). `None` when a part is not there.
fn expand(template: &str, sources: Option<&Path>) -> Option<PathBuf> {
    let mut out = template.to_string();
    if out.contains("{exeDir}") {
        let exe = std::env::current_exe().ok()?;
        out = out.replace("{exeDir}", &exe.parent()?.to_string_lossy());
    }
    if out.contains("{sources}") {
        out = out.replace("{sources}", &sources?.to_string_lossy());
    }
    if out.contains("{repo}") {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        out = out.replace("{repo}", &repo.to_string_lossy());
    }
    let path = PathBuf::from(out);
    Some(std::path::absolute(&path).unwrap_or(path))
}

/// Whether `dir` holds a plugin with the id `id` (its manifest says so).
fn holds_plugin(dir: &Path, id: &str) -> bool {
    std::fs::read_to_string(dir.join("manifest.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|m| m.get("id").and_then(Value::as_str) == Some(id))
}

/// The catalog, each plugin with where it can be installed from on this
/// machine (`source.path`, or null: the person chooses the folder) and
/// whether it is installed.
pub fn catalog(plugins_root: &Path, records: &[PluginRecord], sources: Option<&Path>) -> Value {
    let parsed: Value = serde_json::from_str(CATALOG).unwrap_or_else(|_| json!({"plugins": []}));
    let plugins: Vec<Value> = parsed
        .get("plugins")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|entry| {
                    let id = entry.get("id").and_then(Value::as_str)?;
                    let mut entry = entry.clone();
                    let own = plugins_root.join(id);
                    let found = entry
                        .pointer("/source/lookIn")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .filter_map(|t| expand(t, sources))
                        // Never offer its own installed copy as where to install it from.
                        .find(|p| p != &own && holds_plugin(p, id));
                    if let Some(source) = entry.get_mut("source").and_then(Value::as_object_mut) {
                        source.remove("lookIn");
                        source.insert("path".into(), found.map_or(Value::Null, |p| Value::String(p.to_string_lossy().to_string())));
                    }
                    let installed = records.iter().find(|r| r.id == id);
                    let o = entry.as_object_mut()?;
                    o.insert("installed".into(), Value::Bool(installed.is_some()));
                    o.insert(
                        "installedVersion".into(),
                        installed.and_then(|r| r.manifest.as_ref()).map_or(Value::Null, |m| Value::String(m.version.clone())),
                    );
                    Some(entry)
                })
                .collect()
        })
        .unwrap_or_default();
    json!({ "plugins": plugins })
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// What the routes share: the record, the plugins, and the host that runs checks.
#[derive(Clone)]
pub struct Ctx {
    store: Arc<Mutex<Store>>,
    plugins: PluginRegistryHandle,
    host: Arc<PluginHost>,
}

impl Ctx {
    pub fn new(store: Store, plugins: PluginRegistryHandle, host: Arc<PluginHost>) -> Ctx {
        Ctx { store: Arc::new(Mutex::new(store)), plugins, host }
    }
}

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

fn with_store<T>(ctx: &Ctx, f: impl FnOnce(&mut Store) -> Result<T, String>) -> Result<T, Response> {
    let mut store = ctx.store.lock().map_err(|_| fail(StatusCode::INTERNAL_SERVER_ERROR, "the setup record is unavailable"))?;
    f(&mut store).map_err(|e| fail(StatusCode::BAD_REQUEST, e))
}

async fn get_state(State(ctx): State<Ctx>) -> Response {
    match with_store(&ctx, |s| Ok(s.state().clone())) {
        Ok(state) => Json(state).into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PutBody {
    first_run: FirstRun,
}

async fn put_state(State(ctx): State<Ctx>, Json(body): Json<PutBody>) -> Response {
    match with_store(&ctx, |s| s.set_first_run(body.first_run).map(Clone::clone)) {
        Ok(state) => Json(state).into_response(),
        Err(r) => r,
    }
}

async fn get_catalog(State(ctx): State<Ctx>) -> Response {
    let (root, records) = match ctx.plugins.lock() {
        Ok(reg) => (reg.root().to_path_buf(), reg.list()),
        Err(_) => return fail(StatusCode::INTERNAL_SERVER_ERROR, "the plugin registry is unavailable"),
    };
    let sources = std::env::var_os("OAIY_PLUGIN_SOURCES").filter(|s| !s.is_empty()).map(PathBuf::from);
    Json(catalog(&root, &records, sources.as_deref())).into_response()
}

/// A plugin's record, or the response that says it is not there.
fn record_of(ctx: &Ctx, id: &str) -> Result<PluginRecord, Response> {
    if !valid_plugin_id(id) {
        return Err(fail(StatusCode::BAD_REQUEST, format!("{id:?} is not a plugin id")));
    }
    let reg = ctx.plugins.lock().map_err(|_| fail(StatusCode::INTERNAL_SERVER_ERROR, "the plugin registry is unavailable"))?;
    reg.get(id).cloned().ok_or_else(|| fail(StatusCode::NOT_FOUND, format!("{id} is not installed")))
}

async fn plugin_detail(State(ctx): State<Ctx>, UrlPath(id): UrlPath<String>) -> Response {
    let record = match record_of(&ctx, &id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let state = match with_store(&ctx, |s| Ok(s.state().plugins.get(&id).cloned().unwrap_or_default())) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let setup = declared(&record);
    let capabilities = resolved_capabilities(&record);
    Json(json!({
        "pluginId": id,
        "name": record.manifest.as_ref().map(|m| m.name.clone()).unwrap_or_else(|| id.clone()),
        "setup": setup,
        "capabilities": capabilities,
        "legacyCapabilities": record.legacy_capabilities,
        "unknownCapabilities": record.unknown_capabilities,
        "permissionsAccepted": accepted_covers(state.permissions_accepted.as_ref(), &capabilities),
        "needsSetup": setup.as_ref().is_some_and(|d| d.version > state.version),
        "state": state,
    }))
    .into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepBody {
    #[serde(default = "done_status")]
    status: StepStatus,
}

fn done_status() -> StepStatus {
    StepStatus::Done
}

async fn mark_step(State(ctx): State<Ctx>, UrlPath((id, step)): UrlPath<(String, String)>, body: Option<Json<StepBody>>) -> Response {
    let status = body.map_or(StepStatus::Done, |b| b.status);
    let record = match record_of(&ctx, &id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let Some(setup) = declared(&record) else {
        return fail(StatusCode::CONFLICT, format!("{id} declares no setup"));
    };
    let accepted = if step == PERMISSIONS {
        if status == StepStatus::Skipped {
            return fail(StatusCode::BAD_REQUEST, "the permissions step cannot be skipped");
        }
        // What is accepted is what the plugin asks for now, worked out here: a caller cannot name a narrower list.
        Some(resolved_capabilities(&record))
    } else {
        if find_step(setup, &step).is_none() {
            return fail(StatusCode::NOT_FOUND, format!("{id} has no setup step {step:?}"));
        }
        None
    };
    match with_store(&ctx, |s| s.mark_step(&id, &step, status, accepted).map(Clone::clone)) {
        Ok(state) => Json(state).into_response(),
        Err(r) => r,
    }
}

async fn finish(State(ctx): State<Ctx>, UrlPath(id): UrlPath<String>) -> Response {
    let record = match record_of(&ctx, &id) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let Some(setup) = declared(&record) else {
        return fail(StatusCode::CONFLICT, format!("{id} declares no setup"));
    };
    let capabilities = resolved_capabilities(&record);
    let result = with_store(&ctx, |s| {
        let accepted = s.state().plugins.get(&id).and_then(|p| p.permissions_accepted.as_ref());
        if !accepted_covers(accepted, &capabilities) {
            return Err("accept what the plugin may do first (its permissions step)".into());
        }
        s.finish_plugin(&id, setup.version).map(Clone::clone)
    });
    match result {
        Ok(state) => Json(state).into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct CheckQuery {
    #[serde(default)]
    check: Option<String>,
}

async fn check_step(State(ctx): State<Ctx>, UrlPath((id, step)): UrlPath<(String, String)>, Query(q): Query<CheckQuery>) -> Response {
    let which = match q.check.as_deref() {
        None | Some("done") => Which::Done,
        Some("when") => Which::When,
        Some(other) => return fail(StatusCode::BAD_REQUEST, format!("check is done or when, not {other:?}")),
    };
    if !valid_plugin_id(&id) || !valid_step_id(&step) {
        return fail(StatusCode::BAD_REQUEST, "not a plugin or step id");
    }
    Json(run_check(&ctx.plugins, &ctx.host, &id, &step, which).await).into_response()
}

pub fn router(ctx: Ctx) -> Router {
    Router::new()
        .route("/api/setup", get(get_state).put(put_state))
        .route("/api/setup/catalog", get(get_catalog))
        .route("/api/setup/plugins/:id", get(plugin_detail))
        .route("/api/setup/plugins/:id/steps/:step", post(mark_step))
        .route("/api/setup/plugins/:id/finish", post(finish))
        .route("/api/setup/plugins/:id/check/:step", post(check_step))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sandbox(PathBuf);
    impl Sandbox {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let p = std::env::temp_dir().join(format!("oaiy-setup-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
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

    /// Aokie's v4 manifest (the manifest parser's fixture), and the service
    /// definition its `serviceDefinitions` names.
    const AOKIE_V4: &str = include_str!("plugins/fixtures/aokie-v4.manifest.json");
    const AOKIE_PHONE: &str = include_str!("plugins/fixtures/aokie-phone.definition.json");

    /// The fixture's setup (Aokie's screen steps) with the kinds it does not use yet.
    fn aokie_setup() -> Value {
        let mut setup = serde_json::from_str::<Value>(AOKIE_V4).unwrap()["setup"].clone();
        let steps = setup["steps"].as_array_mut().unwrap();
        steps.push(json!({ "id": "speech", "kind": "requirements", "title": "Hearing and speaking",
            "requires": [ { "kind": "service", "id": "oaiy-voice" }, { "kind": "engineModel", "group": "llm" } ] }));
        steps.push(json!({ "id": "answer", "kind": "host", "action": "phone.answerWithOaiy", "title": "Answer calls and texts with OAIY" }));
        steps.push(json!({ "id": "later", "kind": "host", "action": "phone.teleport", "title": "Needs a newer OAIY" }));
        setup
    }

    /// Aokie at schemaVersion 4 in `root/<id>`, with `setup` (or none).
    fn write_plugin(root: &Path, id: &str, setup: Option<Value>) {
        let dir = root.join(id);
        std::fs::create_dir_all(dir.join("definitions")).unwrap();
        let mut m: Value = serde_json::from_str(AOKIE_V4).unwrap();
        m["id"] = json!(id);
        match setup {
            Some(s) => m["setup"] = s,
            None => {
                m.as_object_mut().unwrap().remove("setup");
            }
        }
        std::fs::write(dir.join("manifest.json"), m.to_string()).unwrap();
        std::fs::write(dir.join("definitions").join("phone.json"), AOKIE_PHONE).unwrap();
    }

    fn record(root: &Path, id: &str) -> PluginRecord {
        let reg = crate::plugins::registry::new_handle(root.to_path_buf());
        let mut r = reg.lock().unwrap();
        r.scan();
        let rec = r.get(id).cloned().expect("the plugin is listed");
        assert!(rec.manifest.is_some(), "the plugin loads: {:?}", rec.reason);
        rec
    }

    /// A check as the manifest carries it.
    fn check(v: Value) -> Check {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn a_new_desktop_starts_the_first_run_wizard_and_one_in_use_does_not() {
        let fresh = Sandbox::new("fresh");
        let store = Store::open(&fresh.0, || None);
        assert!(!store.state().first_run.finished);
        assert!(fresh.0.join(FILE).is_file(), "the record is made at once");

        let used = Sandbox::new("used");
        let store = Store::open(&used.0, || Some("a language model was already chosen in Engines".into()));
        assert!(store.state().first_run.finished);
        assert_eq!(store.state().first_run.migrated.as_deref(), Some("a language model was already chosen in Engines"));
        // Asked only when the record is first made: reopening keeps what was recorded.
        let again = Store::open(&used.0, || panic!("not asked again"));
        assert!(again.state().first_run.finished);
    }

    #[test]
    fn a_record_that_cannot_be_read_is_kept_aside_and_brings_no_wizard() {
        let sb = Sandbox::new("bad");
        std::fs::write(sb.0.join(FILE), "{ not json").unwrap();
        let store = Store::open(&sb.0, || None);
        assert!(store.state().first_run.finished);
        assert!(store.state().first_run.migrated.as_deref().unwrap().contains("could not be read"));
        assert!(sb.0.join("setup.json.unreadable").is_file());
        // And the record is readable again.
        let text = std::fs::read_to_string(sb.0.join(FILE)).unwrap();
        assert!(serde_json::from_str::<SetupState>(&text).is_ok());
    }

    #[test]
    fn signs_of_use_are_a_ready_provider_chatgpt_a_chosen_model_or_a_plugin() {
        let sb = Sandbox::new("signs");
        let plugins = sb.0.join("plugins");
        assert_eq!(in_use_signal(&sb.0, &plugins, &[]), None, "nothing yet");
        // The engines' configuration as a fresh install makes it: no model, so no sign.
        std::fs::create_dir_all(sb.0.join("engines")).unwrap();
        std::fs::write(sb.0.join("engines/oaiy-studio.json"), r#"{"llm":{"default_model":"","models":[]}}"#).unwrap();
        assert_eq!(in_use_signal(&sb.0, &plugins, &[]), None);
        std::fs::write(sb.0.join("engines/oaiy-studio.json"), r#"{"llm":{"default_model":"Qwen3.8-Flash-Next","models":[{"name":"Qwen3.8-Flash-Next","path":"x.gguf"}]}}"#).unwrap();
        assert!(in_use_signal(&sb.0, &plugins, &[]).unwrap().contains("Engines"));
        std::fs::remove_file(sb.0.join("engines/oaiy-studio.json")).unwrap();

        std::fs::create_dir_all(plugins.join(".staging-x")).unwrap();
        std::fs::write(plugins.join(".staging-x/manifest.json"), "{}").unwrap();
        assert_eq!(in_use_signal(&sb.0, &plugins, &[]), None, "an install in progress is not a plugin");
        write_plugin(&plugins, "aokie", None);
        assert!(in_use_signal(&sb.0, &plugins, &[]).unwrap().contains("aokie"));
        std::fs::remove_dir_all(&plugins).unwrap();

        std::fs::create_dir_all(sb.0.join("ai/codex-home")).unwrap();
        std::fs::write(sb.0.join("ai/codex-home/auth.json"), "{}").unwrap();
        assert!(in_use_signal(&sb.0, &plugins, &[]).unwrap().contains("ChatGPT"));
    }

    #[test]
    fn the_first_run_part_is_replaced_whole_but_the_migration_note_stays() {
        let sb = Sandbox::new("put");
        let mut store = Store::open(&sb.0, || Some("a plugin (aokie) was already installed".into()));
        store
            .set_first_run(FirstRun {
                finished: false,
                position: Some("plugin:aokie".into()),
                skipped: vec!["connect".into(), "connect".into()],
                chosen_plugins: vec!["aokie".into()],
                migrated: Some("forged".into()),
            })
            .unwrap();
        let f = &store.state().first_run;
        assert!(!f.finished);
        assert_eq!(f.position.as_deref(), Some("plugin:aokie"));
        assert_eq!(f.skipped, ["connect"], "kept once");
        assert_eq!(f.migrated.as_deref(), Some("a plugin (aokie) was already installed"), "the desktop's note, not the caller's");
        // Saved: a reopen reads the same.
        assert_eq!(Store::open(&sb.0, || None).state(), store.state());
        // Names are checked.
        let bad = FirstRun { chosen_plugins: vec!["../evil".into()], ..FirstRun::default() };
        assert!(store.set_first_run(bad).is_err());
        assert_eq!(store.state().first_run.chosen_plugins, ["aokie"], "a refused change changes nothing");
    }

    #[test]
    fn steps_are_recorded_done_or_skipped_and_a_plugin_finished_at_its_version() {
        let sb = Sandbox::new("steps");
        let mut store = Store::open(&sb.0, || None);
        store.mark_step("aokie", "consent", StepStatus::Skipped, None).unwrap();
        store.mark_step("aokie", "consent", StepStatus::Done, None).unwrap();
        store.mark_step("aokie", "pair", StepStatus::Skipped, None).unwrap();
        store.mark_step("aokie", PERMISSIONS, StepStatus::Done, Some(vec!["flow.run".into()])).unwrap();
        store.finish_plugin("aokie", 2).unwrap();
        let p = &store.state().plugins["aokie"];
        assert_eq!(p.done, ["consent", PERMISSIONS]);
        assert_eq!(p.skipped, ["pair"]);
        assert_eq!(p.version, 2);
        assert_eq!(p.permissions_accepted.as_deref(), Some(&["flow.run".to_string()][..]));
        store.mark_step("aokie", "consent", StepStatus::Todo, None).unwrap();
        assert!(!store.state().plugins["aokie"].done.contains(&"consent".to_string()));
        assert!(store.mark_step("aokie", "Bad Id", StepStatus::Done, None).is_err());
        assert!(store.mark_step("Aokie!", "consent", StepStatus::Done, None).is_err());
        // The file is whole after every change, with nothing left beside it.
        let saved: SetupState = serde_json::from_str(&std::fs::read_to_string(sb.0.join(FILE)).unwrap()).unwrap();
        assert_eq!(&saved, store.state());
        let leftovers: Vec<_> = std::fs::read_dir(&sb.0).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains(".tmp-")).collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn the_record_speaks_camel_case() {
        let mut s = SetupState::default();
        s.first_run.chosen_plugins = vec!["aokie".into()];
        s.plugins.insert("aokie".into(), PluginSetup { version: 1, permissions_accepted: Some(vec![]), ..Default::default() });
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["firstRun"]["chosenPlugins"], json!(["aokie"]));
        assert_eq!(v["plugins"]["aokie"]["permissionsAccepted"], json!([]));
        assert_eq!(v["plugins"]["aokie"]["version"], 1);
    }

    #[test]
    fn a_plugins_setup_is_the_manifests_typed_section_with_its_defaults() {
        let sb = Sandbox::new("declared");
        write_plugin(&sb.0, "aokie", Some(aokie_setup()));
        write_plugin(&sb.0, "plain", None);
        let rec = record(&sb.0, "aokie");
        let d = declared(&rec).expect("declared");
        assert_eq!(d.version, 1, "no version is 1");
        assert_eq!(d.title, "Set up Aokie Phone Bridge");
        let ids: Vec<&str> = d.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["consent", "dongle", "pair", "speech", "answer"], "a host step this OAIY cannot run is left out");
        assert!(find_step(d, "pair").is_some() && find_step(d, "nope").is_none());
        assert!(declared(&record(&sb.0, "plain")).is_none());

        let mut titled = aokie_setup();
        titled["version"] = json!(3);
        titled["title"] = json!("Set up the AI Receptionist");
        write_plugin(&sb.0, "titled", Some(titled));
        let rec = record(&sb.0, "titled");
        let d = declared(&rec).unwrap();
        assert_eq!((d.version, d.title.as_str()), (3, "Set up the AI Receptionist"));
    }

    #[test]
    fn a_checks_conditions_are_its_inline_one_or_its_all_list() {
        let single = check(json!({ "command": "phone.status", "path": "connected", "equals": true }));
        assert_eq!(conditions(&single).len(), 1);
        assert_eq!(conditions(&single)[0].equals, Some(json!(true)));
        let all = check(json!({ "command": "phone.status", "all": [ { "path": "a", "equals": null }, { "path": "b", "present": true }, { "path": "c", "in": [1] }, { "path": "d", "notIn": ["x"] } ] }));
        let list = conditions(&all);
        assert_eq!(list.len(), 4);
        assert_eq!(list[0].equals, Some(Value::Null), "equals null is a test, not a missing one");
        assert_eq!(list[2].one_of, Some(vec![json!(1)]));
    }

    #[test]
    fn a_check_is_judged_on_the_answer_with_a_missing_path_equal_to_null() {
        let c = check(json!({ "command": "phone.status", "all": [
            { "path": "connected", "equals": true },
            { "path": "pairingConfirm", "equals": null },
            { "path": "device.name", "present": true },
            { "path": "mode", "notIn": ["native", "auto"] },
            { "path": "list.1", "in": ["b"] } ] }));
        let ok = json!({ "connected": true, "device": { "name": "Pixel" }, "mode": "dongle", "list": ["a", "b"] });
        assert_eq!(judge(&c, &ok), (true, "phone.status: every condition holds".into()));
        let (passed, why) = judge(&c, &json!({ "connected": false }));
        assert!(!passed);
        assert_eq!(why, "phone.status: connected is false, not true");
        let (passed, why) = judge(&c, &json!({ "connected": true, "pairingConfirm": { "code": "123456" }, "device": { "name": "P" } }));
        assert!(!passed && why.contains("pairingConfirm"), "{why}");
        let (passed, why) = judge(&c, &json!({ "connected": true }));
        assert!(!passed && why.contains("device.name is missing"), "{why}");
        let (passed, why) = judge(&c, &json!({ "connected": true, "device": { "name": "P" }, "mode": "auto" }));
        assert!(!passed && why.contains("mode is \"auto\""), "{why}");
        // present: false passes when the path is missing or null.
        assert!(judge(&check(json!({ "command": "x", "path": "gone", "present": false })), &json!({ "gone": null })).0);
        // A missing path is null: in [null] passes, notIn [null] fails.
        assert!(judge(&check(json!({ "command": "x", "path": "gone", "in": [null] })), &json!({})).0);
        assert!(!judge(&check(json!({ "command": "x", "path": "gone", "notIn": [null] })), &json!({})).0);
    }

    #[test]
    fn the_answer_is_unwrapped_like_a_plugin_screen_does_and_a_refusal_fails() {
        assert_eq!(unwrap_reply(json!({ "ok": true, "data": { "connected": true } })), Ok(json!({ "connected": true })));
        assert_eq!(unwrap_reply(json!({ "connected": true })), Ok(json!({ "connected": true })));
        assert_eq!(unwrap_reply(json!({ "ok": false, "error": { "message": "no radio" } })), Err("no radio".into()));
        let c = check(json!({ "command": "dongle.diagnostics", "path": "radio.initialized", "equals": true }));
        assert_eq!(outcome(&c, Err("no answer within 5 s".into())), Outcome { passed: false, detail: "dongle.diagnostics: no answer within 5 s".into() });
        assert!(outcome(&c, Ok(json!({ "ok": true, "data": { "radio": { "initialized": true } } }))).passed);
    }

    #[test]
    fn a_check_is_planned_only_for_a_declared_command_that_is_not_journalled() {
        let sb = Sandbox::new("plan");
        write_plugin(&sb.0, "aokie", Some(aokie_setup()));
        let mut r = record(&sb.0, "aokie");
        let (connector, c) = plan_check(&r, "pair", Which::Done).unwrap();
        assert_eq!((connector.as_str(), c.command.as_str()), ("aokie", "phone.status"));
        let (_, when) = plan_check(&r, "dongle", Which::When).unwrap();
        assert_eq!(when.command, "settings.get");
        // A step with no `when` is always shown; a step with no `done` is done when its screen says.
        assert!(plan_check(&r, "consent", Which::When).unwrap_err().passed);
        assert!(!plan_check(&r, "speech", Which::Done).unwrap_err().passed);
        assert!(plan_check(&r, "nope", Which::Done).unwrap_err().detail.contains("no setup step"));
        // The parser refuses a journalled check at load; were one there, it is still never sent.
        let setup = r.manifest.as_mut().unwrap().setup.as_mut().unwrap();
        setup.steps.push(serde_json::from_value(json!({ "id": "dial", "kind": "screen", "title": "Bad", "screen": "receptionist-home", "view": "phone",
            "done": { "command": "call.dial", "path": "ok", "equals": true } })).unwrap());
        let journalled = plan_check(&r, "dial", Which::Done).unwrap_err();
        assert!(!journalled.passed && journalled.detail.contains("journalled"), "{}", journalled.detail);
    }

    #[tokio::test]
    async fn a_check_of_a_plugin_that_is_not_running_goes_through_the_gate_and_does_not_pass() {
        let sb = Sandbox::new("gate");
        write_plugin(&sb.0.join("plugins"), "aokie", Some(aokie_setup()));
        let host = PluginHost::new(
            crate::plugins::registry::new_handle(sb.0.join("plugins")),
            crate::bridge::ledger::new_handle(),
            Arc::new(Mutex::new(crate::plugins::TriggerStore::load(sb.0.join("triggers.json")))),
            crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );
        host.registry.lock().unwrap().scan();
        let plugins = host.registry.clone();
        let out = run_check(&plugins, &host, "aokie", "pair", Which::Done).await;
        assert!(!out.passed);
        assert!(out.detail.starts_with("phone.status: "), "{}", out.detail);
        let out = run_check(&plugins, &host, "missing", "pair", Which::Done).await;
        assert_eq!(out, Outcome { passed: false, detail: "missing is not installed".into() });
    }

    #[test]
    fn accepting_covers_what_the_plugin_asks_for_now() {
        let now = vec!["connector.aokie.phone.status".to_string(), "flow.run".to_string()];
        assert!(!accepted_covers(None, &now));
        assert!(accepted_covers(Some(&now.clone()), &now));
        // An update that asks for more asks again.
        assert!(!accepted_covers(Some(&vec!["flow.run".to_string()]), &now));
        assert!(accepted_covers(Some(&vec![]), &[]), "nothing asked, nothing to accept");
    }

    #[test]
    fn the_catalog_says_where_a_plugin_can_be_installed_from_and_never_its_own_copy() {
        let sb = Sandbox::new("catalog");
        let root = sb.0.join("plugins");
        let sources = sb.0.join("sources");
        let v = catalog(&root, &[], None);
        let aokie = v["plugins"].as_array().unwrap().iter().find(|p| p["id"] == "aokie").expect("Aokie is in the catalog").clone();
        assert_eq!(aokie["source"]["kind"], "folder", "installed from a folder on this machine, not downloaded");
        assert!(aokie["source"].get("lookIn").is_none(), "the templates stay on the desktop");
        assert_eq!(aokie["installed"], false);
        assert!(aokie.get("url").is_none() && aokie["source"].get("url").is_none(), "no download address is made up");

        write_plugin(&sources, "aokie", None);
        let v = catalog(&root, &[], Some(&sources));
        let found = v["plugins"][0]["source"]["path"].as_str().expect("found in OAIY_PLUGIN_SOURCES");
        assert!(Path::new(found).ends_with("aokie"), "{found}");
        // A folder with another plugin in it is not Aokie.
        std::fs::remove_dir_all(sources.join("aokie")).unwrap();
        write_plugin(&sources, "other", None);
        std::fs::rename(sources.join("other"), sources.join("aokie")).unwrap();
        assert_eq!(catalog(&root, &[], Some(&sources))["plugins"][0]["source"]["path"], Value::Null);

        write_plugin(&root, "aokie", None);
        let rec = record(&root, "aokie");
        let v = catalog(&root, &[rec], Some(&root));
        assert_eq!(v["plugins"][0]["installed"], true);
        assert_eq!(v["plugins"][0]["installedVersion"], "0.1.0");
        assert_eq!(v["plugins"][0]["source"]["path"], Value::Null, "its own installed copy is not a source");
    }

    #[test]
    fn step_and_plugin_ids_are_checked() {
        for ok in ["consent", "pair", "a", "plugin:aokie", "engine-model"] {
            assert!(valid_step_id(ok), "{ok}");
        }
        let long = "x".repeat(41);
        for bad in ["", "Consent", "1st", "a b", "plugin:", "plugin:../x", long.as_str()] {
            assert!(!valid_step_id(bad), "{bad}");
        }
        assert!(valid_plugin_id("aokie") && valid_plugin_id("my_plugin-2"));
        assert!(!valid_plugin_id("") && !valid_plugin_id("../x") && !valid_plugin_id("Aokie"));
    }
}
