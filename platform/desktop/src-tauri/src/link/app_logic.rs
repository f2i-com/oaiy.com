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
//!                      └─► one `entry` job per script ──► warm script host ──► effects
//!                                                            └─► POST/PUT records
//!                                                            └─► storage marker
//!                                                            └─► connector command
//! ```
//!
//! # Four rules everything here follows
//!
//! **There is no second script engine.** The scripts are JavaScript, and this
//! crate does not embed a JS runtime to run them. Each one becomes a leaf-script
//! `entry` job on the warm script host ([`crate::bridge::script_host`]), which
//! is the same ZIPP engine the flow runner uses — the identical reasoning as
//! `bridge/worker.rs`: a second engine kept "in sync" is the failure this
//! architecture exists to avoid.
//!
//! **One batch per event, not one process per app.** The lane used to wrap an
//! app's scripts into a one-node-per-script graph and spawn the CLI for it, so
//! every plugin event carrying app logic cost a Node start-up per app. Now every
//! script of every app of one event goes to the host in ONE batch (split only
//! when it would not fit — see [`batch_ranges`]), which is the third and last
//! batch an event sends: trigger conditions, then flow-binding conditions after
//! the provider fetch, then this. They are not merged: they are decided at
//! different moments by different owners, and they serialise on the host's own
//! process-wide lock anyway.
//!
//! Because the batch runs first and the effects are applied afterwards, every
//! app's markers are read BEFORE any script runs. That is the same snapshot
//! every script of one app already saw; markers are per app and an app is
//! refused if the provider lists it twice, so no script can observe another's
//! writes either way.
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
use crate::bridge::script_host::{
    batch_ranges, batch_request_with, HostError, JobError, Prelude, ScriptBatch,
};

/// How long a fetched script set is reused.
///
/// Longer than the flow bindings' minute because a script is part of an app's
/// published version rather than something a user toggles, and the document is
/// large — it carries every script of every installed app. Short enough that
/// reinstalling an app is picked up while somebody is still watching.
const CATALOG_TTL: Duration = Duration::from_secs(120);

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
pub struct AppEntry {
    doc: Value,
}

/// One script, once the catalogue names have been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptRef<'a> {
    pub id: &'a str,
    /// The source itself, defining `run(ctx)` in the language below.
    pub source: &'a str,
    /// The logic language it is in, as the engine names languages: [`JAVASCRIPT`]
    /// for a script that names none, or where the provider says nothing of it.
    pub language: &'a str,
}

/// The engine's names for the two logic languages this lane can run a script
/// in. Not a provider's vocabulary: a leaf-script job is OAIY's own protocol
/// (`protocol/v1/script-request.schema.json`), and these are its language ids.
pub const JAVASCRIPT: &str = "javascript";
pub const PYTHON: &str = "python";

/// What a script that names a language in something other than words is said
/// to be in: not JavaScript, so never run as it.
const UNREADABLE_LANGUAGE: &str = "(a language named in something other than words)";

impl AppEntry {
    /// An app as the provider listed it.
    fn listed(doc: Value) -> Self {
        Self { doc }
    }

    /// The sources this app says it takes events from, where the provider's
    /// catalogue says what that list is called. None when it does not, or the
    /// app lists none: nothing is known of what this app takes.
    fn sources<'a>(&'a self, c: &AppLogicCatalogue) -> Vec<&'a str> {
        let Some(list) = c.sources.as_ref() else {
            return Vec::new();
        };
        self.doc
            .get(list)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// A stable id for this app, for the storage file and the logs.
    fn key<'a>(&'a self, c: &AppLogicCatalogue) -> Option<&'a str> {
        let app = self.doc.get(&c.app_block)?;
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
    ///
    /// A script the author switched OFF is not one of them. The flag reads as
    /// "not disabled" — the key is usually absent, and absent means a script
    /// that runs — so only an explicit `false` skips it. Without this the
    /// desktop would keep running headlessly what the browser had stopped
    /// running, and the author would have no way to tell.
    fn event_scripts<'a>(&'a self, c: &AppLogicCatalogue, hook: &str) -> Vec<ScriptRef<'a>> {
        self.doc
            .get(&c.logic_block)
            .and_then(|l| l.get(&c.scripts))
            .and_then(Value::as_array)
            .map(|scripts| {
                scripts
                    .iter()
                    .filter(|s| s.get(&c.script_hook).and_then(Value::as_str) == Some(hook))
                    .filter(|s| s.get(&c.script_enabled).and_then(Value::as_bool) != Some(false))
                    .filter_map(|s| {
                        let source = s.get(&c.script_source).and_then(Value::as_str)?;
                        if source.trim().is_empty() {
                            return None;
                        }
                        Some(ScriptRef {
                            id: s.get(&c.script_id).and_then(Value::as_str).unwrap_or("").trim(),
                            source,
                            language: script_language(c, s),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// This app's forms, as (portable key, the account's own id) pairs. Either
    /// side blank is not a form a script can name, so it is not offered.
    fn forms<'a>(&'a self, c: &AppLogicCatalogue) -> Vec<(&'a str, &'a str)> {
        self.doc
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

    /// The id of the form a script names by that id itself, or by the name the
    /// app shows the form under (where the provider says what that is called).
    fn form_named<'a>(&'a self, c: &AppLogicCatalogue, key: &str) -> Option<&'a str> {
        let forms = self.doc.get(&c.forms).and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
        let id_of = |f: &'a Value| f.get(&c.form_id).and_then(Value::as_str).map(str::trim).filter(|id| !id.is_empty());
        forms.iter().filter_map(id_of).find(|id| *id == key).or_else(|| {
            let label = c.form_label.as_ref()?;
            forms
                .iter()
                .find(|f| f.get(label).and_then(Value::as_str).map(str::trim) == Some(key))
                .and_then(id_of)
        })
    }
}

/// The language a script says it is in, where the catalogue says what that field is called: absent, null or blank
/// is JavaScript, as the provider's own runtime reads it.
fn script_language<'a>(c: &AppLogicCatalogue, script: &'a Value) -> &'a str {
    let Some(field) = c.script_language.as_deref() else {
        return JAVASCRIPT;
    };
    match script.get(field) {
        None | Some(Value::Null) => JAVASCRIPT,
        Some(Value::String(language)) if language.trim().is_empty() => JAVASCRIPT,
        Some(Value::String(language)) => language.trim(),
        Some(_) => UNREADABLE_LANGUAGE,
    }
}

// --- which languages this lane runs ----------------------------------------

/// The logic languages this lane can run an app script in, with `profile` as the
/// prelude the batch would carry.
///
/// JavaScript always: an `entry` job. Python where the provider names the modes
/// of its own profile a Python script is unfolded through
/// ([`AppLogicLanguages::python_modes`]) and `profile` defines every one of
/// them — the runner builds the project from those modes, as the provider's own
/// runtime does, so a script's `run(ctx)` returns its effects here just as it
/// does there. A job that names a mode the request's profile does not define is
/// refused with the WHOLE request, its JavaScript neighbours included, so a mode
/// that is not there is a language this lane does not run.
///
/// What this lane can run is not yet what it is handed: the heartbeat declares
/// a language for the whole desktop, and only one that every lane runs
/// ([`super::heartbeat::declared_languages`]); this lane asks for those alone
/// ([`languages_asked`]).
///
/// [`AppLogicLanguages::python_modes`]: super::descriptor::AppLogicLanguages::python_modes
pub fn languages_run(spec: &AppLogicSpec, profile: Option<&Value>) -> Vec<String> {
    let mut out = vec![JAVASCRIPT.to_string()];
    if python_modes(spec, profile).is_some() {
        out.push(PYTHON.to_string());
    }
    out
}

/// The modes a Python script is unfolded through, when the provider names them
/// and `profile` defines each one.
fn python_modes<'a>(spec: &'a AppLogicSpec, profile: Option<&Value>) -> Option<&'a [String]> {
    let modes = spec.languages.as_ref().map(|l| l.python_modes.as_slice()).filter(|m| !m.is_empty())?;
    let defined: std::collections::BTreeSet<&str> = profile?
        .pointer("/python/modes")?
        .as_array()?
        .iter()
        .filter_map(|mode| mode.get("name").and_then(Value::as_str))
        .collect();
    modes.iter().all(|mode| defined.contains(mode.as_str())).then_some(modes)
}

/// The languages this lane asks the provider for scripts in, and runs, for one
/// event: those the heartbeat declares for this desktop, worked out the way the
/// beat works them out, from the CLI's report and what [`languages_run`] says of
/// `prelude`.
///
/// No more than the beat says, and the reason is the provider's: it hands this
/// desktop the scripts in whatever languages the request names, without weighing
/// them against the beat, and its browser keeps every script in a language the
/// beat does not name. A language asked for here and not declared there would be
/// run in both places.
pub fn languages_asked(spec: &AppLogicSpec, prelude: Prelude<'_>) -> Vec<String> {
    let probe = crate::bridge::script_host::ScriptHost::global().engine_probe();
    super::heartbeat::declared_languages(probe.as_ref(), Some(&languages_run(spec, prelude.profile())))
}

// --- which app an event is for ---------------------------------------------

/// What the provider keeps of who takes which source's events
/// ([`AppLogicSpec::routing`]): the app the account gave each source to, and the
/// apps it counts as able to take each (where its reply says).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Assignments {
    /// Source → the app the account assigned its events to.
    chosen: BTreeMap<String, String>,
    /// Source → the apps the provider counts as able to take its events, with
    /// scripts or without.
    candidates: BTreeMap<String, Vec<String>>,
}

/// Which of the account's apps takes the events of one source, for one event.
///
/// Two apps built on one plugin (the same pack installed twice, say) both
/// act on its events. Run for both, every call wrote its records twice, once
/// into each app, and started the flows of both. The provider's rule is that a
/// source's events go to ONE app, and its account's choice decides:
///
/// * where the account assigned the source to an app, the events are that
///   app's and no other's, however many could take them, one included: the
///   scripts of no other app run, and the flows of no other app start;
/// * where it did not, they go to the only app that could take them, and to
///   none while several could (said once);
/// * where nothing says who takes the source (a provider that publishes no
///   such rule, an event no app could take), every app's work runs, as it did
///   before the rule was read.
///
/// The apps that could take an event (its `candidates`) are those that
/// would act on it, and that the provider says take its source: an app with
/// scripts for the event, or a flow bound to it, that lists the source or that
/// the provider counts as a candidate for it. An app that would act on the
/// event and takes none of its source is not one of them, and does not act on
/// it while one of them is chosen.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Route {
    source: String,
    assigned: Option<String>,
    candidates: Vec<String>,
}

impl Route {
    /// A route that says nothing: every app's work runs, as for a provider that
    /// keeps no such rule.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// The app the event goes to, when one does.
    fn chosen(&self) -> Option<&str> {
        self.assigned.as_deref().or(match self.candidates.as_slice() {
            [only] => Some(only.as_str()),
            _ => None,
        })
    }

    /// Whether `app`'s scripts run for the event: those of the app the account
    /// assigned it to alone, where it did; otherwise those of the one app that
    /// could take it, and every app's while none could.
    fn runs(&self, app: &str) -> bool {
        if self.assigned.is_none() && self.candidates.is_empty() {
            return true;
        }
        self.chosen() == Some(app)
    }

    /// Whether the flows `app` has bound to the event are held back: every app's
    /// but the one the event goes to, wherever that is decided (an assignment,
    /// or candidates), and nobody's while nothing says who takes the source.
    pub fn holds_back(&self, app: &str) -> bool {
        !self.runs(app)
    }

    /// Why no app takes the event, when several could and the account has
    /// chosen none of them: in words for the one who can choose.
    pub fn unchosen(&self) -> Option<String> {
        if self.chosen().is_some() || self.candidates.is_empty() {
            return None;
        }
        Some(format!(
            "{} of this account's apps take events from {:?} ({}) and it is assigned to none of \
             them, so none was given this event: assign {:?} to one app on the provider's site",
            self.candidates.len(),
            self.source,
            self.candidates.join(", "),
            self.source
        ))
    }

    /// The app the account assigned the source to, when it is not `app`: for a
    /// line that says why `app`'s work on the event did not run.
    pub fn assigned_elsewhere(&self, app: &str) -> Option<&str> {
        self.assigned.as_deref().filter(|chosen| *chosen != app)
    }
}

/// Whether any of the account's apps would act on an event: one in `apps` with
/// scripts for events, or one that has a flow bound to it (`bound`). Only then
/// is there anything for the account's assignment to decide, and it is asked for.
pub fn concerns_an_app(spec: &AppLogicSpec, apps: &[AppEntry], bound: &[&str]) -> bool {
    !bound.is_empty()
        || apps
            .iter()
            .any(|app| app.key(&spec.catalogue).is_some() && !app.event_scripts(&spec.catalogue, &spec.event_hook).is_empty())
}

/// Decide which app takes the events of `source`: see [`Route`]. `bound` are the
/// apps with a flow bound to the event; `assignments` what the provider keeps
/// of who takes what, where it was read (none: no app concerns the event, or the
/// provider keeps no such thing).
pub fn route(spec: &AppLogicSpec, apps: &[AppEntry], assignments: Option<&Assignments>, bound: &[&str], source: &str) -> Route {
    let c = &spec.catalogue;
    let counted = assignments.and_then(|a| a.candidates.get(source));
    let takes = |app: &str| {
        counted.is_some_and(|counted| counted.iter().any(|a| a == app))
            || apps.iter().any(|entry| entry.key(c) == Some(app) && entry.sources(c).contains(&source))
    };
    // One entry an app, however often the provider listed it or bound it (a
    // second copy is refused where the scripts are collected, and must not make
    // an app look like two that could take the event).
    let mut candidates: Vec<String> = Vec::new();
    let acting = apps
        .iter()
        .filter(|app| !app.event_scripts(c, &spec.event_hook).is_empty())
        .filter_map(|app| app.key(c))
        .chain(bound.iter().copied());
    for app in acting {
        if !source.is_empty() && takes(app) && !candidates.iter().any(|known| known == app) {
            candidates.push(app.to_string());
        }
    }
    Route {
        source: source.to_string(),
        assigned: assignments.and_then(|a| a.chosen.get(source)).cloned(),
        candidates,
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
    /// A record write that did not land because the provider could not be
    /// reached or was down: the event is sent again later (see `outbox`).
    pub transient: bool,
}

impl Outcome {
    fn err(app: &str, script: &str, effect: &str, detail: impl Into<String>) -> Self {
        Self {
            app: app.to_string(),
            script: script.to_string(),
            effect: effect.to_string(),
            detail: Err(detail.into()),
            transient: false,
        }
    }
}

/// Why a record write did not happen, and whether trying again later can help.
#[derive(Debug)]
struct Failed {
    message: String,
    transient: bool,
}

impl From<String> for Failed {
    fn from(message: String) -> Self {
        Self { message, transient: false }
    }
}

impl Failed {
    fn unreachable(what: &str, e: &reqwest::Error) -> Self {
        Self { message: format!("could not reach the provider to {what}: {}", super::net::unreachable(e)), transient: true }
    }

    /// A refusal, which is worth trying again when the provider was down or busy.
    fn refused(status: reqwest::StatusCode, message: String) -> Self {
        let s = status.as_u16();
        // 409 here is "still being taken from an earlier try" (a clash of
        // values is handled before this); a key refused is mended by linking
        // again, and the event must still be there then.
        Self { message, transient: matches!(s, 401 | 403 | 408 | 409 | 429) || s >= 500 }
    }
}

/// The key a record written for an effect is created under, so the same
/// event sent again (the provider was away) finds the record it made the
/// first time. Kept to 128 characters, as providers cap keys.
fn record_key(event_key: &str, script: &str, index: usize) -> Option<String> {
    if event_key.is_empty() {
        return None;
    }
    let key = format!("applogic:{event_key}:{script}:{index}");
    if key.len() <= 128 {
        return Some(key);
    }
    use sha2::Digest;
    let digest = sha2::Sha256::digest(key.as_bytes());
    Some(format!("applogic:{}", digest.iter().take(24).map(|b| format!("{b:02x}")).collect::<String>()))
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
    /// What the provider keeps of who takes which source's events ([`Catalog::assignments`]), beside the list and
    /// kept as long, but asked for only when an event concerns an app.
    assignments: Mutex<AssignmentsState>,
}

#[derive(Default)]
struct AssignmentsState {
    /// The account they are of ([`account_key`]).
    account: Option<String>,
    fetched: Option<(Instant, Assignments)>,
    retry_at: Option<Instant>,
}

#[derive(Default)]
struct CatalogState {
    /// The account the scripts below are of ([`account_key`]), and the languages they were asked for in: another
    /// account's are never run, nor fallen back on, and a list asked for in other languages is not this one (its
    /// scripts in a language this desktop no longer declares are the browser's again).
    account: Option<(String, Vec<String>)>,
    fetched: Option<(Instant, Vec<AppEntry>)>,
    /// When a failed fetch may be tried again. Without it a provider that is
    /// down costs every single event a full HTTP timeout ON THE EVENT THREAD,
    /// which backs the queue up until events are shed — and a shed event is one
    /// whose records are never written at all.
    retry_at: Option<Instant>,
}

/// Which account a link is, for what is kept of it: its connector, address and account id (else a digest of its
/// credential, never the credential itself).
pub(crate) fn account_key(account: &LinkedAccount) -> String {
    use sha2::{Digest, Sha256};
    let who = match &account.account_id {
        Some(id) if !id.is_empty() => id.clone(),
        _ => Sha256::digest(account.credential.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect(),
    };
    format!("{}|{}|{who}", account.connector_id, account.base_url)
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
            assignments: Mutex::new(AssignmentsState::default()),
        }
    }

    /// What the provider keeps of who takes which source's events
    /// ([`fetch_assignments`]), for an event that concerns one of the account's
    /// apps ([`concerns_an_app`]): an account none of whose apps acts on an event
    /// never asks. Kept as long as the app list is; a refresh that fails is
    /// answered with the last one read, as the list's is, and with none to fall
    /// back on it is an error, and the event waits: run without the account's
    /// choice, an event could be given to an app the account gave it away from.
    pub fn assignments(&self, account: &LinkedAccount, spec: &AppLogicSpec) -> Result<Assignments, String> {
        let key = account_key(account);
        {
            let mut state = self.assignments.lock().unwrap_or_else(|e| e.into_inner());
            if state.account.as_deref() != Some(key.as_str()) {
                *state = AssignmentsState { account: Some(key.clone()), ..AssignmentsState::default() };
            }
            match state.fetched.as_ref() {
                Some((at, held)) if at.elapsed() < CATALOG_TTL => return Ok(held.clone()),
                Some((_, held)) if state.retry_at.is_some_and(|t| Instant::now() < t) => return Ok(held.clone()),
                None if state.retry_at.is_some_and(|t| Instant::now() < t) => {
                    return Err("which of the account's apps takes which events could not be read, and there is no \
                                earlier answer to fall back on"
                        .to_string())
                }
                _ => {}
            }
        }
        let fetched = fetch_assignments(account, spec);
        let mut state = self.assignments.lock().unwrap_or_else(|e| e.into_inner());
        if state.account.as_deref() != Some(key.as_str()) {
            *state = AssignmentsState { account: Some(key), ..AssignmentsState::default() };
        }
        match fetched {
            Ok(held) => {
                state.fetched = Some((Instant::now(), held.clone()));
                state.retry_at = None;
                Ok(held)
            }
            Err(e) => {
                state.retry_at = Some(Instant::now() + FETCH_RETRY_AFTER);
                match state.fetched.as_ref() {
                    Some((at, held)) => {
                        log::warn!(
                            "app logic: could not refresh which app takes which events ({e}); going by what was \
                             read {}s ago",
                            at.elapsed().as_secs()
                        );
                        Ok(held.clone())
                    }
                    None => Err(e),
                }
            }
        }
    }

    /// The account's apps, with their scripts in `languages` ([`languages_asked`]).
    pub fn load(
        &self,
        account: &LinkedAccount,
        spec: &AppLogicSpec,
        languages: &[String],
    ) -> Result<Vec<AppEntry>, String> {
        let key = (account_key(account), languages.to_vec());
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            // (linked to another account since, or asked in other languages: what was kept was another list)
            if state.account.as_ref() != Some(&key) {
                *state = CatalogState { account: Some(key.clone()), ..CatalogState::default() };
            }
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
        match fetch(account, spec, languages) {
            Ok(apps) => {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                // (the account changed while this was fetched: kept for the account it was fetched for)
                *state = CatalogState { account: Some(key), fetched: Some((Instant::now(), apps.clone())), retry_at: None };
                Ok(apps)
            }
            Err(e) => {
                let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if state.account.as_ref() != Some(&key) {
                    *state = CatalogState { account: Some(key), ..CatalogState::default() };
                }
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
        *self.assignments.lock().unwrap_or_else(|e| e.into_inner()) = AssignmentsState::default();
    }

    /// Let the next load ask the provider again at once, keeping what is
    /// held: the outbox, which spaces its own tries, is trying again.
    pub fn retry_now(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).retry_at = None;
        self.assignments.lock().unwrap_or_else(|e| e.into_inner()).retry_at = None;
    }
}

fn client() -> Result<reqwest::blocking::Client, String> {
    super::creds::client_builder(super::net::Keep::Never)
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("could not build the app-logic client: {e}"))
}

/// The account's apps, with their scripts in `languages`.
///
/// The languages are said where the provider reads them
/// ([`AppLogicSpec::languages`]): it lists the scripts in the languages its
/// caller names, and only those, and a caller that names none is handed its
/// JavaScript ones. `languages` is what the heartbeat declares
/// ([`languages_asked`]), so that every script this desktop is handed is one the
/// provider's browser leaves to it, and every one it leaves to it is handed over.
fn fetch(account: &LinkedAccount, spec: &AppLogicSpec, languages: &[String]) -> Result<Vec<AppEntry>, String> {
    let said = spec.languages.as_ref().map(|l| (l.param.as_str(), languages.join(",")));
    Ok(fetch_list(account, &spec.path, said, &spec.catalogue.apps, "app list")?
        .into_iter()
        .map(AppEntry::listed)
        .collect())
}

/// What the provider keeps of who takes which source's events: the account's
/// choices, and the apps it counts as able to take each source, where its reply
/// says ([`super::descriptor::AppLogicRouting::candidates`]; a reply without
/// them is one that counts none, as a map the provider leaves empty is).
fn fetch_assignments(account: &LinkedAccount, spec: &AppLogicSpec) -> Result<Assignments, String> {
    let Some(routing) = spec.routing.as_ref() else {
        return Ok(Assignments::default());
    };
    let what = "list of which app takes which events";
    let mut reply = fetch_reply(account, &routing.path, None, what)?;
    let mut out = Assignments::default();
    for choice in &take_list(&mut reply, &routing.list, what)? {
        let (Some(source), Some(app)) = (
            choice.get(&routing.source).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()),
            choice.get(&routing.app).and_then(Value::as_str).map(str::trim).filter(|a| !a.is_empty()),
        ) else {
            continue;
        };
        out.chosen.insert(source.to_string(), app.to_string());
    }
    if let (Some(key), Some(field)) = (routing.candidates.as_deref(), routing.candidate_app.as_deref()) {
        // (an empty map can come as an empty list, from a provider whose maps and lists are one thing)
        if let Some(map) = reply.get(key).and_then(Value::as_object) {
            for (source, apps) in map {
                let apps: Vec<String> = apps
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|entry| entry.get(field).and_then(Value::as_str))
                    .map(str::trim)
                    .filter(|app| !app.is_empty())
                    .map(str::to_string)
                    .collect();
                out.candidates.insert(source.clone(), apps);
            }
        }
    }
    Ok(out)
}

/// GET one of the lane's lists: the array the reply at `path` holds under
/// `key`, with `query` (a parameter and its value) said when there is one.
/// `what` is the list in words, for the reasons it could not be read.
fn fetch_list(account: &LinkedAccount, path: &str, query: Option<(&str, String)>, key: &str, what: &str) -> Result<Vec<Value>, String> {
    let mut reply = fetch_reply(account, path, query, what)?;
    take_list(&mut reply, key, what)
}

/// The array `reply` holds under `key`, or why it holds none.
fn take_list(reply: &mut Value, key: &str, what: &str) -> Result<Vec<Value>, String> {
    // A missing array is NOT an account with no apps: it is a reply this
    // descriptor does not describe, and treating the two the same would leave
    // an install recording nothing with nothing to show for it.
    let list = reply.get_mut(key).ok_or_else(|| {
        format!(
            "the app-logic lane's reply carries no {key:?}, so this connector's descriptor does not \
             describe it"
        )
    })?;
    match list.take() {
        Value::Array(list) => Ok(list),
        other => Err(format!(
            "the app-logic lane returned {key:?} as a {} rather than the {what}",
            type_name(&other)
        )),
    }
}

/// GET one of the lane's replies, whole.
fn fetch_reply(account: &LinkedAccount, path: &str, query: Option<(&str, String)>, what: &str) -> Result<Value, String> {
    let request = client()?.get(super::oauth::join(&account.base_url, path));
    let request = match query {
        Some((name, value)) => request.query(&[(name, value)]),
        None => request,
    };
    let resp = request
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| format!("could not reach the app-logic lane: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body: Value = super::net::read_json(resp).unwrap_or(Value::Null);
        let refused = format!("the provider refused the {what}");
        let message = body
            .get("message")
            .or_else(|| body.get("error"))
            .and_then(Value::as_str)
            .unwrap_or(&refused);
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "this desktop's key may not read the account's app logic ({message}) — link again"
            ));
        }
        return Err(format!("HTTP {}: {message}", status.as_u16()));
    }
    super::net::read_json_up_to(resp, super::net::MAX_LIST_BYTES)
        .map_err(|e| format!("the app-logic lane returned an unreadable {what}: {e}"))
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
/// "Every app" is every app the event is for (`route`): where the provider says
/// which of the account's apps take its source, that is one of them ([`Route`],
/// which the caller decides once for the event, for this lane and the flows').
///
/// Returns what happened, most useful first for a log: one [`Outcome`] per
/// effect attempted plus one for anything that went wrong before the effects
/// were reached. Never an error — this lane is best effort by construction, and
/// the caller has already dispatched the event locally.
///
/// Three passes, in this order: collect every app's scripts and markers; run
/// them all on the script host; then perform each app's effects. The middle
/// pass is the only one that touches the engine, and an event whose apps carry
/// no event scripts never reaches it — which is what keeps a quiet account from
/// spawning a script host it has no use for.
///
/// A script is run in the language it says it is in, and only where that is
/// one of `languages` (what this desktop declares, [`languages_asked`]): one in
/// another is said and left alone, never run as JavaScript.
///
/// `linked` says whether `account` is still the link this desktop holds, and is
/// asked before each effect: once it is not (the link was forgotten or replaced
/// while the event was handled), nothing more is written for it, here or at the
/// provider, and that is said once as a write that can be tried again, so the
/// event is kept for the next link to the same provider (whose markers and
/// record keys make the second time do only what the first did not).
#[allow(clippy::too_many_arguments)]
pub fn handle_event(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    apps: &[AppEntry],
    route: &Route,
    storage: &StorageStore,
    envelope: &Value,
    host: &dyn ScriptBatch,
    prelude: Prelude<'_>,
    languages: &[String],
    connector: &ConnectorFn,
    linked: &dyn Fn() -> bool,
) -> Vec<Outcome> {
    let mut out = Vec::new();
    let event_key = envelope
        .get("idempotencyKey")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let runs = collect(spec, apps, route, storage, &mut out, envelope);
    if runs.is_empty() {
        return out;
    }

    // The provider names the function its scripts declare. There is no default
    // here on purpose: guessing one would mean a provider whose scripts declare
    // something else runs nothing at all while every event reports success.
    let Some(entry_function) = spec.entry_function.as_deref().filter(|f| !f.is_empty()) else {
        for run in &runs {
            out.push(Outcome::err(
                run.app,
                "-",
                "lane",
                "this provider's descriptor names no entryFunction, so there is no function to \
                 call in its scripts and none of them were run",
            ));
        }
        return out;
    };

    // The provider's scripts are written against the provider's standard
    // library. Without it the same source is a different program — every helper
    // call throws — so nothing is sent and every app is told why. Refusing is
    // not a loss here: the beat that carries this desktop's engine token is
    // dropped in the same state, so the provider's own runtime runs these
    // scripts correctly instead of handing them to a desktop missing half the
    // language.
    if let Some(why) = prelude.missing() {
        for run in &runs {
            out.push(Outcome::err(run.app, "-", "lane", why.to_string()));
        }
        return out;
    }

    let replies = run_on_host(host, spec, entry_function, &runs, prelude, languages);

    for (i, run) in runs.iter().enumerate() {
        if let Some(why) = replies.lane.get(&i) {
            // ONE line for the app, not one per script: the scripts did nothing
            // wrong, the engine was not there.
            out.push(Outcome::err(run.app, "-", "lane", why.clone()));
            continue;
        }
        for (j, script) in run.scripts.iter().enumerate() {
            let name = if script.id.is_empty() {
                "(unnamed script)"
            } else {
                script.id
            };
            let value = match replies.by_job.get(&job_id(i, j)) {
                Some(Ok(v)) => v,
                Some(Err(e)) => {
                    out.push(Outcome::err(run.app, name, "script", e.clone()));
                    continue;
                }
                None => {
                    out.push(Outcome::err(
                        run.app,
                        name,
                        "script",
                        "the script host answered the batch without this script, so nothing it \
                         asked for was performed",
                    ));
                    continue;
                }
            };
            let effects = match effects_of(spec, value) {
                Ok(list) => list,
                Err(e) => {
                    out.push(Outcome::err(run.app, name, "script", e));
                    continue;
                }
            };
            if effects.is_empty() {
                continue;
            }
            let applied = apply_effects(
                account, spec, run.entry, run.app, name, effects, storage, &event_key, connector, linked,
            );
            let went = applied.last().is_some_and(|o| o.effect == LINK_WENT_EFFECT);
            out.extend(applied);
            if went {
                return out;
            }
        }
    }
    out
}

/// What an [`Outcome`] names as its effect when the link went while the event was handled.
const LINK_WENT_EFFECT: &str = "link";

/// The outcome that says the link went while `app`'s effects were being performed: nothing more is written for
/// it, and the event is tried again (a write that can be).
fn link_went(app: &str, script: &str) -> Outcome {
    Outcome {
        app: app.to_string(),
        script: script.to_string(),
        effect: LINK_WENT_EFFECT.to_string(),
        detail: Err(
            "the link this event was being handled for was forgotten or replaced, so nothing more was written for it; \
             the event is kept for the next link to this provider"
                .to_string(),
        ),
        transient: true,
    }
}

/// One app's scripts and the context they run against.
struct AppRun<'a> {
    app: &'a str,
    entry: &'a AppEntry,
    ctx: Value,
    scripts: Vec<ScriptRef<'a>>,
}

/// Resolve every app that has work on this event, with its markers read.
///
/// Anything that stops an app before its scripts — no id, listed twice, markers
/// that would not read — is pushed onto `out` here and the app is left out. An
/// app the event is not for ([`Route`]) is left out without a word: nothing
/// went wrong, the event is another app's.
fn collect<'a>(
    spec: &AppLogicSpec,
    apps: &'a [AppEntry],
    routed: &Route,
    storage: &StorageStore,
    out: &mut Vec<Outcome>,
    envelope: &Value,
) -> Vec<AppRun<'a>> {
    let mut runs = Vec::new();
    // Several apps could take it and the account has chosen none: run for all
    // of them it would be written into each, so it is run for none, and that
    // is said ONCE, for the event, not once an app.
    if let Some(why) = routed.unchosen() {
        out.push(Outcome::err("-", "-", "lane", why));
        return runs;
    }
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
        if !routed.runs(app) {
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
        runs.push(AppRun {
            app,
            entry,
            ctx: json!({ "event": envelope, "storage": seen }),
            scripts,
        });
    }
    runs
}

// --- running the scripts on the warm script host ---------------------------

/// The largest source one script may carry.
///
/// EVERY JavaScript mode of the leaf-script envelope compiles the author's text
/// inside the guest — `entry` wraps it in a `Function` — so the engine's
/// dynamic-code ceiling (`limits.dynamicCodeSourceBytes`) bounds an app-logic
/// script whichever mode it runs in. Over it, the job comes back `resource`
/// with a message about dynamic code that names neither the script nor what to
/// do about it.
///
/// So an oversized script is refused HERE, by name and by size, and its
/// neighbours in the same event still run. There is nowhere earlier to refuse
/// it: this desktop neither authors nor saves app-logic scripts — it runs
/// whatever the provider published, which can grow past the ceiling between two
/// fetches — so "refuse at save" is not a move this side can make.
///
/// Compared in UTF-8 bytes, which is never fewer than the UTF-16 units the
/// engine counts, so this refuses nothing the engine would have run.
/// `the_source_ceiling_is_the_engines_own` pins the figure against the STAGED
/// `zipp/PROFILE.json`, so a release that moves it fails a test rather than
/// drifting into a lane that quietly stops running long scripts.
pub const MAX_SCRIPT_SOURCE_BYTES: usize = 65_536;

/// The job id for the `script`th script of the `app`th run of one event.
///
/// Synthetic, not `<app>/<script>`: both of those are the provider's strings,
/// the request schema bounds a job id at 1..128 characters, and an id that
/// varies with the provider's naming is an id that can collide.
fn job_id(app: usize, script: usize) -> String {
    format!("a{app}s{script}")
}

/// One script as `protocol/v1/script-request.schema.json` describes it.
///
/// `entry` mode: the source is compiled as written, the function the provider
/// names is looked up in it, and it is called with the context as its one
/// argument. Nothing is rewritten on the way — no unboxing, no `return`
/// rewriting, no encoding — because nothing between here and the engine reads
/// the source as anything but source.
///
/// `budgetMs` and `instructionSteps` are the provider's two budgets and are
/// OMITTED when it names neither: the wall-clock default belongs to the CLI and
/// the step default to the engine, and inventing either here would put a figure
/// in this file that nothing else agrees with.
fn script_job(id: &str, spec: &AppLogicSpec, entry_function: &str, script: &ScriptRef, ctx: &Value) -> Value {
    let mut job = json!({
        "id": id,
        "mode": "entry",
        "entry": entry_function,
        "source": script.source,
        "args": [ctx],
    });
    if let Some(ms) = spec.budget_ms {
        job["budgetMs"] = json!(ms);
    }
    if let Some(steps) = spec.instruction_budget {
        job["instructionSteps"] = json!(steps);
    }
    job
}

/// One Python script as a `python-project` job that names the provider's own
/// modes (`modes`, [`python_modes`]): the runner unfolds the author's source
/// through the request's profile into the project the provider's runtime builds,
/// and calls what the mode calls with the context as its one argument, so a
/// script's `run(ctx)` returns the same effects it returns there. The two budgets
/// are a JavaScript job's.
fn python_job(id: &str, spec: &AppLogicSpec, modes: &[String], script: &ScriptRef, ctx: &Value) -> Value {
    let mut job = json!({
        "id": id,
        "language": PYTHON,
        "mode": "python-project",
        "modes": modes,
        "source": script.source,
        "args": [ctx],
    });
    if let Some(ms) = spec.budget_ms {
        job["budgetMs"] = json!(ms);
    }
    if let Some(steps) = spec.instruction_budget {
        job["instructionSteps"] = json!(steps);
    }
    job
}

/// Why a Python script was not run by a desktop that declares Python, when the
/// profile this event was given does not define the modes it is unfolded through.
fn not_unfoldable(spec: &AppLogicSpec) -> String {
    let modes = spec.languages.as_ref().map(|l| l.python_modes.join(", ")).unwrap_or_default();
    format!(
        "this script is in {PYTHON}, and the provider's script profile this event was given does not define the \
         modes it is unfolded through ({modes}), so it was not run: a job naming one would have had every script of \
         the event refused with it"
    )
}

/// Why a script in `language` was not run, when it is not one of `languages`.
fn not_run_in(language: &str, languages: &[String]) -> String {
    format!(
        "this script is in {language}, and this desktop runs this provider's app scripts in {} alone, so it was \
         not run here: it is the provider's own runtime's to run",
        languages.join(" and ")
    )
}

/// What the host said about one event's scripts.
#[derive(Debug, Default)]
struct Replies {
    /// Per job id: the value its entry function returned, or why it did not.
    by_job: BTreeMap<String, Result<Value, String>>,
    /// Per app index: the batch carrying its scripts was never served.
    lane: BTreeMap<usize, String>,
}

/// Run every script of this event on the host, in as few batches as fit, each
/// in its own language where that is one of `languages`.
fn run_on_host(
    host: &dyn ScriptBatch,
    spec: &AppLogicSpec,
    entry_function: &str,
    runs: &[AppRun],
    prelude: Prelude<'_>,
    languages: &[String],
) -> Replies {
    let mut replies = Replies::default();
    let mut jobs: Vec<Value> = Vec::new();
    // Which app each job belongs to, so a batch that is never served can be
    // reported once per app rather than once per script.
    let mut owner: Vec<usize> = Vec::new();
    // Asked of THIS batch's profile, not only of the one the languages were worked out from: a job naming a mode
    // the request's profile does not define refuses the whole request, every JavaScript job in it as well.
    let declares_python = languages.iter().any(|l| l == PYTHON);
    let python = declares_python.then(|| python_modes(spec, prelude.profile())).flatten();
    for (i, run) in runs.iter().enumerate() {
        for (j, script) in run.scripts.iter().enumerate() {
            let id = job_id(i, j);
            let job = match (script.language, python) {
                (JAVASCRIPT, _) => script_job(&id, spec, entry_function, script, &run.ctx),
                (PYTHON, Some(modes)) => python_job(&id, spec, modes, script, &run.ctx),
                (PYTHON, None) if declares_python => {
                    replies.by_job.insert(id, Err(not_unfoldable(spec)));
                    continue;
                }
                (other, _) => {
                    replies.by_job.insert(id, Err(not_run_in(other, languages)));
                    continue;
                }
            };
            if script.source.len() > MAX_SCRIPT_SOURCE_BYTES {
                replies.by_job.insert(
                    id,
                    Err(format!(
                        "this script is {} bytes of source and the engine compiles at most {} \
                         ({} too many), so it was not run — split it or shorten it",
                        script.source.len(),
                        MAX_SCRIPT_SOURCE_BYTES,
                        script.source.len() - MAX_SCRIPT_SOURCE_BYTES
                    )),
                );
                continue;
            }
            jobs.push(job);
            owner.push(i);
        }
    }
    for range in batch_ranges(&jobs, prelude.request_bytes()) {
        let batch = batch_request_with(jobs[range.clone()].to_vec(), prelude.profile());
        match host.run(&batch) {
            Ok(response) => {
                for job in &jobs[range] {
                    let Some(id) = job.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    if let Some(result) = response.result(id) {
                        replies
                            .by_job
                            .insert(id.to_string(), result.outcome.clone().map_err(job_failure));
                    }
                }
            }
            Err(e) => {
                let why = host_failure(&e);
                for app in &owner[range] {
                    replies.lane.entry(*app).or_insert_with(|| why.clone());
                }
            }
        }
    }
    replies
}

/// Why one script did not produce a value.
///
/// `guest` is the script's own doing — it threw, or it does not compile — and
/// keeps the word the lane has always used for it. Every other kind is the
/// engine or the wiring rather than the script, and says so, because the fix is
/// somewhere else entirely.
fn job_failure(e: JobError) -> String {
    match e.kind.as_str() {
        "guest" => format!("the script threw: {}", e.message),
        "timeout" => format!("the script ran past its time budget and was stopped: {}", e.message),
        "resource" => format!("the script exceeded an engine ceiling: {}", e.message),
        other => format!("the script could not be run ({other}): {}", e.message),
    }
}

/// Why a whole batch was never served.
fn host_failure(e: &HostError) -> String {
    format!("{e}; this app's scripts were not run for this event")
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
    linked: &dyn Fn() -> bool,
) -> Vec<Outcome> {
    let f = &spec.fields;
    let mut out = Vec::new();
    let mut record_write_failed = false;

    for (index, effect) in effects.iter().enumerate() {
        // Asked before each, as each is a write (a record, a marker, a command a plugin carries out): one performed
        // after the link went was written with a key the person had asked this desktop to forget.
        if !linked() {
            out.push(link_went(app, script));
            return out;
        }
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

        let key = record_key(event_key, script, index);
        let result: Result<String, Failed> = match operation {
            AppLogicOperation::SubmitRecord => {
                submit(account, spec, entry, effect, key.as_deref()).map(|id| match id {
                    Some(id) => format!("created record {id}"),
                    None => "created a record".to_string(),
                })
            }
            AppLogicOperation::UpdateRecord => update(account, spec, entry, effect, key.as_deref()),
            AppLogicOperation::SetStorage => {
                let key = effect
                    .get(&f.storage_key)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if key.is_empty() {
                    Err("the effect names no key to store under".to_string().into())
                } else if record_write_failed {
                    // Deliberately not stored. The script's own order says a
                    // failed write must not be marked handled, and storing it
                    // anyway would lose the record permanently.
                    Err(format!(
                        "held back {key:?} because a record write earlier in this script \
                         failed, so the event is not marked handled"
                    )
                    .into())
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
                            Failed::from(format!(
                                "{e} — {key:?} was not saved, so this event will be handled again"
                            ))
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
                    _ => Err("the effect names no message to show".to_string().into()),
                }
            }
            AppLogicOperation::ConnectorRequest => {
                connector_request(spec, effect, event_key, script, index, connector).map_err(Failed::from)
            }
        };

        if matches!(
            operation,
            AppLogicOperation::SubmitRecord | AppLogicOperation::UpdateRecord
        ) && result.is_err()
        {
            record_write_failed = true;
        }
        let transient = result.as_ref().err().is_some_and(|f| f.transient);
        out.push(Outcome {
            app: app.to_string(),
            script: script.to_string(),
            effect: kind,
            detail: result.map_err(|f| f.message),
            transient,
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
        // As the provider's own page resolves a key: the portable key first,
        // then the form's id, then the name the app shows it under. An app
        // that was not installed from a pack has no portable keys, and the
        // browser wrote its scripts' records where this lane refused them.
        .or_else(|| entry.form_named(&spec.catalogue, key))
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
    key: Option<&str>,
) -> Result<Option<String>, Failed> {
    let form = form_id(spec, entry, effect)?;
    let form_seg = super::oauth::path_segment(form, "form id")?;
    let values = record(spec, effect)?;
    let url = super::oauth::join(
        &account.base_url,
        &spec.submit_path.replace("{formId}", &form_seg),
    );
    let mut body = json!({ &spec.fields.record: values });
    if let (Some(k), Some(field)) = (key, spec.fields.write_key.as_deref()) {
        // The same event sent again gets the record it made, not a second one.
        body[field] = json!(k);
    }
    let resp = client()?
        .post(&url)
        .bearer_auth(&account.credential)
        .json(&body)
        .send()
        .map_err(|e| Failed::unreachable("create the record", &e))?;
    let status = resp.status();
    let payload: Value = super::net::read_json(resp).unwrap_or(Value::Null);
    if status.as_u16() == 409 && payload.get("conflict").and_then(Value::as_bool) == Some(true) {
        // Made already, under this key, from values the script has since
        // built differently: the record is there.
        return Ok(None);
    }
    if !status.is_success() {
        return Err(Failed::refused(
            status,
            format!("the provider refused the record: HTTP {}: {}", status.as_u16(), refusal(&payload)),
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
    key: Option<&str>,
) -> Result<String, Failed> {
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
                )
                .into());
            }
            find_record(account, spec, form, field, wanted)?
        }
    };

    let Some(id) = target else {
        // Not found. Upsert makes it, otherwise this is a correction for a
        // record that was never written — a clean no-op, said out loud rather
        // than swallowed.
        if effect.get(&f.upsert).and_then(Value::as_bool) == Some(true) {
            let created = submit(account, spec, entry, effect, key)?;
            return Ok(match created {
                Some(id) => format!("no record matched, so one was created ({id})"),
                None => "no record matched, so one was created".to_string(),
            });
        }
        return Ok("no record matched, and the effect does not create one — nothing changed"
            .to_string());
    };

    let form_seg = super::oauth::path_segment(form, "form id")?;
    let id_seg = super::oauth::path_segment(&id, "record id")?;
    let url = super::oauth::join(
        &account.base_url,
        &spec
            .update_path
            .replace("{formId}", &form_seg)
            .replace("{id}", &id_seg),
    );
    let resp = client()?
        .put(&url)
        .bearer_auth(&account.credential)
        .json(&json!({ &f.record: values }))
        .send()
        .map_err(|e| Failed::unreachable("update the record", &e))?;
    let status = resp.status();
    if !status.is_success() {
        let payload: Value = super::net::read_json(resp).unwrap_or(Value::Null);
        return Err(Failed::refused(
            status,
            format!("the provider refused the update to {id}: HTTP {}: {}", status.as_u16(), refusal(&payload)),
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
) -> Result<Option<String>, Failed> {
    // Asked of the provider first, where it filters its listing (`matchFilterParam`): a record
    // the newest page no longer holds is found then. The scan alone missed it, and an effect
    // that creates on a miss made a second one (an SMS thread by its number, once 200 newer
    // threads existed). Only for words in a plainly named field, which is what such a filter
    // takes. A miss, or a listing the provider will not filter (a form whose records it cannot
    // read), falls back to the scan, so nothing the scan found before is lost.
    if let (Some(param), Value::String(text)) = (&spec.match_filter_param, wanted) {
        let plain = !field.is_empty() && field.len() <= 64 && field.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
        if plain {
            let filter = format!(
                "{}={}",
                param.replace("{field}", field),
                url::form_urlencoded::byte_serialize(text.as_bytes()).collect::<String>()
            );
            match scan_records(account, spec, form, field, wanted, Some(&filter)) {
                Ok((Some(id), _)) => return Ok(Some(id)),
                Ok((None, _)) => {}
                // (the provider is away or busy: the event is sent again, as from the scan)
                Err(e) if e.transient => return Err(e),
                Err(_) => {}
            }
        }
    }
    let (found, seen) = scan_records(account, spec, form, field, wanted, None)?;
    if found.is_none() && seen as u32 >= spec.match_scan_limit {
        log::info!(
            "app-logic: no record has {field}={} in the newest {} of this form, which is the              whole page — an older one would not have been seen",
            scalar_text(wanted),
            spec.match_scan_limit
        );
    }
    Ok(found)
}

/// One page of the form's records (`filter`: a query parameter the provider narrows it by),
/// and the id of the first whose `field` holds `wanted`, with how many records the page held.
fn scan_records(
    account: &LinkedAccount,
    spec: &AppLogicSpec,
    form: &str,
    field: &str,
    wanted: &Value,
    filter: Option<&str>,
) -> Result<(Option<String>, usize), Failed> {
    let form_seg = super::oauth::path_segment(form, "form id")?;
    let path = spec.list_path.replace("{formId}", &form_seg);
    let sep = if path.contains('?') { '&' } else { '?' };
    let mut url = format!(
        "{}{sep}limit={}",
        super::oauth::join(&account.base_url, &path),
        spec.match_scan_limit
    );
    if let Some(filter) = filter {
        url.push('&');
        url.push_str(filter);
    }
    let resp = client()?
        .get(&url)
        .bearer_auth(&account.credential)
        .send()
        .map_err(|e| Failed::unreachable("find the record", &e))?;
    let status = resp.status();
    let payload: Value = super::net::read_json_up_to(resp, super::net::MAX_LIST_BYTES).unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(Failed::refused(
            status,
            format!("the provider refused the record listing: HTTP {}: {}", status.as_u16(), refusal(&payload)),
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
    Ok((found.and_then(Value::as_str).map(str::to_string), items.len()))
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
    use crate::bridge::conditions::testing::FakeHost;
    // The two ceilings the planner moved to `script_host` plans against. Only
    // the tests here still name them; the lane itself asks `batch_ranges`.
    use crate::bridge::script_host::{MAX_BATCH_DEADLINE, MAX_REQUEST_BYTES};

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
        AppEntry::listed(json!({
            "app": { "id": "app-1", "slug": "receptionist" },
            "customLogic": { "version": 1, "runtime": "quickjs", "scripts": scripts },
            "forms": forms,
        }))
    }

    fn store(tag: &str) -> (StorageStore, PathBuf) {
        let dir = crate::test_dirs::temp_path(&format!("oaiy-app-logic-{tag}"));
        std::fs::create_dir_all(dir.join("link")).unwrap();
        (StorageStore::open(&dir), dir)
    }

    fn no_connector() -> impl Fn(&str, &str, Option<Value>, &str) -> Result<Value, String> {
        |_, _, _, _| Err("no plugin host in this test".to_string())
    }

    /// What a desktop that declares JavaScript alone runs its app scripts in.
    fn js() -> Vec<String> {
        vec![JAVASCRIPT.to_string()]
    }

    // --- the descriptor is the whole vocabulary ----------------------------


    // --- the provider's prelude --------------------------------------------

    fn profile_doc() -> Value {
        let preamble = "var validators = { email: function (v) { return /@/.test(v); } };";
        json!({
            "v": 1,
            "preamble": preamble,
            "preambleSha256": crate::link::script_profile::sha256_hex(preamble),
            "instructionSteps": 200_000_000u64,
        })
    }

    #[test]
    fn the_providers_prelude_travels_with_the_app_logic_batch() {
        let s = spec();
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1", "slug": "a" },
            "customLogic": { "scripts": [
                { "id": "s1", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("prelude");
        let host = FakeHost::always(json!({}));
        let doc = profile_doc();
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::Profile(&doc),
            &js(),
            &no_connector(),
            &|| true,
        );
        assert!(out.is_empty(), "{out:?}");
        let request = host.only_request();
        assert_eq!(request["profile"], doc, "verbatim, beside the jobs");
        assert_eq!(request["jobs"].as_array().unwrap().len(), 1);
        drop(dir);
    }

    #[test]
    fn app_logic_without_the_providers_prelude_runs_nothing_and_says_so_per_app() {
        // Every script of this provider is written against a library it
        // publishes. Running them without it fails each one separately with a
        // ReferenceError the author cannot act on; refusing the lane says the
        // one true thing once per app.
        let s = spec();
        let apps = vec![
            AppEntry::listed(json!({
                "app": { "id": "app-1", "slug": "a" },
                "customLogic": { "scripts": [
                    { "id": "s1", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
                ] },
                "forms": []
            })),
            AppEntry::listed(json!({
                "app": { "id": "app-2", "slug": "b" },
                "customLogic": { "scripts": [
                    { "id": "s2", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
                ] },
                "forms": []
            })),
        ];
        let (storage, dir) = store("noprelude");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::Missing("the prelude could not be read".into()),
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(host.calls(), 0, "no batch is sent at all");
        assert_eq!(out.len(), 2, "one line per app, not one per script: {out:?}");
        for o in &out {
            assert_eq!(o.effect, "lane");
            assert!(o.detail.clone().unwrap_err().contains("could not be read"));
        }
        drop(dir);
    }

    #[test]
    fn a_batch_leaves_room_for_the_prelude_it_will_carry() {
        // The prelude rides on EVERY batch and is several times the headroom
        // the planner used to keep. Packed to the old budget, a batch would
        // arrive over the host's request cap and be refused whole — every
        // script in it lost, with nothing saying why.
        let fat = "v".repeat(400_000);
        let jobs: Vec<Value> =
            (0..12).map(|_| json!({ "id": "x", "budgetMs": 1000, "args": [fat] })).collect();
        let prelude = "p".repeat(300_000);
        let doc = json!({
            "v": 1,
            "preamble": prelude,
            "preambleSha256": crate::link::script_profile::sha256_hex(&prelude),
        });
        let bytes = serde_json::to_vec(&doc).unwrap().len();

        let with = batch_ranges(&jobs, bytes);
        let without = batch_ranges(&jobs, 0);
        assert_ne!(with, without, "the prelude has to come off the budget, not be hoped for");

        // The old plan, sent carrying the prelude it now has to carry, is over
        // the cap — which is the failure this charge exists to prevent.
        let overflowed =
            serde_json::to_vec(&batch_request_with(jobs[without[0].clone()].to_vec(), Some(&doc)))
                .unwrap()
                .len();
        assert!(
            overflowed > MAX_REQUEST_BYTES,
            "{overflowed} bytes: this test no longer demonstrates anything"
        );
        // The new one fits, batch for batch, and still carries every job.
        for range in &with {
            let request = batch_request_with(jobs[range.clone()].to_vec(), Some(&doc));
            let sent = serde_json::to_vec(&request).unwrap().len();
            assert!(sent <= MAX_REQUEST_BYTES, "{sent} bytes in one batch");
        }
        assert_eq!(with.first().unwrap().start, 0);
        assert_eq!(with.last().unwrap().end, jobs.len(), "nothing is dropped");
    }

    #[test]
    fn no_provider_name_or_effect_type_is_written_into_this_module() {
        // The single most important constraint, asserted against the source
        // rather than trusted. Test fixtures may name a provider; the executing
        // code may not — an effect type or a field name spelled here is a
        // second provider nobody can add without a release.
        let code_only = crate::source_scan::production_code(include_str!("app_logic.rs"));
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
    fn a_scripts_source_reaches_its_job_exactly_as_the_provider_published_it() {
        // The one property the old lane could not have: the source is not
        // wrapped, encoded, or scanned for words the compiler would rewrite. It
        // is the job's `source`, byte for byte, and the function the descriptor
        // names is called with the context as its one argument.
        //
        // Anything less was a real failure once: the graph compiler rewrote
        // `return` inside a function body into an illegal break and killed
        // EVERY script of EVERY app.
        let s = spec();
        let source = "function run(ctx) {\n  return { effects: [] }; // ; and a \"quote\"\n}";
        let script = ScriptRef { id: "one", source, language: JAVASCRIPT };
        let ctx = json!({ "event": { "name": "x" }, "storage": {} });
        let job = script_job("a0s0", &s, "run", &script, &ctx);
        assert_eq!(job["source"], json!(source), "verbatim, not re-encoded");
        assert_eq!(job["mode"], "entry");
        assert_eq!(job["entry"], "run");
        assert_eq!(job["args"], json!([ctx]), "the context is the one argument");
        assert_eq!(job["budgetMs"], json!(1000), "the provider's clock");
        assert_eq!(job["instructionSteps"], json!(200_000_000), "the provider's step budget");
    }

    #[test]
    fn a_provider_that_names_neither_budget_leaves_both_defaults_where_they_live() {
        // The wall clock belongs to the CLI and the step ceiling to the engine.
        // Writing either figure here would put a number in this file that
        // nothing else agrees with.
        let mut s = spec();
        s.budget_ms = None;
        s.instruction_budget = None;
        let job = script_job("a0s0", &s, "run", &ScriptRef { id: "one", source: "x", language: JAVASCRIPT }, &json!({}));
        assert!(job.get("budgetMs").is_none(), "{job}");
        assert!(job.get("instructionSteps").is_none(), "{job}");
    }

    #[test]
    fn one_event_is_one_batch_and_every_script_of_every_app_is_in_it() {
        // The third batch of an event, after the trigger conditions and the
        // flow bindings — not one batch per app, and certainly not one PROCESS
        // per app, which is what every event used to cost.
        let s = spec();
        let app = |id: &str| {
            AppEntry::listed(json!({
                "app": { "id": id },
                "customLogic": { "scripts": [
                    { "id": "a", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
                    { "id": "b", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
                ] },
                "forms": []
            }))
        };
        let apps = vec![app("app-1"), app("app-2")];
        let (storage, dir) = store("onebatch");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert!(out.is_empty(), "{out:?}");
        let request = host.only_request();
        let jobs = request["jobs"].as_array().unwrap();
        assert_eq!(jobs.len(), 4, "two apps, two scripts each, one batch: {request}");
        assert_eq!(
            jobs.iter().map(|j| j["id"].as_str().unwrap()).collect::<Vec<_>>(),
            vec!["a0s0", "a0s1", "a1s0", "a1s1"],
            "ids are synthetic and in publication order"
        );
        // Each app's scripts see THEIR app's markers, so the context cannot be
        // shared between jobs — every job carries its own.
        assert_eq!(jobs[0]["args"], jobs[1]["args"]);
        assert_eq!(request["v"], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_event_with_no_scripts_never_reaches_the_host() {
        // Load-bearing: this runs on every plugin event, and the host is
        // spawned by the first job it is asked to serve. An account whose apps
        // carry no event logic must not start one.
        let s = spec();
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [{ "id": "s", "hook": "onInstall", "source": "x" }] },
            "forms": []
        }))];
        let (storage, dir) = store("nobatch");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert!(out.is_empty());
        assert_eq!(host.calls(), 0, "no scripts, no batch, no host");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_script_the_author_switched_off_is_not_run_headlessly() {
        // The browser stops running a disabled script; without this the desktop
        // would go on running it on every event and the author would have no
        // way to tell. Only an explicit `false` disables — the key is usually
        // absent, and absent is a script that runs.
        let s = spec();
        let app = AppEntry::listed(json!({ "app": { "id": "app-1" }, "customLogic": { "scripts": [
            { "id": "on",      "hook": "onConnectorEvent", "source": "function run(){}", "enabled": true },
            { "id": "default", "hook": "onConnectorEvent", "source": "function run(){}" },
            { "id": "off",     "hook": "onConnectorEvent", "source": "function run(){}", "enabled": false }
        ] } }));
        let ids: Vec<&str> = app
            .event_scripts(&s.catalogue, &s.event_hook)
            .iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec!["on", "default"]);
    }

    #[test]
    fn a_script_over_the_engines_dynamic_code_ceiling_is_refused_by_name_and_its_neighbours_run() {
        // Every JS mode compiles the source inside the guest, so the engine's
        // dynamic-code ceiling bounds an app-logic script whatever the mode. The
        // engine's own answer is `resource` with a message about dynamic code,
        // which names neither the script nor what to do — and this desktop does
        // not save these scripts, so there is no earlier moment to refuse one.
        let s = spec();
        let big = format!("function run(ctx) {{ /* {} */ return {{}}; }}", "x".repeat(MAX_SCRIPT_SOURCE_BYTES));
        assert!(big.len() > MAX_SCRIPT_SOURCE_BYTES);
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "huge",  "hook": "onConnectorEvent", "source": big },
                { "id": "small", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("toobig");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].script, "huge");
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("65536"), "{why}");
        assert!(why.contains(&big.len().to_string()), "{why}");
        // And it never reached the engine: the neighbour is the only job sent.
        let jobs = host.only_request()["jobs"].as_array().unwrap().len();
        assert_eq!(jobs, 1, "the oversized script is refused before the batch is built");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_source_ceiling_is_the_engines_own() {
        // Read from the STAGED engine's profile, never written down twice: a
        // release that moves `dynamicCodeSourceBytes` fails here instead of
        // leaving a lane that quietly refuses scripts the engine would run (or,
        // worse, sends ones it will not).
        // Through `staged_runner`, so "no profile" is a skip on a dev box and a
        // FAILURE under CI — the same policy as every other staged-CLI test
        // here. A silent no-op is how a pinned figure stops being pinned.
        if staged_runner().is_none() {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        }
        let profile = staged_cli_dir().join("zipp").join("PROFILE.json");
        let profile: Value = serde_json::from_str(&std::fs::read_to_string(&profile).unwrap()).unwrap();
        assert_eq!(
            profile["limits"]["dynamicCodeSourceBytes"].as_u64(),
            Some(MAX_SCRIPT_SOURCE_BYTES as u64),
            "the staged engine's dynamic-code ceiling moved"
        );
    }

    #[test]
    fn a_graph_with_more_scripts_than_one_batch_can_carry_is_split_not_truncated() {
        // The host caps ONE batch at 60 s and kills the child when the cap
        // passes — losing every answer in the batch, the scripts that had
        // already finished included. So a long graph becomes consecutive
        // batches: every script still runs, in order, and no batch is bigger
        // than what the host can promise.
        let job = |ms: u64| json!({ "id": "x", "budgetMs": ms, "source": "s" });
        let jobs: Vec<Value> = (0..50).map(|_| job(1000)).collect();
        let ranges = batch_ranges(&jobs, 0);
        assert!(ranges.len() > 1, "50 scripts do not fit one batch");
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, jobs.len(), "nothing is dropped");
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "the batches are contiguous");
        }
        for range in &ranges {
            let batch = batch_request_with(jobs[range.clone()].to_vec(), None);
            assert!(
                crate::bridge::script_host::batch_deadline(&batch) < MAX_BATCH_DEADLINE,
                "a batch must finish inside the host's ceiling, not at it: {range:?}"
            );
        }
        // Everything that fits stays in ONE batch — splitting a short graph
        // would pay a round trip per chunk for nothing.
        assert_eq!(batch_ranges(&(0..5).map(|_| job(1000)).collect::<Vec<_>>(), 0).len(), 1);
        // And one job that is too big all by itself still goes, alone, rather
        // than taking the event's other scripts down with it.
        assert_eq!(batch_ranges(&[job(60_000), job(1000)], 0), vec![0..1, 1..2]);
    }

    #[test]
    fn a_batch_is_split_on_bytes_as_well_as_on_time() {
        // Every job carries its own copy of the context — each one is a fresh
        // engine, so there is nothing to share it through — and one app's
        // markers can be large. Four scripts of one app are then enough to pass
        // the host's request cap, which it refuses whole, unsent.
        let fat = "v".repeat(1_500_000);
        let jobs: Vec<Value> = (0..4).map(|_| json!({ "id": "x", "budgetMs": 1000, "args": [fat] })).collect();
        let ranges = batch_ranges(&jobs, 0);
        assert!(ranges.len() > 1, "six megabytes of context cannot be one request");
        for range in &ranges {
            let bytes = serde_json::to_vec(&batch_request_with(jobs[range.clone()].to_vec(), None)).unwrap().len();
            assert!(bytes <= MAX_REQUEST_BYTES, "{bytes} bytes in one batch");
        }
    }

    #[test]
    fn a_host_that_cannot_serve_is_one_line_for_the_app_not_one_per_script() {
        // The scripts did not disagree about anything; the engine was not
        // there. One line names the app and says its scripts did not run.
        let s = spec();
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "a", "hook": "onConnectorEvent", "source": "function run(){}" },
                { "id": "b", "hook": "onConnectorEvent", "source": "function run(){}" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("hostdown");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &FakeHost::down("no engine on this machine"),
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].effect, "lane");
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("no engine on this machine"), "{why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_script_that_threw_is_its_own_failure_and_the_next_one_still_runs() {
        // The isolation the encoded try/catch used to provide, now the
        // envelope's: one fresh engine per job, so a throw cannot reach a
        // neighbour and cannot fail the batch.
        let s = spec();
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "throws", "hook": "onConnectorEvent", "source": "function run(){ boom(); }" },
                { "id": "after",  "hook": "onConnectorEvent", "source": "function run(){ return {}; }" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("throws");
        let host = FakeHost::by_source(|source| {
            if source.contains("boom") {
                Err(JobError { kind: "guest".into(), message: "boom is not defined".into() })
            } else {
                Ok(json!({}))
            }
        });
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].script, "throws");
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("threw") && why.contains("boom is not defined"), "{why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_kind_that_is_not_the_scripts_fault_says_so_rather_than_blaming_the_script() {
        // "The script threw" sends the author to read a script that is fine.
        assert!(job_failure(JobError { kind: "guest".into(), message: "x".into() }).contains("threw"));
        for (kind, word) in [("timeout", "time budget"), ("resource", "engine ceiling"), ("host", "could not be run")] {
            let why = job_failure(JobError { kind: kind.into(), message: "x".into() });
            assert!(why.contains(word), "{kind}: {why}");
            assert!(!why.contains("threw"), "{kind}: {why}");
        }
    }

    #[test]
    fn a_provider_that_names_no_entry_function_disables_the_lane_and_says_which_app_stopped() {
        // No "run" default in this crate: a provider whose scripts declare
        // something else would otherwise run nothing at all while every event
        // reported success.
        let mut s = spec();
        s.entry_function = None;
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [
                { "id": "a", "hook": "onConnectorEvent", "source": "function run(){}" }
            ] },
            "forms": []
        }))];
        let (storage, dir) = store("noentry");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].app, "app-1");
        assert!(out[0].detail.clone().unwrap_err().contains("entryFunction"));
        assert_eq!(host.calls(), 0, "nothing is sent without a function to call");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
            &|| true,
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
            &|| true,
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
            &|| true,
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
            &|| true,
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
    fn a_record_write_keys_on_the_event_so_sending_it_again_makes_no_second_record() {
        // The outbox sends an event again when the provider was away; the
        // records its scripts wrote the first time must be found, not made twice.
        let a = record_key("evt-7", "calls", 0).unwrap();
        assert_eq!(Some(a.clone()), record_key("evt-7", "calls", 0));
        assert_ne!(Some(a.clone()), record_key("evt-7", "calls", 1));
        assert_ne!(Some(a), record_key("evt-8", "calls", 0));
        let long = record_key(&"e".repeat(200), "calls", 0).unwrap();
        assert!(long.len() <= 128, "{long}");
        assert_eq!(record_key("", "calls", 0), None, "no event key, no record key");
        let d = crate::link::descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        assert!(d.app_logic.unwrap().fields.write_key.is_some(), "the built-in connector says where the key goes");
    }

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
            &|| true,
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
            &|| true,
        );
        assert_eq!(out[0].detail.clone().unwrap(), "updated record resp-1");

        // The provider is asked for the record by its field first (its listing filters), so one
        // older than the newest page is found: the scan alone missed it, and an upsert then made
        // a second record. The value is written as a query takes it.
        let (list, _) = rx.recv().unwrap();
        assert!(list.starts_with("GET /api/v1/forms/form-2/responses?limit="), "{list}");
        assert!(list.contains("&answers.turn_key=c1%3A3 "), "{list}");
        let (put, body) = rx.recv().unwrap();
        assert!(put.starts_with("PUT /api/v1/forms/form-2/responses/resp-1 "), "{put}");
        assert!(body.contains("corrected"), "{body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_match_the_filtered_listing_misses_is_still_looked_for_in_the_newest_page() {
        // A number held where the script wrote its text, a field a filter does not take, a value
        // that is no text: each is found by the scan, as before there was a filter to ask.
        let s = spec();
        let e = entry(json!([{ "id": "form-2", "packFormId": "turns" }]), json!([]));
        let (base, rx) = stub(NO_ROWS, "200 OK");
        let (storage, dir) = store("match-fallback");
        let out = apply_effects(
            &account(base),
            &s,
            &e,
            "app-1",
            "script-1",
            &[
                json!({ "type": "formlogic.updateResponse", "formKey": "turns", "match": { "field": "turn_key", "value": "c9" }, "answers": { "text": "x" } }),
                json!({ "type": "formlogic.updateResponse", "formKey": "turns", "match": { "field": "turn key", "value": "c9" }, "answers": { "text": "x" } }),
                json!({ "type": "formlogic.updateResponse", "formKey": "turns", "match": { "field": "turn_key", "value": 9 }, "answers": { "text": "x" } }),
            ],
            &storage,
            "evt-1",
            &no_connector(),
            &|| true,
        );
        assert!(out.iter().all(|o| o.detail.clone().unwrap().contains("nothing changed")), "{out:?}");
        let mut lists = Vec::new();
        while let Ok((line, _)) = rx.recv_timeout(Duration::from_millis(400)) {
            lists.push(line);
        }
        let filtered: Vec<bool> = lists.iter().map(|l| l.contains("answers.")).collect();
        assert_eq!(filtered, [true, false, false, false], "asked by the field once, then scanned: {lists:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_form_is_found_by_its_key_then_its_id_then_the_name_the_app_shows() {
        // As the provider's own page resolves a key. An app that was not installed from a pack
        // has no portable keys, and its scripts' records were refused here ("no form keyed").
        let s = spec();
        let e = entry(
            json!([
                { "id": "form-1", "packFormId": "calls", "displayName": "form-2" },
                { "id": "form-2", "packFormId": null, "displayName": "Visits" },
            ]),
            json!([]),
        );
        let effect = |key: &str| json!({ "type": "formlogic.submitResponse", "formKey": key, "answers": {} });
        assert_eq!(form_id(&s, &e, &effect("calls")).unwrap(), "form-1");
        assert_eq!(form_id(&s, &e, &effect("form-2")).unwrap(), "form-2", "an id before a name");
        assert_eq!(form_id(&s, &e, &effect("Visits")).unwrap(), "form-2");
        assert!(form_id(&s, &e, &effect("visits")).unwrap_err().contains("no form keyed"));
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
            &|| true,
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
            &|| true,
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
            &|| true,
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
        let e = AppEntry::listed(apps[0].clone());
        assert_eq!(e.key(c), Some("e2e19da6"));
        let scripts = e.event_scripts(c, "onConnectorEvent");
        assert_eq!(scripts.len(), 1, "only the event hook, and not the empty one");
        assert_eq!(scripts[0].id, "on-call");
        // The blank key is not a form a script can name — offering it would let
        // an effect that names no form at all resolve to a real one.
        assert_eq!(e.forms(c), vec![("calls", "fde1f7f6")]);

        // An app with no logic at all is ordinary, not a failure.
        let bare = AppEntry::listed(json!({ "app": { "id": "x" } }));
        assert!(bare.event_scripts(c, "onConnectorEvent").is_empty());
        assert_eq!(bare.key(c), Some("x"));
        // …and the slug stands in when there is no id.
        assert_eq!(
            AppEntry::listed(json!({ "app": { "id": "  ", "slug": "s" } })).key(c),
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
        let why = fetch(&account(base), &s, &js()).unwrap_err();
        assert!(why.contains("does not describe it"), "{why}");

        let (base, _rx) = stub(r#"{"apps":[]}"#, "200 OK");
        assert!(fetch(&account(base), &s, &js()).unwrap().is_empty(), "an empty account is fine");
    }

    #[test]
    fn an_app_with_no_id_is_refused_rather_than_run_without_a_memory() {
        // Its markers would have nowhere to live, so every event it handled
        // would be handled again on the next delivery — forever.
        let s = spec();
        let apps = vec![AppEntry::listed(json!({
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
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &FakeHost::always(json!({})),
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert_eq!(out.len(), 1);
        assert!(out[0].detail.clone().unwrap_err().contains("nowhere to remember"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- against the real runner -------------------------------------------

    /// Where `desktop/scripts/sync-cli.mjs` stages the CLI and its engine.
    fn staged_cli_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("cli")
    }

    /// The bundled CLI, if this checkout has one staged, and a Node to run it.
    ///
    /// Skipped rather than failed when either is missing ON A DEV BOX: the
    /// tests below are about whether the generated graph SURVIVES the real
    /// compiler, and a box without Node cannot answer that either way. Under
    /// CI the skip is a failure — see `require_staged`.
    fn staged_runner() -> Option<(PathBuf, PathBuf)> {
        let cli = staged_cli_dir().join("oaiy.mjs");
        let found = (|| {
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
        })();
        require_staged(found, std::env::var_os("CI").is_some())
    }

    /// Under CI, "nothing staged" is a broken pipeline, not a box without Node:
    /// ci.yml stages the CLI (sync-cli) and puts node on PATH before cargo
    /// test, so a `None` there means the steps were reordered or the staging
    /// silently stopped — and the one test that proves the scripts run on the
    /// real runner would have quietly become a no-op.
    fn require_staged<T>(found: Option<T>, ci: bool) -> Option<T> {
        if found.is_none() && ci {
            panic!(
                "CI must stage the CLI (node desktop/scripts/sync-cli.mjs) and have node on PATH before cargo test"
            );
        }
        found
    }

    #[test]
    #[should_panic(expected = "CI must stage the CLI")]
    fn under_ci_an_unstaged_cli_fails_the_suite_instead_of_skipping() {
        require_staged::<()>(None, true);
    }

    #[test]
    fn on_a_dev_box_an_unstaged_cli_is_a_skip() {
        assert!(require_staged::<()>(None, false).is_none());
        assert_eq!(require_staged(Some(1), true), Some(1));
    }

    #[test]
    fn the_staged_cli_ships_its_engine_and_the_engine_is_the_one_the_bundle_was_built_for() {
        // What `sync-cli.mjs` stages, checked from the Rust side: the bundle,
        // the two worker shells it spawns (workflow, and the leaf-script shell
        // behind `oaiy script`), and the five engine files — with the wasm
        // matching the staged SOURCE.json's digest, the notices matching theirs,
        // and the bundle carrying that digest (the define esbuild baked in).
        // Every digest here is READ from the staged record; none is written
        // down. A desktop that fails this ships a CLI that resolves and then
        // fails every run `engine_unavailable`.
        use sha2::{Digest, Sha256};
        if staged_runner().is_none() {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        }
        let dir = staged_cli_dir();
        for rel in [
            "oaiy.mjs",
            "oaiy-zipp-worker.mjs",
            "oaiy-script-worker.mjs",
            "zipp/zipp_wasm_bg.wasm",
            "zipp/PROFILE.json",
            "zipp/SOURCE.json",
            "zipp/LICENSE-APACHE",
            "zipp/THIRD_PARTY_LICENSES.txt",
        ] {
            assert!(dir.join(rel).is_file(), "staged CLI is missing {rel}");
        }
        let source: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("zipp/SOURCE.json")).unwrap()).unwrap();
        let recorded = source["sha256"].as_str().expect("SOURCE.json records the wasm sha256");
        assert_eq!(recorded.len(), 64);
        let hex = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let wasm = std::fs::read(dir.join("zipp/zipp_wasm_bg.wasm")).unwrap();
        assert_eq!(hex(&wasm), recorded, "the staged wasm is not the one its SOURCE.json records");
        let notices_name = source["notices"]["file"].as_str().unwrap_or("THIRD_PARTY_LICENSES.txt");
        let notices = std::fs::read(dir.join("zipp").join(notices_name)).unwrap();
        assert_eq!(
            hex(&notices),
            source["notices"]["sha256"].as_str().expect("SOURCE.json records the notices sha256"),
            "the staged notices are not the ones SOURCE.json records"
        );
        let bundle = std::fs::read_to_string(dir.join("oaiy.mjs")).unwrap();
        assert!(
            bundle.contains(recorded),
            "the staged oaiy.mjs was not built for the staged engine (it does not carry its wasm digest)"
        );
        assert_eq!(source["release"].as_str().map(|r| r.starts_with('v')), Some(true), "{}", source["release"]);
    }

    #[test]
    fn the_real_scripts_run_on_the_real_engine_and_return_their_effects() {
        // The test that matters. Everything else here is about shapes; this is
        // the one that would have caught "Illegal break statement", which broke
        // EVERY script of EVERY app and produced an empty transcript with a
        // successful-looking desktop.
        //
        // The scripts below are the shipped ones, trimmed to the parts that
        // decide: a hook that must not fire on the wrong event, a dedup that
        // must read the storage it was handed, a throw that must not take its
        // neighbours down, and a timer — which the sandbox has to refuse out
        // loud, since its own `setTimeout` would silently never fire.
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
            { "id": "timer", "hook": "onConnectorEvent", "source":
              "function run(ctx) { setTimeout(function(){}, 10); return {}; }" },
            { "id": "after-the-throw", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { effects: [{ type: 'ui.toast', message: 'still here' }] }; }" },
            { "id": "unboxable", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { effects: [{ type: 'formlogic.submitResponse', formKey: 'f', answers: { text: 'body', url: 'https://example.test' } }] }; }" },
            { "id": "no-host", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { host: typeof process }; }" }
        ]);
        // Through the same reader the lane uses, so these really are the jobs
        // an event would build.
        let app = AppEntry::listed(json!({ "app": { "id": "app-1" }, "customLogic": { "scripts": raw } }));
        let s = spec();
        let refs = app.event_scripts(&s.catalogue, &s.event_hook);
        assert_eq!(refs.len(), 7, "every script here runs on the event hook");

        // ONE warm child for the whole test, handed the staged CLI directly:
        // `OAIY_CLI` is process-global and races the other tests, and under
        // cargo test the bundled lookup never reaches src-tauri/resources.
        let host = crate::bridge::script_host::ScriptHost::with_cli(
            crate::bridge::worker::CliInvocation::Node { script: cli },
            Some(node),
        );
        let run = |ctx: Value| -> Vec<Result<Value, String>> {
            let jobs: Vec<Value> = refs
                .iter()
                .enumerate()
                .map(|(i, script)| script_job(&job_id(0, i), &s, "run", script, &ctx))
                .collect();
            let response = host
                .evaluate(&batch_request_with(jobs, None))
                .unwrap_or_else(|e| panic!("the staged CLI must serve the batch: {e}"));
            // The engine that answered is the one this desktop ships — not the
            // host's own JavaScript.
            assert_eq!(response.engine["name"], "zipp", "{:?}", response.engine);
            (0..refs.len())
                .map(|i| {
                    response
                        .result(&job_id(0, i))
                        .unwrap_or_else(|| panic!("no answer for script {i}"))
                        .outcome
                        .clone()
                        .map_err(job_failure)
                })
                .collect()
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
        // The hazardous text survives intact — semicolons, quotes, a backslash
        // and a multi-byte dash. It crosses as a JSON literal now rather than a
        // percent-encoded blob, and it still has to arrive unchanged.
        assert_eq!(effects["effects"][0]["answers"]["text"], "hello; \"there\" — ok\\done");
        assert_eq!(effects["effects"][0]["answers"]["turn_key"], "c1:3");
        assert_eq!(effects["effects"][1]["key"], "seen-evt-1");
        // A script for a different event returns nothing rather than firing.
        assert_eq!(fresh[1].clone().unwrap(), json!({}));
        // A script that throws reports itself…
        assert!(fresh[2].clone().unwrap_err().contains("threw"));
        // …as does one that reaches for a timer: the guest shims replace
        // `setTimeout` with a stub that throws, at the GLOBAL scope the entry
        // source is compiled in. ZIPP's own `setTimeout` would have returned
        // quietly and never fired, and this script would have looked like a
        // success with its work dropped.
        let timer = fresh[3].clone().unwrap_err();
        assert!(timer.contains("threw"), "{timer}");
        assert!(timer.contains("setTimeout"), "{timer}");
        // …and neither takes the next one with it: one fresh engine per job.
        assert_eq!(fresh[4].clone().unwrap()["effects"][0]["message"], "still here");
        // The old unboxing hazard, still pinned. A record of two fields one of
        // which is called `text` used to come back as the bare text, because the
        // flow runner unboxed a node's result. Nothing unboxes an `entry` job's
        // return value, and this says so rather than assuming it.
        let boxed = fresh[5].clone().unwrap();
        assert_eq!(boxed["effects"][0]["answers"]["text"], "body");
        assert_eq!(boxed["effects"][0]["answers"]["url"], "https://example.test");
        // And the scripts ran inside the ZIPP VM: there is no Node `process` in
        // the scope a script sees. A host that ran them itself would answer
        // "object" here.
        assert_eq!(fresh[6].clone().unwrap()["host"], "undefined");

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

        // Both events were served by ONE child, kept warm between them — the
        // whole point of the lane moving off a process per event.
        let health = host.health();
        assert_eq!(health.children, 1, "one child served both events");
        host.shutdown();
    }


    #[test]
    fn the_real_app_scripts_can_call_the_providers_helpers_only_when_its_prelude_travels() {
        // The same blocker on the other lane, and the more expensive one: an
        // app script that throws writes no record at all, so the install looks
        // healthy and the transcript is simply empty.
        let Some((node, cli)) = staged_runner() else {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        };
        let preamble = "function __isArr(a) { return typeof a == \"object\" && a != null && typeof a.length == \"number\"; }\nvar validators = {\n  email: function(value) {\n    if (typeof value != \"string\") { return false; }\n    return /^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$/.test(value);\n  }\n};\nfunction sum(arr) {\n  if (!__isArr(arr)) { return 0; }\n  let total = 0;\n  let i = 0;\n  while (i < arr.length) {\n    if (typeof arr[i] == \"number\") { total = total + arr[i]; }\n    i = i + 1;\n  }\n  return total;\n}\n";
        let profile = json!({
            "v": 1,
            "preamble": preamble,
            "preambleSha256": crate::link::script_profile::sha256_hex(preamble),
            "instructionSteps": 200_000_000u64,
        });

        // An ordinary app script, written the way the provider's editor teaches
        // it: the standard library, then an effect.
        let raw = json!([{ "id": "s1", "hook": "onConnectorEvent", "source":
            "function run(ctx) {\n  var d = (ctx.event || {}).data || {};\n  if (!validators.email(d.to)) return {};\n  return { effects: [{ type: 'storage.set', key: 'seen', value: sum(d.amounts) }] };\n}" }]);
        let app = AppEntry::listed(json!({ "app": { "id": "app-1" }, "customLogic": { "scripts": raw } }));
        let s = spec();
        let refs = app.event_scripts(&s.catalogue, &s.event_hook);
        let ctx = json!({
            "event": { "name": "x", "idempotencyKey": "evt-1", "data": { "to": "a@b.co", "amounts": [2, 40] } },
            "storage": {},
        });
        let jobs: Vec<Value> =
            refs.iter().enumerate().map(|(i, r)| script_job(&job_id(0, i), &s, "run", r, &ctx)).collect();

        let host = crate::bridge::script_host::ScriptHost::with_cli(
            crate::bridge::worker::CliInvocation::Node { script: cli },
            Some(node),
        );

        // BEFORE.
        let started = Instant::now();
        let bare = host
            .evaluate(&batch_request_with(jobs.clone(), None))
            .expect("the batch is served");
        eprintln!("app logic without the prelude: {:?}", started.elapsed());
        let why = bare.result(&job_id(0, 0)).unwrap().outcome.clone().unwrap_err();
        assert_eq!(why.kind, "guest", "{why:?}");
        assert!(why.message.contains("not defined"), "{}", why.message);

        // AFTER.
        let started = Instant::now();
        let with = host
            .evaluate(&batch_request_with(jobs, Some(&profile)))
            .expect("the batch is served");
        eprintln!("app logic with the prelude: {:?}", started.elapsed());
        let value = with.result(&job_id(0, 0)).unwrap().outcome.clone().expect("the script ran");
        assert_eq!(value["effects"][0]["type"], "storage.set");
        assert_eq!(value["effects"][0]["value"], 42, "and the helper computed the real answer");

        host.shutdown();
    }

    #[test]
    fn the_real_engine_refuses_a_source_over_its_dynamic_code_ceiling() {
        // The backstop behind the local refusal: if the ceiling is ever read
        // wrong here, this is what the lane would actually get back — and it is
        // a `resource` kind, not a script that threw, so it is not reported as
        // the author's bug.
        let Some((node, cli)) = staged_runner() else {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        };
        let host = crate::bridge::script_host::ScriptHost::with_cli(
            crate::bridge::worker::CliInvocation::Node { script: cli },
            Some(node),
        );
        let big = format!(
            "function run(ctx) {{ /* {} */ return {{}}; }}",
            "x".repeat(MAX_SCRIPT_SOURCE_BYTES)
        );
        let script = ScriptRef { id: "huge", source: &big, language: JAVASCRIPT };
        let job = script_job("a0s0", &spec(), "run", &script, &json!({}));
        let response = host.evaluate(&batch_request_with(vec![job], None)).expect("the batch is served");
        let outcome = response.result("a0s0").expect("answered").outcome.clone();
        let e = outcome.expect_err("a source over the ceiling cannot compile");
        assert_eq!(e.kind, "resource", "{e:?}");
        host.shutdown();
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
        let apps = vec![AppEntry::listed(one.clone()), AppEntry::listed(one)];
        let (storage, dir) = store("twice");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &FakeHost::always(json!({})),
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
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
        let apps = vec![AppEntry::listed(json!({
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
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &FakeHost::always(json!({})),
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
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
        let apps = vec![AppEntry::listed(json!({
            "app": { "id": "app-1" },
            "customLogic": { "scripts": [{ "id": "s", "hook": "onInstall", "source": "x" }] },
            "forms": []
        }))];
        let (storage, dir) = store("noscripts");
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &FakeHost::always(json!({})),
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        assert!(out.is_empty());
        assert!(!storage.path().exists(), "nothing was written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- which app an event is for -------------------------------------------

    /// An app with one event script, whose source says which app's it is, that
    /// takes events from `sources` (none: its listing says nothing of them).
    fn app_taking(id: &str, sources: Option<Value>) -> AppEntry {
        let mut doc = json!({
            "app": { "id": id },
            "customLogic": { "scripts": [
                { "id": "s", "hook": "onConnectorEvent", "source": format!("function run(ctx) {{ return {{}}; }} // {id}") }
            ] },
            "forms": []
        });
        if let Some(sources) = sources {
            doc["connectors"] = sources;
        }
        AppEntry::listed(doc)
    }

    /// What the provider keeps of who takes `source`: the account's choice of
    /// `chosen` (none: it made none), and the apps it counts as candidates (none:
    /// its reply names none for the source).
    fn kept(source: &str, chosen: Option<&str>, candidates: &[&str]) -> Assignments {
        let mut out = Assignments::default();
        if let Some(app) = chosen {
            out.chosen.insert(source.to_string(), app.to_string());
        }
        if !candidates.is_empty() {
            out.candidates.insert(source.to_string(), candidates.iter().map(|a| a.to_string()).collect());
        }
        out
    }

    /// The apps whose scripts one event from the plugin `aokie` was run for (by
    /// what each job's source says; none when no batch was sent at all), and
    /// what was said of it: `assignments` is what the provider keeps of who
    /// takes the plugin's events (none: it was not read), and `bound` the apps
    /// with a flow bound to the event.
    fn run_for(
        s: &AppLogicSpec,
        apps: &[AppEntry],
        assignments: Option<&Assignments>,
        bound: &[&str],
        tag: &str,
    ) -> (Vec<String>, Vec<Outcome>) {
        let routed = route(s, apps, assignments, bound, "aokie");
        let (storage, dir) = store(tag);
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            s,
            apps,
            &routed,
            &storage,
            &json!({ "name": "aokie.call.ended", "source": "aokie", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &|| true,
        );
        let ran = match host.calls() {
            0 => Vec::new(),
            _ => host.only_request()["jobs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|job| job["source"].as_str().unwrap().rsplit("// ").next().unwrap().to_string())
                .collect(),
        };
        let _ = std::fs::remove_dir_all(&dir);
        (ran, out)
    }

    #[test]
    fn the_shipped_connector_reads_who_takes_a_sources_events_and_whom_the_account_chose() {
        // FormLogic's names, as data: an app lists the plugins it holds grants
        // for under `connectors` (FlowService::getOwnerAppLogic), the account's
        // choice of one app for a plugin is a row of `assignments`, and the apps
        // that hold a plugin's grants, with scripts or without, are its
        // `candidates` (FlowService::getConnectorAssignments).
        let s = spec();
        assert_eq!(s.catalogue.sources.as_deref(), Some("connectors"));
        let routing = s.routing.expect("the shipped connector says where the choice is read");
        assert_eq!(
            (routing.path.as_str(), routing.list.as_str(), routing.source.as_str(), routing.app.as_str()),
            ("/api/v1/connector-assignments", "assignments", "connectorId", "appId")
        );
        assert_eq!((routing.candidates.as_deref(), routing.candidate_app.as_deref()), (Some("candidates"), Some("appId")));
    }

    #[test]
    fn an_event_goes_to_the_one_app_that_takes_its_source() {
        // Two apps carry event scripts and one of them holds the plugin's
        // grants. Both were run, so the other wrote records of calls it has
        // nothing to do with.
        let apps = [
            app_taking("app-phone", Some(json!(["aokie"]))),
            app_taking("app-fleet", Some(json!(["vehicle"]))),
            app_taking("app-notes", Some(json!([]))),
        ];
        for (assignments, tag) in [(None, "route-one"), (Some(Assignments::default()), "route-one-read")] {
            let (ran, out) = run_for(&spec(), &apps, assignments.as_ref(), &[], tag);
            assert_eq!(ran, ["app-phone"], "{out:?}");
            assert!(out.is_empty(), "an app the event is not for is not a failure: {out:?}");
        }
    }

    #[test]
    fn two_apps_that_take_a_source_get_its_events_in_the_one_the_account_chose() {
        // The same pack installed twice: every call was written into both.
        let apps = [
            app_taking("app-old", Some(json!(["aokie"]))),
            app_taking("app-new", Some(json!(["aokie"]))),
            app_taking("app-notes", None),
        ];
        let (ran, out) = run_for(&spec(), &apps, Some(&kept("aokie", Some("app-new"), &[])), &[], "route-chosen");
        assert_eq!(ran, ["app-new"], "{out:?}");
        assert!(out.is_empty(), "{out:?}");

        // Chosen for another source's events is not chosen for this one's.
        let apps = [
            app_taking("app-old", Some(json!(["aokie", "vehicle"]))),
            app_taking("app-new", Some(json!(["aokie", "vehicle"]))),
        ];
        let (ran, out) = run_for(&spec(), &apps, Some(&kept("vehicle", Some("app-new"), &[])), &[], "route-chosen-elsewhere");
        assert!(ran.is_empty() && out.len() == 1, "{ran:?} {out:?}");
    }

    #[test]
    fn an_assignment_decides_even_where_one_app_alone_could_take_the_source() {
        // The account gave the plugin's events to another of its apps (one with
        // flows and no scripts, say). The one app with scripts for them ran all
        // the same, because it was the only one that could, and wrote records of
        // calls that were not its.
        let apps = [app_taking("app-phone", Some(json!(["aokie"]))), app_taking("app-notes", None)];
        let away = kept("aokie", Some("app-crm"), &[]);
        let (ran, out) = run_for(&spec(), &apps, Some(&away), &[], "route-assigned-away");
        assert!(ran.is_empty(), "no batch is sent at all: {ran:?}");
        assert!(out.is_empty(), "an event that is another app's is not a failure: {out:?}");
        let routed = route(&spec(), &apps, Some(&away), &[], "aokie");
        assert!(routed.holds_back("app-phone") && routed.holds_back("app-notes") && !routed.holds_back("app-crm"));
        assert_eq!(routed.assigned_elsewhere("app-phone"), Some("app-crm"));
        assert_eq!(routed.assigned_elsewhere("app-crm"), None);
        assert!(routed.unchosen().is_none());

        // The same app assigned runs, as it would have alone.
        let (ran, out) = run_for(&spec(), &apps, Some(&kept("aokie", Some("app-phone"), &[])), &[], "route-assigned-here");
        assert_eq!(ran, ["app-phone"], "{out:?}");
    }

    #[test]
    fn two_apps_that_take_a_source_and_no_choice_between_them_run_nothing_and_say_so_once() {
        let apps = [
            app_taking("app-old", Some(json!(["aokie"]))),
            app_taking("app-new", Some(json!(["aokie"]))),
            // (not one that could take it, and not run either: the event is not known to be its)
            app_taking("app-notes", None),
        ];
        let (ran, out) = run_for(&spec(), &apps, Some(&Assignments::default()), &[], "route-unchosen");
        assert!(ran.is_empty(), "no batch is sent at all: {ran:?}");
        assert_eq!(out.len(), 1, "once for the event, not once an app: {out:?}");
        assert_eq!((out[0].effect.as_str(), out[0].transient), ("lane", false));
        let why = out[0].detail.clone().unwrap_err();
        assert!(why.contains("\"aokie\"") && why.contains("app-old, app-new") && why.contains("assign"), "{why}");
    }

    #[test]
    fn two_apps_with_flows_alone_on_a_source_are_told_apart_as_apps_with_scripts_are() {
        // Two apps built on the plugin that carry flows and no scripts are not in
        // the app list, so nothing told them apart, and both started their flows
        // for every call. The provider counts both as able to take its events.
        let s = spec();
        let none: [AppEntry; 0] = [];
        let both = ["app-a", "app-b"];
        let routed = route(&s, &none, Some(&kept("aokie", None, &both)), &both, "aokie");
        assert!(routed.holds_back("app-a") && routed.holds_back("app-b"), "{routed:?}");
        let why = routed.unchosen().expect("said, once");
        assert!(why.contains("app-a, app-b"), "{why}");
        // Assigned: that one's flows start, and the other's do not.
        let routed = route(&s, &none, Some(&kept("aokie", Some("app-b"), &both)), &both, "aokie");
        assert!(routed.holds_back("app-a") && !routed.holds_back("app-b") && routed.unchosen().is_none());

        // One app with scripts and one with flows alone, both counted: as
        // ambiguous as two with scripts, so neither the scripts nor the flows run.
        let apps = [app_taking("app-a", Some(json!(["aokie"])))];
        let (ran, out) = run_for(&s, &apps, Some(&kept("aokie", None, &both)), &["app-b"], "route-flows-and-scripts");
        assert!(ran.is_empty() && out.len() == 1, "{ran:?} {out:?}");

        // A flow of an app the provider does not count as taking the plugin (it
        // holds none of its grants) makes the one that does no less the event's:
        // that one runs, and the other's flow waits, as its scripts would.
        let (ran, out) = run_for(&s, &apps, Some(&kept("aokie", None, &["app-a"])), &["app-crm"], "route-outsider");
        assert_eq!(ran, ["app-a"], "{out:?}");
        let routed = route(&s, &apps, Some(&kept("aokie", None, &["app-a"])), &["app-crm"], "aokie");
        assert!(routed.holds_back("app-crm") && !routed.holds_back("app-a"));
    }

    #[test]
    fn where_nothing_says_who_takes_a_source_every_apps_scripts_run_as_they_did() {
        // No app lists the plugin, and the provider counts none: nothing is
        // known, and nothing changes, for scripts or for flows.
        let unlisted = [app_taking("app-1", None), app_taking("app-2", Some(json!([]))), app_taking("app-3", Some(json!(["vehicle"])))];
        let (ran, out) = run_for(&spec(), &unlisted, Some(&Assignments::default()), &[], "route-unlisted");
        assert_eq!(ran, ["app-1", "app-2", "app-3"], "{out:?}");
        let routed = route(&spec(), &unlisted, Some(&Assignments::default()), &["app-9"], "aokie");
        assert!(!routed.holds_back("app-9") && !routed.holds_back("app-1"), "{routed:?}");

        // A connector that does not name the list: the same apps that would
        // have been told apart are all run, as before the rule was read.
        let mut legacy = spec();
        legacy.catalogue.sources = None;
        legacy.routing = None;
        let both = [app_taking("app-old", Some(json!(["aokie"]))), app_taking("app-new", Some(json!(["aokie"])))];
        let (ran, out) = run_for(&legacy, &both, None, &[], "route-unnamed");
        assert_eq!(ran, ["app-old", "app-new"], "{out:?}");

        // An event that names no source is no plugin's, so it is nobody's in particular.
        let (storage, dir) = store("route-no-source");
        let host = FakeHost::always(json!({}));
        let routed = route(&spec(), &both, Some(&kept("aokie", Some("app-old"), &[])), &[], "");
        handle_event(&account("http://provider.invalid".into()), &spec(), &both, &routed, &storage, &json!({ "name": "x", "idempotencyKey": "evt-1" }), &host, Prelude::None, &js(), &no_connector(), &|| true);
        assert_eq!(host.only_request()["jobs"].as_array().unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_an_app_that_would_act_on_the_event_counts_as_one_that_could_take_it() {
        // An app that holds the plugin's grants and runs nothing on its events
        // (its scripts are for other moments, or switched off, and no flow of it
        // is bound to the event) does not make the one that does ambiguous: that
        // one ran alone before, and still does, though the provider counts both.
        let mut idle = app_taking("app-idle", Some(json!(["aokie"])));
        idle.doc["customLogic"]["scripts"][0]["enabled"] = json!(false);
        let apps = [idle, app_taking("app-phone", Some(json!(["aokie"])))];
        let s = spec();
        let counted = kept("aokie", None, &["app-idle", "app-phone"]);
        let routed = route(&s, &apps, Some(&counted), &[], "aokie");
        assert_eq!((routed.candidates.clone(), routed.chosen()), (vec!["app-phone".to_string()], Some("app-phone")));
        let (ran, out) = run_for(&s, &apps, Some(&counted), &[], "route-idle");
        assert_eq!(ran, ["app-phone"], "{out:?}");

        // Nor does an app the provider listed twice, or bound twice, count as two.
        let twice = [app_taking("app-phone", Some(json!(["aokie"]))), app_taking("app-phone", Some(json!(["aokie"])))];
        assert_eq!(route(&s, &twice, None, &["app-phone", "app-phone"], "aokie").chosen(), Some("app-phone"));

        // Whether there is anything to ask the provider about at all.
        assert!(concerns_an_app(&s, &apps, &[]));
        let quiet = [apps[0].clone()];
        assert!(!concerns_an_app(&s, &quiet, &[]), "scripts that do not run on events concern no event");
        assert!(concerns_an_app(&s, &quiet, &["app-b"]), "a flow bound to the event does");
    }

    #[test]
    fn a_flow_of_an_app_the_event_is_not_for_is_held_back() {
        let s = spec();
        let apps = [
            app_taking("app-old", Some(json!(["aokie"]))),
            app_taking("app-new", Some(json!(["aokie"]))),
            app_taking("app-notes", None),
        ];
        let routed = route(&s, &apps, Some(&kept("aokie", Some("app-new"), &[])), &[], "aokie");
        assert!(routed.holds_back("app-old"));
        assert!(!routed.holds_back("app-new"), "the one the events go to");
        // The account gave the plugin's events to one app, so no other app's
        // flows start for them, those of apps nothing says take it among them.
        assert!(routed.holds_back("app-notes") && routed.holds_back("app-unheard-of"));
        assert!(routed.unchosen().is_none());

        // No choice: every app's are held back, and why is said.
        let apps = [app_taking("app-old", Some(json!(["aokie"]))), app_taking("app-new", Some(json!(["aokie"])))];
        let routed = route(&s, &apps, Some(&Assignments::default()), &[], "aokie");
        assert!(routed.holds_back("app-old") && routed.holds_back("app-new") && routed.holds_back("app-notes"));
        assert!(routed.unchosen().is_some());
        // Another plugin's events are not these apps' to take, so nothing of theirs is held back for them.
        let other = route(&s, &apps, Some(&Assignments::default()), &[], "vehicle");
        assert!(!other.holds_back("app-old") && other.unchosen().is_none());
    }

    #[test]
    fn the_accounts_choice_and_the_providers_candidates_are_read_from_one_reply_kept_and_fallen_back_on() {
        use crate::link::testkit::{Provider, Reply};
        let s = spec();
        let reply = r#"{"assignments":[{"connectorId":"aokie","appId":"app-new","appName":"New"},{"connectorId":"vehicle","appId":"app-gone"},{"appId":"app-old"},{"connectorId":" ","appId":"x"}],"candidates":{"aokie":[{"appId":"app-old","appName":"Old"},{"appId":"app-new"},{"appName":"no id"}],"vehicle":[]},"desktops":[]}"#;
        let server = Provider::start(move |_| Reply::ok(reply));
        let read = fetch_assignments(&account(server.base.clone()), &s).unwrap();
        assert_eq!(server.lines(), ["GET /api/v1/connector-assignments"]);
        assert_eq!(server.requests()[0].header("authorization"), Some("Bearer flk_secret"));
        assert_eq!(
            read.chosen,
            BTreeMap::from([("aokie".to_string(), "app-new".to_string()), ("vehicle".to_string(), "app-gone".to_string())])
        );
        assert_eq!(read.candidates.get("aokie"), Some(&vec!["app-old".to_string(), "app-new".to_string()]));
        assert_eq!(read.candidates.get("vehicle"), Some(&Vec::new()));

        // An account with none: its maps come as empty lists.
        let server = Provider::start(|_| Reply::ok(r#"{"assignments":[],"candidates":[],"desktops":[]}"#));
        assert_eq!(fetch_assignments(&account(server.base.clone()), &s).unwrap(), Assignments::default());

        // A reply the descriptor does not describe is not "nobody chose anything".
        for (refusal, says) in [
            (Reply::status(503, r#"{"message":"busy"}"#), "HTTP 503: busy"),
            (Reply::ok(r#"{"data":[]}"#), "carries no \"assignments\""),
            (Reply::ok(r#"{"assignments":{}}"#), "rather than the list of which app takes which events"),
        ] {
            let refusal = std::sync::Mutex::new(Some(refusal));
            let server = Provider::start(move |_| refusal.lock().unwrap().take().unwrap_or_else(|| Reply::status(500, "{}")));
            let why = fetch_assignments(&account(server.base.clone()), &s).unwrap_err();
            assert!(why.contains(says), "{why}");
        }

        // Kept as long as the app list is, and the last one read stands when a
        // refresh fails; another account's is never this one's.
        let answers = std::sync::Mutex::new(vec![Reply::status(503, r#"{"message":"busy"}"#), Reply::ok(reply)]);
        let server = Provider::start(move |_| answers.lock().unwrap().pop().unwrap_or_else(|| Reply::status(500, "{}")));
        let catalog = Catalog::new();
        let linked = account(server.base.clone());
        let first = catalog.assignments(&linked, &s).unwrap();
        assert_eq!(catalog.assignments(&linked, &s).unwrap(), first, "kept");
        assert_eq!(server.lines().len(), 1, "asked once: {:?}", server.lines());
        let aged = Instant::now().checked_sub(CATALOG_TTL + Duration::from_secs(1)).expect("a clock older than two minutes");
        catalog.assignments.lock().unwrap().fetched.as_mut().unwrap().0 = aged;
        assert_eq!(catalog.assignments(&linked, &s).unwrap(), first, "the refresh failed, and the last one read stands");
        assert_eq!(server.lines().len(), 2);
        let other = LinkedAccount { account_id: Some("another".into()), ..account(server.base.clone()) };
        assert!(catalog.assignments(&other, &s).is_err(), "nothing of another account's is fallen back on");

        // A connector that keeps no such rule asks nothing.
        let mut legacy = spec();
        legacy.catalogue.sources = None;
        legacy.routing = None;
        let server = Provider::start(|_| Reply::status(500, "{}"));
        assert_eq!(fetch_assignments(&account(server.base.clone()), &legacy).unwrap(), Assignments::default());
        assert!(server.lines().is_empty());
    }

    #[test]
    fn app_logic_rejects_traversal_ids_and_does_not_follow_redirects() {
        use crate::link::testkit::{Provider, Reply};
        let elsewhere = Provider::start(|_| Reply::ok(r#"{"apps":[]}"#));
        let moved_to = format!("{}/moved", elsewhere.base);
        let origin = Provider::start(move |_| Reply::redirect(307, &moved_to));
        let acc = account(origin.base.clone());
        let s = spec();

        // Redirects on catalogue fetch and record submit must NOT be followed.
        let err = fetch(&acc, &s, &js()).unwrap_err();
        assert!(err.contains("HTTP 307"), "{err}");
        let ok_app = entry(json!([{ "id": "form-1", "packFormId": "calls" }]), json!([]));
        let sub_err = submit(
            &acc,
            &s,
            &ok_app,
            &json!({ "formKey": "calls", "answers": { "a": 1 } }),
            None,
        )
        .unwrap_err();
        assert!(sub_err.message.contains("HTTP 307"), "{}", sub_err.message);
        assert!(elsewhere.requests().is_empty(), "redirect must not be followed");

        // Traversal formId or recordId is rejected before any HTTP request is made.
        let evil_app = entry(json!([{ "id": "../admin", "packFormId": "calls" }]), json!([]));
        let before = origin.requests().len();
        assert!(submit(
            &acc,
            &s,
            &evil_app,
            &json!({ "formKey": "calls", "answers": { "a": 1 } }),
            None,
        )
        .is_err());
        assert!(update(
            &acc,
            &s,
            &ok_app,
            &json!({ "formKey": "calls", "recordId": "../../evil?x=1", "answers": { "a": 1 } }),
            None,
        )
        .is_err());
        assert_eq!(origin.requests().len(), before, "traversal IDs must not reach the network");
    }

    // --- the language a script is in -------------------------------------------

    /// A profile whose Python contract has the modes `modes` (none: a contract
    /// without modes), each unfolding a script as the provider's app-logic mode
    /// does: the contract's helpers in scope, and `run(ctx)`'s value the answer.
    fn profile_with_modes(modes: &[&str]) -> Value {
        let preamble = "var helper = 1;";
        let mut python = json!({
            "contract": "test-python/1",
            "files": { "helpers.py": "def twice(n):\n    return n * 2\n" },
            "entry": "main",
            "call": "__run__",
        });
        if !modes.is_empty() {
            python["modes"] = modes
                .iter()
                .map(|name| {
                    json!({
                        "name": name,
                        "files": { "main.py": "def __run__(ctx):\n    import logic_block\n    run = getattr(logic_block, 'run', None)\n    return run(ctx) if callable(run) else None\n" },
                        "block": "logic_block.py",
                        "before": "from helpers import *\n",
                        "after": "",
                        "lineOffset": 1,
                    })
                })
                .collect();
        }
        json!({
            "v": 1,
            "preamble": preamble,
            "preambleSha256": crate::link::script_profile::sha256_hex(preamble),
            "python": python,
        })
    }

    /// An app with `scripts`, and nothing else of note.
    fn app_with(scripts: Value) -> AppEntry {
        AppEntry::listed(json!({ "app": { "id": "app-1" }, "customLogic": { "scripts": scripts }, "forms": [] }))
    }

    fn js_and_python() -> Vec<String> {
        vec![JAVASCRIPT.to_string(), PYTHON.to_string()]
    }

    const PY_SOURCE: &str = "def run(ctx):\n    return {}\n";

    #[test]
    fn the_shipped_connector_reads_a_scripts_language_and_unfolds_python_through_its_app_logic_mode() {
        // FormLogic's names, as data: a script says `language` (CustomLogicSanitizer, absent being JavaScript),
        // GET /api/v1/app-logic takes `?languages=` (FlowController::ownerAppLogic), and the profile it serves
        // unfolds an app script through the Python mode `applogic` (formlogic-script-profile.json).
        let s = spec();
        assert_eq!(s.catalogue.script_language.as_deref(), Some("language"));
        let languages = s.languages.as_ref().expect("the shipped connector says which languages it asks for");
        assert_eq!((languages.param.as_str(), languages.python_modes.as_slice()), ("languages", ["applogic".to_string()].as_slice()));
    }

    #[test]
    fn a_python_app_script_is_unfolded_through_the_providers_own_modes_beside_the_javascript_ones() {
        // FormLogic hands Python app scripts to a desktop whose beat names Python. They were run as JavaScript
        // `entry` jobs, which is to say not at all; they go to the runner as Python projects it builds from the
        // provider's own mode, the way the provider's browser builds them.
        let s = spec();
        let doc = profile_with_modes(&["applogic"]);
        let apps = vec![app_with(json!([
            { "id": "js", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
            { "id": "py", "hook": "onConnectorEvent", "language": "python", "source": PY_SOURCE },
            { "id": "named-js", "hook": "onConnectorEvent", "language": " javascript ", "source": "function run(ctx) { return {}; }" },
        ]))];
        let (storage, dir) = store("python-modes");
        let host = FakeHost::always(json!({}));
        let event = json!({ "name": "x", "idempotencyKey": "evt-1" });
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &event,
            &host,
            Prelude::Profile(&doc),
            &js_and_python(),
            &no_connector(),
            &|| true,
        );
        assert!(out.is_empty(), "{out:?}");
        let request = host.only_request();
        assert_eq!(request["profile"], doc, "the modes travel in the profile, beside the jobs");
        let jobs = request["jobs"].as_array().unwrap();
        assert_eq!((jobs[0]["mode"].as_str(), jobs[2]["mode"].as_str()), (Some("entry"), Some("entry")), "{jobs:?}");
        assert_eq!(
            jobs[1],
            json!({
                "id": "a0s1",
                "language": "python",
                "mode": "python-project",
                "modes": ["applogic"],
                "source": PY_SOURCE,
                "args": [{ "event": event, "storage": {} }],
                "budgetMs": s.budget_ms.unwrap(),
                "instructionSteps": s.instruction_budget.unwrap(),
            }),
            "the author's source as written, the same context and the same two budgets as a JavaScript script"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_script_in_a_language_this_desktop_does_not_declare_is_said_and_never_run_as_javascript() {
        // The beat declares JavaScript alone, so the provider's browser runs the Python scripts: one handed over
        // anyway (a provider that does not read the languages asked for) is not this desktop's to run, and is
        // never run as JavaScript, which is what every script used to be.
        let s = spec();
        let doc = profile_with_modes(&["applogic"]);
        let apps = vec![app_with(json!([
            { "id": "js", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
            { "id": "py", "hook": "onConnectorEvent", "language": "python", "source": PY_SOURCE },
            { "id": "rb", "hook": "onConnectorEvent", "language": "ruby", "source": "def run(ctx) end" },
            { "id": "odd", "hook": "onConnectorEvent", "language": 3, "source": "function run(ctx) { return {}; }" },
        ]))];
        let (storage, dir) = store("python-undeclared");
        let host = FakeHost::always(json!({}));
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::Profile(&doc),
            &js(),
            &no_connector(),
            &|| true,
        );
        let request = host.only_request();
        let jobs = request["jobs"].as_array().unwrap();
        assert_eq!(jobs.len(), 1, "only the JavaScript script is sent: {jobs:?}");
        assert_eq!(jobs[0]["mode"], "entry");
        let said: Vec<(&str, String)> = out.iter().map(|o| (o.script.as_str(), o.detail.clone().unwrap_err())).collect();
        assert_eq!(said.iter().map(|(script, _)| *script).collect::<Vec<_>>(), ["py", "rb", "odd"], "{out:?}");
        for ((_, why), language) in said.iter().zip(["python", "ruby", "something other than words"]) {
            assert!(why.contains(language) && why.contains("not run here") && why.contains("javascript alone"), "{why}");
        }
        assert!(out.iter().all(|o| o.effect == "script" && !o.transient), "{out:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn python_is_run_only_where_the_profile_defines_every_mode_the_provider_names() {
        let s = spec();
        assert_eq!(languages_run(&s, None), [JAVASCRIPT], "no profile, nothing to unfold a Python script through");
        assert_eq!(languages_run(&s, Some(&profile_with_modes(&[]))), [JAVASCRIPT], "a contract without modes");
        assert_eq!(languages_run(&s, Some(&profile_with_modes(&["flowModule"]))), [JAVASCRIPT], "without the mode named");
        assert_eq!(languages_run(&s, Some(&profile_with_modes(&["flowModule", "applogic"]))), [JAVASCRIPT, PYTHON]);
        let mut names_none = spec();
        names_none.languages.as_mut().unwrap().python_modes.clear();
        assert_eq!(languages_run(&names_none, Some(&profile_with_modes(&["applogic"]))), [JAVASCRIPT]);
        let mut legacy = spec();
        legacy.languages = None;
        assert_eq!(languages_run(&legacy, Some(&profile_with_modes(&["applogic"]))), [JAVASCRIPT]);

        // And asked again of the profile the batch carries: a job naming a mode the request's profile does not
        // define has the WHOLE request refused, the JavaScript scripts in it too, so the Python one stays behind.
        let apps = vec![app_with(json!([
            { "id": "js", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
            { "id": "py", "hook": "onConnectorEvent", "language": "python", "source": PY_SOURCE },
        ]))];
        let (storage, dir) = store("python-no-mode");
        let host = FakeHost::always(json!({}));
        let other = profile_with_modes(&["flowModule"]);
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::Profile(&other),
            &js_and_python(),
            &no_connector(),
            &|| true,
        );
        let request = host.only_request();
        assert_eq!(request["jobs"].as_array().unwrap().len(), 1, "{request}");
        assert_eq!(out.len(), 1, "{out:?}");
        let why = out[0].detail.clone().unwrap_err();
        assert!(out[0].script == "py" && why.contains("does not define the modes") && why.contains("applogic"), "{why}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_app_list_is_asked_for_in_the_languages_this_desktop_declares_and_kept_for_those_alone() {
        use crate::link::testkit::{Provider, Reply};
        let server = Provider::start(|_| Reply::ok(r#"{"apps":[]}"#));
        let s = spec();
        let catalog = Catalog::new();
        let linked = account(server.base.clone());
        catalog.load(&linked, &s, &js_and_python()).unwrap();
        catalog.load(&linked, &s, &js_and_python()).unwrap();
        // Python no longer declared: a list that holds Python scripts the browser is running again is not kept.
        catalog.load(&linked, &s, &js()).unwrap();
        assert_eq!(
            server.lines(),
            ["GET /api/v1/app-logic?languages=javascript%2Cpython", "GET /api/v1/app-logic?languages=javascript"]
        );

        // A connector that names no languages asks with none, as it did.
        let server = Provider::start(|_| Reply::ok(r#"{"apps":[]}"#));
        let mut legacy = spec();
        legacy.languages = None;
        fetch(&account(server.base.clone()), &legacy, &js_and_python()).unwrap();
        assert_eq!(server.lines(), ["GET /api/v1/app-logic"]);
    }

    #[test]
    fn what_this_lane_asks_for_is_what_the_beat_declares_and_never_more() {
        // FormLogic hands over the scripts in whatever languages the request names, without weighing them against
        // the beat, and its browser keeps every script in a language the beat leaves out. So the lane asks for what
        // the beat declares: with this machine's CLI, which runs JavaScript flows alone, that is JavaScript, though
        // the lane itself could unfold a Python script.
        let s = spec();
        let doc = profile_with_modes(&["applogic"]);
        assert_eq!(languages_run(&s, Some(&doc)), [JAVASCRIPT, PYTHON]);
        let probe = crate::bridge::script_host::ScriptHost::global().engine_probe();
        let flows = crate::link::heartbeat::declared_languages(probe.as_ref(), None);
        let asked = languages_asked(&s, Prelude::Profile(&doc));
        assert!(asked.iter().all(|l| flows.contains(l)), "asked {asked:?} beyond what the CLI runs {flows:?}");
        assert!(asked.iter().any(|l| l == JAVASCRIPT), "{asked:?}");
        assert_eq!(languages_asked(&s, Prelude::None), [JAVASCRIPT]);
    }

    #[test]
    fn a_python_app_script_runs_on_the_real_engine_and_its_effects_are_performed() {
        // The contract end to end, on the runner this desktop ships: a Python script unfolded through a profile's
        // mode is handed the same context a JavaScript one is, answers its effects the same way, and has them
        // performed; one that raises says where, in the author's own lines, and takes nothing down with it.
        let Some((node, cli)) = staged_runner() else {
            eprintln!("no staged CLI or no Node on this machine — skipping");
            return;
        };
        let host = crate::bridge::script_host::ScriptHost::with_cli(
            crate::bridge::worker::CliInvocation::Node { script: cli },
            Some(node),
        );
        let s = spec();
        let doc = profile_with_modes(&["applogic"]);
        let apps = vec![app_with(json!([
            { "id": "py", "hook": "onConnectorEvent", "language": "python", "source":
              "def run(ctx):\n    ev = ctx['event']\n    key = 'seen:' + ev['idempotencyKey']\n    if ctx['storage'].get(key):\n        return None\n    return {'effects': [{'type': 'storage.set', 'key': key, 'value': twice(ev['data']['n'])}]}\n" },
            { "id": "py-throws", "hook": "onConnectorEvent", "language": "python", "source":
              "def run(ctx):\n    return nothing_at_all\n" },
            { "id": "js", "hook": "onConnectorEvent", "source":
              "function run(ctx) { return { effects: [{ type: 'ui.toast', message: 'js ' + ctx.event.data.n }] }; }" },
        ]))];
        let (storage, dir) = store("python-real");
        let event = json!({ "name": "aokie.call.ended", "idempotencyKey": "evt-7", "data": { "n": 21 } });
        let deliver = || {
            handle_event(
                &account("http://provider.invalid".into()),
                &s,
                &apps,
                &Route::unknown(),
                &storage,
                &event,
                &host,
                Prelude::Profile(&doc),
                &js_and_python(),
                &no_connector(),
                &|| true,
            )
        };
        let out = deliver();
        assert_eq!(storage.values("app-1").unwrap().get("seen:evt-7"), Some(&json!(42)), "{out:?}");
        let threw = out.iter().find(|o| o.script == "py-throws").expect("the one that raises is said");
        let why = threw.detail.clone().unwrap_err();
        assert!(why.contains("threw") && why.contains("NameError") && why.contains("logic_block.py:2"), "{why}");
        assert!(out.iter().any(|o| o.script == "js" && o.detail == Ok("[info] js 21".to_string())), "{out:?}");
        // Delivered again, the Python script reads its own marker and asks for nothing.
        let again = deliver();
        assert!(!again.iter().any(|o| o.script == "py"), "{again:?}");
        host.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- a link that goes while an event is handled ------------------------------

    #[test]
    fn the_effects_of_an_event_whose_link_goes_while_they_are_performed_stop_and_the_event_is_kept() {
        // A person who disconnects while an event's effects were being performed: the rest were written all the same,
        // records with the key they had asked this desktop to forget, markers for a link that was no more. Now the
        // link is asked before each effect, and once it has gone nothing more is written and the event is kept.
        let s = spec();
        let apps = vec![app_with(json!([
            { "id": "first", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
            { "id": "second", "hook": "onConnectorEvent", "source": "function run(ctx) { return {}; }" },
        ]))];
        let host = FakeHost::always(json!({ "effects": [
            { "type": "storage.set", "key": "a" },
            { "type": "storage.set", "key": "b" },
        ] }));
        let (storage, dir) = store("link-went-effects");
        // Held for the first effect, and gone from the second on.
        let asked = std::sync::atomic::AtomicUsize::new(0);
        let linked = || asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        let out = handle_event(
            &account("http://provider.invalid".into()),
            &s,
            &apps,
            &Route::unknown(),
            &storage,
            &json!({ "name": "x", "idempotencyKey": "evt-1" }),
            &host,
            Prelude::None,
            &js(),
            &no_connector(),
            &linked,
        );
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].detail, Ok("remembered \"a\"".to_string()));
        assert_eq!((out[1].script.as_str(), out[1].effect.as_str(), out[1].transient), ("first", LINK_WENT_EFFECT, true));
        assert!(out[1].detail.clone().unwrap_err().contains("forgotten or replaced"), "{out:?}");
        let kept = storage.values("app-1").unwrap();
        assert!(kept.contains_key("a") && !kept.contains_key("b"), "{kept:?}");
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 2, "asked before each effect, and of nothing after");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
