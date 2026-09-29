//! Modules: the parts of OAIY that are there only while a plugin provides them.
//!
//! The phone (calls and texts, the Front desk's agents) and the calendar (the
//! phone receptionist's diary) come with a plugin: without one, OAIY shows no
//! Front desk, no phone and no calendar. Everything else is core.
//!
//! ```text
//!   plugin manifests ──► claims()  ─┐
//!   plugin registry  ──► resolve() ─┴─► Snapshot {revision, modules, contributions, warnings}
//!                                          ├── is_enabled("phone")   (routes, leases, the plugin host)
//!                                          ├── GET /api/modules       (ETag = revision)
//!                                          └── GET /api/modules/events (server-sent, on each change)
//! ```
//!
//! # What plugins add
//!
//! Each enabled plugin's contributions are merged into the snapshot: its
//! dashboard sections and pages (`ui.nav`, `ui.sections`), its Overview cards
//! (`ui.overview`, with the `ui.statusCards` polls their `$poll` bindings
//! read), its agent tools (`agentTools`, resolved against its service
//! definitions) and its setup wizard (`setup`). An entry for a module that is
//! off is left out; a bad `ui` entry is left out with a warning. Bindings
//! (`$health.status`) are passed on as declared, for the dashboard to look up,
//! so a health probe does not move the revision.
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
//! A plugin claims modules in its manifest's `modules` section (schemaVersion
//! 4: `{"provides": ["phone", "calendar"]}`, or just the list). Aokie predates the
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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{watch, Notify};

use crate::plugins::manifest::AgentTool;
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

/// What `manifest` claims to provide (its typed `modules` section, or the
/// legacy rule), each claim checked against what the module needs.
pub fn claims(manifest: &PluginManifest) -> Claims {
    let (declared, named, connector) = match &manifest.modules {
        Some(section) => (true, section.provides.clone(), section.connector.as_deref()),
        None if manifest.id == LEGACY_PROVIDER => (false, vec![PHONE.to_string(), CALENDAR.to_string()], None),
        None => (false, Vec::new(), None),
    };
    let mut out = Claims { declared, ..Claims::default() };
    for id in named {
        let id = id.trim().to_string();
        if id.is_empty() || out.claims.iter().any(|c| c.module == id) || out.unknown.contains(&id) {
            continue;
        }
        let Some(def) = def(&id) else {
            out.unknown.push(id);
            continue;
        };
        out.claims.push(check_claim(manifest, def, connector));
    }
    out
}

/// A claim is honoured when one of the plugin's connectors (the one the
/// section names, if it names one) declares every command the module uses.
fn check_claim(manifest: &PluginManifest, def: &'static ModuleDef, named: Option<&str>) -> Claim {
    if def.uses.is_empty() {
        return Claim { module: def.id, connector: None, refused: None };
    }
    let missing = |c: &crate::plugins::manifest::ConnectorDecl| -> Vec<&'static str> {
        def.uses.iter().copied().filter(|cmd| !c.commands.iter().any(|x| x == cmd)).collect()
    };
    let connectors: Vec<&crate::plugins::manifest::ConnectorDecl> =
        manifest.connectors.iter().filter(|c| named.map_or(true, |n| c.id == n)).collect();
    if let (Some(n), true) = (named, connectors.is_empty()) {
        let refused = format!("{} claims {} through the connector \"{n}\", which it does not declare.", manifest.name, def.noun);
        return Claim { module: def.id, connector: None, refused: Some(refused) };
    }
    if let Some(c) = connectors.iter().find(|c| missing(c).is_empty()) {
        return Claim { module: def.id, connector: Some(c.id.clone()), refused: None };
    }
    let refused = match connectors.iter().min_by_key(|c| missing(c).len()) {
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
    pub contributions: Contributions,
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
    let on = |id: &str| modules.iter().any(|m| m.id == id && m.enabled);
    let (contributions, more) = contribute(&records, &on);
    warnings.extend(more);
    Resolved { modules, contributions, warnings }
}

// ---- What plugins add to the dashboard and the app --------------------------------

/// The dashboard's own sections a plugin page may join, as a tab after the section's own.
pub const BUILTIN_SECTIONS: &[&str] = &["agent", "flows", "calendar", "engines", "services", "connections", "settings"];

/// The sidebar's groups a plugin's own section may be in (Work when it says none).
pub const SECTION_GROUPS: &[&str] = &["Home", "Work", "Setup"];

/// What an overview contribution may be.
pub const OVERVIEW_KINDS: &[&str] = &["hero", "status", "tile"];

/// The parts of a plugin's health report (`PluginRecord.last_health`) a `$health` binding may read.
pub const HEALTH_FIELDS: &[&str] = &["status", "detail", "components"];

/// The most often a `$poll` binding's command is sent.
pub const MIN_POLL_MS: u64 = 5000;

/// What the plugins providing (or not) the modules add to the dashboard and
/// the Agent app. Only plugins that load and are not turned off add anything,
/// and an entry for a module that is off is left out. Everything is a
/// declaration: `$health` and `$poll` bindings are resolved by the dashboard.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Contributions {
    /// Sidebar sections and their pages: a built-in section's extra tabs, a
    /// plugin's own section, or a page that is a section of its own.
    pub sections: Vec<SectionContribution>,
    /// Overview cards (`hero`, `status`, `tile`).
    pub overview: Vec<OverviewContribution>,
    /// The `ui.statusCards` polls the overview's `$poll` bindings read.
    pub polls: Vec<PollContribution>,
    pub agent: AgentContributions,
    /// The plugins with a setup wizard (whether it is finished is the wizard's own, in `/api/setup`).
    pub setup: Vec<SetupContribution>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentContributions {
    pub tools: Vec<AgentTool>,
}

/// One sidebar section plugins add or extend.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionContribution {
    /// A built-in section's id (`connections`: its pages go after the section's
    /// own tabs); `plugin-section:<pluginId>:<id>` for a plugin's own
    /// (`ui.sections`); or, for a page with no section, the page's view id.
    pub id: String,
    pub builtin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugin_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugin_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    /// Home, Work or Setup (a plugin's own section; a built-in keeps its own).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    /// The module it is shown with (it is left out while that is off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// Its pages, as tabs (after a built-in section's own).
    pub pages: Vec<PageContribution>,
}

/// One plugin screen in the dashboard (`ui.nav[]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageContribution {
    /// `plugin:<pluginId>:<navId>`.
    pub view: String,
    pub plugin_id: String,
    pub plugin_name: String,
    pub nav_id: String,
    pub label: String,
    /// The `ui.screens` id it shows.
    pub screen: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
}

/// One overview card (`ui.overview[]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverviewContribution {
    pub plugin_id: String,
    pub plugin_name: String,
    pub id: String,
    /// `hero`, `status` or `tile`.
    pub kind: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// Its texts by name (`headline`, `body`, `value`, …): plain text, or
    /// `$health.<path>` / `$poll.<statusCardId>.<path>` for the dashboard to look up.
    pub bind: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cta: Option<Cta>,
    /// Where clicking it goes (a tile's `nav`), when that page is shown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
}

/// A card's button: a label and the page it opens.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cta {
    pub label: String,
    pub view: String,
}

/// A read-only command of a plugin's, polled for `$poll.<id>.<path>` bindings.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PollContribution {
    pub plugin_id: String,
    /// The `ui.statusCards` id.
    pub id: String,
    pub connector: String,
    pub command: String,
    /// At least [`MIN_POLL_MS`].
    pub interval_ms: u64,
}

/// A plugin with a setup wizard.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupContribution {
    pub plugin_id: String,
    pub title: String,
    pub version: u32,
    /// How many steps it declares (those this OAIY can run).
    pub steps: usize,
}

/// A `ui` id: letters, digits, `-`, `_` and `.` (no `:`, which view ids are split on).
fn is_ui_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn text(o: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    o.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// A `ui` list of objects (a missing list is empty; a list that is not one, and an entry that is not an object, are warned about).
fn ui_list<'a>(ui: Option<&'a Value>, key: &str, warn: &mut Vec<String>) -> Vec<(usize, &'a serde_json::Map<String, Value>)> {
    let Some(v) = ui.and_then(|u| u.get(key)) else { return Vec::new() };
    let Some(list) = v.as_array() else {
        warn.push(format!("ui.{key} is not a list, so it is left out."));
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, item) in list.iter().enumerate() {
        match item.as_object() {
            Some(o) => out.push((i, o)),
            None => warn.push(format!("ui.{key}[{i}] is not an object, so it is left out.")),
        }
    }
    out
}

/// `ui.<key>[i] ("id")`, for a warning.
fn entry(key: &str, i: usize, o: &serde_json::Map<String, Value>) -> String {
    match o.get("id").and_then(Value::as_str) {
        Some(id) => format!("ui.{key}[{i}] ({id:?})"),
        None => format!("ui.{key}[{i}]"),
    }
}

/// A module an entry names: `Err` (with why) when this OAIY does not know it.
fn entry_module(o: &serde_json::Map<String, Value>) -> Result<Option<String>, String> {
    match o.get("module") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if def(id).is_some() => Ok(Some(id.clone())),
        Some(other) => Err(format!("names the module {other}, which this OAIY does not know")),
    }
}

/// Is `binding` one the dashboard can look up (or plain text)? `Some(card)` for a `$poll` binding.
fn check_binding(binding: &str, cards: &BTreeMap<String, PollContribution>) -> Result<Option<String>, String> {
    if let Some(path) = binding.strip_prefix("$health.") {
        let first = path.split('.').next().unwrap_or("");
        if !HEALTH_FIELDS.contains(&first) || path.split('.').any(str::is_empty) {
            return Err(format!(
                "names {binding}, which is not in a plugin's health report ({})",
                HEALTH_FIELDS.join(", ")
            ));
        }
        return Ok(None);
    }
    if let Some(rest) = binding.strip_prefix("$poll.") {
        let (card, path) = rest.split_once('.').unwrap_or((rest, ""));
        if path.is_empty() || path.split('.').any(str::is_empty) {
            return Err(format!("names {binding}, which needs a path: $poll.<statusCardId>.<path>"));
        }
        if !cards.contains_key(card) {
            return Err(format!("names {binding}, but there is no usable ui.statusCards entry {card:?}"));
        }
        return Ok(Some(card.to_string()));
    }
    if binding.starts_with('$') {
        return Err(format!("names {binding}, which is not $health.<path> or $poll.<statusCardId>.<path>"));
    }
    Ok(None)
}

/// What one plugin adds.
struct PluginContributions {
    /// Its sections (pages joining a built-in section are in `builtin`).
    sections: Vec<SectionContribution>,
    builtin: Vec<(&'static str, PageContribution)>,
    overview: Vec<OverviewContribution>,
    polls: Vec<PollContribution>,
}

/// What `rec` (enabled, its manifest loaded) adds to the dashboard, with what is wrong with its `ui`.
fn plugin_ui(rec: &PluginRecord, m: &PluginManifest, on: &dyn Fn(&str) -> bool, warn: &mut Vec<String>) -> PluginContributions {
    let ui = m.extra.get("ui");
    let screens = m.ui_screen_ids();
    let name = m.name.clone();
    let mut out = PluginContributions { sections: Vec::new(), builtin: Vec::new(), overview: Vec::new(), polls: Vec::new() };

    // ui.sections: the plugin's own sections, which its pages may name.
    struct Own {
        id: String,
        label: String,
        icon: Option<String>,
        group: String,
        module: Option<String>,
    }
    let mut own: Vec<Own> = Vec::new();
    for (i, s) in ui_list(ui, "sections", warn) {
        let at = entry("sections", i, s);
        let Some(id) = text(s, "id").filter(|id| is_ui_id(id)) else {
            warn.push(format!("{at} needs an id of letters, digits, '-', '_' or '.', so it is left out."));
            continue;
        };
        if own.iter().any(|o| o.id == id) || BUILTIN_SECTIONS.contains(&id.as_str()) {
            warn.push(format!("{at}: the id {id:?} is taken (twice, or by one of OAIY's sections), so it is left out."));
            continue;
        }
        let Some(label) = text(s, "label") else {
            warn.push(format!("{at} has no label, so it is left out."));
            continue;
        };
        let module = match entry_module(s) {
            Ok(m) => m,
            Err(why) => {
                warn.push(format!("{at} {why}, so it is left out."));
                continue;
            }
        };
        let group = group_of(s, &at, warn);
        own.push(Own { id, label, icon: text(s, "icon"), group, module });
    }

    // ui.nav: its pages. Each page's section: none (a section of its own), a built-in, or one of its own.
    let mut pages_shown: BTreeMap<String, String> = BTreeMap::new(); // navId -> view, for the pages shown now
    let mut nav_ids: BTreeSet<String> = BTreeSet::new();
    let mut used_own: BTreeSet<String> = BTreeSet::new();
    for (i, n) in ui_list(ui, "nav", warn) {
        let at = entry("nav", i, n);
        let Some(id) = text(n, "id").filter(|id| is_ui_id(id)) else {
            warn.push(format!("{at} needs an id of letters, digits, '-', '_' or '.', so it is left out."));
            continue;
        };
        if !nav_ids.insert(id.clone()) {
            warn.push(format!("{at}: the id {id:?} is used twice, so it is left out."));
            continue;
        }
        let Some(label) = text(n, "label") else {
            warn.push(format!("{at} has no label, so it is left out."));
            continue;
        };
        let Some(screen) = text(n, "screen").filter(|s| screens.contains(s)) else {
            warn.push(format!("{at} names no screen in ui.screens, so it is left out."));
            continue;
        };
        let module = match entry_module(n) {
            Ok(m) => m,
            Err(why) => {
                warn.push(format!("{at} {why}, so it is left out."));
                continue;
            }
        };
        let page = PageContribution {
            view: format!("plugin:{}:{id}", rec.id),
            plugin_id: rec.id.clone(),
            plugin_name: name.clone(),
            nav_id: id.clone(),
            label: label.clone(),
            screen,
            icon: text(n, "icon"),
            badge: text(n, "badge"),
            module: module.clone(),
        };
        let page_on = module.as_deref().map_or(true, on);
        let section = text(n, "section");
        if let Some(builtin) = section.as_deref().and_then(|s| BUILTIN_SECTIONS.iter().copied().find(|b| *b == s)) {
            // A built-in section that is there only with its module (the Calendar).
            let section_on = def(builtin).map_or(true, |d| on(d.id));
            if page_on && section_on {
                pages_shown.insert(id, page.view.clone());
                out.builtin.push((builtin, page));
            }
            continue;
        }
        if let Some(o) = section.as_deref().and_then(|s| own.iter().find(|o| o.id == s)) {
            used_own.insert(o.id.clone());
            if page_on && o.module.as_deref().map_or(true, on) {
                pages_shown.insert(id, page.view.clone());
                let sid = format!("plugin-section:{}:{}", rec.id, o.id);
                match out.sections.iter_mut().find(|s| s.id == sid) {
                    Some(s) => s.pages.push(page),
                    None => out.sections.push(SectionContribution {
                        id: sid,
                        builtin: false,
                        plugin_id: Some(rec.id.clone()),
                        plugin_name: Some(name.clone()),
                        label: Some(o.label.clone()),
                        icon: o.icon.clone(),
                        group: Some(o.group.clone()),
                        badge: None,
                        module: o.module.clone(),
                        pages: vec![page],
                    }),
                }
            }
            continue;
        }
        if let Some(s) = &section {
            warn.push(format!("{at} names the section {s:?}, which is neither one of OAIY's nor in ui.sections, so it is a section of its own."));
        }
        let group = group_of(n, &at, warn);
        if page_on {
            pages_shown.insert(id, page.view.clone());
            out.sections.push(SectionContribution {
                id: page.view.clone(),
                builtin: false,
                plugin_id: Some(rec.id.clone()),
                plugin_name: Some(name.clone()),
                label: Some(label),
                icon: page.icon.clone(),
                group: Some(group),
                badge: page.badge.clone(),
                module,
                pages: vec![page],
            });
        }
    }
    for o in own.iter().filter(|o| !used_own.contains(&o.id)) {
        warn.push(format!("ui.sections {:?} has no pages: no ui.nav entry names it.", o.id));
    }

    // ui.statusCards: the polls `$poll.<id>` reads. Only a declared, read-only command.
    let mut cards: BTreeMap<String, PollContribution> = BTreeMap::new();
    for (i, c) in ui_list(ui, "statusCards", warn) {
        let at = entry("statusCards", i, c);
        let Some(id) = text(c, "id").filter(|id| is_ui_id(id)) else {
            warn.push(format!("{at} needs an id of letters, digits, '-', '_' or '.', so it is left out."));
            continue;
        };
        let poll = c.get("poll").and_then(Value::as_object);
        let Some(command) = poll.and_then(|p| text(p, "command")) else {
            warn.push(format!("{at} has no poll.command, so it is left out."));
            continue;
        };
        let Some(connector) = m.connector_for(&command) else {
            warn.push(format!("{at} polls {command:?}, which no connector of this plugin declares, so it is left out."));
            continue;
        };
        if m.is_journalled(&command) {
            warn.push(format!("{at} polls {command:?}, which is journalled (it changes something), so it is left out."));
            continue;
        }
        let interval_ms = poll.and_then(|p| p.get("intervalMs")).and_then(Value::as_u64).unwrap_or(MIN_POLL_MS).max(MIN_POLL_MS);
        if cards.contains_key(&id) {
            warn.push(format!("{at}: the id {id:?} is used twice, so it is left out."));
            continue;
        }
        cards.insert(id.clone(), PollContribution { plugin_id: rec.id.clone(), id, connector: connector.id.clone(), command, interval_ms });
    }

    // ui.overview: cards on the Overview.
    let mut card_ids: BTreeSet<String> = BTreeSet::new();
    let mut polled: BTreeSet<String> = BTreeSet::new();
    'cards: for (i, o) in ui_list(ui, "overview", warn) {
        let at = entry("overview", i, o);
        let Some(id) = text(o, "id").filter(|id| is_ui_id(id)) else {
            warn.push(format!("{at} needs an id of letters, digits, '-', '_' or '.', so it is left out."));
            continue;
        };
        if !card_ids.insert(id.clone()) {
            warn.push(format!("{at}: the id {id:?} is used twice, so it is left out."));
            continue;
        }
        let kind = text(o, "kind").unwrap_or_else(|| "hero".into());
        if !OVERVIEW_KINDS.contains(&kind.as_str()) {
            warn.push(format!("{at} is a {kind:?}, which is not one of {}, so it is left out.", OVERVIEW_KINDS.join(", ")));
            continue;
        }
        let module = match entry_module(o) {
            Ok(m) => m,
            Err(why) => {
                warn.push(format!("{at} {why}, so it is left out."));
                continue;
            }
        };
        // A page it opens: one of the plugin's `ui.nav` ids (no view while that page is not shown).
        let open = |nav: &str, what: &str, warn: &mut Vec<String>| -> Result<Option<String>, ()> {
            if !nav_ids.contains(nav) {
                warn.push(format!("{at}: {what} names {nav:?}, which is not one of its ui.nav pages, so it is left out."));
                return Err(());
            }
            Ok(pages_shown.get(nav).cloned())
        };
        let mut bind = BTreeMap::new();
        let mut cta = None;
        let mut uses: Vec<String> = Vec::new();
        if let Some(b) = o.get("bind") {
            let Some(b) = b.as_object() else {
                warn.push(format!("{at}: bind is not an object, so it is left out."));
                continue;
            };
            for (key, value) in b {
                if key == "cta" {
                    let Some(c) = value.as_object() else {
                        warn.push(format!("{at}: bind.cta is not an object, so it is left out."));
                        continue 'cards;
                    };
                    let Some(nav) = text(c, "nav") else {
                        warn.push(format!("{at}: bind.cta names no nav, so it is left out."));
                        continue 'cards;
                    };
                    let Ok(view) = open(&nav, "bind.cta", warn) else { continue 'cards };
                    let label = text(c, "label").unwrap_or_else(|| format!("Open {}", text(o, "title").unwrap_or_else(|| name.clone())));
                    cta = view.map(|view| Cta { label, view });
                    continue;
                }
                let Some(s) = value.as_str() else {
                    warn.push(format!("{at}: bind.{key} is not text or a binding, so it is left out."));
                    continue 'cards;
                };
                match check_binding(s, &cards) {
                    Ok(card) => uses.extend(card),
                    Err(why) => {
                        warn.push(format!("{at}: bind.{key} {why}, so it is left out."));
                        continue 'cards;
                    }
                }
                bind.insert(key.clone(), s.to_string());
            }
        }
        let view = match text(o, "nav") {
            Some(nav) => match open(&nav, "nav", warn) {
                Ok(v) => v,
                Err(()) => continue,
            },
            None => None,
        };
        if module.as_deref().is_some_and(|id| !on(id)) {
            continue;
        }
        polled.extend(uses);
        out.overview.push(OverviewContribution {
            plugin_id: rec.id.clone(),
            plugin_name: name.clone(),
            id,
            kind,
            title: text(o, "title").unwrap_or_else(|| name.clone()),
            icon: text(o, "icon"),
            module,
            bind,
            cta,
            view,
        });
    }
    out.polls = polled.into_iter().filter_map(|id| cards.remove(&id)).collect();
    out
}

/// A section's group: Home, Work or Setup (Work, with a warning, for another).
fn group_of(o: &serde_json::Map<String, Value>, at: &str, warn: &mut Vec<String>) -> String {
    match text(o, "group") {
        None => "Work".into(),
        Some(g) if SECTION_GROUPS.contains(&g.as_str()) => g,
        Some(g) => {
            warn.push(format!("{at} is in the group {g:?}, which is not one of {}, so it is in Work.", SECTION_GROUPS.join(", ")));
            "Work".into()
        }
    }
}

/// What the enabled plugins in `records` (sorted by id) add, with what is wrong with it.
fn contribute(records: &[&PluginRecord], on: &dyn Fn(&str) -> bool) -> (Contributions, Vec<String>) {
    let mut out = Contributions::default();
    let mut warnings = Vec::new();
    let mut builtin: Vec<(&'static str, PageContribution)> = Vec::new();
    for rec in records.iter().filter(|r| !r.user_disabled) {
        let Some(m) = rec.manifest.as_ref() else { continue };
        let mut warn: Vec<String> = m.warnings.clone();
        let parts = plugin_ui(rec, m, on, &mut warn);
        out.sections.extend(parts.sections);
        builtin.extend(parts.builtin);
        out.overview.extend(parts.overview);
        out.polls.extend(parts.polls);
        for tool in &m.resolved_agent_tools {
            if let Some(other) = out.agent.tools.iter().find(|t| t.name == tool.name) {
                warn.push(format!(
                    "its agent tool {:?} is left out: {} has one of that name (the lowest plugin id is used).",
                    tool.name, other.plugin_id
                ));
                continue;
            }
            out.agent.tools.push(AgentTool { plugin_id: rec.id.clone(), ..tool.clone() });
        }
        if let Some(setup) = &m.setup {
            out.setup.push(SetupContribution {
                plugin_id: rec.id.clone(),
                // (Loading fills in a missing title; a manifest built in memory may not have one.)
                title: if setup.title.trim().is_empty() { format!("Set up {}", m.name.trim()) } else { setup.title.clone() },
                version: setup.version,
                steps: setup.steps.len(),
            });
        }
        warnings.extend(warn.into_iter().map(|w| format!("{}: {w}", rec.id)));
    }
    // The built-in sections' extra tabs, in the sidebar's order, before the plugins' own sections.
    let mut extended: Vec<SectionContribution> = Vec::new();
    for id in BUILTIN_SECTIONS {
        let pages: Vec<PageContribution> = builtin.iter().filter(|(b, _)| b == id).map(|(_, p)| p.clone()).collect();
        if !pages.is_empty() {
            extended.push(SectionContribution {
                id: id.to_string(),
                builtin: true,
                plugin_id: None,
                plugin_name: None,
                label: None,
                icon: None,
                group: None,
                badge: None,
                module: None,
                pages,
            });
        }
    }
    extended.append(&mut out.sections);
    out.sections = extended;
    (out, warnings)
}

// ---- The snapshot, and its revision ---------------------------------------------

/// The modules as `/api/modules` shows them.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// Moves when (and only when) anything below changes.
    pub revision: u64,
    pub modules: Vec<Module>,
    /// What the plugins add to the dashboard and the Agent app.
    pub contributions: Contributions,
    pub warnings: Vec<String>,
}

impl Snapshot {
    fn empty() -> Self {
        Self { revision: 0, modules: Vec::new(), contributions: Contributions::default(), warnings: Vec::new() }
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
        if self.current.revision > 0
            && self.current.modules == resolved.modules
            && self.current.contributions == resolved.contributions
            && self.current.warnings == resolved.warnings
        {
            return (self.current.clone(), false);
        }
        self.current = Arc::new(Snapshot {
            revision: self.current.revision + 1,
            modules: resolved.modules,
            contributions: resolved.contributions,
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
            trust: None,
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
            trust: None,
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
        // (A loaded manifest cannot name an unknown module; one built in memory can.)
        m["modules"] = json!(["calendar", "fax"]);
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
        // Aokie with no ui adds nothing.
        assert_eq!(v["contributions"], json!({"sections": [], "overview": [], "polls": [], "agent": {"tools": []}, "setup": []}));
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

    // ---- contributions ----------------------------------------------------------

    /// Aokie's manifest at schemaVersion 4 (the fixture `plugins::manifest` loads).
    fn aokie_v4() -> Value {
        serde_json::from_str(include_str!("../plugins/fixtures/aokie-v4.manifest.json")).unwrap()
    }

    /// A manifest with its agent tools resolved against Aokie's phone definition, as loading does.
    fn loaded(body: Value) -> PluginManifest {
        let mut m = manifest(body);
        let phone: crate::plugins::definitions::ServiceDefinition =
            serde_json::from_str(include_str!("../plugins/fixtures/aokie-phone.definition.json")).unwrap();
        m.resolved_agent_tools = m.resolve_agent_tools(&[phone]).expect("the tools resolve");
        m
    }

    fn loaded_record(body: Value, state: PluginState, user_disabled: bool) -> PluginRecord {
        let mut r = record(body.clone(), state, user_disabled);
        r.manifest = Some(loaded(body));
        r
    }

    /// Aokie at v4 with the design's additions: its page and hero shown with the phone,
    /// a tile polling data delivery, and two agent tools.
    fn aokie_v4_full() -> Value {
        let mut v = aokie_v4();
        v["ui"]["nav"][0]["module"] = json!("phone");
        v["ui"]["overview"][0]["module"] = json!("phone");
        v["ui"]["overview"].as_array_mut().unwrap().push(json!({
            "id": "outbox", "kind": "tile", "title": "Waiting to send", "icon": "cloud",
            "bind": { "value": "$poll.data-delivery.outbox.pending" }, "nav": "receptionist"
        }));
        v["agentTools"] = json!([
            { "action": "aokie.phone/sms.threads", "name": "phone_sms_threads", "audience": ["project", "runner"] },
            { "action": "aokie.phone/call.dial", "name": "phone_call", "audience": ["runner"], "confirm": "Call {number} and say: {openingLine}" }
        ]);
        v
    }

    fn contributions_json(r: &Resolved) -> Value {
        serde_json::to_value(&r.contributions).unwrap()
    }

    #[test]
    fn a_v4_aokie_contributes_its_page_cards_tools_and_setup() {
        let r = resolve(&[loaded_record(aokie_v4_full(), PluginState::Crashed, false)]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let c = contributions_json(&r);
        assert_eq!(
            c["sections"],
            json!([{
                "id": "plugin:aokie:receptionist", "builtin": false, "pluginId": "aokie", "pluginName": "Aokie Phone Bridge",
                "label": "AI Receptionist", "icon": "phone", "group": "Work", "badge": "New", "module": "phone",
                "pages": [{ "view": "plugin:aokie:receptionist", "pluginId": "aokie", "pluginName": "Aokie Phone Bridge", "navId": "receptionist",
                            "label": "AI Receptionist", "screen": "receptionist-home", "icon": "phone", "badge": "New", "module": "phone" }]
            }])
        );
        assert_eq!(
            c["overview"][0],
            json!({
                "pluginId": "aokie", "pluginName": "Aokie Phone Bridge", "id": "aokie-hero", "kind": "hero", "title": "Aokie receptionist",
                "icon": "phone", "module": "phone", "bind": { "headline": "$health.status", "body": "$health.detail" },
                "cta": { "label": "Open AI Receptionist", "view": "plugin:aokie:receptionist" }
            }),
            "bindings are passed on as declared, for the dashboard to look up"
        );
        assert_eq!(c["overview"][1]["kind"], "tile");
        assert_eq!(c["overview"][1]["view"], "plugin:aokie:receptionist");
        assert_eq!(c["overview"][1]["bind"], json!({ "value": "$poll.data-delivery.outbox.pending" }));
        assert_eq!(
            c["polls"],
            json!([{ "pluginId": "aokie", "id": "data-delivery", "connector": "aokie", "command": "dongle.diagnostics", "intervalMs": 10000 }]),
            "only the polls a card reads"
        );
        let tools = c["agent"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools[0],
            json!({
                "pluginId": "aokie", "name": "phone_sms_threads", "action": "aokie.phone/sms.threads", "definition": "aokie.phone", "actionId": "sms.threads",
                "description": "Every conversation on the paired phone, most recent first.", "inputSchema": { "type": "object" },
                "sideEffects": "none", "audience": ["project", "runner"], "timeoutMs": 20000
            })
        );
        assert_eq!(tools[1]["sideEffects"], "external-write");
        assert_eq!(tools[1]["confirm"], "Call {number} and say: {openingLine}");
        assert_eq!(tools[1]["inputSchema"]["required"], json!(["number", "openingLine"]));
        assert_eq!(c["setup"], json!([{ "pluginId": "aokie", "title": "Set up Aokie Phone Bridge", "version": 1, "steps": 3 }]));
    }

    #[test]
    fn a_turned_off_or_broken_plugin_contributes_nothing() {
        let off = resolve(&[loaded_record(aokie_v4_full(), PluginState::Disabled, true)]);
        assert_eq!(off.contributions, Contributions::default());
        let mut broken = loaded_record(aokie_v4_full(), PluginState::Disabled, false);
        broken.manifest = None;
        assert_eq!(resolve(&[broken]).contributions, Contributions::default());
    }

    /// A plugin with no phone of its own, whose entries go with the phone.
    fn acme(ui: Value) -> Value {
        json!({
            "schemaVersion": 4, "id": "acme", "name": "Acme Tools", "version": "1.0.0", "pluginApiVersion": 1,
            "entry": {"kind": "process", "command": "acme.exe"},
            "connectors": [{"id": "acme", "commands": ["stats.get", "stats.reset"]}],
            "commands": {"journalled": ["stats.reset"]},
            "ui": ui,
        })
    }

    fn screens() -> Value {
        json!([{ "id": "home", "entry": "ui/index.html" }, { "id": "logs", "entry": "ui/logs.html" }])
    }

    #[test]
    fn entries_for_a_module_that_is_off_are_left_out() {
        let ui = json!({
            "screens": screens(),
            "nav": [ { "id": "calls", "label": "Call stats", "screen": "home", "module": "phone" },
                     { "id": "logs", "label": "Logs", "screen": "logs" } ],
            "overview": [ { "id": "calls-hero", "kind": "hero", "title": "Calls", "module": "phone", "bind": { "cta": { "nav": "calls" } } },
                          { "id": "logs-tile", "kind": "tile", "title": "Logs", "nav": "calls" } ]
        });
        // No phone: the phone's page and card are left out; the tile stays, with no page to open.
        let r = resolve(&[record(acme(ui.clone()), PluginState::Running, false)]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let views: Vec<&str> = r.contributions.sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(views, vec!["plugin:acme:logs"]);
        let cards: Vec<&str> = r.contributions.overview.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(cards, vec!["logs-tile"]);
        assert_eq!(r.contributions.overview[0].view, None, "its page is not shown now");
        // With Aokie providing the phone, they are there.
        let r = resolve(&[record(acme(ui), PluginState::Running, false), record(aokie_manifest(), PluginState::Running, false)]);
        let views: Vec<&str> = r.contributions.sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(views, vec!["plugin:acme:calls", "plugin:acme:logs"]);
        assert_eq!(r.contributions.overview.len(), 2);
        assert_eq!(r.contributions.overview[0].cta, Some(Cta { label: "Open Calls".into(), view: "plugin:acme:calls".into() }));
        assert_eq!(r.contributions.overview[1].view.as_deref(), Some("plugin:acme:calls"));
    }

    #[test]
    fn a_page_joins_a_builtin_section_or_the_plugins_own() {
        let ui = json!({
            "screens": screens(),
            "sections": [ { "id": "tools", "label": "Acme", "icon": "bot", "group": "Setup" } ],
            "nav": [ { "id": "keys", "label": "Acme keys", "screen": "home", "section": "connections" },
                     { "id": "stats", "label": "Stats", "screen": "home", "section": "tools" },
                     { "id": "logs", "label": "Logs", "screen": "logs", "section": "tools" },
                     { "id": "diary", "label": "Acme diary", "screen": "home", "section": "calendar" } ]
        });
        let r = resolve(&[record(acme(ui.clone()), PluginState::Running, false)]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let c = contributions_json(&r);
        assert_eq!(c["sections"][0]["id"], "connections");
        assert_eq!(c["sections"][0]["builtin"], true);
        assert_eq!(c["sections"][0]["pages"][0]["view"], "plugin:acme:keys");
        assert!(c["sections"][0].get("label").is_none(), "a built-in keeps its own label");
        assert_eq!(c["sections"][1]["id"], "plugin-section:acme:tools");
        assert_eq!(c["sections"][1]["label"], "Acme");
        assert_eq!(c["sections"][1]["group"], "Setup");
        let tabs: Vec<&str> = c["sections"][1]["pages"].as_array().unwrap().iter().map(|p| p["navId"].as_str().unwrap()).collect();
        assert_eq!(tabs, vec!["stats", "logs"]);
        assert_eq!(c["sections"].as_array().unwrap().len(), 2, "no Calendar, so no page in it: {c}");
        // With the calendar on, the page joins it (after Connections, in the sidebar's order).
        let r = resolve(&[record(acme(ui), PluginState::Running, false), record(aokie_manifest(), PluginState::Running, false)]);
        let ids: Vec<&str> = r.contributions.sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["calendar", "connections", "plugin-section:acme:tools"]);
    }

    #[test]
    fn bad_ui_entries_are_left_out_with_a_warning() {
        let ui = json!({
            "screens": screens(),
            "sections": [ { "id": "empty", "label": "Nothing here" }, { "id": "agent", "label": "Not yours" } ],
            "nav": [ { "id": "ok", "label": "Fine", "screen": "home", "group": "Sideways", "section": "elsewhere" },
                     { "id": "noscreen", "label": "No screen", "screen": "missing" },
                     { "id": "fax", "label": "Fax", "screen": "home", "module": "fax" },
                     { "label": "No id", "screen": "home" },
                     { "id": "ok", "label": "Twice", "screen": "home" } ],
            "statusCards": [ { "id": "stats", "poll": { "command": "stats.get", "intervalMs": 1000 } },
                             { "id": "reset", "poll": { "command": "stats.reset" } },
                             { "id": "ghost", "poll": { "command": "ghost.get" } } ],
            "overview": [ { "id": "typo", "kind": "hero", "bind": { "body": "$health.detial" } },
                          { "id": "spin", "kind": "carousel" },
                          { "id": "reset-tile", "kind": "tile", "bind": { "value": "$poll.reset.count" } },
                          { "id": "math", "kind": "tile", "bind": { "value": "$eval(1+1)" } },
                          { "id": "lost", "kind": "hero", "bind": { "cta": { "nav": "nowhere" } } },
                          { "id": "number", "kind": "tile", "bind": { "value": 7 } },
                          { "id": "stats-tile", "kind": "tile", "title": "Stats", "bind": { "value": "$poll.stats.count", "caption": "counted" } } ]
        });
        let r = resolve(&[record(acme(ui), PluginState::Running, false)]);
        let said = |needle: &str| assert!(r.warnings.iter().any(|w| w.starts_with("acme: ") && w.contains(needle)), "{needle}: {:#?}", r.warnings);
        said("ui.sections \"empty\" has no pages");
        said("ui.sections[1] (\"agent\"): the id \"agent\" is taken");
        said("ui.nav[0] (\"ok\") is in the group \"Sideways\"");
        said("ui.nav[0] (\"ok\") names the section \"elsewhere\"");
        said("ui.nav[1] (\"noscreen\") names no screen");
        said("ui.nav[2] (\"fax\") names the module \"fax\"");
        said("ui.nav[3] needs an id");
        said("ui.nav[4] (\"ok\"): the id \"ok\" is used twice");
        said("ui.statusCards[1] (\"reset\") polls \"stats.reset\", which is journalled");
        said("ui.statusCards[2] (\"ghost\") polls \"ghost.get\", which no connector");
        said("ui.overview[0] (\"typo\"): bind.body names $health.detial, which is not in a plugin's health report");
        said("ui.overview[1] (\"spin\") is a \"carousel\"");
        said("ui.overview[2] (\"reset-tile\"): bind.value names $poll.reset.count, but there is no usable ui.statusCards entry \"reset\"");
        said("ui.overview[3] (\"math\"): bind.value names $eval(1+1)");
        said("ui.overview[4] (\"lost\"): bind.cta names \"nowhere\"");
        said("ui.overview[5] (\"number\"): bind.value is not text");
        assert_eq!(r.warnings.len(), 16, "{:#?}", r.warnings);
        // What is fine stays: the page (a section of its own, in Work), and the one good card with its poll (5 s at the most often).
        let s = &r.contributions.sections;
        assert_eq!((s.len(), s[0].id.as_str(), s[0].group.as_deref()), (1, "plugin:acme:ok", Some("Work")));
        let cards: Vec<&str> = r.contributions.overview.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(cards, vec!["stats-tile"]);
        assert_eq!(r.contributions.overview[0].bind.get("caption").map(String::as_str), Some("counted"), "plain text is text");
        assert_eq!(
            r.contributions.polls,
            vec![PollContribution { plugin_id: "acme".into(), id: "stats".into(), connector: "acme".into(), command: "stats.get".into(), interval_ms: MIN_POLL_MS }]
        );
    }

    #[test]
    fn load_warnings_reach_the_snapshot() {
        let mut rec = loaded_record(aokie_v4(), PluginState::Running, false);
        rec.manifest.as_mut().unwrap().warnings.push("setup.steps[3] (\"answer\"): the host step \"phone.hologram\" needs a newer OAIY, so it is left out.".into());
        let r = resolve(&[rec]);
        assert!(r.warnings.iter().any(|w| w.starts_with("aokie: setup.steps[3]") && w.contains("newer OAIY")), "{:?}", r.warnings);
    }

    #[test]
    fn one_tool_name_is_given_to_the_lowest_plugin_id() {
        let mut other = aokie_v4_full();
        other["id"] = json!("zed-phone");
        other["name"] = json!("Zed Phone");
        other.as_object_mut().unwrap().remove("setup");
        other["ui"] = json!({});
        other["modules"] = json!([]);
        let r = resolve(&[
            loaded_record(other, PluginState::Running, false),
            loaded_record(aokie_v4_full(), PluginState::Running, false),
        ]);
        let names: Vec<(&str, &str)> = r.contributions.agent.tools.iter().map(|t| (t.plugin_id.as_str(), t.name.as_str())).collect();
        assert_eq!(names, vec![("aokie", "phone_sms_threads"), ("aokie", "phone_call")]);
        assert!(r.warnings.iter().any(|w| w.starts_with("zed-phone: its agent tool \"phone_sms_threads\" is left out: aokie")), "{:?}", r.warnings);
    }

    #[test]
    fn a_change_in_what_plugins_add_moves_the_revision() {
        let mut p = Publisher::default();
        let (first, _) = p.update(resolve(&[loaded_record(aokie_v4_full(), PluginState::Running, false)]));
        let (same, changed) = p.update(resolve(&[loaded_record(aokie_v4_full(), PluginState::Running, false)]));
        assert!(!changed && same.revision == first.revision);
        let mut fewer = aokie_v4_full();
        fewer["agentTools"] = json!([]);
        let (next, changed) = p.update(resolve(&[loaded_record(fewer, PluginState::Running, false)]));
        assert!(changed, "only the tools changed");
        assert_eq!(next.revision, first.revision + 1);
        assert!(next.contributions.agent.tools.is_empty());
        // A health report is not part of it: a new one does not move the revision.
        let mut healthy = loaded_record(aokie_v4_full(), PluginState::Running, false);
        healthy.manifest.as_mut().unwrap().agent_tools.clear();
        healthy.manifest.as_mut().unwrap().resolved_agent_tools.clear();
        healthy.last_health = Some(json!({"status": "ok", "detail": "Phone connected"}));
        let (again, changed) = p.update(resolve(&[healthy]));
        assert!(!changed && again.revision == next.revision);
    }

    #[test]
    fn leases_belong_to_their_module() {
        assert_eq!(lease_module("answer-calls").map(|d| d.id), Some(PHONE));
        assert_eq!(lease_module("answer-texts").map(|d| d.id), Some(PHONE));
        assert!(lease_module("answer-tasks").is_none(), "flows' tasks are core");
    }
}
