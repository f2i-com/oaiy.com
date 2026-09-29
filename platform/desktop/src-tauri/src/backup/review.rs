//! What a restore lets the person look at and tick before it brings anything back.
//!
//! Most of what a backup holds is data: contacts, the calendar, conversations, voices. Some of it is
//! more than data, because a program acts on it: a service template names a program to run, a flow
//! and a trigger run when something happens, a provider list says where the AI's words (and keys) are
//! sent, a connector descriptor says where a link goes, the settings say what OAIY and the Agent may
//! do. A hostile backup restored by mistake could use every one of those, so none of them comes back
//! unless the person ticked its class, and the dry run lists every item of every class by name and by
//! what it does. Data comes back without a tick.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use super::table::{filter_json, table, Class, Why};
use super::{BackupError, ErrorKind, Result};

/// The classes of things that can act, each ticked (or not) on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RestoreClass {
    /// What OAIY and the Agent may do: the Agent's switch, what was accepted for plugins, its model.
    Settings,
    /// Service templates (each names a program to run) and which services start with OAIY.
    Templates,
    /// Flows, triggers and the run journal.
    Flows,
    /// The AI providers OAIY's gateway talks to, and where.
    Providers,
    /// Connector descriptors: where a link to a provider goes.
    Connections,
    /// The settings of plugins OAIY knows how to back up.
    Plugins,
    /// The Agent's own settings: its providers, its network gate, how it answers calls and texts.
    AgentSettings,
    /// Voice clips and settings: what callers hear.
    Voices,
    /// What is remembered about people: read by the AI as facts and instructions before it answers them.
    Memory,
    /// Outreach campaigns: texts and calls to a list of people. They come back paused.
    Outreach,
    /// The Agent's projects, conversations, brief and knowledge files: read by the Agent as its context
    /// and instructions.
    AgentData,
}

impl RestoreClass {
    pub const ALL: [RestoreClass; 11] = [
        RestoreClass::Settings,
        RestoreClass::Templates,
        RestoreClass::Flows,
        RestoreClass::Providers,
        RestoreClass::Connections,
        RestoreClass::Plugins,
        RestoreClass::AgentSettings,
        RestoreClass::Voices,
        RestoreClass::Memory,
        RestoreClass::Outreach,
        RestoreClass::AgentData,
    ];

    pub fn id(self) -> &'static str {
        match self {
            RestoreClass::Settings => "settings",
            RestoreClass::Templates => "templates",
            RestoreClass::Flows => "flows",
            RestoreClass::Providers => "providers",
            RestoreClass::Connections => "connections",
            RestoreClass::Plugins => "plugins",
            RestoreClass::AgentSettings => "agentSettings",
            RestoreClass::Voices => "voices",
            RestoreClass::Memory => "memory",
            RestoreClass::Outreach => "outreach",
            RestoreClass::AgentData => "agentData",
        }
    }

    pub fn from_id(id: &str) -> Option<RestoreClass> {
        Self::ALL.into_iter().find(|c| c.id() == id)
    }

    pub fn label(self) -> &'static str {
        match self {
            RestoreClass::Settings => "Settings that decide what OAIY and the Agent may do",
            RestoreClass::Templates => "Service templates and what starts with OAIY",
            RestoreClass::Flows => "Flows, triggers and run history",
            RestoreClass::Providers => "AI providers (the addresses OAIY sends your AI requests to)",
            RestoreClass::Connections => "Connector descriptors (where a link to a provider goes)",
            RestoreClass::Plugins => "Plugin settings",
            RestoreClass::AgentSettings => "The Agent's own settings (its providers, network gate, and how it answers calls and texts)",
            RestoreClass::Voices => "Voices your callers hear",
            RestoreClass::Memory => "Contacts and what is remembered about people",
            RestoreClass::Outreach => "Outreach campaigns (texts and calls to a list of people)",
            RestoreClass::AgentData => "The Agent's projects, conversations, brief and knowledge files",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            RestoreClass::Settings => "These switch on what OAIY and the Agent are allowed to do: whether the Agent may change OAIY, which plugin permissions count as accepted, which model the Agent uses.",
            RestoreClass::Templates => "A service template names a program that OAIY runs, and can start it with OAIY at every start. Only tick templates you recognise as yours.",
            RestoreClass::Flows => "A flow runs when it is triggered and can call your AI providers, send messages and run scripts; a trigger decides when. Runs that were waiting are never brought back.",
            RestoreClass::Providers => "Where your AI requests, and your conversations in them, are sent. Keys come back only if you also tick the keys box.",
            RestoreClass::Connections => "A connector descriptor points OAIY's link at a provider's address.",
            RestoreClass::Plugins => "The settings of a plugin. PINs, keys and values sealed to another computer are never in them.",
            RestoreClass::AgentSettings => "Which servers the Agent talks to, whether its network gate is open, whether it answers calls and texts by itself, and the instructions it answers them by. A provider at another address arrives without a key, beside yours.",
            RestoreClass::Voices => "A voice is what your callers hear. A sample or a setting from a file that was not made by you would speak to them in your name.",
            RestoreClass::Memory => "The receptionist and the Agent read what is remembered about a person, and the notes for the receptionist, before they answer them. It is read as instructions, so a file that was not made by you could steer what they say.",
            RestoreClass::Outreach => "A campaign texts or calls the people on its list. A restored campaign is always PAUSED: it is never running and nothing is scheduled. It is listed by name with the number of people, and you start each one yourself.",
            RestoreClass::AgentData => "The Agent reads its projects, conversations, the front desk's brief and its knowledge files as context and instructions: the brief wins over what the phone's agents would otherwise say. Each project and file is listed by name and size.",
        }
    }
}

/// What the person chose to bring back besides the data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ticks {
    pub classes: BTreeSet<RestoreClass>,
    /// Bring back the API keys the backup holds (only where the provider lists are ticked too).
    pub keys: bool,
}

impl Ticks {
    /// Only data.
    pub fn none() -> Self {
        Self::default()
    }

    /// Every class (not the keys).
    pub fn all() -> Self {
        Self { classes: RestoreClass::ALL.into_iter().collect(), keys: false }
    }

    pub fn from_ids(classes: &[String], keys: bool) -> Result<Self> {
        let mut set = BTreeSet::new();
        for id in classes {
            set.insert(RestoreClass::from_id(id).ok_or_else(|| BackupError::new(ErrorKind::Unsupported, "That choice is not one this version of OAIY knows."))?);
        }
        Ok(Self { classes: set, keys })
    }

    pub fn has(&self, class: RestoreClass) -> bool {
        self.classes.contains(&class)
    }

    pub fn ids(&self) -> Vec<String> {
        self.classes.iter().map(|c| c.id().to_string()).collect()
    }
}

/// One thing a restore could bring back that can act, described for the person.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReviewItem {
    pub class: RestoreClass,
    /// Where it is in the backup.
    pub name: String,
    pub title: String,
    /// What it does, in a sentence.
    pub what: String,
}

/// A class as the dry run lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ClassInfo {
    pub id: String,
    pub label: String,
    pub description: String,
    pub count: usize,
}

/// The most items of these classes one backup may hold and still be reviewed.
pub const MAX_REVIEW_ITEMS: usize = 2000;

/// Cut `text` to `max` characters.
pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let mut cut: String = text.chars().take(max.saturating_sub(1)).collect();
        cut.push('…');
        cut
    }
}

fn s<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// What is known about this computer that a description compares with.
#[derive(Default)]
pub struct Local {
    /// The ids of the service templates here (the built-in ones are seeded before any restore).
    pub template_ids: HashSet<String>,
    /// The ids of the connectors OAIY ships.
    pub builtin_connectors: HashSet<String>,
}

impl Local {
    pub fn read(data_dir: &Path) -> Self {
        let mut template_ids = HashSet::new();
        if let Ok(entries) = std::fs::read_dir(data_dir.join("templates")) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Some(id) = std::fs::read(entry.path()).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| s(&v, "id").map(str::to_string)) {
                    template_ids.insert(id);
                }
            }
        }
        let builtin_connectors = crate::link::descriptor::load_all(&data_dir.join("does-not-exist")).into_iter().map(|d| d.id).collect();
        Self { template_ids, builtin_connectors }
    }
}

/// How the description of a file that could not be read begins (see [`is_unreadable`]).
const UNREADABLE: &str = "Could not be read (";

/// What is said of a file too large to look at.
pub const TOO_LARGE: &str = "Too large to look at: it is not brought back.";

fn unreadable(class: RestoreClass, name: &str, why: &str) -> ReviewItem {
    ReviewItem { class, name: name.to_string(), title: clip(name.rsplit('/').next().unwrap_or(name), 120), what: format!("{UNREADABLE}{why}): OAIY would not load it, so it is not brought back.") }
}

/// Whether the description says its file is not brought back because it could not be read (or is too
/// large to be looked at). The dry run says so, and staging leaves such a file out: the two agree.
pub fn is_unreadable(item: &ReviewItem) -> bool {
    item.what.starts_with(UNREADABLE) || item.what == TOO_LARGE
}

/// Describe one restorable file of a class that can act.
pub fn describe(class: RestoreClass, name: &str, bytes: &[u8], local: &Local, backup_templates: &HashSet<String>) -> Vec<ReviewItem> {
    let value = || serde_json::from_slice::<Value>(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes));
    match name {
        "services-autostart.json" => match serde_json::from_slice::<Vec<String>>(bytes) {
            Ok(ids) => ids
                .into_iter()
                .map(|id| {
                    let known = if backup_templates.contains(&id) {
                        "its template is in this backup"
                    } else if local.template_ids.contains(&id) {
                        "its template is already here"
                    } else {
                        "it has no template, so it is left out"
                    };
                    ReviewItem { class, name: name.to_string(), title: clip(&id, 120), what: format!("Starts with OAIY at every start ({known}).") }
                })
                .collect(),
            Err(_) => vec![unreadable(class, name, "it is not a list of services")],
        },
        "triggers.json" => match value() {
            // Read the way OAIY's own trigger store reads it: a list, each entry a binding, and an entry that is not
            // one is skipped (so it is not described as if it would run).
            Ok(Value::Array(rows)) => {
                let mut items = Vec::new();
                let mut ignored = 0usize;
                for row in &rows {
                    match serde_json::from_value::<crate::bridge::triggers::TriggerBinding>(row.clone()) {
                        Ok(b) => {
                            let mode = format!("{:?}", b.mode).to_lowercase();
                            let off = if b.enabled { "" } else { ", switched off" };
                            let when = b.condition.as_deref().filter(|c| !c.trim().is_empty()).map(|c| format!(" and only if this holds: {}", clip(c, 120))).unwrap_or_default();
                            items.push(ReviewItem {
                                class,
                                name: name.to_string(),
                                title: clip(&b.id, 120),
                                what: clip(&format!("When \"{}\" happens{when}, runs the flow \"{}\" ({mode}{off}).", b.event, b.flow_id), 500),
                            });
                        }
                        Err(_) => ignored += 1,
                    }
                }
                if items.is_empty() && ignored > 0 {
                    vec![unreadable(class, name, &format!("none of its {ignored} entries is a trigger OAIY would load"))]
                } else {
                    if ignored > 0 {
                        items.push(ReviewItem {
                            class,
                            name: name.to_string(),
                            title: "Entries that will not load".to_string(),
                            what: format!("{ignored} entr{} in the file {} not triggers OAIY would load, and {} ignored.", if ignored == 1 { "y" } else { "ies" }, if ignored == 1 { "is" } else { "are" }, if ignored == 1 { "is" } else { "are" }),
                        });
                    }
                    items
                }
            }
            _ => vec![unreadable(class, name, "it is not a list of triggers")],
        },
        "bridge/ledger.jsonl" => {
            let (kept, left_out) = super::sanitize::ledger_finished_only(bytes);
            let finished = kept.iter().filter(|b| **b == b'\n').count();
            vec![ReviewItem {
                class,
                name: name.to_string(),
                title: "Run history".to_string(),
                what: format!("{finished} finished run records are brought back. {left_out} records of runs that were waiting or running are left out, so nothing starts by itself."),
            }]
        }
        "ai/providers.json" => match value() {
            Ok(v) => v
                .get("providers")
                .and_then(Value::as_array)
                .map(|providers| {
                    providers
                        .iter()
                        .map(|p| {
                            let id = s(p, "id").unwrap_or("(no id)");
                            let key = if s(p, "apiKey").is_some_and(|k| !k.trim().is_empty()) { "; has an API key (brought back only with the keys box)" } else { "; no key" };
                            let local_net = if p.get("allowLocal").and_then(Value::as_bool).unwrap_or(false) { "; may use this computer's own addresses" } else { "" };
                            ReviewItem {
                                class,
                                name: name.to_string(),
                                title: clip(&format!("{} ({id})", s(p, "name").unwrap_or(id)), 120),
                                what: clip(&format!("{} requests go to {}{key}{local_net}.", s(p, "protocol").unwrap_or("openai"), s(p, "baseUrl").unwrap_or("(no address)")), 300),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "control.json" => {
            let on = value().ok().and_then(|v| v.get("agentMayChange").and_then(Value::as_bool)).unwrap_or(false);
            vec![ReviewItem {
                class,
                name: name.to_string(),
                title: "The Agent's switch".to_string(),
                what: if on { "Lets the Agent set up and change OAIY for you: ON.".to_string() } else { "Lets the Agent set up and change OAIY for you: off.".to_string() },
            }]
        }
        "setup.json" => match value() {
            Ok(v) => {
                let plugins = v.get("plugins").and_then(Value::as_object);
                let accepted: Vec<String> = plugins
                    .map(|m| m.iter().filter(|(_, p)| p.get("permissionsAccepted").is_some_and(|a| a.as_array().is_some_and(|a| !a.is_empty()))).map(|(id, _)| id.clone()).collect())
                    .unwrap_or_default();
                vec![ReviewItem {
                    class,
                    name: name.to_string(),
                    title: "Setup record".to_string(),
                    what: clip(
                        &if accepted.is_empty() {
                            "Records how far setup got. No plugin permissions are marked as accepted.".to_string()
                        } else {
                            format!("Records how far setup got and marks the permissions of these plugins as ACCEPTED: {}.", accepted.join(", "))
                        },
                        300,
                    ),
                }]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "agent.json" => {
            let source = value().ok().and_then(|v| v.get("model").and_then(|m| s(m, "source").map(str::to_string))).unwrap_or_else(|| "(not set)".to_string());
            vec![ReviewItem { class, name: name.to_string(), title: "The Agent's model".to_string(), what: clip(&format!("The Agent thinks with: {source}."), 200) }]
        }
        _ if name.starts_with("templates/") => match value() {
            Ok(v) => {
                let id = s(&v, "id").unwrap_or("");
                let run = v.get("run");
                let command = run.and_then(|r| s(r, "command")).unwrap_or("(none)");
                let args: Vec<String> = run.and_then(|r| r.get("args")).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
                let mut what = format!("Runs \"{}{}\"", command, if args.is_empty() { String::new() } else { format!(" {}", args.join(" ")) });
                if let Some(install) = v.get("install").filter(|i| s(i, "kind") == Some("script")) {
                    what.push_str(&format!("; install script {}", [s(install, "windows"), s(install, "unix")].into_iter().flatten().collect::<Vec<_>>().join(" / ")));
                }
                if let Some(files) = v.get("files").and_then(Value::as_object).filter(|f| !f.is_empty()) {
                    what.push_str(&format!("; writes {} script file(s): {}", files.len(), files.keys().take(6).cloned().collect::<Vec<_>>().join(", ")));
                }
                if let Some(paths) = v.get("uninstall").and_then(|u| u.get("paths")).and_then(Value::as_array).filter(|p| !p.is_empty()) {
                    what.push_str(&format!("; deletes {} path(s) when uninstalled", paths.len()));
                }
                if let Some(env) = run.and_then(|r| r.get("env")).and_then(Value::as_object).filter(|e| !e.is_empty()) {
                    what.push_str(&format!("; sets {} environment variable(s): {}", env.len(), env.keys().take(6).cloned().collect::<Vec<_>>().join(", ")));
                }
                if let Some(cwd) = run.and_then(|r| s(r, "cwd")) {
                    what.push_str(&format!("; runs in {cwd}"));
                }
                if v.get("autostart").and_then(Value::as_bool).unwrap_or(false) {
                    what.push_str("; STARTS with OAIY once installed");
                }
                if !id.is_empty() && local.template_ids.contains(id) {
                    what.push_str("; replaces your template of the same id");
                }
                vec![ReviewItem { class, name: name.to_string(), title: clip(&format!("{} ({id})", s(&v, "name").unwrap_or(id)), 120), what: clip(&what, 600) }]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        _ if name.starts_with("flows/") => match value() {
            Ok(v) => {
                let title = s(&v, "name").or_else(|| s(&v, "title")).unwrap_or_else(|| name.rsplit('/').next().unwrap_or(name));
                let nodes = v.get("nodes").and_then(Value::as_array).map(|n| n.len()).unwrap_or(0);
                let mut kinds: Vec<String> = v.get("nodes").and_then(Value::as_array).map(|n| n.iter().filter_map(|x| s(x, "type").or_else(|| x.get("data").and_then(|d| s(d, "type"))).map(str::to_string)).collect()).unwrap_or_default();
                kinds.sort();
                kinds.dedup();
                // A flow can also be offered to the Agent as a tool, or run before or after one of the Agent's own tools.
                let mut extra = String::new();
                if let Some(tool) = v.get("oaiyTool") {
                    extra.push_str(&format!(" It is offered to the Agent as the tool \"{}\".", clip(s(tool, "name").unwrap_or("?"), 60)));
                }
                if let Some(hook) = v.get("oaiyToolHook") {
                    extra.push_str(&format!(" It runs {} the Agent's \"{}\" tool.", clip(s(hook, "mode").unwrap_or("around"), 20), clip(s(hook, "tool").unwrap_or("?"), 60)));
                }
                vec![ReviewItem {
                    class,
                    name: name.to_string(),
                    title: clip(title, 120),
                    what: clip(&format!("A flow with {nodes} step(s){}.{extra}", if kinds.is_empty() { String::new() } else { format!(": {}", kinds.into_iter().take(8).collect::<Vec<_>>().join(", ")) }), 500),
                }]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        _ if name.starts_with("connectors/") => match value() {
            Ok(v) => {
                let id = s(&v, "id").unwrap_or("(no id)");
                let overrides = if local.builtin_connectors.contains(id) { "; REPLACES the connector OAIY ships with this id" } else { "" };
                vec![ReviewItem {
                    class,
                    name: name.to_string(),
                    title: clip(&format!("{} ({id})", s(&v, "name").unwrap_or(id)), 120),
                    what: clip(&format!("A link to a provider, prefilled with the address {}{overrides}.", s(&v, "defaultBaseUrl").unwrap_or("(none)")), 300),
                }]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "callers.json" => match value() {
            Ok(v) => {
                let entries = v.get("contacts").and_then(Value::as_array).map(Vec::len).or_else(|| v.as_array().map(Vec::len)).or_else(|| v.as_object().map(|o| o.len())).unwrap_or(0);
                vec![ReviewItem {
                    class,
                    name: name.to_string(),
                    title: "Contacts and what is remembered about callers".to_string(),
                    what: format!("{entries} entr{}: the names, facts and notes that the receptionist and the Agent read about a person before they answer them.", if entries == 1 { "y" } else { "ies" }),
                }]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        _ if name.starts_with("plugin-data/") => {
            let plugin = name.split('/').nth(1).unwrap_or("?");
            match table().key_table(&format!("plugin.{plugin}")) {
                Some(keys) => match value() {
                    Ok(v) => {
                        let found = filter_json(keys, &v, &|_| true);
                        let mut items: Vec<ReviewItem> = found.kept.iter().filter(|k| k.row.class == Class::Runs && !matches!(k.value, Value::Object(_))).map(|k| key_item(name, k)).collect();
                        if items.is_empty() {
                            items.push(ReviewItem { class, name: name.to_string(), title: clip(&format!("Settings of the \"{plugin}\" plugin"), 120), what: "Nothing in it acts: only settings that cannot act (or nothing) come back, and those come back without a tick.".to_string() });
                        }
                        items
                    }
                    Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
                },
                None => vec![unreadable(class, name, "OAIY does not know this plugin's settings")],
            }
        }
        _ => Vec::new(),
    }
}

/// A voice file, by its name and size: it is audio, and is never read.
pub fn describe_voice(class: RestoreClass, name: &str, size: u64) -> ReviewItem {
    ReviewItem { class, name: name.to_string(), title: clip(name.rsplit('/').next().unwrap_or(name), 120), what: format!("A voice file ({} KB): what your callers hear.", size.div_ceil(1024)) }
}

/// A value as a person is shown it: text is cut, with how long it is.
pub fn show_value(value: &Value) -> String {
    match value {
        Value::String(text) => {
            let length = text.chars().count();
            if length > 160 {
                format!("\"{}\" ({length} characters in all)", clip(text, 160))
            } else {
                format!("\"{text}\"")
            }
        }
        Value::Array(items) => format!("a list of {}", items.len()),
        Value::Object(map) => format!("{} entr{}", map.len(), if map.len() == 1 { "y" } else { "ies" }),
        other => other.to_string(),
    }
}

/// What one key that acts is, for the dry run: by its key, its name and its value.
pub fn key_item(file: &str, kept: &super::table::Kept) -> ReviewItem {
    let class = kept.row.tick.unwrap_or(RestoreClass::Settings);
    let mut what = format!("Sets {} to {}.", kept.path, show_value(&kept.value));
    if !kept.row.reason.is_empty() {
        what.push(' ');
        what.push_str(&kept.row.reason);
    }
    ReviewItem { class, name: format!("{file}#{}", kept.path), title: clip(&kept.row.what, 120), what: clip(&what, 700) }
}

/// Something in a backup that is not brought back, and why: for the list of what is not restored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NotRestored {
    pub name: String,
    pub why: String,
}

/// What a settings file holds that is not brought back at all (excluded, or not in the table), by key.
pub fn keys_not_restored(file: &str, table_name: &str, bytes: &[u8]) -> Vec<NotRestored> {
    let Some(keys) = table().key_table(table_name) else { return Vec::new() };
    let Ok(v) = serde_json::from_slice::<Value>(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)) else { return Vec::new() };
    let found = filter_json(keys, &v, &|_| true);
    let mut out: Vec<NotRestored> = found
        .left
        .iter()
        .map(|l| NotRestored {
            name: clip(&format!("{file}#{}", l.path), 200),
            why: clip(
                &match (&l.why, l.row) {
                    (Why::Unknown, _) => "not restored: unknown item".to_string(),
                    (Why::Excluded, Some(row)) => format!("not restored: {}{}", row.reason, row.redo.as_ref().map(|r| format!(" To do again: {r}")).unwrap_or_default()),
                    (Why::BadValue(why), _) => format!("not restored: its value is not one this version accepts ({why})"),
                    (Why::NotTicked(class), _) => format!("only with the tick \"{}\"", class.label()),
                    (Why::Excluded, None) => "not restored".to_string(),
                },
                400,
            ),
        })
        .collect();
    if found.left_more > 0 {
        out.push(NotRestored { name: clip(file, 200), why: format!("and {} more keys that are not restored", found.left_more) });
    }
    out
}

/// What the Agent's settings in a backup say, for the dry run: its providers and where they point, its
/// network gate, how it answers calls and texts, its image and video service.
pub fn describe_agent_settings(settings_json: &[u8]) -> Vec<ReviewItem> {
    let class = RestoreClass::AgentSettings;
    let name = "agent/idb/settings.json";
    let Ok(v) = serde_json::from_slice::<Value>(settings_json) else {
        return vec![unreadable(class, name, "it is not valid JSON")];
    };
    let mut items = Vec::new();
    for p in v.get("providers").and_then(Value::as_array).into_iter().flatten() {
        let id = s(p, "id").unwrap_or("(no id)");
        let key = if s(p, "apiKey").is_some_and(|k| !k.trim().is_empty()) { "; has an API key (brought back only with the keys box, and only where yours has none)" } else { "" };
        items.push(ReviewItem {
            class,
            name: name.to_string(),
            title: clip(&format!("{} ({id})", s(p, "name").unwrap_or(id)), 120),
            what: clip(&format!("Agent provider of type {} at {}{key}. If yours of the same id is at another address, this one arrives beside it, without a key.", s(p, "type").unwrap_or("?"), s(p, "baseUrl").or_else(|| s(p, "endpoint")).unwrap_or("(default address)")), 400),
        });
    }
    if let Some(gate) = v.get("gate") {
        items.push(ReviewItem {
            class,
            name: name.to_string(),
            title: "The network gate".to_string(),
            what: clip(&format!("Which sites the Agent's code may reach: mode {}, {} allowed, {} denied.", s(gate, "mode").unwrap_or("?"), gate.get("allow").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0), gate.get("deny").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0)), 300),
        });
    }
    if let Some(m) = v.get("messages") {
        let flag = |k: &str| if m.get(k).and_then(Value::as_bool).unwrap_or(false) { "ON" } else { "off" };
        items.push(ReviewItem {
            class,
            name: name.to_string(),
            title: "Calls and texts".to_string(),
            what: format!("The Agent answers texts by itself: {}; answers calls: {}; rings missed calls back: {}.", flag("answer"), flag("calls"), flag("callBack")),
        });
    }
    if let Some(media) = v.get("media").filter(|m| s(m, "baseUrl").is_some_and(|u| !u.is_empty())) {
        items.push(ReviewItem { class, name: name.to_string(), title: "Images, video and audio".to_string(), what: clip(&format!("The image and video service is at {}.", s(media, "baseUrl").unwrap_or("?")), 300) });
    }
    items
}
