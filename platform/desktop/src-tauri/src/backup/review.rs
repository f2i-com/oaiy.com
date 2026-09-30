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

pub use super::parts::ReviewItem;
use super::parts::{quoted, short, some_of, text_problem, Parts};
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
    /// The calendar's words: the business's name, the services and the appointments' names and notes, which the receptionist
    /// reads before it answers and says to callers.
    Calendar,
    /// The phone's earlier conversations (each call and text thread): loaded into a model as what was said before, and
    /// handed back to it by the earlier_conversations tool.
    Conversations,
    /// Outreach campaigns: texts and calls to a list of people. They come back paused.
    Outreach,
    /// The Agent's projects, conversations, brief and knowledge files: read by the Agent as its context
    /// and instructions.
    AgentData,
}

impl RestoreClass {
    pub const ALL: [RestoreClass; 13] = [
        RestoreClass::Settings,
        RestoreClass::Templates,
        RestoreClass::Flows,
        RestoreClass::Providers,
        RestoreClass::Connections,
        RestoreClass::Plugins,
        RestoreClass::AgentSettings,
        RestoreClass::Voices,
        RestoreClass::Memory,
        RestoreClass::Calendar,
        RestoreClass::Conversations,
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
            RestoreClass::Calendar => "calendar",
            RestoreClass::Conversations => "conversations",
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
            RestoreClass::Memory => "Contacts and notes your receptionist reads",
            RestoreClass::Calendar => "Calendar text your receptionist reads",
            RestoreClass::Conversations => "Earlier conversations (calls and texts)",
            RestoreClass::Outreach => "Outreach campaigns (texts and calls to a list of people)",
            RestoreClass::AgentData => "The Agent's projects, brief and knowledge files",
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
            RestoreClass::Calendar => "The receptionist reads the business's name, the services (their names, prices and descriptions) and, for a caller, their appointments before it answers, says them to callers, and the Agent reads each appointment's name and notes. The phone also sends every appointment it has no copy of at FormLogic to your linked FormLogic account, so the appointments come only with the tick too. Opening hours and the steps between the times offered are brought back without a tick.",
            RestoreClass::Conversations => "Every call and text thread is loaded into the model as what was said before, and the earlier_conversations tool hands what was said in earlier calls and texts back to it, so a conversation from a file that was not made by you could steer what the receptionist says next. Each is listed by size.",
            RestoreClass::Outreach => "A campaign texts or calls the people on its list. A restored campaign is always PAUSED: it is never running and nothing is scheduled. It is listed by name with the number of people, and you start each one yourself.",
            RestoreClass::AgentData => "The Agent reads its projects (their files and their own conversations), the front desk's brief and its knowledge files as context and instructions: the brief wins over what the phone's agents would otherwise say. Each project and file is listed by name and size.",
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

/// Cut `text` to `max` characters, with how long it was in all when it is cut, and the characters a person cannot see made visible (it is
/// [`short`]: there is one way the dry run cuts a text).
pub fn clip(text: &str, max: usize) -> String {
    short(text, max)
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

/// How the description of a file that hides text begins (see [`is_unreadable`] and [`hidden_text_in`]).
const HIDES: &str = "Not brought back: it hides text (";

/// What is said of a file too large to look at.
pub const TOO_LARGE: &str = "Too large to look at: it is not brought back.";

fn unreadable(class: RestoreClass, name: &str, why: &str) -> ReviewItem {
    unreadable_as(class, name, name.rsplit('/').next().unwrap_or(name), why)
}

/// A thing that could not be read, listed under this title: said so, and not brought back.
pub fn unreadable_as(class: RestoreClass, name: &str, title: &str, why: &str) -> ReviewItem {
    Parts::new("unreadable").fixed("problem", format!("{UNREADABLE}{why}): OAIY would not load it, so it is not brought back.")).item(class, name, title)
}

/// A thing that is too large to look at: said so, and not brought back.
pub fn too_large(class: RestoreClass, name: &str, title: &str) -> ReviewItem {
    Parts::new("unreadable").fixed("problem", TOO_LARGE).item(class, name, title)
}

/// Whether the description says its file is not brought back because it could not be read (or is too
/// large to be looked at). The dry run says so, and staging leaves such a file out: the two agree.
pub fn is_unreadable(item: &ReviewItem) -> bool {
    item.what.starts_with(UNREADABLE) || item.what.starts_with(HIDES) || item.what == TOO_LARGE
}

/// The first value of a JSON document that hides text (see [`super::parts::text_problem`]), by where it is, and why: a key or a text.
fn first_hidden_text(value: &Value, at: &str) -> Option<String> {
    let here = |key: &str| if at.is_empty() { key.to_string() } else { format!("{at}.{key}") };
    match value {
        Value::String(text) => text_problem(text).map(|why| format!("{}: {why}", short(at, 80))),
        Value::Array(items) => items.iter().enumerate().find_map(|(i, v)| first_hidden_text(v, &format!("{at}[{i}]"))),
        Value::Object(map) => map.iter().find_map(|(key, v)| text_problem(key).map(|why| format!("a key of {}: {why}", short(&here(key), 80))).or_else(|| first_hidden_text(v, &here(key)))),
        _ => None,
    }
}

/// Why a file of the desktop that is copied whole, and that holds words a model reads or a caller hears (a flow's descriptions and prompts,
/// a template's, a trigger's condition, a connector's, the notes about callers, the setup, agent and control records), is not brought back:
/// a value in it hides text. The dry run says it, and staging leaves the file out.
fn hidden_text_in(name: &str, bytes: &[u8]) -> Option<String> {
    let watched = name.starts_with("flows/") || name.starts_with("templates/") || name.starts_with("connectors/") || matches!(name, "triggers.json" | "callers.json" | "setup.json" | "agent.json" | "control.json" | "services-autostart.json");
    if !watched {
        return None;
    }
    let value: Value = serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes)).ok()?;
    first_hidden_text(&value, "")
}

/// `3 places`, `1 place`.
fn words(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// One address of a descriptor, and where it is.
struct Address {
    path: String,
    value: String,
}

/// The host an address goes to (`//host/path` is read as `https://host/path`), or `(not an address)`.
fn host_of(address: &str) -> String {
    let text = address.trim();
    let full = if text.starts_with("//") { format!("https:{text}") } else { text.to_string() };
    reqwest::Url::parse(&full).ok().and_then(|u| u.host_str().map(|h| h.to_lowercase())).unwrap_or_else(|| "(not an address)".to_string())
}

/// Whether a value is an address: a text under a key that ends in `Url` or `url`, or one under a key that ends in `Path` or `path` when it
/// is itself an address (it names a host, or starts with `//`).
fn is_address(key: &str, value: &Value) -> bool {
    let lower = key.to_lowercase();
    matches!(value, Value::String(text) if lower.ends_with("url") || (lower.ends_with("path") && (text.contains("://") || text.starts_with("//"))))
}

/// Every address in a JSON document, with where it is. Every one is kept, however many a place holds (an array of five thousand steps is
/// five thousand addresses), so that what is said of a place can count them and name their hosts.
fn collect_addresses(value: &Value, at: &str, out: &mut Vec<Address>, visited: &mut usize) {
    *visited += 1;
    if *visited > 200_000 {
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                let here = if at.is_empty() { key.clone() } else { format!("{at}.{key}") };
                match v {
                    Value::String(text) if is_address(key, v) => out.push(Address { path: here, value: text.clone() }),
                    _ => collect_addresses(v, &here, out, visited),
                }
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                collect_addresses(v, &format!("{at}[{i}]"), out, visited);
            }
        }
        _ => {}
    }
}

/// The addresses of a descriptor by the place (top-level key) they are in, for the places that have any.
fn addresses_by_place(descriptor: &Value) -> Vec<(String, Vec<Address>)> {
    let Value::Object(map) = descriptor else { return Vec::new() };
    let mut visited = 0usize;
    let mut places = Vec::new();
    for (key, v) in map {
        let mut here = Vec::new();
        match v {
            Value::String(text) if is_address(key, v) => here.push(Address { path: key.clone(), value: text.clone() }),
            _ => collect_addresses(v, key, &mut here, &mut visited),
        }
        if !here.is_empty() {
            places.push((key.clone(), here));
        }
    }
    places
}

/// The keys of a descriptor that are addresses by their names (`...Url`, `...Path`), by place, as the descriptor OAIY ships has them: what
/// a descriptor of its own has, each said by name and in this order whatever else its object holds. (Most of them are relative paths
/// there; a descriptor that puts an address in one is the case.) A key added to the shipped descriptor is one of them. Only keys of the
/// descriptor itself: an entry of an array is not.
fn reference_keys() -> &'static std::collections::BTreeMap<String, Vec<String>> {
    static KEYS: std::sync::OnceLock<std::collections::BTreeMap<String, Vec<String>>> = std::sync::OnceLock::new();
    fn named_like_addresses(value: &Value, at: &str, out: &mut Vec<String>) {
        if let Value::Object(map) = value {
            for (key, v) in map {
                let here = if at.is_empty() { key.clone() } else { format!("{at}.{key}") };
                let lower = key.to_lowercase();
                match v {
                    Value::String(_) if lower.ends_with("url") || lower.ends_with("path") => out.push(here),
                    _ => named_like_addresses(v, &here, out),
                }
            }
        }
    }
    KEYS.get_or_init(|| {
        let reference: Value = serde_json::from_str(include_str!("../../resources/connectors/formlogic.json")).unwrap_or(Value::Null);
        let mut by_place = std::collections::BTreeMap::new();
        if let Value::Object(map) = &reference {
            for (place, v) in map {
                let mut keys = Vec::new();
                match v {
                    Value::String(_) if place.to_lowercase().ends_with("url") || place.to_lowercase().ends_with("path") => keys.push(place.clone()),
                    _ => named_like_addresses(v, place, &mut keys),
                }
                by_place.insert(place.clone(), keys);
            }
        }
        by_place
    })
}
/// The places of a connector that are parts of its description, by the key of the descriptor and the label of the part, in the order they are
/// said (the address it is prefilled with is a part of its own).
const KNOWN_PLACES: [(&str, &str); 11] = [
    ("auth", "auth"),
    ("healthPath", "health"),
    ("heartbeat", "heartbeat"),
    ("relay", "relay"),
    ("desktopFlows", "desktopFlows"),
    ("desktopAi", "desktopAi"),
    ("flows", "flows"),
    ("appLogic", "appLogic"),
    ("dataNode", "dataNode"),
    ("scriptProfile", "scriptProfile"),
    ("docsUrl", "docs"),
];

/// How many other places (a key of the descriptor that is not one of those it has) are named, and how many hosts of a place.
const MAX_OTHER_PLACES: usize = 12;
const MAX_HOSTS_SAMPLED: usize = 4;

/// The hosts some addresses go to: the first few, and how many more.
fn hosts_said(addresses: &[&Address]) -> String {
    let hosts: BTreeSet<String> = addresses.iter().map(|a| host_of(&a.value)).collect();
    let named = hosts.iter().take(MAX_HOSTS_SAMPLED).map(|h| short(h, 80)).collect::<Vec<_>>().join(", ");
    if hosts.len() > MAX_HOSTS_SAMPLED {
        format!("{named} and {} more hosts", hosts.len() - MAX_HOSTS_SAMPLED)
    } else {
        named
    }
}

/// What one place of a descriptor sends to: the addresses of the keys the shipped descriptor has, each by its key and in its order
/// (whatever else the object of a key holds), then every other address of the place (a key the shipped descriptor has not, an entry of a
/// list or a map) as a count and a sample of the hosts they go to. What is said before the count is bounded by the shipped descriptor, so
/// no number of other addresses, and no length of any, can push one of its own keys out.
fn place_said(place: &str, all: &[Address]) -> String {
    let reference = reference_keys().get(place).cloned().unwrap_or_default();
    let named: Vec<&Address> = reference.iter().filter_map(|k| all.iter().find(|a| a.path == *k)).collect();
    let mut text = named.iter().map(|a| format!("{} = {}", a.path, short(&a.value, 100))).collect::<Vec<_>>().join("; ");
    let other: Vec<&Address> = all.iter().filter(|a| !reference.contains(&a.path)).collect();
    if !other.is_empty() {
        if !text.is_empty() {
            text.push_str("; ");
        }
        text.push_str(&format!("{} besides, to {}", words(other.len(), "other address", "other addresses"), hosts_said(&other)));
    }
    format!("{text}.")
}

/// Describe one restorable file of a class that can act.
pub fn describe(class: RestoreClass, name: &str, bytes: &[u8], local: &Local, backup_templates: &HashSet<String>) -> Vec<ReviewItem> {
    // A value that says one thing to a person and another to a model is not brought back: nothing of the file is described as if it would be.
    if let Some(why) = hidden_text_in(name, bytes) {
        return vec![Parts::new("unreadable")
            .fixed("problem", format!("{HIDES}{why}): a value in it says one thing to a person and another to a model, so nothing of it comes back."))
            .item(class, name, name.rsplit('/').next().unwrap_or(name))];
    }
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
                    Parts::new("autostart").fixed("starts", format!("Starts with OAIY at every start ({known}).")).item(class, name, &id)
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
                            // How it runs and whether it is on come first, each a part of its own, then which flow it runs, then what starts it, then
                            // what it waits for: an event, a flow or a condition written as long as it may be cannot push another out.
                            let mut parts = Parts::new("trigger").fixed("mode", format!("Runs in the mode {mode}."));
                            if !b.enabled {
                                parts = parts.fixed("state", "It is switched off.");
                            }
                            parts = parts.fixed("runs", format!("Runs the flow \"{}\".", b.flow_id)).fixed("when", format!("When \"{}\" happens.", b.event));
                            if let Some(c) = b.condition.as_deref().filter(|c| !c.trim().is_empty()) {
                                parts = parts.sample("condition", format!("Only if this holds: {c}."));
                            }
                            items.push(parts.item(class, name, &b.id));
                        }
                        Err(_) => ignored += 1,
                    }
                }
                if items.is_empty() && ignored > 0 {
                    vec![unreadable(class, name, &format!("none of its {ignored} entries is a trigger OAIY would load"))]
                } else {
                    if ignored > 0 {
                        items.push(
                            Parts::new("more")
                                .fixed("count", format!("{ignored} entr{} in the file {} not triggers OAIY would load, and {} ignored.", if ignored == 1 { "y" } else { "ies" }, if ignored == 1 { "is" } else { "are" }, if ignored == 1 { "is" } else { "are" }))
                                .item(class, name, "Entries that will not load"),
                        );
                    }
                    items
                }
            }
            _ => vec![unreadable(class, name, "it is not a list of triggers")],
        },
        "bridge/ledger.jsonl" => {
            let (kept, left_out) = super::sanitize::ledger_finished_only(bytes);
            let finished = kept.iter().filter(|b| **b == b'\n').count();
            vec![Parts::new("ledger")
                .fixed("records", format!("{finished} finished run records are brought back. {left_out} records of runs that were waiting or running are left out, so nothing starts by itself."))
                .item(class, name, "Run history")]
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
                            // Where it goes is the last thing said, cut on its own: a long address cannot push out what it may reach (its own
                            // computer's addresses, which relaxes the network guard) or whether it has a key.
                            let mut parts = Parts::new("provider-list")
                                .fixed("protocol", format!("Protocol: {}.", s(p, "protocol").unwrap_or("openai")))
                                .fixed("key", if s(p, "apiKey").is_some_and(|k| !k.trim().is_empty()) { "It has an API key (brought back only with the keys box)." } else { "No key." });
                            if p.get("allowLocal").and_then(Value::as_bool).unwrap_or(false) {
                                parts = parts.fixed("local", "May use this computer's own addresses.");
                            }
                            parts.sample("address", format!("Requests go to {}.", s(p, "baseUrl").unwrap_or("(no address)"))).item(class, name, &format!("{} ({id})", s(p, "name").unwrap_or(id)))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "control.json" => {
            let on = value().ok().and_then(|v| v.get("agentMayChange").and_then(Value::as_bool)).unwrap_or(false);
            vec![Parts::new("control").fixed("switch", if on { "Lets the Agent set up and change OAIY for you: ON." } else { "Lets the Agent set up and change OAIY for you: off." }).item(class, name, "The Agent's switch")]
        }
        "setup.json" => match value() {
            Ok(v) => {
                let plugins = v.get("plugins").and_then(Value::as_object);
                let accepted: Vec<String> = plugins
                    .map(|m| m.iter().filter(|(_, p)| p.get("permissionsAccepted").is_some_and(|a| a.as_array().is_some_and(|a| !a.is_empty()))).map(|(id, _)| id.clone()).collect())
                    .unwrap_or_default();
                let parts = Parts::new("setup");
                let parts = if accepted.is_empty() {
                    parts.fixed("accepted", "Records how far setup got. No plugin permissions are marked as accepted.")
                } else {
                    // How many first, and then the first few by name: a list of hundreds of plugins is a sample, and says how long it is.
                    parts.fixed("accepted", format!("Records how far setup got and marks the permissions of {} as ACCEPTED.", words(accepted.len(), "plugin", "plugins"))).sample("names", format!("They are: {}.", some_of(&accepted, MAX_ACCEPTED_NAMED, 40)))
                };
                vec![parts.item(class, name, "Setup record")]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "agent.json" => {
            let source = value().ok().and_then(|v| v.get("model").and_then(|m| s(m, "source").map(str::to_string))).unwrap_or_else(|| "(not set)".to_string());
            vec![Parts::new("agent-model").fixed("model", format!("The Agent thinks with: {source}.")).item(class, name, "The Agent's model")]
        }
        _ if name.starts_with("templates/") => match value() {
            Ok(v) => {
                let id = s(&v, "id").unwrap_or("");
                let run = v.get("run");
                let command = run.and_then(|r| s(r, "command")).unwrap_or("(none)");
                let args: Vec<String> = run.and_then(|r| r.get("args")).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
                // Every fact about what it does is a part of its own, worked out before anything is cut and said in a fixed order, so a
                // long command line, or thousands of files, paths and variables, hide nothing: each part is cut on its own.
                let mut parts = Parts::new("template");
                if v.get("autostart").and_then(Value::as_bool).unwrap_or(false) {
                    parts = parts.fixed("autostart", "STARTS with OAIY once installed.");
                }
                if !id.is_empty() && local.template_ids.contains(id) {
                    parts = parts.fixed("replaces", "Replaces your template of the same id.");
                }
                parts = parts.fixed("runs", format!("Runs \"{}{}\".", short(command, 120), if args.is_empty() { String::new() } else { format!(" {}", short(&args.join(" "), 200)) }));
                if let Some(install) = v.get("install").filter(|i| s(i, "kind") == Some("script")) {
                    parts = parts.fixed("install", format!("Install script {}.", short(&[s(install, "windows"), s(install, "unix")].into_iter().flatten().collect::<Vec<_>>().join(" / "), 160)));
                }
                if let Some(files) = v.get("files").and_then(Value::as_object).filter(|f| !f.is_empty()) {
                    let listed: Vec<String> = files.iter().take(6).map(|(k, body)| format!("{} ({} bytes)", short(k, 40), body.as_str().map(str::len).unwrap_or(0))).collect();
                    parts = parts.fixed("writes", format!("Writes {} script file(s): {}.", files.len(), listed.join(", ")));
                }
                if let Some(paths) = v.get("uninstall").and_then(|u| u.get("paths")).and_then(Value::as_array).filter(|p| !p.is_empty()) {
                    parts = parts.fixed("deletes", format!("Deletes {} path(s) when uninstalled: {}.", paths.len(), paths.iter().filter_map(Value::as_str).take(3).map(|p| short(p, 60)).collect::<Vec<_>>().join(", ")));
                }
                if let Some(env) = run.and_then(|r| r.get("env")).and_then(Value::as_object).filter(|e| !e.is_empty()) {
                    parts = parts.fixed("env", format!("Sets {} environment variable(s): {}.", env.len(), env.keys().take(6).map(|k| short(k, 30)).collect::<Vec<_>>().join(", ")));
                }
                if let Some(cwd) = run.and_then(|r| s(r, "cwd")) {
                    parts = parts.fixed("cwd", format!("Runs in {}.", short(cwd, 120)));
                }
                if let Some(marker) = s(&v, "installedMarker").filter(|m| !m.is_empty()) {
                    parts = parts.fixed("marker", format!("Writes a marker file at {}.", short(marker, 120)));
                }
                if let Some(health) = v.get("health").and_then(|h| s(h, "url")) {
                    parts = parts.fixed("health", format!("Asks {} after it starts.", short(health, 120)));
                }
                if let Some(docs) = s(&v, "docsUrl").filter(|d| !d.is_empty()) {
                    parts = parts.fixed("docs", format!("Links to {}.", short(docs, 120)));
                }
                vec![parts.item(class, name, &format!("{} ({id})", s(&v, "name").unwrap_or(id)))]
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
                // What the flow does to the Agent (a tool it offers, and what the model is told of it, a tool it runs around) comes first, each
                // part cut on its own; the kinds of its steps are a sample after them.
                let mut parts = Parts::new("flow").fixed("steps", format!("A flow with {nodes} step(s)."));
                if let Some(tool) = v.get("oaiyTool") {
                    parts = parts.fixed("tool", format!("It is offered to the Agent as the tool \"{}\".", short(s(tool, "name").unwrap_or("?"), 60)));
                    if let Some(description) = s(tool, "description").map(str::trim).filter(|d| !d.is_empty()) {
                        parts = parts.fixed("tool-description", format!("The model is told: \"{description}\"."));
                    }
                    // What the model is asked for: the labels of the flow's inputs.
                    let labels: Vec<String> = v
                        .get("nodes")
                        .and_then(Value::as_array)
                        .map(|n| n.iter().filter(|x| s(x, "type").is_some_and(|t| ["input_text", "input_file", "input_folder", "input_audio", "input_video"].contains(&t))).map(|x| x.get("data").and_then(|d| s(d, "label")).or_else(|| s(x, "id")).unwrap_or("?").to_string()).collect())
                        .unwrap_or_default();
                    if !labels.is_empty() {
                        parts = parts.fixed("tool-inputs", format!("The model is asked for {}: {}.", words(labels.len(), "input", "inputs"), some_of(&labels, MAX_INPUTS_NAMED, 40)));
                    }
                }
                if let Some(hook) = v.get("oaiyToolHook") {
                    parts = parts.fixed("hook", format!("It runs {} the Agent's \"{}\" tool.", short(s(hook, "mode").unwrap_or("around"), 20), short(s(hook, "tool").unwrap_or("?"), 60)));
                }
                if !kinds.is_empty() {
                    parts = parts.sample("kinds", format!("Its steps are of the kinds: {}.", some_of(&kinds, MAX_KINDS_NAMED, 30)));
                }
                vec![parts.item(class, name, title)]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        _ if name.starts_with("connectors/") => match value() {
            Ok(v) => {
                let id = s(&v, "id").unwrap_or("(no id)");
                let places = addresses_by_place(&v);
                let place = |key: &str| places.iter().find(|(k, _)| k == key).map(|(_, a)| a.as_slice());
                // What replaces one OAIY ships, the address it is prefilled with and what it asks to be allowed come first; then each place it
                // sends to, by its own key, in a fixed order, each cut on its own: where it signs in, its health check, its heartbeat, where
                // it sends events. A place with thousands of addresses (or thirty more keys of its own) cannot push another out.
                let mut parts = Parts::new("connector");
                if local.builtin_connectors.contains(id) {
                    parts = parts.fixed("replaces", "REPLACES the connector OAIY ships with this id.");
                }
                parts = parts.fixed("prefilled", format!("A link to a provider, prefilled with the address {}.", short(s(&v, "defaultBaseUrl").unwrap_or("(none)"), 160)));
                if let Some(scopes) = v.pointer("/auth/scopes").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>()).filter(|s| !s.is_empty()) {
                    // How many first, and then the first few by name: a real descriptor asks for a handful.
                    parts = parts.fixed("scopes", format!("Asks to be allowed ({}): {}.", scopes.len(), some_of(&scopes, MAX_SCOPES_NAMED, 50)));
                }
                for (key, label) in KNOWN_PLACES {
                    if let Some(all) = place(key) {
                        parts = parts.fixed(label, place_said(key, all));
                    }
                }
                let total: usize = places.iter().map(|(_, a)| a.len()).sum();
                if total > 0 {
                    let everything: Vec<&Address> = places.iter().flat_map(|(_, a)| a.iter()).collect();
                    let hosts: BTreeSet<String> = everything.iter().map(|a| host_of(&a.value)).collect();
                    parts = parts.sample("summary", format!("It holds {} in {}, to {}.", words(total, "address", "addresses"), words(places.len(), "place", "places"), words(hosts.len(), "host", "hosts")));
                }
                let others: Vec<&(String, Vec<Address>)> = places.iter().filter(|(k, _)| k != "defaultBaseUrl" && !KNOWN_PLACES.iter().any(|(n, _)| n == k)).collect();
                if !others.is_empty() {
                    let said = others.iter().take(MAX_OTHER_PLACES).map(|(k, a)| format!("{}: {} to {}", short(k, 40), words(a.len(), "address", "addresses"), hosts_said(&a.iter().collect::<Vec<_>>()))).collect::<Vec<_>>().join("; ");
                    parts = parts.sample("other-places", format!("Other places it sends to: {said}{}.", if others.len() > MAX_OTHER_PLACES { format!(" and {} more places", others.len() - MAX_OTHER_PLACES) } else { String::new() }));
                }
                vec![parts.item(class, name, &format!("{} ({id})", s(&v, "name").unwrap_or(id)))]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "callers.json" => match value() {
            Ok(v) => {
                let entries = v.get("contacts").and_then(Value::as_array).map(Vec::len).or_else(|| v.as_array().map(Vec::len)).or_else(|| v.as_object().map(|o| o.len())).unwrap_or(0);
                vec![Parts::new("callers")
                    .fixed("entries", format!("{entries} entr{}: the names, facts and notes that the receptionist and the Agent read about a person before they answer them.", if entries == 1 { "y" } else { "ies" }))
                    .item(class, name, "Contacts and what is remembered about callers")]
            }
            Err(_) => vec![unreadable(class, name, "it is not valid JSON")],
        },
        "calendar/calendar.json" => match value() {
            Ok(v) => describe_calendar(class, name, &v),
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
                            items.push(Parts::new("nothing").fixed("nothing", "Nothing in it is a setting OAIY restores: what is not left out is call handling, and none of it is in this file.").item(class, name, &format!("Settings of the \"{plugin}\" plugin")));
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

/// How many of the scopes a connector asks for, of the plugins a setup record marks as accepted, and of a flow's inputs, are named (the rest
/// are counted).
const MAX_SCOPES_NAMED: usize = 20;
const MAX_ACCEPTED_NAMED: usize = 10;
const MAX_INPUTS_NAMED: usize = 8;
const MAX_KINDS_NAMED: usize = 8;

/// The most services and appointments the dry run lists one by one (the rest are counted).
const MAX_CALENDAR_LISTED: usize = 40;

/// The calendar in a backup, by every word in it that the receptionist or the Agent reads: the business's and the
/// receptionist's names, whether it texts, and for each service and each appointment the words of it, by value (cut, with how
/// long each is). It is built from the key table, so a key added to the table with words in it is listed by construction.
fn describe_calendar(class: RestoreClass, name: &str, document: &Value) -> Vec<ReviewItem> {
    let Some(keys) = table().key_table("calendar") else { return vec![unreadable(class, name, "OAIY does not know how to read it")] };
    let found = filter_json(keys, document, &|_| true);
    let mut items: Vec<ReviewItem> = found.kept.iter().filter(|k| k.row.class == Class::Runs && !k.path.contains("[]") && !matches!(k.value, Value::Object(_) | Value::Array(_))).map(|k| key_item(name, k)).collect();
    // The services and the appointments: one item for each, with its words by value.
    let mut groups: std::collections::BTreeMap<(&str, usize), Vec<&super::table::Kept>> = std::collections::BTreeMap::new();
    for kept in &found.kept {
        if let (Some(list), Some(at)) = (kept.path.split("[]").next().filter(|_| kept.path.contains("[]")), kept.at.first()) {
            groups.entry((if list == "appointments" { "appointments" } else { "settings.services" }, *at)).or_default().push(kept);
        }
    }
    let position = |path: &str| keys.keys.iter().position(|r| r.path == path).unwrap_or(usize::MAX);
    for list in ["settings.services", "appointments"] {
        let mut total = 0usize;
        let mut words_of = 0usize;
        for ((_, at), kept) in groups.iter().filter(|((l, _), _)| *l == list) {
            total += 1;
            let mut says: Vec<&&super::table::Kept> = kept.iter().filter(|k| k.row.class == Class::Runs).collect();
            says.sort_by_key(|k| position(&k.path));
            if says.is_empty() {
                continue;
            }
            words_of += 1;
            if words_of > MAX_CALENDAR_LISTED {
                continue;
            }
            let when = kept.iter().find(|k| k.path.ends_with(".start")).and_then(|k| k.value.as_str()).map(|s| format!(" at {}", short(s, 20))).unwrap_or_default();
            let (title, parts, what) = if list == "appointments" {
                (format!("Appointment {}{when}", at + 1), Parts::new("calendar-appointment"), "The receptionist tells the caller who booked it what it is for, the Agent reads its name and notes, and the phone sends it to your linked FormLogic account.")
            } else {
                (format!("Service {}", at + 1), Parts::new("calendar-service"), "The receptionist reads it before it answers and says it to callers.")
            };
            let mut parts = parts.fixed("about", what);
            for k in says {
                parts = parts.fixed(&k.path, format!("{}: {}.", k.row.what, show_value(&k.value)));
            }
            items.push(parts.item(class, &format!("{name}#{list}[{at}]"), &title));
        }
        if words_of > MAX_CALENDAR_LISTED {
            let more = words_of - MAX_CALENDAR_LISTED;
            let noun = if list == "appointments" { "appointments" } else { "services" };
            items.push(
                Parts::new("more")
                    .fixed("count", format!("{more} more of the {total} {noun} have words in them that are read in the same way; they come back with the same tick."))
                    .item(class, &format!("{name}#{list}"), &format!("and {more} more {noun}")),
            );
        }
    }
    // (A calendar with no words in it, only hours and times, has nothing to tick: it says nothing here.)
    items
}

/// A voice file, by its name and size: it is audio, and is never read.
pub fn describe_voice(class: RestoreClass, name: &str, size: u64) -> ReviewItem {
    Parts::new("voice").fixed("file", format!("A voice file ({} KB): what your callers hear.", size.div_ceil(1024))).item(class, name, name.rsplit('/').next().unwrap_or(name))
}

/// A value as a person is shown it: text is cut, with how long it is.
pub fn show_value(value: &Value) -> String {
    match value {
        // What a person reads: the characters they cannot see are made visible before the length is counted.
        Value::String(text) => quoted(text, 160),
        Value::Array(items) if items.is_empty() => "an empty list".to_string(),
        Value::Array(items) if items.iter().all(Value::is_string) => {
            let shown: Vec<String> = items.iter().take(5).filter_map(Value::as_str).map(|s| clip(s, 40)).collect();
            let more = items.len().saturating_sub(5);
            format!("{}{}", shown.join(", "), if more > 0 { format!(" and {more} more") } else { String::new() })
        }
        Value::Array(items) => format!("a list of {}", items.len()),
        Value::Object(map) => format!("{} entr{}", map.len(), if map.len() == 1 { "y" } else { "ies" }),
        other => other.to_string(),
    }
}

/// What one key that acts is, for the dry run: by its key, its name and its value.
pub fn key_item(file: &str, kept: &super::table::Kept) -> ReviewItem {
    let class = kept.row.tick.unwrap_or(RestoreClass::Settings);
    // The value is the fixed part; why it matters is a sample after it, each cut on its own.
    let mut parts = Parts::new("setting").fixed("sets", format!("Sets {} to {}.", kept.path, show_value(&kept.value)));
    if !kept.row.reason.is_empty() {
        parts = parts.sample("why", &kept.row.reason);
    }
    parts.item(class, &format!("{file}#{}", kept.path), &kept.row.what)
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
                    (Why::KeyWithoutAddress, _) => format!("not restored: {}", super::table::KEY_WITHOUT_ADDRESS),
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
