//! Plugin manifest: parse, validate, and resolve the capability surface.
//!
//! A plugin is a directory under `<data_dir>/plugins/<id>/` holding a
//! `manifest.json` and an executable. The manifest is the plugin's entire
//! declared permission surface — nothing it did not declare can be invoked.
//!
//! # Invalid manifests are loud
//!
//! A malformed manifest yields [`ManifestError`], which the registry surfaces as
//! `disabled` **with the reason attached**. It is never a silent skip. A plugin
//! that quietly fails to appear is indistinguishable from one that was never
//! installed, and the user's next move — reinstalling software that is already
//! there — does not help.
//!
//! # Wildcards are expanded here, not matched at call time
//!
//! `protocol/README.md` requires it: *"Wildcards are expanded at grant time,
//! never matched at call time."* So `connector.aokie.*` is resolved against the
//! connector's **declared command list** when the manifest loads, producing an
//! explicit set of exact capability strings.
//!
//! Matching the pattern per-call instead looks equivalent and is not. It means
//! the grant silently widens every time the plugin adds a command: a user who
//! approved `connector.aokie.*` when it meant six read-only calls has, after an
//! update, approved `call.dial` and `sms.send` without being asked. Expanding at
//! load time makes the grant a fixed list you can show someone.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Plugin API protocol versions this host implements.
///
/// A plugin declaring anything outside this range is refused rather than
/// attempted — a partially-understood plugin is worse than a disabled one,
/// because it runs.
pub const SUPPORTED_PLUGIN_API: std::ops::RangeInclusive<u32> = 1..=1;

/// Manifest `schemaVersion` values this host can read.
///
/// Aokie ships `3`. Older first-party manifests exist at `1` and `2`, and the
/// differences are additive, so all of them parse. `4` adds `modules`,
/// `agentTools` and `setup` (see [`V4_SECTIONS`]).
pub const SUPPORTED_SCHEMA_VERSIONS: std::ops::RangeInclusive<u32> = 1..=4;

/// The sections schemaVersion 4 adds. Under an older schemaVersion each is
/// refused, not half-honoured: a host that predates the section would ignore
/// it, and a plugin that believes it declared something it did not is worse
/// than one that fails to load.
pub const V4_SECTIONS: &[&str] = &["modules", "agentTools", "setup"];

/// This desktop's version, which a manifest's `minDesktopVersion` is held to.
pub const DESKTOP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where an agent tool may be offered: the project conversation, the Front
/// desk's runner, or one of the Front desk's own conversations.
pub const AGENT_AUDIENCES: &[&str] = &["project", "runner", "session:sms", "session:call", "session:task"];

/// The audience of an agent tool that does not name one.
pub const DEFAULT_AGENT_AUDIENCE: &str = "project";

/// The host's own setup steps a plugin's `setup` may use (`kind: "host"`).
/// Another action is left out with a warning, so a newer plugin still loads.
pub const HOST_SETUP_ACTIONS: &[&str] = &["phone.answerWithOaiy", "calendar.business"];

/// Cap on the declared capability surface, after wildcard expansion. A plugin
/// asking for thousands of capabilities is either broken or hostile, and an
/// unbounded list is a UI that cannot be reviewed by a person.
const MAX_CAPABILITIES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    Unreadable(String),
    Malformed(String),
    /// Present and parseable, but this host cannot honour it.
    Unsupported(String),
    Invalid(String),
}

impl ManifestError {
    /// The sentence shown next to a `disabled` plugin. Must say what is wrong
    /// well enough to act on.
    pub fn reason(&self) -> String {
        match self {
            ManifestError::Unreadable(m) => format!("manifest.json could not be read: {m}"),
            ManifestError::Malformed(m) => format!("manifest.json is not valid JSON: {m}"),
            ManifestError::Unsupported(m) => format!("unsupported by this version of OAIY Desktop: {m}"),
            ManifestError::Invalid(m) => format!("manifest.json is invalid: {m}"),
        }
    }
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason())
    }
}

/// How to launch the plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginEntry {
    /// Only `process` today. An unknown kind is refused rather than assumed.
    pub kind: String,
    /// Executable path, **relative to the plugin directory**. Absolute paths and
    /// `..` traversal are refused — see [`PluginManifest::resolve_entry`].
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorDecl {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Exact command names. This list is the allow-list the gateway checks
    /// against, and the expansion source for `connector.<id>.*`.
    #[serde(default)]
    pub commands: Vec<String>,
}

/// Commands with side effects a retry must not repeat.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandsDecl {
    /// The gateway requires an `idempotencyKey` for these. Sending an SMS twice
    /// because a socket blipped is not recoverable by apologising.
    #[serde(default)]
    pub journalled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub publisher: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    pub plugin_api_version: u32,
    #[serde(default)]
    pub min_desktop_version: Option<String>,
    pub entry: PluginEntry,
    /// Declared permission surface, wildcards allowed. Resolve with
    /// [`PluginManifest::resolved_capabilities`] before enforcing anything.
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub connectors: Vec<ConnectorDecl>,
    /// Event names the plugin may emit. An event it did not declare is dropped.
    #[serde(default)]
    pub events: Vec<String>,
    #[serde(default)]
    pub commands: Option<CommandsDecl>,
    /// schemaVersion 4: the built-in modules this plugin provides (the phone,
    /// the calendar). Read by `modules::claims`; without it, the plugin id
    /// `aokie` provides both (the legacy rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modules: Option<ModulesDecl>,
    /// schemaVersion 4: the plugin's own service-definition actions offered
    /// to the agent as tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty", deserialize_with = "de_agent_tools")]
    pub agent_tools: Vec<AgentToolDecl>,
    /// schemaVersion 4: the plugin's setup wizard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<SetupDecl>,
    /// Unknown keys (`ui`, `data`, `serviceDefinitions`, …) are retained rather
    /// than rejected: the manifest schema is additive within a major, so a
    /// newer plugin must not be refused for carrying a field this host has no
    /// opinion about.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// What loading it found worth saying without refusing it (a setup step
    /// this host cannot run, an unreadable `minDesktopVersion`). Shown in
    /// `/api/modules` `warnings`; never read from the file.
    #[serde(skip)]
    pub warnings: Vec<String>,
    /// `agentTools`, each resolved against the service-definition action it
    /// names (the name, description and input schema the agent sees). Worked
    /// out at load, so the modules snapshot never reads the disk.
    #[serde(skip)]
    pub resolved_agent_tools: Vec<AgentTool>,
}

fn default_schema_version() -> u32 {
    1
}

// ---- schemaVersion 4: modules ------------------------------------------------------

/// `modules`: `{"provides": ["phone", "calendar"], "connector": "aokie"}`, or
/// just the list `["phone", "calendar"]`. Written back as the object form.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModulesDecl {
    pub provides: Vec<String>,
    /// The connector that serves the phone (optional: otherwise the first
    /// connector declaring every command the phone uses).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector: Option<String>,
}

impl<'de> Deserialize<'de> for ModulesDecl {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        const SHAPE: &str = "modules must be a list of module ids, or {\"provides\": [...]}";
        let ids = |list: &[Value]| -> Result<Vec<String>, D::Error> {
            let mut out: Vec<String> = Vec::new();
            for item in list {
                let id = item.as_str().ok_or_else(|| D::Error::custom(format!("{SHAPE}: {item} is not a module id")))?;
                let id = id.trim().to_string();
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            Ok(out)
        };
        match Value::deserialize(d)? {
            Value::Array(list) => Ok(Self { provides: ids(&list)?, connector: None }),
            Value::Object(o) => {
                let provides = match o.get("provides") {
                    Some(Value::Array(list)) => ids(list)?,
                    _ => return Err(D::Error::custom(format!("{SHAPE}: \"provides\" is missing or not a list"))),
                };
                let connector = match o.get("connector") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(c)) => Some(c.trim().to_string()),
                    Some(_) => return Err(D::Error::custom("modules.connector must be a connector id")),
                };
                Ok(Self { provides, connector })
            }
            _ => Err(D::Error::custom(SHAPE)),
        }
    }
}

// ---- schemaVersion 4: agent tools ----------------------------------------------------

/// One `agentTools[]` entry: a service-definition action of the plugin's own,
/// offered to the agent under `name`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolDecl {
    /// `<service definition id>/<action id>`, e.g. `aokie.phone/sms.threads`.
    pub action: String,
    /// What the agent calls it: `^[a-z][a-z0-9_]{2,47}$`, unique in the plugin.
    pub name: String,
    /// Instead of the action's own description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Where it is offered (a subset of [`AGENT_AUDIENCES`]); absent: `["project"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience: Option<Vec<String>>,
    /// What the person is asked before it runs (`Call {number}`). Required
    /// when the action's `sideEffects` is not `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

impl AgentToolDecl {
    /// The audience it names, or the default.
    pub fn audience(&self) -> Vec<String> {
        self.audience.clone().unwrap_or_else(|| vec![DEFAULT_AGENT_AUDIENCE.to_string()])
    }
}

fn de_agent_tools<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<AgentToolDecl>, D::Error> {
    let list = match Value::deserialize(d)? {
        Value::Array(list) => list,
        Value::Null => return Ok(Vec::new()),
        _ => return Err(D::Error::custom("agentTools must be a list")),
    };
    list.into_iter()
        .enumerate()
        .map(|(i, v)| serde_json::from_value(v).map_err(|e| D::Error::custom(format!("agentTools[{i}]: {e}"))))
        .collect()
}

/// An agent tool as the agent sees it: an `agentTools[]` entry with the
/// name, description and input schema of the action it names.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTool {
    pub plugin_id: String,
    pub name: String,
    /// `<definition>/<actionId>`, as declared.
    pub action: String,
    /// The service definition's id (`aokie.phone`): the invoke route's first part.
    pub definition: String,
    /// The action's id (`sms.threads`): the invoke route's second part.
    pub action_id: String,
    pub description: String,
    pub input_schema: Value,
    /// The action's, as its definition says (absent: not said, so treated as having some).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_effects: Option<String>,
    pub audience: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl AgentTool {
    /// Does running it change anything (its action's `sideEffects` is not `none`)?
    pub fn has_side_effects(&self) -> bool {
        self.side_effects.as_deref() != Some("none")
    }
}

/// `^[a-z][a-z0-9_]{2,47}$`
fn is_valid_tool_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && (3..=48).contains(&s.len())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

// ---- schemaVersion 4: setup -----------------------------------------------------------

/// `setup`: the plugin's own setup wizard, run by the host.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupDecl {
    /// Moves when the plugin's setup changes enough to be run again (default 1).
    pub version: u32,
    /// The wizard's title (default: "Set up <plugin name>").
    pub title: String,
    pub steps: Vec<SetupStep>,
}

impl<'de> Deserialize<'de> for SetupDecl {
    /// Each step is read on its own, so a refusal says which one.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default = "default_setup_version")]
            version: u32,
            #[serde(default)]
            title: String,
            steps: Vec<Value>,
        }
        let raw = Raw::deserialize(d).map_err(|e| D::Error::custom(format!("setup: {e}")))?;
        let steps = raw
            .steps
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let id = v.get("id").and_then(Value::as_str).map(|s| format!(" ({s:?})")).unwrap_or_default();
                serde_json::from_value(v).map_err(|e| D::Error::custom(format!("setup.steps[{i}]{id}: {e}")))
            })
            .collect::<Result<Vec<SetupStep>, D::Error>>()?;
        Ok(Self { version: raw.version, title: raw.title, steps })
    }
}

fn default_setup_version() -> u32 {
    1
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// One step of a plugin's setup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupStep {
    /// Unique in the setup: `^[a-z][a-z0-9-]{0,39}$`.
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The person may skip it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub optional: bool,
    /// The step is shown only while this check passes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Check>,
    #[serde(flatten)]
    pub kind: StepKind,
}

/// What a setup step does (`kind`), and what it needs for that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum StepKind {
    /// The capabilities the person grants. The host always shows it first,
    /// declared or not.
    Permissions {},
    /// Services and engine models the plugin needs on this computer.
    Requirements { requires: Vec<Requirement> },
    /// A few of the plugin's settings, read and written through its connector.
    Settings {
        fields: Vec<SettingField>,
        /// Default `{"command": "settings.get", "path": "settings"}`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read: Option<SettingsRead>,
        /// Default `{"command": "settings.set"}`, with `{"<key>": value, ...}`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        write: Option<SettingsWrite>,
    },
    /// One of the plugin's `ui.screens`, in setup mode, showing `view`.
    Screen {
        screen: String,
        view: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        done: Option<Check>,
    },
    /// One of the host's own steps ([`HOST_SETUP_ACTIONS`]).
    Host { action: String },
}

impl StepKind {
    pub fn name(&self) -> &'static str {
        match self {
            StepKind::Permissions {} => "permissions",
            StepKind::Requirements { .. } => "requirements",
            StepKind::Settings { .. } => "settings",
            StepKind::Screen { .. } => "screen",
            StepKind::Host { .. } => "host",
        }
    }
}

/// Something a `requirements` step needs. There is no model id: an engine
/// model is met by whatever the person chose in Engines for that group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Requirement {
    Service {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        why: Option<String>,
    },
    EngineModel {
        group: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        why: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Bool,
    Choice,
    Text,
    Number,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingField {
    pub key: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind: FieldType,
    /// A choice's options (and only a choice's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<ChoiceOption>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChoiceOption {
    pub value: Value,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsRead {
    pub command: String,
    /// Where the settings are in the answer (absent: the whole answer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsWrite {
    pub command: String,
}

/// Present, even as `null` (a plain `Option<Value>` reads `null` as absent,
/// and `"equals": null` is a real test).
fn some_value<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

/// One test of a command's answer: a `path` and exactly one operator.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    /// Dot-separated. A missing path equals null.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub path: String,
    #[serde(default, deserialize_with = "some_value", skip_serializing_if = "Option::is_none")]
    pub equals: Option<Value>,
    /// The path exists and is not null (`true`), or the reverse (`false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub present: Option<bool>,
    #[serde(default, rename = "in", skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_in: Option<Vec<Value>>,
}

impl Condition {
    fn operators(&self) -> usize {
        [self.equals.is_some(), self.present.is_some(), self.one_of.is_some(), self.not_in.is_some()]
            .iter()
            .filter(|x| **x)
            .count()
    }

    fn validate(&self, what: &str) -> Result<(), String> {
        if !is_valid_path(&self.path) {
            return Err(format!("{what} needs a dot-separated path, not {:?}", self.path));
        }
        match self.operators() {
            1 => Ok(()),
            0 => Err(format!("{what} has no test: give one of equals, present, in or notIn")),
            _ => Err(format!("{what} has more than one test: give exactly one of equals, present, in or notIn")),
        }
    }
}

/// A check (a step's `done` or `when`): a read-only command of the plugin's,
/// sent with no payload, and a test of its answer. Either one condition
/// inline, or `all` of a list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub command: String,
    #[serde(flatten)]
    pub condition: Condition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all: Option<Vec<Condition>>,
}

fn is_valid_path(s: &str) -> bool {
    !s.is_empty() && s.len() <= 256 && s.split('.').all(|seg| !seg.is_empty())
}

/// `^[a-z][a-z0-9-]{0,39}$`
fn is_valid_step_id(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && s.len() <= 40
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

// ---- minDesktopVersion -----------------------------------------------------------------

/// A semantic version: `major.minor.patch`, an optional `-pre.release`, and
/// build metadata (`+...`, ignored). `None` when `s` is not one.
fn parse_semver(s: &str) -> Option<(u64, u64, u64, Vec<String>)> {
    let s = s.trim();
    let s = s.split_once('+').map_or(s, |(v, _)| v);
    let (core, pre) = match s.split_once('-') {
        Some((core, pre)) => (core, Some(pre)),
        None => (s, None),
    };
    let nums: Vec<&str> = core.split('.').collect();
    if nums.len() != 3 || nums.iter().any(|n| n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let pre: Vec<String> = match pre {
        None => Vec::new(),
        Some(p) => {
            let ids: Vec<String> = p.split('.').map(str::to_string).collect();
            if ids.iter().any(|x| x.is_empty() || !x.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')) {
                return None;
            }
            ids
        }
    };
    Some((nums[0].parse().ok()?, nums[1].parse().ok()?, nums[2].parse().ok()?, pre))
}

/// Semver order: the numbers, then a pre-release below its release, its
/// identifiers compared numerically when both are numbers.
fn semver_cmp(a: &(u64, u64, u64, Vec<String>), b: &(u64, u64, u64, Vec<String>)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let by_numbers = (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2));
    if by_numbers != Ordering::Equal {
        return by_numbers;
    }
    match (a.3.is_empty(), b.3.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }
    for (x, y) in a.3.iter().zip(b.3.iter()) {
        let o = match (x.parse::<u64>(), y.parse::<u64>()) {
            (Ok(x), Ok(y)) => x.cmp(&y),
            (Ok(_), Err(_)) => Ordering::Less,
            (Err(_), Ok(_)) => Ordering::Greater,
            (Err(_), Err(_)) => x.cmp(y),
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    a.3.len().cmp(&b.3.len())
}

/// The most a `manifest.json` may hold. It is a hand-written description of a plugin,
/// a few kilobytes; a file this large is not one, and it is read whole into memory
/// (and hashed) by the package check, so it is bounded.
pub const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;

/// `manifest.json`, whole, at most [`MAX_MANIFEST_BYTES`].
fn read_manifest_bytes(path: &Path) -> Result<Vec<u8>, ManifestError> {
    use std::io::Read as _;
    let file = std::fs::File::open(path).map_err(|e| ManifestError::Unreadable(e.to_string()))?;
    let mut raw = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|e| ManifestError::Unreadable(e.to_string()))?;
    if raw.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(ManifestError::Invalid(format!(
            "it is larger than {} MiB, which no manifest is",
            MAX_MANIFEST_BYTES / (1024 * 1024)
        )));
    }
    Ok(raw)
}

/// Is `wanted` (a manifest's `minDesktopVersion`) newer than `have`?
/// `Err` when either is not a semantic version.
pub fn needs_newer_desktop(wanted: &str, have: &str) -> Result<bool, String> {
    let w = parse_semver(wanted).ok_or_else(|| format!("{wanted:?} is not a version like 1.2.3"))?;
    let h = parse_semver(have).ok_or_else(|| format!("this desktop's version {have:?} is not a version like 1.2.3"))?;
    Ok(semver_cmp(&w, &h) == std::cmp::Ordering::Greater)
}

impl PluginManifest {
    /// Read and validate `<dir>/manifest.json`.
    pub fn load(dir: &Path) -> Result<Self, ManifestError> {
        Self::read(dir).map(|(manifest, _)| manifest)
    }

    /// Read and validate `<dir>/manifest.json`, and give back the bytes that were parsed.
    ///
    /// For a caller that must know the manifest it holds is the one a package check
    /// verified (see [`super::trust`]): it compares those bytes, instead of reading the
    /// file a second time and trusting that it did not change in between.
    pub fn read(dir: &Path) -> Result<(Self, Vec<u8>), ManifestError> {
        let raw = read_manifest_bytes(&dir.join("manifest.json"))?;
        let manifest = Self::parse(&raw, dir)?;
        Ok((manifest, raw))
    }

    /// Validate the bytes of a `manifest.json` that has already been read. `dir` is the
    /// plugin's folder, for what the manifest points at (its entry, its service
    /// definitions).
    pub fn parse(raw: &[u8], dir: &Path) -> Result<Self, ManifestError> {
        let raw = std::str::from_utf8(raw)
            .map_err(|_| ManifestError::Unreadable("stream did not contain valid UTF-8".into()))?;
        let value: Value = serde_json::from_str(raw).map_err(|e| ManifestError::Malformed(e.to_string()))?;
        // Before the sections are read: under an older schemaVersion the
        // reason is the version, even when the section is malformed too.
        refuse_newer_sections(&value)?;
        let mut manifest: PluginManifest =
            serde_json::from_value(value).map_err(|e| ManifestError::Invalid(e.to_string()))?;
        manifest.validate(dir)?;
        Ok(manifest)
    }

    fn validate(&mut self, dir: &Path) -> Result<(), ManifestError> {
        if !SUPPORTED_SCHEMA_VERSIONS.contains(&self.schema_version) {
            return Err(ManifestError::Unsupported(format!(
                "manifest schemaVersion {} (this host reads {}-{})",
                self.schema_version,
                SUPPORTED_SCHEMA_VERSIONS.start(),
                SUPPORTED_SCHEMA_VERSIONS.end()
            )));
        }
        self.warnings.clear();
        if let Some(wanted) = self.min_desktop_version.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            match needs_newer_desktop(wanted, DESKTOP_VERSION) {
                Ok(true) => {
                    return Err(ManifestError::Unsupported(format!(
                        "{} needs OAIY Desktop {wanted} or later (this is {DESKTOP_VERSION})",
                        self.name
                    )))
                }
                Ok(false) => {}
                Err(why) => self.warnings.push(format!("minDesktopVersion {why}, so it is not checked.")),
            }
        }
        if !SUPPORTED_PLUGIN_API.contains(&self.plugin_api_version) {
            return Err(ManifestError::Unsupported(format!(
                "pluginApiVersion {} (this host speaks {}-{})",
                self.plugin_api_version,
                SUPPORTED_PLUGIN_API.start(),
                SUPPORTED_PLUGIN_API.end()
            )));
        }

        if !is_valid_id(&self.id) {
            return Err(ManifestError::Invalid(format!(
                "id {:?} must be lowercase letters, digits and hyphens, starting with a letter",
                self.id
            )));
        }
        if self.name.trim().is_empty() {
            return Err(ManifestError::Invalid("name is empty".into()));
        }
        if self.entry.kind != "process" {
            return Err(ManifestError::Unsupported(format!(
                "entry.kind {:?} (only \"process\" is supported)",
                self.entry.kind
            )));
        }
        // The entry path is attacker-controlled data in a file we found on disk,
        // so it is validated as a path, not merely as a string.
        self.resolve_entry(dir)?;
        super::definitions::validate_for_plugin(dir, self).map_err(ManifestError::Invalid)?;

        for c in &self.connectors {
            if !is_valid_id(&c.id) {
                return Err(ManifestError::Invalid(format!(
                    "connector id {:?} must be lowercase letters, digits and hyphens",
                    c.id
                )));
            }
            for cmd in &c.commands {
                if !is_valid_command(cmd) {
                    return Err(ManifestError::Invalid(format!(
                        "connector {} declares an invalid command {cmd:?} (expected dot-namespaced, e.g. \"call.answer\")",
                        c.id
                    )));
                }
            }
        }

        // Every declared event must be namespaced under something this plugin
        // OWNS — its own id or one of its connector ids. Without this, the
        // `source` pin in the runtime is cosmetic: the trigger dispatcher matches
        // bindings on NAME alone, so a plugin declaring `aokie.call.incoming` in
        // its own manifest (source = itself, which passes the runtime check)
        // fires every binding meant for the real Aokie — forging another
        // subsystem's events with zero declared capabilities. Tying the event
        // namespace to an owned prefix is what makes the source pin enforceable.
        let owned_prefixes: BTreeSet<&str> = std::iter::once(self.id.as_str())
            .chain(self.connectors.iter().map(|c| c.id.as_str()))
            .collect();
        for e in &self.events {
            if !is_valid_event_name(e) {
                return Err(ManifestError::Invalid(format!(
                    "event name {e:?} must be dot-namespaced, e.g. \"aokie.call.incoming\""
                )));
            }
            let first = e.split('.').next().unwrap_or("");
            if !owned_prefixes.contains(first) {
                return Err(ManifestError::Invalid(format!(
                    "event {e:?} is namespaced under {first:?}, which this plugin does not own; \
                     declare events under the plugin id ({:?}) or a declared connector id",
                    self.id
                )));
            }
        }

        // A journalled command that is not declared is a manifest bug that
        // silently disables the idempotency requirement — exactly the direction
        // you do not want a mistake to fail in.
        if let Some(cmds) = &self.commands {
            let declared: BTreeSet<&str> = self
                .connectors
                .iter()
                .flat_map(|c| c.commands.iter().map(String::as_str))
                .collect();
            for j in &cmds.journalled {
                if !declared.contains(j.as_str()) {
                    return Err(ManifestError::Invalid(format!(
                        "commands.journalled lists {j:?}, which no connector declares"
                    )));
                }
            }
        }

        let resolved = self.resolved_capabilities();
        if resolved.len() > MAX_CAPABILITIES {
            return Err(ManifestError::Invalid(format!(
                "declares {} capabilities after wildcard expansion (max {MAX_CAPABILITIES})",
                resolved.len()
            )));
        }

        // schemaVersion 4's sections (refused under an older one before this).
        self.validate_modules()?;
        self.resolved_agent_tools = if self.agent_tools.is_empty() {
            Vec::new()
        } else {
            let definitions = super::definitions::load_for_plugin(dir, self);
            self.resolve_agent_tools(&definitions).map_err(ManifestError::Invalid)?
        };
        self.validate_setup()?;

        Ok(())
    }

    /// The connector declaring `command`, if one does.
    pub fn connector_for(&self, command: &str) -> Option<&ConnectorDecl> {
        self.connectors.iter().find(|c| c.commands.iter().any(|x| x == command))
    }

    /// The ids of the plugin's `ui.screens`.
    pub fn ui_screen_ids(&self) -> BTreeSet<String> {
        self.extra
            .get("ui")
            .and_then(|u| u.get("screens"))
            .and_then(Value::as_array)
            .map(|list| list.iter().filter_map(|s| s.get("id").and_then(Value::as_str)).map(str::to_string).collect())
            .unwrap_or_default()
    }

    /// A command only read from (a check's, or a settings step's `read`):
    /// declared by one of the plugin's connectors, and not journalled.
    fn read_only_command(&self, command: &str, what: &str) -> Result<(), String> {
        if self.connector_for(command).is_none() {
            return Err(format!("{what} sends {command:?}, which no connector of this plugin declares"));
        }
        if self.is_journalled(command) {
            return Err(format!(
                "{what} sends {command:?}, which is journalled (it changes something); it may only read"
            ));
        }
        Ok(())
    }

    fn validate_check(&self, check: &Check, what: &str) -> Result<(), String> {
        self.read_only_command(&check.command, what)?;
        match &check.all {
            Some(list) => {
                if !check.condition.path.is_empty() || check.condition.operators() > 0 {
                    return Err(format!("{what} gives a condition and \"all\": give one or the other"));
                }
                if list.is_empty() {
                    return Err(format!("{what} has an empty \"all\""));
                }
                for (i, c) in list.iter().enumerate() {
                    c.validate(&format!("{what}.all[{i}]"))?;
                }
                Ok(())
            }
            None => check.condition.validate(what),
        }
    }

    /// `modules`: known modules only, and a connector the plugin declares.
    fn validate_modules(&self) -> Result<(), ManifestError> {
        let Some(decl) = &self.modules else { return Ok(()) };
        for id in &decl.provides {
            if crate::modules::def(id).is_none() {
                let known: Vec<&str> = crate::modules::BUILTIN.iter().map(|d| d.id).collect();
                return Err(ManifestError::Invalid(format!(
                    "modules names {id:?}, which is not a module this OAIY knows ({})",
                    known.join(", ")
                )));
            }
        }
        if let Some(c) = &decl.connector {
            if !self.connectors.iter().any(|x| &x.id == c) {
                return Err(ManifestError::Invalid(format!(
                    "modules.connector names {c:?}, which is not one of this plugin's connectors"
                )));
            }
        }
        Ok(())
    }

    /// `agentTools`, each resolved against the action it names in `definitions`
    /// (the plugin's own service definitions).
    pub fn resolve_agent_tools(
        &self,
        definitions: &[super::definitions::ServiceDefinition],
    ) -> Result<Vec<AgentTool>, String> {
        let mut out = Vec::new();
        let mut names = BTreeSet::new();
        for (i, t) in self.agent_tools.iter().enumerate() {
            if !is_valid_tool_name(&t.name) {
                return Err(format!(
                    "agentTools[{i}]: the name {:?} must be 3 to 48 lowercase letters, digits and underscores, starting with a letter",
                    t.name
                ));
            }
            let what = format!("agentTools[{i}] ({:?})", t.name);
            if !names.insert(t.name.as_str()) {
                return Err(format!("{what}: the name is used twice"));
            }
            let Some((def_id, action_id)) = t.action.split_once('/').filter(|(d, a)| !d.is_empty() && !a.is_empty()) else {
                return Err(format!("{what}: the action {:?} must be \"<service definition id>/<action id>\"", t.action));
            };
            let Some(def) = definitions.iter().find(|d| d.id == def_id) else {
                return Err(format!("{what}: {def_id:?} is not one of this plugin's service definitions"));
            };
            let Some(action) = def.action(action_id) else {
                return Err(format!("{what}: the service definition {def_id:?} has no action {action_id:?}"));
            };
            let declared = t.audience();
            if declared.is_empty() {
                return Err(format!("{what}: the audience is empty (leave it out for [\"project\"])"));
            }
            let mut audience: Vec<String> = Vec::new();
            for a in declared {
                if !AGENT_AUDIENCES.contains(&a.as_str()) {
                    return Err(format!("{what}: the audience {a:?} is not one of {}", AGENT_AUDIENCES.join(", ")));
                }
                if !audience.contains(&a) {
                    audience.push(a);
                }
            }
            let confirm = t.confirm.as_deref().map(str::trim).filter(|c| !c.is_empty()).map(str::to_string);
            let effects = action.side_effects.clone();
            if effects.as_deref() != Some("none") && confirm.is_none() {
                let why = match effects.as_deref() {
                    Some(e) => format!("has side effects ({e})"),
                    None => "does not say it has no side effects".to_string(),
                };
                return Err(format!(
                    "{what}: {} {why}, so the tool needs a \"confirm\" template (what the person is asked before it runs)",
                    t.action
                ));
            }
            let text = |s: Option<&str>| s.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
            let description = text(t.description.as_deref())
                .or_else(|| text(action.description.as_deref()))
                .or_else(|| text(action.title.as_deref()))
                .unwrap_or_else(|| action_id.to_string());
            let input_schema = match &action.input_schema {
                Some(schema @ Value::Object(_)) => schema.clone(),
                _ => serde_json::json!({ "type": "object" }),
            };
            out.push(AgentTool {
                plugin_id: self.id.clone(),
                name: t.name.clone(),
                action: t.action.clone(),
                definition: def_id.to_string(),
                action_id: action_id.to_string(),
                description,
                input_schema,
                side_effects: effects,
                audience,
                confirm,
                timeout_ms: action.timeout_ms,
            });
        }
        Ok(out)
    }

    /// `setup`: every refusal in the pinned contract; a host step this OAIY
    /// cannot run is left out with a warning.
    fn validate_setup(&mut self) -> Result<(), ManifestError> {
        let Some(mut setup) = self.setup.take() else { return Ok(()) };
        let result = self.check_setup(&mut setup);
        self.setup = Some(setup);
        result.map_err(ManifestError::Invalid)
    }

    fn check_setup(&mut self, setup: &mut SetupDecl) -> Result<(), String> {
        if setup.version == 0 {
            return Err("setup.version must be 1 or more".into());
        }
        if setup.title.trim().is_empty() {
            setup.title = format!("Set up {}", self.name.trim());
        }
        let screens = self.ui_screen_ids();
        let mut ids = BTreeSet::new();
        let mut kept = Vec::with_capacity(setup.steps.len());
        for (i, step) in std::mem::take(&mut setup.steps).into_iter().enumerate() {
            let what = format!("setup.steps[{i}] ({:?})", step.id);
            if !is_valid_step_id(&step.id) {
                return Err(format!(
                    "{what}: the id must be up to 40 lowercase letters, digits and hyphens, starting with a letter"
                ));
            }
            if !ids.insert(step.id.clone()) {
                return Err(format!("{what}: the id is used twice"));
            }
            if step.title.trim().is_empty() {
                return Err(format!("{what}: the title is empty"));
            }
            if let Some(when) = &step.when {
                self.validate_check(when, &format!("{what}.when"))?;
            }
            match &step.kind {
                StepKind::Permissions {} => {}
                StepKind::Requirements { requires } => {
                    if requires.is_empty() {
                        return Err(format!("{what}: a requirements step with nothing in requires"));
                    }
                    for (j, r) in requires.iter().enumerate() {
                        match r {
                            Requirement::Service { id, .. } if id.trim().is_empty() => {
                                return Err(format!("{what}.requires[{j}]: the service id is empty"))
                            }
                            Requirement::EngineModel { group, .. } if group.trim().is_empty() => {
                                return Err(format!("{what}.requires[{j}]: the engine group is empty"))
                            }
                            _ => {}
                        }
                    }
                }
                StepKind::Settings { fields, read, write } => {
                    if fields.is_empty() {
                        return Err(format!("{what}: a settings step with no fields"));
                    }
                    let mut keys = BTreeSet::new();
                    for (j, f) in fields.iter().enumerate() {
                        let field = format!("{what}.fields[{j}]");
                        if f.key.trim().is_empty() {
                            return Err(format!("{field}: the key is empty"));
                        }
                        if !keys.insert(f.key.as_str()) {
                            return Err(format!("{field}: the key {:?} is used twice", f.key));
                        }
                        if f.label.trim().is_empty() {
                            return Err(format!("{field}: the label is empty"));
                        }
                        match (f.kind, &f.options) {
                            (FieldType::Choice, Some(options)) if !options.is_empty() => {}
                            (FieldType::Choice, _) => return Err(format!("{field}: a choice needs options")),
                            (_, Some(_)) => return Err(format!("{field}: options are only for a choice")),
                            _ => {}
                        }
                    }
                    let read_command = read.as_ref().map_or("settings.get", |r| r.command.as_str());
                    self.read_only_command(read_command, &format!("{what}.read"))?;
                    if let Some(path) = read.as_ref().and_then(|r| r.path.as_deref()) {
                        if !is_valid_path(path) {
                            return Err(format!("{what}.read: the path {path:?} must be dot-separated"));
                        }
                    }
                    let write_command = write.as_ref().map_or("settings.set", |w| w.command.as_str());
                    if self.connector_for(write_command).is_none() {
                        return Err(format!(
                            "{what}.write sends {write_command:?}, which no connector of this plugin declares"
                        ));
                    }
                }
                StepKind::Screen { screen, view, done } => {
                    if !screens.contains(screen) {
                        return Err(format!("{what}: it shows the screen {screen:?}, which is not in ui.screens"));
                    }
                    if view.trim().is_empty() {
                        return Err(format!("{what}: the view is empty"));
                    }
                    if let Some(done) = done {
                        self.validate_check(done, &format!("{what}.done"))?;
                    }
                }
                StepKind::Host { action } => {
                    if !HOST_SETUP_ACTIONS.contains(&action.as_str()) {
                        self.warnings.push(format!(
                            "{what}: the host step {action:?} needs a newer OAIY, so it is left out."
                        ));
                        continue;
                    }
                }
            }
            kept.push(step);
        }
        setup.steps = kept;
        Ok(())
    }

    /// Absolute path to the executable, confined to the plugin directory.
    ///
    /// Refuses absolute paths and any `..` component. Without this, a manifest
    /// could name `../../../../Windows/System32/cmd.exe`, or an absolute path to
    /// anything on the box, and the host would dutifully launch it — turning
    /// "drop a folder in the plugins dir" into arbitrary code execution with a
    /// plausible-looking manifest as cover.
    pub fn resolve_entry(&self, dir: &Path) -> Result<PathBuf, ManifestError> {
        let raw = self.entry.command.trim();
        if raw.is_empty() {
            return Err(ManifestError::Invalid("entry.command is empty".into()));
        }
        let candidate = Path::new(raw);
        if candidate.is_absolute() {
            return Err(ManifestError::Invalid(format!(
                "entry.command {raw:?} must be relative to the plugin directory"
            )));
        }
        for part in candidate.components() {
            match part {
                std::path::Component::Normal(_) => {}
                std::path::Component::CurDir => {}
                other => {
                    return Err(ManifestError::Invalid(format!(
                        "entry.command {raw:?} contains a disallowed path component {other:?}"
                    )));
                }
            }
        }
        // Windows accepts both separators; a manifest using '\' must not slip a
        // traversal past the component walk above.
        if raw.split(['/', '\\']).any(|seg| seg == "..") {
            return Err(ManifestError::Invalid(format!(
                "entry.command {raw:?} must not traverse outside the plugin directory"
            )));
        }
        Ok(dir.join(candidate))
    }

    /// The exact capability strings this plugin is granted.
    ///
    /// Wildcards are expanded against declared commands **here**, so the result
    /// is a fixed list. An unexpandable wildcard (a pattern matching no declared
    /// command) is dropped rather than kept as a pattern — keeping it would
    /// reintroduce call-time matching through the back door.
    pub fn resolved_capabilities(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for cap in &self.capabilities {
            let cap = cap.trim();
            if cap.is_empty() {
                continue;
            }
            if let Some(prefix) = cap.strip_suffix('*') {
                // `connector.aokie.*` -> one entry per declared command.
                for conn in &self.connectors {
                    for cmd in &conn.commands {
                        let exact = format!("connector.{}.{}", conn.id, cmd);
                        if exact.starts_with(prefix) {
                            out.insert(exact);
                        }
                    }
                }
                // Non-connector wildcards (`flow.*`) expand over the host's own
                // capability list.
                for host_cap in HOST_CAPABILITIES {
                    if host_cap.starts_with(prefix) {
                        out.insert((*host_cap).to_string());
                    }
                }
                continue;
            }
            out.insert(canonical_capability(cap).to_string());
        }
        out
    }

    /// Declared capabilities that were rewritten from a pre-OAIY spelling.
    ///
    /// Surfaced so the plugins UI can say "this plugin uses legacy capability
    /// names" rather than the normalisation being invisible and permanent.
    pub fn legacy_capabilities(&self) -> Vec<(String, String)> {
        self.capabilities
            .iter()
            .map(|c| c.trim())
            .filter_map(|c| {
                let canon = canonical_capability(c);
                (canon != c).then(|| (c.to_string(), canon.to_string()))
            })
            .collect()
    }

    /// Declared capabilities this host will never grant anything for.
    ///
    /// A capability that is neither a host capability nor a declared connector
    /// command grants exactly nothing — which is safe, but silent. `typo.flow.run`
    /// and FormLogic-era names like `companion.admission` both land here, and both
    /// mean a plugin author believes they have a permission they do not. Reporting
    /// them turns a mystery `capability_denied` into a manifest warning.
    pub fn unknown_capabilities(&self) -> Vec<String> {
        let mut declared: BTreeSet<String> = BTreeSet::new();
        for conn in &self.connectors {
            for cmd in &conn.commands {
                declared.insert(format!("connector.{}.{}", conn.id, cmd));
            }
        }
        self.resolved_capabilities()
            .into_iter()
            .filter(|c| !HOST_CAPABILITIES.contains(&c.as_str()) && !declared.contains(c))
            .collect()
    }

    /// Does this plugin hold `cap`? Exact match against the resolved set.
    ///
    /// Deliberately not a prefix or glob test. See the module docs.
    pub fn grants(&self, cap: &str) -> bool {
        self.resolved_capabilities().contains(cap)
    }

    /// Is `command` on `connector_id` declared? The gateway checks this *before*
    /// forwarding, so an undeclared command never reaches the plugin process.
    pub fn declares_command(&self, connector_id: &str, command: &str) -> bool {
        self.connectors
            .iter()
            .any(|c| c.id == connector_id && c.commands.iter().any(|x| x == command))
    }

    /// Does this command require an `idempotencyKey`?
    pub fn is_journalled(&self, command: &str) -> bool {
        self.commands
            .as_ref()
            .is_some_and(|c| c.journalled.iter().any(|j| j == command))
    }

    /// May the plugin emit `name`? Undeclared events are dropped and logged, so
    /// a plugin cannot invent event names to reach triggers it was not granted.
    pub fn declares_event(&self, name: &str) -> bool {
        self.events.iter().any(|e| e == name)
    }
}

/// Host capabilities a plugin may request, beyond connector commands.
///
/// `oaiy.flow.run` lets a plugin ask the host to run a flow — which is how Aokie
/// turns an incoming call into an orchestration. It is gated because a plugin
/// that can start arbitrary flows can reach every capability those flows hold.
/// `oaiy.companion.admission` lets a plugin broker COMPANION DEVICE TRUST — the
/// pairing ceremony that lets an approved phone carry a live call's audio. It is
/// gated because the holder decides which devices the owner is even asked to
/// approve, and a plugin that could enrol its own devices would make the
/// owner's confirmation ceremonial.
pub const HOST_CAPABILITIES: &[&str] = &[
    "oaiy.flow.run",
    "oaiy.events.publish",
    "oaiy.services.read",
    "oaiy.companion.admission",
];

/// Pre-OAIY spellings of host capabilities, mapped to their canonical names.
///
/// Plugins written against FormLogic Desktop declare the bare `flow.run`. Aokie's
/// shipped manifest does exactly this, and it lives in its own repository on its
/// own release cycle — so a host that only recognised `oaiy.flow.run` would load
/// Aokie successfully, show it as running, and then refuse its first
/// `flow.run` RPC with `capability_denied`. A permission failure at call time,
/// from a manifest that plainly intended the permission, is the worst available
/// outcome: it looks like a bug in the plugin.
///
/// So legacy names are normalised **at load**, into the canonical name, and
/// enforcement only ever compares canonical names. This is a rename, not a
/// widening: `flow.run` grants precisely what `oaiy.flow.run` grants and nothing
/// more. [`PluginManifest::legacy_capabilities`] reports which ones were
/// rewritten so the UI can nudge a plugin author instead of the drift becoming
/// permanent and invisible.
pub const LEGACY_HOST_CAPABILITY_ALIASES: &[(&str, &str)] = &[
    ("flow.run", "oaiy.flow.run"),
    // Aokie's shipped manifest declares the bare name; without this mapping the
    // host would load it, show it running, and then refuse the pairing screen
    // for a permission the manifest plainly intended.
    ("companion.admission", "oaiy.companion.admission"),
    ("events.publish", "oaiy.events.publish"),
    ("services.read", "oaiy.services.read"),
];

/// A schemaVersion 4 section under an older schemaVersion is refused, with
/// the version as the reason.
fn refuse_newer_sections(manifest: &Value) -> Result<(), ManifestError> {
    let version = manifest.get("schemaVersion").and_then(Value::as_u64).unwrap_or(1);
    if version >= 4 {
        return Ok(());
    }
    match V4_SECTIONS.iter().find(|key| manifest.get(**key).is_some()) {
        Some(key) => Err(ManifestError::Invalid(format!(
            "{key} needs schemaVersion 4, and this manifest declares {version}: a section is refused under an older schemaVersion rather than half-honoured"
        ))),
        None => Ok(()),
    }
}

fn canonical_capability(cap: &str) -> &str {
    for (legacy, canonical) in LEGACY_HOST_CAPABILITY_ALIASES {
        if cap == *legacy {
            return canonical;
        }
    }
    cap
}

fn is_valid_id(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    s.len() <= 64 && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Dot-namespaced, at least two segments: `call.answer`, `dongle.installDriver`.
fn is_valid_command(s: &str) -> bool {
    if s.len() > 96 || !s.contains('.') {
        return false;
    }
    let segs: Vec<&str> = s.split('.').collect();
    if segs.len() < 2 || segs.iter().any(|x| x.is_empty()) {
        return false;
    }
    segs[0].starts_with(|c: char| c.is_ascii_lowercase())
        && segs
            .iter()
            .all(|seg| seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
}

/// Same shape as a command but conventionally three segments
/// (`aokie.call.incoming`). Matches `event-envelope.schema.json`'s pattern.
fn is_valid_event_name(s: &str) -> bool {
    is_valid_command(s) && s.len() <= 128
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_manifest(body: &serde_json::Value) -> tempdir::TempPluginDir {
        let dir = tempdir::TempPluginDir::new();
        fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_string_pretty(body).unwrap(),
        )
        .unwrap();
        // The entry must exist for realistic tests; resolve_entry does not
        // require existence, but the runner will.
        fs::write(dir.path().join("plugin.exe"), b"stub").unwrap();
        dir
    }

    fn base() -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 3,
            "id": "aokie",
            "name": "Aokie Phone Bridge",
            "version": "0.1.0",
            "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe", "args": ["--stdio"] },
            "capabilities": ["oaiy.flow.run", "connector.aokie.*"],
            "connectors": [{
                "id": "aokie",
                "commands": ["call.answer", "call.dial", "sms.send", "phone.status"]
            }],
            "events": ["aokie.call.incoming", "aokie.call.ended"],
            "commands": { "journalled": ["call.dial", "sms.send"] }
        })
    }

    #[test]
    fn a_well_formed_manifest_loads() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).expect("should load");
        assert_eq!(m.id, "aokie");
        assert_eq!(m.plugin_api_version, 1);
        assert_eq!(m.connectors.len(), 1);
    }

    #[test]
    fn a_manifest_is_read_and_parsed_from_the_same_bytes() {
        // What a package check hashes is what is parsed: `read` gives back the bytes it
        // parsed, and parsing those bytes again needs no second look at the file.
        let d = write_manifest(&base());
        let (m, raw) = PluginManifest::read(d.path()).expect("should read");
        assert_eq!(raw, fs::read(d.path().join("manifest.json")).unwrap());
        fs::remove_file(d.path().join("manifest.json")).unwrap();
        let again = PluginManifest::parse(&raw, d.path()).expect("the bytes are the manifest");
        assert_eq!((again.id, again.entry.command), (m.id, m.entry.command));
    }

    #[test]
    fn a_manifest_that_is_not_text_or_is_far_too_large_does_not_load() {
        let d = tempdir::TempPluginDir::new();
        fs::write(d.path().join("manifest.json"), [0xff, 0xfe, 0x00]).unwrap();
        assert!(matches!(PluginManifest::load(d.path()), Err(ManifestError::Unreadable(_))));

        fs::write(d.path().join("manifest.json"), vec![b' '; MAX_MANIFEST_BYTES as usize + 1]).unwrap();
        let err = PluginManifest::load(d.path()).unwrap_err();
        assert!(matches!(err, ManifestError::Invalid(_)), "{err:?}");
        assert!(err.reason().contains("larger than"), "{err:?}");
    }

    // --- wildcard expansion: the security-relevant part -------------------

    #[test]
    fn a_connector_wildcard_expands_to_exact_commands() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).unwrap();
        let caps = m.resolved_capabilities();

        for cmd in ["call.answer", "call.dial", "sms.send", "phone.status"] {
            assert!(
                caps.contains(&format!("connector.aokie.{cmd}")),
                "expected connector.aokie.{cmd} in {caps:?}"
            );
        }
        assert!(
            !caps.iter().any(|c| c.contains('*')),
            "no pattern may survive expansion: {caps:?}"
        );
    }

    #[test]
    fn expansion_does_not_widen_when_the_plugin_adds_a_command() {
        // The whole reason expansion happens at load time. A grant reviewed by a
        // user must not silently cover commands added later.
        let mut before = base();
        before["connectors"][0]["commands"] = serde_json::json!(["call.answer"]);
        // journalled must stay a subset of the declared commands — validation
        // refuses otherwise, which is how this test first failed.
        before["commands"] = serde_json::json!({ "journalled": [] });
        let d1 = write_manifest(&before);
        let caps_before = PluginManifest::load(d1.path()).unwrap().resolved_capabilities();

        let d2 = write_manifest(&base());
        let caps_after = PluginManifest::load(d2.path()).unwrap().resolved_capabilities();

        assert!(caps_before.contains("connector.aokie.call.answer"));
        assert!(!caps_before.contains("connector.aokie.sms.send"));
        assert!(
            caps_after.contains("connector.aokie.sms.send"),
            "the later manifest does cover it — the point is the two sets differ"
        );
        assert!(caps_after.len() > caps_before.len());
    }

    #[test]
    fn grants_is_an_exact_match_not_a_prefix_match() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(m.grants("connector.aokie.call.answer"));
        // A prefix must NOT be treated as a grant.
        assert!(!m.grants("connector.aokie.call"));
        assert!(!m.grants("connector.aokie"));
        // Nor may a pattern be usable as a capability.
        assert!(!m.grants("connector.aokie.*"));
        // Nor a command the connector never declared.
        assert!(!m.grants("connector.aokie.call.hangup"));
    }

    #[test]
    fn a_host_wildcard_expands_over_host_capabilities_only() {
        let mut v = base();
        v["capabilities"] = serde_json::json!(["oaiy.*"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        let caps = m.resolved_capabilities();
        assert!(caps.contains("oaiy.flow.run"));
        assert!(caps.contains("oaiy.services.read"));
        // It must not reach connector commands.
        assert!(!caps.iter().any(|c| c.starts_with("connector.")), "{caps:?}");
    }

    #[test]
    fn a_wildcard_matching_nothing_yields_nothing() {
        let mut v = base();
        v["capabilities"] = serde_json::json!(["connector.nosuch.*"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(
            m.resolved_capabilities().is_empty(),
            "an unexpandable pattern must not be retained as a pattern"
        );
    }

    // --- legacy capability names ------------------------------------------

    #[test]
    fn the_formlogic_spelling_of_flow_run_is_normalised() {
        // Aokie's shipped manifest declares the bare `flow.run`. Without this,
        // the plugin loads, shows as running, and is refused capability_denied on
        // its first RPC — a permission failure from a manifest that plainly
        // intended the permission.
        let mut v = base();
        v["capabilities"] = serde_json::json!(["flow.run"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(m.grants("oaiy.flow.run"), "the legacy name must grant the canonical one");
        assert!(
            !m.resolved_capabilities().contains("flow.run"),
            "the legacy spelling must not survive into the enforced set"
        );
    }

    #[test]
    fn normalisation_is_a_rename_not_a_widening() {
        let mut legacy = base();
        legacy["capabilities"] = serde_json::json!(["flow.run"]);
        let d1 = write_manifest(&legacy);
        let a = PluginManifest::load(d1.path()).unwrap().resolved_capabilities();

        let mut canon = base();
        canon["capabilities"] = serde_json::json!(["oaiy.flow.run"]);
        let d2 = write_manifest(&canon);
        let b = PluginManifest::load(d2.path()).unwrap().resolved_capabilities();

        assert_eq!(a, b, "the legacy name must grant exactly what the canonical one grants");
    }

    #[test]
    fn rewritten_capabilities_are_reported() {
        let mut v = base();
        v["capabilities"] = serde_json::json!(["flow.run", "connector.aokie.call.answer"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        assert_eq!(
            m.legacy_capabilities(),
            vec![("flow.run".to_string(), "oaiy.flow.run".to_string())],
            "so the UI can nudge the author instead of the drift becoming permanent"
        );
    }

    #[test]
    fn a_canonical_manifest_reports_no_legacy_names() {
        let d = write_manifest(&base());
        assert!(PluginManifest::load(d.path()).unwrap().legacy_capabilities().is_empty());
    }

    #[test]
    fn capabilities_that_grant_nothing_are_reported() {
        // A capability that is neither a host capability nor a declared
        // connector command grants nothing, safely but silently. Reporting it
        // turns a mystery capability_denied into a manifest warning.
        let mut v = base();
        v["capabilities"] = serde_json::json!([
            "oaiy.flow.runn",           // typo
            "aokie.hardware.seize",     // invented; no host or connector meaning
            "connector.aokie.call.answer",
            "flow.run",
            // `companion.admission` used to belong on this list. It is now a
            // real host capability with a legacy mapping, so it must NOT be
            // reported — Aokie's shipped manifest declares exactly that name.
            "companion.admission",
        ]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        let unknown = m.unknown_capabilities();
        assert!(unknown.contains(&"oaiy.flow.runn".to_string()), "{unknown:?}");
        assert!(unknown.contains(&"aokie.hardware.seize".to_string()), "{unknown:?}");
        assert!(
            !unknown.contains(&"companion.admission".to_string()),
            "the legacy spelling now maps to a real host capability: {unknown:?}"
        );
        assert!(
            m.resolved_capabilities().contains("oaiy.companion.admission"),
            "and it must resolve to the canonical name"
        );
        assert!(!unknown.contains(&"oaiy.flow.run".to_string()), "normalised, so known");
        assert!(
            !unknown.contains(&"connector.aokie.call.answer".to_string()),
            "a declared command is known"
        );
    }

    #[test]
    fn an_unknown_capability_still_grants_nothing() {
        let mut v = base();
        v["capabilities"] = serde_json::json!(["companion.admission"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(!m.grants("oaiy.flow.run"));
        assert!(!m.grants("connector.aokie.call.answer"));
    }

    #[test]
    fn flow_run_must_be_declared_to_be_held() {
        let mut v = base();
        v["capabilities"] = serde_json::json!(["connector.aokie.call.answer"]);
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(
            !m.grants("oaiy.flow.run"),
            "a plugin that can start arbitrary flows reaches every capability those flows hold"
        );
    }

    // --- entry path confinement ------------------------------------------

    #[test]
    fn a_traversing_entry_command_is_refused() {
        for bad in [
            "../../../../Windows/System32/cmd.exe",
            "..\\..\\evil.exe",
            "sub/../../escape.exe",
        ] {
            let mut v = base();
            v["entry"]["command"] = serde_json::json!(bad);
            let d = write_manifest(&v);
            let err = PluginManifest::load(d.path()).expect_err(&format!("{bad} must be refused"));
            assert!(
                matches!(err, ManifestError::Invalid(_)),
                "{bad}: got {err:?}"
            );
        }
    }

    #[test]
    fn an_absolute_entry_command_is_refused() {
        let mut v = base();
        #[cfg(windows)]
        {
            v["entry"]["command"] = serde_json::json!("C:\\Windows\\System32\\cmd.exe");
        }
        #[cfg(not(windows))]
        {
            v["entry"]["command"] = serde_json::json!("/bin/sh");
        }
        let d = write_manifest(&v);
        assert!(PluginManifest::load(d.path()).is_err());
    }

    #[test]
    fn a_nested_relative_entry_is_allowed() {
        let mut v = base();
        v["entry"]["command"] = serde_json::json!("bin/plugin.exe");
        let d = write_manifest(&v);
        let m = PluginManifest::load(d.path()).expect("nested paths are legitimate");
        let resolved = m.resolve_entry(d.path()).unwrap();
        assert!(resolved.starts_with(d.path()));
        assert!(resolved.ends_with("plugin.exe"));
    }

    // --- version compatibility -------------------------------------------

    #[test]
    fn an_unsupported_plugin_api_is_refused_with_a_reason() {
        let mut v = base();
        v["pluginApiVersion"] = serde_json::json!(99);
        let d = write_manifest(&v);
        let err = PluginManifest::load(d.path()).unwrap_err();
        assert!(matches!(err, ManifestError::Unsupported(_)));
        let reason = err.reason();
        assert!(reason.contains("99"), "the reason must name the version: {reason}");
        assert!(reason.contains("speaks"), "and what we do support: {reason}");
    }

    #[test]
    fn an_unsupported_schema_version_is_refused() {
        let mut v = base();
        v["schemaVersion"] = serde_json::json!(9);
        let d = write_manifest(&v);
        assert!(matches!(
            PluginManifest::load(d.path()).unwrap_err(),
            ManifestError::Unsupported(_)
        ));
    }

    #[test]
    fn aokies_schema_version_3_is_readable() {
        let d = write_manifest(&base());
        assert_eq!(PluginManifest::load(d.path()).unwrap().schema_version, 3);
    }

    #[test]
    fn an_unknown_entry_kind_is_refused() {
        let mut v = base();
        v["entry"]["kind"] = serde_json::json!("wasm");
        let d = write_manifest(&v);
        assert!(matches!(
            PluginManifest::load(d.path()).unwrap_err(),
            ManifestError::Unsupported(_)
        ));
    }

    // --- forward compatibility -------------------------------------------

    #[test]
    fn unknown_top_level_fields_are_retained_not_refused() {
        let mut v = base();
        v["ui"] = serde_json::json!({ "nav": [{ "id": "receptionist", "label": "AI Receptionist" }] });
        v["serviceDefinitions"] = serde_json::json!([{ "definitionFile": "definitions/phone.json" }]);
        v["somethingFromTheFuture"] = serde_json::json!(true);
        let d = write_manifest(&v);
        fs::create_dir_all(d.path().join("definitions")).unwrap();
        fs::write(d.path().join("definitions/phone.json"), r#"{"id":"demo.phone","name":"Phone","actions":[]}"#).unwrap();
        let m = PluginManifest::load(d.path()).expect("additive fields must not break loading");
        assert!(m.extra.contains_key("ui"));
        assert!(m.extra.contains_key("serviceDefinitions"));
        assert!(m.extra.contains_key("somethingFromTheFuture"));
    }

    // --- malformed input --------------------------------------------------

    #[test]
    fn a_missing_manifest_is_a_typed_error() {
        let d = tempdir::TempPluginDir::new();
        assert!(matches!(
            PluginManifest::load(d.path()).unwrap_err(),
            ManifestError::Unreadable(_)
        ));
    }

    #[test]
    fn invalid_json_is_a_typed_error() {
        let d = tempdir::TempPluginDir::new();
        fs::write(d.path().join("manifest.json"), b"{ not json").unwrap();
        assert!(matches!(
            PluginManifest::load(d.path()).unwrap_err(),
            ManifestError::Malformed(_)
        ));
    }

    #[test]
    fn every_error_reason_is_non_empty_and_actionable() {
        // The registry shows these next to a disabled plugin; a blank or
        // uninformative reason is the silent-skip failure in another costume.
        let cases = [
            ManifestError::Unreadable("no such file".into()),
            ManifestError::Malformed("expected `,`".into()),
            ManifestError::Unsupported("pluginApiVersion 99".into()),
            ManifestError::Invalid("id is empty".into()),
        ];
        for c in cases {
            let r = c.reason();
            assert!(r.len() > 20, "reason too terse: {r:?}");
            assert!(r.contains("manifest") || r.contains("OAIY"), "{r:?}");
        }
    }

    #[test]
    fn a_bad_id_is_refused() {
        for bad in ["Aokie", "9lives", "has space", "has_underscore", ""] {
            let mut v = base();
            v["id"] = serde_json::json!(bad);
            let d = write_manifest(&v);
            assert!(
                PluginManifest::load(d.path()).is_err(),
                "id {bad:?} must be refused"
            );
        }
    }

    #[test]
    fn a_bad_command_name_is_refused() {
        for bad in ["answer", "call.", ".answer", "call..answer", "call answer"] {
            let mut v = base();
            v["connectors"][0]["commands"] = serde_json::json!([bad]);
            v["capabilities"] = serde_json::json!([]);
            v["commands"] = serde_json::json!({ "journalled": [] });
            let d = write_manifest(&v);
            assert!(
                PluginManifest::load(d.path()).is_err(),
                "command {bad:?} must be refused"
            );
        }
    }

    #[test]
    fn a_plugin_cannot_declare_another_subsystems_events() {
        // The forgery fix: the runtime pins event `source` to the plugin, but the
        // dispatcher matches bindings on NAME. So an evil plugin declaring
        // `aokie.call.incoming` (source = itself) fired every real-Aokie binding.
        // The manifest now refuses events not namespaced under something the
        // plugin owns.
        let mut v = base();
        v["id"] = serde_json::json!("evil");
        v["connectors"] = serde_json::json!([]);
        v["capabilities"] = serde_json::json!([]);
        v["commands"] = serde_json::json!({ "journalled": [] });
        v["events"] = serde_json::json!(["aokie.call.incoming"]);
        let d = write_manifest(&v);
        let err = PluginManifest::load(d.path()).expect_err("forged namespace must be refused");
        assert!(err.reason().contains("does not own"), "{}", err.reason());
    }

    #[test]
    fn a_plugin_may_namespace_events_under_its_id_or_a_connector() {
        // aokie: id == aokie, connector id == aokie, events aokie.* — the real
        // shipped shape, which must still load.
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).expect("owned namespace loads");
        assert!(m.declares_event("aokie.call.incoming"));

        // A connector-namespaced event on a differently-named plugin is fine.
        let mut v = base();
        v["id"] = serde_json::json!("acme-weather");
        v["connectors"] = serde_json::json!([{ "id": "weather", "commands": ["forecast.get"] }]);
        v["capabilities"] = serde_json::json!(["connector.weather.forecast.get"]);
        v["commands"] = serde_json::json!({ "journalled": [] });
        v["events"] = serde_json::json!(["weather.updated"]);
        let d = write_manifest(&v);
        assert!(PluginManifest::load(d.path()).is_ok(), "connector-owned namespace is legitimate");
    }

    #[test]
    fn an_un_namespaced_event_is_refused() {
        let mut v = base();
        v["events"] = serde_json::json!(["incoming"]);
        let d = write_manifest(&v);
        assert!(PluginManifest::load(d.path()).is_err());
    }

    #[test]
    fn a_journalled_command_nobody_declares_is_refused() {
        // Otherwise the idempotency requirement silently does not apply to it.
        let mut v = base();
        v["commands"] = serde_json::json!({ "journalled": ["call.teleport"] });
        let d = write_manifest(&v);
        let err = PluginManifest::load(d.path()).unwrap_err();
        assert!(err.reason().contains("call.teleport"), "{err:?}");
    }

    // --- gateway helpers --------------------------------------------------

    #[test]
    fn undeclared_commands_never_reach_the_plugin() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(m.declares_command("aokie", "call.answer"));
        assert!(!m.declares_command("aokie", "call.hangup"));
        assert!(!m.declares_command("other", "call.answer"));
    }

    #[test]
    fn journalled_commands_are_identified() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(m.is_journalled("sms.send"), "sending an SMS twice is not undoable");
        assert!(m.is_journalled("call.dial"));
        assert!(!m.is_journalled("phone.status"), "a read is safe to retry");
    }

    #[test]
    fn undeclared_events_are_identified() {
        let d = write_manifest(&base());
        let m = PluginManifest::load(d.path()).unwrap();
        assert!(m.declares_event("aokie.call.incoming"));
        assert!(
            !m.declares_event("aokie.call.invented"),
            "a plugin must not invent event names to reach triggers it was not granted"
        );
    }

    // --- schemaVersion 4: modules, agentTools, setup ----------------------

    /// Aokie's shipped manifest with the proposed v4 fragment merged in (its
    /// `_notes` dropped): `modules` and a three-step `setup`.
    const AOKIE_V4: &str = include_str!("fixtures/aokie-v4.manifest.json");
    /// Aokie's `definitions/phone.json`, which its `serviceDefinitions` names.
    const AOKIE_PHONE: &str = include_str!("fixtures/aokie-phone.definition.json");

    fn aokie_v4() -> serde_json::Value {
        serde_json::from_str(AOKIE_V4).unwrap()
    }

    /// Aokie's shipped manifest as it is today: schemaVersion 3, no v4 sections.
    fn aokie_v3() -> serde_json::Value {
        let mut v = aokie_v4();
        v["schemaVersion"] = serde_json::json!(3);
        let o = v.as_object_mut().unwrap();
        o.remove("modules");
        o.remove("setup");
        v
    }

    fn write_aokie(body: &serde_json::Value) -> tempdir::TempPluginDir {
        let d = write_manifest(body);
        fs::create_dir_all(d.path().join("definitions")).unwrap();
        fs::write(d.path().join("definitions/phone.json"), AOKIE_PHONE).unwrap();
        d
    }

    fn load_aokie(body: &serde_json::Value) -> Result<PluginManifest, ManifestError> {
        let d = write_aokie(body);
        PluginManifest::load(d.path())
    }

    /// The reason `body` is refused (it must be).
    fn refusal(body: &serde_json::Value) -> String {
        load_aokie(body).expect_err("must be refused").reason()
    }

    fn with_tools(tools: serde_json::Value) -> serde_json::Value {
        let mut v = aokie_v4();
        v["agentTools"] = tools;
        v
    }

    fn with_steps(steps: serde_json::Value) -> serde_json::Value {
        let mut v = aokie_v4();
        v["setup"] = serde_json::json!({ "version": 2, "title": "Set up the AI Receptionist", "steps": steps });
        v
    }

    #[test]
    fn the_live_aokie_manifest_still_loads() {
        // schemaVersion 3, minDesktopVersion 0.1.0 on a 0.1.0 desktop.
        let m = load_aokie(&aokie_v3()).expect("the shipped Aokie loads");
        assert_eq!(m.schema_version, 3);
        assert_eq!(m.min_desktop_version.as_deref(), Some("0.1.0"));
        assert!(m.modules.is_none() && m.setup.is_none() && m.agent_tools.is_empty());
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
    }

    #[test]
    fn aokies_v4_manifest_loads() {
        let m = load_aokie(&aokie_v4()).expect("the v4 fixture loads");
        assert_eq!(m.schema_version, 4);
        assert_eq!(m.modules.as_ref().unwrap().provides, vec!["phone", "calendar"]);
        let setup = m.setup.as_ref().unwrap();
        assert_eq!(setup.version, 1, "no version: 1");
        assert_eq!(setup.title, "Set up Aokie Phone Bridge", "no title: the plugin's name");
        let ids: Vec<&str> = setup.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["consent", "dongle", "pair"]);
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
        // `equals: null` is a test, not an absent operator.
        let StepKind::Screen { done: Some(done), .. } = &setup.steps[0].kind else { panic!("a screen step with a check") };
        let all = done.all.as_ref().unwrap();
        assert_eq!(all[2].path, "blocked");
        assert_eq!(all[2].equals, Some(serde_json::Value::Null));
        // And the modules claim reads it.
        assert!(crate::modules::claims(&m).declared);
        assert!(crate::modules::provided_by(&m).contains(crate::modules::PHONE));
    }

    #[test]
    fn the_v4_sections_reach_the_dashboard_in_camel_case() {
        let mut v = with_tools(serde_json::json!([
            { "action": "aokie.phone/sms.threads", "name": "phone_sms_threads", "audience": ["project", "runner"] }
        ]));
        v["modules"] = serde_json::json!(["phone", "calendar"]);
        let m = load_aokie(&v).unwrap();
        let out = serde_json::to_value(&m).unwrap();
        assert_eq!(out["modules"], serde_json::json!({ "provides": ["phone", "calendar"] }), "the list form is written back as the object form");
        assert_eq!(out["agentTools"], serde_json::json!([{ "action": "aokie.phone/sms.threads", "name": "phone_sms_threads", "audience": ["project", "runner"] }]));
        let steps = &out["setup"]["steps"];
        assert_eq!(out["setup"]["version"], 1);
        assert_eq!(out["setup"]["title"], "Set up Aokie Phone Bridge");
        assert_eq!(steps[0]["kind"], "screen");
        assert_eq!(steps[0]["done"]["all"][2], serde_json::json!({ "path": "blocked", "equals": null }));
        assert_eq!(steps[1]["when"], serde_json::json!({ "command": "settings.get", "path": "settings.transportMode", "notIn": ["native", "auto"] }));
        assert!(steps[0].get("optional").is_none() && steps[0].get("when").is_none(), "absent optional fields are skipped: {}", steps[0]);
        assert!(out.get("warnings").is_none() && out.get("resolvedAgentTools").is_none(), "worked out by the host, not the manifest's");
        // A v3 manifest has none of them.
        let v3 = serde_json::to_value(load_aokie(&aokie_v3()).unwrap()).unwrap();
        assert!(v3.get("modules").is_none() && v3.get("agentTools").is_none() && v3.get("setup").is_none());
    }

    #[test]
    fn a_v4_section_under_an_older_schema_version_is_refused_for_its_version() {
        for (key, value) in [
            ("modules", serde_json::json!({ "provides": ["phone"] })),
            ("agentTools", serde_json::json!([])),
            ("setup", serde_json::json!({ "steps": [] })),
            // Malformed as well: the version is still the reason.
            ("setup", serde_json::json!("not a setup")),
        ] {
            let mut v = aokie_v3();
            v[key] = value;
            let reason = refusal(&v);
            assert!(reason.contains(key) && reason.contains("schemaVersion 4") && reason.contains("declares 3"), "{reason}");
        }
        // And under 1 or 2 alike.
        let mut v = base();
        v["schemaVersion"] = serde_json::json!(2);
        v["modules"] = serde_json::json!(["phone"]);
        let d = write_manifest(&v);
        assert!(PluginManifest::load(d.path()).unwrap_err().reason().contains("declares 2"));
    }

    #[test]
    fn a_module_this_oaiy_does_not_know_is_refused() {
        let mut v = aokie_v4();
        v["modules"] = serde_json::json!({ "provides": ["phone", "fax"] });
        let reason = refusal(&v);
        assert!(reason.contains("\"fax\"") && reason.contains("phone, calendar"), "{reason}");
    }

    #[test]
    fn a_modules_connector_the_plugin_does_not_declare_is_refused() {
        let mut v = aokie_v4();
        v["modules"] = serde_json::json!({ "provides": ["phone"], "connector": "aokie" });
        let m = load_aokie(&v).expect("its own connector is fine");
        assert_eq!(m.modules.unwrap().connector.as_deref(), Some("aokie"));
        v["modules"] = serde_json::json!({ "provides": ["phone"], "connector": "someone-else" });
        assert!(refusal(&v).contains("\"someone-else\""));
    }

    #[test]
    fn a_malformed_modules_section_is_refused() {
        for bad in [serde_json::json!({ "provides": "phone" }), serde_json::json!("phone"), serde_json::json!([{ "id": "phone" }])] {
            let mut v = aokie_v4();
            v["modules"] = bad.clone();
            let reason = refusal(&v);
            assert!(reason.contains("modules must be"), "{bad}: {reason}");
        }
    }

    #[test]
    fn agent_tools_take_their_name_description_and_schema_from_the_action() {
        let v = with_tools(serde_json::json!([
            { "action": "aokie.phone/sms.threads", "name": "phone_sms_threads", "audience": ["project", "runner", "runner"] },
            { "action": "aokie.phone/call.dial", "name": "phone_call", "audience": ["runner"], "confirm": "Call {number} and say: {openingLine}",
              "description": "Ring someone for the business." },
            { "action": "aokie.phone/sms.thread", "name": "phone_sms_thread" }
        ]));
        let m = load_aokie(&v).unwrap();
        let tools = &m.resolved_agent_tools;
        assert_eq!(tools.len(), 3);
        let threads = &tools[0];
        assert_eq!((threads.definition.as_str(), threads.action_id.as_str()), ("aokie.phone", "sms.threads"));
        assert_eq!(threads.description, "Every conversation on the paired phone, most recent first.");
        assert_eq!(threads.side_effects.as_deref(), Some("none"));
        assert!(!threads.has_side_effects());
        assert_eq!(threads.audience, vec!["project", "runner"], "said twice, offered once");
        assert_eq!(threads.input_schema, serde_json::json!({ "type": "object" }));
        assert_eq!(threads.plugin_id, "aokie");
        let call = &tools[1];
        assert_eq!(call.description, "Ring someone for the business.", "the manifest's description wins");
        assert!(call.has_side_effects());
        assert_eq!(call.confirm.as_deref(), Some("Call {number} and say: {openingLine}"));
        assert_eq!(call.input_schema["required"], serde_json::json!(["number", "openingLine"]));
        assert_eq!(call.timeout_ms, Some(30000));
        assert_eq!(tools[2].audience, vec!["project"], "no audience: the project conversation");
    }

    #[test]
    fn a_bad_agent_tool_is_refused() {
        for (tool, says) in [
            (serde_json::json!({ "action": "weather.forecast/get", "name": "forecast" }), "not one of this plugin's service definitions"),
            (serde_json::json!({ "action": "aokie.phone/sms.teleport", "name": "teleport" }), "has no action \"sms.teleport\""),
            (serde_json::json!({ "action": "sms.threads", "name": "threads" }), "<service definition id>/<action id>"),
            (serde_json::json!({ "action": "aokie.phone/sms.threads", "name": "Threads" }), "lowercase letters"),
            (serde_json::json!({ "action": "aokie.phone/sms.threads", "name": "ab" }), "3 to 48"),
            (serde_json::json!({ "action": "aokie.phone/sms.threads", "name": "threads", "audience": ["everyone"] }), "\"everyone\" is not one of"),
            (serde_json::json!({ "action": "aokie.phone/sms.threads", "name": "threads", "audience": [] }), "audience is empty"),
            (serde_json::json!({ "action": "aokie.phone/sms.send", "name": "send_text" }), "needs a \"confirm\" template"),
            (serde_json::json!({ "action": "aokie.phone/sms.send", "name": "send_text", "confirm": "  " }), "needs a \"confirm\" template"),
            (serde_json::json!({ "name": "no_action" }), "agentTools[0]: missing field `action`"),
        ] {
            let reason = refusal(&with_tools(serde_json::json!([tool])));
            assert!(reason.contains(says), "{tool}: {reason}");
        }
        let twice = with_tools(serde_json::json!([
            { "action": "aokie.phone/sms.threads", "name": "threads" },
            { "action": "aokie.phone/phone.status", "name": "threads" }
        ]));
        assert!(refusal(&twice).contains("used twice"));
    }

    #[test]
    fn an_action_that_does_not_say_it_is_harmless_needs_a_confirm_template() {
        let d = write_aokie(&with_tools(serde_json::json!([{ "action": "aokie.phone/status.read", "name": "status_read" }])));
        fs::write(
            d.path().join("definitions/phone.json"),
            r#"{"id":"aokie.phone","name":"Phone","actions":[{"id":"status.read","transport":{"kind":"plugin-command","command":"phone.status"}}]}"#,
        )
        .unwrap();
        let reason = PluginManifest::load(d.path()).unwrap_err().reason();
        assert!(reason.contains("does not say it has no side effects"), "{reason}");
    }

    #[test]
    fn duplicate_or_malformed_step_ids_are_refused() {
        let screen = |id: &str| serde_json::json!({ "id": id, "kind": "screen", "title": "A", "screen": "receptionist-home", "view": "phone" });
        assert!(refusal(&with_steps(serde_json::json!([screen("pair"), screen("pair")]))).contains("used twice"));
        for bad in ["Pair", "9pair", "pair_phone", "a-very-long-step-id-that-runs-past-forty-chars"] {
            assert!(refusal(&with_steps(serde_json::json!([screen(bad)]))).contains("the id must be"), "{bad}");
        }
    }

    #[test]
    fn a_screen_step_naming_a_missing_screen_is_refused() {
        let v = with_steps(serde_json::json!([{ "id": "pair", "kind": "screen", "title": "Pair", "screen": "nowhere", "view": "phone" }]));
        assert!(refusal(&v).contains("\"nowhere\", which is not in ui.screens"));
    }

    #[test]
    fn a_check_must_read_a_declared_command_that_is_not_journalled() {
        let step = |done: serde_json::Value| {
            with_steps(serde_json::json!([{ "id": "pair", "kind": "screen", "title": "Pair", "screen": "receptionist-home", "view": "phone", "done": done }]))
        };
        let undeclared = refusal(&step(serde_json::json!({ "command": "phone.teleport", "path": "paired", "equals": true })));
        assert!(undeclared.contains("\"phone.teleport\", which no connector"), "{undeclared}");
        let journalled = refusal(&step(serde_json::json!({ "command": "phone.connect", "path": "ok", "equals": true })));
        assert!(journalled.contains("journalled"), "{journalled}");
        // `when` is held to the same rule.
        let when = with_steps(serde_json::json!([{ "id": "pair", "kind": "screen", "title": "Pair", "screen": "receptionist-home", "view": "phone",
            "when": { "command": "sms.send", "path": "ok", "equals": true } }]));
        assert!(refusal(&when).contains("journalled"));
    }

    #[test]
    fn a_check_has_exactly_one_test_per_condition() {
        let step = |done: serde_json::Value| {
            with_steps(serde_json::json!([{ "id": "pair", "kind": "screen", "title": "Pair", "screen": "receptionist-home", "view": "phone", "done": done }]))
        };
        for (done, says) in [
            (serde_json::json!({ "command": "phone.status", "path": "paired" }), "has no test"),
            (serde_json::json!({ "command": "phone.status", "path": "paired", "equals": true, "present": true }), "more than one test"),
            (serde_json::json!({ "command": "phone.status", "equals": true }), "needs a dot-separated path"),
            (serde_json::json!({ "command": "phone.status", "path": "a..b", "equals": true }), "needs a dot-separated path"),
            (serde_json::json!({ "command": "phone.status", "all": [] }), "empty \"all\""),
            (serde_json::json!({ "command": "phone.status", "path": "paired", "equals": true, "all": [{ "path": "connected", "present": true }] }), "one or the other"),
            (serde_json::json!({ "command": "phone.status", "all": [{ "path": "connected" }] }), "all[0] has no test"),
        ] {
            let reason = refusal(&step(done.clone()));
            assert!(reason.contains(says), "{done}: {reason}");
        }
        for good in [
            serde_json::json!({ "command": "phone.status", "path": "paired", "present": true }),
            serde_json::json!({ "command": "phone.status", "path": "state", "in": ["a", "b"] }),
            serde_json::json!({ "command": "phone.status", "all": [{ "path": "connected", "equals": true }, { "path": "pairingConfirm", "equals": null }] }),
        ] {
            load_aokie(&step(good.clone())).unwrap_or_else(|e| panic!("{good}: {}", e.reason()));
        }
    }

    #[test]
    fn a_settings_step_reads_and_writes_through_declared_commands() {
        let step = |extra: serde_json::Value| {
            let mut s = serde_json::json!({ "id": "behaviour", "kind": "settings", "title": "How calls are handled", "optional": true,
                "fields": [{ "key": "holdAndCallWaiting", "label": "Hold and call waiting", "type": "bool" }] });
            s.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            with_steps(serde_json::json!([s]))
        };
        let m = load_aokie(&step(serde_json::json!({}))).expect("the defaults, settings.get and settings.set, are declared");
        assert!(m.setup.unwrap().steps[0].optional);
        let read = refusal(&step(serde_json::json!({ "read": { "command": "settings.fetch" } })));
        assert!(read.contains(".read sends \"settings.fetch\", which no connector"), "{read}");
        let journalled = refusal(&step(serde_json::json!({ "read": { "command": "sms.send" } })));
        assert!(journalled.contains(".read") && journalled.contains("journalled"), "{journalled}");
        let write = refusal(&step(serde_json::json!({ "write": { "command": "settings.put" } })));
        assert!(write.contains(".write sends \"settings.put\", which no connector"), "{write}");
        // A journalled write is fine (writing is what it is for).
        load_aokie(&step(serde_json::json!({ "write": { "command": "phone.connect" } }))).unwrap();
        // The defaults must be declared too.
        let mut undeclared = step(serde_json::json!({}));
        let commands = undeclared["connectors"][0]["commands"].as_array_mut().unwrap();
        commands.retain(|c| c != "settings.get");
        undeclared["capabilities"] = serde_json::json!([]);
        assert!(refusal(&undeclared).contains("\"settings.get\", which no connector"));
    }

    #[test]
    fn a_settings_field_is_well_formed() {
        let fields = |f: serde_json::Value| with_steps(serde_json::json!([{ "id": "behaviour", "kind": "settings", "title": "Calls", "fields": f }]));
        for (f, says) in [
            (serde_json::json!([]), "no fields"),
            (serde_json::json!([{ "key": "mode", "label": "Mode", "type": "choice" }]), "a choice needs options"),
            (serde_json::json!([{ "key": "on", "label": "On", "type": "bool", "options": [{ "value": 1, "label": "One" }] }]), "only for a choice"),
            (serde_json::json!([{ "key": "on", "label": "On", "type": "bool" }, { "key": "on", "label": "Again", "type": "text" }]), "used twice"),
            (serde_json::json!([{ "key": "on", "label": "On", "type": "toggle" }]), "unknown variant `toggle`"),
        ] {
            let reason = refusal(&fields(f.clone()));
            assert!(reason.contains(says), "{f}: {reason}");
        }
        load_aokie(&fields(serde_json::json!([{ "key": "mode", "label": "Mode", "type": "choice", "help": "How.",
            "options": [{ "value": "desktop_realtime", "label": "OAIY" }, { "value": 2, "label": "Two" }] }]))).unwrap();
    }

    #[test]
    fn requirements_name_a_service_or_an_engine_group_and_no_model() {
        let req = |r: serde_json::Value| with_steps(serde_json::json!([{ "id": "speech", "kind": "requirements", "title": "Hearing and speaking", "requires": r }]));
        let m = load_aokie(&req(serde_json::json!([
            { "kind": "service", "id": "oaiy-voice", "why": "Hears callers." },
            { "kind": "engineModel", "group": "llm", "why": "Answers calls." }
        ])))
        .unwrap();
        let StepKind::Requirements { requires } = &m.setup.as_ref().unwrap().steps[0].kind else { panic!() };
        assert_eq!(requires[1], Requirement::EngineModel { group: "llm".into(), why: Some("Answers calls.".into()) });
        assert!(refusal(&req(serde_json::json!([]))).contains("nothing in requires"));
        assert!(refusal(&req(serde_json::json!([{ "kind": "service", "id": " " }]))).contains("service id is empty"));
        assert!(refusal(&req(serde_json::json!([{ "kind": "model", "id": "qwen" }]))).contains("unknown variant `model`"));
    }

    #[test]
    fn a_structurally_broken_step_is_refused_and_says_which() {
        for (step, says) in [
            (serde_json::json!({ "id": "x", "kind": "wizardry", "title": "X" }), "setup.steps[0] (\"x\"): unknown variant `wizardry`"),
            (serde_json::json!({ "id": "x", "kind": "screen", "title": "X", "view": "phone" }), "missing field `screen`"),
            (serde_json::json!({ "id": "x", "kind": "screen", "title": "X", "screen": "receptionist-home" }), "missing field `view`"),
            (serde_json::json!({ "id": "x", "kind": "screen", "title": " ", "screen": "receptionist-home", "view": "phone" }), "the title is empty"),
            (serde_json::json!({ "id": "x", "kind": "host", "title": "X" }), "missing field `action`"),
        ] {
            let reason = refusal(&with_steps(serde_json::json!([step.clone()])));
            assert!(reason.contains(says), "{step}: {reason}");
        }
        let mut v = aokie_v4();
        v["setup"] = serde_json::json!({ "version": 0, "steps": [] });
        assert!(refusal(&v).contains("setup.version must be 1 or more"));
        v["setup"] = serde_json::json!({ "title": "No steps" });
        assert!(refusal(&v).contains("setup: missing field `steps`"));
    }

    #[test]
    fn a_host_step_this_oaiy_does_not_know_is_left_out_with_a_warning() {
        let v = with_steps(serde_json::json!([
            { "id": "permissions", "kind": "permissions", "title": "What it may do" },
            { "id": "answer", "kind": "host", "action": "phone.answerWithOaiy", "title": "Answer calls and texts with OAIY" },
            { "id": "hologram", "kind": "host", "action": "phone.hologram", "title": "Beam callers in" },
            { "id": "business", "kind": "host", "action": "calendar.business", "title": "Your business" }
        ]));
        let m = load_aokie(&v).expect("a newer plugin still loads");
        let ids: Vec<&str> = m.setup.as_ref().unwrap().steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["permissions", "answer", "business"]);
        assert_eq!(m.warnings.len(), 1, "{:?}", m.warnings);
        assert!(m.warnings[0].contains("\"phone.hologram\" needs a newer OAIY"), "{:?}", m.warnings);
        let setup = m.setup.as_ref().unwrap();
        assert_eq!((setup.version, setup.title.as_str()), (2, "Set up the AI Receptionist"));
    }

    #[test]
    fn a_plugin_needing_a_newer_desktop_is_refused() {
        let mut v = aokie_v3();
        v["minDesktopVersion"] = serde_json::json!("99.0.0");
        let err = load_aokie(&v).unwrap_err();
        assert!(matches!(err, ManifestError::Unsupported(_)));
        assert!(err.reason().contains("needs OAIY Desktop 99.0.0 or later") && err.reason().contains(DESKTOP_VERSION), "{}", err.reason());
        for fine in [DESKTOP_VERSION, "0.0.9", "0.1.0-beta.1", "0.1.0+build.7"] {
            v["minDesktopVersion"] = serde_json::json!(fine);
            load_aokie(&v).unwrap_or_else(|e| panic!("{fine}: {}", e.reason()));
        }
    }

    #[test]
    fn an_unreadable_min_desktop_version_is_a_warning() {
        let mut v = aokie_v3();
        for odd in ["soon", "1.2", "v0.9.0"] {
            v["minDesktopVersion"] = serde_json::json!(odd);
            let m = load_aokie(&v).unwrap_or_else(|e| panic!("{odd}: {}", e.reason()));
            assert_eq!(m.warnings.len(), 1, "{odd}: {:?}", m.warnings);
            assert!(m.warnings[0].contains("minDesktopVersion") && m.warnings[0].contains(odd), "{:?}", m.warnings);
        }
    }

    #[test]
    fn versions_compare_as_semver() {
        let newer = |a: &str, b: &str| needs_newer_desktop(a, b).unwrap();
        assert!(newer("0.2.0", "0.1.9"));
        assert!(newer("0.10.0", "0.9.0"), "numerically, not as text");
        assert!(!newer("0.1.0", "0.1.0"));
        assert!(!newer("0.1.0-beta", "0.1.0"), "a pre-release is below its release");
        assert!(newer("0.1.0", "0.1.0-rc.1"));
        assert!(newer("1.0.0-rc.10", "1.0.0-rc.9"));
        assert!(newer("1.0.0-beta", "1.0.0-alpha"));
        assert!(newer("1.0.0-alpha.1", "1.0.0-alpha"));
        assert!(!newer("1.0.0+later", "1.0.0"), "build metadata does not count");
        assert!(needs_newer_desktop("1.x.0", "0.1.0").is_err());
    }

    /// Minimal scratch directory helper, so these tests need no dev-dependency.
    mod tempdir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicU32, Ordering};

        static N: AtomicU32 = AtomicU32::new(0);

        pub struct TempPluginDir(PathBuf);

        impl TempPluginDir {
            pub fn new() -> Self {
                let n = N.fetch_add(1, Ordering::Relaxed);
                let p = std::env::temp_dir().join(format!(
                    "oaiy-plugin-test-{}-{n}",
                    std::process::id()
                ));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(&p).expect("create temp plugin dir");
                Self(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempPluginDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
}
