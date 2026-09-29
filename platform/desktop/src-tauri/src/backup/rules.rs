//! What a backup holds and what it leaves out, and why.
//!
//! The rule is an allow-list with a deny-list in front of it. A file is backed up only if it is
//! one of the personal stores named in [`include_category`], and never if it matches a rule in
//! [`RULES`] first: every credential, key and machine-bound secret is denied by name, so a new
//! secret file added by later code is not swept in by accident (it is not on the allow-list) and
//! an old one cannot be added by a badly named store (the deny-list runs first). Anything the
//! rules do not recognise is left out and listed, so the person can see what was not saved.
//!
//! The same rules decide what a restore accepts: a name in a backup that these rules would leave
//! out (a credential, a path into `plugins/`) makes the whole backup refused, so a restore never
//! touches an excluded item.
//!
//! Paths are relative to the data folder, with forward slashes, and matched without regard to case.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a backed-up file belongs to, for the summaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Category {
    Contacts,
    Calendar,
    Flows,
    Settings,
    History,
    Voices,
    Templates,
    PluginData,
    Providers,
    Connectors,
    Agent,
}

impl Category {
    pub const ALL: [Category; 11] = [
        Category::Contacts,
        Category::Calendar,
        Category::Flows,
        Category::Settings,
        Category::History,
        Category::Voices,
        Category::Templates,
        Category::PluginData,
        Category::Providers,
        Category::Connectors,
        Category::Agent,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Category::Contacts => "contacts",
            Category::Calendar => "calendar",
            Category::Flows => "flows",
            Category::Settings => "settings",
            Category::History => "history",
            Category::Voices => "voices",
            Category::Templates => "templates",
            Category::PluginData => "pluginData",
            Category::Providers => "providers",
            Category::Connectors => "connectors",
            Category::Agent => "agent",
        }
    }

    /// How the person reads it.
    pub fn label(self) -> &'static str {
        match self {
            Category::Contacts => "Contacts and what is remembered about callers",
            Category::Calendar => "Calendar",
            Category::Flows => "Flows and triggers",
            Category::Settings => "Settings and setup",
            Category::History => "Run history and the Agent's change log",
            Category::Voices => "Voices",
            Category::Templates => "Service templates you edited or added",
            Category::PluginData => "Plugin data",
            Category::Providers => "AI providers and their API keys",
            Category::Connectors => "Connector descriptors",
            Category::Agent => "Agent conversations and projects",
        }
    }
}

/// One thing a backup leaves out, as the manifest records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Excluded {
    pub pattern: String,
    pub reason: String,
    /// What the person does again after a restore because of it (a credential that was left out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redo: Option<String>,
}

/// A reason to leave a file or folder out.
pub struct Rule {
    pub id: &'static str,
    /// How the manifest shows the rule.
    pub pattern: &'static str,
    pub reason: &'static str,
    pub redo: Option<&'static str>,
    /// A folder this rule leaves out is still walked, because a rule of its own applies to something
    /// inside it (the link folder holds the credential, which has its own reason and its own redo).
    descend: bool,
    matches: fn(&str) -> bool,
}

impl Rule {
    fn record(&self) -> Excluded {
        Excluded { pattern: self.pattern.to_string(), reason: self.reason.to_string(), redo: self.redo.map(str::to_string) }
    }
}

/// `p` is `dir` or is inside it.
fn under(p: &str, dir: &str) -> bool {
    p == dir || (p.len() > dir.len() && p.starts_with(dir) && p.as_bytes()[dir.len()] == b'/')
}

fn file_name(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// A name that looks like it holds a key or a credential. A key file is left out wherever it sits;
/// plugins keep their own pairing and sign-in state in their data folders, and what they bind to
/// this computer cannot be restored on another, so a plugin data path with such a word in it is left
/// out too (the words are not applied to your flows or contacts, whose names are yours).
fn secret_looking(p: &str) -> bool {
    let in_plugin_data = under(p, "plugin-data");
    p.split('/').any(|part| {
        let ext_hit = [".key", ".pem", ".pfx", ".p12", ".dpapi", ".sealed"].iter().any(|e| part.ends_with(e));
        let word_hit = in_plugin_data
            && ["token", "secret", "credential", "password", "passwd", "pairing", "roster", "outbox", "keystore", "private", "cookie"]
                .iter()
                .any(|w| part.contains(w));
        ext_hit || word_hit || part == "auth.json"
    })
}

pub const RULE_PROVIDER_KEYS: &str = "provider-keys";
pub const RULE_TEMPLATE_SEED: &str = "built-in-template";

static RULES: &[Rule] = &[
    Rule {
        id: "link-account",
        pattern: "link/account.json",
        reason: "The FormLogic link credential and this computer's instance id: a second live copy would clash with the original.",
        redo: Some("Link FormLogic again (Connections); the old computer's entry can be removed in FormLogic."),
        descend: false,
        matches: |p| p == "link/account.json",
    },
    Rule {
        id: "tunnel-identity",
        pattern: "desktop-e2e-identity.key",
        reason: "The tunnel identity key: browsers pinned the old key, and a key is never copied into a backup.",
        redo: Some("Your browsers will ask once to trust this computer again."),
        descend: false,
        matches: |p| p == "desktop-e2e-identity.key" || p == "desktop-e2e-published.json",
    },
    Rule {
        id: "data-node-key",
        pattern: "data-node-signing.key",
        reason: "The data node's signing key.",
        redo: Some("Enrol this computer as a data node again and approve it in FormLogic."),
        descend: false,
        matches: |p| p == "data-node-signing.key",
    },
    Rule {
        id: "companion",
        pattern: "companion/**",
        reason: "The phone pairing: this computer's key for the phone, the roster of trusted phones and the relay's bearer.",
        redo: Some("Pair your phone again and enter the relay key again."),
        descend: false,
        matches: |p| under(p, "companion"),
    },
    Rule {
        id: "pairings",
        pattern: "bridge/pairings.json",
        reason: "Tokens of paired browsers and apps.",
        redo: Some("Pair your browsers and apps with OAIY again."),
        descend: false,
        matches: |p| p == "bridge/pairings.json",
    },
    Rule {
        id: "codex-signin",
        pattern: "ai/codex-home/**",
        reason: "The ChatGPT sign-in, which belongs to the Codex program.",
        redo: Some("Sign in to ChatGPT again."),
        descend: false,
        matches: |p| under(p, "ai/codex-home"),
    },
    Rule {
        id: "engines",
        pattern: "engines/**",
        reason: "The engines' programs, models and settings (their settings hold a gateway key and the Hugging Face token).",
        redo: Some("Choose your engine models again, and enter your Hugging Face token if you use one."),
        descend: false,
        matches: |p| under(p, "engines"),
    },
    Rule {
        id: "hf-token",
        pattern: "hf-token",
        reason: "The Hugging Face token.",
        redo: Some("Enter your Hugging Face token again if you use one."),
        descend: false,
        matches: |p| file_name(p) == "hf-token",
    },
    Rule {
        id: RULE_PROVIDER_KEYS,
        pattern: "ai/providers.json",
        reason: "Your API provider keys. Tick \"Include my API provider keys\" to add them.",
        redo: Some("Enter your API provider keys again."),
        descend: false,
        matches: |p| p == "ai/providers.json",
    },
    Rule {
        id: "link-state",
        pattern: "link/**",
        reason: "The FormLogic link's own state (events waiting to be sent, delivery markers, its cache): it belongs to the link, which you make again.",
        redo: None,
        descend: true,
        matches: |p| under(p, "link"),
    },
    Rule {
        id: "keys",
        pattern: "keys/**",
        reason: "Keys and vault files are never put into a backup.",
        redo: None,
        descend: false,
        matches: |p| under(p, "keys") || under(p, "vault"),
    },
    Rule {
        id: "secret-names",
        pattern: "*.key, *.pem, *.dpapi, auth.json, and in plugin data anything named *token*, *secret*, *credential*, *password*, *pairing*, *outbox*",
        reason: "Looks like a key, a sign-in or something a plugin ties to this computer: those cannot be restored on another, and a backup never holds a credential.",
        redo: Some("Sign in or pair again in each plugin that asks."),
        descend: false,
        matches: secret_looking,
    },
    Rule {
        id: "programs",
        pattern: "models/**, python/**, venvs/**, node/**, bin/**",
        reason: "Programs and downloaded models: large, and installed or downloaded again.",
        redo: None,
        descend: false,
        matches: |p| ["models", "python", "venvs", "node", "bin"].iter().any(|d| under(p, d)),
    },
    Rule {
        id: "logs-and-caches",
        pattern: "logs/**, tmp/**, cache/**, *.log",
        reason: "Logs, caches and temporary files.",
        redo: None,
        descend: false,
        matches: |p| ["logs", "tmp", "cache"].iter().any(|d| under(p, d)) || file_name(p).ends_with(".log") || file_name(p).ends_with(".log.1"),
    },
    Rule {
        id: "rollback",
        pattern: "*.bak, *.tmp, *.corrupt, *.part, plugins/.backup-*",
        reason: "Rollback copies and half-written files: they can hold older plaintext.",
        redo: None,
        descend: false,
        matches: |p| {
            let n = file_name(p);
            n.ends_with(".bak") || n.ends_with(".tmp") || n.ends_with(".corrupt") || n.ends_with(".part") || p.split('/').any(|c| c.starts_with(".backup-"))
        },
    },
    Rule {
        id: "plugins",
        pattern: "plugins/**",
        reason: "Installed plugin programs, their trust decisions and switches: install and trust the plugins again on the new computer.",
        redo: Some("Install your plugins again and accept what they may do."),
        descend: false,
        matches: |p| under(p, "plugins"),
    },
    Rule {
        id: "rebuilt",
        pattern: "scripts/**, model-catalog.json, services-running.json, .*.seed",
        reason: "Rebuilt by OAIY from what it ships or from your templates.",
        redo: None,
        descend: false,
        matches: |p| under(p, "scripts") || p == "model-catalog.json" || p == "services-running.json" || (file_name(p).starts_with('.') && file_name(p).ends_with(".seed")),
    },
    Rule {
        id: "own-state",
        pattern: "restore/**, backup/**",
        reason: "The backup's and restore's own working folders.",
        redo: None,
        descend: false,
        matches: |p| under(p, "restore") || under(p, "backup") || p == ".oaiy-write-test",
    },
];

/// The rule for an unedited built-in template, which a restore has no need to bring back.
static TEMPLATE_SEED_RULE: Rule = Rule {
    id: RULE_TEMPLATE_SEED,
    pattern: "templates/<built-in>.json",
    reason: "A built-in service template you did not edit: OAIY makes it again.",
    redo: None,
    descend: false,
    matches: |_| false,
};

/// Every rule that can leave something out, for the documentation and the tests.
pub fn all_rules() -> impl Iterator<Item = &'static Rule> {
    RULES.iter().chain(std::iter::once(&TEMPLATE_SEED_RULE))
}

/// Plugins whose data OAIY knows how to back up, and the files of each that are: only these, each
/// cleaned of PINs, keys, tokens and values sealed to a computer (see [`super::sanitize`]). A plugin
/// keeps its own pairings, queues and sealed state in its data folder, and none of it can be told
/// apart from a settings file by name, so a plugin's data is opt-in, plugin by plugin and file by
/// file. (Aokie's `settings.json` holds its settings and, sealed to the computer, the manager's PIN;
/// its pairing store, PIN throttle and outbox stay behind.)
pub const PLUGIN_POLICIES: &[(&str, &[&str])] = &[("aokie", &["settings.json"])];

/// `plugin-data/<id>/<rest>`: the plugin id, and what is inside its folder.
fn plugin_parts(p: &str) -> Option<(&str, &str)> {
    let rest = p.strip_prefix("plugin-data/")?;
    let (id, inside) = rest.split_once('/').unwrap_or((rest, ""));
    Some((id, inside))
}

fn plugin_known(id: &str) -> bool {
    PLUGIN_POLICIES.iter().any(|(known, _)| *known == id)
}

fn plugin_file_listed(p: &str) -> bool {
    plugin_parts(p).is_some_and(|(id, inside)| !inside.is_empty() && PLUGIN_POLICIES.iter().any(|(known, files)| *known == id && files.contains(&inside)))
}

/// Voice clips and the small files that go with them.
fn voice_file_ok(p: &str) -> bool {
    let name = file_name(p);
    name == "chosen" || ["wav", "mp3", "ogg", "flac", "m4a", "opus", "txt", "json"].iter().any(|e| name.rsplit_once('.').is_some_and(|(_, ext)| ext == *e))
}

/// What the allow-list says about a file that no deny rule caught: the category it is kept under and
/// whether it is a secret (written private when restored).
fn include_category(p: &str) -> Option<(Category, bool)> {
    let slashes = p.matches('/').count();
    match p {
        "callers.json" => Some((Category::Contacts, false)),
        "triggers.json" | "bridge/ledger.jsonl" => Some((Category::Flows, false)),
        "setup.json" | "agent.json" | "control.json" => Some((Category::Settings, false)),
        "services-autostart.json" => Some((Category::Templates, false)),
        "control-log.jsonl" | "control-log.jsonl.1" | "bridge/deadletters.jsonl" => Some((Category::History, false)),
        "ai/providers.json" => Some((Category::Providers, true)),
        "calendar/calendar.json" => Some((Category::Calendar, false)),
        _ if under(p, "flows") && p != "flows" => Some((Category::Flows, false)),
        _ if under(p, "connectors") && p != "connectors" && slashes == 1 && p.ends_with(".json") => Some((Category::Connectors, false)),
        _ if under(p, "voices") && p != "voices" && slashes == 1 && voice_file_ok(p) => Some((Category::Voices, false)),
        _ if under(p, "templates") && slashes == 1 && p.ends_with(".json") => Some((Category::Templates, false)),
        // plugin-data/<id>/<file>: only the files of a plugin on the list
        _ if plugin_file_listed(p) => Some((Category::PluginData, false)),
        _ => None,
    }
}

/// Folders the walk goes into: the ones that can hold something on the allow-list.
fn dir_may_hold_personal_data(p: &str) -> bool {
    ["calendar", "flows", "connectors", "voices", "templates", "plugin-data", "bridge", "ai"].iter().any(|d| under(p, d))
}

/// What to do with a path.
#[derive(Clone, Copy)]
pub enum Decision {
    Include { category: Category, secret: bool },
    Exclude(&'static Rule),
    /// Not on the allow-list and not denied by name: left out, and listed as not recognised.
    Unknown,
}

/// Decide about one relative path (`include_keys`: the person ticked "Include my API provider keys").
pub fn classify(rel: &str, include_keys: bool) -> Decision {
    let p = rel.to_lowercase();
    for rule in RULES {
        if rule.id == RULE_PROVIDER_KEYS && include_keys {
            continue;
        }
        if (rule.matches)(&p) {
            return Decision::Exclude(rule);
        }
    }
    match include_category(&p) {
        Some((category, secret)) => Decision::Include { category, secret },
        None => Decision::Unknown,
    }
}

/// Whether the walk should go into this folder, or leave it out (and why).
pub fn classify_dir(rel: &str, include_keys: bool) -> Decision {
    let p = rel.to_lowercase();
    for rule in RULES {
        if rule.id == RULE_PROVIDER_KEYS && include_keys {
            continue;
        }
        if (rule.matches)(&p) {
            return Decision::Exclude(rule);
        }
    }
    if dir_may_hold_personal_data(&p) {
        // Not a category of its own: its files decide.
        return Decision::Include { category: Category::Settings, secret: false };
    }
    Decision::Unknown
}

/// The category a name in a backup belongs to, or why the backup must be refused for holding it.
/// Used when a backup is read: nothing that a backup never holds may come in through one. What a
/// backup says about itself (that it holds keys, say) plays no part: whether the provider list comes
/// back, and with or without its keys, is what the person ticks when restoring.
pub fn category_of_backup_entry(name: &str) -> Result<(Category, bool), String> {
    if name == super::AGENT_ENTRY {
        return Ok((Category::Agent, false));
    }
    match classify(name, true) {
        Decision::Include { category, secret } => Ok((category, secret)),
        Decision::Exclude(rule) => Err(format!("it holds an item OAIY never backs up ({})", rule.pattern)),
        Decision::Unknown => Err("it holds an item that OAIY does not recognise".to_string()),
    }
}

/// How a file is cleaned as it is copied into a backup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sanitize {
    None,
    /// The calendar, without its FormLogic sync state.
    Calendar,
    /// A plugin's settings, without PINs, keys, tokens and sealed values.
    PluginJson,
}

/// One file the plan will copy.
#[derive(Clone, Debug)]
pub struct PlanItem {
    pub rel: String,
    pub abs: PathBuf,
    pub size: u64,
    pub category: Category,
    pub secret: bool,
    pub sanitize: Sanitize,
}

/// What a walk of the data folder found.
#[derive(Debug, Default)]
pub struct Plan {
    pub items: Vec<PlanItem>,
    /// What was left out, one record per rule (or per unrecognised top-level name).
    pub excluded: Vec<Excluded>,
    /// Symbolic links and junctions, which are never followed.
    pub links: Vec<String>,
    /// Entries that could not be read.
    pub unreadable: Vec<String>,
}

/// Whether a directory entry is a link that must not be followed: a symbolic link, a junction, or
/// any other reparse point.
pub(crate) fn is_link(entry_meta: &std::fs::Metadata) -> bool {
    if entry_meta.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if entry_meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return true;
        }
    }
    false
}

/// Walk `data_dir` and decide about everything in it, without following a link and without going
/// into a folder that is left out (a models folder can be tens of gigabytes).
pub fn plan(data_dir: &Path, include_keys: bool) -> Plan {
    let mut plan = Plan::default();
    let mut excluded: BTreeMap<String, Excluded> = BTreeMap::new();
    let mut stack: Vec<(PathBuf, String)> = vec![(data_dir.to_path_buf(), String::new())];
    while let Some((dir, rel_dir)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => {
                if !rel_dir.is_empty() {
                    plan.unreadable.push(rel_dir.clone());
                }
                continue;
            }
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel = if rel_dir.is_empty() { name.clone() } else { format!("{rel_dir}/{name}") };
            let abs = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&abs) else {
                plan.unreadable.push(rel);
                continue;
            };
            if is_link(&meta) {
                plan.links.push(rel);
                continue;
            }
            if meta.is_dir() {
                if let Some(record) = plugin_folder_exclusion(&rel) {
                    excluded.entry(record.pattern.clone()).or_insert(record);
                    continue;
                }
                match classify_dir(&rel, include_keys) {
                    Decision::Include { .. } => stack.push((abs, rel)),
                    Decision::Exclude(rule) if rule.descend => stack.push((abs, rel)),
                    Decision::Exclude(rule) => {
                        excluded.entry(rule.pattern.to_string()).or_insert_with(|| rule.record());
                    }
                    Decision::Unknown => {
                        note_unknown(&mut excluded, &rel, true);
                    }
                }
                continue;
            }
            if !meta.is_file() {
                continue;
            }
            match classify(&rel, include_keys) {
                Decision::Include { category, secret } => {
                    if category == Category::Templates && template_is_unedited(&abs) {
                        excluded.entry(TEMPLATE_SEED_RULE.pattern.to_string()).or_insert_with(|| TEMPLATE_SEED_RULE.record());
                        continue;
                    }
                    let sanitize = match category {
                        Category::Calendar => Sanitize::Calendar,
                        Category::PluginData => Sanitize::PluginJson,
                        _ => Sanitize::None,
                    };
                    plan.items.push(PlanItem { rel, abs, size: meta.len(), category, secret, sanitize });
                }
                Decision::Exclude(rule) => {
                    excluded.entry(rule.pattern.to_string()).or_insert_with(|| rule.record());
                }
                Decision::Unknown => match plugin_other_files(&rel) {
                    Some(record) => {
                        excluded.entry(record.pattern.clone()).or_insert(record);
                    }
                    None => note_unknown(&mut excluded, &rel, false),
                },
            }
        }
    }
    plan.items.sort_by(|a, b| a.rel.cmp(&b.rel));
    plan.links.sort();
    plan.unreadable.sort();
    plan.excluded = excluded.into_values().collect();
    plan
}

/// A plugin's folder that OAIY has no policy for is left out whole, and listed.
fn plugin_folder_exclusion(rel: &str) -> Option<Excluded> {
    let lower = rel.to_lowercase();
    let (id, inside) = plugin_parts(&lower)?;
    (inside.is_empty() && !plugin_known(id)).then(|| Excluded {
        pattern: format!("plugin-data/{id}/"),
        reason: "OAIY does not know how to back up this plugin's data safely: only plugins it knows are backed up, file by file.".to_string(),
        redo: Some("Set the plugin up again on the new computer.".to_string()),
    })
}

/// The files of a known plugin that are not on its list are left out, and listed once.
fn plugin_other_files(rel: &str) -> Option<Excluded> {
    let lower = rel.to_lowercase();
    let (id, inside) = plugin_parts(&lower)?;
    (plugin_known(id) && !inside.is_empty()).then(|| Excluded {
        pattern: format!("plugin-data/{id}/**"),
        reason: "Only this plugin's settings file is backed up: the rest of its data holds pairings, sealed values, queues and other state that belongs to this computer.".to_string(),
        redo: Some("Pair the plugin's devices and set its PIN again.".to_string()),
    })
}

fn note_unknown(excluded: &mut BTreeMap<String, Excluded>, rel: &str, dir: bool) {
    let top = rel.split('/').next().unwrap_or(rel);
    let pattern = if dir || rel.contains('/') { format!("{top}/") } else { top.to_string() };
    excluded.entry(pattern.clone()).or_insert_with(|| Excluded {
        pattern,
        reason: "Not recognised as personal data, so it is not backed up.".to_string(),
        redo: None,
    });
}

/// A template file that is the copy OAIY seeded: its `.<name>.seed` snapshot is the same bytes.
fn template_is_unedited(abs: &Path) -> bool {
    let Some(name) = abs.file_name().and_then(|n| n.to_str()) else { return false };
    let Some(dir) = abs.parent() else { return false };
    match (std::fs::read(abs), std::fs::read(dir.join(format!(".{name}.seed")))) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
