//! Tests of the classification table (`table.json`): that it holds together, that it says what the tests
//! and the docs say it does, and, above all, that a store that is added to OAIY without being classified
//! makes a test fail, so that it cannot become restorable by accident.
//!
//! The scanners read the real sources: the desktop's own (every name it uses for a file or a folder under
//! its data folder) and the Agent's page (every name it uses in its browser storage), and check each
//! against the table's `scan` section. They read text, so they are heuristics, and they say so: what they
//! cannot see (a name built at run time) is caught by the fixtures instead, which list a used installation's
//! data folder and the Agent's storage and check the table's answer for each line.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::review::RestoreClass;
use super::table::{filter_json, merge_into_local, splice_generated, table, union_lines, Class, Table, ValueType, Why};

const TABLE_JSON: &str = include_str!("table.json");
const DATA_DIR_LISTING: &str = include_str!("testdata/data-dir-listing.txt");
const AGENT_STORAGE_LISTING: &str = include_str!("testdata/agent-storage-listing.txt");
const AGENT_SETTINGS_EXPORT: &str = include_str!("testdata/agent-settings-export.json");
const AOKIE_SCHEMA: &str = include_str!("testdata/aokie-settings-schema.v1.json");

/// Where Aokie's shared settings schema was copied from, and what it must still be.
const AOKIE_SCHEMA_PROVENANCE: &str = "copied byte for byte from aokie.com docs/contracts/aokie-settings-schema.v1.json at commit cb298dd0618c652363776d1d87ea9909dd850263 (a copy also lives in that repository and its SETTING_SPECS table is test-locked to it)";
const AOKIE_SCHEMA_SHA256_LF: &str = "329c872f8ad51200452deef52f7bacde2b1b73660841cea405cd01caf4738530";

fn lf(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---- the table holds together -----------------------------------------------------------------------

#[test]
fn the_table_loads_and_holds_together() {
    let parsed = Table::parse(TABLE_JSON).unwrap_or_else(|e| panic!("{e}"));
    assert!(parsed.problems().is_empty(), "{:?}", parsed.problems());
    // Every kind that can be ticked is used by something, so no tick is a switch that does nothing.
    let mut used: BTreeSet<RestoreClass> = BTreeSet::new();
    for row in parsed.desktop.iter().chain(&parsed.agent) {
        used.extend(row.tick);
    }
    for kt in parsed.key_tables.values() {
        used.extend(kt.keys.iter().filter_map(|k| k.tick));
    }
    for class in RestoreClass::ALL {
        assert!(used.contains(&class), "{} is ticked by nothing in the table", class.id());
    }
    // A row that runs things says what it is, and one that leaves something out says why.
    for row in parsed.desktop.iter().chain(&parsed.agent) {
        assert!(!row.what.trim().is_empty(), "{}", row.id);
        if row.class == Class::Excluded {
            assert!(row.reason.len() > 10, "{} says why", row.id);
        }
    }
}

#[test]
fn a_table_that_does_not_hold_together_does_not_load() {
    let broken = |edit: &dyn Fn(&mut Value)| {
        let mut v: Value = serde_json::from_str(TABLE_JSON).unwrap();
        edit(&mut v);
        Table::parse(&v.to_string()).err()
    };
    assert!(broken(&|v| v["desktop"][0]["class"] = json!("maybe")).is_some(), "an unknown class");
    assert!(broken(&|v| v["desktop"][0]["tick"] = json!("settings")).is_some(), "an excluded row cannot be ticked");
    assert!(broken(&|v| v["desktop"][0]["reason"] = json!("")).is_some(), "an excluded row says why");
    assert!(broken(&|v| v["desktop"][0]["id"] = json!("callers")).is_some(), "an id used twice");
    assert!(broken(&|v| v["desktop"][0]["frobnicate"] = json!(1)).is_some(), "a field this version does not know");
    let runs = TABLE_JSON.find("\"class\": \"runs\"").expect("a row that runs");
    assert!(runs > 0);
    assert!(broken(&|v| {
        let rows = v["desktop"].as_array_mut().unwrap();
        let row = rows.iter_mut().find(|r| r["id"] == "callers").unwrap();
        row.as_object_mut().unwrap().remove("tick");
    })
    .is_some(), "a row that runs things needs a tick");
    assert!(broken(&|v| v["desktop"][0]["merge"] = json!("clobber")).is_some(), "an unknown merge");
    assert!(broken(&|v| {
        let keys = v["keyTables"]["plugin.aokie"]["keys"].as_array_mut().unwrap();
        keys.retain(|k| k["path"] != "settings");
    })
    .is_some(), "a key that comes back needs a container that comes back");
}

// ---- globs ------------------------------------------------------------------------------------------

#[test]
fn globs_match_the_way_the_table_means_them() {
    let g = |pattern: &str, path: &str, fold: bool| {
        let row = Table::parse(&TABLE_JSON.replace("\"paths\": [\"link/account.json\"]", &format!("\"paths\": [{}]", serde_json::to_string(pattern).unwrap()))).unwrap();
        row.desktop[0].matches(path, fold)
    };
    assert!(g("dir/**", "dir", true) && g("dir/**", "dir/a", true) && g("dir/**", "dir/a/b/c", true));
    assert!(!g("dir/**", "dirx/a", true) && !g("dir/**", "other/dir/a", true));
    assert!(g("**/*.key", "a.key", true) && g("**/*.key", "x/y/z.KEY", true) && !g("**/*.key", "a.keys", true));
    assert!(g("voices/*.wav", "voices/a.wav", true) && !g("voices/*.wav", "voices/deeper/a.wav", true));
    assert!(g("opfs/*/.backup-*/**", "opfs/front-desk/.backup-2026-09-29/sessions/index.json", false));
    assert!(!g("opfs/*/.backup-*/**", "opfs/front-desk/backup-2026/x", false));
    assert!(g("a/b.json", "A/B.JSON", true) && !g("a/b.json", "A/B.JSON", false), "case counts only where the names are case sensitive");
    assert!(!g("callers.json", "callers.json/x", true) && !g("callers.json", "x/callers.json", true));
    // A folder is walked when something that comes back could be in it.
    let t = table();
    assert!(t.desktop_holds_under("plugin-data") && t.desktop_holds_under("plugin-data/aokie") && t.desktop_holds_under("flows") && t.desktop_holds_under("ai"));
    assert!(!t.desktop_holds_under("plugin-data/other") && !t.desktop_holds_under("models") && !t.desktop_holds_under("ai/codex-home") && !t.desktop_holds_under("callers.json"));
}

// ---- the fixtures: a used installation, and the Agent's storage --------------------------------------

fn listing(text: &str) -> Vec<(String, String)> {
    lf(text).lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| {
        let (path, expected) = l.split_once('\t').unwrap_or_else(|| panic!("a line of the listing has a path, a tab and an answer: {l:?}"));
        (path.to_string(), expected.trim().to_string())
    }).collect()
}

#[test]
fn a_used_installations_data_folder_is_classified_as_the_fixture_says() {
    let t = table();
    let mut wrong = Vec::new();
    for (path, expected) in listing(DATA_DIR_LISTING) {
        let got = t.desktop_row(&path, true).map(|r| r.class.id()).unwrap_or("unknown");
        if got != expected {
            wrong.push(format!("{path}: the fixture says {expected}, the table says {got}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    // The provider list is left out of a backup unless the keys box is ticked, and is classified as one that acts once it is in.
    assert_eq!(t.desktop_row("ai/providers.json", false).map(|r| r.class), Some(Class::Excluded));
    assert_eq!(t.desktop_row("ai/providers.json", true).map(|r| r.class), Some(Class::Runs));
}

#[test]
fn the_agents_storage_is_classified_as_the_fixture_says() {
    let t = table();
    let mut wrong = Vec::new();
    for (name, expected) in listing(AGENT_STORAGE_LISTING) {
        let got = t.agent_row(&name).map(|r| r.class.id()).unwrap_or("unknown");
        if got != expected {
            wrong.push(format!("{name}: the fixture says {expected}, the table says {got}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    // What the phone's agents are told, what they say to callers and who is texted or called is never data.
    for name in ["opfs/front-desk/files/brief.md", "opfs/front-desk/files/knowledge/prices.md", "opfs/front-desk/callers.json", "opfs/front-desk/outreach/out-x.json", "opfs/front-desk/outreach/index.json", "idb/settings.json"] {
        assert_eq!(t.agent_row(name).map(|r| r.class), Some(Class::Runs), "{name}");
    }
    let dnc = t.agent_row("opfs/front-desk/outreach/do-not-contact.json").unwrap();
    assert_eq!((dnc.class, dnc.merge.as_deref()), (Class::Data, Some("union")), "the do-not-contact list can only grow");
    assert_eq!(t.agent_row("opfs/front-desk/callbacks.json").map(|r| r.class), Some(Class::Excluded), "missed calls waiting to be rung back never come back");
}

#[test]
fn every_key_of_the_agents_settings_export_is_classified() {
    let export: Value = serde_json::from_str(AGENT_SETTINGS_EXPORT).unwrap();
    let keys = table().key_table("agent.settings").unwrap();
    let all = filter_json(keys, &export, &|_| true);
    let unknown: Vec<&str> = all.left.iter().filter(|l| l.why == Why::Unknown).map(|l| l.path.as_str()).collect();
    assert!(unknown.is_empty(), "these keys of the Agent's settings are not in the table: {unknown:?}");
    // With nothing ticked nothing in the Agent's settings comes back: every one of them is read by the Agent or by a call
    // (the audit found none that only decides how something is shown).
    let none = filter_json(keys, &export, &|row| row.class == Class::Data);
    let kept: Vec<&str> = none.kept.iter().map(|k| k.path.as_str()).collect();
    assert!(kept.is_empty(), "{kept:?}");
    assert!(none.value.get("providers").is_none() || none.value["providers"].as_array().is_some_and(|p| p.is_empty()));
    assert!(none.value["messages"].get("instructions").is_none() && none.value["media"].get("baseUrl").is_none() && none.value.get("gate").is_none_or(|g| g.as_object().is_some_and(|o| o.is_empty())));
    // Whatever is ticked, an address that was read from a service and a project that was open on another computer stay out.
    let everything = filter_json(keys, &export, &|_| true);
    assert!(everything.value["media"].get("endpoints").is_none() && everything.value.get("lastProjectId").is_none() && everything.value["providers"][0].get("detectedContext").is_none());
    assert_eq!(everything.value["providers"][0]["baseUrl"], "https://api.openai.com/v1");
}

#[test]
fn the_vendored_aokie_schema_is_the_one_the_table_was_written_against() {
    use sha2::{Digest, Sha256};
    let digest = super::hex(&Sha256::digest(lf(AOKIE_SCHEMA).as_bytes()));
    assert_eq!(digest, AOKIE_SCHEMA_SHA256_LF, "the schema fixture changed: {AOKIE_SCHEMA_PROVENANCE}");
}

#[test]
fn every_setting_in_aokies_schema_is_classified_and_typed_as_the_schema_says() {
    let schema: Value = serde_json::from_str(AOKIE_SCHEMA).unwrap();
    let keys = table().key_table("plugin.aokie").unwrap();
    let mut in_schema: BTreeSet<String> = BTreeSet::new();
    for spec in schema["settings"].as_array().expect("the schema lists its settings") {
        let key = spec["key"].as_str().unwrap();
        in_schema.insert(format!("settings.{key}"));
        let row = keys.row(&format!("settings.{key}")).unwrap_or_else(|| panic!("settings.{key} is in Aokie's schema and not in the table: decide what a restore does with it"));
        let kind = spec["type"].as_str().unwrap();
        if kind == "endpointUrl" {
            assert_eq!(row.class, Class::Excluded, "settings.{key} is an address that audio or words go to: it is never restored");
            continue;
        }
        if row.class == Class::Excluded {
            continue;
        }
        match (kind, row.ty.as_ref().unwrap()) {
            ("bool", ValueType::Bool) => {}
            ("int", ValueType::Int { min, max }) => assert_eq!((*min, *max), (spec["min"].as_i64().unwrap(), spec["max"].as_i64().unwrap()), "settings.{key}"),
            ("string", ValueType::Str { max_chars }) => assert_eq!(*max_chars as u64, spec["maxChars"].as_u64().unwrap(), "settings.{key}"),
            ("enum", ValueType::Enum(options)) => {
                let theirs: Vec<&str> = spec["options"].as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
                assert_eq!(options.iter().map(String::as_str).collect::<Vec<_>>(), theirs, "settings.{key}");
            }
            (kind, ty) => panic!("settings.{key}: the schema says {kind}, the table says {ty:?}"),
        }
    }
    // What the table lists beyond the schema is what Aokie's settings file holds besides it, and one setting Aokie
    // keeps outside its schema (its consent posture): each listed on purpose.
    let beyond: BTreeSet<&str> = ["settings", "settings.consentMode", "preferredDongle", "pairedDevices", "configVersion", "dialLedger"].into_iter().collect();
    for row in &keys.keys {
        assert!(in_schema.contains(&row.path) || beyond.contains(row.path.as_str()), "{} is in the table and not in the schema: say why", row.path);
    }
    // Everything that reads callers' audio or words to somewhere else, or grants access, is excluded.
    for key in ["aiEndpoint", "sttEndpoint", "ttsEndpoint", "audioTranscriptEndpoint", "realtimeVoiceEndpoint", "realtimeVoiceDestination", "consentMode", "outboundEnabled", "managerNumbers", "managerPin", "acceptPattern", "maxDailyDials", "legacyPairingPin"] {
        assert_eq!(keys.row(&format!("settings.{key}")).unwrap().class, Class::Excluded, "settings.{key}");
    }
    // What a model reads as instructions, or that is spoken to callers, needs a tick.
    for key in ["persona", "greeting", "screenMessage", "blockedMessage", "autoAnswer", "aiReceptionist", "blockedNumbers"] {
        assert_eq!(keys.row(&format!("settings.{key}")).unwrap().class, Class::Runs, "settings.{key}");
    }
    // A key that is not in the table is not restored, and is said not to be.
    let unknown = filter_json(keys, &json!({ "settings": { "brandNewSwitch": true, "greeting": "hi" } }), &|_| true);
    assert!(unknown.value["settings"].get("brandNewSwitch").is_none());
    assert!(unknown.left.iter().any(|l| l.path == "settings.brandNewSwitch" && l.why == Why::Unknown));
}

/// The audit found that every setting of the phone plugin is call handling (what callers hear, who is answered, when a call ends):
/// no key of its table comes back without a tick. A key added to the table as data has to be argued for in the audit's terms.
#[test]
fn no_key_of_the_phone_plugins_settings_comes_back_without_a_tick() {
    let keys = table().key_table("plugin.aokie").unwrap();
    let data: Vec<&str> = keys.keys.iter().filter(|k| k.class == Class::Data).map(|k| k.path.as_str()).collect();
    assert!(data.is_empty(), "these keys of the phone plugin's settings are data: {data:?}");
    for key in ["ttsVoice", "maxSilenceSecs", "bargeSensitivity", "sttEndpointMs", "realtimeVoice", "ttsEngine", "aiModel"] {
        let row = keys.row(&format!("settings.{key}")).unwrap();
        assert_eq!((row.class, row.tick), (Class::Runs, Some(RestoreClass::Plugins)), "settings.{key}");
    }
}

// ---- filtering a document by its key table ------------------------------------------------------------

#[test]
fn a_document_is_filtered_key_by_key_and_value_by_value() {
    let keys = table().key_table("plugin.aokie").unwrap();
    let hostile = json!({
        "settings": {
            "greeting": "hello",
            "persona": "x".repeat(4001),
            "bargeSensitivity": 99999,
            "autoAnswer": "yes",
            "ttsEngine": "attacker",
            "ttsVoice": "dpapi:AAAA",
            "realtimeMaxOutputTokens": 512,
            "a.b": 1,
            "settings.persona": "smuggled",
            "nested": { "greeting": "deeper" }
        },
        "settings.greeting": "smuggled at the top",
        "pairedDevices": [{ "address": "AA" }]
    });
    let got = filter_json(keys, &hostile, &|_| true);
    assert_eq!(got.value, json!({ "settings": { "greeting": "hello", "realtimeMaxOutputTokens": 512 } }));
    let why = |path: &str| got.left.iter().find(|l| l.path == path).map(|l| l.why.clone());
    assert!(matches!(why("settings.persona"), Some(Why::BadValue(_))), "a text longer than its limit is refused");
    assert!(matches!(why("settings.bargeSensitivity"), Some(Why::BadValue(_))), "a number outside its limits is refused");
    assert!(matches!(why("settings.autoAnswer"), Some(Why::BadValue(_))), "a value of the wrong kind is refused");
    assert!(matches!(why("settings.ttsEngine"), Some(Why::BadValue(_))), "a choice that is not one of the choices is refused");
    assert!(matches!(why("settings.ttsVoice"), Some(Why::BadValue(_))), "a value that is sealed to a computer is refused");
    assert_eq!(why("settings.nested"), Some(Why::Unknown));
    assert_eq!(why("pairedDevices"), Some(Why::Excluded));
    assert_eq!(why("settings.a.b"), Some(Why::Unknown), "a key that could pass for a path is never looked up");
    // A smuggled key cannot pose as a listed one: what is at the top under a dotted name is not a setting.
    assert!(got.value.get("settings.greeting").is_none());
    // A document that is not an object says so.
    let odd = filter_json(keys, &json!([1, 2]), &|_| true);
    assert!(odd.kept.is_empty() && matches!(odd.left[0].why, Why::BadValue(_)));
}

#[test]
fn what_a_backup_brings_is_put_into_what_is_here_and_the_rest_stays() {
    let keys = table().key_table("plugin.aokie").unwrap();
    let here = json!({ "settings": { "managerPin": "dpapi1:LOCAL", "greeting": "mine", "aiEndpoint": "http://127.0.0.1:1/v1", "blockedNumbers": "0400 000 222\n0400 000 333" }, "configVersion": 4 });
    let theirs = json!({ "settings": { "greeting": "theirs", "managerPin": "1111", "aiEndpoint": "http://attacker.example", "blockedNumbers": "0400 000 111\n+61 400 000 333" }, "configVersion": 99 });
    let got = merge_into_local(Some(&here), &filter_json(keys, &theirs, &|_| true));
    assert_eq!(got["settings"]["greeting"], "theirs");
    assert_eq!(got["settings"]["managerPin"], "dpapi1:LOCAL");
    assert_eq!(got["settings"]["aiEndpoint"], "http://127.0.0.1:1/v1");
    assert_eq!(got["configVersion"], 4);
    let blocked: Vec<&str> = got["settings"]["blockedNumbers"].as_str().unwrap().lines().collect();
    assert_eq!(blocked, ["0400 000 222", "0400 000 333", "0400 000 111"], "the same number written another way is not added twice, and none of ours is lost");
    // With no file here there is nothing to keep: the result is only what the table let through.
    let bare = merge_into_local(None, &filter_json(keys, &theirs, &|_| true));
    assert_eq!(bare, json!({ "settings": { "greeting": "theirs", "blockedNumbers": "0400 000 111\n+61 400 000 333" } }));
}

#[test]
fn a_list_kept_one_a_line_only_grows() {
    assert_eq!(union_lines("a\nb", "c\nA", 100), "a\nb\nc");
    assert_eq!(union_lines("", "0400 111 222", 100), "0400 111 222");
    assert_eq!(union_lines("0400 111 222", "", 100), "0400 111 222");
    assert_eq!(union_lines("+61400111222", "0400 111 222", 100), "+61400111222", "a number is the same number however it is written");
    // What is here is never cut, even when it fills the limit; what is added stops at it.
    let long = "1".repeat(10);
    assert_eq!(union_lines(&long, &"2".repeat(10), 15), long);
}

// ---- reading the sources ----------------------------------------------------------------------------

/// The text of each `.rs` file under `dir` (relative name and text), without the test modules and without the
/// `backup` module itself (it is the thing being checked).
fn rust_sources(dir: &Path) -> Vec<(String, String)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            if path.is_dir() {
                if rel != "backup" && !rel.ends_with("testdata") {
                    walk(root, &path, out);
                }
            } else if rel.ends_with(".rs") && !rel.ends_with("/tests.rs") && rel != "tests.rs" && !rel.contains("/tests/") {
                out.push((rel, lf(&std::fs::read_to_string(&path).unwrap())));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out
}

/// The part of a source file that is not its test module.
fn production_part(source: &str) -> &str {
    let mut from = 0;
    while let Some(at) = source[from..].find("#[cfg(test)]") {
        let start = from + at;
        let rest = source[start + "#[cfg(test)]".len()..].trim_start();
        if rest.starts_with("mod ") || rest.starts_with("pub mod ") || rest.starts_with("pub(crate) mod ") {
            return &source[..start];
        }
        from = start + 1;
    }
    source
}

/// Every plain string literal of some Rust source, and whether it is the argument of a `.join(` call.
fn rust_literals(source: &str) -> Vec<(String, bool)> {
    let b: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && b.get(i + 1) == Some(&'*') {
            let mut depth = 1;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else if (c == 'r' || (c == 'b' && b.get(i + 1) == Some(&'r'))) && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_')) && {
            let mut j = i + if c == 'b' { 2 } else { 1 };
            while b.get(j) == Some(&'#') {
                j += 1;
            }
            b.get(j) == Some(&'"')
        } {
            let mut j = i + if c == 'b' { 2 } else { 1 };
            let mut hashes = 0;
            while b.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            j += 1;
            loop {
                if j >= b.len() {
                    break;
                }
                if b[j] == '"' && (0..hashes).all(|k| b.get(j + 1 + k) == Some(&'#')) {
                    j += 1 + hashes;
                    break;
                }
                j += 1;
            }
            i = j;
        } else if c == '"' {
            let start = i;
            let mut j = i + 1;
            let mut text = String::new();
            let mut plain = true;
            while j < b.len() && b[j] != '"' {
                if b[j] == '\\' {
                    plain = false;
                    j += 2;
                    continue;
                }
                text.push(b[j]);
                j += 1;
            }
            let before: String = b[..start].iter().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect();
            if plain {
                out.push((text, before.trim_end().ends_with(".join(")));
            }
            i = j + 1;
        } else if c == '\'' {
            // A character, or a lifetime.
            if b.get(i + 1) == Some(&'\\') {
                i += 2;
                while i < b.len() && b[i] != '\'' {
                    i += 1;
                }
                i += 1;
            } else if b.get(i + 2) == Some(&'\'') {
                i += 3;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

const STORE_EXTENSIONS: &[&str] = &["json", "jsonl", "md", "txt", "key", "pem", "db", "sqlite", "log", "bak", "tmp", "seed", "wav", "mp3", "dpapi", "sealed", "toml", "lock", "cfg", "yaml", "yml"];

/// Whether a literal could be the name of a file or a folder that something is kept in.
fn store_like(literal: &str, after_join: bool) -> bool {
    let plain = !literal.is_empty() && literal.len() <= 80 && literal.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'));
    if !plain || literal.starts_with("../") || !literal.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    // A bare extension (".json", ".tmp") is a piece of a name, not a name.
    if literal.strip_prefix('.').is_some_and(|rest| STORE_EXTENSIONS.contains(&rest) || rest.starts_with("json.")) {
        return false;
    }
    let has_extension = literal.rsplit_once('.').is_some_and(|(_, e)| STORE_EXTENSIONS.contains(&e));
    has_extension || after_join
}

/// Whether the mapping given for a name is one a reader can follow: the id of a row of the table, or a statement
/// that it is not a store, with why.
fn check_mapping(t: &Table, name: &str, value: &str) -> Option<String> {
    if let Some(reason) = value.strip_prefix("not-a-store:") {
        return (reason.trim().len() < 8).then(|| format!("{name}: say why it is not a store"));
    }
    let known = t.desktop.iter().chain(&t.agent).any(|r| r.id == value) || t.key_tables.contains_key(value) || t.key_tables.values().any(|k| k.keys.iter().any(|key| key.path == value));
    (!known).then(|| format!("{name}: \"{value}\" is not a row of the table"))
}

#[test]
fn every_name_the_desktop_uses_for_a_store_is_classified_or_declared_not_a_store() {
    let t = table();
    let mut missing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (file, source) in rust_sources(&manifest_dir().join("src")) {
        for (literal, after_join) in rust_literals(production_part(&source)) {
            if !store_like(&literal, after_join) {
                continue;
            }
            seen.insert(literal.clone());
            if !t.scan.desktop_names.contains_key(&literal) {
                missing.entry(literal).or_default().insert(file.clone());
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these names are used by the desktop's code for a file or a folder and are not in table.json's scan.desktopNames (map each to the row that classifies it, or to \"not-a-store: why\"): {missing:#?}"
    );
    // The other way: a name in the table that no code uses any more is a stale entry (or the scanner is blind).
    let stale: Vec<&String> = t.scan.desktop_names.keys().filter(|n| !seen.contains(*n)).collect();
    assert!(stale.is_empty(), "these names are in scan.desktopNames and no source uses them: {stale:?}");
    let bad: Vec<String> = t.scan.desktop_names.iter().filter_map(|(n, v)| check_mapping(t, n, v)).collect();
    assert!(bad.is_empty(), "{bad:#?}");
}

// ---- the Agent's page ---------------------------------------------------------------------------------

fn app_sources() -> Vec<(String, String)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            if path.is_dir() {
                if !matches!(rel.as_str(), "node_modules" | "softn/knowledge") {
                    walk(root, &path, out);
                }
            } else if (rel.ends_with(".ts") || rel.ends_with(".tsx")) && !rel.ends_with(".test.ts") && !rel.ends_with(".test.tsx") && !rel.ends_with(".d.ts") {
                out.push((rel, lf(&std::fs::read_to_string(&path).unwrap())));
            }
        }
    }
    let root = manifest_dir().join("../../../app/src");
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out
}

/// Every plain string literal of some TypeScript source, with the text just before it on the same line.
fn ts_literals(source: &str) -> Vec<(String, String)> {
    let b: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && b.get(i + 1) == Some(&'*') {
            i += 2;
            while i < b.len() && !(b[i] == '*' && b.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i += 2;
        } else if c == '\'' || c == '"' || c == '`' {
            let quote = c;
            let start = i;
            let mut j = i + 1;
            let mut text = String::new();
            let mut plain = true;
            while j < b.len() && b[j] != quote && b[j] != '\n' {
                if b[j] == '\\' {
                    plain = false;
                    j += 2;
                    continue;
                }
                if quote == '`' && b[j] == '$' && b.get(j + 1) == Some(&'{') {
                    plain = false;
                }
                text.push(b[j]);
                j += 1;
            }
            let line_start = b[..start].iter().rposition(|x| *x == '\n').map(|p| p + 1).unwrap_or(0);
            let before: String = b[line_start..start].iter().collect();
            if plain {
                out.push((text, before));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn every_name_the_agents_page_stores_under_is_classified_or_declared_not_a_store() {
    let t = table();
    let sources = app_sources();
    // 1. Only the modules the table lists touch the browser's storage: the private file system, IndexedDB, or a folder the person picks.
    let mut touching: BTreeSet<String> = BTreeSet::new();
    for (file, source) in &sources {
        if ["navigator.storage", "indexedDB", "createWritable", "getFileHandle", "getDirectoryHandle", "showDirectoryPicker", "showSaveFilePicker"].iter().any(|p| source.contains(p)) {
            touching.insert(file.clone());
        }
    }
    let listed: BTreeSet<String> = t.scan.agent_modules.keys().cloned().collect();
    assert_eq!(touching, listed, "the modules of the Agent's page that touch browser storage must be exactly the ones in scan.agentModules (a module that starts to store something is a store to classify)");
    // The two that are listed and do not store anything must go on not doing so.
    let not_storing: [(&str, &[&str]); 2] = [("main.ts", &["createWritable", "getFileHandle", "getDirectoryHandle", "indexedDB", "getDirectory("]), ("vfs/transfer.ts", &["navigator.storage", "indexedDB", "getDirectory("])];
    for (file, forbidden) in not_storing {
        let source = &sources.iter().find(|(f, _)| f == file).unwrap_or_else(|| panic!("{file} is gone: update scan.agentModules")).1;
        for word in forbidden {
            assert!(!source.contains(word), "{file} is listed as not storing anything in the Agent's storage, and now uses {word}: classify what it stores");
        }
    }
    // 2. Every name the modules that store use is classified.
    let storing = ["desktop/backup.ts", "settings.ts", "vfs/projects.ts"];
    let calls = ["get", "put", "getDirectoryHandle", "getFileHandle", "removeEntry", "readJson", "writeBytes", "dirAt"];
    let mut missing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (file, source) in &sources {
        if !storing.contains(&file.as_str()) {
            continue;
        }
        for (literal, before) in ts_literals(source) {
            let plain = !literal.is_empty() && literal.len() <= 80 && literal.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'));
            let has_extension = literal.rsplit_once('.').is_some_and(|(_, e)| ["json", "jsonl", "md"].contains(&e));
            // The first argument of a call that stores or reads by name: `get('gate')`, `dirAt(this.dir, 'sessions', true)`.
            let called = before.rfind('(').is_some_and(|open| {
                let mut name = &before[..open];
                if name.ends_with('>') {
                    name = &name[..name.rfind('<').unwrap_or(name.len())];
                }
                let ident: String = name.chars().rev().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect::<Vec<_>>().into_iter().rev().collect();
                calls.contains(&ident.as_str()) && !before[open..].contains(')')
            });
            // (A bare extension, ".json", is a piece of a name and not a name.)
            let bare_extension = literal.strip_prefix('.').is_some_and(|rest| STORE_EXTENSIONS.contains(&rest));
            if !plain || bare_extension || !(called || has_extension) || literal.starts_with("../") || literal.starts_with("./") || literal.starts_with("http") {
                continue;
            }
            seen.insert(literal.clone());
            if !t.scan.agent_names.contains_key(&literal) {
                missing.entry(literal).or_default().insert(file.clone());
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these names are used by the Agent's page to store something and are not in table.json's scan.agentNames (map each to the row or key that classifies it, or to \"not-a-store: why\"): {missing:#?}"
    );
    let stale: Vec<&String> = t.scan.agent_names.keys().filter(|n| !seen.contains(*n)).collect();
    assert!(stale.is_empty(), "these names are in scan.agentNames and no source of the page uses them: {stale:?}");
    let bad: Vec<String> = t.scan.agent_names.iter().filter_map(|(n, v)| check_mapping(t, n, v)).collect();
    assert!(bad.is_empty(), "{bad:#?}");
    // The page's own layout is the one the fixture lists.
    let layout = std::fs::read_to_string(manifest_dir().join("../../../app/src/vfs/projects.ts")).map(|s| lf(&s)).unwrap();
    for part in ["/projects/<id>/project.json", "/projects/<id>/files/...", "/projects/<id>/chat.json"] {
        assert!(layout.contains(part), "vfs/projects.ts no longer says it keeps {part}: check the Agent's storage fixture");
    }
}

// ---- the audit: data is what no code turns into behaviour -----------------------------------------------

fn repo_root() -> PathBuf {
    manifest_dir().join("../../..")
}

/// Every place the table says something is read: `<path from the repository root>#<a name that is in that file>`.
fn all_readers(t: &Table) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for row in t.desktop.iter().chain(&t.agent) {
        out.extend(row.readers.iter().map(|r| (format!("row {}", row.id), r.clone())));
    }
    for table in t.key_tables.values() {
        for key in &table.keys {
            out.extend(key.readers.iter().map(|r| (format!("key {}: {}", table.name, key.path), r.clone())));
        }
    }
    out.extend(t.audit.iter().flat_map(|a| a.readers.iter().map(|r| (format!("audit note {}", a.item), r.clone()))));
    out
}

/// An annotation cannot go stale: each place it names is a file that is there and holds the name.
#[test]
fn every_place_the_audit_names_is_a_file_that_holds_the_name() {
    let t = table();
    let readers = all_readers(t);
    assert!(readers.len() >= 30, "the audit names what reads things: {} places", readers.len());
    let mut wrong = Vec::new();
    for (who, reader) in readers {
        let Some((path, symbol)) = reader.split_once('#') else {
            wrong.push(format!("{who}: {reader:?} is not <path>#<name>"));
            continue;
        };
        match std::fs::read_to_string(repo_root().join(path)) {
            Ok(text) if lf(&text).contains(symbol) => {}
            Ok(_) => wrong.push(format!("{who}: {path} does not hold {symbol:?}")),
            Err(_) => wrong.push(format!("{who}: there is no file {path}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The data values that exist, and nothing else: a new one is a deliberate edit of this list, made after the audit's questions
/// were asked of it (see docs/BACKUP.md, "The audit").
#[test]
fn the_values_that_are_data_are_the_ones_the_audit_found() {
    let t = table();
    let mut data: Vec<String> = t.desktop.iter().chain(&t.agent).filter(|r| r.class == Class::Data).map(|r| format!("row {}", r.id)).collect();
    for table in t.key_tables.values() {
        data.extend(table.keys.iter().filter(|k| k.class == Class::Data && !matches!(k.ty, Some(ValueType::Object | ValueType::Objects { .. }))).map(|k| format!("{}: {}", table.name, k.path)));
    }
    data.sort();
    let expected = [
        "calendar: appointments[].createdAt", "calendar: appointments[].id", "calendar: appointments[].minutes", "calendar: appointments[].source", "calendar: appointments[].start",
        "calendar: appointments[].status", "calendar: appointments[].updatedAt", "calendar: settings.horizonDays", "calendar: settings.hours", "calendar: settings.noticeMinutes",
        "calendar: settings.slotMinutes", "row agent-do-not-contact",
    ];
    assert_eq!(data, expected, "a value that is data has been added or removed: ask the audit's questions of it, and say so here");
    // None of them is text, and each of them says what reads it and that it feeds no behaviour (the loader holds that; this is the test that it does).
    for table in t.key_tables.values() {
        for key in table.keys.iter().filter(|k| k.class == Class::Data && !matches!(k.ty, Some(ValueType::Object | ValueType::Objects { .. }))) {
            assert!(!key.ty.as_ref().unwrap().carries_words(), "{}: {}", table.name, key.path);
            assert!(key.reads.as_deref().unwrap_or("").to_lowercase().contains(super::table::FEEDS_NOTHING) && !key.readers.is_empty(), "{}: {}", table.name, key.path);
        }
    }
    let dnc = t.agent.iter().find(|r| r.id == "agent-do-not-contact").unwrap();
    assert!(dnc.keyless_because.is_some() && dnc.merge.as_deref() == Some("union"), "the one data file without a key table is checked by its merge");
}

/// The loader holds the rule itself: a table in which something is data without saying what reads it, or that carries words, is
/// not a table OAIY starts with.
#[test]
fn a_table_that_calls_something_data_without_the_audit_does_not_load() {
    let broken = |edit: &dyn Fn(&mut Value)| {
        let mut v: Value = serde_json::from_str(TABLE_JSON).unwrap();
        edit(&mut v);
        Table::parse(&v.to_string()).err()
    };
    fn key<'a>(v: &'a mut Value, table: &str, path: &str) -> &'a mut Value {
        let at = v["keyTables"][table]["keys"].as_array().unwrap().iter().position(|k| k["path"] == path).unwrap();
        &mut v["keyTables"][table]["keys"][at]
    }
    let says = |e: Option<String>, what: &str| assert!(e.as_ref().is_some_and(|e| e.contains(what)), "{what:?} in {e:?}");
    says(broken(&|v| { key(v, "calendar", "settings.slotMinutes").as_object_mut().unwrap().remove("reads"); }), "does not say what reads it");
    says(broken(&|v| { key(v, "calendar", "settings.slotMinutes")["reads"] = json!("Read by the calendar."); }), "does not say what reads it");
    says(broken(&|v| { key(v, "calendar", "settings.slotMinutes")["readers"] = json!([]); }), "does not name the code that reads it");
    // The calendar hole, exactly: the business's name as data, with an annotation that says the right words.
    says(broken(&|v| {
        let k = key(v, "calendar", "settings.business");
        k["class"] = json!("data");
        k.as_object_mut().unwrap().remove("tick");
        k["reads"] = json!("Nothing reads it and it feeds no behaviour.");
        k["readers"] = json!(["platform/desktop/src-tauri/src/calendar/mod.rs#Settings"]);
    }), "can carry words");
    // Every type that can carry words is refused as data, whatever the annotation says: a string, an address, a list of strings and
    // a map of them or of scalars (a number and a switch are the values that carry none).
    for (ty, extra) in [
        ("string", json!({ "maxChars": 20 })),
        ("url", json!({ "maxChars": 200 })),
        ("strings", json!({ "maxItems": 3, "maxChars": 20 })),
        ("strings-map", json!({ "maxItems": 3, "maxChars": 20 })),
        ("scalars-map", json!({ "maxItems": 3, "maxChars": 20 })),
    ] {
        says(
            broken(&|v| {
                let k = key(v, "calendar", "settings.slotMinutes").as_object_mut().unwrap();
                for gone in ["min", "max"] {
                    k.remove(gone);
                }
                k.insert("type".into(), json!(ty));
                for (name, value) in extra.as_object().unwrap() {
                    k.insert(name.clone(), value.clone());
                }
            }),
            "can carry words",
        );
    }
    // (And the values that carry none are data when the audit is whole: the table as it is has such keys.)
    assert!(Table::parse(TABLE_JSON).is_ok());
    says(broken(&|v| { v["agent"].as_array_mut().unwrap().iter_mut().find(|r| r["id"] == "agent-do-not-contact").unwrap().as_object_mut().unwrap().remove("keylessBecause"); }), "does not say why it needs none");
    says(broken(&|v| { v["agent"].as_array_mut().unwrap().iter_mut().find(|r| r["id"] == "agent-do-not-contact").unwrap().as_object_mut().unwrap().remove("reads"); }), "does not say what reads it");
}

// ---- the documentation is generated from the table ----------------------------------------------------

fn docs_path() -> PathBuf {
    manifest_dir().join("../../../docs/BACKUP.md")
}

#[test]
fn the_backup_docs_are_generated_from_the_table() {
    let t = table();
    let path = docs_path();
    let doc = lf(&std::fs::read_to_string(&path).expect("docs/BACKUP.md"));
    let blocks = [("classification-table", t.render_table()), ("tick-kinds", t.render_kinds()), ("audit", t.render_audit())];
    let mut expected = doc.clone();
    for (name, block) in &blocks {
        expected = splice_generated(&expected, name, block).unwrap_or_else(|| panic!("docs/BACKUP.md has no <!-- BEGIN GENERATED: {name} --> ... <!-- END GENERATED: {name} --> block"));
    }
    if std::env::var("OAIY_REGENERATE_DOCS").is_ok() {
        let crlf = std::fs::read_to_string(&path).map(|s| s.contains("\r\n")).unwrap_or(false);
        std::fs::write(&path, if crlf { expected.replace('\n', "\r\n") } else { expected.clone() }).unwrap();
        return;
    }
    assert!(
        doc == expected,
        "docs/BACKUP.md does not match platform/desktop/src-tauri/src/backup/table.json. Regenerate it with: OAIY_REGENERATE_DOCS=1 cargo test --no-default-features --lib the_backup_docs_are_generated_from_the_table"
    );
    // The prose does not contradict the table on the points that matter.
    for stale in ["Nothing is ticked, so only your data comes back", "Nothing here is brought back unless you tick it"] {
        assert!(!doc.contains(stale), "{stale}");
    }
}
