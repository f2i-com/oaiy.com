//! Service definitions a plugin contributes.
//!
//! A plugin can ship more than connectors: `manifest.serviceDefinitions[]` points
//! at JSON files describing an INVOCABLE ACTION SURFACE — named actions with a
//! title, an input schema, declared side effects and a timeout. The Aokie plugin
//! contributes `aokie.phone` (dial, speak, hang up, send/read SMS, status).
//!
//! Every action's transport is `plugin-command`, i.e. it resolves to one of the
//! plugin's own connector commands. So a definition is a *documented, schema'd
//! facade* over the command surface: the same call a flow could make by hand,
//! but discoverable, named, and carrying the metadata a UI needs to render it.
//! Invoking one therefore goes through the ordinary gated connector path — it
//! grants nothing extra.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::manifest::PluginManifest;

/// How an action reaches its implementation. Only `plugin-command` is supported;
/// anything else is retained so it can be SHOWN but refused on invoke, rather
/// than making the whole definition unloadable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transport {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DefinitionAction {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// e.g. `none` | `external-write` — lets a UI warn before a call is placed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side_effects: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    pub transport: Transport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceDefinition {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub actions: Vec<DefinitionAction>,
    /// Which plugin contributed this. Set by the loader, never read from the
    /// file — provenance must not be something a package can claim for itself.
    #[serde(default)]
    pub plugin_id: String,
}

impl ServiceDefinition {
    pub fn action(&self, id: &str) -> Option<&DefinitionAction> {
        self.actions.iter().find(|a| a.id == id)
    }
}

/// A declared `serviceDefinitions[]` entry.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DefinitionRef {
    definition_file: String,
}

/// Load every definition a plugin contributes, from its manifest + directory.
///
/// Never fails the plugin: a bad definition file is skipped (the plugin still
/// works, it just contributes less), because refusing to load a working phone
/// bridge over a typo in an optional catalog entry would be the wrong trade.
pub fn load_for_plugin(dir: &Path, manifest: &PluginManifest) -> Vec<ServiceDefinition> {
    let Some(refs) = manifest.extra.get("serviceDefinitions") else {
        return Vec::new();
    };
    let Ok(refs) = serde_json::from_value::<Vec<DefinitionRef>>(refs.clone()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for r in refs {
        let rel = r.definition_file.replace('\\', "/");
        // The path comes from the package, so it must not escape it.
        if rel.starts_with('/') || rel.split('/').any(|s| s == ".." || s == ".") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(dir.join(&rel)) else { continue };
        let Ok(mut def) = serde_json::from_str::<ServiceDefinition>(&raw) else { continue };
        if def.id.trim().is_empty() {
            continue;
        }
        def.plugin_id = manifest.id.clone();
        out.push(def);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with(refs: serde_json::Value) -> PluginManifest {
        let mut m: PluginManifest = serde_json::from_str(
            r#"{"schemaVersion":1,"id":"aokie","name":"A","version":"1","pluginApiVersion":1,
                "entry":{"kind":"process","command":"x"}}"#,
        )
        .unwrap();
        m.extra.insert("serviceDefinitions".into(), refs);
        m
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("oaiy-defs-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(p.join("definitions")).unwrap();
        p
    }

    #[test]
    fn loads_a_definition_and_stamps_its_plugin() {
        let dir = tmpdir("ok");
        std::fs::write(
            dir.join("definitions/phone.json"),
            r#"{"id":"aokie.phone","name":"Aokie Phone","category":"Telephony","actions":[
                 {"id":"call.dial","title":"Place a call","sideEffects":"external-write",
                  "transport":{"kind":"plugin-command","command":"call.dial"}}]}"#,
        )
        .unwrap();
        let defs = load_for_plugin(&dir, &manifest_with(serde_json::json!([{"definitionFile":"definitions/phone.json"}])));
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].id, "aokie.phone");
        // Provenance is stamped by the loader, not taken from the file.
        assert_eq!(defs[0].plugin_id, "aokie");
        let a = defs[0].action("call.dial").unwrap();
        assert_eq!(a.transport.command.as_deref(), Some("call.dial"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_definition_cannot_claim_another_plugins_provenance() {
        let dir = tmpdir("prov");
        std::fs::write(
            dir.join("definitions/phone.json"),
            r#"{"id":"x","name":"X","pluginId":"someone-else","actions":[]}"#,
        )
        .unwrap();
        let defs = load_for_plugin(&dir, &manifest_with(serde_json::json!([{"definitionFile":"definitions/phone.json"}])));
        assert_eq!(defs[0].plugin_id, "aokie", "the loader overwrites any claimed pluginId");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn paths_that_escape_the_package_are_skipped() {
        let dir = tmpdir("escape");
        let defs = load_for_plugin(
            &dir,
            &manifest_with(serde_json::json!([{"definitionFile":"../../../secrets.json"}])),
        );
        assert!(defs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_definition_is_skipped_not_fatal() {
        let dir = tmpdir("broken");
        std::fs::write(dir.join("definitions/bad.json"), b"{ not json").unwrap();
        std::fs::write(
            dir.join("definitions/good.json"),
            r#"{"id":"ok","name":"OK","actions":[]}"#,
        )
        .unwrap();
        let defs = load_for_plugin(
            &dir,
            &manifest_with(serde_json::json!([
                {"definitionFile":"definitions/bad.json"},
                {"definitionFile":"definitions/good.json"}
            ])),
        );
        assert_eq!(defs.len(), 1, "the good one still loads");
        assert_eq!(defs[0].id, "ok");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_declared_definitions_is_simply_empty() {
        let dir = tmpdir("none");
        let m: PluginManifest = serde_json::from_str(
            r#"{"schemaVersion":1,"id":"a","name":"A","version":"1","pluginApiVersion":1,
                "entry":{"kind":"process","command":"x"}}"#,
        )
        .unwrap();
        assert!(load_for_plugin(&dir, &m).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
