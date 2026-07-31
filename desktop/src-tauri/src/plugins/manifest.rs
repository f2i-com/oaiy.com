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

use serde::{Deserialize, Serialize};
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
/// differences are additive, so all three parse.
pub const SUPPORTED_SCHEMA_VERSIONS: std::ops::RangeInclusive<u32> = 1..=3;

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
    /// Unknown keys (`ui`, `data`, `serviceDefinitions`, …) are retained rather
    /// than rejected: the manifest schema is additive within a major, so a
    /// newer plugin must not be refused for carrying a field this host has no
    /// opinion about.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_schema_version() -> u32 {
    1
}

impl PluginManifest {
    /// Read and validate `<dir>/manifest.json`.
    pub fn load(dir: &Path) -> Result<Self, ManifestError> {
        let path = dir.join("manifest.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| ManifestError::Unreadable(e.to_string()))?;
        let manifest: PluginManifest =
            serde_json::from_str(&raw).map_err(|e| ManifestError::Malformed(e.to_string()))?;
        manifest.validate(dir)?;
        Ok(manifest)
    }

    fn validate(&self, dir: &Path) -> Result<(), ManifestError> {
        if !SUPPORTED_SCHEMA_VERSIONS.contains(&self.schema_version) {
            return Err(ManifestError::Unsupported(format!(
                "manifest schemaVersion {} (this host reads {}-{})",
                self.schema_version,
                SUPPORTED_SCHEMA_VERSIONS.start(),
                SUPPORTED_SCHEMA_VERSIONS.end()
            )));
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
