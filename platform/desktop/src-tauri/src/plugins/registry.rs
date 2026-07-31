//! Plugin registry: scan the plugins root, hold state, gate connector commands.
//!
//! ```text
//!   installed ──► stopped ──► starting ──► running
//!                    ▲                       │
//!                    │                       ├──► unhealthy   (process alive, health failing)
//!                    └───────────────────────┴──► crashed      (process gone)
//!
//!   disabled  — user opt-out, or a manifest this host cannot honour.
//!               Never auto-started.
//! ```
//!
//! # Every non-running state carries a reason
//!
//! [`PluginRecord::reason`] is populated for every state that is not `Running`.
//! A plugin sitting in the list as "disabled" with no explanation is barely
//! better than one that never appeared: the user cannot tell an unsupported API
//! version from a corrupt download from their own earlier opt-out, and all three
//! have different fixes.
//!
//! # A plugin directory that fails to load is still listed
//!
//! It appears as `disabled` with the parse error attached. Skipping it would make
//! a broken plugin indistinguishable from an uninstalled one, and the natural
//! response — reinstall — does not help when the manifest is the problem.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::manifest::{ManifestError, PluginManifest};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginState {
    /// On disk, manifest valid, never started this session.
    Installed,
    Stopped,
    Starting,
    Running,
    /// Process alive but failing health probes. Deliberately distinct from
    /// `Crashed`: the process still holds its hardware (Aokie holds a USB
    /// dongle), so killing it to "fix" the state can be worse than leaving it.
    Unhealthy,
    Crashed,
    /// User opt-out, or a manifest this host cannot honour.
    Disabled,
}

impl PluginState {
    /// The `capability-manifest` unavailability reason for this state.
    ///
    /// Mapped explicitly rather than defaulted, because the codes are what a
    /// consumer branches on to decide what to tell the user — and the first cut
    /// of this reported `plugin_crashed` for a plugin that had simply never been
    /// started, which tells someone their software is broken when it is merely
    /// idle. "Start it" and "it crashed, look at the logs" are different
    /// instructions.
    ///
    /// Returns `None` for states that ARE available.
    pub fn unavailable_reason(self, user_disabled: bool) -> Option<&'static str> {
        match self {
            PluginState::Running | PluginState::Unhealthy => None,
            // Present and fine, just not started.
            PluginState::Installed | PluginState::Stopped => Some("service_stopped"),
            // Mid-start: not yet usable, and not a fault.
            PluginState::Starting => Some("service_stopped"),
            PluginState::Crashed => Some("plugin_crashed"),
            // A host refusal (bad manifest, unsupported API) is not the user's
            // opt-out, and the fixes differ.
            PluginState::Disabled if user_disabled => Some("plugin_disabled"),
            PluginState::Disabled => Some("not_installed"),
        }
    }

    /// Can a connector command be forwarded right now?
    pub fn accepts_commands(self) -> bool {
        // `Unhealthy` still accepts: health is a coarse signal, and refusing
        // every command because one probe timed out would make a slow plugin
        // unusable rather than merely slow.
        matches!(self, PluginState::Running | PluginState::Unhealthy)
    }
}

/// Why a command was refused. Maps to `protocol/v1/error.schema.json` codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateRefusal {
    /// No plugin declares this connector.
    ConnectorMissing { connector_id: String },
    /// The plugin exists but is not in a state that can serve commands.
    ConnectorUnavailable {
        connector_id: String,
        state: PluginState,
        reason: Option<String>,
    },
    /// The command is not in the manifest.
    CapabilityDenied {
        connector_id: String,
        command: String,
    },
    /// A side-effecting command arrived without an idempotency key.
    IdempotencyRequired {
        command: String,
    },
}

impl GateRefusal {
    /// The closed-taxonomy code a caller branches on.
    pub fn code(&self) -> &'static str {
        match self {
            GateRefusal::ConnectorMissing { .. } => "capability_unavailable",
            GateRefusal::ConnectorUnavailable { .. } => "capability_unavailable",
            GateRefusal::CapabilityDenied { .. } => "capability_denied",
            GateRefusal::IdempotencyRequired { .. } => "invalid_request",
        }
    }

    /// The sentence a person reads. `capability_unavailable` must be actionable —
    /// the protocol requires it — so these name what to do, not just what failed.
    pub fn message(&self) -> String {
        match self {
            GateRefusal::ConnectorMissing { connector_id } => format!(
                "No installed plugin provides the \"{connector_id}\" connector. \
                 Install it in OAIY Desktop → Plugins."
            ),
            GateRefusal::ConnectorUnavailable { connector_id, state, reason } => {
                let base = format!(
                    "The \"{connector_id}\" connector is not available: its plugin is {state:?}."
                );
                match reason {
                    Some(r) => format!("{base} {r} Start it in OAIY Desktop → Plugins."),
                    None => format!("{base} Start it in OAIY Desktop → Plugins."),
                }
            }
            GateRefusal::CapabilityDenied { connector_id, command } => format!(
                "The \"{connector_id}\" plugin does not declare the command \"{command}\", \
                 so it was refused before the plugin was contacted."
            ),
            GateRefusal::IdempotencyRequired { command } => format!(
                "\"{command}\" has side effects that must not be repeated, so it requires an \
                 idempotencyKey."
            ),
        }
    }
}

/// One plugin as the registry sees it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginRecord {
    pub id: String,
    pub state: PluginState,
    /// Populated for every state that is not `Running`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub dir: PathBuf,
    /// `None` when the manifest could not be loaded — the record still exists so
    /// the failure is visible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PluginManifest>,
    /// Capability names rewritten from a pre-OAIY spelling, for a UI nudge.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub legacy_capabilities: Vec<(String, String)>,
    /// Declared capabilities that grant nothing — a typo, or a FormLogic-era
    /// name with no OAIY equivalent.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unknown_capabilities: Vec<String>,
    /// True when the user explicitly turned it off, as opposed to the host
    /// refusing it. Both read as `Disabled`, and they are not the same problem.
    pub user_disabled: bool,
    pub restart_attempts: u32,
}

impl PluginRecord {
    fn from_manifest(dir: PathBuf, m: PluginManifest) -> Self {
        let legacy = m.legacy_capabilities();
        let unknown = m.unknown_capabilities();
        Self {
            id: m.id.clone(),
            state: PluginState::Installed,
            reason: Some("Not started yet.".into()),
            dir,
            manifest: Some(m),
            legacy_capabilities: legacy,
            unknown_capabilities: unknown,
            user_disabled: false,
            restart_attempts: 0,
        }
    }

    fn from_error(dir: PathBuf, id: String, err: &ManifestError) -> Self {
        Self {
            id,
            state: PluginState::Disabled,
            reason: Some(err.reason()),
            dir,
            manifest: None,
            legacy_capabilities: Vec::new(),
            unknown_capabilities: Vec::new(),
            user_disabled: false,
            restart_attempts: 0,
        }
    }

    /// Is this plugin usable at all? A record with no manifest never becomes
    /// runnable however many times the user presses Start.
    pub fn is_loadable(&self) -> bool {
        self.manifest.is_some()
    }
}

#[derive(Default)]
pub struct PluginRegistry {
    root: PathBuf,
    /// Plugin ids the user has explicitly turned off.
    ///
    /// Persisted, because `scan()` rebuilds every `PluginRecord` from its
    /// manifest and `from_manifest` hardcodes `user_disabled: false`. While
    /// nothing started plugins on its own that was merely forgetful; once boot
    /// autostart existed it became a policy INVERSION — every launch would
    /// start the very plugins the user had turned off, and for the phone bridge
    /// that means seizing a Bluetooth dongle they had deliberately released.
    disabled: std::collections::BTreeSet<String>,
    /// Keyed by plugin id, ordered so listings are stable rather than
    /// hash-order — a list that reshuffles between polls is unusable in a UI.
    plugins: BTreeMap<String, PluginRecord>,
}

impl PluginRegistry {
    /// `<plugins root>/disabled.json` — the user's opt-outs.
    fn disabled_path(root: &std::path::Path) -> PathBuf {
        root.join("disabled.json")
    }

    fn load_disabled(root: &std::path::Path) -> std::collections::BTreeSet<String> {
        std::fs::read_to_string(Self::disabled_path(root))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Atomic (`.tmp` + rename): a crash mid-write must not leave a truncated
    /// file that reads as "nothing is disabled" and starts everything.
    fn persist_disabled(&self) {
        let path = Self::disabled_path(&self.root);
        let write = || -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension("json.tmp");
            let body = serde_json::to_string(&self.disabled).map_err(std::io::Error::other)?;
            std::fs::write(&tmp, body)?;
            std::fs::rename(&tmp, &path)
        };
        if let Err(e) = write() {
            log::warn!("could not persist disabled plugins to {}: {e}", path.display());
        }
    }

    pub fn new(root: PathBuf) -> Self {
        Self {
            disabled: Self::load_disabled(&root),
            root,
            plugins: BTreeMap::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Rescan the plugins root.
    ///
    /// Preserves the live state of plugins already known: a rescan triggered by
    /// `GET /api/plugins` must not knock a running plugin back to `Installed`.
    /// Only genuinely new directories get a fresh record, and directories that
    /// disappeared are dropped.
    pub fn scan(&mut self) -> ScanReport {
        let mut report = ScanReport::default();
        let entries = match std::fs::read_dir(&self.root) {
            Ok(e) => e,
            Err(_) => {
                // A missing plugins root is a normal first run, not an error.
                self.plugins.retain(|_, r| r.state == PluginState::Running);
                return report;
            }
        };

        let mut seen: Vec<String> = Vec::new();
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let dir_name = entry.file_name().to_string_lossy().to_string();

            match PluginManifest::load(&dir) {
                Ok(m) => {
                    // The manifest's own id wins over the directory name, but a
                    // mismatch is worth refusing: two directories both claiming
                    // `aokie` would silently shadow each other, and which one won
                    // would depend on directory iteration order.
                    if m.id != dir_name {
                        let rec = PluginRecord::from_error(
                            dir.clone(),
                            dir_name.clone(),
                            &ManifestError::Invalid(format!(
                                "id {:?} does not match its directory name {dir_name:?}; \
                                 rename one so the plugin has a single identity",
                                m.id
                            )),
                        );
                        seen.push(dir_name.clone());
                        report.invalid += 1;
                        self.plugins.insert(dir_name, rec);
                        continue;
                    }
                    let id = m.id.clone();
                    seen.push(id.clone());
                    match self.plugins.get_mut(&id) {
                        Some(existing) if existing.is_loadable() => {
                            // Refresh the manifest but keep runtime state.
                            existing.legacy_capabilities = m.legacy_capabilities();
                            existing.unknown_capabilities = m.unknown_capabilities();
                            existing.manifest = Some(m);
                            report.unchanged += 1;
                        }
                        _ => {
                            self.plugins
                                .insert(id, PluginRecord::from_manifest(dir, m));
                            report.added += 1;
                        }
                    }
                }
                Err(e) => {
                    seen.push(dir_name.clone());
                    report.invalid += 1;
                    // Overwrite: a manifest that used to be valid and is now
                    // broken must stop reporting as fine.
                    self.plugins
                        .insert(dir_name.clone(), PluginRecord::from_error(dir, dir_name, &e));
                }
            }
        }

        // Drop records whose directory is gone, unless still running — killing
        // the record while the process lives would orphan a child we supervise.
        self.plugins
            .retain(|id, r| seen.contains(id) || r.state == PluginState::Running);
        // Records were rebuilt from manifests above, which resets user_disabled
        // to false — re-apply the user's persisted choice before anyone reads it.
        self.apply_disabled();
        report
    }

    pub fn list(&self) -> Vec<PluginRecord> {
        self.plugins.values().cloned().collect()
    }

    pub fn get(&self, id: &str) -> Option<&PluginRecord> {
        self.plugins.get(id)
    }

    pub fn insert(&mut self, record: PluginRecord) {
        self.plugins.insert(record.id.clone(), record);
    }

    /// Drop a plugin from the in-memory registry after its directory has been
    /// removed. `scan()` only ADDS what it finds on disk, so without this an
    /// uninstalled plugin would linger in the listing until the next restart.
    pub fn forget(&mut self, id: &str) -> bool {
        self.plugins.remove(id).is_some()
    }

    /// Move a plugin to `state`, with a reason for every non-running state.
    ///
    /// Takes the reason as `Option` but stores a fallback when a non-running
    /// state arrives without one, so the "every state explains itself" property
    /// holds even if a caller forgets.
    pub fn set_state(&mut self, id: &str, state: PluginState, reason: Option<String>) {
        if let Some(rec) = self.plugins.get_mut(id) {
            rec.state = state;
            rec.reason = match (state, reason) {
                (PluginState::Running, _) => None,
                (_, Some(r)) => Some(r),
                (s, None) => Some(format!("{s:?} (no reason recorded)")),
            };
        }
    }

    /// User opt-out. Distinct from a host refusal, though both read `Disabled`.
    /// Re-apply the persisted opt-outs to freshly-scanned records.
    ///
    /// Called at the end of `scan()`: records are rebuilt from their manifests
    /// there, so without this the user's choice is silently discarded on every
    /// rescan — and a rescan happens on every plugin listing.
    fn apply_disabled(&mut self) {
        for (id, rec) in self.plugins.iter_mut() {
            if self.disabled.contains(id) {
                rec.user_disabled = true;
                if rec.state != PluginState::Running {
                    rec.state = PluginState::Disabled;
                    rec.reason = Some("Turned off in OAIY Desktop → Plugins.".into());
                }
            }
        }
    }

    pub fn set_user_disabled(&mut self, id: &str, disabled: bool) {
        if disabled {
            self.disabled.insert(id.to_string());
        } else {
            self.disabled.remove(id);
        }
        self.persist_disabled();
        if let Some(rec) = self.plugins.get_mut(id) {
            rec.user_disabled = disabled;
            if disabled {
                rec.state = PluginState::Disabled;
                rec.reason = Some("Turned off in OAIY Desktop → Plugins.".into());
            } else if rec.is_loadable() {
                rec.state = PluginState::Stopped;
                rec.reason = Some("Turned on, not started yet.".into());
            }
        }
    }

    pub fn note_restart(&mut self, id: &str) -> u32 {
        match self.plugins.get_mut(id) {
            Some(rec) => {
                rec.restart_attempts += 1;
                rec.restart_attempts
            }
            None => 0,
        }
    }

    pub fn reset_restarts(&mut self, id: &str) {
        if let Some(rec) = self.plugins.get_mut(id) {
            rec.restart_attempts = 0;
        }
    }

    /// Plugins that should be started at boot: loadable, not user-disabled.
    pub fn autostart_ids(&self) -> Vec<String> {
        self.plugins
            .values()
            .filter(|r| r.is_loadable() && !r.user_disabled && r.state != PluginState::Disabled)
            .map(|r| r.id.clone())
            .collect()
    }

    /// Which plugin serves `connector_id`?
    fn owner_of(&self, connector_id: &str) -> Option<&PluginRecord> {
        self.plugins.values().find(|r| {
            r.manifest
                .as_ref()
                .is_some_and(|m| m.connectors.iter().any(|c| c.id == connector_id))
        })
    }

    /// Decide whether a connector command may be forwarded.
    ///
    /// Checked **before** the plugin is contacted, in the order the SDK contract
    /// specifies. An undeclared command must never reach the plugin process — a
    /// plugin is untrusted code, and "it will validate defensively" is a hope,
    /// not a control.
    pub fn gate(
        &self,
        connector_id: &str,
        command: &str,
        idempotency_key: Option<&str>,
    ) -> Result<&PluginRecord, GateRefusal> {
        let Some(rec) = self.owner_of(connector_id) else {
            return Err(GateRefusal::ConnectorMissing {
                connector_id: connector_id.to_string(),
            });
        };
        let manifest = rec
            .manifest
            .as_ref()
            .expect("owner_of only matches records with a manifest");

        if !rec.state.accepts_commands() {
            return Err(GateRefusal::ConnectorUnavailable {
                connector_id: connector_id.to_string(),
                state: rec.state,
                reason: rec.reason.clone(),
            });
        }

        if !manifest.declares_command(connector_id, command) {
            return Err(GateRefusal::CapabilityDenied {
                connector_id: connector_id.to_string(),
                command: command.to_string(),
            });
        }

        // A journalled command has effects a retry must not repeat. Requiring the
        // key here, rather than trusting the caller, is what makes "sent the SMS
        // twice because a socket blipped" impossible rather than unlikely.
        if manifest.is_journalled(command)
            && idempotency_key.map(str::trim).unwrap_or("").is_empty()
        {
            return Err(GateRefusal::IdempotencyRequired {
                command: command.to_string(),
            });
        }

        Ok(rec)
    }

    /// May `plugin_id` emit `event_name`?
    ///
    /// An undeclared event is dropped. Otherwise a plugin could invent event
    /// names to reach triggers it was never granted — the event bus would become
    /// a way around the capability model.
    pub fn may_emit(&self, plugin_id: &str, event_name: &str) -> bool {
        self.plugins
            .get(plugin_id)
            .and_then(|r| r.manifest.as_ref())
            .is_some_and(|m| m.declares_event(event_name))
    }

    /// Does `plugin_id` hold `capability`? Exact match on the resolved set.
    pub fn grants(&self, plugin_id: &str, capability: &str) -> bool {
        self.plugins
            .get(plugin_id)
            .and_then(|r| r.manifest.as_ref())
            .is_some_and(|m| m.grants(capability))
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub added: usize,
    pub unchanged: usize,
    pub invalid: usize,
}

pub type PluginRegistryHandle = Arc<Mutex<PluginRegistry>>;

pub fn new_handle(root: PathBuf) -> PluginRegistryHandle {
    Arc::new(Mutex::new(PluginRegistry::new(root)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static N: AtomicU32 = AtomicU32::new(0);

    struct Root(PathBuf);
    impl Root {
        fn new() -> Self {
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("oaiy-plugreg-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
        /// Write a plugin dir. `id` doubles as the directory name unless
        /// `dir_name` overrides it.
        fn plugin(&self, dir_name: &str, body: serde_json::Value) {
            let d = self.0.join(dir_name);
            fs::create_dir_all(&d).unwrap();
            fs::write(
                d.join("manifest.json"),
                serde_json::to_string_pretty(&body).unwrap(),
            )
            .unwrap();
            fs::write(d.join("plugin.exe"), b"stub").unwrap();
        }
    }
    impl Drop for Root {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn manifest(id: &str) -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 3,
            "id": id,
            "name": format!("{id} plugin"),
            "version": "0.1.0",
            "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe" },
            "capabilities": ["flow.run", "connector.aokie.*"],
            "connectors": [{ "id": "aokie", "commands": ["call.answer", "sms.send"] }],
            "events": ["aokie.call.incoming"],
            "commands": { "journalled": ["sms.send"] }
        })
    }

    // --- scanning ---------------------------------------------------------

    #[test]
    fn a_valid_plugin_is_discovered() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        let report = reg.scan();
        assert_eq!(report.added, 1);
        assert_eq!(report.invalid, 0);
        let rec = reg.get("aokie").expect("discovered");
        assert_eq!(rec.state, PluginState::Installed);
        assert!(rec.is_loadable());
    }

    #[test]
    fn a_broken_plugin_is_listed_disabled_with_the_parse_error() {
        // Skipping it would make a broken plugin indistinguishable from an
        // uninstalled one, and reinstalling does not fix a bad manifest.
        let root = Root::new();
        let d = root.path().join("busted");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("manifest.json"), b"{ not json").unwrap();

        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        assert_eq!(reg.scan().invalid, 1);
        let rec = reg.get("busted").expect("must still be listed");
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(!rec.is_loadable());
        let reason = rec.reason.as_deref().unwrap_or("");
        assert!(reason.contains("not valid JSON"), "{reason}");
    }

    #[test]
    fn an_unsupported_api_version_is_disabled_with_a_reason_naming_it() {
        let root = Root::new();
        let mut m = manifest("future");
        m["pluginApiVersion"] = serde_json::json!(42);
        root.plugin("future", m);
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        let reason = reg.get("future").unwrap().reason.clone().unwrap();
        assert!(reason.contains("42"), "{reason}");
    }

    #[test]
    fn an_id_that_disagrees_with_its_directory_is_refused() {
        // Two dirs both claiming `aokie` would shadow each other, and which won
        // would depend on directory iteration order.
        let root = Root::new();
        root.plugin("some-folder", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        assert_eq!(reg.scan().invalid, 1);
        let rec = reg.get("some-folder").unwrap();
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(rec.reason.as_deref().unwrap().contains("directory name"));
    }

    #[test]
    fn a_missing_plugins_root_is_not_an_error() {
        let mut reg = PluginRegistry::new(PathBuf::from(r"Z:\definitely\not\here"));
        assert_eq!(reg.scan(), ScanReport::default());
        assert!(reg.list().is_empty());
    }

    #[test]
    fn loose_files_in_the_root_are_ignored() {
        let root = Root::new();
        fs::write(root.path().join("README.txt"), b"hi").unwrap();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        assert_eq!(reg.scan().added, 1);
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn a_rescan_does_not_reset_a_running_plugin() {
        // `GET /api/plugins` rescans; knocking a running plugin back to
        // Installed would make the UI lie and invite a duplicate start.
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);

        let report = reg.scan();
        assert_eq!(report.added, 0);
        assert_eq!(report.unchanged, 1);
        assert_eq!(reg.get("aokie").unwrap().state, PluginState::Running);
    }

    #[test]
    fn a_removed_plugin_is_dropped_unless_it_is_running() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        fs::remove_dir_all(root.path().join("aokie")).unwrap();

        reg.scan();
        assert!(reg.get("aokie").is_none(), "a gone plugin should disappear");

        // But not while we still supervise its process.
        root.plugin("aokie", manifest("aokie"));
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        fs::remove_dir_all(root.path().join("aokie")).unwrap();
        reg.scan();
        assert!(
            reg.get("aokie").is_some(),
            "dropping the record would orphan a child we supervise"
        );
    }

    #[test]
    fn a_manifest_that_becomes_invalid_stops_reporting_as_fine() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        assert!(reg.get("aokie").unwrap().is_loadable());

        fs::write(root.path().join("aokie").join("manifest.json"), b"broken").unwrap();
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(!rec.is_loadable());
    }

    #[test]
    fn listings_are_ordered_not_hash_ordered() {
        // A list that reshuffles between polls is unusable in a UI.
        let root = Root::new();
        for id in ["zulu", "alpha", "mike"] {
            root.plugin(id, manifest(id));
        }
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        let ids: Vec<String> = reg.list().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, vec!["alpha", "mike", "zulu"]);
    }

    // --- state + reasons --------------------------------------------------

    #[test]
    fn every_non_running_state_carries_a_reason() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();

        for st in [
            PluginState::Installed,
            PluginState::Stopped,
            PluginState::Starting,
            PluginState::Unhealthy,
            PluginState::Crashed,
            PluginState::Disabled,
        ] {
            // Deliberately pass None — the registry must still record something.
            reg.set_state("aokie", st, None);
            let rec = reg.get("aokie").unwrap();
            assert!(
                rec.reason.as_deref().is_some_and(|r| !r.is_empty()),
                "{st:?} left no reason"
            );
        }

        reg.set_state("aokie", PluginState::Running, Some("ignored".into()));
        assert!(
            reg.get("aokie").unwrap().reason.is_none(),
            "a running plugin needs no excuse"
        );
    }

    #[test]
    fn a_never_started_plugin_does_not_report_as_crashed() {
        // Found by driving the live API: discovery reported `plugin_crashed` for a
        // freshly installed plugin, which tells a user their software broke when
        // it is merely idle. "Start it" and "it crashed, check the logs" are
        // different instructions.
        assert_eq!(
            PluginState::Installed.unavailable_reason(false),
            Some("service_stopped")
        );
        assert_eq!(
            PluginState::Stopped.unavailable_reason(false),
            Some("service_stopped")
        );
        assert_eq!(
            PluginState::Starting.unavailable_reason(false),
            Some("service_stopped"),
            "mid-start is not a fault"
        );
        assert_eq!(
            PluginState::Crashed.unavailable_reason(false),
            Some("plugin_crashed")
        );
    }

    #[test]
    fn an_available_state_has_no_unavailability_reason() {
        assert_eq!(PluginState::Running.unavailable_reason(false), None);
        assert_eq!(
            PluginState::Unhealthy.unavailable_reason(false),
            None,
            "unhealthy still serves commands, so it is not unavailable"
        );
    }

    #[test]
    fn a_user_optout_reads_differently_from_a_host_refusal() {
        assert_eq!(
            PluginState::Disabled.unavailable_reason(true),
            Some("plugin_disabled")
        );
        assert_eq!(
            PluginState::Disabled.unavailable_reason(false),
            Some("not_installed"),
            "a manifest this host cannot honour is not the user's opt-out"
        );
    }

    #[test]
    fn every_unavailability_reason_is_in_the_protocol_enum() {
        // `capability-manifest.schema.json` closes this set; an invented code
        // would fail schema validation at the consumer.
        const ALLOWED: &[&str] = &[
            "not_installed",
            "service_stopped",
            "connection_missing",
            "plugin_disabled",
            "plugin_crashed",
            "permission_denied",
            "unsupported_platform",
            "hardware_missing",
        ];
        for st in [
            PluginState::Installed,
            PluginState::Stopped,
            PluginState::Starting,
            PluginState::Running,
            PluginState::Unhealthy,
            PluginState::Crashed,
            PluginState::Disabled,
        ] {
            for ud in [true, false] {
                if let Some(r) = st.unavailable_reason(ud) {
                    assert!(ALLOWED.contains(&r), "{st:?}/{ud} -> {r:?} is not in the schema enum");
                }
            }
        }
    }

    #[test]
    fn user_disabled_is_distinct_from_a_host_refusal() {
        // Both read Disabled, and they are not the same problem: one is fixed by
        // a toggle, the other by a new plugin build.
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut m = manifest("future");
        m["pluginApiVersion"] = serde_json::json!(42);
        root.plugin("future", m);

        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_user_disabled("aokie", true);

        let a = reg.get("aokie").unwrap();
        let f = reg.get("future").unwrap();
        assert_eq!(a.state, PluginState::Disabled);
        assert_eq!(f.state, PluginState::Disabled);
        assert!(a.user_disabled);
        assert!(!f.user_disabled, "the host refused this one; the user did not");
        assert!(a.reason.as_deref().unwrap().contains("Turned off"));
    }

    #[test]
    fn re_enabling_returns_a_plugin_to_stopped() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_user_disabled("aokie", true);
        reg.set_user_disabled("aokie", false);
        assert_eq!(reg.get("aokie").unwrap().state, PluginState::Stopped);
    }

    #[test]
    fn re_enabling_an_unloadable_plugin_does_not_make_it_runnable() {
        let root = Root::new();
        let d = root.path().join("busted");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("manifest.json"), b"nope").unwrap();
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_user_disabled("busted", false);
        assert_eq!(
            reg.get("busted").unwrap().state,
            PluginState::Disabled,
            "a toggle cannot fix a manifest"
        );
    }

    #[test]
    fn a_users_opt_out_survives_a_restart_and_a_rescan() {
        // Regression: `user_disabled` lived only in memory and `scan()` rebuilds
        // every record from its manifest, so the flag reset to false on each
        // scan. Harmless while nothing started plugins by itself — but boot
        // autostart turned it into a policy INVERSION: the plugins the user
        // turned off were the ones started for them, every launch. For the phone
        // bridge that means seizing a Bluetooth dongle they deliberately freed.
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        root.plugin("other", manifest("other"));

        {
            let mut reg = PluginRegistry::new(root.path().to_path_buf());
            reg.scan();
            reg.set_user_disabled("aokie", true);
            // A rescan in the SAME process must not resurrect it either.
            reg.scan();
            assert!(reg.get("aokie").unwrap().user_disabled);
            assert_eq!(reg.autostart_ids(), vec!["other".to_string()]);
        }

        // A fresh registry over the same root — i.e. the next app launch.
        let mut reopened = PluginRegistry::new(root.path().to_path_buf());
        reopened.scan();
        assert!(reopened.get("aokie").unwrap().user_disabled, "the opt-out must survive a restart");
        assert_eq!(reopened.get("aokie").unwrap().state, PluginState::Disabled);
        assert_eq!(reopened.autostart_ids(), vec!["other".to_string()]);

        // And turning it back on sticks, so the file is not write-only.
        reopened.set_user_disabled("aokie", false);
        let mut again = PluginRegistry::new(root.path().to_path_buf());
        again.scan();
        assert!(!again.get("aokie").unwrap().user_disabled);
        assert_eq!(again.autostart_ids().len(), 2);
    }

    #[test]
    fn autostart_skips_disabled_and_unloadable_plugins() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        root.plugin("other", manifest("other"));
        let d = root.path().join("busted");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("manifest.json"), b"nope").unwrap();

        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_user_disabled("other", true);

        assert_eq!(reg.autostart_ids(), vec!["aokie".to_string()]);
    }

    #[test]
    fn restart_attempts_count_and_reset() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        assert_eq!(reg.note_restart("aokie"), 1);
        assert_eq!(reg.note_restart("aokie"), 2);
        reg.reset_restarts("aokie");
        assert_eq!(reg.get("aokie").unwrap().restart_attempts, 0);
    }

    // --- the connector gate -----------------------------------------------

    fn running_registry() -> (Root, PluginRegistry) {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        (root, reg)
    }

    #[test]
    fn a_declared_command_on_a_running_plugin_passes() {
        let (_r, reg) = running_registry();
        assert!(reg.gate("aokie", "call.answer", None).is_ok());
    }

    #[test]
    fn an_undeclared_command_is_refused_before_the_plugin_is_contacted() {
        let (_r, reg) = running_registry();
        let err = reg.gate("aokie", "call.hangup", None).unwrap_err();
        assert_eq!(err.code(), "capability_denied");
        assert!(
            err.message().contains("before the plugin was contacted"),
            "{}",
            err.message()
        );
    }

    #[test]
    fn an_unknown_connector_is_actionable() {
        let (_r, reg) = running_registry();
        let err = reg.gate("nosuch", "x.y", None).unwrap_err();
        assert_eq!(err.code(), "capability_unavailable");
        assert!(err.message().contains("Plugins"), "must say where to go: {}", err.message());
    }

    #[test]
    fn a_stopped_plugin_refuses_commands_and_says_why() {
        let (_r, mut reg) = running_registry();
        reg.set_state("aokie", PluginState::Crashed, Some("Exited with code 1.".into()));
        let err = reg.gate("aokie", "call.answer", None).unwrap_err();
        assert_eq!(err.code(), "capability_unavailable");
        let msg = err.message();
        assert!(msg.contains("Crashed"), "{msg}");
        assert!(msg.contains("Exited with code 1."), "surface the reason: {msg}");
    }

    #[test]
    fn an_unhealthy_plugin_still_accepts_commands() {
        // Health is coarse. Refusing everything because one probe timed out makes
        // a slow plugin unusable rather than merely slow.
        let (_r, mut reg) = running_registry();
        reg.set_state("aokie", PluginState::Unhealthy, Some("3 missed probes.".into()));
        assert!(reg.gate("aokie", "call.answer", None).is_ok());
    }

    #[test]
    fn a_journalled_command_requires_an_idempotency_key() {
        let (_r, reg) = running_registry();
        let err = reg.gate("aokie", "sms.send", None).unwrap_err();
        assert!(matches!(err, GateRefusal::IdempotencyRequired { .. }));
        assert_eq!(err.code(), "invalid_request");

        assert!(reg.gate("aokie", "sms.send", Some("sms:42")).is_ok());
        // Blank and whitespace-only keys are not keys.
        assert!(reg.gate("aokie", "sms.send", Some("")).is_err());
        assert!(reg.gate("aokie", "sms.send", Some("   ")).is_err());
    }

    #[test]
    fn a_read_only_command_needs_no_idempotency_key() {
        let (_r, reg) = running_registry();
        assert!(reg.gate("aokie", "call.answer", None).is_ok());
    }

    #[test]
    fn the_gate_checks_state_before_the_command_allow_list() {
        // Order matters for the message the user sees: a stopped plugin should
        // say "start it", not "that command does not exist".
        let (_r, mut reg) = running_registry();
        reg.set_state("aokie", PluginState::Stopped, Some("Not started.".into()));
        let err = reg.gate("aokie", "call.hangup", None).unwrap_err();
        assert_eq!(err.code(), "capability_unavailable");
    }

    // --- events + capabilities -------------------------------------------

    #[test]
    fn an_undeclared_event_may_not_be_emitted() {
        let (_r, reg) = running_registry();
        assert!(reg.may_emit("aokie", "aokie.call.incoming"));
        assert!(
            !reg.may_emit("aokie", "aokie.call.invented"),
            "the event bus must not become a way around the capability model"
        );
        assert!(!reg.may_emit("nosuch", "aokie.call.incoming"));
    }

    #[test]
    fn capability_checks_go_through_the_resolved_set() {
        let (_r, reg) = running_registry();
        // Declared as the legacy `flow.run`, enforced as the canonical name.
        assert!(reg.grants("aokie", "oaiy.flow.run"));
        assert!(reg.grants("aokie", "connector.aokie.sms.send"));
        assert!(!reg.grants("aokie", "connector.aokie.call.hangup"));
        assert!(!reg.grants("nosuch", "oaiy.flow.run"));
    }

    #[test]
    fn legacy_and_unknown_capabilities_are_surfaced_on_the_record() {
        let root = Root::new();
        let mut m = manifest("aokie");
        m["capabilities"] = serde_json::json!(["flow.run", "companion.admission"]);
        root.plugin("aokie", m);
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(
            rec.legacy_capabilities,
            vec![("flow.run".to_string(), "oaiy.flow.run".to_string())]
        );
        assert_eq!(rec.unknown_capabilities, vec!["companion.admission".to_string()]);
    }
}
