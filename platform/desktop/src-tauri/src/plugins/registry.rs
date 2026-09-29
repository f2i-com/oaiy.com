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
use super::trust::{LaunchPermit, PackageTrust, TrustService, PACKAGE_MANIFEST_FILE};

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
    /// Latest successful supervisor probe, for plugin screens and diagnostics.
    /// Only the health contract fields are retained; never the whole RPC reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_health: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_health_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_health_error: Option<String>,
    /// Who made the package and whether it is still what they made (see
    /// [`super::trust`]). `None` only when the manifest could not be loaded, so there
    /// was no plugin to judge.
    ///
    /// A package that may not run (`quarantined`, or `unsigned` in a release build) has
    /// NO `manifest` here. The manifest is what everything else reads to decide what a
    /// plugin brings: its modules, pages, agent tools, setup, service definitions and
    /// screens. A folder that failed its signature must not contribute any of them, and
    /// leaving each of those readers to remember to ask would be the way one of them
    /// forgot. It is listed the way a plugin with a broken manifest is: disabled, with
    /// the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust: Option<PackageTrust>,
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
            last_health: None,
            last_health_at: None,
            last_health_error: None,
            trust: None,
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
            last_health: None,
            last_health_at: None,
            last_health_error: None,
            trust: None,
        }
    }

    /// A plugin whose package may not run: listed as disabled, with why, and without a
    /// manifest (see [`PluginRecord::trust`]).
    fn withheld(dir: PathBuf, id: String, trust: PackageTrust) -> Self {
        Self {
            id,
            state: PluginState::Disabled,
            reason: Some(format!("Not started. {}", trust.reason.as_deref().unwrap_or("Its package is not trusted."))),
            dir,
            manifest: None,
            legacy_capabilities: Vec::new(),
            unknown_capabilities: Vec::new(),
            user_disabled: false,
            restart_attempts: 0,
            last_health: None,
            last_health_at: None,
            last_health_error: None,
            trust: Some(trust),
        }
    }

    fn with_trust(mut self, trust: PackageTrust) -> Self {
        self.trust = Some(trust);
        self
    }

    /// Does its package's trust say it may not be started? (A plugin already running when
    /// its folder stopped verifying keeps running with the manifest it was started from,
    /// and reads no other from the folder, but says so here, and is not started again.)
    pub fn refused_by_trust(&self) -> bool {
        self.trust.as_ref().is_some_and(|t| !t.allows_launch())
    }

    /// Is this plugin usable at all? A record with no manifest never becomes
    /// runnable however many times the user presses Start.
    pub fn is_loadable(&self) -> bool {
        self.manifest.is_some()
    }

    fn clear_health(&mut self) {
        self.last_health = None;
        self.last_health_at = None;
        self.last_health_error = None;
    }
}

/// Health is plugin-supplied data. Keep its documented display fields only,
/// with a small total bound so repeated listings cannot grow without limit.
fn health_snapshot(value: &serde_json::Value) -> Result<serde_json::Value, String> {
    let status = value.get("status").and_then(serde_json::Value::as_str)
        .filter(|status| !status.is_empty() && status.len() <= 64)
        .ok_or_else(|| "The plugin returned an invalid health status.".to_string())?;
    let mut snapshot = serde_json::json!({ "status": status });
    if let Some(detail) = value.get("detail").and_then(serde_json::Value::as_str) {
        snapshot["detail"] = serde_json::json!(detail);
    }
    if let Some(components) = value.get("components").filter(|v| v.is_object()) {
        snapshot["components"] = components.clone();
    }
    if snapshot.to_string().len() > 64 * 1024 {
        return Err("The plugin health report exceeds the 64 KiB display limit.".into());
    }
    Ok(snapshot)
}

pub struct PluginRegistry {
    root: PathBuf,
    /// Verifies the packages in `root`: at each scan, and (through [`Self::trust`])
    /// immediately before each launch and at install.
    trust: Arc<TrustService>,
    /// Plugins whose legacy `data` folder could not be moved out of a signed bundle,
    /// said once each rather than on every scan.
    migration_warned: std::collections::BTreeSet<String>,
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

    /// A registry over `root` under this build's trust policy and pinned publishers.
    pub fn new(root: PathBuf) -> Self {
        let trust = TrustService::for_root(&root);
        Self::with_trust(root, trust)
    }

    /// A registry that verifies packages with `trust`.
    pub fn with_trust(root: PathBuf, trust: Arc<TrustService>) -> Self {
        Self {
            disabled: Self::load_disabled(&root),
            root,
            trust,
            migration_warned: Default::default(),
            plugins: BTreeMap::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The package verifier, for the callers that must check outside a scan: a launch,
    /// an install, the person's decision to trust a package.
    pub fn trust(&self) -> Arc<TrustService> {
        self.trust.clone()
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
            // Installer staging/rollback directories are not installed plugins.
            // A concurrent scan or interrupted cleanup must never expose them.
            if dir_name.starts_with('.') {
                continue;
            }

            // The bytes are kept: the trust check is told which manifest.json this scan
            // built the plugin from, so a verdict is never applied to a file it is not about.
            match PluginManifest::read(&dir) {
                Ok((m, manifest_json)) => {
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

                    // A process that is up holds its folder open. It is left as it is:
                    // what a scan learns about its package is shown, and is acted on
                    // when it next starts.
                    let live = self.plugins.get(&id).is_some_and(|r| {
                        r.is_loadable()
                            && matches!(r.state, PluginState::Starting | PluginState::Running | PluginState::Unhealthy)
                    });
                    if !live {
                        self.move_legacy_data_out_of_a_signed_bundle(&id, &dir);
                    }
                    let trust = self.trust.assess_read(&dir, &id, &manifest_json);

                    if !trust.allows_launch() && !live {
                        // Quarantined, or unsigned and not trusted: listed with why, and
                        // with nothing of its manifest for anyone to build on.
                        self.plugins.insert(id.clone(), PluginRecord::withheld(dir, id, trust));
                        report.invalid += 1;
                        continue;
                    }
                    match self.plugins.get_mut(&id) {
                        Some(existing) if existing.is_loadable() => {
                            if trust.allows_launch() {
                                // Refresh the manifest but keep runtime state.
                                existing.legacy_capabilities = m.legacy_capabilities();
                                existing.unknown_capabilities = m.unknown_capabilities();
                                existing.manifest = Some(m);
                            } else if existing.trust.as_ref().map(|t| t.state) != Some(trust.state) {
                                // What a live process may do is what it was started with: its
                                // commands, its events and its capabilities. A manifest read
                                // from a folder that no longer verifies is not adopted, or an
                                // edit to manifest.json would widen a running plugin's
                                // permissions on the strength of a file no signature covers.
                                // Said when it happens, not on every scan after.
                                log::warn!(
                                    "plugin {id} is running but its package no longer verifies, so it will not be started again: {}",
                                    trust.reason.as_deref().unwrap_or("not trusted")
                                );
                            }
                            existing.trust = Some(trust);
                            report.unchanged += 1;
                        }
                        _ => {
                            self.plugins
                                .insert(id, PluginRecord::from_manifest(dir, m).with_trust(trust));
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

    /// Old installs kept the plugin's own state in `<plugin>/data`, inside the bundle.
    /// In a signed bundle that folder is a set of files the signature does not list, so
    /// the package would be quarantined before it ever started and got the chance to move
    /// it (the move used to be made at start). Made here, before the package is looked at,
    /// and only when the move is what makes the package verify (see
    /// [`TrustService::legacy_data_may_move`]): a package that fails its check for another
    /// reason, or whose signature lists a `data` folder of its own, is left exactly as
    /// found, and a link named `data` is never moved (see `migrate_legacy_data_dir`).
    fn move_legacy_data_out_of_a_signed_bundle(&mut self, id: &str, dir: &Path) {
        if !dir.join("data").is_dir() || !dir.join(PACKAGE_MANIFEST_FILE).is_file() {
            return;
        }
        if !self.trust.legacy_data_may_move(dir, id) {
            return;
        }
        match super::runner::migrate_legacy_data_dir(dir) {
            Ok(true) => log::info!("moved {}/data out of the signed plugin bundle so its signature can verify", dir.display()),
            Ok(false) => {}
            Err(e) => {
                if self.migration_warned.insert(id.to_string()) {
                    log::warn!("legacy plugin data dir not migrated: {e}");
                }
            }
        }
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

    /// A launch found the package is not what it was at the last scan (or never was):
    /// take the plugin's manifest away and say why, exactly as a scan would have.
    ///
    /// Only for a plugin that is not running: this is called by a start that has just
    /// been refused.
    pub fn withhold(&mut self, id: &str, trust: PackageTrust) {
        let Some(existing) = self.plugins.get(id) else { return };
        let record = PluginRecord::withheld(existing.dir.clone(), id.to_string(), trust);
        self.plugins.insert(id.to_string(), record);
        self.apply_disabled();
    }

    /// A launch has verified the package and parsed its manifest from the bytes it hashed:
    /// the record follows what is about to run. What its gate, its events and its
    /// capabilities allow is then what the process was started from, and not an earlier
    /// scan's reading of the file; and a record a scan had withheld on a verdict that
    /// was stale (the launch checks the bytes, the scan may only have looked at the
    /// folder's sizes and times) is a plugin again, with the verdict that was just made.
    pub fn adopt_launch(&mut self, id: &str, permit: &LaunchPermit) {
        if let Some(rec) = self.plugins.get_mut(id) {
            let manifest = permit.manifest().clone();
            rec.legacy_capabilities = manifest.legacy_capabilities();
            rec.unknown_capabilities = manifest.unknown_capabilities();
            rec.manifest = Some(manifest);
            rec.trust = Some(permit.trust().clone());
        }
    }

    /// Drop a plugin from the in-memory registry after its directory has been
    /// removed, and the trust the person gave it: a package installed again is a new
    /// decision. `scan()` only ADDS what it finds on disk, so without this an
    /// uninstalled plugin would linger in the listing until the next restart.
    pub fn forget(&mut self, id: &str) -> bool {
        self.trust.forget_local(id);
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
            if !state.accepts_commands() {
                rec.clear_health();
            }
            rec.reason = match (state, reason) {
                (PluginState::Running, _) => None,
                (_, Some(r)) => Some(r),
                (s, None) => Some(format!("{s:?} (no reason recorded)")),
            };
        }
    }

    /// Record a probe only while the plugin is live. A failed probe removes a
    /// previously green snapshot immediately, independently of restart policy.
    pub fn note_health(&mut self, id: &str, outcome: Result<&serde_json::Value, String>) {
        let Some(rec) = self.plugins.get_mut(id) else { return };
        if !rec.state.accepts_commands() {
            rec.clear_health();
            return;
        }
        match outcome.and_then(health_snapshot) {
            Ok(snapshot) => {
                rec.last_health = Some(snapshot);
                rec.last_health_at = Some(chrono::Utc::now());
                rec.last_health_error = None;
            }
            Err(error) => {
                rec.clear_health();
                rec.last_health_error = Some(error.chars().take(2048).collect());
            }
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
                    rec.clear_health();
                    // A package that failed its check keeps saying so: that is the fact
                    // that matters when the person turns the plugin back on.
                    if !rec.refused_by_trust() {
                        rec.reason = Some("Turned off in OAIY Desktop → Plugins.".into());
                    }
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
            rec.clear_health();
            if disabled {
                rec.state = PluginState::Disabled;
                if !rec.refused_by_trust() {
                    rec.reason = Some("Turned off in OAIY Desktop → Plugins.".into());
                }
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

    /// The manifest of the plugin that serves `connector_id`: the one the gate
    /// reads, so what is declared and journalled here is what the gate will see.
    pub fn manifest_for_connector(&self, connector_id: &str) -> Option<&PluginManifest> {
        self.owner_of(connector_id).and_then(|r| r.manifest.as_ref())
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
            // A plugin whose package may not run has no manifest here, so it declares no
            // connector, and "no plugin provides it, install one" would be the wrong thing
            // to tell a caller of `aokie` when `aokie` is installed and quarantined. A
            // plugin's connector carries its own name (the screens' `PluginHost.command`
            // assumes as much), which is enough to say what happened.
            if let Some(held) = self.plugins.get(connector_id).filter(|r| r.manifest.is_none() && r.refused_by_trust()) {
                return Err(GateRefusal::ConnectorUnavailable {
                    connector_id: connector_id.to_string(),
                    state: held.state,
                    reason: held.reason.clone(),
                });
            }
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
    fn installer_staging_and_backups_are_not_discovered() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        root.plugin(".staging-test", manifest("aokie"));
        root.plugin(".backup-aokie-test", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        let report = reg.scan();
        assert_eq!(report.added, 1);
        assert_eq!(report.invalid, 0);
    }

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
    fn plugin_listings_keep_the_latest_component_health_across_rescans() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        reg.note_health("aokie", Ok(&serde_json::json!({
            "status": "ok", "detail": null,
            "components": { "responder": { "mode": "agent", "ready": true }, "voice": { "available": true } },
            "unexpectedPrivateField": "must not leave the host",
        })));
        reg.scan(); // GET /api/plugins follows this exact list/rescan path.
        let wire = serde_json::to_value(reg.list()).unwrap();
        assert_eq!(wire[0]["lastHealth"]["status"], "ok");
        assert_eq!(wire[0]["lastHealth"]["components"]["responder"]["ready"], true);
        assert!(wire[0]["lastHealthAt"].is_string());
        assert!(wire[0]["lastHealth"].get("unexpectedPrivateField").is_none());
        assert!(wire[0].get("lastHealthError").is_none());
    }

    #[test]
    fn failed_health_probes_remove_stale_success_and_successful_probes_recover() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        reg.note_health("aokie", Ok(&serde_json::json!({ "status": "ok" })));
        reg.note_health("aokie", Err("Health probe timed out.".into()));
        let rec = reg.get("aokie").unwrap();
        assert!(rec.last_health.is_none());
        assert!(rec.last_health_at.is_none());
        assert_eq!(rec.last_health_error.as_deref(), Some("Health probe timed out."));

        reg.set_state("aokie", PluginState::Unhealthy, Some("Probe failed.".into()));
        reg.note_health("aokie", Ok(&serde_json::json!({ "status": "degraded", "detail": "LLM unavailable" })));
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.last_health.as_ref().unwrap()["detail"], "LLM unavailable");
        assert!(rec.last_health_error.is_none());
    }

    #[test]
    fn stopped_or_restarted_plugins_never_inherit_the_previous_health_report() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        for state in [PluginState::Stopped, PluginState::Starting, PluginState::Crashed, PluginState::Disabled] {
            reg.set_state("aokie", PluginState::Running, None);
            reg.note_health("aokie", Ok(&serde_json::json!({ "status": "ok" })));
            reg.set_state("aokie", state, None);
            // A late reply after the process stopped cannot restore its report.
            reg.note_health("aokie", Ok(&serde_json::json!({ "status": "ok" })));
            assert!(reg.get("aokie").unwrap().last_health.is_none());
            assert!(reg.get("aokie").unwrap().last_health_at.is_none());
        }
    }

    #[test]
    fn malformed_or_oversized_health_is_reported_without_expanding_plugin_listings() {
        assert!(health_snapshot(&serde_json::json!({ "components": {} })).is_err());
        assert!(health_snapshot(&serde_json::json!({ "status": 5 })).is_err());
        assert!(health_snapshot(&serde_json::json!({
            "status": "ok", "components": { "detail": "x".repeat(64 * 1024) },
        })).unwrap_err().contains("64 KiB"));
        let snapshot = health_snapshot(&serde_json::json!({
            "status": "ok", "detail": null, "components": [], "unrelated": "ignored",
        })).unwrap();
        assert_eq!(snapshot, serde_json::json!({ "status": "ok" }));
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
        // `companion.admission` is the second legacy alias the host understands,
        // so the unknown example has to be something the host really has no
        // meaning for — otherwise this test would pass by accident the moment
        // the name it used became real.
        m["capabilities"] = serde_json::json!(["flow.run", "companion.admission", "aokie.hardware.seize"]);
        root.plugin("aokie", m);
        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(
            rec.legacy_capabilities,
            vec![
                ("flow.run".to_string(), "oaiy.flow.run".to_string()),
                (
                    "companion.admission".to_string(),
                    "oaiy.companion.admission".to_string()
                )
            ]
        );
        assert_eq!(rec.unknown_capabilities, vec!["aokie.hardware.seize".to_string()]);
    }

    // --- package trust ------------------------------------------------------

    use crate::plugins::trust::tests::TestKey;
    use crate::plugins::trust::{Publishers, TrustPolicy, TrustState};

    fn trust_service(root: &Root, policy: TrustPolicy, publishers: Publishers) -> Arc<TrustService> {
        TrustService::new(policy, publishers, root.path().join("trusted-plugins.json"))
    }

    /// A registry under a release build's rules that pins `key` for the plugin `aokie`.
    fn release_registry(root: &Root, key: &TestKey) -> PluginRegistry {
        let trust = trust_service(root, TrustPolicy::release(), key.pinned_for("Aokie", &["aokie"]));
        PluginRegistry::with_trust(root.path().to_path_buf(), trust)
    }

    fn signed_aokie(root: &Root, key: &TestKey) -> PathBuf {
        root.plugin("aokie", manifest("aokie"));
        let dir = root.path().join("aokie");
        key.sign(&dir, "aokie-plugin", "0.1.0");
        dir
    }

    #[test]
    fn a_signed_plugin_that_verifies_is_listed_with_its_publisher_and_keeps_its_manifest() {
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        signed_aokie(&root, &key);
        let mut reg = release_registry(&root, &key);
        assert_eq!(reg.scan().added, 1);

        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Installed);
        assert!(rec.is_loadable(), "a verified plugin's manifest is what everything builds on");
        let trust = rec.trust.as_ref().unwrap();
        assert_eq!(trust.state, TrustState::Verified);
        assert_eq!(trust.publisher.as_deref(), Some("Aokie"));
        assert_eq!(reg.autostart_ids(), vec!["aokie".to_string()]);

        // As the listing carries it.
        let wire = serde_json::to_value(reg.list()).unwrap();
        assert_eq!(wire[0]["trust"]["state"], "verified");
        assert_eq!(wire[0]["trust"]["publisher"], "Aokie");
        assert_eq!(wire[0]["trust"]["keyId"], "fl-test-2026a");
        assert!(wire[0]["trust"].get("reason").is_none());
    }

    #[test]
    fn a_tampered_signed_plugin_is_listed_disabled_with_why_and_gives_nothing_to_build_on() {
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        let dir = signed_aokie(&root, &key);
        fs::write(dir.join("plugin.exe"), b"tampered").unwrap();

        // Even a developer's build: a package that carries a signature is not waived.
        let trust = trust_service(&root, TrustPolicy::developer(), key.pinned_for("Aokie", &["aokie"]));
        let mut reg = PluginRegistry::with_trust(root.path().to_path_buf(), trust);
        let report = reg.scan();
        assert_eq!((report.added, report.invalid), (0, 1));

        let rec = reg.get("aokie").expect("it is listed, not dropped");
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(!rec.user_disabled, "the person did not turn it off");
        assert!(!rec.is_loadable(), "no manifest: no module, page, tool, screen or setup step comes from it");
        assert!(rec.refused_by_trust());
        let reason = rec.reason.as_deref().unwrap();
        assert!(reason.starts_with("Not started."), "{reason}");
        assert!(reason.contains("Quarantined: digest mismatch: plugin.exe"), "{reason}");
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Quarantined);
        assert!(reg.autostart_ids().is_empty());

        // Turning it on does not revive it.
        reg.set_user_disabled("aokie", false);
        assert_eq!(reg.get("aokie").unwrap().state, PluginState::Disabled);
        assert!(reg.autostart_ids().is_empty());

        // And what a caller of its connector is told says why, rather than "not installed".
        let refusal = reg.gate("aokie", "call.answer", None).unwrap_err();
        assert_eq!(refusal.code(), "capability_unavailable");
        assert!(refusal.message().contains("Quarantined"), "{}", refusal.message());
        assert!(!refusal.message().contains("No installed plugin provides"), "{}", refusal.message());
    }

    #[test]
    fn what_a_quarantined_plugin_would_have_contributed_is_not_contributed() {
        // The modules, pages and tools read the manifest; a record without one gives them nothing.
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        let dir = signed_aokie(&root, &key);
        let mut reg = release_registry(&root, &key);
        reg.scan();
        let before = crate::modules::resolve(&reg.list());
        assert!(before.modules.iter().any(|m| m.enabled), "a verified aokie provides the phone");

        fs::write(dir.join("plugin.exe"), b"tampered").unwrap();
        reg.scan();
        let after = crate::modules::resolve(&reg.list());
        assert!(after.modules.iter().all(|m| !m.enabled), "a quarantined aokie provides nothing");
        let why = after.modules.iter().find_map(|m| m.reason.clone()).unwrap();
        assert!(why.contains("could not be loaded") && why.contains("Quarantined"), "{why}");
    }

    #[test]
    fn an_unsigned_plugin_is_held_back_in_a_release_build() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let trust = trust_service(&root, TrustPolicy::release(), Publishers::default());
        let mut reg = PluginRegistry::with_trust(root.path().to_path_buf(), trust);
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Disabled);
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Unsigned);
        assert!(!rec.is_loadable());
        assert!(reg.autostart_ids().is_empty());
        assert!(rec.reason.as_deref().unwrap().contains("trust this exact package"));
    }

    #[test]
    fn the_owners_unsigned_local_aokie_lists_and_autostarts_exactly_as_before_under_this_build() {
        // The dev flow: `tauri dev`, an unsigned local Aokie folder. `new` is the
        // constructor the desktop uses, so this is this build's own policy (a debug build
        // here, as under `tauri dev`), not one chosen by the test.
        let root = Root::new();
        let dir = root.path().join("aokie");
        fs::create_dir_all(dir.join("definitions")).unwrap();
        fs::write(dir.join("manifest.json"), include_str!("fixtures/aokie-v4.manifest.json")).unwrap();
        fs::write(dir.join("definitions").join("phone.json"), include_str!("fixtures/aokie-phone.definition.json")).unwrap();
        fs::write(dir.join("aokie-plugin.exe"), b"a local build").unwrap();

        let mut reg = PluginRegistry::new(root.path().to_path_buf());
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Installed, "{:?}", rec.reason);
        assert!(rec.is_loadable(), "its manifest, and so its pages, tools and setup, are there");
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::UnsignedDev);
        assert_eq!(reg.autostart_ids(), vec!["aokie".to_string()]);
        // What it brings to the product is there: the phone and the calendar.
        let resolved = crate::modules::resolve(&reg.list());
        assert!(resolved.modules.iter().all(|m| m.enabled), "{:?}", resolved.modules.iter().map(|m| (&m.id, &m.reason)).collect::<Vec<_>>());
        // Not running yet, and not refused for its package either.
        assert!(matches!(
            reg.gate("aokie", "phone.status", None),
            Err(GateRefusal::ConnectorUnavailable { state: PluginState::Installed, .. })
        ));
    }

    #[test]
    fn trusting_an_unsigned_plugin_brings_it_in_and_a_change_takes_it_out_again() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let dir = root.path().join("aokie");
        let trust = trust_service(&root, TrustPolicy::release(), Publishers::default());
        let mut reg = PluginRegistry::with_trust(root.path().to_path_buf(), trust.clone());
        reg.scan();
        assert!(!reg.get("aokie").unwrap().is_loadable());

        trust.trust_local(&dir, "aokie").unwrap();
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::TrustedLocal);
        assert_eq!(rec.state, PluginState::Installed, "startable now: {:?}", rec.reason);
        assert!(rec.is_loadable());
        assert_eq!(reg.autostart_ids(), vec!["aokie".to_string()]);

        // Any change and it is a package nobody trusted.
        fs::write(dir.join("plugin.exe"), b"a different build").unwrap();
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Unsigned);
        assert!(!rec.is_loadable());
        assert!(rec.reason.as_deref().unwrap().contains("changed since you trusted it"), "{:?}", rec.reason);

        // Uninstalling forgets the trust too, so the same bytes back again are a new decision.
        fs::write(dir.join("plugin.exe"), b"stub").unwrap();
        reg.scan();
        assert!(reg.get("aokie").unwrap().is_loadable(), "the trusted bytes are back");
        reg.forget("aokie");
        reg.scan();
        assert!(!reg.get("aokie").unwrap().is_loadable());
    }

    #[test]
    fn a_running_plugin_whose_package_stops_verifying_keeps_running_and_says_so() {
        // A file changed under a live phone line must not kill the call, but the plugin
        // is not started again, and the listing says why.
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        let dir = signed_aokie(&root, &key);
        let mut reg = release_registry(&root, &key);
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);

        fs::write(dir.join("ui-extra.txt"), b"dropped in").unwrap();
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Running);
        assert!(rec.is_loadable(), "the live process still has its commands");
        assert!(reg.gate("aokie", "call.answer", None).is_ok());
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Quarantined);
        assert!(rec.refused_by_trust());

        // When it exits, the next scan (a start begins with one) withholds it.
        reg.set_state("aokie", PluginState::Crashed, Some("Exited with code 1.".into()));
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(!rec.is_loadable());
        assert!(reg.autostart_ids().is_empty());
    }

    #[test]
    fn a_launch_that_finds_the_package_changed_withholds_the_plugin_like_a_scan_would() {
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        signed_aokie(&root, &key);
        let mut reg = release_registry(&root, &key);
        reg.scan();
        assert!(reg.get("aokie").unwrap().is_loadable());

        let verdict = crate::plugins::trust::PackageTrust {
            state: TrustState::Quarantined,
            publisher: None,
            key_id: None,
            version: None,
            reason: Some("Quarantined: digest mismatch: plugin.exe.".into()),
            trusted_at: None,
        };
        reg.withhold("aokie", verdict);
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Disabled);
        assert!(!rec.is_loadable());
        assert!(rec.reason.as_deref().unwrap().contains("digest mismatch"));
    }

    #[test]
    fn a_turned_off_plugin_that_failed_its_check_keeps_saying_so() {
        let root = Root::new();
        root.plugin("aokie", manifest("aokie"));
        let trust = trust_service(&root, TrustPolicy::release(), Publishers::default());
        let mut reg = PluginRegistry::with_trust(root.path().to_path_buf(), trust);
        reg.scan();
        reg.set_user_disabled("aokie", true);
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert!(rec.user_disabled);
        assert!(rec.reason.as_deref().unwrap().contains("trust this exact package"), "{:?}", rec.reason);
    }

    #[test]
    fn a_legacy_data_folder_in_a_signed_bundle_is_moved_out_before_the_bundle_is_checked() {
        // Older installs kept the plugin's state in <plugin>/data, inside the bundle, and
        // moved it out when the plugin started. A signed bundle with it still there would
        // be quarantined before it could start.
        let base = Root::new();
        let plugins = base.path().join("plugins");
        fs::create_dir_all(&plugins).unwrap();
        let key = TestKey::generate("fl-test-2026a");
        let dir = plugins.join("aokie");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), manifest("aokie").to_string()).unwrap();
        fs::write(dir.join("plugin.exe"), b"stub").unwrap();
        key.sign(&dir, "aokie-plugin", "0.1.0");
        fs::create_dir_all(dir.join("data")).unwrap();
        fs::write(dir.join("data").join("settings.json"), b"{\"paired\":true}").unwrap();

        let trust = TrustService::new(TrustPolicy::release(), key.pinned_for("Aokie", &["aokie"]), plugins.join("trusted-plugins.json"));
        let mut reg = PluginRegistry::with_trust(plugins.clone(), trust);
        reg.scan();

        assert_eq!(reg.get("aokie").unwrap().trust.as_ref().unwrap().state, TrustState::Verified);
        assert!(!dir.join("data").exists());
        assert_eq!(
            fs::read(crate::plugins::runner::plugin_data_dir(&dir).join("settings.json")).unwrap(),
            b"{\"paired\":true}",
            "the plugin's state moved with it"
        );
    }

    // --- the state older versions kept in `<plugin>/data` ------------------------

    /// A signed `aokie` bundle at `<base>/plugins/aokie` (so the plugin's state moves to
    /// `<base>/plugin-data/aokie`, inside the scratch), with `files` written before it is
    /// signed, and a registry that pins the key for it under a release build's rules.
    fn signed_bundle(base: &Root, files: &[(&str, &[u8])]) -> (PathBuf, PluginRegistry) {
        let plugins = base.path().join("plugins");
        let dir = plugins.join("aokie");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), manifest("aokie").to_string()).unwrap();
        fs::write(dir.join("plugin.exe"), b"stub").unwrap();
        for (rel, bytes) in files {
            fs::create_dir_all(dir.join(rel).parent().unwrap()).unwrap();
            fs::write(dir.join(rel), bytes).unwrap();
        }
        let key = TestKey::generate("fl-test-2026a");
        key.sign(&dir, "aokie-plugin", "0.1.0");
        let trust = TrustService::new(TrustPolicy::release(), key.pinned_for("Aokie", &["aokie"]), plugins.join("trusted-plugins.json"));
        (dir, PluginRegistry::with_trust(plugins, trust))
    }

    #[test]
    fn a_junction_named_data_in_a_signed_bundle_is_not_moved_into_the_plugins_data_dir() {
        // `data` as a link to a folder elsewhere: moved, the plugin's data folder would be
        // a pointer to it, and the bundle would then verify with the link gone from it.
        use crate::plugins::trust::tests::{dir_link, remove_dir_link};
        let base = Root::new();
        let (dir, mut reg) = signed_bundle(&base, &[]);
        let elsewhere = base.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("keep.txt"), b"not the plugin's").unwrap();
        if !dir_link(&dir.join("data"), &elsewhere) {
            return; // this machine cannot make one
        }

        reg.scan();
        let trust = reg.get("aokie").unwrap().trust.clone().unwrap();
        let plugin_data = crate::plugins::runner::plugin_data_dir(&dir);
        let became_a_link = fs::symlink_metadata(&plugin_data).map(|m| m.file_type().is_symlink()).unwrap_or(false);
        // Only the links are removed, before anything can fail with them in place.
        remove_dir_link(&dir.join("data"));
        if became_a_link {
            remove_dir_link(&plugin_data);
        }
        assert!(!became_a_link, "the plugin's data folder became a link to a folder outside the plugin");
        assert_eq!(trust.state, TrustState::Quarantined);
        assert!(trust.reason.unwrap().contains("a symbolic link is present: data"));
        assert_eq!(fs::read(elsewhere.join("keep.txt")).unwrap(), b"not the plugin's");
    }

    #[test]
    fn a_data_folder_the_signature_lists_stays_in_the_bundle_and_verifies() {
        let base = Root::new();
        let (dir, mut reg) = signed_bundle(&base, &[("data/models.bin", b"model bytes" as &[u8])]);
        reg.scan();
        assert_eq!(reg.get("aokie").unwrap().trust.as_ref().unwrap().state, TrustState::Verified);
        assert!(dir.join("data").join("models.bin").exists(), "signed content is not the plugin's state");
        assert!(!crate::plugins::runner::plugin_data_dir(&dir).exists());
    }

    #[test]
    fn a_package_that_fails_its_check_is_left_as_found_including_its_data_folder() {
        // Moving the state out is what makes a package verify; it is not done to one that
        // will be quarantined anyway.
        let base = Root::new();
        let (dir, mut reg) = signed_bundle(&base, &[]);
        fs::create_dir_all(dir.join("data")).unwrap();
        fs::write(dir.join("data").join("settings.json"), b"{\"paired\":true}").unwrap();
        fs::write(dir.join("plugin.exe"), b"tampered").unwrap();

        reg.scan();
        assert_eq!(reg.get("aokie").unwrap().trust.as_ref().unwrap().state, TrustState::Quarantined);
        assert!(dir.join("data").join("settings.json").exists(), "nothing was moved");
        assert!(!crate::plugins::runner::plugin_data_dir(&dir).exists());
    }

    // --- a live plugin and a folder that stopped verifying ---------------------

    /// A manifest that declares less than `manifest()`: one command, no flow capability, no event.
    fn narrow_manifest(id: &str) -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 3,
            "id": id,
            "name": format!("{id} plugin"),
            "version": "0.1.0",
            "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe" },
            "capabilities": ["connector.aokie.*"],
            "connectors": [{ "id": "aokie", "commands": ["call.answer"] }],
            "events": [],
        })
    }

    /// What `manifest()` declares and `narrow_manifest()` does not: a journalled command, the
    /// flow capability, an event.
    fn widened_by_an_edit(reg: &PluginRegistry) -> (bool, bool, bool) {
        (
            reg.gate("aokie", "sms.send", Some("k")).is_ok(),
            reg.grants("aokie", "oaiy.flow.run"),
            reg.may_emit("aokie", "aokie.call.incoming"),
        )
    }

    #[test]
    fn a_running_plugin_does_not_adopt_a_manifest_from_a_folder_that_stopped_verifying() {
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        root.plugin("aokie", narrow_manifest("aokie"));
        let dir = root.path().join("aokie");
        key.sign(&dir, "aokie-plugin", "0.1.0");
        let mut reg = release_registry(&root, &key);
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        assert_eq!(widened_by_an_edit(&reg), (false, false, false), "the signed manifest declares none of them");

        // Somebody edits manifest.json under the running plugin: another command, the flow
        // capability and an event.
        fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest("aokie")).unwrap()).unwrap();
        reg.scan();

        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.state, PluginState::Running, "a live call is not dropped");
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Quarantined);
        assert!(rec.refused_by_trust());
        assert_eq!(widened_by_an_edit(&reg), (false, false, false), "an edit no signature covers widens nothing");
        assert!(reg.gate("aokie", "call.answer", None).is_ok(), "and it still serves what it started with");
        assert_eq!(
            rec.manifest.as_ref().unwrap().connectors[0].commands,
            vec!["call.answer".to_string()],
            "the record still holds the manifest the process was started from"
        );
    }

    /// Rewrite a file with other bytes of the same length and put its modified time back:
    /// the folder's stat fingerprint does not move.
    fn swap_keeping_size_and_time(path: &Path, bytes: &[u8]) {
        let before = fs::metadata(path).unwrap().modified().unwrap();
        assert_eq!(fs::metadata(path).unwrap().len(), bytes.len() as u64);
        fs::write(path, bytes).unwrap();
        fs::File::options().write(true).open(path).unwrap().set_modified(before).unwrap();
    }

    #[test]
    fn a_manifest_swapped_behind_the_fingerprint_is_not_built_into_a_plugin() {
        // The same command name, spelled differently but at the same length: a manifest
        // that declares another command, swapped in with its time put back. A scan that
        // took the folder's fingerprint for proof would list it verified and let it be
        // called.
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        let signed = serde_json::to_string_pretty(&narrow_manifest("aokie")).unwrap();
        let swapped = signed.replace("call.answer", "sms.sendnow");
        assert_eq!(signed.len(), swapped.len());
        root.plugin("aokie", narrow_manifest("aokie"));
        let dir = root.path().join("aokie");
        key.sign(&dir, "aokie-plugin", "0.1.0");
        let mut reg = release_registry(&root, &key);
        reg.scan();
        assert_eq!(reg.get("aokie").unwrap().trust.as_ref().unwrap().state, TrustState::Verified);

        swap_keeping_size_and_time(&dir.join("manifest.json"), swapped.as_bytes());
        reg.scan();
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.trust.as_ref().unwrap().state, TrustState::Quarantined, "{:?}", rec.trust);
        assert!(!rec.is_loadable(), "nothing is built from the swapped manifest");
        assert!(reg.gate("aokie", "sms.sendnow", Some("k")).is_err());
    }

    #[test]
    fn a_launch_makes_the_record_follow_the_manifest_it_was_started_from() {
        let root = Root::new();
        let key = TestKey::generate("fl-test-2026a");
        root.plugin("aokie", narrow_manifest("aokie"));
        let dir = root.path().join("aokie");
        key.sign(&dir, "aokie-plugin", "0.1.0");
        let mut reg = release_registry(&root, &key);
        reg.scan();
        assert_eq!(widened_by_an_edit(&reg), (false, false, false));

        // A newer signed release is put in place after the scan; the launch verifies it
        // and starts from it, and the record follows.
        fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest("aokie")).unwrap()).unwrap();
        key.sign(&dir, "aokie-plugin", "0.2.0");
        let permit = reg.trust().authorize_launch(&dir, "aokie").expect("the new release verifies");
        reg.adopt_launch("aokie", &permit);
        let rec = reg.get("aokie").unwrap();
        assert_eq!(rec.trust.as_ref().unwrap().version.as_deref(), Some("0.2.0"));
        reg.set_state("aokie", PluginState::Running, None);
        assert_eq!(widened_by_an_edit(&reg), (true, true, true), "what the process was started from is what the gate allows");
    }

    #[test]
    fn an_unsigned_plugin_edited_while_it_runs_is_refreshed_in_a_developer_build_as_before() {
        // The owner's flow: `tauri dev`, a plugin folder with no signature. Nothing checks
        // it, so an edit to its manifest is picked up without a restart, as it always was.
        let root = Root::new();
        root.plugin("aokie", narrow_manifest("aokie"));
        let dir = root.path().join("aokie");
        let trust = trust_service(&root, TrustPolicy::developer(), Publishers::default());
        let mut reg = PluginRegistry::with_trust(root.path().to_path_buf(), trust);
        reg.scan();
        reg.set_state("aokie", PluginState::Running, None);
        assert_eq!(widened_by_an_edit(&reg), (false, false, false));

        fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest("aokie")).unwrap()).unwrap();
        reg.scan();
        assert_eq!(reg.get("aokie").unwrap().trust.as_ref().unwrap().state, TrustState::UnsignedDev);
        assert_eq!(widened_by_an_edit(&reg), (true, true, true), "the edit is adopted");
    }
}
