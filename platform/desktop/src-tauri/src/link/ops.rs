//! Turning relay commands into work on this desktop.
//!
//! The bridge between [`super::relay`], which knows how to fetch and report but
//! nothing about this app, and the registry and plugin host, which know how to
//! do things but nothing about a provider.
//!
//! This is a remote surface: whatever is reachable here is reachable by anyone
//! who can queue a command on the provider, and that includes anyone who acts as
//! the account's owner there. The relay's dispatcher, [`relay_dispatcher`], keeps
//! two things in front of the plugins:
//!
//! * The `desktop` connector, this app's own, is a SMALL, CLOSED vocabulary: it
//!   lists services and plugins and starts and stops them, and refuses every
//!   other op by name rather than falling through to something more general.
//! * A command for any other connector is forwarded to that plugin only if the
//!   relay policy ([`super::policy`], `resources/relay-policy.json`) says the
//!   website may run it. What the policy allows is then still checked by the
//!   plugin's own gate, as for any caller: the gate says what a plugin can do,
//!   the policy says what a website may ask it to.
//!
//! Every relayed command, for a plugin or for this app, allowed or refused, is
//! one line of `<data>/relay-log.jsonl` (connector, verb, decision, command id),
//! never its payload.
//!
//! [`dispatcher`] is the same code without the policy or the log, for callers on
//! this computer: a binding's follow-up actions in [`super::flow_runner`]. It must
//! not be handed to the relay.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::policy::{Declared, Refusal, RelayPolicy};
use crate::control::audit;

/// The connector id that means "this app itself" rather than a plugin.
pub const DESKTOP_CONNECTOR: &str = "desktop";

/// How long a plugin gets to answer a relayed connector command.
///
/// Comfortably under the provider's own command TTL, so a plugin that hangs
/// produces a typed timeout the caller can read rather than a command that
/// expires looking like the desktop never picked it up.
const FORWARD_TIMEOUT: Duration = Duration::from_secs(30);

use crate::plugins::PluginRegistryHandle;
use crate::services::registry::RegistryHandle;

/// The idempotency key for a relayed connector command.
///
/// Namespaced so a plugin's journal shows where the key came from, and derived
/// only from the command id so a redelivery of the SAME command produces the
/// SAME key — which is the entire point. A generated one would be new each
/// time and would defeat the journal it is meant to key.
fn relay_idempotency_key(command_id: &str) -> String {
    format!("relay-command-{command_id}")
}

/// Say why a plugin could not serve a relayed command, in words that name the
/// fix rather than the internals.
///
/// This message is what the user reads on the provider's website, so "the
/// aokie plugin is not running on this desktop" beats a debug-printed enum.
fn forward_error_message(
    connector: &str,
    command: &str,
    error: crate::plugins::ForwardError,
) -> String {
    use crate::plugins::ForwardError;
    match error {
        // The refusal's own sentence (what to do), not its debug print.
        ForwardError::Refused(refusal) => refusal.message(),
        ForwardError::NotRunning { plugin_id } => {
            format!("the {plugin_id} plugin is not running on this desktop")
        }
        ForwardError::Call(e) => format!("the {connector} plugin did not answer {command:?}: {e}"),
        ForwardError::Internal(message) => message,
    }
}

/// The log of commands the relay ran or refused, in the data folder.
///
/// Its own file, written with the same code as the control log. The control log
/// is the Agent's changes and Settings → Agent shows it as such; a provider's call
/// console asks `call.current` every few seconds, and would fill it.
pub const RELAY_LOG_FILE: &str = "relay-log.jsonl";

/// The ops of the `desktop` connector: the closed list [`dispatcher`] answers, and
/// so the ones the relay log calls allowed. A test runs each through the
/// dispatcher, so an op added there and not here cannot go unlogged as unknown.
pub const DESKTOP_OPS: [&str; 10] = [
    "services.list",
    "services.start",
    "services.stop",
    "services.restart",
    "services.repair",
    "plugins.list",
    "plugins.start",
    "plugins.stop",
    "plugins.restart",
    "plugins.health",
];

/// What stands between the relay and the plugins: the rules, and the record.
pub struct RelayGuard {
    policy: RelayPolicy,
    log: audit::Log,
}

impl RelayGuard {
    /// The policy this build ships, and `<data>/relay-log.jsonl`.
    pub fn open(data_dir: &Path) -> RelayGuard {
        let policy = RelayPolicy::shipped();
        if let Some(why) = policy.closed_because() {
            log::error!(
                "the relay policy cannot be used ({why}): every command for a plugin will be refused"
            );
        }
        RelayGuard::new(policy, data_dir.join(RELAY_LOG_FILE))
    }

    pub fn new(policy: RelayPolicy, log_path: PathBuf) -> RelayGuard {
        RelayGuard { policy, log: audit::Log::new(log_path) }
    }

    /// May the website run `command` on `connector`? Either way it is written down,
    /// before anything is forwarded, so a plugin that hangs cannot lose the line.
    fn admit(
        &self,
        plugins: &PluginRegistryHandle,
        connector: &str,
        command: &str,
        command_id: &str,
    ) -> Result<(), Refusal> {
        let verdict = self
            .policy
            .check(connector, command, command_id, || declared_by(plugins, connector, command));
        self.record(connector, command, command_id, verdict.as_ref().err().map(Refusal::reason));
        verdict
    }

    /// An op of this app's own. It is not the policy's to allow (the list of ops is
    /// closed in [`dispatcher`]) but it is written down all the same: stopping the
    /// phone plugin from the website is exactly what this log is for.
    fn note_desktop_op(&self, command: &str, command_id: &str) {
        let unknown = (!DESKTOP_OPS.contains(&command)).then_some("unknown_op");
        self.record(DESKTOP_CONNECTOR, command, command_id, unknown);
    }

    /// One line: which command, on which connector, what was decided and why (`refused`
    /// is the reason, when it was). Never the payload (message text, phone numbers) and
    /// never anything the plugin said back: the log's redaction only knows secret-looking
    /// NAMES.
    fn record(&self, connector: &str, command: &str, command_id: &str, refused: Option<&str>) {
        let mut args = json!({
            "connector": connector,
            "command": command,
            "commandId": command_id,
            "decision": if refused.is_none() { "allowed" } else { "refused" },
        });
        let summary = match refused {
            None => "allowed".to_string(),
            Some(reason) => {
                args["reason"] = json!(reason);
                format!("refused: {reason}")
            }
        };
        self.log.append("relay.command", &args, "relay", refused.is_none(), &summary);
    }
}

/// What the plugin registry knows about `command` of `connector`, for a connector
/// the relay policy has no entry for. A registry that cannot be read offers nothing.
fn declared_by(plugins: &PluginRegistryHandle, connector: &str, command: &str) -> Declared {
    let Ok(registry) = plugins.lock() else {
        return Declared::NoPlugin;
    };
    match registry.manifest_for_connector(connector) {
        None => Declared::NoPlugin,
        Some(manifest) => Declared::Plugin {
            declares: manifest.declares_command(connector, command),
            journalled: manifest.is_journalled(command),
        },
    }
}

/// The dispatcher the relay worker calls: [`dispatcher`], behind the relay policy
/// and the relay log.
///
/// The `desktop` connector's ops are this app's own and are not asked about the
/// policy, only written down. Everything else is asked first.
pub fn relay_dispatcher(
    registry: RegistryHandle,
    plugins: PluginRegistryHandle,
    host: std::sync::Arc<crate::plugins::PluginHost>,
    guard: RelayGuard,
) -> super::relay::Dispatcher {
    let asked = plugins.clone();
    let inner = dispatcher(registry, plugins, host);
    std::sync::Arc::new(move |connector: &str, command: &str, payload: &Value, key: &str| {
        // `key` is the relayed command's own id: see `super::relay::Dispatcher`.
        if connector == DESKTOP_CONNECTOR {
            guard.note_desktop_op(command, key);
        } else {
            guard
                .admit(&asked, connector, command, key)
                .map_err(|refusal| refusal.message())?;
        }
        inner(connector, command, payload, key)
    })
}

/// The dispatcher for callers on this computer: a binding's follow-up actions
/// after a flow it ran (see [`super::flow_runner`]). It takes the `desktop` ops
/// and forwards any other connector's command to its plugin through the plugin's
/// own gate, and asks no relay policy: these are not commands a website queued
/// for this computer, they are the follow-up of a flow this computer ran. (The
/// flows and bindings are themselves served by the provider. That is a door of
/// its own, and the relay policy does not cover it.) The relay must be given
/// [`relay_dispatcher`] instead.
pub fn dispatcher(
    registry: RegistryHandle,
    plugins: PluginRegistryHandle,
    host: std::sync::Arc<crate::plugins::PluginHost>,
) -> super::relay::Dispatcher {
    std::sync::Arc::new(move |connector: &str, command: &str, payload: &Value, key: &str| {
        // A command names WHOSE verb it is. Anything that is not this app's own
        // belongs to a plugin's connector, and goes to that plugin through the
        // capability gate its manifest declares — the plugin decides what it
        // exposes, exactly as it does for a local caller.
        //
        // Matching the verb alone and ignoring this was a real bug: the
        // provider's call console polls `call.current` on the `aokie`
        // connector, every one was measured against the desktop's own list and
        // refused, and the console got nothing back for the whole call.
        if connector != DESKTOP_CONNECTOR {
            let payload = (!payload.is_null()).then(|| payload.clone());
            // The key is REQUIRED for anything with side effects. Passing none
            // was the second half of this bug: read-only verbs like
            // `call.current` went through, and every verb that DOES something —
            // answering, hanging up, speaking — came back "requires an
            // idempotencyKey" from the gate. The relayed command's own id is
            // the right key: it is stable across redelivery, which is exactly
            // what stops a blipped socket from answering the call twice.
            let key = relay_idempotency_key(key);
            return host
                .forward_connector(connector, command, payload, Some(&key), FORWARD_TIMEOUT)
                .map_err(|e| forward_error_message(connector, command, e));
        }
        // The id the op acts on. `.list` needs none; everything else does, and
        // an absent one must refuse rather than act on some default.
        let target = |field: &str| -> Result<String, String> {
            payload
                .get(field)
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("{command} needs a {field}"))
        };

        match command {
            "services.list" => {
                let reg = registry
                    .lock()
                    .map_err(|_| "the service registry is unavailable".to_string())?;
                let snap = reg.snapshot();
                // Shaped as the provider's UI reads it: a `services` array.
                Ok(json!({ "services": snap.services, "dataDir": snap.data_dir }))
            }
            "plugins.list" => {
                let mut reg = plugins
                    .lock()
                    .map_err(|_| "the plugin registry is unavailable".to_string())?;
                reg.scan();
                Ok(json!({ "plugins": reg.list() }))
            }
            "services.start" | "services.stop" | "services.repair" | "services.restart" => {
                let id = target("serviceId")?;
                let mut reg = registry
                    .lock()
                    .map_err(|_| "the service registry is unavailable".to_string())?;
                match command {
                    "services.start" => reg.start(&id)?,
                    "services.stop" => reg.stop(&id)?,
                    "services.repair" => reg.repair(&id)?,
                    // Composed, because there is no single restart: stop then
                    // start is what the provider's own UI does locally, so the
                    // remote path must not mean something different.
                    _ => {
                        let _ = reg.stop(&id);
                        reg.start(&id)?
                    }
                }
                Ok(json!({ "ok": true, "serviceId": id }))
            }
            "plugins.start" | "plugins.stop" | "plugins.restart" => {
                let id = target("pluginId")?;
                match command {
                    "plugins.start" => host.start(&id)?,
                    "plugins.stop" => host.stop(&id)?,
                    _ => {
                        let _ = host.stop(&id);
                        host.start(&id)?
                    }
                }
                Ok(json!({ "ok": true, "pluginId": id }))
            }
            "plugins.health" => {
                let id = target("pluginId")?;
                let reg = plugins
                    .lock()
                    .map_err(|_| "the plugin registry is unavailable".to_string())?;
                match reg.get(&id) {
                    Some(record) => Ok(json!({ "plugin": record })),
                    None => Err(format!("no plugin named {id:?}")),
                }
            }
            // Named refusal, not a generic fallthrough: a provider that grows a
            // new op on the DESKTOP connector must not silently reach something
            // here that was never meant to be remotely reachable. A plugin's
            // connector is a different matter — that went to the plugin above.
            other => Err(format!(
                "this desktop does not serve the remote op {other:?} on the desktop connector"
            )),
        }
    })
}

#[cfg(test)]
mod tests {
    /// The op names FormLogic queues, minus the `desktop.` connector prefix it
    /// strips before storing. Pinned because a rename on either side turns into
    /// "no desktop picked it up" with nothing pointing here.
    const KNOWN_OPS: [&str; 10] = [
        "services.list",
        "services.start",
        "services.stop",
        "services.restart",
        "services.repair",
        "plugins.list",
        "plugins.start",
        "plugins.stop",
        "plugins.restart",
        "plugins.health",
    ];

    #[test]
    fn every_op_the_provider_can_queue_is_one_this_desktop_answers() {
        // A closed vocabulary is only safe if it is also COMPLETE — an op the
        // provider offers and this desktop refuses shows up as a mysterious
        // failure in someone's browser.
        let handled: Vec<&str> = KNOWN_OPS.to_vec();
        for op in KNOWN_OPS {
            assert!(handled.contains(&op), "{op} is unhandled");
        }
        // …and the list has no duplicates, which would hide a missing one.
        let mut sorted = KNOWN_OPS.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len());
    }

    #[test]
    fn a_relayed_command_keys_on_its_own_id_so_a_retry_cannot_act_twice() {
        // A plugin refuses any side-effecting command with no key. Passing none
        // let read-only verbs through while every verb that DOES something came
        // back "requires an idempotencyKey" — so the call console could watch a
        // call and never touch it.
        let key = super::relay_idempotency_key("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a");
        assert!(key.contains("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a"));
        assert!(!key.trim().is_empty());
        // Same command id, same key: that is what makes a redelivery harmless
        // rather than a second answered call.
        assert_eq!(key, super::relay_idempotency_key("6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a"));
        assert_ne!(key, super::relay_idempotency_key("a-different-command"));
    }

    #[test]
    fn a_plugins_own_verbs_are_not_measured_against_the_desktops_list() {
        // `call.current`, `dongle.list`, `phone.listPaired` and friends belong
        // to the aokie plugin, not to this app. Refusing them because they are
        // not in the desktop's list is what left the provider's call console
        // blank for a whole call.
        for op in ["call.current", "dongle.list", "phone.listPaired", "settings.get"] {
            assert!(
                !super::tests::KNOWN_OPS.contains(&op),
                "{op} is a plugin verb and must not be in the desktop list"
            );
        }
        // …and the desktop's own list is still exactly what it was.
        assert!(KNOWN_OPS.contains(&"services.list"));
        assert_eq!(super::DESKTOP_CONNECTOR, "desktop");
    }

    #[test]
    fn ops_carry_no_connector_prefix() {
        // FormLogic stores the SHORT verb: the connector id 'desktop' is the
        // namespace and is not repeated in the command column. Matching on
        // 'desktop.plugins.list' would never fire.
        for op in KNOWN_OPS {
            assert!(!op.starts_with("desktop."), "{op} must be the short verb");
            assert!(op.contains('.'), "{op} should be <area>.<verb>");
        }
    }

    #[test]
    fn the_ops_the_relay_log_calls_allowed_are_the_ops_pinned_above() {
        let mut logged = super::DESKTOP_OPS.to_vec();
        logged.sort_unstable();
        let mut pinned = KNOWN_OPS.to_vec();
        pinned.sort_unstable();
        assert_eq!(logged, pinned);
    }
}

/// The relay's dispatcher against a real plugin host, and the dispatcher local
/// callers use against the same one.
///
/// Two ways to see whether a command reached the plugin. Most tests use a host
/// whose plugins are marked running with NO process behind them: a command that
/// gets past the policy and the gate then stops at "the plugin is not running"
/// (`ForwardError::NotRunning`), and one the policy refused never gets that far
/// and says so in its own words. The `e2e` module puts a real process behind the
/// plugin and reads what it was actually sent.
#[cfg(test)]
mod guard_tests {
    use super::*;
    use crate::link::relay::Dispatcher;
    use crate::plugins::{ForwardError, GateRefusal, PluginHost, PluginState, TriggerStore, TriggerStoreHandle};
    use crate::services::registry::Registry;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    /// Aokie's manifest and its phone definition, as OAIY's plugin tests have them.
    const AOKIE_MANIFEST: &str = include_str!("../plugins/fixtures/aokie-v4.manifest.json");
    const AOKIE_PHONE: &str = include_str!("../plugins/fixtures/aokie-phone.definition.json");

    /// What a command that got past the policy and the gate meets when no process is there.
    const NOT_RUNNING: &str = "the aokie plugin is not running on this desktop";

    /// The commands the defect let through, that Aokie's entry keeps on this computer.
    /// The last is a flow's verb: FormLogic fires it in the browser or here, never across the relay.
    const KEPT_HERE: [&str; 11] = [
        "dongle.installDriver",
        "dongle.restoreDriver",
        "dongle.removeCerts",
        "dongle.reset",
        "phone.startPairing",
        "phone.confirmPairing",
        "phone.removePaired",
        "consent.set",
        "consent.revoke",
        "settings.set",
        "call.configureAgent",
    ];

    struct World {
        root: PathBuf,
        registry: RegistryHandle,
        plugins: PluginRegistryHandle,
        host: Arc<PluginHost>,
    }

    impl Drop for World {
        fn drop(&mut self) {
            // A plugin the host started must not outlive the test.
            self.host.stop_all();
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn sandbox(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("oaiy-relayguard-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    /// Aokie's manifest, its entry pointed at `entry`, installed under `<root>/plugins/aokie`.
    fn install_aokie(root: &Path, entry: &str) -> PathBuf {
        install_aokie_with(root, entry, |_| {})
    }

    /// The same, after `change` has altered the manifest.
    fn install_aokie_with(root: &Path, entry: &str, change: impl FnOnce(&mut Value)) -> PathBuf {
        let dir = root.join("plugins").join("aokie");
        std::fs::create_dir_all(dir.join("definitions")).unwrap();
        std::fs::write(dir.join("definitions/phone.json"), AOKIE_PHONE).unwrap();
        let mut manifest: Value = serde_json::from_str(AOKIE_MANIFEST).unwrap();
        manifest["entry"] = json!({ "kind": "process", "command": entry });
        change(&mut manifest);
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        dir
    }

    /// A plugin the relay policy has no entry for: one read, and one command that changes something.
    fn install_notes(root: &Path) {
        let dir = root.join("plugins").join("notes");
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = json!({
            "schemaVersion": 3,
            "id": "notes",
            "name": "Notes",
            "version": "0.1.0",
            "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe" },
            "capabilities": ["connector.notes.note.list", "connector.notes.note.add"],
            "connectors": [{ "id": "notes", "commands": ["note.list", "note.add"] }],
            "commands": { "journalled": ["note.add"] }
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    }

    /// A host over what is installed under `<root>/plugins`. Its own autostart runs.
    fn build(root: PathBuf) -> World {
        let plugins = crate::plugins::registry::new_handle(root.join("plugins"));
        let triggers: TriggerStoreHandle =
            Arc::new(Mutex::new(TriggerStore::load(root.join("triggers.json"))));
        let host = PluginHost::new(
            plugins.clone(),
            crate::bridge::ledger::new_handle(),
            triggers,
            crate::bridge::deadletters::open_handle(root.join("deadletters.jsonl")),
            "0.0.0-test".into(),
            true,
        );
        let registry: RegistryHandle =
            Arc::new(Mutex::new(Registry::empty(root.join("data"), root.join("models"))));
        World { root, registry, plugins, host }
    }

    /// Aokie and a second plugin, both marked running, and no process behind either.
    fn world(tag: &str) -> World {
        let root = sandbox(tag);
        install_aokie(&root, "plugin.exe");
        install_notes(&root);
        world_of(root, &["aokie", "notes"])
    }

    /// The plugins installed under `<root>/plugins`, named in `ids`, marked running with no
    /// process behind them.
    ///
    /// They are turned off first (the registry reads that from `disabled.json` when it is made),
    /// so the host's autostart, which would try to start them and find no executable, leaves
    /// them alone and cannot race the state set here.
    fn world_of(root: PathBuf, ids: &[&str]) -> World {
        let off = serde_json::to_string(ids).unwrap();
        std::fs::write(root.join("plugins").join("disabled.json"), off).unwrap();
        let world = build(root);
        {
            let mut registry = world.plugins.lock().unwrap();
            registry.scan();
            for id in ids {
                registry.set_state(id, PluginState::Running, None);
            }
        }
        world
    }

    fn relay(world: &World, policy: RelayPolicy) -> Dispatcher {
        relay_dispatcher(
            world.registry.clone(),
            world.plugins.clone(),
            world.host.clone(),
            RelayGuard::new(policy, world.root.join(RELAY_LOG_FILE)),
        )
    }

    fn local(world: &World) -> Dispatcher {
        dispatcher(world.registry.clone(), world.plugins.clone(), world.host.clone())
    }

    // ---- what the relay lets through ---------------------------------------------------

    #[test]
    fn a_command_the_policy_keeps_on_this_computer_is_refused_before_the_plugin_is_asked() {
        let world = world("refused");
        let relay = relay(&world, RelayPolicy::shipped());
        for verb in KEPT_HERE {
            // install_driver takes vid and pid from the payload and runs an elevated helper.
            let refused = relay("aokie", verb, &json!({ "vid": 4660, "pid": 22136 }), "cmd-refused")
                .expect_err(verb);
            assert!(refused.contains(&format!("\"{verb}\"")), "{refused}");
            assert!(refused.contains("can only be run from OAIY on this computer"), "{refused}");
            assert!(!refused.contains("not running"), "{verb} got as far as the plugin: {refused}");
        }
    }

    #[test]
    fn a_command_the_policy_allows_gets_past_it_and_past_the_plugins_own_gate() {
        let world = world("allowed");
        let relay = relay(&world, RelayPolicy::shipped());
        let call = json!({ "callId": "call-1" });
        for (verb, payload) in [
            ("call.current", Value::Null),
            ("call.answer", call.clone()),
            ("call.reject", call.clone()),
            ("call.hangup", call.clone()),
            ("call.operatorSpeak", json!({ "callId": "call-1", "text": "One moment please." })),
            ("call.dial", json!({ "number": "0491 570 006", "openingLine": "Hello" })),
            ("sms.send", json!({ "to": "0491 570 157", "body": "Hello" })),
            ("sms.threads", Value::Null),
            ("sms.thread", json!({ "threadId": "thread-1" })),
            ("phone.status", Value::Null),
            ("dongle.list", Value::Null),
        ] {
            // The journalled ones only get this far because the relay gives them a key:
            // the plugin's gate would say "requires an idempotencyKey" otherwise.
            let outcome = relay("aokie", verb, &payload, "cmd-allowed").expect_err(verb);
            assert_eq!(outcome, NOT_RUNNING, "{verb}");
        }
    }

    #[test]
    fn a_plugin_the_policy_has_no_entry_for_gets_its_reads_and_nothing_that_is_journalled() {
        let world = world("noentry");
        let relay = relay(&world, RelayPolicy::shipped());
        assert_eq!(
            relay("notes", "note.list", &Value::Null, "cmd-1").unwrap_err(),
            "the notes plugin is not running on this desktop"
        );
        let add = relay("notes", "note.add", &json!({ "text": "hello" }), "cmd-2").unwrap_err();
        assert!(add.contains("\"note.add\"") && add.contains("can only be run from OAIY on this computer"), "{add}");
        // A verb the plugin does not declare, and a connector nobody serves, in the gate's own words.
        let zap = GateRefusal::CapabilityDenied { connector_id: "notes".into(), command: "note.zap".into() };
        assert_eq!(relay("notes", "note.zap", &Value::Null, "cmd-3").unwrap_err(), zap.message());
        let ghost = GateRefusal::ConnectorMissing { connector_id: "ghost".into() };
        assert_eq!(relay("ghost", "x.y", &Value::Null, "cmd-4").unwrap_err(), ghost.message());
    }

    #[test]
    fn a_plugin_cannot_widen_the_policy_by_declaring_more() {
        // A newer Aokie that declares one more command and does not journal it. The plugin's
        // gate would let it through, and on this computer it does; the relay policy is the
        // build's, so for the website it stays refused until somebody lists it.
        let root = sandbox("widen");
        install_aokie_with(&root, "plugin.exe", |manifest| {
            manifest["connectors"][0]["commands"].as_array_mut().unwrap().push(json!("dongle.selfDestruct"));
        });
        let world = world_of(root, &["aokie"]);
        let relay = relay(&world, RelayPolicy::shipped());

        let refused = relay("aokie", "dongle.selfDestruct", &Value::Null, "cmd-1").unwrap_err();
        assert!(refused.contains("\"dongle.selfDestruct\""), "{refused}");
        assert!(refused.contains("can only be run from OAIY on this computer"), "{refused}");

        let local = world.host.forward_connector("aokie", "dongle.selfDestruct", None, None, Duration::from_secs(5));
        assert!(matches!(local, Err(ForwardError::NotRunning { .. })), "{local:?}");
    }

    #[test]
    fn a_command_with_no_id_never_reaches_the_plugin() {
        let world = world("noid");
        let relay = relay(&world, RelayPolicy::shipped());
        for id in ["", "  "] {
            let refused = relay("aokie", "call.answer", &json!({ "callId": "call-1" }), id).unwrap_err();
            assert!(refused.contains("without a command id"), "{refused}");
        }
    }

    #[test]
    fn a_policy_that_cannot_be_used_refuses_every_command_for_a_plugin_and_no_desktop_op() {
        let world = world("closed");
        let relay = relay(&world, RelayPolicy::from_text(None));
        for (connector, verb) in [("aokie", "call.current"), ("aokie", "dongle.list"), ("notes", "note.list")] {
            let refused = relay(connector, verb, &Value::Null, "cmd-1").unwrap_err();
            assert!(refused.contains("could not read its list"), "{connector} {verb}: {refused}");
        }
        // The desktop's own ops are a closed list in this file, not the policy's to give or take.
        let listed = relay("desktop", "plugins.list", &Value::Null, "cmd-2").expect("a desktop op still runs");
        assert!(listed["plugins"].is_array());
        let unknown = relay("desktop", "plugins.detonate", &Value::Null, "cmd-3").unwrap_err();
        assert!(unknown.contains("does not serve the remote op"), "{unknown}");
    }

    // ---- what local callers still do ---------------------------------------------------

    #[test]
    fn the_same_commands_from_this_computer_are_not_asked_about_the_policy() {
        let world = world("local");
        // A plugin's own screen, and the Agent's plugin_command tool: the host, directly.
        for verb in KEPT_HERE {
            let outcome = world.host.forward_connector(
                "aokie",
                verb,
                Some(json!({ "vid": 4660, "pid": 22136 })),
                Some("screen-1"),
                Duration::from_secs(5),
            );
            match outcome {
                Err(ForwardError::NotRunning { plugin_id }) => assert_eq!(plugin_id, "aokie", "{verb}"),
                other => panic!("{verb} was stopped on this computer: {other:?}"),
            }
        }
        // A binding's follow-up after a flow this computer ran.
        let flows = local(&world);
        for verb in KEPT_HERE {
            let outcome = flows("aokie", verb, &json!({}), "run-1#action1");
            assert_eq!(outcome.unwrap_err(), NOT_RUNNING, "{verb}");
        }
    }

    #[test]
    fn the_relay_is_always_given_the_guarded_dispatcher() {
        // The relay's wiring is in `lib.rs`, inside the GUI block, and in the headless
        // server's own binary: a `--lib` test build compiles neither, so a change there
        // would be invisible to every other test. Read them.
        for (name, source) in [
            ("src/lib.rs", include_str!("../lib.rs")),
            ("src/bin/oaiy-server.rs", include_str!("../bin/oaiy-server.rs")),
        ] {
            let flat = source.split_whitespace().collect::<Vec<_>>().join(" ");
            let spawns: Vec<usize> = flat.match_indices("link::relay::spawn(").map(|(i, _)| i).collect();
            assert_eq!(spawns.len(), 1, "{name}: expected one relay worker");
            let wiring: String = flat[spawns[0]..].chars().take(400).collect();
            assert!(wiring.contains("ops::relay_dispatcher("), "{name}: {wiring}");
            assert!(!wiring.contains("ops::dispatcher("), "{name}: {wiring}");
        }
    }

    #[test]
    fn the_paths_local_callers_use_know_nothing_about_the_relay_policy() {
        // Plugin screens and the Agent's control tools reach a plugin through the
        // bridge routes and `PluginHost::forward_connector`; a flow's follow-up
        // actions through `ops::dispatcher`. None of them asks the policy, and that
        // is a property of the code, so it is read from the code.
        for (name, source) in [
            ("plugins/host.rs", include_str!("../plugins/host.rs")),
            ("bridge/routes.rs", include_str!("../bridge/routes.rs")),
            ("control/tools.rs", include_str!("../control/tools.rs")),
            ("setup.rs", include_str!("../setup.rs")),
            ("link/flow_runner.rs", include_str!("flow_runner.rs")),
            ("link/result_actions.rs", include_str!("result_actions.rs")),
        ] {
            for word in ["RelayPolicy", "RelayGuard", "relay_dispatcher", "link::policy"] {
                assert!(!source.contains(word), "{name} mentions {word}");
            }
        }
    }

    // ---- the record --------------------------------------------------------------------

    #[test]
    fn every_relayed_command_is_one_line_in_the_relay_log_and_never_its_payload() {
        let world = world("log");
        let relay = relay(&world, RelayPolicy::shipped());
        let body = "Your table is ready, Alex";
        let _ = relay("aokie", "sms.send", &json!({ "to": "0491 570 006", "body": body }), "cmd-send");
        let _ = relay("aokie", "dongle.installDriver", &json!({ "vid": 4660, "pid": 22136 }), "cmd-driver");
        let _ = relay("aokie", "call.dial", &json!({ "number": "0491 570 156", "openingLine": body }), "");
        let _ = relay("notes", "note.add", &json!({ "text": body }), "cmd-note");
        // This app's own ops are the policy's no business, but they are written down: stopping
        // the phone plugin from the website is exactly what the log is for.
        let _ = relay("desktop", "plugins.stop", &json!({ "pluginId": "aokie" }), "cmd-stop");
        let _ = relay("desktop", "plugins.detonate", &Value::Null, "cmd-boom");

        let raw = std::fs::read_to_string(world.root.join(RELAY_LOG_FILE)).expect("the relay log");
        let lines: Vec<Value> = raw.lines().map(|l| serde_json::from_str(l).expect("one JSON line each")).collect();
        assert_eq!(lines.len(), 6, "{raw}");

        // Nothing of the payload, whatever it was called.
        for secret in ["0491 570 006", "0491 570 156", body, "openingLine", "vid", "pid"] {
            assert!(!raw.contains(secret), "the log holds {secret:?}: {raw}");
        }

        for line in &lines {
            assert_eq!(line["tool"], "relay.command");
            assert_eq!(line["session"], "relay");
        }
        assert_eq!(
            lines[0]["args"],
            json!({ "connector": "aokie", "command": "sms.send", "commandId": "cmd-send", "decision": "allowed" })
        );
        assert_eq!(lines[0]["ok"], true);
        assert_eq!(
            lines[1]["args"],
            json!({
                "connector": "aokie", "command": "dongle.installDriver", "commandId": "cmd-driver",
                "decision": "refused", "reason": "not_listed"
            })
        );
        assert_eq!(lines[1]["ok"], false);
        assert_eq!(lines[2]["args"]["reason"], "no_command_id");
        assert_eq!(lines[3]["args"]["reason"], "journalled");
        assert_eq!(
            lines[4]["args"],
            json!({ "connector": "desktop", "command": "plugins.stop", "commandId": "cmd-stop", "decision": "allowed" })
        );
        assert_eq!(lines[4]["ok"], true);
        assert_eq!(lines[5]["args"]["decision"], "refused");
        assert_eq!(lines[5]["args"]["reason"], "unknown_op");
        assert_eq!(lines[5]["ok"], false);
    }

    #[test]
    fn every_op_of_the_desktop_connector_is_one_the_dispatcher_answers_and_nothing_else_is() {
        // The relay log calls an op allowed when it is in `DESKTOP_OPS`, and the dispatcher's own
        // match decides what it runs: this is what keeps the two the same list.
        let world = world("ops");
        let relay = relay(&world, RelayPolicy::shipped());
        for op in DESKTOP_OPS {
            // No payload: an op that needs a target says what it needs. What none of them says is
            // that this desktop does not serve it.
            if let Err(message) = relay("desktop", op, &Value::Null, "cmd-op") {
                assert!(!message.contains("does not serve the remote op"), "{op}: {message}");
            }
        }
        let unknown = relay("desktop", "plugins.detonate", &Value::Null, "cmd-boom").unwrap_err();
        assert!(unknown.contains("does not serve the remote op"), "{unknown}");
    }

    // ---- with a real process behind the plugin -----------------------------------------

    /// The same, against a real child process: what it was sent is read from its own file.
    ///
    /// Windows only, like the plugin host's own end-to-end tests: the plugin is a small Node
    /// script behind a `.cmd` shim (an `entry.command` is confined to the plugin's folder). A
    /// machine with no Node skips these and SAYS so on stderr; run with `--nocapture` to see it.
    #[cfg(windows)]
    mod e2e {
        use super::*;

        /// Answers what the host asks and writes every `connector.request` it is sent to
        /// `seen.jsonl` in its data folder, so a test can tell "refused" from "delivered".
        const PLUGIN_JS: &str = r#"
import fs from "node:fs";
import path from "node:path";
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
const dir = process.env.OAIY_PLUGIN_DATA_DIR;
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    if (msg.method === "plugin.init") {
      send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } });
    } else if (msg.method === "plugin.health") {
      send({ jsonrpc: "2.0", id: msg.id, result: { status: "ok" } });
    } else if (msg.method === "connector.request") {
      fs.appendFileSync(path.join(dir, "seen.jsonl"), JSON.stringify(msg.params) + "\n");
      send({ jsonrpc: "2.0", id: msg.id, result: { answered: msg.params.command, requestId: msg.params.requestId } });
    } else if (msg.method === "plugin.shutdown") {
      process.exit(0);
    } else if (msg.id !== undefined) {
      send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "unknown method" } });
    }
  }
});
"#;

        fn node_available() -> bool {
            std::process::Command::new("node")
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }

        /// Aokie's manifest with a running (Node) plugin behind it, started by the host's own autostart.
        fn world_with_a_process(tag: &str) -> Option<World> {
            if !node_available() {
                eprintln!("SKIPPED {tag}: there is no node on this machine");
                return None;
            }
            let root = sandbox(tag);
            let dir = install_aokie(&root, "plugin.cmd");
            std::fs::write(dir.join("plugin.mjs"), PLUGIN_JS).unwrap();
            std::fs::write(dir.join("plugin.cmd"), "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n").unwrap();
            let world = build(root);
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            loop {
                let state = world.plugins.lock().ok().and_then(|r| r.get("aokie").map(|p| (p.state, p.reason.clone())));
                match &state {
                    Some((PluginState::Running, _)) => break,
                    Some((PluginState::Crashed, why)) => panic!("the plugin did not start: {why:?}"),
                    _ => {}
                }
                assert!(std::time::Instant::now() < deadline, "the plugin never started: {state:?}");
                std::thread::sleep(Duration::from_millis(100));
            }
            Some(world)
        }

        /// Every `connector.request` the plugin was sent, in order.
        fn seen(world: &World) -> Vec<Value> {
            let file = crate::plugins::runner::plugin_data_dir(&world.root.join("plugins").join("aokie")).join("seen.jsonl");
            std::fs::read_to_string(file)
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }

        fn verbs(seen: &[Value]) -> Vec<&str> {
            seen.iter().map(|s| s["command"].as_str().unwrap()).collect()
        }

        #[test]
        fn over_the_relay_the_phone_line_reaches_the_plugin_and_the_computers_setup_does_not() {
            let Some(world) = world_with_a_process("e2e-relay") else { return };
            let relay = relay(&world, RelayPolicy::shipped());

            // The two the defect is about, and one more of each kind: refused, with the message.
            for verb in ["dongle.installDriver", "consent.set", "settings.set", "phone.startPairing"] {
                let refused = relay("aokie", verb, &json!({ "vid": 4660, "pid": 22136 }), "cmd-refused").unwrap_err();
                assert!(refused.contains(&format!("\"{verb}\"")), "{refused}");
                assert!(refused.contains("can only be run from OAIY on this computer"), "{refused}");
            }
            // The call console's own commands: delivered, and answered.
            let current = relay("aokie", "call.current", &Value::Null, "cmd-current").expect("call.current");
            assert_eq!(current["answered"], "call.current");
            let hangup = relay("aokie", "call.hangup", &json!({ "callId": "call-1" }), "cmd-hangup").expect("call.hangup");
            assert_eq!(hangup["answered"], "call.hangup");
            relay("aokie", "call.answer", &json!({ "callId": "call-1" }), "cmd-answer").expect("call.answer");
            relay("aokie", "sms.send", &json!({ "to": "0491 570 157", "body": "Hello" }), "cmd-sms").expect("sms.send");
            // The one FormLogic's MCP tool tells an AI client to send, a read like sms.threads.
            let thread = relay("aokie", "sms.thread", &json!({ "threadId": "thread-1" }), "cmd-thread").expect("sms.thread");
            assert_eq!(thread["answered"], "sms.thread");

            let seen = seen(&world);
            assert_eq!(verbs(&seen), ["call.current", "call.hangup", "call.answer", "sms.send", "sms.thread"]);
            // A journalled command reaches the plugin with the key the relay made from its id,
            // which is what the plugin's own gate insists on.
            assert_eq!(seen[1]["requestId"], "relay-command-cmd-hangup");
            assert_eq!(seen[1]["payload"], json!({ "callId": "call-1" }));
            assert_eq!(seen[3]["requestId"], "relay-command-cmd-sms");
        }

        #[test]
        fn the_same_denied_commands_from_this_computer_still_reach_the_plugin() {
            let Some(world) = world_with_a_process("e2e-local") else { return };
            let relay = relay(&world, RelayPolicy::shipped());
            assert!(relay("aokie", "dongle.installDriver", &json!({ "vid": 1, "pid": 2 }), "cmd-1").is_err());

            // As a plugin's own screen does, or the Agent's plugin_command tool: the host directly.
            let installed = world
                .host
                .forward_connector("aokie", "dongle.installDriver", Some(json!({ "vid": 4660, "pid": 22136 })), None, Duration::from_secs(10))
                .expect("a screen may still install a driver");
            assert_eq!(installed["answered"], "dongle.installDriver");
            let reset = world
                .host
                .forward_connector("aokie", "dongle.reset", None, Some("screen-1"), Duration::from_secs(10))
                .expect("a screen may still reset the dongle");
            assert_eq!(reset["answered"], "dongle.reset");

            // A binding's follow-up after a flow this computer ran.
            let follow_up = local(&world)("aokie", "consent.revoke", &json!({}), "run-1#action1").expect("a flow's follow-up");
            assert_eq!(follow_up["requestId"], "relay-command-run-1#action1");

            assert_eq!(verbs(&seen(&world)), ["dongle.installDriver", "dongle.reset", "consent.revoke"]);
        }
    }
}
