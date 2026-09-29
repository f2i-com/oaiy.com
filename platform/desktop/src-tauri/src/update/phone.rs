//! Whether a phone call is live, asked of every running plugin that provides the phone.
//!
//! OAIY's own call route (`voice::live_call_count`) sees only the calls that reach it: those
//! the phone plugin sends through OAIY's realtime stream. A plugin that runs its own speech
//! pipeline (a "legacy" mode), a call it is screening, or one it is holding for the caller never
//! touches that route, yet stopping the plugin for an update drops all of them. So before an
//! install OAIY also asks EVERY running plugin that provides the phone module (by the module
//! registry's claims, whether or not OAIY's phone module is on and whether or not the plugin is
//! the provider OAIY chose: a second claimant, or one turned off in Plugins that still runs, can
//! hold a call too), by a read-only connector command, whether a call is live. Nothing here knows
//! any plugin by name: the command is the first the module names
//! ([`crate::modules::ModuleDef::live`]) that the plugin's connector declares:
//!
//! - `call.switchboard`: `{foreground, waiting, parked, ...}`, each `null` or a call: the
//!   foreground call (ringing or on the line), the caller knocking, and the call on hold;
//! - `call.current`: `{call: null | {...}}`, the foreground call only.
//!
//! What comes back is one of four things ([`LineState`]):
//!
//! - **`NoPlugin`**: no plugin that provides the phone is running (stopped, crashed, not yet
//!   started, turned off): none holds a call, and stopping them drops none;
//! - **`Idle`**: every plugin that is running answered, and no call is live;
//! - **`Live`**: one answered that a call is ringing, on the line, waiting or on hold (one is enough);
//! - **`Unknown`**: a plugin is running and did not give an answer that can be read: no answer in
//!   time, an error, something that is not the shape above, no command that says, or it is still
//!   starting. An install does not go on while OAIY cannot tell.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::modules::{self, PHONE};
use crate::plugins::{CallError, ForwardError, PluginHost, PluginState};

/// How long the plugin has to answer.
pub const ASK_TIMEOUT: Duration = Duration::from_secs(3);

/// An answer is kept this long for the status the window polls; an install asks again.
const KEEP: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineState {
    /// No plugin that provides the phone is running.
    NoPlugin,
    Idle,
    /// `count` calls (ringing, on the line, waiting or on hold) at the plugin `plugin` (the plugins, joined, when more than one has some).
    Live { plugin: String, count: usize },
    /// The plugin `plugin` is running and did not say, and why (the plugins, joined, when more than one did not).
    Unknown { plugin: String, why: String },
}

/// Something that can be asked whether the phone has a call live.
pub trait Line: Send + Sync {
    /// The state of the line. `fresh`: ask now, whatever was answered a moment ago.
    fn ask(&self, fresh: bool) -> LineState;
}

/// The phone plugins, asked through the host.
pub struct PluginLine {
    host: Arc<PluginHost>,
    timeout: Duration,
    last: Mutex<Option<(Instant, LineState)>>,
}

impl PluginLine {
    pub fn new(host: Arc<PluginHost>) -> PluginLine {
        PluginLine::with_timeout(host, ASK_TIMEOUT)
    }

    pub fn with_timeout(host: Arc<PluginHost>, timeout: Duration) -> PluginLine {
        PluginLine { host, timeout, last: Mutex::new(None) }
    }

    /// Ask the plugins now: EVERY plugin that provides the phone module and is running, whether or not OAIY's phone module is on
    /// (a plugin turned off in Plugins can still be running) and whether or not it is the provider OAIY chose (a second plugin that
    /// claims the phone loses the choice but not its calls). Any of them can hold a call the others know nothing of, and stopping
    /// it for an update drops that call just the same.
    fn look(&self) -> LineState {
        let mut records = self.host.registry.lock().unwrap_or_else(|e| e.into_inner()).list();
        records.sort_by(|a, b| a.id.cmp(&b.id));
        let mut live: Vec<(String, usize)> = Vec::new();
        let mut unknown: Vec<(String, String)> = Vec::new();
        let mut idle = false;
        for record in &records {
            let Some(manifest) = record.manifest.as_ref() else { continue };
            let claims = modules::claims(manifest);
            let Some(claim) = claims.get(PHONE).filter(|c| c.refused.is_none()) else { continue };
            let name = manifest.name.clone();
            match record.state {
                state if state.accepts_commands() => match self.ask_plugin(&name, manifest, claim.connector.as_deref()) {
                    LineState::Live { count, .. } => live.push((name, count)),
                    LineState::Unknown { why, .. } => unknown.push((name, why)),
                    LineState::Idle => idle = true,
                    LineState::NoPlugin => {}
                },
                // A process that has not answered its start yet: it may be picking up a call, and cannot say.
                PluginState::Starting => unknown.push((name, "it is still starting".to_string())),
                // No process (never started, stopped, crashed, turned off): it holds no call.
                _ => {}
            }
        }
        // Any call blocks; else any plugin that cannot say; else it is quiet, if there is a plugin to be quiet.
        if !live.is_empty() {
            let count = live.iter().map(|(_, n)| n).sum();
            return LineState::Live { plugin: live.into_iter().map(|(n, _)| n).collect::<Vec<_>>().join(" and "), count };
        }
        if !unknown.is_empty() {
            let plugin = unknown.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" and ");
            let why = if unknown.len() == 1 { unknown[0].1.clone() } else { unknown.iter().map(|(n, w)| format!("{n}: {w}")).collect::<Vec<_>>().join("; ") };
            return LineState::Unknown { plugin, why };
        }
        if idle { LineState::Idle } else { LineState::NoPlugin }
    }

    /// Ask one running plugin (`name`, `manifest`) through the connector that serves its phone claim.
    fn ask_plugin(&self, name: &str, manifest: &crate::plugins::PluginManifest, connector: Option<&str>) -> LineState {
        let unknown = |why: String| LineState::Unknown { plugin: name.to_string(), why };
        let Some(connector) = connector else {
            return unknown("its claim names no connector to ask".to_string());
        };
        let declared: Vec<&String> = manifest.connectors.iter().find(|c| c.id == connector).map(|c| c.commands.iter().collect()).unwrap_or_default();
        let live = modules::def(PHONE).map(|d| d.live).unwrap_or(&[]);
        let Some(command) = live.iter().copied().find(|c| declared.iter().any(|d| d.as_str() == *c)) else {
            return unknown(format!("it declares no command that says whether a call is live ({})", live.join(" or ")));
        };
        match self.host.forward_connector(connector, command, None, None, self.timeout) {
            Ok(reply) => match crate::setup::unwrap_reply(reply).map_err(|e| format!("it answered with an error ({e})")).and_then(|data| count_calls(command, &data)) {
                Ok(0) => LineState::Idle,
                Ok(count) => LineState::Live { plugin: name.to_string(), count },
                Err(why) => unknown(why),
            },
            // (setup's wording names its own 5 s deadline; this look has its own.)
            Err(ForwardError::Call(CallError::Timeout { .. })) => unknown(format!("it did not answer within {} s", self.timeout.as_secs_f32().max(0.1))),
            Err(e) => unknown(crate::setup::describe_forward_error(e)),
        }
    }
}

impl Line for PluginLine {
    fn ask(&self, fresh: bool) -> LineState {
        if !fresh {
            if let Some((at, state)) = self.last.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                if at.elapsed() < KEEP {
                    return state.clone();
                }
            }
        }
        let state = self.look();
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), state.clone()));
        state
    }
}

/// How many calls a plugin's answer to `command` says are live, or why the answer cannot be read.
///
/// Strict on purpose: a key that is missing, or a value that is neither `null` nor an object, is not "no call".
pub fn count_calls(command: &str, data: &Value) -> Result<usize, String> {
    let slot = |key: &str, required: bool| -> Result<usize, String> {
        match data.get(key) {
            Some(Value::Null) => Ok(0),
            Some(Value::Object(_)) => Ok(1),
            Some(_) => Err(format!("its answer to {command} has a \"{key}\" that is neither empty nor a call")),
            None if required => Err(format!("its answer to {command} has no \"{key}\"")),
            None => Ok(0),
        }
    };
    if !data.is_object() {
        return Err(format!("its answer to {command} is not an object"));
    }
    match command {
        "call.switchboard" => Ok(slot("foreground", true)? + slot("waiting", false)? + slot("parked", false)?),
        "call.current" => slot("call", true),
        other => Err(format!("{other} is not a command whose answer can be read")),
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;

    /// A Line the test sets, counting how it was asked.
    pub struct FakeLine {
        pub state: Mutex<LineState>,
        pub asked_fresh: std::sync::atomic::AtomicUsize,
        pub asked_cached: std::sync::atomic::AtomicUsize,
    }

    impl FakeLine {
        pub fn new(state: LineState) -> Arc<FakeLine> {
            Arc::new(FakeLine { state: Mutex::new(state), asked_fresh: Default::default(), asked_cached: Default::default() })
        }

        pub fn set(&self, state: LineState) {
            *self.state.lock().unwrap() = state;
        }
    }

    impl Line for FakeLine {
        fn ask(&self, fresh: bool) -> LineState {
            let counter = if fresh { &self.asked_fresh } else { &self.asked_cached };
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.state.lock().unwrap().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_switchboard_with_a_call_in_any_place_counts_it() {
        let idle = json!({"foreground": null, "waiting": null, "parked": null, "revision": 0});
        assert_eq!(count_calls("call.switchboard", &idle), Ok(0));
        let call = json!({"callId": "c1", "from": "+61491570006", "state": "ringing"});
        // Ringing (a foreground call that is not yet answered), knocking (waiting) and on hold (parked) each count.
        assert_eq!(count_calls("call.switchboard", &json!({"foreground": call, "waiting": null, "parked": null})), Ok(1));
        assert_eq!(count_calls("call.switchboard", &json!({"foreground": null, "waiting": call, "parked": null})), Ok(1));
        assert_eq!(count_calls("call.switchboard", &json!({"foreground": null, "waiting": null, "parked": call})), Ok(1));
        assert_eq!(count_calls("call.switchboard", &json!({"foreground": call, "waiting": call, "parked": call})), Ok(3));
        // A plugin that leaves out waiting and parked (an older one) still answers for the foreground.
        assert_eq!(count_calls("call.switchboard", &json!({"foreground": call})), Ok(1));
    }

    #[test]
    fn the_foreground_only_command_counts_a_call_and_an_empty_one_is_no_call() {
        assert_eq!(count_calls("call.current", &json!({"call": null, "companionMedia": null})), Ok(0));
        assert_eq!(count_calls("call.current", &json!({"call": {"callId": "c1", "state": "active"}})), Ok(1));
    }

    #[test]
    fn an_answer_that_is_not_the_shape_is_never_no_call() {
        for (command, data) in [
            ("call.switchboard", json!({})),
            ("call.switchboard", json!({"waiting": null, "parked": null})),
            ("call.switchboard", json!({"foreground": "yes"})),
            ("call.switchboard", json!({"foreground": null, "parked": 3})),
            ("call.switchboard", json!("banana")),
            ("call.switchboard", json!([])),
            ("call.switchboard", json!(null)),
            ("call.current", json!({})),
            ("call.current", json!({"call": false})),
            ("call.current", json!({"call": []})),
            ("call.current", json!(7)),
            ("phone.status", json!({"call": null})),
        ] {
            assert!(count_calls(command, &data).is_err(), "{command} {data}");
        }
    }
}

/// The phone plugin asked for real: a stand-in plugin (testdata/fake-phone-plugin.mjs) run as a child process under a real
/// plugin host, answering the way a behavior file beside it says. They need Node, as the plugin tests that start a real child do,
/// and end quietly where there is none (a missing toolchain is not a defect in the code under test).
#[cfg(test)]
mod process_tests {
    use super::fake::FakeLine;
    use super::*;
    use crate::plugins::trust::{Publishers, TrustPolicy, TrustService};
    use crate::plugins::{PluginRegistry, PluginState, TriggerStore};
    use crate::update::blockers::{compute, Activity, Probes};
    use serde_json::json;
    use std::path::{Path, PathBuf};

    const SCRIPT: &str = include_str!("testdata/fake-phone-plugin.mjs");
    /// Everything the phone module needs a connector to declare, and the two commands that say whether a call is live.
    const PHONE_COMMANDS: [&str; 5] = ["phone.status", "sms.send", "settings.get", "settings.set", "call.dial"];

    /// A folder of one test's own, and the turn to use the machine's process table: these tests start real Node children, and the whole
    /// suite runs beside them (the voice tests are timed), so at most one of them runs at a time.
    struct Sandbox(PathBuf, #[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl Sandbox {
        fn new(tag: &str) -> Sandbox {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
            let turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!("oaiy-phone-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Sandbox(dir, turn)
        }
    }

    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn has_node() -> bool {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAS.get_or_init(|| std::process::Command::new("node").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false))
    }

    /// A plugin host with an unsigned-plugin-friendly registry and nothing started.
    fn host(sb: &Sandbox) -> Arc<PluginHost> {
        let plugins = sb.0.join("plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        let trust = TrustService::new(TrustPolicy::developer(), Publishers::default(), plugins.join("trusted-plugins.json"));
        let registry = Arc::new(Mutex::new(PluginRegistry::with_trust(plugins, trust)));
        PluginHost::assemble(
            registry,
            crate::bridge::ledger::new_handle(),
            Arc::new(Mutex::new(TriggerStore::load(sb.0.join("triggers.json")))),
            crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl")),
            "0.1.0".into(),
            true,
        )
    }

    fn behave(dir: &Path, behavior: serde_json::Value) {
        std::fs::write(dir.join("behavior.json"), behavior.to_string()).unwrap();
    }

    /// Put the stand-in plugin `line` in the sandbox: a phone provider whose connector declares the phone's commands and `live_commands`.
    fn install(sb: &Sandbox, live_commands: &[&str], behavior: serde_json::Value) -> PathBuf {
        install_as(sb, "line", live_commands, behavior, Some(claim("line")))
    }

    /// The `modules` section of a plugin that provides the phone through its connector `connector`.
    fn claim(connector: &str) -> serde_json::Value {
        json!({ "provides": ["phone"], "connector": connector })
    }

    /// The same under another id. `modules` is the manifest's `modules` section (None: it has none, as Aokie's has not).
    fn install_as(sb: &Sandbox, id: &str, live_commands: &[&str], behavior: serde_json::Value, modules: Option<serde_json::Value>) -> PathBuf {
        let dir = sb.0.join("plugins").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plugin.mjs"), SCRIPT).unwrap();
        let shim = if cfg!(windows) { "plugin.cmd" } else { "plugin.sh" };
        if cfg!(windows) {
            std::fs::write(dir.join(shim), "@echo off\r\nnode \"%~dp0plugin.mjs\" %*\r\n").unwrap();
        } else {
            std::fs::write(dir.join(shim), "#!/bin/sh\nexec node \"$(dirname \"$0\")/plugin.mjs\" \"$@\"\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(dir.join(shim), std::fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        let commands: Vec<&str> = PHONE_COMMANDS.iter().copied().chain(live_commands.iter().copied()).collect();
        let name = id.chars().next().map(|first| first.to_uppercase().collect::<String>() + &id[first.len_utf8()..]).unwrap_or_default() + " Test Plugin";
        let mut manifest = json!({
            "schemaVersion": 4, "id": id, "name": name, "version": "0.1.0", "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": shim },
            "capabilities": commands.iter().map(|c| format!("connector.{id}.{c}")).collect::<Vec<_>>(),
            "connectors": [{ "id": id, "name": "Line Test", "commands": commands }],
            "events": [],
        });
        if let Some(modules) = modules {
            manifest["modules"] = modules;
        }
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        behave(&dir, behavior);
        dir
    }

    /// Start `line` and wait until the host says it is running.
    fn start(host: &Arc<PluginHost>) {
        start_as(host, "line");
    }

    fn start_as(host: &Arc<PluginHost>, id: &str) {
        // A scan makes the records again from the folders: once, when the plugin is not known yet (a running one is left as it is).
        if host.registry.lock().unwrap().get(id).is_none() {
            host.registry.lock().unwrap().scan();
        }
        host.start(id).unwrap_or_else(|e| panic!("the stand-in plugin {id} does not start: {e}\nits log: {:?}", host.logs(id, Some(20))));
        let deadline = Instant::now() + Duration::from_secs(20);
        while host.registry.lock().unwrap().get(id).map(|r| r.state) != Some(PluginState::Running) {
            assert!(Instant::now() < deadline, "the stand-in plugin {id} never came up: {:?}", host.registry.lock().unwrap().get(id).map(|r| r.reason.clone()));
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn asked(dir: &Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("asked.log")).unwrap_or_default().lines().map(str::to_string).collect()
    }

    fn call() -> serde_json::Value {
        json!({ "callId": "call_1", "from": "+61491570006", "direction": "inbound", "state": "active" })
    }

    fn idle_board() -> serde_json::Value {
        json!({ "mode": "answer", "data": { "foreground": null, "waiting": null, "parked": null, "revision": 3 } })
    }

    /// The state of the line with the plugin running and behaving as `behavior`, its connector declaring `live`.
    fn state_with(tag: &str, live: &[&str], behavior: serde_json::Value) -> Option<(LineState, Vec<String>)> {
        if !has_node() {
            return None;
        }
        let sb = Sandbox::new(tag);
        let host = host(&sb);
        let dir = install(&sb, live, behavior);
        start(&host);
        let state = PluginLine::with_timeout(host.clone(), Duration::from_millis(800)).ask(true);
        let commands = asked(&dir);
        host.stop("line").unwrap();
        Some((state, commands))
    }

    #[test]
    fn a_call_in_the_plugins_own_pipeline_that_oaiys_line_does_not_know_blocks() {
        // The plugin runs its own speech pipeline (a legacy mode): the call never reaches OAIY, whose own count is 0.
        let Some((state, commands)) = state_with("legacy", &["call.switchboard", "call.current"], json!({ "mode": "answer", "data": { "foreground": call(), "waiting": null, "parked": null } })) else { return };
        assert_eq!(state, LineState::Live { plugin: "Line Test Plugin".into(), count: 1 });
        assert_eq!(commands, ["call.switchboard"], "the authoritative command, and only it");
        // ...and through the blockers, with nothing on OAIY's own line: the install is blocked, and the message says who reported it.
        let readings = crate::update::blockers::Readings { phone: state, ..Default::default() };
        let blockers = compute(Some(&readings), Duration::from_secs(3600));
        assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["phoneCall"]);
        assert!(blockers[0].message.contains("Line Test Plugin"));
        assert!(!blockers[0].message.contains("OAIY's own line"), "it says who reported it: the plugin, not OAIY's line");
    }

    #[test]
    fn a_ringing_call_a_waiting_caller_and_a_call_on_hold_each_count() {
        for (name, board) in [
            ("ringing", json!({ "foreground": { "callId": "c", "state": "ringing" }, "waiting": null, "parked": null })),
            ("waiting", json!({ "foreground": null, "waiting": { "callId": "c", "from": "+61491570006" }, "parked": null })),
            ("held", json!({ "foreground": null, "waiting": null, "parked": { "callId": "c", "from": "+61491570006" } })),
        ] {
            let Some((state, _)) = state_with(name, &["call.switchboard"], json!({ "mode": "answer", "data": board })) else { return };
            assert_eq!(state, LineState::Live { plugin: "Line Test Plugin".into(), count: 1 }, "{name}");
        }
    }

    #[test]
    fn an_idle_plugin_does_not_block() {
        let Some((state, _)) = state_with("idle", &["call.switchboard", "call.current"], idle_board()) else { return };
        assert_eq!(state, LineState::Idle);
        let readings = crate::update::blockers::Readings { phone: state, ..Default::default() };
        assert!(compute(Some(&readings), Duration::from_secs(3600)).is_empty());
    }

    #[test]
    fn a_plugin_that_only_has_the_foreground_command_is_asked_that() {
        let Some((state, commands)) = state_with("current", &["call.current"], json!({ "mode": "answer", "data": { "call": call() } })) else { return };
        assert_eq!(state, LineState::Live { plugin: "Line Test Plugin".into(), count: 1 });
        assert_eq!(commands, ["call.current"]);
        let Some((state, _)) = state_with("current-idle", &["call.current"], json!({ "mode": "answer", "data": { "call": null } })) else { return };
        assert_eq!(state, LineState::Idle);
    }

    #[test]
    fn a_plugin_that_never_answers_blocks_and_the_reason_says_so() {
        let started = Instant::now();
        let Some((state, _)) = state_with("hang", &["call.switchboard"], json!({ "mode": "hang" })) else { return };
        match state {
            LineState::Unknown { plugin, why } => {
                assert_eq!(plugin, "Line Test Plugin");
                assert!(why.contains("did not answer"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        assert!(started.elapsed() < Duration::from_secs(15), "it gave up at its own deadline, not the plugin's");
    }

    #[test]
    fn a_plugin_that_answers_with_an_error_blocks() {
        let Some((state, _)) = state_with("error", &["call.switchboard"], json!({ "mode": "error" })) else { return };
        match state {
            LineState::Unknown { why, .. } => assert!(why.contains("the radio is not there"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_plugin_that_answers_something_unreadable_blocks() {
        for (name, data) in [("garbage", json!("banana")), ("empty", json!({})), ("odd", json!({ "foreground": 7 }))] {
            let Some((state, _)) = state_with(name, &["call.switchboard"], json!({ "mode": "answer", "data": data })) else { return };
            assert!(matches!(state, LineState::Unknown { .. }), "{name}: {state:?}");
        }
    }

    #[test]
    fn a_plugin_that_declares_no_command_that_says_blocks_and_says_which_it_looked_for() {
        let Some((state, commands)) = state_with("nocommand", &[], idle_board()) else { return };
        match state {
            LineState::Unknown { why, .. } => assert!(why.contains("call.switchboard or call.current"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(commands.is_empty(), "nothing was sent: {commands:?}");
    }

    #[test]
    fn a_stopped_plugin_and_a_plugin_that_is_not_installed_do_not_block() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("stopped");
        let host = host(&sb);
        let line = PluginLine::with_timeout(host.clone(), Duration::from_millis(800));
        // Nothing installed at all.
        assert_eq!(line.ask(true), LineState::NoPlugin);
        // Installed, never started.
        let dir = install(&sb, &["call.switchboard"], json!({ "mode": "answer", "data": { "foreground": call() } }));
        host.registry.lock().unwrap().scan();
        assert_eq!(line.ask(true), LineState::NoPlugin);
        // Running, then stopped: it holds no call.
        start(&host);
        assert!(matches!(line.ask(true), LineState::Live { .. }));
        host.stop("line").unwrap();
        assert_eq!(line.ask(true), LineState::NoPlugin);
        assert_eq!(asked(&dir).len(), 1, "a stopped plugin is not asked");
    }

    #[test]
    fn a_status_may_come_from_what_was_asked_a_moment_ago_and_a_decision_never_does() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("cache");
        let host = host(&sb);
        let dir = install(&sb, &["call.switchboard"], idle_board());
        start(&host);
        let line = PluginLine::with_timeout(host.clone(), Duration::from_millis(800));
        assert_eq!(line.ask(false), LineState::Idle);
        // A call begins.
        behave(&dir, json!({ "mode": "answer", "data": { "foreground": call(), "waiting": null, "parked": null } }));
        assert_eq!(line.ask(false), LineState::Idle, "the polled status answers from what it kept");
        assert!(matches!(line.ask(true), LineState::Live { .. }), "the decision asks the plugin again");
        assert_eq!(asked(&dir).len(), 2);
        host.stop("line").unwrap();
    }

    #[test]
    fn the_look_after_the_flush_asks_the_plugin_again_and_stops_the_install_for_a_call_that_began_meanwhile() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("flush");
        let host = host(&sb);
        let dir = install(&sb, &["call.switchboard"], idle_board());
        start(&host);
        // A real updater over the real phone line: the phone line is the plugin.
        let line: Arc<dyn Line> = Arc::new(PluginLine::with_timeout(host.clone(), Duration::from_millis(800)));
        let (updater, _, now) = crate::update::updater::tests::ready();
        updater.set_activity(Arc::new(OnlyThePhone(line)));
        let dir2 = dir.clone();
        // The Agent's flush takes seconds; a call comes in on the phone meanwhile.
        let flush = move || behave(&dir2, json!({ "mode": "answer", "data": { "foreground": call(), "waiting": null, "parked": null } }));
        let stopped = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        struct Part(Arc<Mutex<Vec<&'static str>>>);
        impl crate::update::install::Part for Part {
            fn name(&self) -> &'static str {
                "a part"
            }
            fn stop(&self) -> Result<(), String> {
                self.0.lock().unwrap().push("stop");
                Ok(())
            }
            fn start(&self) -> Result<(), String> {
                self.0.lock().unwrap().push("start");
                Ok(())
            }
        }
        let part = Part(stopped.clone());
        let parts: Vec<&dyn crate::update::install::Part> = vec![&part];
        let hand_off = |_: &crate::update::VerifiedPackage| -> Result<(), String> { panic!("the installer must not be started") };
        // The uptime the Fake gave the helper does not apply to the real probes: the clock is an hour after the start.
        let clock = move || now;
        let outcome = crate::update::install::perform(&updater, &crate::update::install::Steps { flush: &flush, parts: &parts, hand_off: &hand_off, clock: &clock });
        match outcome {
            crate::update::install::Outcome::Refused(crate::update::updater::InstallRefusal::Blocked(blockers)) => assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["phoneCall"], "{blockers:?}"),
            other => panic!("{other:?}"),
        }
        assert!(stopped.lock().unwrap().is_empty(), "nothing was stopped");
        assert_eq!(updater.status_at(now).state, crate::update::updater::State::Ready, "the download is kept");
        host.stop("line").unwrap();
    }

    /// The manifest of the phone plugin OAIY is used with, as that plugin's repository has it (a copy: testdata/README.txt).
    const REAL_MANIFEST: &str = include_str!("testdata/aokie-manifest.json");
    /// The service definition that manifest requires (the registry refuses a manifest whose definition file is missing).
    const REAL_DEFINITION: &str = include_str!("testdata/aokie-phone-definition.json");

    #[test]
    fn the_real_phone_plugins_manifest_lets_the_live_call_commands_through_the_gate_with_no_key() {
        // A command that is journalled needs an idempotency key, and this asks with none: were either live-call command
        // journalled (or not declared) in the manifest of the plugin that is really used, every user of it would be told
        // "can't tell" for ever. The manifest is read the way the registry reads it and the gate is the one connector requests go through.
        let sb = Sandbox::new("real-manifest");
        let host = host(&sb);
        let dir = sb.0.join("plugins").join("aokie");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("manifest.json"), REAL_MANIFEST).unwrap();
        std::fs::create_dir_all(dir.join("definitions")).unwrap();
        std::fs::write(dir.join("definitions").join("phone.json"), REAL_DEFINITION).unwrap();
        std::fs::write(dir.join("aokie-plugin.exe"), "not a program").unwrap();
        host.registry.lock().unwrap().scan();
        let why_not = host.registry.lock().unwrap().get("aokie").map(|r| r.reason.clone());
        host.registry.lock().unwrap().set_state("aokie", PluginState::Running, None);
        let registry = host.registry.lock().unwrap();
        let record = registry.get("aokie").expect("the plugin is found");
        let manifest = record.manifest.as_ref().unwrap_or_else(|| panic!("its manifest loads: {why_not:?}"));
        let claims = modules::claims(manifest);
        assert!(claims.provides(PHONE), "{claims:?}");
        let connector = claims.get(PHONE).and_then(|c| c.connector.clone()).expect("the claim names the connector that serves it");
        let live = modules::def(PHONE).unwrap().live;
        // What OAIY would ask: the first command the connector declares.
        assert_eq!(live.iter().find(|c| manifest.declares_command(&connector, c)), Some(&"call.switchboard"));
        for command in live {
            assert!(registry.gate(&connector, command, None).is_ok(), "{command}: {:?}", registry.gate(&connector, command, None).err());
        }
        // The same gate does refuse a journalled command asked without a key: what this test relies on is real.
        assert!(matches!(registry.gate(&connector, "call.answer", None), Err(crate::plugins::GateRefusal::IdempotencyRequired { .. })));
    }

    /// What an install asks, with the phone line the real one and nothing else read: the real `Probes` also read what the whole
    /// process shares (the call hub, the agent's tasks, plugin installs), which other tests in this process set and clear.
    struct OnlyThePhone(Arc<dyn Line>);

    impl crate::update::blockers::Activity for OnlyThePhone {
        fn read(&self, fresh: bool) -> crate::update::blockers::Readings {
            crate::update::blockers::Readings { phone: self.0.ask(fresh), ..Default::default() }
        }

        fn calls(&self) -> crate::update::blockers::CallReadings {
            crate::update::blockers::CallReadings { hub_calls: 0, phone: self.0.ask(true) }
        }
    }

    fn board(foreground: Option<serde_json::Value>) -> serde_json::Value {
        json!({ "mode": "answer", "data": { "foreground": foreground, "waiting": null, "parked": null } })
    }

    #[test]
    fn a_plugin_turned_off_in_plugins_that_is_still_running_is_still_asked() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("disabled");
        let host = host(&sb);
        let dir = install(&sb, &["call.switchboard"], board(Some(call())));
        start(&host);
        // Turned off in Plugins while its process runs (a rescan leaves a running plugin running): OAIY's phone module is off, the call is not.
        {
            let mut registry = host.registry.lock().unwrap();
            registry.set_user_disabled("line", true);
            registry.set_state("line", PluginState::Running, None);
        }
        let records = host.registry.lock().unwrap().list();
        let phone = modules::resolve(&records).modules.into_iter().find(|m| m.id == PHONE).unwrap();
        assert!(!phone.enabled, "the phone module is off: {phone:?}");
        let state = PluginLine::with_timeout(host.clone(), Duration::from_millis(800)).ask(true);
        assert_eq!(state, LineState::Live { plugin: "Line Test Plugin".into(), count: 1 });
        assert_eq!(asked(&dir), ["call.switchboard"]);
        host.stop("line").unwrap();
    }

    #[test]
    fn every_running_plugin_that_provides_the_phone_is_asked_not_only_the_one_oaiy_chose() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("two");
        let host = host(&sb);
        let a = install_as(&sb, "a-line", &["call.switchboard"], idle_board(), Some(claim("a-line")));
        let b = install_as(&sb, "b-line", &["call.switchboard"], idle_board(), Some(claim("b-line")));
        start_as(&host, "a-line");
        start_as(&host, "b-line");
        let records = host.registry.lock().unwrap().list();
        let phone = modules::resolve(&records).modules.into_iter().find(|m| m.id == PHONE).unwrap();
        assert_eq!(phone.provider.as_ref().unwrap().plugin_id, "a-line", "OAIY uses the lowest id");
        let line = PluginLine::with_timeout(host.clone(), Duration::from_millis(800));
        assert_eq!(line.ask(true), LineState::Idle);
        // The one OAIY did not choose holds a call.
        behave(&b, board(Some(call())));
        assert_eq!(line.ask(true), LineState::Live { plugin: "B-line Test Plugin".into(), count: 1 });
        // The one it chose does, and the other does not.
        behave(&b, idle_board());
        behave(&a, board(Some(call())));
        assert_eq!(line.ask(true), LineState::Live { plugin: "A-line Test Plugin".into(), count: 1 });
        // Both do: both are named and counted.
        behave(&b, board(Some(call())));
        assert_eq!(line.ask(true), LineState::Live { plugin: "A-line Test Plugin and B-line Test Plugin".into(), count: 2 });
        // One cannot say and the other is quiet: that blocks, and names the one that cannot say.
        behave(&a, idle_board());
        behave(&b, json!({ "mode": "hang" }));
        match line.ask(true) {
            LineState::Unknown { plugin, why } => {
                assert_eq!(plugin, "B-line Test Plugin");
                assert!(why.contains("did not answer"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        // One cannot say and the other has a call: a call is what is said.
        behave(&a, board(Some(call())));
        assert!(matches!(line.ask(true), LineState::Live { .. }));
        host.stop("a-line").unwrap();
        host.stop("b-line").unwrap();
    }

    #[test]
    fn a_plugin_that_does_not_claim_the_phone_is_never_asked_and_aokies_old_rule_still_counts() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("claims");
        let host = host(&sb);
        let line = PluginLine::with_timeout(host.clone(), Duration::from_millis(800));
        let other = install_as(&sb, "other", &["call.switchboard"], board(Some(call())), Some(json!({ "provides": [] })));
        start_as(&host, "other");
        assert_eq!(line.ask(true), LineState::NoPlugin);
        assert!(asked(&other).is_empty(), "a plugin that does not provide the phone is not asked");
        host.stop("other").unwrap();
        // Aokie predates the modules section: a plugin with that id and no section provides the phone.
        let aokie = install_as(&sb, "aokie", &["call.switchboard"], board(Some(call())), None);
        start_as(&host, "aokie");
        assert_eq!(line.ask(true), LineState::Live { plugin: "Aokie Test Plugin".into(), count: 1 });
        assert_eq!(asked(&aokie), ["call.switchboard"]);
        host.stop("aokie").unwrap();
    }

    #[test]
    fn a_plugin_that_is_still_starting_cannot_say_and_one_that_is_not_running_holds_no_call() {
        let sb = Sandbox::new("states");
        let host = host(&sb);
        install(&sb, &["call.switchboard"], board(Some(call())));
        host.registry.lock().unwrap().scan();
        let line = PluginLine::with_timeout(host.clone(), Duration::from_millis(800));
        host.registry.lock().unwrap().set_state("line", PluginState::Starting, None);
        match line.ask(true) {
            LineState::Unknown { plugin, why } => assert_eq!((plugin.as_str(), why.as_str()), ("Line Test Plugin", "it is still starting")),
            other => panic!("{other:?}"),
        }
        // The plugin is not asked at all in these: no process, so no call.
        for state in [PluginState::Stopped, PluginState::Crashed, PluginState::Installed, PluginState::Disabled] {
            host.registry.lock().unwrap().set_state("line", state, None);
            assert_eq!(line.ask(true), LineState::NoPlugin, "{state:?}");
        }
    }

    #[test]
    fn a_call_that_begins_while_the_engines_are_stopping_is_seen_by_the_look_before_the_plugins_stop() {
        if !has_node() {
            return;
        }
        let sb = Sandbox::new("last-look");
        let host = host(&sb);
        let dir = install(&sb, &["call.switchboard"], idle_board());
        start(&host);
        // The real updater over the real phone line, and two stand-in parts: the first (the engines) takes its time to stop, and in that
        // time a call reaches the phone plugin; the second is the plugins, which hold the phone.
        let line: Arc<dyn Line> = Arc::new(PluginLine::with_timeout(host.clone(), Duration::from_millis(800)));
        let (updater, _, now) = crate::update::updater::tests::ready();
        updater.set_activity(Arc::new(OnlyThePhone(line)));
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        struct StandIn {
            name: &'static str,
            log: Arc<Mutex<Vec<String>>>,
            on_stop: Option<Box<dyn Fn() + Send + Sync>>,
            holds_calls: bool,
        }
        impl crate::update::install::Part for StandIn {
            fn name(&self) -> &'static str {
                self.name
            }
            fn stop(&self) -> Result<(), String> {
                self.log.lock().unwrap().push(format!("stop {}", self.name));
                if let Some(on_stop) = &self.on_stop {
                    on_stop();
                }
                Ok(())
            }
            fn start(&self) -> Result<(), String> {
                self.log.lock().unwrap().push(format!("start {}", self.name));
                Ok(())
            }
            fn holds_calls(&self) -> bool {
                self.holds_calls
            }
        }
        let dir2 = dir.clone();
        let engines = StandIn { name: "the engines", log: log.clone(), on_stop: Some(Box::new(move || behave(&dir2, board(Some(call()))))), holds_calls: false };
        let plugins = StandIn { name: "the plugins", log: log.clone(), on_stop: None, holds_calls: true };
        let parts: Vec<&dyn crate::update::install::Part> = vec![&engines, &plugins];
        let hand_off = |_: &crate::update::VerifiedPackage| -> Result<(), String> { panic!("the installer must not be started") };
        let (flush, clock) = (|| {}, move || now);
        let outcome = crate::update::install::perform(&updater, &crate::update::install::Steps { flush: &flush, parts: &parts, hand_off: &hand_off, clock: &clock });
        match outcome {
            crate::update::install::Outcome::Refused(crate::update::updater::InstallRefusal::Blocked(blockers)) => assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["phoneCall"], "{blockers:?}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(*log.lock().unwrap(), ["stop the engines", "start the engines"], "the plugins were never stopped, and the engines are running again");
        assert_eq!(updater.status_at(now).state, crate::update::updater::State::Ready);
        host.stop("line").unwrap();
    }

    #[test]
    fn the_blockers_read_a_live_plugin_call_even_when_the_call_hub_knows_of_none() {
        let line = FakeLine::new(LineState::Live { plugin: "Line Test Plugin".into(), count: 2 });
        let probes = Probes { phone: Some(line.clone()), ..Default::default() };
        let readings = probes.read(true);
        assert_eq!(readings.phone, LineState::Live { plugin: "Line Test Plugin".into(), count: 2 });
        assert_eq!(line.asked_fresh.load(std::sync::atomic::Ordering::SeqCst), 1);
        let readings = probes.read(false);
        assert_eq!(readings.phone, LineState::Live { plugin: "Line Test Plugin".into(), count: 2 });
        assert_eq!(line.asked_cached.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The look before the plugins stop reads the phone line afresh too (the hub is shared with other tests, so only the line is compared).
        let calls = probes.calls();
        assert_eq!(calls.phone, LineState::Live { plugin: "Line Test Plugin".into(), count: 2 });
        assert_eq!(line.asked_fresh.load(std::sync::atomic::Ordering::SeqCst), 2, "the look asked afresh, whatever was kept");
    }
}
