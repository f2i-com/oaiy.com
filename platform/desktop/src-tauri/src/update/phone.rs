//! Whether a phone call is live, asked of the plugin that provides the phone.
//!
//! OAIY's own call route (`voice::live_call_count`) sees only the calls that reach it: those
//! the phone plugin sends through OAIY's realtime stream. A plugin that runs its own speech
//! pipeline (a "legacy" mode), a call it is screening, or one it is holding for the caller never
//! touches that route, yet stopping the plugin for an update drops all of them. So before an
//! install OAIY also asks the plugin that provides the phone module, by a read-only connector
//! command, whether a call is live. Nothing here knows any plugin by name: the provider is
//! whichever plugin the module registry says provides `phone`, and the command is the first the
//! module names ([`crate::modules::ModuleDef::live`]) that the provider's connector declares:
//!
//! - `call.switchboard`: `{foreground, waiting, parked, ...}`, each `null` or a call: the
//!   foreground call (ringing or on the line), the caller knocking, and the call on hold;
//! - `call.current`: `{call: null | {...}}`, the foreground call only.
//!
//! What comes back is one of four things ([`LineState`]):
//!
//! - **`NoPlugin`**: nothing provides the phone, or its plugin is not running (stopped, crashed, not
//!   yet started): it holds no call, and stopping it drops none;
//! - **`Idle`**: it answered, and no call is live;
//! - **`Live`**: it answered, and a call is ringing, on the line, waiting or on hold;
//! - **`Unknown`**: it is running and did not give an answer that can be read: no answer in time, an
//!   error, something that is not the shape above, or no command that says. An install does not go
//!   on while OAIY cannot tell.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::modules::{self, PHONE};
use crate::plugins::{CallError, ForwardError, PluginHost};

/// How long the plugin has to answer.
pub const ASK_TIMEOUT: Duration = Duration::from_secs(3);

/// An answer is kept this long for the status the window polls; an install asks again.
const KEEP: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineState {
    /// Nothing provides the phone, or its plugin is not running.
    NoPlugin,
    Idle,
    /// `count` calls (ringing, on the line, waiting or on hold) at the plugin `plugin`.
    Live { plugin: String, count: usize },
    /// The plugin `plugin` is running and did not say, and why.
    Unknown { plugin: String, why: String },
}

/// Something that can be asked whether the phone has a call live.
pub trait Line: Send + Sync {
    /// The state of the line. `fresh`: ask now, whatever was answered a moment ago.
    fn ask(&self, fresh: bool) -> LineState;
}

/// The phone plugin, asked through the host.
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

    /// Ask the plugin now.
    fn look(&self) -> LineState {
        let records = self.host.registry.lock().unwrap_or_else(|e| e.into_inner()).list();
        let resolved = modules::resolve(&records);
        let Some(module) = resolved.modules.iter().find(|m| m.id == PHONE).filter(|m| m.enabled) else { return LineState::NoPlugin };
        let Some(provider) = module.provider.as_ref() else { return LineState::NoPlugin };
        // A plugin that is not serving holds no call.
        if !provider.state.accepts_commands() {
            return LineState::NoPlugin;
        }
        let name = provider.name.clone();
        let unknown = |why: String| LineState::Unknown { plugin: name.clone(), why };
        let Some(connector) = provider.connector.clone() else {
            return unknown("its claim names no connector to ask".to_string());
        };
        let declared: Vec<String> = records
            .iter()
            .find(|r| r.id == provider.plugin_id)
            .and_then(|r| r.manifest.as_ref())
            .and_then(|m| m.connectors.iter().find(|c| c.id == connector))
            .map(|c| c.commands.clone())
            .unwrap_or_default();
        let live = modules::def(PHONE).map(|d| d.live).unwrap_or(&[]);
        let Some(command) = live.iter().copied().find(|c| declared.iter().any(|d| d == c)) else {
            return unknown(format!("it declares no command that says whether a call is live ({})", live.join(" or ")));
        };
        match self.host.forward_connector(&connector, command, None, None, self.timeout) {
            Ok(reply) => match crate::setup::unwrap_reply(reply).map_err(|e| format!("it answered with an error ({e})")).and_then(|data| count_calls(command, &data)) {
                Ok(0) => LineState::Idle,
                Ok(count) => LineState::Live { plugin: name, count },
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

    struct Sandbox(PathBuf);

    impl Sandbox {
        fn new(tag: &str) -> Sandbox {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!("oaiy-phone-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Sandbox(dir)
        }
    }

    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn has_node() -> bool {
        std::process::Command::new("node").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
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

    /// Put the stand-in plugin `line` in the sandbox: a phone provider whose connector declares `phone_commands` and `live_commands`.
    fn install(sb: &Sandbox, live_commands: &[&str], behavior: serde_json::Value) -> PathBuf {
        let dir = sb.0.join("plugins").join("line");
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
        let manifest = json!({
            "schemaVersion": 4, "id": "line", "name": "Line Test Plugin", "version": "0.1.0", "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": shim },
            "capabilities": commands.iter().map(|c| format!("connector.line.{c}")).collect::<Vec<_>>(),
            "connectors": [{ "id": "line", "name": "Line Test", "commands": commands }],
            "modules": { "provides": ["phone"], "connector": "line" },
            "events": [],
        });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        behave(&dir, behavior);
        dir
    }

    /// Start it and wait until the host says it is running.
    fn start(host: &Arc<PluginHost>) {
        host.registry.lock().unwrap().scan();
        host.start("line").unwrap_or_else(|e| panic!("the stand-in plugin does not start: {e}\nits log: {:?}", host.logs("line", Some(20))));
        let deadline = Instant::now() + Duration::from_secs(20);
        while host.registry.lock().unwrap().get("line").map(|r| r.state) != Some(PluginState::Running) {
            assert!(Instant::now() < deadline, "the stand-in plugin never came up: {:?}", host.registry.lock().unwrap().get("line").map(|r| r.reason.clone()));
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
        // A real updater over real probes: the phone line is the plugin.
        let line: Arc<dyn Line> = Arc::new(PluginLine::with_timeout(host.clone(), Duration::from_millis(800)));
        let probes = Probes { phone: Some(line), ..Default::default() };
        let (updater, _, now) = crate::update::updater::tests::ready();
        updater.set_activity(Arc::new(probes));
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
            // (Other tests in this process may add a reason of their own to the list; the plugin's call is in it.)
            crate::update::install::Outcome::Refused(crate::update::updater::InstallRefusal::Blocked(blockers)) => assert!(blockers.iter().any(|b| b.code == "phoneCall"), "{blockers:?}"),
            other => panic!("{other:?}"),
        }
        assert!(stopped.lock().unwrap().is_empty(), "nothing was stopped");
        assert_eq!(updater.status_at(now).state, crate::update::updater::State::Ready, "the download is kept");
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
    }
}
