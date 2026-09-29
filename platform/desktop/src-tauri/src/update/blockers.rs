//! What must stop an update from being installed, and why.
//!
//! OAIY answers a phone line, so an update must not interrupt a live call, and it must not
//! throw away work in flight either. [`compute`] turns what the app is doing into the list of
//! reasons the install button is off, each in words the owner can read; an empty list means
//! nothing is in the way. Every reason blocks by itself.
//!
//! What is checked for a call is exactly two things (see [`super::phone`]): the calls on OAIY's own
//! call route (the realtime stream the phone plugin uses when it hands a call's speech to OAIY), and
//! what every running plugin that provides the phone says when it is asked, by a read-only command, whether a
//! call is ringing, on the line, waiting or on hold. A running phone plugin that cannot be asked, or
//! answers something unreadable, blocks too: OAIY does not guess that no call is live.
//!
//! What the app is doing comes from an [`Activity`]: the real one ([`Probes`]) asks the call hub, the
//! phone plugin, the agent's task list, the downloader, the engines and the service registry; the
//! tests give a fake one. A status the window polls may be served from a short cache; the decision to
//! install is not (`fresh`).

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use super::phone::{Line, LineState};
use crate::services::downloads::DownloadsHandle;
use crate::services::node_runtime::NodeHandle;
use crate::services::python::PythonHandle;
use crate::services::registry::RegistryHandle;

/// OAIY has to have been up this long before it can restart itself: right after a start, services are
/// still coming up (a model loading, a phone plugin connecting), and a restart then would be a loop.
pub const MIN_UPTIME: Duration = Duration::from_secs(120);

/// One reason an update cannot be installed now.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Blocker {
    /// Stable name of the reason: `call`, `phoneCall`, `callUnknown`, `agentTask`, `download`, `mediaJob`,
    /// `enginesUnknown`, `installing`, `migration`, `starting`.
    pub code: &'static str,
    /// The reason, for the person to read.
    pub message: String,
}

impl Blocker {
    fn new(code: &'static str, message: impl Into<String>) -> Blocker {
        Blocker { code, message: message.into() }
    }
}

/// What the engines are doing that a restart would end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnginesState {
    /// They answered (or are not running, which is the same): media jobs and catalog downloads running.
    Known { media: usize, downloads: usize },
    /// They are running and did not answer, and why: not the same as idle.
    Unknown(String),
}

/// Everything an update looks at, read at one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Readings {
    /// Calls on OAIY's own call route.
    pub hub_calls: usize,
    /// The phone plugin's word on whether a call is live.
    pub phone: LineState,
    /// Tasks a flow gave the agent that it has not answered.
    pub agent_tasks: usize,
    /// Model or file downloads queued or running in OAIY's own downloader (a paused one is not: it resumes).
    pub downloads: usize,
    pub engines: EnginesState,
    /// Things being installed now (a service, a plugin, Python, Node), by name.
    pub installing: Vec<String>,
    /// The data folder being moved.
    pub migrating: bool,
}

impl Default for Readings {
    /// A quiet app with no phone plugin.
    fn default() -> Readings {
        Readings { hub_calls: 0, phone: LineState::NoPlugin, agent_tasks: 0, downloads: 0, engines: EnginesState::Known { media: 0, downloads: 0 }, installing: Vec::new(), migrating: false }
    }
}

/// What the call sources say, and nothing else: the look an install takes just before the plugins are stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallReadings {
    /// Calls on OAIY's own call route.
    pub hub_calls: usize,
    /// The phone plugins' word on whether a call is live.
    pub phone: LineState,
}

/// What the app is doing, as far as an update cares.
pub trait Activity: Send + Sync {
    /// Look at everything. `fresh`: ask every source now (an install), rather than accept an answer kept a few seconds (the status a window polls).
    fn read(&self, fresh: bool) -> Readings;

    /// Only the call sources, always asked afresh. An install takes this look right before it stops the plugins, when the engines and
    /// the script host are already stopped: what asks the engines a question would find them gone, and a call is all that matters then.
    fn calls(&self) -> CallReadings;
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 { one.to_string() } else { many.replace("{n}", &n.to_string()) }
}

/// The reasons an install must wait, given what the app is doing (None: nothing is known yet, which is
/// still starting up) and how long it has been running.
pub fn compute(readings: Option<&Readings>, uptime: Duration) -> Vec<Blocker> {
    let mut out = Vec::new();
    let Some(r) = readings else {
        out.push(Blocker::new("starting", "OAIY is still starting up. Try again in a minute."));
        return out;
    };
    out.extend(call_blockers(&CallReadings { hub_calls: r.hub_calls, phone: r.phone.clone() }));
    if r.agent_tasks > 0 {
        out.push(Blocker::new("agentTask", plural(r.agent_tasks, "The Agent is working on a task from a flow.", "The Agent is working on {n} tasks from flows.")));
    }
    let engine_downloads = match &r.engines {
        EnginesState::Known { downloads, .. } => *downloads,
        EnginesState::Unknown(_) => 0,
    };
    let downloads = r.downloads + engine_downloads;
    if downloads > 0 {
        out.push(Blocker::new("download", plural(downloads, "A model or file is downloading.", "{n} models or files are downloading.")));
    }
    match &r.engines {
        EnginesState::Known { media, .. } if *media > 0 => {
            out.push(Blocker::new("mediaJob", plural(*media, "The engines are making a picture, video or other media.", "The engines are making {n} pictures, videos or other media.")));
        }
        EnginesState::Unknown(why) => out.push(Blocker::new("enginesUnknown", format!("OAIY can't tell whether the engines are busy: {why}. It does not restart while it can't tell."))),
        _ => {}
    }
    if !r.installing.is_empty() {
        out.push(Blocker::new("installing", format!("Something is installing: {}.", r.installing.join(", "))));
    }
    if r.migrating {
        out.push(Blocker::new("migration", "The data folder is being moved."));
    }
    if uptime < MIN_UPTIME {
        let left = (MIN_UPTIME - uptime).as_secs().max(1);
        out.push(Blocker::new("starting", format!("OAIY started less than {} minutes ago; it can update in {left} seconds.", MIN_UPTIME.as_secs() / 60)));
    }
    out
}

/// The reasons the calls give: one on OAIY's own line, one a phone plugin reports, a phone plugin that cannot say.
pub fn call_blockers(r: &CallReadings) -> Vec<Blocker> {
    let mut out = Vec::new();
    if r.hub_calls > 0 {
        out.push(Blocker::new("call", plural(r.hub_calls, "A phone call is in progress on OAIY's own line.", "{n} phone calls are in progress on OAIY's own line.")));
    }
    match &r.phone {
        LineState::NoPlugin | LineState::Idle => {}
        LineState::Live { plugin, count } => out.push(Blocker::new(
            "phoneCall",
            plural(*count, &format!("{plugin} reports a phone call (ringing, in progress or on hold)."), &format!("{plugin} reports {{n}} phone calls (ringing, in progress or on hold).")),
        )),
        LineState::Unknown { plugin, why } => out.push(Blocker::new(
            "callUnknown",
            format!("OAIY can't tell whether a phone call is live: {plugin} did not give an answer ({why}). It does not restart while it can't tell; stopping that plugin (Connections, Plugins, Stop) lets it."),
        )),
    }
    out
}

/// What the running app is doing: the handles it asks. A part not attached counts as idle. (An updater
/// with no [`Activity`] at all is "still starting": see [`compute`].)
#[derive(Clone, Default)]
pub struct Probes {
    pub downloads: Option<DownloadsHandle>,
    pub registry: Option<RegistryHandle>,
    pub python: Option<PythonHandle>,
    pub node: Option<NodeHandle>,
    /// The engines' media jobs and catalog downloads in progress (the desktop only: there are no engines in the headless server).
    /// Asked with `fresh` when an install is decided.
    pub engines: Option<Arc<dyn Fn(bool) -> EnginesState + Send + Sync>>,
    /// The data folder is being moved.
    pub migration: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// The plugin that provides the phone.
    pub phone: Option<Arc<dyn Line>>,
}

impl Activity for Probes {
    fn read(&self, fresh: bool) -> Readings {
        let mut installing: Vec<String> = Vec::new();
        if let Some(registry) = &self.registry {
            // A poisoned lock is read anyway: this only asks a question.
            let registry = registry.lock().unwrap_or_else(|e| e.into_inner());
            installing.extend(registry.installing_ids().into_iter().map(|id| format!("the service {id}")));
        }
        if self.python.as_ref().is_some_and(|p| p.job_running()) {
            installing.push("Python".to_string());
        }
        if self.node.as_ref().is_some_and(|n| n.is_installing()) {
            installing.push("Node".to_string());
        }
        if crate::plugins::install::in_progress() {
            installing.push("a plugin".to_string());
        }
        Readings {
            hub_calls: crate::voice::live_call_count(),
            phone: self.phone.as_ref().map_or(LineState::NoPlugin, |line| line.ask(fresh)),
            agent_tasks: crate::agent_tasks::pending_count(),
            downloads: self.downloads.as_ref().map_or(0, |d| d.active_count()),
            engines: self.engines.as_ref().map_or(EnginesState::Known { media: 0, downloads: 0 }, |f| f(fresh)),
            installing,
            migrating: self.migration.as_ref().is_some_and(|f| f()),
        }
    }

    fn calls(&self) -> CallReadings {
        CallReadings { hub_calls: crate::voice::live_call_count(), phone: self.phone.as_ref().map_or(LineState::NoPlugin, |line| line.ask(true)) }
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// An Activity the test sets up; it counts how often it was read and how often that was `fresh`.
    #[derive(Default)]
    pub struct Fake {
        pub state: Mutex<Readings>,
        pub reads: AtomicUsize,
        pub fresh_reads: AtomicUsize,
        /// How often only the calls were looked at (just before the plugins stop).
        pub call_reads: AtomicUsize,
    }

    /// The parts of [`Readings`] the tests set one at a time (the old names kept, so a test reads as before).
    #[derive(Default, Clone)]
    pub struct FakeState {
        pub calls: usize,
        pub tasks: usize,
        pub downloads: usize,
        pub media: usize,
        pub installing: Vec<String>,
        pub migrating: bool,
        pub phone: Option<LineState>,
        pub engines_unknown: Option<String>,
    }

    impl Fake {
        /// Change what is read.
        pub fn set(&self, f: impl FnOnce(&mut FakeState)) {
            let mut s = FakeState::default();
            {
                let current = self.state.lock().unwrap();
                s.calls = current.hub_calls;
                s.tasks = current.agent_tasks;
                s.downloads = current.downloads;
                s.media = match &current.engines {
                    EnginesState::Known { media, .. } => *media,
                    EnginesState::Unknown(_) => 0,
                };
                s.installing = current.installing.clone();
                s.migrating = current.migrating;
                s.phone = Some(current.phone.clone());
                s.engines_unknown = match &current.engines {
                    EnginesState::Unknown(why) => Some(why.clone()),
                    _ => None,
                };
            }
            f(&mut s);
            let mut current = self.state.lock().unwrap();
            current.hub_calls = s.calls;
            current.agent_tasks = s.tasks;
            current.downloads = s.downloads;
            current.engines = match s.engines_unknown {
                Some(why) => EnginesState::Unknown(why),
                None => EnginesState::Known { media: s.media, downloads: 0 },
            };
            current.installing = s.installing;
            current.migrating = s.migrating;
            current.phone = s.phone.unwrap_or(LineState::NoPlugin);
        }
    }

    impl Activity for Fake {
        fn read(&self, fresh: bool) -> Readings {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if fresh {
                self.fresh_reads.fetch_add(1, Ordering::SeqCst);
            }
            self.state.lock().unwrap().clone()
        }

        fn calls(&self) -> CallReadings {
            self.call_reads.fetch_add(1, Ordering::SeqCst);
            let state = self.state.lock().unwrap();
            CallReadings { hub_calls: state.hub_calls, phone: state.phone.clone() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::Fake;
    use super::*;

    const LONG: Duration = Duration::from_secs(3600);

    fn codes(activity: &Fake, uptime: Duration) -> Vec<&'static str> {
        compute(Some(&activity.read(false)), uptime).iter().map(|b| b.code).collect()
    }

    fn blockers(activity: &Fake, uptime: Duration) -> Vec<Blocker> {
        compute(Some(&activity.read(false)), uptime)
    }

    fn live(count: usize) -> LineState {
        LineState::Live { plugin: "Aokie Phone Bridge".into(), count }
    }

    fn unknown(why: &str) -> LineState {
        LineState::Unknown { plugin: "Aokie Phone Bridge".into(), why: why.into() }
    }

    #[test]
    fn nothing_going_on_and_up_for_a_while_blocks_nothing() {
        assert!(blockers(&Fake::default(), LONG).is_empty());
        assert!(blockers(&Fake::default(), MIN_UPTIME).is_empty());
    }

    #[test]
    fn a_live_call_on_oaiys_own_line_blocks_by_itself_and_says_where() {
        let a = Fake::default();
        a.set(|s| s.calls = 1);
        let b = blockers(&a, LONG);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].code, "call");
        assert!(b[0].message.contains("phone call") && b[0].message.contains("OAIY's own line"), "{}", b[0].message);
        a.set(|s| s.calls = 2);
        assert!(blockers(&a, LONG)[0].message.starts_with("2 phone calls"));
    }

    #[test]
    fn a_call_the_phone_plugin_reports_blocks_by_itself_and_names_the_plugin() {
        // A call in the plugin's own pipeline never reaches OAIY's line: the hub says 0.
        let a = Fake::default();
        a.set(|s| s.phone = Some(live(1)));
        let b = blockers(&a, LONG);
        assert_eq!(b.iter().map(|b| b.code).collect::<Vec<_>>(), ["phoneCall"]);
        assert!(b[0].message.contains("Aokie Phone Bridge") && b[0].message.contains("ringing, in progress or on hold"), "{}", b[0].message);
        a.set(|s| s.phone = Some(live(2)));
        assert!(blockers(&a, LONG)[0].message.contains("2 phone calls"));
    }

    #[test]
    fn a_phone_plugin_that_cannot_say_blocks_and_says_it_cannot_tell() {
        let a = Fake::default();
        a.set(|s| s.phone = Some(unknown("the plugin did not answer call.switchboard within 3.0s")));
        let b = blockers(&a, LONG);
        assert_eq!(b.iter().map(|b| b.code).collect::<Vec<_>>(), ["callUnknown"]);
        assert!(b[0].message.starts_with("OAIY can't tell whether a phone call is live"), "{}", b[0].message);
        // It names the way out, step by step: the page, the tab, the button (PluginsPanel's Stop).
        assert!(b[0].message.ends_with("stopping that plugin (Connections, Plugins, Stop) lets it."), "{}", b[0].message);
        assert!(b[0].message.contains("did not answer call.switchboard"), "{}", b[0].message);
    }

    #[test]
    fn an_idle_phone_plugin_and_no_phone_plugin_block_nothing() {
        for state in [LineState::Idle, LineState::NoPlugin] {
            let a = Fake::default();
            a.set(|s| s.phone = Some(state.clone()));
            assert!(blockers(&a, LONG).is_empty(), "{state:?}");
        }
    }

    #[test]
    fn a_running_agent_task_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.tasks = 1);
        assert_eq!(codes(&a, LONG), ["agentTask"]);
        assert!(blockers(&a, LONG)[0].message.contains("Agent"));
    }

    #[test]
    fn a_download_in_progress_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.downloads = 3);
        assert_eq!(codes(&a, LONG), ["download"]);
        assert!(blockers(&a, LONG)[0].message.starts_with("3 models or files"));
    }

    #[test]
    fn a_download_the_engines_make_blocks_like_one_of_ours() {
        let a = Fake::default();
        a.state.lock().unwrap().engines = EnginesState::Known { media: 0, downloads: 2 };
        assert_eq!(codes(&a, LONG), ["download"]);
    }

    #[test]
    fn an_engine_media_job_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.media = 1);
        assert_eq!(codes(&a, LONG), ["mediaJob"]);
    }

    #[test]
    fn engines_that_are_running_and_do_not_answer_block_and_say_they_cannot_tell() {
        let a = Fake::default();
        a.set(|s| s.engines_unknown = Some("they did not answer within 2 seconds".into()));
        let b = blockers(&a, LONG);
        assert_eq!(b.iter().map(|b| b.code).collect::<Vec<_>>(), ["enginesUnknown"]);
        assert!(b[0].message.contains("can't tell") && b[0].message.contains("did not answer within 2 seconds"), "{}", b[0].message);
    }

    #[test]
    fn a_service_or_plugin_installing_blocks_by_itself_and_is_named() {
        let a = Fake::default();
        a.set(|s| s.installing = vec!["the service comfyui".into(), "a plugin".into()]);
        let b = blockers(&a, LONG);
        assert_eq!(b.iter().map(|b| b.code).collect::<Vec<_>>(), ["installing"]);
        assert!(b[0].message.contains("the service comfyui") && b[0].message.contains("a plugin"));
    }

    #[test]
    fn a_data_folder_move_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.migrating = true);
        assert_eq!(codes(&a, LONG), ["migration"]);
    }

    #[test]
    fn an_app_that_started_less_than_two_minutes_ago_blocks_by_itself() {
        let a = Fake::default();
        assert_eq!(codes(&a, Duration::from_secs(0)), ["starting"]);
        assert_eq!(codes(&a, Duration::from_secs(119)), ["starting"]);
        assert!(codes(&a, Duration::from_secs(120)).is_empty());
        let message = &blockers(&a, Duration::from_secs(90))[0].message;
        assert!(message.contains("2 minutes") && message.contains("30 seconds"), "{message}");
    }

    #[test]
    fn nothing_known_yet_is_still_starting_and_blocks() {
        let blockers = compute(None, LONG);
        assert_eq!(blockers.len(), 1);
        assert_eq!(blockers[0].code, "starting");
    }

    #[test]
    fn every_reason_at_once_is_listed_in_the_order_a_person_would_care() {
        let a = Fake::default();
        a.set(|s| {
            s.calls = 1;
            s.phone = Some(live(1));
            s.tasks = 1;
            s.downloads = 1;
            s.media = 1;
            s.installing = vec!["Python".into()];
            s.migrating = true;
        });
        assert_eq!(codes(&a, Duration::from_secs(5)), ["call", "phoneCall", "agentTask", "download", "mediaJob", "installing", "migration", "starting"]);
        // ...and an unreadable phone plugin and unreadable engines take their place in it.
        a.set(|s| {
            s.phone = Some(unknown("no answer"));
            s.engines_unknown = Some("no answer".into());
        });
        assert_eq!(codes(&a, LONG), ["call", "callUnknown", "agentTask", "download", "enginesUnknown", "installing", "migration"]);
    }

    #[test]
    fn the_codes_are_all_different_so_the_window_can_key_by_them() {
        let a = Fake::default();
        a.set(|s| {
            s.calls = 1;
            s.phone = Some(live(1));
            s.tasks = 1;
            s.downloads = 1;
            s.media = 1;
            s.installing = vec!["Python".into()];
            s.migrating = true;
        });
        let mut all = codes(&a, Duration::from_secs(5));
        let before = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), before);
    }
}
