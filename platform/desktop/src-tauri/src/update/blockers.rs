//! What must stop an update from being installed, and why.
//!
//! OAIY answers a phone line, so an update must never interrupt a live call, and it must not
//! throw away work in flight either. [`compute`] turns what the app is doing into the list of
//! reasons the install button is off, each in words the owner can read; an empty list means
//! nothing is in the way. Every reason blocks by itself.
//!
//! What the app is doing comes from an [`Activity`]: the real one ([`Probes`]) asks the call hub,
//! the agent's task list, the downloader, the engines and the service registry; the tests give a
//! fake one.

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

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
    /// Stable name of the reason: `call`, `agentTask`, `download`, `mediaJob`, `installing`, `migration`, `starting`.
    pub code: &'static str,
    /// The reason, for the person to read.
    pub message: String,
}

impl Blocker {
    fn new(code: &'static str, message: impl Into<String>) -> Blocker {
        Blocker { code, message: message.into() }
    }
}

/// What the app is doing, as far as an update cares.
pub trait Activity: Send + Sync {
    /// Phone calls in progress now.
    fn live_calls(&self) -> usize;
    /// Tasks a flow gave the agent that it has not answered.
    fn agent_tasks(&self) -> usize;
    /// Model or file downloads queued or running (a paused one is not: it resumes).
    fn downloads(&self) -> usize;
    /// Pictures, video, music and the like the engines are making.
    fn media_jobs(&self) -> usize;
    /// Things being installed now (a service, a plugin, Python, Node), by name.
    fn installing(&self) -> Vec<String>;
    /// The data folder being moved.
    fn migrating(&self) -> bool;
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 { one.to_string() } else { many.replace("{n}", &n.to_string()) }
}

/// The reasons an install must wait, given what the app is doing (None: nothing is known yet, which is
/// still starting up) and how long it has been running.
pub fn compute(activity: Option<&dyn Activity>, uptime: Duration) -> Vec<Blocker> {
    let mut out = Vec::new();
    let Some(activity) = activity else {
        out.push(Blocker::new("starting", "OAIY is still starting up. Try again in a minute."));
        return out;
    };
    let calls = activity.live_calls();
    if calls > 0 {
        out.push(Blocker::new("call", plural(calls, "A phone call is in progress. OAIY never restarts during a call.", "{n} phone calls are in progress. OAIY never restarts during a call.")));
    }
    let tasks = activity.agent_tasks();
    if tasks > 0 {
        out.push(Blocker::new("agentTask", plural(tasks, "The Agent is working on a task from a flow.", "The Agent is working on {n} tasks from flows.")));
    }
    let downloads = activity.downloads();
    if downloads > 0 {
        out.push(Blocker::new("download", plural(downloads, "A model or file is downloading.", "{n} models or files are downloading.")));
    }
    let media = activity.media_jobs();
    if media > 0 {
        out.push(Blocker::new("mediaJob", plural(media, "The engines are making a picture, video or other media.", "The engines are making {n} pictures, videos or other media.")));
    }
    let installing = activity.installing();
    if !installing.is_empty() {
        out.push(Blocker::new("installing", format!("Something is installing: {}.", installing.join(", "))));
    }
    if activity.migrating() {
        out.push(Blocker::new("migration", "The data folder is being moved."));
    }
    if uptime < MIN_UPTIME {
        let left = (MIN_UPTIME - uptime).as_secs().max(1);
        out.push(Blocker::new("starting", format!("OAIY started less than {} minutes ago; it can update in {left} seconds.", MIN_UPTIME.as_secs() / 60)));
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
    pub engines: Option<Arc<dyn Fn() -> (usize, usize) + Send + Sync>>,
    /// The data folder is being moved.
    pub migration: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl Activity for Probes {
    fn live_calls(&self) -> usize {
        crate::voice::live_call_count()
    }

    fn agent_tasks(&self) -> usize {
        crate::agent_tasks::pending_count()
    }

    fn downloads(&self) -> usize {
        let own = self.downloads.as_ref().map_or(0, |d| d.active_count());
        let engines = self.engines.as_ref().map_or(0, |f| f().1);
        own + engines
    }

    fn media_jobs(&self) -> usize {
        self.engines.as_ref().map_or(0, |f| f().0)
    }

    fn installing(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        if let Some(registry) = &self.registry {
            // A poisoned lock is read anyway: this only asks a question.
            let registry = registry.lock().unwrap_or_else(|e| e.into_inner());
            names.extend(registry.installing_ids().into_iter().map(|id| format!("the service {id}")));
        }
        if self.python.as_ref().is_some_and(|p| p.job_running()) {
            names.push("Python".to_string());
        }
        if self.node.as_ref().is_some_and(|n| n.is_installing()) {
            names.push("Node".to_string());
        }
        if crate::plugins::install::in_progress() {
            names.push("a plugin".to_string());
        }
        names
    }

    fn migrating(&self) -> bool {
        self.migration.as_ref().is_some_and(|f| f())
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use std::sync::Mutex;

    /// An Activity the test sets up.
    #[derive(Default)]
    pub struct Fake(pub Mutex<FakeState>);

    #[derive(Default, Clone)]
    pub struct FakeState {
        pub calls: usize,
        pub tasks: usize,
        pub downloads: usize,
        pub media: usize,
        pub installing: Vec<String>,
        pub migrating: bool,
    }

    impl Fake {
        pub fn set(&self, f: impl FnOnce(&mut FakeState)) {
            f(&mut self.0.lock().unwrap());
        }
    }

    impl Activity for Fake {
        fn live_calls(&self) -> usize {
            self.0.lock().unwrap().calls
        }
        fn agent_tasks(&self) -> usize {
            self.0.lock().unwrap().tasks
        }
        fn downloads(&self) -> usize {
            self.0.lock().unwrap().downloads
        }
        fn media_jobs(&self) -> usize {
            self.0.lock().unwrap().media
        }
        fn installing(&self) -> Vec<String> {
            self.0.lock().unwrap().installing.clone()
        }
        fn migrating(&self) -> bool {
            self.0.lock().unwrap().migrating
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::Fake;
    use super::*;

    const LONG: Duration = Duration::from_secs(3600);

    fn codes(activity: &Fake, uptime: Duration) -> Vec<&'static str> {
        compute(Some(activity), uptime).iter().map(|b| b.code).collect()
    }

    #[test]
    fn nothing_going_on_and_up_for_a_while_blocks_nothing() {
        assert!(compute(Some(&Fake::default()), LONG).is_empty());
        assert!(compute(Some(&Fake::default()), MIN_UPTIME).is_empty());
    }

    #[test]
    fn a_live_call_blocks_by_itself_and_says_so() {
        let a = Fake::default();
        a.set(|s| s.calls = 1);
        let blockers = compute(Some(&a), LONG);
        assert_eq!(blockers.len(), 1);
        assert_eq!(blockers[0].code, "call");
        assert!(blockers[0].message.contains("phone call") && blockers[0].message.contains("never restarts"));
        a.set(|s| s.calls = 2);
        assert!(compute(Some(&a), LONG)[0].message.starts_with("2 phone calls"));
    }

    #[test]
    fn a_running_agent_task_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.tasks = 1);
        assert_eq!(codes(&a, LONG), ["agentTask"]);
        assert!(compute(Some(&a), LONG)[0].message.contains("Agent"));
    }

    #[test]
    fn a_download_in_progress_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.downloads = 3);
        assert_eq!(codes(&a, LONG), ["download"]);
        assert!(compute(Some(&a), LONG)[0].message.starts_with("3 models or files"));
    }

    #[test]
    fn an_engine_media_job_blocks_by_itself() {
        let a = Fake::default();
        a.set(|s| s.media = 1);
        assert_eq!(codes(&a, LONG), ["mediaJob"]);
    }

    #[test]
    fn a_service_or_plugin_installing_blocks_by_itself_and_is_named() {
        let a = Fake::default();
        a.set(|s| s.installing = vec!["the service comfyui".into(), "a plugin".into()]);
        let blockers = compute(Some(&a), LONG);
        assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["installing"]);
        assert!(blockers[0].message.contains("the service comfyui") && blockers[0].message.contains("a plugin"));
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
        let message = &compute(Some(&a), Duration::from_secs(90))[0].message;
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
            s.tasks = 1;
            s.downloads = 1;
            s.media = 1;
            s.installing = vec!["Python".into()];
            s.migrating = true;
        });
        assert_eq!(codes(&a, Duration::from_secs(5)), ["call", "agentTask", "download", "mediaJob", "installing", "migration", "starting"]);
    }
}
