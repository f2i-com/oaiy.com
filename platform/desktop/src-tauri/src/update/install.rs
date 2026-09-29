//! The safe install sequence, and the stop it shares with quitting.
//!
//! An install ends with the installer replacing the running program, so before it OAIY stops
//! everything it runs the way it does when it quits, and if anything goes wrong on the way it
//! starts again what it had stopped, so a failed update leaves a working app rather than half of
//! one. The order is fixed, and is what the tests pin:
//!
//! 1. the update is taken ([`Updater::begin_install`]): a verified download, and NOTHING in the way
//!    (a live call, a running task, a download, a media job, an install, a young process). If
//!    anything is, nothing has been touched;
//! 2. the Agent page is asked to save its work (bounded: the install goes on if it does not answer);
//! 3. the blockers are looked at again, because the flush takes seconds and a call may have begun;
//! 4. each [`Part`] is stopped, in the order quitting stops them;
//! 5. the installer is handed the verified bytes ([`Steps::hand_off`]), which on Windows starts the
//!    installer and exits this process, so nothing after it runs.
//!
//! A part that will not stop, or a hand-off that fails, restarts the parts already stopped (last
//! stopped, first started) and the update is `failed`, with the reason in words.
//!
//! The parts are the same list quitting uses ([`stop_best_effort`]): one order, defined in one place.

use std::time::Instant;

use super::blockers::Blocker;
use super::updater::{InstallRefusal, Updater};
use super::verify::VerifiedPackage;

/// One thing the app runs that must be stopped before it exits, and started again if it does not.
pub trait Part: Send + Sync {
    fn name(&self) -> &'static str;
    fn stop(&self) -> Result<(), String>;
    /// Bring back what `stop` took away (as it was, not as a fresh start would).
    fn start(&self) -> Result<(), String>;
}

/// Quitting: stop every part in order, whatever one of them says (the app is going away either way).
pub fn stop_best_effort(parts: &[&dyn Part]) {
    for part in parts {
        if let Err(e) = part.stop() {
            log::warn!("stopping {} on exit: {e}", part.name());
        }
    }
}

/// What the sequence needs from the app.
pub struct Steps<'a> {
    /// Ask the Agent page to save its work and wait a little for the answer. Never fails: no answer is not a reason to stop.
    pub flush: &'a dyn Fn(),
    /// Everything the app runs, in the order quitting stops it.
    pub parts: &'a [&'a dyn Part],
    /// Give the installer the verified bytes. On Windows it starts the installer and exits this process.
    pub hand_off: &'a dyn Fn(&VerifiedPackage) -> Result<(), String>,
    /// The time (for how long the app has been up), so a test can set it.
    pub clock: &'a dyn Fn() -> Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Nothing was stopped: the update cannot be installed now.
    Refused(InstallRefusal),
    /// The installer has the bytes and the app is stopped (Windows never gets here: the process exits in the hand-off).
    HandedOff,
    /// Something went wrong after the app began to stop; what was stopped has been started again.
    Failed { message: String, restarted: Vec<&'static str>, not_restarted: Vec<(&'static str, String)> },
}

/// Run the sequence for the update `updater` holds. See the module docs for the order.
pub fn perform(updater: &Updater, steps: &Steps) -> Outcome {
    // 1. The update, and the blockers.
    let package = match updater.begin_install((steps.clock)()) {
        Ok(package) => package,
        Err(refusal) => return Outcome::Refused(refusal),
    };
    // 2. The Agent's work is saved.
    (steps.flush)();
    // 3. Nothing may have begun in the meantime.
    let blockers: Vec<Blocker> = updater.blockers((steps.clock)());
    if !blockers.is_empty() {
        updater.return_to_ready(package);
        return Outcome::Refused(InstallRefusal::Blocked(blockers));
    }
    // 4. Everything is stopped, in order.
    let mut stopped: Vec<&dyn Part> = Vec::new();
    for part in steps.parts {
        log::info!("update: stopping {}", part.name());
        stopped.push(*part);
        if let Err(e) = part.stop() {
            return failed(updater, &stopped, format!("{} would not stop ({e})", part.name()));
        }
    }
    // 5. The installer takes over.
    log::info!("update: handing the installer {} bytes", package.len());
    match (steps.hand_off)(&package) {
        Ok(()) => Outcome::HandedOff,
        Err(e) => failed(updater, &stopped, format!("the installer could not be started ({e})")),
    }
}

/// Start what was stopped again, last stopped first, and record the failure.
fn failed(updater: &Updater, stopped: &[&dyn Part], why: String) -> Outcome {
    let mut restarted = Vec::new();
    let mut not_restarted = Vec::new();
    for part in stopped.iter().rev() {
        match part.start() {
            Ok(()) => restarted.push(part.name()),
            Err(e) => not_restarted.push((part.name(), e)),
        }
    }
    let mut message = format!("The update could not be installed: {why}.");
    if not_restarted.is_empty() {
        message.push_str(" OAIY started again what it had stopped, and is running as before.");
    } else {
        let names: Vec<&str> = not_restarted.iter().map(|(n, _)| *n).collect();
        message.push_str(&format!(" OAIY could not start {} again: quit OAIY from its tray icon and open it again.", names.join(" and ")));
    }
    log::warn!("update: {message}");
    updater.fail_install(message.clone());
    Outcome::Failed { message, restarted, not_restarted }
}

#[cfg(test)]
mod tests {
    use super::super::updater::tests::{package, ready};
    use super::super::updater::State;
    use super::*;
    use std::sync::{Arc, Mutex};

    type Log = Arc<Mutex<Vec<String>>>;

    /// A part that records what is asked of it and can be told to fail.
    struct Fakepart {
        name: &'static str,
        log: Log,
        fail_stop: bool,
        fail_start: bool,
    }

    impl Fakepart {
        fn new(name: &'static str, log: &Log) -> Fakepart {
            Fakepart { name, log: log.clone(), fail_stop: false, fail_start: false }
        }
    }

    impl Part for Fakepart {
        fn name(&self) -> &'static str {
            self.name
        }
        fn stop(&self) -> Result<(), String> {
            self.log.lock().unwrap().push(format!("stop {}", self.name));
            if self.fail_stop { Err("stuck".into()) } else { Ok(()) }
        }
        fn start(&self) -> Result<(), String> {
            self.log.lock().unwrap().push(format!("start {}", self.name));
            if self.fail_start { Err("gone".into()) } else { Ok(()) }
        }
    }

    fn parts(log: &Log) -> Vec<Fakepart> {
        ["engines", "script host", "plugins", "services"].into_iter().map(|n| Fakepart::new(n, log)).collect()
    }

    fn run(u: &Updater, now: Instant, parts: &[Fakepart], log: &Log, flush: impl Fn() + 'static, hand_off_fails: bool) -> Outcome {
        let refs: Vec<&dyn Part> = parts.iter().map(|p| p as &dyn Part).collect();
        let hand_log = log.clone();
        let hand_off = move |package: &VerifiedPackage| {
            hand_log.lock().unwrap().push(format!("hand off {} bytes", package.len()));
            if hand_off_fails { Err("no permission".to_string()) } else { Ok(()) }
        };
        let clock = move || now;
        perform(u, &Steps { flush: &flush, parts: &refs, hand_off: &hand_off, clock: &clock })
    }

    fn log_of(log: &Log) -> Vec<String> {
        log.lock().unwrap().clone()
    }

    #[test]
    fn the_steps_run_in_order_blockers_then_flush_then_each_part_then_the_installer() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let flush_log = log.clone();
        let parts = parts(&log);
        let outcome = run(&u, now, &parts, &log, move || flush_log.lock().unwrap().push("flush".into()), false);
        assert_eq!(outcome, Outcome::HandedOff);
        assert_eq!(log_of(&log), ["flush", "stop engines", "stop script host", "stop plugins", "stop services", "hand off 9 bytes"]);
        assert_eq!(u.status_at(now).state, State::Installing);
    }

    #[test]
    fn a_blocker_present_at_the_start_means_nothing_at_all_is_touched() {
        use super::super::blockers::fake::FakeState;
        let cases: Vec<(&str, fn(&mut FakeState))> = vec![
            ("a call", |s| s.calls = 1),
            ("a task", |s| s.tasks = 1),
            ("a download", |s| s.downloads = 1),
            ("a media job", |s| s.media = 1),
            ("an install", |s| s.installing = vec!["Python".into()]),
            ("a data folder move", |s| s.migrating = true),
        ];
        for (what, set) in cases {
            let (u, fake, now) = ready();
            fake.set(set);
            let log: Log = Log::default();
            let flush_log = log.clone();
            let parts = parts(&log);
            let outcome = run(&u, now, &parts, &log, move || flush_log.lock().unwrap().push("flush".into()), false);
            assert!(matches!(outcome, Outcome::Refused(InstallRefusal::Blocked(_))), "{what}: {outcome:?}");
            assert!(log_of(&log).is_empty(), "{what}: nothing is asked of anything, not even the flush: {:?}", log_of(&log));
            assert_eq!(u.status_at(now).state, State::Ready, "{what}");
        }
    }

    #[test]
    fn a_call_that_begins_during_the_flush_stops_the_install_before_anything_is_stopped() {
        let (u, fake, now) = ready();
        let log: Log = Log::default();
        let flush_log = log.clone();
        let parts = parts(&log);
        // The flush takes seconds; in that time a call comes in.
        let outcome = run(&u, now, &parts, &log, move || {
            flush_log.lock().unwrap().push("flush".into());
            fake.set(|s| s.calls = 1);
        }, false);
        match outcome {
            Outcome::Refused(InstallRefusal::Blocked(blockers)) => assert_eq!(blockers[0].code, "call"),
            other => panic!("{other:?}"),
        }
        assert_eq!(log_of(&log), ["flush"]);
        // The download is not lost: the owner can press the button again when the call is over.
        assert_eq!(u.status_at(now).state, State::Ready);
    }

    #[test]
    fn nothing_downloaded_and_verified_means_no_install_and_nothing_stopped() {
        let (u, _, now) = super::super::updater::tests::available();
        let log: Log = Log::default();
        let parts = parts(&log);
        assert_eq!(run(&u, now, &parts, &log, || {}, false), Outcome::Refused(InstallRefusal::NotReady));
        assert!(log_of(&log).is_empty());
    }

    #[test]
    fn a_hand_off_that_fails_starts_everything_again_last_stopped_first_and_is_failed() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let parts = parts(&log);
        let outcome = run(&u, now, &parts, &log, || {}, true);
        match outcome {
            Outcome::Failed { message, restarted, not_restarted } => {
                assert_eq!(restarted, ["services", "plugins", "script host", "engines"]);
                assert!(not_restarted.is_empty());
                assert!(message.contains("the installer could not be started (no permission)"), "{message}");
                assert!(message.contains("started again what it had stopped"), "{message}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "stop services", "hand off 9 bytes", "start services", "start plugins", "start script host", "start engines"]);
        let s = u.status_at(now);
        assert_eq!(s.state, State::Failed);
        assert!(s.error.unwrap().contains("could not be installed"));
    }

    #[test]
    fn a_part_that_will_not_stop_starts_the_ones_before_it_again_and_the_installer_is_never_started() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[2].fail_stop = true;
        let outcome = run(&u, now, &parts, &log, || {}, false);
        match outcome {
            Outcome::Failed { message, restarted, .. } => {
                // The part that failed may be half stopped: it is started too.
                assert_eq!(restarted, ["plugins", "script host", "engines"]);
                assert!(message.contains("plugins would not stop (stuck)"), "{message}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "start plugins", "start script host", "start engines"]);
        assert!(!log_of(&log).iter().any(|l| l.starts_with("hand off")));
        assert_eq!(u.status_at(now).state, State::Failed);
    }

    #[test]
    fn a_part_that_cannot_be_started_again_is_reported_and_the_rest_are_still_started() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[3].fail_start = true;
        let outcome = run(&u, now, &parts, &log, || {}, true);
        match outcome {
            Outcome::Failed { message, restarted, not_restarted } => {
                assert_eq!(restarted, ["plugins", "script host", "engines"]);
                assert_eq!(not_restarted, [("services", "gone".to_string())]);
                assert!(message.contains("could not start services again: quit OAIY from its tray icon"), "{message}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn quitting_stops_every_part_in_the_same_order_and_carries_on_past_one_that_fails() {
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[1].fail_stop = true;
        let refs: Vec<&dyn Part> = parts.iter().map(|p| p as &dyn Part).collect();
        stop_best_effort(&refs);
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "stop services"]);
    }

    #[test]
    fn a_verified_package_is_what_the_installer_is_handed() {
        // The hand-off gets the bytes the verified package holds and nothing else.
        let (u, _, now) = ready();
        let seen: Arc<Mutex<Vec<u8>>> = Arc::default();
        let seen2 = seen.clone();
        let hand_off = move |p: &VerifiedPackage| {
            *seen2.lock().unwrap() = p.bytes().to_vec();
            Ok(())
        };
        let flush = || {};
        let clock = move || now;
        let outcome = perform(&u, &Steps { flush: &flush, parts: &[], hand_off: &hand_off, clock: &clock });
        assert_eq!(outcome, Outcome::HandedOff);
        assert_eq!(*seen.lock().unwrap(), package("0.2.0").bytes());
    }
}
