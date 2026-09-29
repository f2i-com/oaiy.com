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
//! 4. each [`Part`] is stopped, in the order quitting stops them; but right before the part that holds the
//!    phone (the plugins) the CALLS are looked at once more, from the sources alone: the stops before it take
//!    seconds, and a call may have begun. If one has, the parts already stopped are started again and the
//!    update goes back to `ready`, nothing installed;
//! 5. the installer is handed the verified bytes ([`Steps::hand_off`]), which on Windows starts the
//!    installer and exits this process, so nothing after it runs.
//!
//! A part that will not stop, or a hand-off that fails, restarts the parts already stopped (last
//! stopped, first started) and the update is `failed`, with the reason in words.
//!
//! So does a PANIC anywhere after the install began (in the flush, a part's stop, the hand-off): the
//! sequence holds an [`Unwind`] guard, which on the way out of a panic starts again what was stopped
//! (the part that was in the middle of stopping too) and fails the update, so the app is not left
//! half stopped with the state stuck on "installing". This needs unwinding panics: this build's
//! release profile does not set `panic = "abort"` (Cargo.toml has no profile section). A part that
//! never returns from `stop` is not recovered: each part's own stop is bounded (the engines' shutdown,
//! the services' kill timeouts), and a hang inside one is a follow-up (docs/UPDATES.md, "Not yet").
//!
//! The parts are the same list quitting uses ([`stop_best_effort`]): one order, defined in one place.

use std::panic::{catch_unwind, AssertUnwindSafe};
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
    /// Stopping this part ends a phone call (the plugins hold the phone). The calls are looked at again right before it.
    fn holds_calls(&self) -> bool {
        false
    }
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
    // From here the update is `installing`: whatever ends this function without an outcome (a panic) is undone by the guard.
    let mut unwind = Unwind { updater, stopped: Vec::new(), done: false };
    // 2. The Agent's work is saved.
    (steps.flush)();
    // 3. Nothing may have begun in the meantime.
    let blockers: Vec<Blocker> = updater.blockers_fresh((steps.clock)());
    if !blockers.is_empty() {
        unwind.done = true;
        updater.return_to_ready(package);
        return Outcome::Refused(InstallRefusal::Blocked(blockers));
    }
    // 4. Everything is stopped, in order. (A part is on the list BEFORE it is asked to stop: one that panics or fails halfway is started again too.)
    for part in steps.parts {
        if part.holds_calls() {
            // The parts before this one took time to stop: a call may have begun, and this is the stop that would end it. The look asks
            // only the call sources (the engines are gone by now, and what asks them would find nothing).
            let blockers = updater.call_blockers();
            if !blockers.is_empty() {
                unwind.done = true;
                return put_back(updater, package, &unwind.stopped, blockers);
            }
        }
        log::info!("update: stopping {}", part.name());
        unwind.stopped.push(*part);
        if let Err(e) = part.stop() {
            unwind.done = true;
            return failed(updater, &unwind.stopped, format!("{} would not stop ({e})", part.name()));
        }
    }
    // 5. The installer takes over.
    log::info!("update: handing the installer {} bytes", package.len());
    let handed = (steps.hand_off)(&package);
    unwind.done = true;
    match handed {
        Ok(()) => Outcome::HandedOff,
        Err(e) => failed(updater, &unwind.stopped, format!("the installer could not be started ({e})")),
    }
}

/// What ends an install that is cut short by a panic: while it is alive and not `done`, dropping it (which is what unwinding does) starts
/// again every part that was stopped and marks the update failed. Nothing in a drop may panic again (that aborts the process), so
/// the restarts are each caught.
struct Unwind<'a> {
    updater: &'a Updater,
    stopped: Vec<&'a dyn Part>,
    /// The sequence chose its own outcome: there is nothing to undo.
    done: bool,
}

impl Drop for Unwind<'_> {
    fn drop(&mut self) {
        if !self.done {
            log::error!("update: the install stopped unexpectedly; starting again what was stopped");
            failed(self.updater, &self.stopped, "it stopped unexpectedly, because of an internal error".to_string());
        }
    }
}

/// Start what was stopped again, last stopped first: which were, and which could not be. A part whose start panics is reported like one
/// that fails, and the others are still started.
fn start_again(stopped: &[&dyn Part]) -> (Vec<&'static str>, Vec<(&'static str, String)>) {
    let mut restarted = Vec::new();
    let mut not_restarted = Vec::new();
    for part in stopped.iter().rev() {
        match catch_unwind(AssertUnwindSafe(|| part.start())) {
            Ok(Ok(())) => restarted.push(part.name()),
            Ok(Err(e)) => not_restarted.push((part.name(), e)),
            Err(_) => not_restarted.push((part.name(), "it failed while starting".to_string())),
        }
    }
    (restarted, not_restarted)
}

/// A call (or a phone plugin that could not say) was found right before the plugins were to be stopped: start what was stopped again and go
/// back to `ready` with the download, nothing installed. If something cannot be started again the update is `failed` instead, saying what.
fn put_back(updater: &Updater, package: VerifiedPackage, stopped: &[&dyn Part], blockers: Vec<Blocker>) -> Outcome {
    log::warn!("update: {}; putting back what was stopped", blockers.iter().map(|b| b.message.as_str()).collect::<Vec<_>>().join(" "));
    let (restarted, not_restarted) = start_again(stopped);
    if not_restarted.is_empty() {
        updater.return_to_ready(package);
        return Outcome::Refused(InstallRefusal::Blocked(blockers));
    }
    let names: Vec<&str> = not_restarted.iter().map(|(n, _)| *n).collect();
    let message = format!(
        "The update was stopped because {} OAIY could not start {} again: quit OAIY from its tray icon and open it again.",
        blockers.iter().map(|b| b.message.as_str()).collect::<Vec<_>>().join(" "),
        names.join(" and ")
    );
    log::warn!("update: {message}");
    updater.fail_install(message.clone());
    Outcome::Failed { message, restarted, not_restarted }
}

/// Start what was stopped again, last stopped first, and record the failure.
fn failed(updater: &Updater, stopped: &[&dyn Part], why: String) -> Outcome {
    let (restarted, not_restarted) = start_again(stopped);
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

    /// A part that records what is asked of it and can be told to fail, or to panic, or to do something once it has stopped.
    struct Fakepart {
        name: &'static str,
        log: Log,
        fail_stop: bool,
        fail_start: bool,
        panic_stop: bool,
        panic_start: bool,
        holds_calls: bool,
        after_stop: Option<Arc<dyn Fn() + Send + Sync>>,
    }

    impl Fakepart {
        fn new(name: &'static str, log: &Log) -> Fakepart {
            Fakepart { name, log: log.clone(), fail_stop: false, fail_start: false, panic_stop: false, panic_start: false, holds_calls: name == "plugins", after_stop: None }
        }
    }

    impl Part for Fakepart {
        fn name(&self) -> &'static str {
            self.name
        }
        fn stop(&self) -> Result<(), String> {
            self.log.lock().unwrap().push(format!("stop {}", self.name));
            if self.panic_stop {
                panic!("an injected panic while stopping {}", self.name);
            }
            if let Some(after) = &self.after_stop {
                after();
            }
            if self.fail_stop { Err("stuck".into()) } else { Ok(()) }
        }
        fn start(&self) -> Result<(), String> {
            self.log.lock().unwrap().push(format!("start {}", self.name));
            if self.panic_start {
                panic!("an injected panic while starting {}", self.name);
            }
            if self.fail_start { Err("gone".into()) } else { Ok(()) }
        }
        fn holds_calls(&self) -> bool {
            self.holds_calls
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
        assert_eq!(log_of(&log), ["flush", "stop engines", "stop script host", "stop plugins", "stop services", "hand off 12 bytes"]);
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
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "stop services", "hand off 12 bytes", "start services", "start plugins", "start script host", "start engines"]);
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

    /// `perform` with a panic somewhere in it: what came out of the panic (its message), with the updater as it is left.
    fn perform_that_panics(u: &Updater, now: Instant, parts: &[Fakepart], log: &Log, flush: impl Fn() + 'static, hand_off_panics: bool) -> Option<String> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let refs: Vec<&dyn Part> = parts.iter().map(|p| p as &dyn Part).collect();
            let hand_log = log.clone();
            let hand_off = move |package: &VerifiedPackage| {
                hand_log.lock().unwrap().push(format!("hand off {} bytes", package.len()));
                if hand_off_panics {
                    panic!("an injected panic in the hand-off");
                }
                Ok(())
            };
            let clock = move || now;
            perform(u, &Steps { flush: &flush, parts: &refs, hand_off: &hand_off, clock: &clock })
        }));
        result.err().map(|payload| payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())
    }

    fn assert_failed_and_unstuck(u: &Updater, now: Instant) {
        let s = u.status_at(now);
        assert_eq!(s.state, State::Failed, "the update is not left on installing");
        let error = s.error.unwrap();
        assert!(error.contains("it stopped unexpectedly, because of an internal error"), "{error}");
        assert!(error.contains("started again what it had stopped"), "{error}");
    }

    #[test]
    fn a_call_that_begins_while_the_first_parts_are_stopping_puts_them_back_and_the_plugins_are_never_stopped() {
        let (u, fake, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        // The engines take their time to stop; a call comes in meanwhile.
        let f = fake.clone();
        parts[0].after_stop = Some(Arc::new(move || f.set(|s| s.calls = 1)));
        let outcome = run(&u, now, &parts, &log, || {}, false);
        match outcome {
            Outcome::Refused(InstallRefusal::Blocked(blockers)) => assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["call"]),
            other => panic!("{other:?}"),
        }
        // What was stopped is started again, last stopped first; the plugins (the phone) and everything after them were never touched.
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "start script host", "start engines"]);
        // The look before the plugins asked the call sources alone: no second full reading of the engines it had just stopped.
        let (call_reads, reads) = (fake.call_reads.load(std::sync::atomic::Ordering::SeqCst), fake.reads.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(call_reads, 1);
        assert_eq!(reads, 2, "the button and the look after the flush read everything; the look before the plugins did not");
        assert_eq!(u.status_at(now).state, State::Ready, "the download is kept: the owner can press the button again when the call is over");
    }

    #[test]
    fn a_phone_plugin_that_cannot_say_at_the_last_look_puts_it_back_the_same_way() {
        use super::super::phone::LineState;
        let (u, fake, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        let f = fake.clone();
        parts[1].after_stop = Some(Arc::new(move || f.set(|s| s.phone = Some(LineState::Unknown { plugin: "Aokie Phone Bridge".into(), why: "it did not answer within 3 s".into() }))));
        match run(&u, now, &parts, &log, || {}, false) {
            Outcome::Refused(InstallRefusal::Blocked(blockers)) => assert_eq!(blockers.iter().map(|b| b.code).collect::<Vec<_>>(), ["callUnknown"]),
            other => panic!("{other:?}"),
        }
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "start script host", "start engines"]);
        assert_eq!(u.status_at(now).state, State::Ready);
    }

    #[test]
    fn a_part_that_cannot_be_put_back_after_a_late_call_makes_the_update_failed_and_says_which() {
        let (u, fake, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        let f = fake.clone();
        parts[1].after_stop = Some(Arc::new(move || f.set(|s| s.calls = 1)));
        parts[0].fail_start = true;
        match run(&u, now, &parts, &log, || {}, false) {
            Outcome::Failed { message, restarted, not_restarted } => {
                assert_eq!(restarted, ["script host"]);
                assert_eq!(not_restarted, [("engines", "gone".to_string())]);
                assert!(message.contains("phone call is in progress") && message.contains("could not start engines again"), "{message}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(u.status_at(now).state, State::Failed);
    }

    #[test]
    fn with_no_part_that_holds_calls_there_is_no_extra_look() {
        let (u, fake, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[2].holds_calls = false;
        assert_eq!(run(&u, now, &parts, &log, || {}, false), Outcome::HandedOff);
        assert_eq!(fake.call_reads.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn a_panic_in_a_part_that_is_stopping_starts_it_and_the_ones_before_it_again_and_fails_the_update() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[2].panic_stop = true;
        let panicked = perform_that_panics(&u, now, &parts, &log, || {}, false).expect("the panic goes on up: it is not swallowed");
        assert!(panicked.contains("injected panic while stopping plugins"), "{panicked}");
        // The part that was in the middle of stopping is started too, then the ones before it, last stopped first; the installer never had the bytes.
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "start plugins", "start script host", "start engines"]);
        assert_failed_and_unstuck(&u, now);
    }

    #[test]
    fn a_panic_in_the_hand_off_starts_everything_again_and_fails_the_update() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let parts = parts(&log);
        let panicked = perform_that_panics(&u, now, &parts, &log, || {}, true).expect("the panic goes on up");
        assert!(panicked.contains("injected panic in the hand-off"), "{panicked}");
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "stop services", "hand off 12 bytes", "start services", "start plugins", "start script host", "start engines"]);
        assert_failed_and_unstuck(&u, now);
    }

    #[test]
    fn a_panic_in_the_flush_before_anything_is_stopped_fails_the_update_and_stops_nothing() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let parts = parts(&log);
        let panicked = perform_that_panics(&u, now, &parts, &log, || panic!("an injected panic in the flush"), false).expect("the panic goes on up");
        assert!(panicked.contains("injected panic in the flush"), "{panicked}");
        assert!(log_of(&log).is_empty(), "nothing was stopped, so nothing is started: {:?}", log_of(&log));
        assert_failed_and_unstuck(&u, now);
    }

    #[test]
    fn a_part_that_panics_while_being_started_again_is_reported_and_the_rest_are_still_started() {
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let mut parts = parts(&log);
        parts[3].panic_stop = true;
        parts[2].panic_start = true;
        perform_that_panics(&u, now, &parts, &log, || {}, false).expect("the panic goes on up");
        // services panicked stopping, and plugins panic starting again: script host and engines are started all the same.
        assert_eq!(log_of(&log), ["stop engines", "stop script host", "stop plugins", "stop services", "start services", "start plugins", "start script host", "start engines"]);
        let s = u.status_at(now);
        assert_eq!(s.state, State::Failed);
        assert!(s.error.unwrap().contains("could not start plugins again"), "the part that could not be started is named");
    }

    #[test]
    fn a_sequence_that_ends_by_itself_is_not_undone_by_the_guard() {
        // A hand-off that succeeds ends the process on Windows and the app carries on stopped on Linux: the guard does not start anything again.
        let (u, _, now) = ready();
        let log: Log = Log::default();
        let parts = parts(&log);
        assert_eq!(run(&u, now, &parts, &log, || {}, false), Outcome::HandedOff);
        assert!(!log_of(&log).iter().any(|l| l.starts_with("start ")), "{:?}", log_of(&log));
        assert_eq!(u.status_at(now).state, State::Installing);
        // Nor a refusal.
        let (u, fake, now) = ready();
        fake.set(|s| s.calls = 1);
        let log: Log = Log::default();
        let parts = self::parts(&log);
        assert!(matches!(run(&u, now, &parts, &log, || {}, false), Outcome::Refused(_)));
        assert!(log_of(&log).is_empty());
        assert_eq!(u.status_at(now).state, State::Ready);
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
