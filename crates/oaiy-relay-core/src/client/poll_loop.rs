//! The poll loop: a blocking driver that acts on [`poll::decide`] and does nothing else.
//!
//! It runs on one thread (a tokio `spawn_blocking` task on the desktop, a foreground service's worker on the phone), owns the one poll that is in flight (P1), and is stopped or
//! told about a network change from other threads through a [`PollHandle`]. What it does in each pass:
//!
//! 1. **Prove the relay** when [`poll::proof_due`] says (before the first poll of the process, after a pause of 60 seconds or more, after a network change, every 300 seconds),
//!    and send no poll and no token until it has (P9). A proof that does not verify ends the loop: the relay is "not who it was", the credential is kept and not sent.
//! 2. **Poll** with the stored `since` and `epoch`, `wait` taken from `info.wait.default` (at most `info.wait.max`), and the timeout of `wait + 10` seconds (P1).
//! 3. **Persist** what the answer accepted (the items and the cursor and epoch together) **before** the poll that carries the new `since` is sent: that poll is the
//!    acknowledgement and the relay deletes what it acknowledges (P2). A write that fails moves nothing and is paced as a failure.
//! 4. **Decide**, report, and **pause** as the decision says: the pause is slept in the clock's sleep and ended by the handle.
//!
//! The loop never retries a stop, never has a budget that ends it, and never reads a clock or draws a random number itself: the client's clock and random source do (so a test
//! with a fake clock runs a loop of hours in no time and a seeded source makes it repeatable).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::ids::Token;
use crate::json::Json;
use crate::poll::{self, Action, Counters, DecideInput, Decision, Outcome, PollInfo, ProofDue, ProofResult, Report};

use super::http::{Cancel, TransportError};
use super::relay::{PollRequest, ProveError, RelayClient};
use super::status::{ConnectionState, Event, StatusSink};
use super::store::{AcceptedItem, Item, PersistBatch, PollStore};

/// How the loop is set.
#[derive(Debug, Clone)]
pub struct PollLoopConfig {
    /// `limit` of each poll (1 to 64).
    pub limit: u32,
    /// The protocol level of this client, to judge a `426` against `info.minClient`.
    pub level: u32,
}

impl Default for PollLoopConfig {
    fn default() -> Self {
        PollLoopConfig { limit: 32, level: 1 }
    }
}

/// Why the loop ended.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEnd {
    /// The handle stopped it.
    Cancelled,
    /// A stop of P2 or P9: the action to take, what the relay said, and the counters as they were (they are reported with the stop).
    Stopped {
        /// The action: `forget_credential`, `refresh_or_reenrol`, `update_client`, `report_defect`, `report_relay_changed`.
        action: Action,
        /// The status that ended it, when there was one.
        status: Option<u16>,
        /// The relay's error code, when it sent one.
        code: Option<String>,
        /// The counters at the stop.
        counters: Counters,
    },
    /// The store could not be read at the start: the loop did not start (a damaged cursor is never treated as a first run).
    StoreUnreadable(String),
}

struct HandleState {
    stop: AtomicBool,
    network_changed: AtomicBool,
    current: Mutex<Option<Cancel>>,
}

/// What other threads use to stop the loop or tell it the network changed. Cloning it shares it.
#[derive(Clone)]
pub struct PollHandle(Arc<HandleState>);

impl PollHandle {
    fn new() -> PollHandle {
        PollHandle(Arc::new(HandleState { stop: AtomicBool::new(false), network_changed: AtomicBool::new(false), current: Mutex::new(None) }))
    }

    fn interrupt(&self) {
        if let Ok(c) = self.0.current.lock() {
            if let Some(c) = c.as_ref() {
                c.cancel();
            }
        }
    }

    /// Ends the loop: the poll in flight is abandoned and no further request is made.
    pub fn stop(&self) {
        self.0.stop.store(true, Ordering::SeqCst);
        self.interrupt();
    }

    /// The network changed: the poll in flight (or the pause) is ended, the relay is proved again, and the next poll starts no sooner than 250 ms after the one that was ended (P1).
    pub fn network_changed(&self) {
        self.0.network_changed.store(true, Ordering::SeqCst);
        self.interrupt();
    }

    fn stopped(&self) -> bool {
        self.0.stop.load(Ordering::SeqCst)
    }

    fn take_network_changed(&self) -> bool {
        self.0.network_changed.swap(false, Ordering::SeqCst)
    }

    fn arm(&self) -> Cancel {
        let c = Cancel::new();
        if let Ok(mut cur) = self.0.current.lock() {
            *cur = Some(c.clone());
        }
        // A stop or a network change that came before the cancel was armed is not lost.
        if self.stopped() || self.0.network_changed.load(Ordering::SeqCst) {
            c.cancel();
        }
        c
    }
}

/// The loop.
pub struct PollLoop<S: PollStore> {
    client: Arc<RelayClient>,
    token: Token,
    store: S,
    sink: Arc<dyn StatusSink>,
    config: PollLoopConfig,
    handle: PollHandle,
}

impl<S: PollStore> PollLoop<S> {
    /// A loop for `client` with the device's `token`, writing what it accepts to `store` and its events to `sink`.
    pub fn new(client: Arc<RelayClient>, token: Token, store: S, sink: Arc<dyn StatusSink>, config: PollLoopConfig) -> PollLoop<S> {
        PollLoop { client, token, store, sink, config, handle: PollHandle::new() }
    }

    /// The handle other threads stop the loop with.
    pub fn handle(&self) -> PollHandle {
        self.handle.clone()
    }

    /// The store, after the loop has ended (a test looks at what was accepted).
    pub fn into_store(self) -> S {
        self.store
    }

    /// The store, while the loop is not running.
    pub fn store(&self) -> &S {
        &self.store
    }

    fn emit(&self, e: Event) {
        self.sink.event(&e);
    }

    fn sleep(&self, seconds: f64) -> bool {
        if seconds <= 0.0 {
            return !self.handle.stopped();
        }
        let cancel = self.handle.arm();
        let slept = self.client.clock().sleep(Duration::from_secs_f64(seconds), &cancel);
        // A network change ends the pause too, and is not a stop.
        slept || !self.handle.stopped()
    }

    /// Runs until stopped, ended by a stop of the rules, or cancelled through the handle.
    #[allow(unused_assignments)]
    pub fn run(&mut self) -> LoopEnd {
        let mut cursor = match self.store.load() {
            Ok(c) => c,
            Err(e) => return LoopEnd::StoreUnreadable(e.to_string()),
        };
        let clock = self.client.clock().clone();
        let mut counters = Counters::default();
        let mut state = ConnectionState::Idle;
        let mut process_start = true;
        let mut last_proof: Option<Duration> = None;
        let mut longest_pause_s = 0u64;
        let mut last_start: Option<Duration> = None;
        let mut warned_clock = false;
        let mut poll_info = PollInfo::default();
        let mut wait_s = 20u64;
        let mut last_reported: Option<Report> = None;

        macro_rules! set_state {
            ($s:expr) => {
                if state != $s {
                    state = $s;
                    self.emit(Event::State($s));
                }
            };
        }

        loop {
            if self.handle.stopped() {
                return LoopEnd::Cancelled;
            }
            let network_changed = self.handle.take_network_changed();
            let due = ProofDue {
                process_start,
                network_changed,
                longest_pause_s,
                seconds_since_proof: last_proof.map_or(u64::MAX, |t| clock.monotonic().saturating_sub(t).as_secs()),
            };
            // P9: the relay's identity before any token is sent.
            if poll::proof_due(&due) {
                set_state!(ConnectionState::Proving);
                let cancel = self.handle.arm();
                let mut asked = None;
                let result = match self.client.prove(&cancel) {
                    Ok(proved) => {
                        poll_info = PollInfo::from_info(&proved.info);
                        wait_s = proved.info.wait.default.min(proved.info.wait.max);
                        ProofResult::Verified
                    }
                    Err(ProveError::NoAnswer(super::relay::ClientError::Cancelled)) => continue,
                    Err(ProveError::NoAnswer(e)) => {
                        asked = e.relay().and_then(|r| r.retry_after);
                        ProofResult::NoAnswer
                    }
                    Err(ProveError::Invalid(_)) => ProofResult::Invalid,
                };
                let d = poll::decide_proof(counters, result, self.client.jitter(), cursor.since, asked);
                counters = d.counters;
                self.emit(Event::Answer { outcome: d.outcome, status: None, counters });
                match d.outcome {
                    Outcome::Proved => {
                        process_start = false;
                        longest_pause_s = 0;
                        last_proof = Some(clock.monotonic());
                        self.emit(Event::InfoProved);
                        continue;
                    }
                    Outcome::Stop => {
                        set_state!(ConnectionState::Suspect);
                        self.emit(Event::Action(Action::ReportRelayChanged));
                        return LoopEnd::Stopped { action: Action::ReportRelayChanged, status: None, code: None, counters };
                    }
                    _ => {
                        for r in &d.reports {
                            if last_reported != Some(*r) {
                                self.emit(Event::Report(*r));
                                last_reported = Some(*r);
                            }
                            if *r == Report::Unreachable {
                                set_state!(ConnectionState::Unreachable);
                            }
                        }
                        longest_pause_s = longest_pause_s.max(d.pause_s as u64);
                        if !self.sleep(d.pause_s) {
                            return LoopEnd::Cancelled;
                        }
                        continue;
                    }
                }
            }

            // P1: a poll that replaces another starts no sooner than 250 ms after the one it replaces.
            if let Some(started) = last_start {
                let wait_ms = poll::replace_wait_ms(clock.monotonic().saturating_sub(started).as_millis() as u64);
                if network_changed && wait_ms > 0 && !self.sleep(wait_ms as f64 / 1000.0) {
                    return LoopEnd::Cancelled;
                }
            }

            let request = PollRequest { since: cursor.since, epoch: cursor.epoch.clone(), wait_s, limit: self.config.limit };
            let cancel = self.handle.arm();
            last_start = Some(clock.monotonic());
            let reply = match self.client.poll(&self.token, &request, &cancel) {
                Ok(r) => r,
                // The proof lapsed under us (a loop that slept a very long time on a host that suspends): prove again before anything is sent.
                Err(super::relay::ClientError::NotProved) => {
                    process_start = true;
                    continue;
                }
                Err(super::relay::ClientError::Suspect) => {
                    set_state!(ConnectionState::Suspect);
                    return LoopEnd::Stopped { action: Action::ReportRelayChanged, status: None, code: None, counters };
                }
                Err(_) => return LoopEnd::Cancelled,
            };
            if reply.transport == Some(TransportError::Cancelled) {
                // A poll that this client ended (a stop, or a network change): its answer is discarded and the next poll is its replacement.
                continue;
            }

            // P2: persist what was accepted before the poll that acknowledges it can be sent.
            let adoption = poll::assess(reply.status, reply.body.as_ref(), cursor.since);
            let mut persisted = true;
            let mut accepted_items = Vec::new();
            if let Some(a) = &adoption {
                if let Some(items) = reply.body.as_ref().and_then(|b| b.get("items")).and_then(Json::as_array) {
                    for &i in &a.accepted {
                        let raw = items[i].clone();
                        let seq = raw.get("seq").and_then(Json::as_uint53).unwrap_or(0);
                        accepted_items.push(AcceptedItem { seq, item: Item::from_json(&raw), raw });
                    }
                }
                let batch = PersistBatch { items: &accepted_items, since: a.since, epoch: &a.epoch, reset: a.reset };
                persisted = self.store.persist(&batch).is_ok();
            }

            let min_client_above_ours = if reply.status == Some(426) {
                let cancel = self.handle.arm();
                self.client.read_info(&cancel).map(|i| i.min_client > u64::from(self.config.level)).unwrap_or(false)
            } else {
                false
            };
            let d: Decision = poll::decide(&DecideInput {
                counters,
                info: poll_info,
                answer: reply.answer(),
                since: cursor.since,
                persisted,
                // This loop has one poll in flight and discards the answer of one it cancelled, so a `superseded` answer is always another process's doing.
                we_replaced: false,
                min_client_above_ours,
                now_epoch: Some(clock.unix_now()),
                u: self.client.jitter(),
            });
            counters = d.counters;
            self.emit(Event::Answer { outcome: d.outcome, status: reply.status, counters });

            if d.outcome == Outcome::Progress {
                if let Some(a) = &adoption {
                    cursor.since = a.since;
                    cursor.epoch = Some(a.epoch.clone());
                    self.emit(Event::Accepted { count: accepted_items.len(), since: a.since });
                    if a.reset {
                        self.emit(Event::MailboxReset);
                    }
                }
            }
            // The epoch the relay gave is stored when none is (a first run, or the retry after a first 400 that left it out): "an omitted epoch is no check, the answer carries the
            // relay's current epoch, which the client stores, and a later mismatch is an ordinary reset" (P2).
            if cursor.epoch.is_none() && matches!(d.outcome, Outcome::Idle | Outcome::Superseded) {
                if let Some(epoch) = reply.body.as_ref().and_then(|b| b.get_str("epoch")).map(str::to_string) {
                    let batch = PersistBatch { items: &[], since: cursor.since, epoch: &epoch, reset: false };
                    if self.store.persist(&batch).is_ok() {
                        cursor.epoch = Some(epoch);
                    }
                }
            }
            if matches!(d.outcome, Outcome::Progress | Outcome::Superseded | Outcome::Idle | Outcome::Flow) {
                set_state!(ConnectionState::Connected);
                last_reported = None;
            }
            for r in &d.reports {
                if last_reported != Some(*r) || *r != Report::Unreachable {
                    self.emit(Event::Report(*r));
                }
                last_reported = Some(*r);
                if *r == Report::Unreachable {
                    set_state!(ConnectionState::Unreachable);
                }
            }
            if !warned_clock && self.client.clock_mismatch() {
                warned_clock = true;
                self.emit(Event::ClockMismatch { offset_s: self.client.relay_offset_s().unwrap_or(0.0).round() as i64 });
            }

            if let Some(action) = d.action {
                self.emit(Event::Action(action));
                match action {
                    Action::ClearEpoch => {
                        let _ = self.store.clear_epoch();
                        cursor.epoch = None;
                    }
                    // Every client of this loop has one poll in flight, which has just been answered: there is nothing of its own to cancel.
                    Action::CancelOwnPolls => {}
                    Action::ForgetCredential
                    | Action::RefreshOrReenrol
                    | Action::UpdateClient
                    | Action::ReportDefect
                    | Action::ReportRelayChanged => {
                        set_state!(match action {
                            Action::ForgetCredential => ConnectionState::Revoked,
                            Action::RefreshOrReenrol => ConnectionState::Rejected,
                            Action::UpdateClient => ConnectionState::UpgradeRequired,
                            _ => ConnectionState::Stopped,
                        });
                        let code = reply.body.as_ref().and_then(|b| b.get("error")).and_then(|e| e.get_str("code")).map(str::to_string);
                        return LoopEnd::Stopped { action, status: reply.status, code, counters };
                    }
                }
            }

            longest_pause_s = longest_pause_s.max(d.pause_s as u64);
            if !self.sleep(d.pause_s) {
                return LoopEnd::Cancelled;
            }
        }
    }
}
