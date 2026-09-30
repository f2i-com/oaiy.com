//! The lanes a password verification runs in (design 4.7.2 and 4.7.4): what keeps 64 MiB passes from filling a
//! small server, and what keeps an attacker's queue from keeping the owner out.
//!
//! - **The anonymous lane** runs **2** verifications at a time with a waiting queue of **5**; more get `429` at
//!   once. **At most 2 attempts (running plus queued) per address**, counted when they enqueue, so one address
//!   cannot fill the queue before its failures are counted. In slow mode (`throttle.rs`) it runs **one** at a time
//!   with **5 seconds** between the starts of consecutive ones: 12 a minute, 17,280 a day at most.
//! - **One extra slot is reserved** for the session lane (`elevate`, the `current` check of a password change) and
//!   the known-device lane (a login that presents a valid `dev` cookie), so that neither ever waits behind the
//!   anonymous queue. That is 2 + 1 verifications at once: 3 x 64 MiB = 192 MiB at most.
//! - **The session lane** is per session: 5 wrong answers revoke that session, and at most 10 verifications per
//!   session per hour.
//! - **The fence**: every hash, whoever asks (a login, a setup, a password change, the console's reset), goes
//!   through one gate of 3 in front of the engine, so the bound holds for a caller that forgot its lane.
//!
//! The scheduler of the anonymous lane is a pure state machine that is told the time and whether slow mode is on,
//! so its rules (the queue, the cap per address, the spacing) are tested with no waiting; the async parts are thin.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use zeroize::Zeroizing;

use super::clock::Clock;
use super::password::{HashError, PasswordEngine, Verdict};
use super::throttle::{LoginThrottle, SLOW_SPACING_MS};

/// Verifications the anonymous lane runs at once.
pub const RUNNING: usize = 2;
/// Requests that may wait behind them.
pub const QUEUE: usize = 5;
/// Attempts of one address, running plus queued.
pub const PER_ADDRESS: usize = 2;
/// The reserved slot's waiters before a `429`: the lane of the owner is not a queue for anyone else.
pub const RESERVED_WAITERS: usize = 5;
/// Hashes at once, in all: the fence.
pub const FENCE: usize = RUNNING + 1;
/// Wrong answers in a session's lane that revoke it.
pub const SESSION_WRONG_LIMIT: u32 = 5;
/// Verifications a session may ask for in an hour.
pub const SESSION_PER_HOUR: usize = 10;
const HOUR_MS: u64 = 3_600_000;
/// How often a waiting request looks again (it is also woken when a place frees).
const POLL: Duration = Duration::from_millis(20);

/// Why a request was not admitted (a `429` at once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    /// Five are already waiting.
    QueueFull,
    /// This address already has two attempts running or waiting.
    PerAddress,
    /// The reserved slot has five waiting.
    ReservedBusy,
}

impl Reject {
    /// The seconds a client is told to wait: the spacing in slow mode, a moment otherwise.
    pub fn retry_after_s(self, slow: bool) -> u64 {
        if slow {
            SLOW_SPACING_MS / 1000
        } else {
            match self {
                Reject::ReservedBusy => 1,
                _ => 2,
            }
        }
    }
}

/// What a waiting ticket is told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    /// It may start now.
    Go,
    /// Not yet.
    Wait,
}

/// The scheduler of the anonymous lane: pure, told the time.
#[derive(Debug, Default)]
pub struct Scheduler {
    running: usize,
    waiting: VecDeque<u64>,
    /// Attempts per address, running plus waiting.
    held: HashMap<String, usize>,
    /// When the last verification started (zero: none has).
    last_start: u64,
    next_ticket: u64,
}

impl Scheduler {
    /// Ask for a place. The per-address cap is checked before the queue, and both when the request arrives.
    pub fn enqueue(&mut self, key: &str) -> Result<u64, Reject> {
        if self.held.get(key).copied().unwrap_or(0) >= PER_ADDRESS {
            return Err(Reject::PerAddress);
        }
        if self.waiting.len() >= QUEUE {
            return Err(Reject::QueueFull);
        }
        *self.held.entry(key.to_string()).or_default() += 1;
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.waiting.push_back(ticket);
        Ok(ticket)
    }

    /// May `ticket` start at `now`? Only the head of the queue may, and only when fewer than the limit are
    /// running (2, or 1 in slow mode) and, in slow mode, 5 seconds after the last start.
    pub fn poll(&mut self, ticket: u64, now: u64, slow: bool) -> Poll {
        let limit = if slow { 1 } else { RUNNING };
        let spaced =
            !slow || self.last_start == 0 || now.saturating_sub(self.last_start) >= SLOW_SPACING_MS;
        if self.waiting.front() == Some(&ticket) && self.running < limit && spaced {
            self.waiting.pop_front();
            self.running += 1;
            self.last_start = now;
            Poll::Go
        } else {
            Poll::Wait
        }
    }

    /// A verification that started has finished.
    pub fn finish(&mut self, key: &str) {
        self.running = self.running.saturating_sub(1);
        self.release(key);
    }

    /// A request that was waiting went away.
    pub fn cancel(&mut self, ticket: u64, key: &str) {
        self.waiting.retain(|t| *t != ticket);
        self.release(key);
    }

    fn release(&mut self, key: &str) {
        if let Some(n) = self.held.get_mut(key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.held.remove(key);
            }
        }
    }

    pub fn running(&self) -> usize {
        self.running
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Addresses with an attempt running or waiting.
    pub fn addresses_held(&self) -> usize {
        self.held.len()
    }
}

/// The anonymous lane: the scheduler, and the waiting.
pub struct AnonLane {
    scheduler: Mutex<Scheduler>,
    notify: Notify,
    clock: Arc<dyn Clock>,
    throttle: Arc<LoginThrottle>,
}

/// A place in the anonymous lane. Dropping it frees the place.
pub struct AnonPermit {
    lane: Arc<AnonLane>,
    key: String,
}

impl Drop for AnonPermit {
    fn drop(&mut self) {
        self.lane.lock().finish(&self.key);
        self.lane.notify.notify_waiters();
    }
}

/// A request waiting for its turn: if the client goes away, its place goes with it.
struct Waiting {
    lane: Arc<AnonLane>,
    ticket: u64,
    key: String,
    armed: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.armed {
            self.lane.lock().cancel(self.ticket, &self.key);
            self.lane.notify.notify_waiters();
        }
    }
}

impl AnonLane {
    pub fn new(clock: Arc<dyn Clock>, throttle: Arc<LoginThrottle>) -> Arc<AnonLane> {
        Arc::new(AnonLane {
            scheduler: Mutex::new(Scheduler::default()),
            notify: Notify::new(),
            clock,
            throttle,
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Scheduler> {
        self.scheduler.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait for a turn. `Err` at once when the address already has two attempts or five are waiting.
    pub async fn acquire(self: &Arc<Self>, key: &str) -> Result<AnonPermit, Reject> {
        let ticket = self.lock().enqueue(key)?;
        let mut waiting = Waiting {
            lane: self.clone(),
            ticket,
            key: key.to_string(),
            armed: true,
        };
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let now = self.clock.now_ms();
            let slow = self.throttle.is_slow();
            if self.lock().poll(ticket, now, slow) == Poll::Go {
                waiting.armed = false;
                return Ok(AnonPermit {
                    lane: self.clone(),
                    key: key.to_string(),
                });
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = tokio::time::sleep(POLL) => {}
            }
        }
    }

    pub fn running(&self) -> usize {
        self.lock().running()
    }

    pub fn waiting(&self) -> usize {
        self.lock().waiting()
    }
}

/// The reserved slot: one verification at a time, for the session lane and the known-device lane.
pub struct ReservedLane {
    slot: Arc<Semaphore>,
    waiting: AtomicUsize,
}

struct ReservedWaiter<'a>(&'a AtomicUsize);

impl Drop for ReservedWaiter<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ReservedLane {
    pub fn new() -> Arc<ReservedLane> {
        Arc::new(ReservedLane {
            slot: Arc::new(Semaphore::new(1)),
            waiting: AtomicUsize::new(0),
        })
    }

    /// The slot, waiting only behind the lane's own users, and not at all behind the anonymous queue.
    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit, Reject> {
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= RESERVED_WAITERS {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(Reject::ReservedBusy);
        }
        let _waiter = ReservedWaiter(&self.waiting);
        self.slot
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Reject::ReservedBusy)
    }
}

/// The verifications, behind the fence.
pub struct Hasher {
    engine: Arc<dyn PasswordEngine>,
    fence: Arc<Semaphore>,
}

impl Hasher {
    pub fn new(engine: Arc<dyn PasswordEngine>) -> Arc<Hasher> {
        Arc::new(Hasher {
            engine,
            fence: Arc::new(Semaphore::new(FENCE)),
        })
    }

    /// Verify `password` (normalised) against `stored`, on a blocking thread, behind the fence.
    pub async fn verify(&self, password: Zeroizing<String>, stored: String) -> Verdict {
        let Ok(_permit) = self.fence.clone().acquire_owned().await else {
            return Verdict::Mismatch;
        };
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || engine.verify(password.as_bytes(), &stored))
            .await
            .unwrap_or(Verdict::Mismatch)
    }

    /// Hash `password` (normalised), on a blocking thread, behind the fence.
    pub async fn hash(&self, password: Zeroizing<String>) -> Result<String, HashError> {
        let _permit = self
            .fence
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| HashError::Argon2("the hasher is shut down".into()))?;
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || engine.hash(password.as_bytes()))
            .await
            .map_err(|e| HashError::Argon2(e.to_string()))?
    }
}

/// What the session lane says of a verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionGate {
    Open,
    /// Ten in the last hour: seconds until the oldest ages out.
    Limited {
        retry_after_s: u64,
    },
}

#[derive(Default)]
struct SessionState {
    /// Consecutive wrong answers.
    wrong: u32,
    /// The times of the verifications in the last hour, oldest first.
    times: VecDeque<u64>,
}

/// The per-session counters of `elevate` and the `current` check of a password change.
pub struct SessionLane {
    state: Mutex<HashMap<String, SessionState>>,
    clock: Arc<dyn Clock>,
}

impl SessionLane {
    pub fn new(clock: Arc<dyn Clock>) -> SessionLane {
        SessionLane {
            state: Mutex::new(HashMap::new()),
            clock,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, SessionState>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Count a verification for `id`, unless it has had ten in the last hour.
    pub fn begin(&self, id: &str) -> SessionGate {
        let now = self.clock.now_ms();
        let mut map = self.lock();
        // Sessions that went quiet an hour ago are forgotten, so the table follows the live sessions.
        if map.len() > 256 {
            map.retain(|_, s| {
                s.times
                    .back()
                    .is_some_and(|t| now.saturating_sub(*t) < HOUR_MS)
            });
        }
        let s = map.entry(id.to_string()).or_default();
        while s
            .times
            .front()
            .is_some_and(|t| now.saturating_sub(*t) >= HOUR_MS)
        {
            s.times.pop_front();
        }
        if s.times.len() >= SESSION_PER_HOUR {
            let oldest = *s.times.front().unwrap_or(&now);
            let wait = HOUR_MS - now.saturating_sub(oldest);
            return SessionGate::Limited {
                retry_after_s: wait.div_ceil(1000).max(1),
            };
        }
        s.times.push_back(now);
        SessionGate::Open
    }

    /// A wrong answer. `true` when it is the fifth in a row: the caller revokes the session.
    pub fn wrong(&self, id: &str) -> bool {
        let mut map = self.lock();
        let s = map.entry(id.to_string()).or_default();
        s.wrong += 1;
        if s.wrong >= SESSION_WRONG_LIMIT {
            map.remove(id);
            true
        } else {
            false
        }
    }

    /// A right answer clears the run of wrong ones.
    pub fn right(&self, id: &str) {
        if let Some(s) = self.lock().get_mut(id) {
            s.wrong = 0;
        }
    }

    /// The session ended.
    pub fn forget(&self, id: &str) {
        self.lock().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::clock::ManualClock;
    use std::sync::atomic::AtomicBool;

    const T0: u64 = 1_790_000_000_000;

    // The scheduler, on no clock at all ----------------------------------------------------------

    fn go(s: &mut Scheduler, ticket: u64, now: u64, slow: bool) -> Poll {
        s.poll(ticket, now, slow)
    }

    #[test]
    fn two_run_at_once_and_the_third_waits_for_a_place() {
        let mut s = Scheduler::default();
        let a = s.enqueue("a").unwrap();
        let b = s.enqueue("b").unwrap();
        let c = s.enqueue("c").unwrap();
        assert_eq!(go(&mut s, a, T0, false), Poll::Go);
        assert_eq!(go(&mut s, b, T0, false), Poll::Go);
        assert_eq!(go(&mut s, c, T0, false), Poll::Wait, "two are running");
        assert_eq!(s.running(), 2);
        s.finish("a");
        assert_eq!(go(&mut s, c, T0, false), Poll::Go);
        assert_eq!((s.running(), s.waiting()), (2, 0));
    }

    #[test]
    fn a_queue_of_five_and_the_eighth_request_is_refused_at_once() {
        let mut s = Scheduler::default();
        // Two running, five waiting, from seven different addresses, each asking as it arrives (a request
        // takes a place in the queue and looks at once whether it may start).
        let mut tickets = Vec::new();
        for i in 0..7 {
            let t = s.enqueue(&format!("addr-{i}")).unwrap();
            tickets.push(t);
            let expected = if i < 2 { Poll::Go } else { Poll::Wait };
            assert_eq!(go(&mut s, t, T0, false), expected, "request {i}");
        }
        assert_eq!((s.running(), s.waiting()), (2, 5));
        assert_eq!(s.enqueue("addr-7"), Err(Reject::QueueFull));
        // One finishing frees a place in the queue for the next arrival, not before.
        s.finish("addr-0");
        assert_eq!(go(&mut s, tickets[2], T0, false), Poll::Go);
        assert!(s.enqueue("addr-7").is_ok());
        assert_eq!(s.enqueue("addr-8"), Err(Reject::QueueFull));
    }

    #[test]
    fn only_the_head_of_the_queue_may_start_and_the_order_is_arrival() {
        let mut s = Scheduler::default();
        let t: Vec<u64> = (0..4)
            .map(|i| s.enqueue(&format!("a{i}")).unwrap())
            .collect();
        assert_eq!(go(&mut s, t[1], T0, false), Poll::Wait, "not the head");
        assert_eq!(go(&mut s, t[0], T0, false), Poll::Go);
        assert_eq!(go(&mut s, t[1], T0, false), Poll::Go);
        assert_eq!(go(&mut s, t[3], T0, false), Poll::Wait);
        assert_eq!(go(&mut s, t[2], T0, false), Poll::Wait, "two are running");
    }

    #[test]
    fn an_address_may_hold_two_attempts_running_or_queued_counted_when_they_enqueue() {
        let mut s = Scheduler::default();
        let a1 = s.enqueue("same").unwrap();
        let _a2 = s.enqueue("same").unwrap();
        assert_eq!(
            s.enqueue("same"),
            Err(Reject::PerAddress),
            "the third of one address"
        );
        // Counted at enqueue: nothing has started, and it is refused all the same.
        assert_eq!(s.running(), 0);
        // Another address is unaffected, and the queue is not filled by one address.
        assert!(s.enqueue("other").is_ok());
        // Once one of them finishes, a place opens for the address.
        assert_eq!(go(&mut s, a1, T0, false), Poll::Go);
        s.finish("same");
        assert!(s.enqueue("same").is_ok());
    }

    #[test]
    fn one_address_cannot_fill_the_queue_and_the_owner_behind_it_still_gets_a_place() {
        let mut s = Scheduler::default();
        // An attacker from one address: two are admitted, the rest refused by the address cap.
        let mut admitted = 0;
        let mut refused = 0;
        for _ in 0..50 {
            match s.enqueue("attacker") {
                Ok(_) => admitted += 1,
                Err(Reject::PerAddress) => refused += 1,
                Err(other) => panic!("{other:?}"),
            }
        }
        assert_eq!((admitted, refused), (2, 48));
        assert!(s.enqueue("someone-else").is_ok());
    }

    #[test]
    fn a_request_that_goes_away_gives_its_place_back() {
        let mut s = Scheduler::default();
        let a = s.enqueue("a").unwrap();
        let b = s.enqueue("a").unwrap();
        assert_eq!(s.enqueue("a"), Err(Reject::PerAddress));
        s.cancel(a, "a");
        assert_eq!(s.waiting(), 1);
        assert!(s.enqueue("a").is_ok());
        // The remaining ticket is now the head.
        assert_eq!(go(&mut s, b, T0, false), Poll::Go);
        s.finish("a");
        s.cancel(999, "a");
        s.cancel(999, "a");
        assert_eq!(
            s.addresses_held(),
            0,
            "the counts fall to zero and are removed"
        );
    }

    #[test]
    fn slow_mode_runs_one_at_a_time_with_five_seconds_between_starts() {
        let mut s = Scheduler::default();
        let t: Vec<u64> = (0..4)
            .map(|i| s.enqueue(&format!("a{i}")).unwrap())
            .collect();
        assert_eq!(
            go(&mut s, t[0], T0, true),
            Poll::Go,
            "the first is not made to wait"
        );
        // The first is still running: nobody else starts, whatever the time.
        assert_eq!(go(&mut s, t[1], T0 + 60_000, true), Poll::Wait);
        s.finish("a0");
        // Finished at once, but the next start is 5 seconds after the last start.
        assert_eq!(go(&mut s, t[1], T0 + 4_999, true), Poll::Wait);
        assert_eq!(go(&mut s, t[1], T0 + 5_000, true), Poll::Go);
        s.finish("a1");
        assert_eq!(go(&mut s, t[2], T0 + 9_999, true), Poll::Wait);
        assert_eq!(go(&mut s, t[2], T0 + 10_000, true), Poll::Go);
        s.finish("a2");
    }

    #[test]
    fn slow_mode_is_twelve_starts_a_minute_however_fast_they_arrive_and_finish() {
        // A request every second for two minutes, each finishing at once: 12 starts in each minute.
        let mut s = Scheduler::default();
        let mut keys: HashMap<u64, String> = HashMap::new();
        let mut starts = Vec::new();
        for sec in 0..120u64 {
            let now = T0 + sec * 1000;
            let key = format!("visitor-{sec}");
            if let Ok(t) = s.enqueue(&key) {
                keys.insert(t, key);
            }
            while let Some(head) = s.waiting.front().copied() {
                if s.poll(head, now, true) == Poll::Go {
                    starts.push(sec);
                    s.finish(&keys[&head]);
                } else {
                    break;
                }
            }
        }
        assert_eq!(starts.iter().filter(|s| **s < 60).count(), 12, "{starts:?}");
        assert_eq!(
            starts.iter().filter(|s| **s >= 60).count(),
            12,
            "{starts:?}"
        );
        assert!(starts.windows(2).all(|w| w[1] - w[0] >= 5), "{starts:?}");
        // 12 a minute is 17,280 a day.
        assert_eq!(12 * 60 * 24, 17_280);
    }

    #[test]
    fn slow_mode_going_off_lets_two_run_again_and_going_on_holds_the_second() {
        let mut s = Scheduler::default();
        let a = s.enqueue("a").unwrap();
        let b = s.enqueue("b").unwrap();
        assert_eq!(go(&mut s, a, T0, false), Poll::Go);
        // Slow mode came on while one runs: the limit is one, so the second waits although only one runs.
        assert_eq!(go(&mut s, b, T0 + 10_000, true), Poll::Wait);
        assert_eq!(go(&mut s, b, T0 + 10_000, false), Poll::Go);
        assert_eq!(s.running(), 2);
    }

    // The async lane ----------------------------------------------------------------------------

    fn lane() -> (Arc<AnonLane>, Arc<LoginThrottle>, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(T0));
        let throttle = Arc::new(LoginThrottle::new(clock.clone()));
        (
            AnonLane::new(clock.clone(), throttle.clone()),
            throttle,
            clock,
        )
    }

    async fn settle() {
        tokio::time::sleep(Duration::from_millis(80)).await;
    }

    #[tokio::test]
    async fn a_waiting_request_starts_when_a_place_frees_and_not_before() {
        let (lane, _t, _c) = lane();
        let a = lane.acquire("a").await.unwrap();
        let b = lane.acquire("b").await.unwrap();
        let started = Arc::new(AtomicBool::new(false));
        let waiter = {
            let (lane, started) = (lane.clone(), started.clone());
            tokio::spawn(async move {
                let p = lane.acquire("c").await.unwrap();
                started.store(true, Ordering::SeqCst);
                drop(p);
            })
        };
        settle().await;
        assert!(!started.load(Ordering::SeqCst), "two are running");
        assert_eq!((lane.running(), lane.waiting()), (2, 1));
        drop(a);
        waiter.await.unwrap();
        assert!(started.load(Ordering::SeqCst));
        drop(b);
        assert_eq!((lane.running(), lane.waiting()), (0, 0));
    }

    #[tokio::test]
    async fn a_request_dropped_while_it_waits_gives_its_place_back() {
        let (lane, _t, _c) = lane();
        let _a = lane.acquire("a").await.unwrap();
        let _b = lane.acquire("b").await.unwrap();
        let waiter = {
            let lane = lane.clone();
            tokio::spawn(async move {
                let _ = lane.acquire("c").await;
            })
        };
        settle().await;
        assert_eq!(lane.waiting(), 1);
        waiter.abort();
        let _ = waiter.await;
        settle().await;
        assert_eq!(lane.waiting(), 0, "the client went away");
        // Its address may try again at once.
        let again = {
            let lane = lane.clone();
            tokio::spawn(async move {
                let _ = lane.acquire("c").await;
            })
        };
        settle().await;
        assert_eq!(lane.waiting(), 1);
        again.abort();
    }

    #[tokio::test]
    async fn slow_mode_spaces_the_starts_by_the_fake_clock_and_lets_the_owner_past_on_the_reserved_slot(
    ) {
        let (lane, throttle, clock) = lane();
        // 20 failures from 20 rotating addresses: slow mode.
        for i in 0..20 {
            throttle.address_failed(&format!("2001:db8:{i:x}::"));
        }
        assert!(throttle.is_slow());
        let reserved = ReservedLane::new();
        let starts = Arc::new(Mutex::new(Vec::<u64>::new()));
        let mut tasks = Vec::new();
        for i in 0..4 {
            let (lane, starts, clock) = (lane.clone(), starts.clone(), clock.clone());
            tasks.push(tokio::spawn(async move {
                let p = lane.acquire(&format!("visitor-{i}")).await.unwrap();
                starts.lock().unwrap().push(clock.now_ms());
                drop(p);
            }));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        settle().await;
        assert_eq!(
            starts.lock().unwrap().len(),
            1,
            "one started; the rest wait 5 seconds each"
        );
        // The owner's lane is not behind them: the reserved slot is free at once.
        let owner = tokio::time::timeout(Duration::from_millis(500), reserved.acquire()).await;
        assert!(
            owner.is_ok(),
            "the reserved slot did not wait behind the anonymous queue"
        );
        drop(owner);
        for expected in 2..=4 {
            clock.advance(SLOW_SPACING_MS);
            settle().await;
            assert_eq!(starts.lock().unwrap().len(), expected);
        }
        for t in tasks {
            t.await.unwrap();
        }
        let times = starts.lock().unwrap().clone();
        assert!(
            times.windows(2).all(|w| w[1] - w[0] >= SLOW_SPACING_MS),
            "{times:?}"
        );
    }

    #[tokio::test]
    async fn the_reserved_slot_serialises_its_own_users_and_refuses_the_sixth_waiter() {
        let reserved = ReservedLane::new();
        let first = reserved.acquire().await.unwrap();
        let mut waiters = Vec::new();
        // The number is the design's, written out: the test does not follow the constant.
        assert_eq!(RESERVED_WAITERS, 5);
        for _ in 0..5 {
            let r = reserved.clone();
            waiters.push(tokio::spawn(async move { r.acquire().await.map(drop) }));
        }
        settle().await;
        // Five are waiting behind the one holding it: the next is refused at once.
        assert_eq!(reserved.acquire().await.err(), Some(Reject::ReservedBusy));
        drop(first);
        for w in waiters {
            assert_eq!(w.await.unwrap(), Ok(()));
        }
        assert!(reserved.acquire().await.is_ok(), "and it is free again");
    }

    // The memory bound -----------------------------------------------------------------------------

    /// An engine that holds each call for a moment and records how many run at once, by lane (the first byte).
    struct Gauge {
        now: [AtomicUsize; 2],
        most: [AtomicUsize; 2],
        most_all: AtomicUsize,
        all: AtomicUsize,
        calls: AtomicUsize,
    }

    impl Gauge {
        fn new() -> Arc<Gauge> {
            Arc::new(Gauge {
                now: [AtomicUsize::new(0), AtomicUsize::new(0)],
                most: [AtomicUsize::new(0), AtomicUsize::new(0)],
                most_all: AtomicUsize::new(0),
                all: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
            })
        }

        fn run(&self, lane: usize) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let n = self.now[lane].fetch_add(1, Ordering::SeqCst) + 1;
            self.most[lane].fetch_max(n, Ordering::SeqCst);
            let all = self.all.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_all.fetch_max(all, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(25));
            self.all.fetch_sub(1, Ordering::SeqCst);
            self.now[lane].fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl PasswordEngine for Gauge {
        fn hash(&self, password: &[u8]) -> Result<String, HashError> {
            self.run(usize::from(password.first() == Some(&b'R')));
            Ok("hash".into())
        }

        fn verify(&self, password: &[u8], _stored: &str) -> Verdict {
            self.run(usize::from(password.first() == Some(&b'R')));
            Verdict::Mismatch
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_flood_never_runs_more_than_two_anonymous_and_one_reserved_verification_at_once() {
        let (anon, _throttle, _clock) = lane();
        let reserved = ReservedLane::new();
        let gauge = Gauge::new();
        let hasher = Hasher::new(gauge.clone());
        let mut tasks = Vec::new();
        // 60 anonymous requests from 60 addresses (some are refused at once: the queue is five) and 30 requests
        // of the reserved lane, all at the same moment.
        for i in 0..60 {
            let (anon, hasher) = (anon.clone(), hasher.clone());
            tasks.push(tokio::spawn(async move {
                if let Ok(p) = anon.acquire(&format!("addr-{i}")).await {
                    let _ = hasher
                        .verify(Zeroizing::new("A password".into()), "x".into())
                        .await;
                    drop(p);
                    1
                } else {
                    0
                }
            }));
        }
        for _ in 0..30 {
            let (reserved, hasher) = (reserved.clone(), hasher.clone());
            tasks.push(tokio::spawn(async move {
                if let Ok(p) = reserved.acquire().await {
                    let _ = hasher
                        .verify(Zeroizing::new("R password".into()), "x".into())
                        .await;
                    drop(p);
                    1
                } else {
                    0
                }
            }));
        }
        let mut done = 0;
        for t in tasks {
            done += t.await.unwrap();
        }
        assert!(done >= 2, "some ran ({done})");
        assert!(
            done < 90,
            "and the flood was turned away in part ({done} of 90 were admitted)"
        );
        let (a, r, all) = (
            gauge.most[0].load(Ordering::SeqCst),
            gauge.most[1].load(Ordering::SeqCst),
            gauge.most_all.load(Ordering::SeqCst),
        );
        assert!(a <= RUNNING, "{a} anonymous verifications at once");
        assert!(r <= 1, "{r} reserved verifications at once");
        assert!(all <= FENCE, "{all} verifications at once in all");
        assert!(a >= 1 && r >= 1, "both lanes ran ({a}, {r})");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_fence_holds_a_caller_that_forgot_its_lane_to_three_at_once() {
        let gauge = Gauge::new();
        let hasher = Hasher::new(gauge.clone());
        let mut tasks = Vec::new();
        for i in 0..24 {
            let hasher = hasher.clone();
            tasks.push(tokio::spawn(async move {
                let pw = Zeroizing::new(if i % 2 == 0 { "A pw" } else { "R pw" }.to_string());
                if i % 3 == 0 {
                    let _ = hasher.hash(pw).await;
                } else {
                    let _ = hasher.verify(pw, "x".into()).await;
                }
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(gauge.calls.load(Ordering::SeqCst), 24);
        assert!(gauge.most_all.load(Ordering::SeqCst) <= FENCE);
        assert!(
            gauge.most_all.load(Ordering::SeqCst) >= 2,
            "and they did run together"
        );
    }

    #[tokio::test]
    async fn the_hasher_reports_what_the_engine_says() {
        struct Fixed;
        impl PasswordEngine for Fixed {
            fn hash(&self, p: &[u8]) -> Result<String, HashError> {
                Ok(format!("h:{}", p.len()))
            }
            fn verify(&self, p: &[u8], stored: &str) -> Verdict {
                if stored == "right" && p == b"pw" {
                    Verdict::Match { rehash: true }
                } else {
                    Verdict::Mismatch
                }
            }
        }
        let h = Hasher::new(Arc::new(Fixed));
        assert_eq!(
            h.hash(Zeroizing::new("abcd".into())).await,
            Ok("h:4".to_string())
        );
        assert_eq!(
            h.verify(Zeroizing::new("pw".into()), "right".into()).await,
            Verdict::Match { rehash: true }
        );
        assert_eq!(
            h.verify(Zeroizing::new("pw".into()), "wrong".into()).await,
            Verdict::Mismatch
        );
    }

    // The session lane ---------------------------------------------------------------------------

    #[test]
    fn five_wrong_answers_in_a_row_revoke_the_session_and_a_right_one_clears_the_run() {
        let clock = Arc::new(ManualClock::new(T0));
        let lane = SessionLane::new(clock);
        for _ in 0..4 {
            assert!(!lane.wrong("s1"));
        }
        lane.right("s1");
        for _ in 0..4 {
            assert!(!lane.wrong("s1"), "cleared by the right answer");
        }
        assert!(lane.wrong("s1"), "the fifth");
        // Another session is on its own.
        for _ in 0..4 {
            assert!(!lane.wrong("s2"));
        }
        lane.forget("s2");
        for _ in 0..4 {
            assert!(!lane.wrong("s2"));
        }
    }

    #[test]
    fn a_session_may_verify_ten_times_an_hour_and_the_eleventh_is_told_when_to_come_back() {
        let clock = Arc::new(ManualClock::new(T0));
        let lane = SessionLane::new(clock.clone());
        for i in 0..10 {
            assert_eq!(lane.begin("s"), SessionGate::Open, "{i}");
            clock.advance(60_000);
        }
        // 10 minutes in: the eleventh, 50 minutes until the first ages out.
        assert_eq!(
            lane.begin("s"),
            SessionGate::Limited {
                retry_after_s: 3000
            }
        );
        assert_eq!(lane.begin("other"), SessionGate::Open);
        // A moment later the first is an hour old.
        clock.advance(50 * 60_000);
        assert_eq!(lane.begin("s"), SessionGate::Open);
        assert!(
            matches!(lane.begin("s"), SessionGate::Limited { .. }),
            "the second is still inside"
        );
    }

    #[test]
    fn a_refusal_says_how_long_to_wait_and_slow_mode_says_five_seconds() {
        assert_eq!(Reject::QueueFull.retry_after_s(true), 5);
        assert_eq!(Reject::PerAddress.retry_after_s(true), 5);
        assert_eq!(Reject::QueueFull.retry_after_s(false), 2);
        assert_eq!(Reject::ReservedBusy.retry_after_s(false), 1);
    }
}
