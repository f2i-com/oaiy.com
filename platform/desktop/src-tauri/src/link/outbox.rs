//! Plugin events on their way to the linked account.
//!
//! A plugin's event (Aokie's calls and texts) is handled on this machine
//! first: local triggers, and the calendar's record of an appointment asked
//! for on a call. When the desktop is linked, the same event also goes to the
//! account: it may start the account's flows (a run reserved at FormLogic,
//! which may text the caller) and run its apps' logic scripts (records
//! written there). That half needs FormLogic, which may not be there, and it
//! must not hold up the first.
//!
//! So the event is written here, one file per event in `<data>/link/outbox`,
//! before the plugin is told it arrived, and a thread of its own sends them to
//! the account, oldest first (a call's end must not reach FormLogic before its
//! start). A failure that waiting can mend (no connection, FormLogic down or
//! busy, its flows not readable yet) keeps the event, and those after it, for
//! the next try: five seconds, doubling to five minutes, and at once when the
//! link's heartbeat reaches FormLogic again. One FormLogic refuses outright
//! goes to the dead letters with its reason, and the rest carry on.
//!
//! Sending again is harmless: a run is reserved under the event's own key,
//! and a record under one made from the event, the script and the effect.
//!
//! An event queued for one link is never sent over another: if the desktop is
//! linked again since (perhaps to another account), it goes to the dead
//! letters, where a person can send it on. While unlinked, events wait.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// More than this waiting and a new event goes to the dead letters instead.
const MAX_QUEUED: usize = 5000;
const FIRST_RETRY: Duration = Duration::from_secs(5);
const MAX_RETRY: Duration = Duration::from_secs(300);
/// How long the sending thread sleeps when there is nothing to do.
const IDLE: Duration = Duration::from_secs(30);

/// What sending one event to the account came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery {
    /// Sent, or nothing on the account wanted it.
    Done,
    /// Not now: FormLogic could not be reached, or was down. Kept, and tried again.
    Later(String),
    /// Refused: it goes to the dead letters, with why.
    Refused(String),
}

/// One queued event, as kept on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    plugin: String,
    /// The account it is for: its id, or its address.
    account: String,
    envelope: Value,
    queued_at: String,
    #[serde(default)]
    attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

/// How the outbox stands, for the dashboard.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// Events waiting to reach the account.
    pub waiting: usize,
    /// When the oldest of them happened here.
    pub oldest_at: Option<String>,
    pub last_error: Option<String>,
    pub last_sent_at: Option<String>,
    pub next_attempt_at: Option<String>,
}

pub struct Outbox {
    dir: PathBuf,
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    next_seq: u64,
    /// What is on disk, in order: the file's number, and the request id of an
    /// `aokie.appointment.requested` event (the calendar waits for those).
    queued: BTreeMap<u64, Option<String>>,
    oldest_at: Option<String>,
    failures: u32,
    failed_at: Option<DateTime<Utc>>,
    retry_at: Option<Instant>,
    retry_at_wall: Option<DateTime<Utc>>,
    last_error: Option<String>,
    last_sent_at: Option<DateTime<Utc>>,
}

/// The outbox of the running desktop, for the calendar's question below.
static CURRENT: Mutex<Option<Weak<Outbox>>> = Mutex::new(None);

/// Request ids of `aokie.appointment.requested` events not yet sent: FormLogic's
/// own flow has not had them, so it has not made its record of the request yet.
pub fn waiting_request_ids() -> HashSet<String> {
    CURRENT.lock().ok().and_then(|g| g.as_ref().and_then(Weak::upgrade)).map(|o| o.waiting_request_ids()).unwrap_or_default()
}

/// How the running desktop's outbox stands (nothing waiting when there is none).
pub fn current_status() -> Status {
    CURRENT.lock().ok().and_then(|g| g.as_ref().and_then(Weak::upgrade)).map(|o| o.status()).unwrap_or_default()
}

/// Which link an event is kept for: the desktop connection the link made, or
/// the provider's address for a link from before connections had ids.
pub fn account_key(account: &super::LinkedAccount) -> String {
    account.account_id.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| account.base_url.clone())
}

fn request_id(envelope: &Value) -> Option<String> {
    (envelope.get("name").and_then(Value::as_str) == Some("aokie.appointment.requested"))
        .then(|| envelope.pointer("/data/requestId").and_then(Value::as_str).map(str::to_string))
        .flatten()
        .filter(|s| !s.is_empty())
}

fn stamp(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl Outbox {
    /// The outbox in `data_dir/link/outbox`, with whatever an earlier run left in it.
    pub fn open(data_dir: &Path) -> Arc<Outbox> {
        let dir = data_dir.join("link").join("outbox");
        let mut state = State::default();
        if let Ok(read) = std::fs::read_dir(&dir) {
            for f in read.flatten() {
                let name = f.file_name().to_string_lossy().to_string();
                let Some(seq) = name.strip_suffix(".json").and_then(|n| n.parse::<u64>().ok()) else { continue };
                let entry = std::fs::read_to_string(f.path()).ok().and_then(|t| serde_json::from_str::<Entry>(&t).ok());
                let Some(entry) = entry else {
                    // Unreadable: kept aside rather than retried for ever or lost.
                    let _ = std::fs::rename(f.path(), f.path().with_extension("json.unreadable"));
                    continue;
                };
                state.queued.insert(seq, request_id(&entry.envelope));
                if state.oldest_at.as_ref().map_or(true, |o| entry.queued_at < *o) {
                    state.oldest_at = Some(entry.queued_at.clone());
                }
                state.next_seq = state.next_seq.max(seq + 1);
            }
        }
        Arc::new(Outbox { dir, state: Mutex::new(state), wake: Condvar::new() })
    }

    /// Make this the outbox the calendar asks about.
    pub fn make_current(self: &Arc<Self>) {
        *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::downgrade(self));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn path(&self, seq: u64) -> PathBuf {
        self.dir.join(format!("{seq:020}.json"))
    }

    fn write(&self, seq: u64, entry: &Entry) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("could not make {}: {e}", self.dir.display()))?;
        let path = self.path(seq);
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string(entry).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, text).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("could not keep {}: {e}", path.display()))
    }

    /// Keep an event to send to `account`. Err when it could not be written
    /// (the plugin is then not told it arrived, and sends it again), or when
    /// too many are already waiting (`full`).
    pub fn enqueue(&self, plugin: &str, account: &str, envelope: &Value) -> Result<(), Enqueue> {
        let mut state = self.lock();
        if state.queued.len() >= MAX_QUEUED {
            return Err(Enqueue::Full);
        }
        let seq = state.next_seq;
        let queued_at = stamp(Utc::now());
        let entry = Entry { plugin: plugin.to_string(), account: account.to_string(), envelope: envelope.clone(), queued_at: queued_at.clone(), attempts: 0, last_error: None };
        self.write(seq, &entry).map_err(Enqueue::NotWritten)?;
        state.next_seq = seq + 1;
        state.queued.insert(seq, request_id(envelope));
        state.oldest_at.get_or_insert(queued_at);
        drop(state);
        self.wake.notify_all();
        Ok(())
    }

    pub fn status(&self) -> Status {
        let s = self.lock();
        Status {
            waiting: s.queued.len(),
            oldest_at: s.oldest_at.clone().filter(|_| !s.queued.is_empty()),
            last_error: s.last_error.clone(),
            last_sent_at: s.last_sent_at.map(stamp),
            next_attempt_at: s.retry_at_wall.filter(|_| !s.queued.is_empty()).map(stamp),
        }
    }

    /// The last try failed: the next is a retry.
    pub fn retrying(&self) -> bool {
        self.lock().failures > 0
    }

    fn waiting_request_ids(&self) -> HashSet<String> {
        self.lock().queued.values().flatten().cloned().collect()
    }

    /// Send what is waiting, oldest first, until one cannot go yet.
    ///
    /// `account` is the account linked now (None: unlinked, and all waits).
    /// `heard_at` is when the link last reached FormLogic: a failure older
    /// than that is tried again at once. `deliver` sends one event; `dead`
    /// keeps one that cannot be sent, with why.
    pub fn send_due(
        &self,
        account: Option<&str>,
        heard_at: Option<DateTime<Utc>>,
        deliver: &dyn Fn(&str, &Value) -> Delivery,
        dead: &dyn Fn(&str, &Value, String),
    ) {
        let Some(account) = account else { return };
        loop {
            let seq = {
                let s = self.lock();
                let heard = s.failed_at.is_some_and(|f| heard_at.is_some_and(|h| h > f));
                if !heard && s.retry_at.is_some_and(|t| Instant::now() < t) {
                    return;
                }
                match s.queued.keys().next() {
                    Some(seq) => *seq,
                    None => return,
                }
            };
            let path = self.path(seq);
            let entry = match std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Entry>(&t).ok()) {
                Some(e) => e,
                None => {
                    // Gone or spoiled on disk: nothing to send.
                    if path.exists() {
                        let _ = std::fs::rename(&path, path.with_extension("json.unreadable"));
                    }
                    self.forget(seq);
                    continue;
                }
            };
            if entry.account != account {
                dead(
                    &entry.plugin,
                    &entry.envelope,
                    "not sent: this desktop was linked again, or to another account, after the event was kept; \
                     redrive it if it belongs to the account linked now"
                        .into(),
                );
                self.forget(seq);
                continue;
            }
            match deliver(&entry.plugin, &entry.envelope) {
                Delivery::Done => {
                    self.forget(seq);
                    let mut s = self.lock();
                    s.failures = 0;
                    s.failed_at = None;
                    s.retry_at = None;
                    s.retry_at_wall = None;
                    s.last_error = None;
                    s.last_sent_at = Some(Utc::now());
                }
                Delivery::Refused(why) => {
                    dead(&entry.plugin, &entry.envelope, why);
                    self.forget(seq);
                }
                Delivery::Later(why) => {
                    let mut kept = entry;
                    kept.attempts += 1;
                    kept.last_error = Some(why.clone());
                    let _ = self.write(seq, &kept);
                    let mut s = self.lock();
                    if s.last_error.as_deref() != Some(why.as_str()) {
                        // Once per reason, not once per try.
                        log::warn!("events for the linked account are waiting ({} queued): {why}", s.queued.len());
                    }
                    s.failures += 1;
                    let wait = FIRST_RETRY.saturating_mul(1u32 << s.failures.saturating_sub(1).min(10)).min(MAX_RETRY);
                    let now = Utc::now();
                    s.failed_at = Some(now);
                    s.retry_at = Some(Instant::now() + wait);
                    s.retry_at_wall = Some(now + chrono::Duration::from_std(wait).unwrap_or_default());
                    s.last_error = Some(why);
                    return;
                }
            }
        }
    }

    fn forget(&self, seq: u64) {
        let _ = std::fs::remove_file(self.path(seq));
        let mut s = self.lock();
        s.queued.remove(&seq);
        if s.queued.is_empty() {
            s.oldest_at = None;
        } else if let Some(next) = s.queued.keys().next().copied() {
            s.oldest_at = std::fs::read_to_string(self.path(next)).ok().and_then(|t| serde_json::from_str::<Entry>(&t).ok()).map(|e| e.queued_at);
        }
    }

    /// Sleep until an event is queued, a retry is due, or `IDLE` passes.
    pub fn wait_for_work(&self) {
        let s = self.lock();
        let wait = match (s.queued.is_empty(), s.retry_at) {
            (false, Some(t)) => t.saturating_duration_since(Instant::now()).min(IDLE).max(Duration::from_millis(50)),
            (false, None) => Duration::from_millis(50),
            (true, _) => IDLE,
        };
        let _ = self.wake.wait_timeout(s, wait);
    }
}

/// Why an event could not be kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Enqueue {
    Full,
    NotWritten(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::RefCell;

    fn dir() -> PathBuf {
        std::env::temp_dir().join(format!("oaiy-outbox-{}", uuid::Uuid::new_v4().simple()))
    }

    fn event(name: &str, key: &str) -> Value {
        json!({"name": name, "idempotencyKey": key, "data": {"requestId": format!("req-{key}")}})
    }

    #[test]
    fn events_wait_while_formlogic_is_away_and_go_in_order_when_it_is_back() {
        let d = dir();
        let o = Outbox::open(&d);
        o.enqueue("aokie", "acct", &event("aokie.call.started", "1")).unwrap();
        o.enqueue("aokie", "acct", &event("aokie.appointment.requested", "2")).unwrap();
        o.enqueue("aokie", "acct", &event("aokie.call.ended", "3")).unwrap();
        assert_eq!(o.waiting_request_ids(), HashSet::from(["req-2".to_string()]), "the calendar knows which request has not gone");

        let tried = RefCell::new(Vec::new());
        let down = |_: &str, e: &Value| {
            tried.borrow_mut().push(e["idempotencyKey"].as_str().unwrap().to_string());
            Delivery::Later("formlogic.com can't be reached: it refused the connection (is it running?)".into())
        };
        o.send_due(Some("acct"), None, &down, &|_, _, _| panic!("nothing is lost"));
        assert_eq!(*tried.borrow(), ["1"], "the first waits, and the rest wait behind it");
        let st = o.status();
        assert_eq!(st.waiting, 3);
        assert!(st.last_error.unwrap().contains("refused"));
        assert!(st.next_attempt_at.is_some());

        // Not again before its time...
        o.send_due(Some("acct"), None, &down, &|_, _, _| {});
        assert_eq!(tried.borrow().len(), 1);
        // ...unless the heartbeat has since reached FormLogic.
        let sent = RefCell::new(Vec::new());
        let up = |_: &str, e: &Value| {
            sent.borrow_mut().push(e["idempotencyKey"].as_str().unwrap().to_string());
            Delivery::Done
        };
        o.send_due(Some("acct"), Some(Utc::now() + chrono::Duration::seconds(1)), &up, &|_, _, _| panic!("nothing is lost"));
        assert_eq!(*sent.borrow(), ["1", "2", "3"], "in the order they happened");
        assert_eq!(o.status().waiting, 0);
        assert!(o.waiting_request_ids().is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn what_is_waiting_survives_a_restart() {
        let d = dir();
        Outbox::open(&d).enqueue("aokie", "acct", &event("aokie.call.started", "1")).unwrap();
        Outbox::open(&d).enqueue("aokie", "acct", &event("aokie.call.ended", "2")).unwrap();
        let again = Outbox::open(&d);
        assert_eq!(again.status().waiting, 2);
        let sent = RefCell::new(Vec::new());
        again.send_due(Some("acct"), None, &|_: &str, e: &Value| {
            sent.borrow_mut().push(e["idempotencyKey"].as_str().unwrap().to_string());
            Delivery::Done
        }, &|_, _, _| {});
        assert_eq!(*sent.borrow(), ["1", "2"]);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn one_formlogic_refuses_goes_to_the_dead_letters_and_the_rest_carry_on() {
        let d = dir();
        let o = Outbox::open(&d);
        o.enqueue("aokie", "acct", &event("aokie.call.started", "bad")).unwrap();
        o.enqueue("aokie", "acct", &event("aokie.call.ended", "good")).unwrap();
        let dead = RefCell::new(Vec::new());
        o.send_due(
            Some("acct"),
            None,
            &|_: &str, e: &Value| if e["idempotencyKey"] == "bad" { Delivery::Refused("HTTP 400: no such flow".into()) } else { Delivery::Done },
            &|_, e, why| dead.borrow_mut().push((e["idempotencyKey"].as_str().unwrap().to_string(), why)),
        );
        assert_eq!(*dead.borrow(), [("bad".to_string(), "HTTP 400: no such flow".to_string())]);
        assert_eq!(o.status().waiting, 0);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn events_wait_while_unlinked_and_never_go_to_another_account() {
        let d = dir();
        let o = Outbox::open(&d);
        o.enqueue("aokie", "acct-1", &event("aokie.call.started", "1")).unwrap();
        o.send_due(None, None, &|_: &str, _: &Value| panic!("not while unlinked"), &|_, _, _| panic!("kept"));
        assert_eq!(o.status().waiting, 1);
        let dead = RefCell::new(0);
        o.send_due(Some("acct-2"), None, &|_: &str, _: &Value| panic!("not to another account"), &|_, _, why| {
            assert!(why.contains("linked again"), "{why}");
            *dead.borrow_mut() += 1;
        });
        assert_eq!((*dead.borrow(), o.status().waiting), (1, 0));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn tries_wait_longer_each_time_up_to_five_minutes() {
        let d = dir();
        let o = Outbox::open(&d);
        o.enqueue("aokie", "acct", &event("aokie.call.started", "1")).unwrap();
        let mut waits = Vec::new();
        for _ in 0..9 {
            o.lock().retry_at = None;
            o.send_due(Some("acct"), None, &|_: &str, _: &Value| Delivery::Later("down".into()), &|_, _, _| {});
            let s = o.lock();
            waits.push(s.retry_at.unwrap().saturating_duration_since(Instant::now()).as_secs() + 1);
        }
        assert_eq!(waits[0], 5);
        assert_eq!(waits[1], 10);
        assert_eq!(*waits.last().unwrap(), 300);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_full_outbox_says_so_instead_of_growing_without_end() {
        let d = dir();
        let o = Outbox::open(&d);
        o.lock().queued.extend((0..MAX_QUEUED as u64).map(|i| (i + 1_000_000, None)));
        assert_eq!(o.enqueue("aokie", "acct", &event("aokie.call.started", "x")), Err(Enqueue::Full));
        let _ = std::fs::remove_dir_all(d);
    }
}
