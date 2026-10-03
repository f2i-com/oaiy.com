//! What the user is told: the connection state, the events of the loop, and the health answer.
//!
//! Nothing here carries a credential, a header or a body: an event says what happened in the words of README 5.1.1 (the outcome, the action, the report), and a sink (a status
//! line in the desktop's Settings, a notification of the phone's foreground service, a log) decides how to show it. A message the relay supplied (`error.message`, a `ctl`
//! notice) is data, shown as plain text under the label "message from your relay (unverified)" and never acted on.

use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::json;
use crate::poll::{Action, Counters, Outcome, Report};

/// What the connection to the relay is, for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not started.
    Idle,
    /// Proving the relay's identity (before any token is sent).
    Proving,
    /// The relay answered the last poll (a `429` counts: it proves the relay answered).
    Connected,
    /// Three failures in a row: the relay is reported unreachable. The loop goes on at the capped pauses.
    Unreachable,
    /// `401`: the relay does not accept the credential. The client refreshes it once if it can, otherwise it stops and asks to enrol again.
    Rejected,
    /// `401 revoked`: the device was revoked. The credential is forgotten and a new key is required.
    Revoked,
    /// `426` and `minClient` above this client's level: update the product.
    UpgradeRequired,
    /// The identity proof did not verify: "not who it was". The credential is kept and not sent.
    Suspect,
    /// Any other `4xx`: a defect of the client or a relay that is switched off. The loop has ended.
    Stopped,
}

/// Something that happened in the loop.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The state changed.
    State(ConnectionState),
    /// An answer was judged: its outcome and the counters after it (no body, no header).
    Answer {
        /// The outcome.
        outcome: Outcome,
        /// The HTTP status, when there was a response.
        status: Option<u16>,
        /// The counters after.
        counters: Counters,
    },
    /// A report of P7.
    Report(Report),
    /// An action of P7.
    Action(Action),
    /// Accepted items were written to the store.
    Accepted {
        /// How many.
        count: usize,
        /// The `since` that was stored.
        since: u64,
    },
    /// A reset was adopted: "mailbox reset: in-flight items may be lost".
    MailboxReset,
    /// The relay's clock differs from ours by more than 60 seconds (a warning for the owner; nothing is judged by it).
    ClockMismatch {
        /// The offset, in seconds.
        offset_s: i64,
    },
    /// `info` was (re)read and proved.
    InfoProved,
}

/// Where events go.
pub trait StatusSink: Send + Sync {
    /// Called from the loop's thread: keep it short.
    fn event(&self, event: &Event);
}

/// A sink that drops everything.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl StatusSink for NullSink {
    fn event(&self, _: &Event) {}
}

/// A sink that keeps everything, for tests.
#[derive(Default)]
pub struct RecordingSink(Mutex<Vec<Event>>);

impl RecordingSink {
    /// An empty recorder.
    pub fn new() -> RecordingSink {
        RecordingSink::default()
    }

    /// What was recorded, in order.
    pub fn events(&self) -> Vec<Event> {
        self.0.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// The states, in order.
    pub fn states(&self) -> Vec<ConnectionState> {
        self.events().into_iter().filter_map(|e| if let Event::State(s) = e { Some(s) } else { None }).collect()
    }

    /// The reports, in order.
    pub fn reports(&self) -> Vec<Report> {
        self.events().into_iter().filter_map(|e| if let Event::Report(r) = e { Some(r) } else { None }).collect()
    }
}

impl StatusSink for RecordingSink {
    fn event(&self, event: &Event) {
        if let Ok(mut v) = self.0.lock() {
            v.push(event.clone());
        }
    }
}

/// `GET /v1/health`: exactly `{"ok":true,"time":N,"authHeaderSeen":bool}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Health {
    /// The relay's time.
    pub time: u64,
    /// True when the request carried an `Authorization` header through the web stack (a host that strips it can be told apart from a revoked device).
    pub auth_header_seen: bool,
}

impl Health {
    /// Reads the answer.
    pub fn parse(body: &[u8]) -> Result<Health> {
        let doc = json::parse(body)?;
        if doc.get("ok").and_then(json::Json::as_bool) != Some(true) {
            return Err(Error::Invalid("health: ok"));
        }
        Ok(Health {
            time: doc.get_uint53("time").ok_or(Error::Invalid("health: time"))?,
            auth_header_seen: doc.get("authHeaderSeen").and_then(json::Json::as_bool).ok_or(Error::Invalid("health: authHeaderSeen"))?,
        })
    }
}
