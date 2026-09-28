//! The calendar syncs with FormLogic while the desktop is linked to it, and
//! needs it for nothing: every change is made here first, and goes to
//! FormLogic when FormLogic can be reached.
//!
//! FormLogic keeps appointments as records of the Aokie receptionist pack's
//! `appointments` form (answers `caller_name`, `service`, `date`, `time`,
//! `status`, `phone`, `call_id`, `request_id`, `source`, `notes`), reached with
//! the link's key: `GET /api/v1/app-logic` names the form, `GET/POST/PUT/DELETE
//! /api/v1/forms/{form}/responses[/{id}]` read and write its records. Hours,
//! services and lengths are the desktop's alone (FormLogic has none).
//!
//! A sync runs every minute, a moment after a change here, and when asked
//! (`POST /api/calendar/sync`):
//!
//! 1. **Pull.** The records that changed since the last sync, with the ones
//!    deleted since then (`?updatedSince=`). A FormLogic without that, or a
//!    first sync, lists them all; a local appointment whose record is not
//!    among them is looked up, and gone only if FormLogic says so (404).
//!    Each record pairs with a local appointment by its id, or by its
//!    `request_id`: a call's request id (FormLogic's own flow records a call's
//!    request too), or `oaiy:<local id>` for one made here.
//! 2. **Merge.** What changed on one side goes to the other. When both
//!    changed, a final status (cancelled, done) wins, else the later change.
//! 3. **Push.** Local changes go with `If-Match` (FormLogic's etag), so one
//!    made in FormLogic meanwhile is merged rather than overwritten. A new
//!    appointment is first looked for by its `request_id` (a create that
//!    landed unanswered, or FormLogic's record of the same call), and only
//!    then created. One from a call waits five minutes, and while its event
//!    is still on its way to FormLogic, as FormLogic's flow makes its own.
//! 4. **Deletions.** An appointment deleted here leaves a tombstone until
//!    FormLogic's copy is deleted, so it does not come back; one deleted in
//!    FormLogic is deleted here. A deletion wins over an edit on the other
//!    side. If FormLogic's flow and this desktop both recorded one call, the
//!    desktop's copy is deleted and FormLogic's kept.
//!
//! Offline, each failed sync waits longer before the next (one minute,
//! doubling to ten), and a sync runs as soon as the link's heartbeat reaches
//! FormLogic again. One record FormLogic refuses is set aside with the reason
//! (and sent again once it changes here); it does not stop the rest.
//!
//! A local appointment remembers its FormLogic copy in `formlogic`: `{id,
//! updatedAt, etag, syncedAt}` (FormLogic's version then, and the local
//! `updatedAt` it was in step with).

use std::collections::HashSet;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{Appointment, Book, Calendar, Change, NewAppointment, Status, Tombstone};
use crate::link::LinkHandle;

/// How often a sync runs while all is well.
const EVERY: Duration = Duration::from_secs(60);
/// The longest wait between tries while FormLogic cannot be reached.
const BACKOFF_MAX: Duration = Duration::from_secs(600);
/// How often the sync thread looks at whether a sync is due.
const TICK: Duration = Duration::from_secs(5);
/// A change here is sent this long after the last one, so a burst goes as one.
const SETTLE: Duration = Duration::from_secs(2);
/// How long a manual sync waits for an answer before saying it is still going.
const ASK_WAIT: Duration = Duration::from_secs(10);
/// How long a call's request is left for FormLogic's own flow to record.
const CALL_GRACE: chrono::Duration = chrono::Duration::minutes(5);
/// How long a tombstone is kept for a copy of a deleted request that turns up late.
const KEEP_TOMBSTONES: chrono::Duration = chrono::Duration::days(30);
/// A pull asks from a little before the last one, for a write that landed as it ran.
const OVERLAP: chrono::Duration = chrono::Duration::seconds(10);
const PAGE: usize = 500;
const MAX_PAGES: usize = 40;
/// Local appointments missing from a full listing that are looked up per sync.
const MAX_LOOKUPS: usize = 25;
const EPOCH: &str = "1970-01-01 00:00:00";

/// Where the sync has got to, kept with the calendar.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct State {
    /// FormLogic's appointments form the local records are paired with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<String>,
    /// FormLogic's time when the last whole pull began: the next asks for what changed since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_at: Option<String>,
    /// FormLogic records to delete: this desktop's copy of a call FormLogic's
    /// flow also recorded, or FormLogic's copy of a request deleted here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discard: Vec<String>,
}

/// An appointment's FormLogic copy, as kept in its `formlogic`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Remote {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    etag: Option<String>,
    /// The local `updatedAt` it was in step with.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    synced_at: String,
    /// This desktop made FormLogic's record.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    created: bool,
    /// A create was sent: FormLogic may have it even with no answer yet.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    sent: bool,
    /// FormLogic refused this version; sent again once it changes here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    refused: Option<Refused>,
    /// The answers both sides last agreed on (or, before it was ever paired,
    /// how it began here): what tells a change made here from one made there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Refused {
    message: String,
    /// The local `updatedAt` refused.
    of: String,
}

fn remote(a: &Appointment) -> Remote {
    a.formlogic.clone().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

fn set_remote(a: &mut Appointment, r: Remote) {
    a.formlogic = (r != Remote::default()).then(|| serde_json::to_value(r).ok()).flatten();
}

/// A new appointment's `formlogic`: only how it began.
pub(super) fn first_version(a: &Appointment) -> Option<Value> {
    serde_json::to_value(Remote { base: Some(answers(a)), ..Default::default() }).ok()
}

/// The answers a person edits, on either side.
const FIELDS: [&str; 7] = ["caller_name", "service", "date", "time", "status", "phone", "notes"];

/// One field's value on both sides the same (a time with or without seconds).
fn same(field: &str, x: &Value, y: &Value) -> bool {
    let text = |v: &Value| v.as_str().unwrap_or("").trim().to_string();
    let (x, y) = (text(x), text(y));
    if field == "time" {
        x.get(..5).unwrap_or(&x) == y.get(..5).unwrap_or(&y)
    } else {
        x == y
    }
}

/// A three-way merge of one record: the fields to take from FormLogic. A
/// field changed on one side only goes as that side has it; one changed on
/// both goes to a final status (cancelled, done), else to the later change.
fn merge(base: &Value, ours: &Value, theirs: &Value, theirs_later: bool) -> Vec<&'static str> {
    FIELDS
        .into_iter()
        .filter(|f| {
            let (b, o, t) = (&base[*f], &ours[*f], &theirs[*f]);
            if same(f, o, t) || same(f, t, b) {
                false
            } else if same(f, o, b) {
                true
            } else if *f == "status" {
                let final_there = is_final(status_here(t.as_str().unwrap_or("")));
                let final_here = is_final(status_here(o.as_str().unwrap_or("")));
                if final_there != final_here { final_there } else { theirs_later }
            } else {
                theirs_later
            }
        })
        .collect()
}

/// The change that takes `take` from FormLogic's answers into a local appointment.
fn change_of(ours: &Value, theirs: &Value, take: &[&str]) -> Change {
    let from = |f: &str| if take.contains(&f) { &theirs[f] } else { &ours[f] };
    let text = |f: &str| take.contains(&f).then(|| theirs[f].as_str().unwrap_or("").to_string());
    Change {
        start: (take.contains(&"date") || take.contains(&"time")).then(|| change_from(&json!({"date": from("date"), "time": from("time")})).start).flatten(),
        status: text("status").map(|s| status_here(&s)),
        service: text("service"),
        name: text("caller_name"),
        phone: text("phone"),
        notes: text("notes"),
        ..Default::default()
    }
}

/// The key FormLogic's copy carries as `request_id`.
fn request_key(a: &Appointment) -> String {
    a.request_id.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| format!("oaiy:{}", a.id))
}

/// Changed here since it was last in step with FormLogic.
fn changed_here(a: &Appointment, r: &Remote) -> bool {
    let since = if r.synced_at.is_empty() { &a.created_at } else { &r.synced_at };
    a.updated_at != *since
}

/// FormLogic refused it as it is now.
fn refused_now(a: &Appointment, r: &Remote) -> bool {
    r.refused.as_ref().is_some_and(|x| x.of == a.updated_at)
}

/// The tombstone an appointment deleted here leaves, when FormLogic has or
/// may have a copy of it: paired, sent, or from a call (FormLogic's own flow
/// records those).
pub(super) fn tombstone_for(a: &Appointment) -> Option<Tombstone> {
    let r = remote(a);
    if r.id.is_none() && !r.sent && a.request_id.is_none() {
        return None;
    }
    Some(Tombstone { id: a.id.clone(), formlogic_id: r.id, request_key: request_key(a), deleted_at: stamp_now(), checked: false })
}

fn stamp_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ---- what the Calendar page and the Overview are told ----------------------

/// Changes made here that FormLogic has not had yet.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Pending {
    pub creates: usize,
    pub updates: usize,
    pub deletes: usize,
    pub total: usize,
}

/// A record FormLogic would not take, and why.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Problem {
    pub id: String,
    pub message: String,
}

/// How the sync stands.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub linked: bool,
    /// `unlinked`, `waiting` (not tried yet), `syncing`, `synced`, `offline`
    /// (FormLogic could not be reached) or `error` (it answered, and refused).
    pub state: &'static str,
    /// When the last sync was tried.
    pub at: Option<String>,
    /// When a sync last went through.
    pub last_success_at: Option<String>,
    pub next_attempt_at: Option<String>,
    /// What the last sync brought here, sent there and removed here.
    pub pulled: usize,
    pub pushed: usize,
    pub removed: usize,
    pub pending: Pending,
    pub error: Option<String>,
    pub problems: Vec<Problem>,
}

impl Report {
    const fn new() -> Self {
        Self {
            linked: false,
            state: "waiting",
            at: None,
            last_success_at: None,
            next_attempt_at: None,
            pulled: 0,
            pushed: 0,
            removed: 0,
            pending: Pending { creates: 0, updates: 0, deletes: 0, total: 0 },
            error: None,
            problems: Vec::new(),
        }
    }
}

/// What is waiting to go to FormLogic.
fn pending(book: &Book) -> Pending {
    let mut p = Pending::default();
    for a in &book.appointments {
        let r = remote(a);
        if refused_now(a, &r) {
            continue;
        }
        if r.id.is_none() {
            p.creates += 1;
        } else if changed_here(a, &r) {
            p.updates += 1;
        }
    }
    p.deletes = book.deleted.iter().filter(|t| t.formlogic_id.is_some() || !t.checked).count();
    p.total = p.creates + p.updates + p.deletes;
    p
}

// ---- the sync thread ---------------------------------------------------------

struct Runner {
    report: Report,
    failures: u32,
    next_at: Option<Instant>,
    next_at_wall: Option<DateTime<Utc>>,
    failed_at: Option<DateTime<Utc>>,
    nudged_at: Option<Instant>,
    asked: bool,
    runs: u64,
}

static RUNNER: Mutex<Runner> = Mutex::new(Runner { report: Report::new(), failures: 0, next_at: None, next_at_wall: None, failed_at: None, nudged_at: None, asked: false, runs: 0 });
static WAKE: Condvar = Condvar::new();
static LINK: Mutex<Option<LinkHandle>> = Mutex::new(None);
/// One sync at a time: the loop and a manual one must not both create a record.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
/// The appointments form found for a FormLogic (its address, the form).
static FORM: Mutex<Option<(String, String)>> = Mutex::new(None);

fn runner() -> std::sync::MutexGuard<'static, Runner> {
    RUNNER.lock().unwrap_or_else(|e| e.into_inner())
}

/// A change was made here: sync a moment from now (when linked).
pub fn nudge() {
    runner().nudged_at = Some(Instant::now());
    WAKE.notify_all();
}

/// How the sync stands now.
pub fn last() -> Report {
    let mut report = runner().report.clone();
    report.linked = LINK.lock().ok().and_then(|g| g.clone()).and_then(|l| l.account()).is_some();
    if !report.linked {
        report.state = "unlinked";
    }
    if let Some(cal) = super::shared() {
        if let Ok(book) = cal.book.lock() {
            report.pending = pending(&book);
            if report.last_success_at.is_none() {
                report.last_success_at = book.sync.last_success_at.clone();
            }
        }
    }
    report
}

/// Sync now, waiting a little for it: the report then (`syncing` if it is still going).
pub fn now() -> Report {
    let before = {
        let mut r = runner();
        r.asked = true;
        r.runs
    };
    WAKE.notify_all();
    let deadline = Instant::now() + ASK_WAIT;
    let mut r = runner();
    while r.runs == before {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        r = WAKE.wait_timeout(r, left).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
    }
    drop(r);
    last()
}

/// Sync while linked, on a thread of its own.
pub fn spawn(link: LinkHandle) {
    *LINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(link.clone());
    {
        // The first look is a little after startup.
        let mut r = runner();
        r.next_at = Some(Instant::now() + Duration::from_secs(20));
    }
    std::thread::Builder::new()
        .name("calendar-sync".into())
        .spawn(move || loop {
            {
                let r = runner();
                let _ = WAKE.wait_timeout(r, TICK);
            }
            let Some(cal) = super::shared() else { continue };
            let linked = link.account().is_some();
            let (asked, due) = {
                let mut r = runner();
                if !linked {
                    // Nothing to sync with: forget any backoff, so a new link syncs at once.
                    r.failures = 0;
                    r.failed_at = None;
                    r.nudged_at = None;
                    r.next_at_wall = None;
                    r.report.state = "unlinked";
                    let asked = std::mem::take(&mut r.asked);
                    if asked {
                        r.runs += 1;
                        WAKE.notify_all();
                    }
                    continue;
                }
                let now = Instant::now();
                let heard = r.failed_at.is_some_and(|failed| link.status().last_heartbeat_at.is_some_and(|beat| beat > failed));
                let due = r.nudged_at.is_some_and(|n| now >= n + SETTLE) || heard || r.next_at.map_or(true, |t| now >= t);
                (std::mem::take(&mut r.asked), due)
            };
            // Only while the phone receptionist (and so the calendar) is installed, unless asked.
            if asked || (due && super::available()) {
                run_once(cal, &link);
            }
        })
        .ok();
}

fn run_once(cal: &Calendar, link: &LinkHandle) {
    let Some(account) = link.account() else { return };
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut r = runner();
        r.nudged_at = None;
        r.report.state = "syncing";
    }
    let ctx = Ctx { now: Utc::now(), waiting: waiting_requests() };
    let result = Api::new(&account.base_url, &account.credential).and_then(|api| sync(cal, &api, &ctx));
    let mut r = runner();
    let at = Utc::now();
    r.report.at = Some(at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
    match result {
        Ok(out) => {
            r.failures = 0;
            r.failed_at = None;
            r.next_at = Some(Instant::now() + EVERY);
            r.next_at_wall = Some(at + chrono::Duration::from_std(EVERY).unwrap_or_default());
            r.report.state = "synced";
            r.report.last_success_at = r.report.at.clone();
            r.report.error = None;
            r.report.pulled = out.pulled;
            r.report.pushed = out.pushed;
            r.report.removed = out.removed;
            r.report.problems = out.problems;
        }
        Err(fail) => {
            r.failures += 1;
            r.failed_at = Some(at);
            let wait = backoff(r.failures);
            r.next_at = Some(Instant::now() + wait);
            r.next_at_wall = Some(at + chrono::Duration::from_std(wait).unwrap_or_default());
            let (state, message) = match fail {
                Failure::Offline(m) => ("offline", m),
                Failure::Refused(m) | Failure::Local(m) => ("error", m),
                Failure::FormGone => ("error", "FormLogic's appointments form has gone".to_string()),
            };
            // Said once per change of state, not every try.
            if r.report.state != state || r.report.error.as_deref() != Some(message.as_str()) {
                log::warn!("calendar sync: {message}");
            }
            r.report.state = state;
            r.report.error = Some(message);
        }
    }
    r.report.next_attempt_at = r.next_at_wall.map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
    r.runs += 1;
    WAKE.notify_all();
}

/// The wait after `failures` failed tries in a row: a minute, doubling, at most ten.
fn backoff(failures: u32) -> Duration {
    let doubled = EVERY.saturating_mul(1u32 << failures.saturating_sub(1).min(8));
    doubled.min(BACKOFF_MAX)
}

/// Request ids of calls whose event is still on its way to FormLogic.
fn waiting_requests() -> HashSet<String> {
    crate::link::outbox::waiting_request_ids()
}

// ---- FormLogic's API -----------------------------------------------------------

/// Why a sync stopped.
#[derive(Debug)]
pub(crate) enum Failure {
    /// FormLogic could not be reached, or is not answering properly: tried again later.
    Offline(String),
    /// It answered and refused (a revoked key, no appointments form).
    Refused(String),
    /// The appointments form is not there any more.
    FormGone,
    /// The calendar could not be saved here.
    Local(String),
}

pub(crate) struct Api {
    base: String,
    key: String,
    http: reqwest::blocking::Client,
}

struct Reply {
    status: u16,
    body: Value,
}

impl Api {
    pub(crate) fn new(base: &str, key: &str) -> Result<Self, Failure> {
        let http = reqwest::blocking::Client::builder()
            // Short enough that "Sync now" answers inside the page's own wait.
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| Failure::Local(e.to_string()))?;
        Ok(Self { base: base.trim_end_matches('/').to_string(), key: key.to_string(), http })
    }

    fn send(&self, method: reqwest::Method, path: &str, body: Option<&Value>, if_match: Option<&str>) -> Result<Reply, Failure> {
        let mut req = self.http.request(method, format!("{}{path}", self.base)).bearer_auth(&self.key);
        if let Some(b) = body {
            req = req.json(b);
        }
        if let Some(etag) = if_match {
            req = req.header("If-Match", format!("\"{etag}\""));
        }
        let resp = req.send().map_err(|e| Failure::Offline(crate::link::net::unreachable(&e)))?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().unwrap_or(Value::Null);
        match status {
            401 | 403 => Err(Failure::Refused(format!("FormLogic no longer accepts this desktop's key ({}): link it again", said(&body, status)))),
            408 | 429 | 500..=599 => Err(Failure::Offline(format!("FormLogic is not answering properly (HTTP {status})"))),
            _ => Ok(Reply { status, body }),
        }
    }

    fn get(&self, path: &str) -> Result<Reply, Failure> {
        self.send(reqwest::Method::GET, path, None, None)
    }

    /// The appointments form: the app-logic entry of the receptionist pack's `appointments` form.
    fn form(&self) -> Result<String, Failure> {
        if let Some((base, id)) = FORM.lock().ok().and_then(|g| g.clone()) {
            if base == self.base {
                return Ok(id);
            }
        }
        let apps = self.get("/api/v1/app-logic")?;
        if apps.status != 200 {
            return Err(Failure::Refused(format!("FormLogic would not list its apps: {}", said(&apps.body, apps.status))));
        }
        let id = find_form(&apps.body).ok_or_else(|| Failure::Refused("this FormLogic account has no appointments form (the Aokie receptionist pack)".into()))?;
        *FORM.lock().unwrap_or_else(|e| e.into_inner()) = Some((self.base.clone(), id.clone()));
        Ok(id)
    }

    fn forget_form(&self) {
        let mut g = FORM.lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref().is_some_and(|(base, _)| *base == self.base) {
            *g = None;
        }
    }
}

/// FormLogic's own words for a refusal.
fn said(body: &Value, status: u16) -> String {
    let message = body.get("message").or_else(|| body.get("error")).and_then(Value::as_str).unwrap_or("").trim();
    if message.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {message}")
    }
}

fn enc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The id of the object with `packFormId: "appointments"`, anywhere in the app-logic listing.
fn find_form(v: &Value) -> Option<String> {
    match v {
        Value::Object(o) => {
            if o.get("packFormId").and_then(Value::as_str) == Some("appointments") {
                if let Some(id) = o.get("formId").or_else(|| o.get("id")).and_then(Value::as_str) {
                    return Some(id.to_string());
                }
            }
            o.values().find_map(find_form)
        }
        Value::Array(a) => a.iter().find_map(find_form),
        _ => None,
    }
}

/// FormLogic's time (`YYYY-MM-DD HH:MM:SS`, UTC) or an RFC 3339 one.
fn when(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s).map(|t| t.with_timezone(&Utc)).ok().or_else(|| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok().map(|t| t.and_utc()))
}

fn status_there(s: Status) -> &'static str {
    match s {
        Status::Requested => "requested",
        Status::Confirmed => "confirmed",
        Status::Done => "completed",
        Status::Cancelled | Status::Declined => "cancelled",
    }
}

fn status_here(s: &str) -> Status {
    match s {
        "confirmed" => Status::Confirmed,
        "completed" => Status::Done,
        "cancelled" | "no-show" => Status::Cancelled,
        _ => Status::Requested,
    }
}

fn source_there(s: &str) -> &'static str {
    match s {
        "call" => "call",
        "text" => "sms",
        "agent" => "flow",
        _ => "manual",
    }
}

fn source_here(s: &str) -> &'static str {
    match s {
        "call" => "call",
        "sms" => "text",
        _ => "formlogic",
    }
}

/// An appointment as FormLogic's answers.
pub fn answers(a: &Appointment) -> Value {
    let (date, time) = a.start.split_once('T').unwrap_or((a.start.as_str(), ""));
    json!({
        "caller_name": a.name,
        // The form needs a service; a call may not have named one.
        "service": if a.service.trim().is_empty() { "Appointment" } else { a.service.as_str() },
        "date": date,
        "time": time,
        "status": status_there(a.status),
        "phone": a.phone,
        "notes": a.notes,
        "source": source_there(&a.source),
        "call_id": a.call_id.clone().unwrap_or_default(),
        "request_id": request_key(a),
    })
}

/// `answers`, keeping FormLogic's word for a status that means the same here
/// ("no-show" stays "no-show" while it is cancelled here).
fn answers_over(a: &Appointment, base: Option<&Value>) -> Value {
    let mut v = answers(a);
    if let Some(theirs) = base.and_then(|b| b.get("status")).and_then(Value::as_str) {
        if status_here(theirs) == a.status || (a.status == Status::Declined && status_here(theirs) == Status::Cancelled) {
            v["status"] = json!(theirs);
        }
    }
    v
}

/// What a FormLogic record says, as a change of the local appointment.
fn change_from(answers: &Value) -> Change {
    let text = |k: &str| answers.get(k).and_then(Value::as_str).map(str::to_string);
    let start = match (text("date"), text("time")) {
        (Some(d), Some(t)) if !d.is_empty() && !t.is_empty() => Some(format!("{d}T{}", &t[..t.len().min(5)])),
        (Some(d), _) if !d.is_empty() => Some(format!("{d}T09:00")),
        _ => None,
    };
    Change {
        status: text("status").map(|s| status_here(&s)),
        start,
        service: text("service"),
        name: text("caller_name"),
        phone: text("phone"),
        notes: text("notes"),
        ..Default::default()
    }
}

fn is_final(s: Status) -> bool {
    matches!(s, Status::Cancelled | Status::Done | Status::Declined)
}

/// A FormLogic record's id and version.
struct Stamp {
    id: String,
    updated_at: String,
    etag: Option<String>,
}

impl Stamp {
    fn of(r: &Value) -> Option<Stamp> {
        let text = |k: &str| r.get(k).and_then(Value::as_str).map(str::to_string);
        Some(Stamp { id: text("id")?, updated_at: text("updatedAt").unwrap_or_default(), etag: text("etag") })
    }
}

fn text_of(answers: &Value, key: &str) -> Option<String> {
    answers.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

// ---- one sync --------------------------------------------------------------------

/// What a sync is told about the moment it runs.
pub(crate) struct Ctx {
    pub now: DateTime<Utc>,
    /// Request ids of calls whose event is still on its way to FormLogic.
    pub waiting: HashSet<String>,
}

#[derive(Default, Debug)]
pub(crate) struct Outcome {
    pub pulled: usize,
    pub pushed: usize,
    pub removed: usize,
    pub problems: Vec<Problem>,
}

/// One sync: pull and merge, then push. Stops at the first sign FormLogic
/// cannot be reached, keeping what it had done.
pub(crate) fn sync(cal: &Calendar, api: &Api, ctx: &Ctx) -> Result<Outcome, Failure> {
    let mut out = Outcome::default();
    let form = match pull(cal, api, &mut out) {
        Err(Failure::FormGone) => {
            // The pack was installed again, with a new form: find it.
            api.forget_form();
            pull(cal, api, &mut out)?
        }
        other => other?,
    };
    push(cal, api, &form, ctx, &mut out)?;
    let stamp = ctx.now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    cal.with(|b| {
        b.sync.last_success_at = Some(stamp.clone());
        Ok(())
    })
    .map_err(Failure::Local)?;
    Ok(out)
}

fn save<T>(cal: &Calendar, change: impl FnOnce(&mut Book) -> Result<T, String>) -> Result<T, Failure> {
    cal.with(change).map_err(Failure::Local)
}

/// What a pull brought.
#[derive(Default)]
struct Pulled {
    records: Vec<Value>,
    deleted: Vec<String>,
    /// FormLogic's time as the pull began, when it says it.
    server_time: Option<String>,
    /// Every record FormLogic has (not only the changed ones).
    full: bool,
}

/// Pull, and merge what came: the form's id.
fn pull(cal: &Calendar, api: &Api, out: &mut Outcome) -> Result<String, Failure> {
    let form = api.form()?;
    let cursor = save(cal, |b| {
        if b.sync.form.as_deref() != Some(form.as_str()) {
            if b.sync.form.is_some() {
                start_over(b);
            }
            b.sync.form = Some(form.clone());
            b.sync.cursor = None;
        }
        Ok(b.sync.cursor.clone())
    })?;
    // What changed and every deletion since the last sync, when FormLogic can
    // say; else everything, compared whole.
    let changed = match cursor.as_deref() {
        Some(c) => list(api, &form, Some(c))?,
        None => None,
    };
    let mut pulled = match changed {
        Some(p) => p,
        None => list(api, &form, None)?.unwrap_or_default(),
    };
    if pulled.full {
        look_up_missing(cal, api, &form, &mut pulled)?;
    }
    save(cal, |b| {
        apply(b, &pulled, out);
        if let Some(t) = &pulled.server_time {
            b.sync.cursor = Some(t.clone());
        }
        Ok(())
    })?;
    Ok(form)
}

/// A different form than before (the pack was installed again): the pairings
/// with the old one mean nothing, so every appointment is matched afresh.
fn start_over(b: &mut Book) {
    for a in &mut b.appointments {
        let base = remote(a).base;
        set_remote(a, Remote { base, ..Default::default() });
    }
    for t in &mut b.deleted {
        t.formlogic_id = None;
        t.checked = false;
    }
    b.sync.discard.clear();
}

/// The form's records: changed since `cursor`, or all (`cursor` None). None
/// when the changes since `cursor` cannot be told whole (a FormLogic without
/// `updatedSince`, or deletions older than it keeps).
fn list(api: &Api, form: &str, cursor: Option<&str>) -> Result<Option<Pulled>, Failure> {
    let path = format!("/api/v1/forms/{form}/responses");
    let since = cursor
        .and_then(when)
        .map(|t| (t - OVERLAP).format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| EPOCH.to_string());
    let first = page(api, &format!("{path}?updatedSince={}&limit={PAGE}", enc(&since)))?;
    let Some(server_time) = first.get("serverTime").and_then(Value::as_str).map(str::to_string) else {
        // A FormLogic from before updatedSince: it listed its newest records. Read them all.
        if cursor.is_some() {
            return Ok(None);
        }
        let mut records = responses(&first);
        let mut n = records.len();
        let mut pages = 1;
        while n == PAGE && pages < MAX_PAGES {
            let next = page(api, &format!("{path}?limit={PAGE}&offset={}", pages * PAGE))?;
            let more = responses(&next);
            n = more.len();
            records.extend(more);
            pages += 1;
        }
        return Ok(Some(Pulled { records, deleted: Vec::new(), server_time: None, full: true }));
    };
    let deleted_since = first.get("deletedSince").and_then(Value::as_str).unwrap_or("").to_string();
    let mut whole = first.get("deletedComplete").and_then(Value::as_bool).unwrap_or(false) && !deleted_since.is_empty() && when(&deleted_since) <= when(&since);
    let mut records = responses(&first);
    let mut deleted = deleted_ids(&first);
    let mut n = records.len();
    let mut pages = 1;
    while n == PAGE && pages < MAX_PAGES {
        let (at, id) = match records.last() {
            Some(r) => (r.get("updatedAt").and_then(Value::as_str).unwrap_or("").to_string(), r.get("id").and_then(Value::as_str).unwrap_or("").to_string()),
            None => break,
        };
        let next = page(api, &format!("{path}?updatedSince={}&afterId={}&limit={PAGE}", enc(&at), enc(&id)))?;
        let more = responses(&next);
        n = more.len();
        records.extend(more);
        deleted.extend(deleted_ids(&next));
        whole &= next.get("deletedComplete").and_then(Value::as_bool).unwrap_or(false);
        pages += 1;
    }
    if cursor.is_some() && !whole {
        return Ok(None);
    }
    // A listing cut short is not whole: the next sync asks from the same place.
    let server_time = (n < PAGE).then_some(server_time);
    Ok(Some(Pulled { records, deleted, server_time, full: cursor.is_none() }))
}

fn page(api: &Api, path: &str) -> Result<Value, Failure> {
    let reply = api.get(path)?;
    match reply.status {
        200 => Ok(reply.body),
        404 => Err(Failure::FormGone),
        s => Err(Failure::Refused(format!("FormLogic would not list the appointments: {}", said(&reply.body, s)))),
    }
}

fn responses(v: &Value) -> Vec<Value> {
    v.get("responses").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn deleted_ids(v: &Value) -> Vec<String> {
    v.get("deleted")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|d| d.get("id").and_then(Value::as_str).map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Local appointments paired with a record a full listing did not have are
/// looked up: gone only if FormLogic says so, as a listing read in pages can
/// miss one that moved while it was read.
fn look_up_missing(cal: &Calendar, api: &Api, form: &str, pulled: &mut Pulled) -> Result<(), Failure> {
    let listed: HashSet<String> = pulled.records.iter().filter_map(|r| r.get("id").and_then(Value::as_str).map(str::to_string)).chain(pulled.deleted.iter().cloned()).collect();
    let missing: Vec<String> = cal.list(None, None).iter().filter_map(|a| remote(a).id).filter(|id| !listed.contains(id)).take(MAX_LOOKUPS).collect();
    for rid in missing {
        let reply = api.get(&format!("/api/v1/forms/{form}/responses/{}", enc(&rid)))?;
        match reply.status {
            404 => pulled.deleted.push(rid),
            200 => {
                if let Some(r) = reply.body.get("response") {
                    pulled.records.push(r.clone());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Merge a pull into the book.
fn apply(b: &mut Book, pulled: &Pulled, out: &mut Outcome) {
    // Deleted in FormLogic: deleted here, whatever changed here meanwhile.
    for rid in &pulled.deleted {
        if let Some(i) = b.appointments.iter().position(|a| remote(a).id.as_deref() == Some(rid.as_str())) {
            b.appointments.remove(i);
            out.removed += 1;
        }
        b.deleted.retain(|t| t.formlogic_id.as_deref() != Some(rid.as_str()) || !t.request_key.starts_with("oaiy:"));
        for t in &mut b.deleted {
            if t.formlogic_id.as_deref() == Some(rid.as_str()) {
                t.formlogic_id = None;
                t.checked = true;
            }
        }
        b.sync.discard.retain(|d| d != rid);
    }
    for r in &pulled.records {
        let Some(stamp) = Stamp::of(r) else { continue };
        let rid = stamp.id.clone();
        if pulled.deleted.contains(&rid) || b.sync.discard.contains(&rid) {
            continue;
        }
        let answers = r.get("answers").cloned().unwrap_or(Value::Null);
        let key = text_of(&answers, "request_id");
        // Deleted here: FormLogic's copy is deleted by the push, not brought back.
        if b.deleted.iter().any(|t| t.formlogic_id.as_deref() == Some(rid.as_str())) {
            continue;
        }
        if let Some(k) = &key {
            if b.deleted.iter().any(|t| t.request_key == *k) {
                b.sync.discard.push(rid);
                continue;
            }
        }
        let by_id = b.appointments.iter().position(|a| remote(a).id.as_deref() == Some(rid.as_str()));
        let found = by_id.or_else(|| key.as_ref().and_then(|k| b.appointments.iter().position(|a| request_key(a) == *k)));
        let Some(i) = found else {
            insert(b, &stamp, &answers, out);
            continue;
        };
        let mine = remote(&b.appointments[i]);
        match mine.id.as_deref() {
            Some(r0) if r0 != rid => {
                if mine.created {
                    // Made here and by FormLogic's flow for the same call:
                    // FormLogic's is kept (its texts refer to it), this one's goes.
                    b.sync.discard.push(r0.to_string());
                    reconcile(b, i, &stamp, &answers, out);
                } else {
                    // FormLogic has two of its own; so does this calendar.
                    insert(b, &stamp, &answers, out);
                }
            }
            _ => reconcile(b, i, &stamp, &answers, out),
        }
    }
}

/// A record new to this desktop.
fn insert(b: &mut Book, stamp: &Stamp, answers: &Value, out: &mut Outcome) {
    let c = change_from(answers);
    let Some(start) = c.start.clone() else { return };
    let made = b.create(NewAppointment {
        service: c.service.clone().unwrap_or_default(),
        start,
        status: c.status,
        name: c.name.clone().unwrap_or_default(),
        phone: c.phone.clone().unwrap_or_default(),
        notes: c.notes.clone().unwrap_or_default(),
        source: source_here(answers.get("source").and_then(Value::as_str).unwrap_or("")).to_string(),
        ..Default::default()
    });
    let Ok(made) = made else { return };
    if let Some(a) = b.appointments.iter_mut().find(|a| a.id == made.id) {
        a.request_id = text_of(answers, "request_id");
        a.call_id = text_of(answers, "call_id");
        let synced_at = a.updated_at.clone();
        set_remote(a, Remote { id: Some(stamp.id.clone()), updated_at: stamp.updated_at.clone(), etag: stamp.etag.clone(), synced_at, base: Some(answers.clone()), ..Default::default() });
    }
    out.pulled += 1;
}

/// Bring a local appointment and its FormLogic record into step, as far as
/// the pull can: FormLogic's side is taken here now; this side is left for the
/// push, which writes it over exactly the version seen here.
fn reconcile(b: &mut Book, i: usize, stamp: &Stamp, theirs: &Value, out: &mut Outcome) {
    let a = &b.appointments[i];
    let mine = remote(a);
    let paired = mine.id.as_deref() == Some(stamp.id.as_str());
    let theirs_changed = !paired
        || match (&mine.etag, &stamp.etag) {
            (Some(x), Some(y)) => x != y,
            _ => mine.updated_at != stamp.updated_at,
        };
    if !theirs_changed {
        // Nothing new there; a change made here is sent by the push. A record
        // this desktop made is answered without its etag: it is learnt here,
        // so the push can say which version it writes over.
        if mine.etag.is_none() && stamp.etag.is_some() {
            let mut r = mine;
            r.etag = stamp.etag.clone();
            set_remote(&mut b.appointments[i], r);
        }
        return;
    }
    let ours_changed = changed_here(a, &mine);
    let ours = answers_over(a, mine.base.as_ref());
    let take: Vec<&'static str> = if !ours_changed {
        FIELDS.to_vec()
    } else {
        let theirs_later = when(&stamp.updated_at) > when(&a.updated_at);
        match &mine.base {
            Some(base) => merge(base, &ours, theirs, theirs_later),
            // Nothing to tell whose change is whose: one side's record wins,
            // a final status first, else the later.
            None => {
                let final_there = is_final(change_from(theirs).status.unwrap_or(Status::Requested));
                let take_all = if final_there != is_final(a.status) { final_there } else { theirs_later };
                if take_all { FIELDS.to_vec() } else { Vec::new() }
            }
        }
    };
    let id = a.id.clone();
    let created = paired && mine.created;
    if !take.is_empty() && b.change(&id, change_of(&ours, theirs, &take)).is_ok() {
        out.pulled += 1;
    }
    let a = &mut b.appointments[i];
    if a.request_id.is_none() && !text_of(theirs, "request_id").is_some_and(|k| k.starts_with("oaiy:")) {
        a.request_id = text_of(theirs, "request_id");
    }
    if a.call_id.is_none() {
        a.call_id = text_of(theirs, "call_id");
    }
    let now = answers_over(a, Some(theirs));
    let agree = FIELDS.iter().all(|f| same(f, &now[*f], &theirs[*f]));
    // Taken from FormLogic alone, or agreeing now: in step. Otherwise still
    // changed here, for the push to write over exactly the version just seen.
    let synced_at = if !ours_changed || agree { a.updated_at.clone() } else { mine.synced_at.clone() };
    set_remote(a, Remote {
        id: Some(stamp.id.clone()),
        updated_at: stamp.updated_at.clone(),
        etag: stamp.etag.clone(),
        synced_at,
        created,
        base: Some(theirs.clone()),
        refused: if agree { None } else { mine.refused.clone() },
        ..Default::default()
    });
}

/// Send what changed here: changes, new appointments, deletions.
fn push(cal: &Calendar, api: &Api, form: &str, ctx: &Ctx, out: &mut Outcome) -> Result<(), Failure> {
    let path = format!("/api/v1/forms/{form}/responses");
    for a in cal.list(None, None) {
        let r = remote(&a);
        if r.id.is_some() && changed_here(&a, &r) && !refused_now(&a, &r) {
            put_one(cal, api, &path, &a, out)?;
        }
    }
    for a in cal.list(None, None) {
        let r = remote(&a);
        if r.id.is_some() || refused_now(&a, &r) {
            continue;
        }
        let key = request_key(&a);
        if a.source == "call" && (when(&a.created_at).is_some_and(|t| ctx.now - t < CALL_GRACE) || ctx.waiting.contains(&key)) {
            continue;
        }
        create_one(cal, api, &path, &a, out)?;
    }
    delete_tombstoned(cal, api, &path, ctx, out)
}

/// Send a change of a record FormLogic has, over the version last seen.
fn put_one(cal: &Calendar, api: &Api, path: &str, a: &Appointment, out: &mut Outcome) -> Result<(), Failure> {
    let mut a = a.clone();
    // Twice at most: once over the version last seen, once more over the one
    // FormLogic says is there now, if this side still wins the merge.
    for _ in 0..2 {
        let r = remote(&a);
        let Some(rid) = r.id.clone() else { return Ok(()) };
        let body = answers_over(&a, r.base.as_ref());
        let reply = api.send(reqwest::Method::PUT, &format!("{path}/{}", enc(&rid)), Some(&json!({ "answers": body })), r.etag.as_deref())?;
        match reply.status {
            200..=299 => {
                let sent = a.updated_at.clone();
                let got = reply.body.get("response");
                let stamp = got.and_then(Stamp::of);
                let agreed = got.and_then(|g| g.get("answers")).filter(|v| v.is_object()).cloned().unwrap_or(body);
                save(cal, |b| {
                    if let Some(x) = b.appointments.iter_mut().find(|x| x.id == a.id) {
                        let mut now = remote(x);
                        if let Some(s) = &stamp {
                            now.updated_at = s.updated_at.clone();
                            now.etag = s.etag.clone();
                        }
                        // In step with what was sent; a change made since is sent next time.
                        now.synced_at = sent;
                        now.refused = None;
                        now.base = Some(agreed);
                        set_remote(x, now);
                    }
                    Ok(())
                })?;
                out.pushed += 1;
                return Ok(());
            }
            412 => {
                // Changed in FormLogic since it was last seen: merge, then maybe send again.
                let Some(current) = reply.body.get("response") else { return Ok(()) };
                let Some(stamp) = Stamp::of(current) else { return Ok(()) };
                let theirs = current.get("answers").cloned().unwrap_or(Value::Null);
                let again = save(cal, |b| {
                    if let Some(i) = b.appointments.iter().position(|x| x.id == a.id) {
                        reconcile(b, i, &stamp, &theirs, out);
                        let x = &b.appointments[i];
                        let r = remote(x);
                        return Ok(changed_here(x, &r).then(|| x.clone()));
                    }
                    Ok(None)
                })?;
                match again {
                    Some(x) => a = x,
                    None => return Ok(()),
                }
            }
            404 => {
                // Deleted in FormLogic: deleted here too.
                save(cal, |b| {
                    b.appointments.retain(|x| x.id != a.id);
                    Ok(())
                })?;
                out.removed += 1;
                return Ok(());
            }
            s => return refuse(cal, &a, said(&reply.body, s), out),
        }
    }
    Ok(())
}

/// Send a new appointment, unless FormLogic already has it.
fn create_one(cal: &Calendar, api: &Api, path: &str, a: &Appointment, out: &mut Outcome) -> Result<(), Failure> {
    let key = request_key(a);
    if let Some(found) = find_by_key(cal, api, path, &key)? {
        let Some(stamp) = Stamp::of(&found) else { return Ok(()) };
        let theirs = found.get("answers").cloned().unwrap_or(Value::Null);
        return save(cal, |b| {
            if let Some(i) = b.appointments.iter().position(|x| x.id == a.id) {
                reconcile(b, i, &stamp, &theirs, out);
            }
            Ok(())
        });
    }
    // Marked before it goes: if the answer is lost, FormLogic may have it, and
    // deleting it here must still reach FormLogic.
    save(cal, |b| {
        if let Some(x) = b.appointments.iter_mut().find(|x| x.id == a.id) {
            let mut r = remote(x);
            r.sent = true;
            set_remote(x, r);
        }
        Ok(())
    })?;
    let body = answers(a);
    // Keyed by what is sent, so a retry of the same answers is the same record,
    // and changed answers are not refused as a clash.
    let key_hash = {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(body.to_string().as_bytes());
        digest.iter().take(6).map(|b| format!("{b:02x}")).collect::<String>()
    };
    let reply = api.send(reqwest::Method::POST, path, Some(&json!({ "answers": body.clone(), "idempotencyKey": format!("oaiy:appt:{}:{key_hash}", a.id) })), None)?;
    match reply.status {
        200 | 201 => {
            let rec = reply.body.get("response").unwrap_or(&reply.body);
            let Some(stamp) = Stamp::of(rec) else {
                return refuse(cal, a, "FormLogic made the record but gave no id".into(), out);
            };
            let sent = a.updated_at.clone();
            save(cal, |b| {
                if let Some(x) = b.appointments.iter_mut().find(|x| x.id == a.id) {
                    set_remote(x, Remote { id: Some(stamp.id.clone()), updated_at: stamp.updated_at.clone(), etag: stamp.etag.clone(), synced_at: sent, created: true, base: Some(body.clone()), ..Default::default() });
                }
                Ok(())
            })?;
            out.pushed += 1;
            Ok(())
        }
        // Still being taken from an earlier try: the next sync finds it.
        409 => Ok(()),
        s => refuse(cal, a, said(&reply.body, s), out),
    }
}

/// FormLogic's record carrying `request_id` = `key`, if it has one (the oldest).
fn find_by_key(cal: &Calendar, api: &Api, path: &str, key: &str) -> Result<Option<Value>, Failure> {
    let reply = api.get(&format!("{path}?answers.request_id={}&limit=10", enc(key)))?;
    if reply.status != 200 {
        // A form that cannot be searched by answers: create without looking.
        return Ok(None);
    }
    let (discard, gone): (Vec<String>, Vec<String>) = cal
        .book
        .lock()
        .map(|b| (b.sync.discard.clone(), b.deleted.iter().filter_map(|t| t.formlogic_id.clone()).collect()))
        .unwrap_or_default();
    Ok(responses(&reply.body)
        .into_iter()
        .filter(|r| text_of(r.get("answers").unwrap_or(&Value::Null), "request_id").as_deref() == Some(key))
        .filter(|r| r.get("id").and_then(Value::as_str).is_some_and(|id| !discard.iter().any(|d| d == id) && !gone.iter().any(|g| g == id)))
        .last())
}

/// FormLogic would not take it as it is: set aside, with why, until it changes here.
fn refuse(cal: &Calendar, a: &Appointment, message: String, out: &mut Outcome) -> Result<(), Failure> {
    let of = a.updated_at.clone();
    save(cal, |b| {
        if let Some(x) = b.appointments.iter_mut().find(|x| x.id == a.id) {
            let mut r = remote(x);
            r.refused = Some(Refused { message: message.clone(), of });
            set_remote(x, r);
        }
        Ok(())
    })?;
    out.problems.push(Problem { id: a.id.clone(), message });
    Ok(())
}

/// Delete in FormLogic what was deleted here, and this desktop's extra copies.
fn delete_tombstoned(cal: &Calendar, api: &Api, path: &str, ctx: &Ctx, out: &mut Outcome) -> Result<(), Failure> {
    let (tombstones, discard) = cal.book.lock().map(|b| (b.deleted.clone(), b.sync.discard.clone())).unwrap_or_default();
    for t in tombstones {
        let mut targets: Vec<String> = t.formlogic_id.clone().into_iter().collect();
        if !t.checked {
            // Never paired, or FormLogic's own copy of a call's request: look for it.
            if let Some(found) = find_by_key(cal, api, path, &t.request_key)? {
                if let Some(id) = found.get("id").and_then(Value::as_str) {
                    if !targets.iter().any(|x| x == id) {
                        targets.push(id.to_string());
                    }
                }
            }
        }
        let mut done = true;
        for rid in &targets {
            let reply = api.send(reqwest::Method::DELETE, &format!("{path}/{}", enc(rid)), None, None)?;
            match reply.status {
                200..=299 | 404 => {}
                s => {
                    done = false;
                    out.problems.push(Problem { id: t.id.clone(), message: format!("FormLogic would not delete it: {}", said(&reply.body, s)) });
                }
            }
        }
        if done {
            save(cal, |b| {
                if t.request_key.starts_with("oaiy:") {
                    // Made here: nothing else will make a copy of it.
                    b.deleted.retain(|x| x.id != t.id);
                } else if let Some(x) = b.deleted.iter_mut().find(|x| x.id == t.id) {
                    // A call's request: kept a while, as its event may come again.
                    x.formlogic_id = None;
                    x.checked = true;
                }
                Ok(())
            })?;
        }
    }
    for rid in discard {
        let reply = api.send(reqwest::Method::DELETE, &format!("{path}/{}", enc(&rid)), None, None)?;
        if matches!(reply.status, 200..=299 | 404) {
            save(cal, |b| {
                b.sync.discard.retain(|d| *d != rid);
                Ok(())
            })?;
            out.removed += 1;
        }
    }
    // Tombstones of calls' requests go after a while.
    let cutoff = ctx.now - KEEP_TOMBSTONES;
    save(cal, |b| {
        b.deleted.retain(|t| !(t.checked && t.formlogic_id.is_none() && when(&t.deleted_at).is_some_and(|d| d < cutoff)));
        Ok(())
    })
}

#[cfg(test)]
mod tests;
