//! The calendar syncs with FormLogic while the desktop is linked to it.
//!
//! FormLogic keeps appointments as records of the Aokie receptionist pack's
//! `appointments` form (answers `caller_name`, `service`, `date`, `time`,
//! `status`, `phone`, `call_id`, `request_id`, `source`, `notes`), reached with
//! the link's key: `GET /api/v1/app-logic` names the form, `GET/POST/PUT
//! /api/v1/forms/{form}/responses[/{id}]` read and write its records. Hours,
//! services and lengths are the desktop's alone (FormLogic has none).
//!
//! Every minute, and when asked (`POST /api/calendar/sync`):
//! - each FormLogic record is paired with a local appointment by its id, or by
//!   the call's request id (both sides record a call's request: Aokie tells the
//!   desktop, and FormLogic's own flow writes it);
//! - what changed since the last sync goes the way it changed: a FormLogic
//!   change comes here, a local one goes there; when both changed, a final
//!   status (cancelled, done) wins, else the later change;
//! - a local appointment FormLogic has not seen is added there. One from a
//!   call waits five minutes first: FormLogic's flow makes its own record of
//!   it (and texts the caller), and the next pull pairs the two.
//!
//! A local appointment remembers its FormLogic copy as `formlogic: {id,
//! updatedAt, syncedAt}`: FormLogic's `updatedAt` then, and the local
//! `updatedAt` it was in step with.

use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde_json::{json, Value};

use super::{Appointment, Calendar, Change, NewAppointment, Status};
use crate::link::LinkHandle;

const EVERY: Duration = Duration::from_secs(60);
/// How long a call's request is left for FormLogic's own flow to record.
const CALL_GRACE: chrono::Duration = chrono::Duration::minutes(5);

/// How the last sync went, for the Calendar page.
#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub linked: bool,
    pub at: Option<String>,
    pub pulled: usize,
    pub pushed: usize,
    pub error: Option<String>,
}

static LAST: Mutex<Option<Report>> = Mutex::new(None);
static LINK: Mutex<Option<LinkHandle>> = Mutex::new(None);
/// The form's id, once found.
static FORM: Mutex<Option<String>> = Mutex::new(None);

pub fn last() -> Report {
    LAST.lock().ok().and_then(|g| g.clone()).unwrap_or_default()
}

/// Sync every minute while linked (on a thread of its own).
pub fn spawn(link: LinkHandle) {
    *LINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(link);
    std::thread::Builder::new()
        .name("calendar-sync".into())
        .spawn(|| loop {
            std::thread::sleep(Duration::from_secs(20));
            // Only while the phone receptionist (and so the calendar) is installed.
            if super::available() {
                let _ = now();
            }
            std::thread::sleep(EVERY.saturating_sub(Duration::from_secs(20)));
        })
        .ok();
}

/// Sync now: the report of this run.
pub fn now() -> Report {
    let report = match (super::shared(), LINK.lock().ok().and_then(|g| g.clone()).and_then(|l| l.account())) {
        (None, _) => Report { error: Some("the calendar is not open".into()), ..Default::default() },
        (_, None) => Report::default(),
        (Some(cal), Some(account)) => {
            let api = Api { base: account.base_url.trim_end_matches('/').to_string(), key: account.credential.clone() };
            match sync(cal, &api) {
                Ok((pulled, pushed)) => Report { linked: true, at: Some(Utc::now().to_rfc3339()), pulled, pushed, error: None },
                Err(e) => {
                    log::warn!("calendar sync: {e}");
                    Report { linked: true, at: Some(Utc::now().to_rfc3339()), error: Some(e), ..Default::default() }
                }
            }
        }
    };
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(report.clone());
    report
}

struct Api {
    base: String,
    key: String,
}

impl Api {
    fn client() -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| e.to_string())
    }

    fn send(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value, String> {
        let mut req = Self::client()?.request(method.clone(), format!("{}{path}", self.base)).bearer_auth(&self.key);
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().map_err(|e| format!("FormLogic did not answer: {e}"))?;
        let status = resp.status();
        let value: Value = resp.json().unwrap_or(Value::Null);
        if !status.is_success() {
            let message = value.get("message").and_then(Value::as_str).unwrap_or("");
            return Err(format!("FormLogic refused {method} {path}: HTTP {} {message}", status.as_u16()));
        }
        Ok(value)
    }

    /// The appointments form: the app-logic entry of the receptionist pack's `appointments` form.
    fn form(&self) -> Result<String, String> {
        if let Some(id) = FORM.lock().ok().and_then(|g| g.clone()) {
            return Ok(id);
        }
        let apps = self.send(reqwest::Method::GET, "/api/v1/app-logic", None)?;
        let id = find_form(&apps).ok_or("this FormLogic account has no appointments form (the Aokie receptionist pack)")?;
        *FORM.lock().unwrap_or_else(|e| e.into_inner()) = Some(id.clone());
        Ok(id)
    }
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
        "service": a.service,
        "date": date,
        "time": time,
        "status": status_there(a.status),
        "phone": a.phone,
        "notes": a.notes,
        "source": source_there(&a.source),
        "call_id": a.call_id.clone().unwrap_or_default(),
        "request_id": a.request_id.clone().unwrap_or_default(),
    })
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

/// One sync: (records brought here, records sent there).
fn sync(cal: &Calendar, api: &Api) -> Result<(usize, usize), String> {
    let form = api.form()?;
    let page = api.send(reqwest::Method::GET, &format!("/api/v1/forms/{form}/responses?limit=1000"), None)?;
    let records = page.get("responses").and_then(Value::as_array).cloned().unwrap_or_default();
    let (mut pulled, mut pushed) = (0, 0);
    let mut seen = std::collections::HashSet::new();

    for r in &records {
        let Some(rid) = r.get("id").and_then(Value::as_str) else { continue };
        seen.insert(rid.to_string());
        let answers = r.get("answers").cloned().unwrap_or(Value::Null);
        let remote_at = r.get("updatedAt").and_then(Value::as_str).unwrap_or("").to_string();
        let request = answers.get("request_id").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let local = cal.list(None, None).into_iter().find(|a| {
            a.formlogic.as_ref().and_then(|f| f.get("id")).and_then(Value::as_str) == Some(rid) || (request.is_some() && a.request_id == request)
        });
        let Some(local) = local else {
            // New to this desktop: made here, in step with FormLogic.
            let change = change_from(&answers);
            let Some(start) = change.start.clone() else { continue };
            let made = cal.create(NewAppointment {
                service: change.service.clone().unwrap_or_default(),
                start,
                status: change.status,
                name: change.name.clone().unwrap_or_default(),
                phone: change.phone.clone().unwrap_or_default(),
                notes: change.notes.clone().unwrap_or_default(),
                source: source_here(answers.get("source").and_then(Value::as_str).unwrap_or("")).to_string(),
                ..Default::default()
            })?;
            let text = |k: &str| answers.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
            cal.mark_synced(&made.id, rid, &remote_at, text("request_id"), text("call_id"))?;
            pulled += 1;
            continue;
        };
        let synced = local.formlogic.as_ref();
        let remote_changed = synced.and_then(|f| f.get("updatedAt")).and_then(Value::as_str) != Some(remote_at.as_str());
        // Never synced: a local change is one made after it was first recorded.
        let local_since = synced.and_then(|f| f.get("syncedAt")).and_then(Value::as_str).unwrap_or(&local.created_at);
        let local_changed = local.updated_at != local_since;
        let take_remote = match (remote_changed, local_changed) {
            (true, false) => true,
            (false, true) => false,
            (false, false) => {
                if synced.is_none() {
                    cal.mark_synced(&local.id, rid, &remote_at, None, None)?;
                }
                continue;
            }
            (true, true) => {
                let theirs = change_from(&answers).status.unwrap_or(Status::Requested);
                if is_final(theirs) != is_final(local.status) {
                    is_final(theirs)
                } else {
                    when(&remote_at) > when(&local.updated_at)
                }
            }
        };
        if take_remote {
            cal.update(&local.id, change_from(&answers))?;
            cal.mark_synced(&local.id, rid, &remote_at, None, None)?;
            pulled += 1;
        } else {
            let put = api.send(reqwest::Method::PUT, &format!("/api/v1/forms/{form}/responses/{rid}"), Some(json!({ "answers": answers_of(cal, &local.id)? })))?;
            let at = updated_at(&put).unwrap_or(remote_at);
            cal.mark_synced(&local.id, rid, &at, None, None)?;
            pushed += 1;
        }
    }

    // Local appointments FormLogic has not seen.
    let now = Utc::now();
    for a in cal.list(None, None) {
        if a.formlogic.is_some() || a.source == "formlogic" {
            continue;
        }
        if a.source == "call" && when(&a.created_at).is_some_and(|t| now - t < CALL_GRACE) {
            continue;
        }
        let made = api.send(
            reqwest::Method::POST,
            &format!("/api/v1/forms/{form}/responses"),
            Some(json!({ "answers": answers(&a), "idempotencyKey": format!("oaiy:appt:{}:v1", a.id) })),
        )?;
        let rid = made.get("id").or_else(|| made.get("response").and_then(|r| r.get("id"))).and_then(Value::as_str).ok_or("FormLogic made the record but gave no id")?.to_string();
        let at = updated_at(&made).unwrap_or_default();
        cal.mark_synced(&a.id, &rid, &at, None, None)?;
        pushed += 1;
    }
    Ok((pulled, pushed))
}

fn answers_of(cal: &Calendar, id: &str) -> Result<Value, String> {
    cal.get(id).map(|a| answers(&a)).ok_or_else(|| format!("no appointment {id}"))
}

fn updated_at(v: &Value) -> Option<String> {
    v.get("updatedAt").or_else(|| v.get("response").and_then(|r| r.get("updatedAt"))).and_then(Value::as_str).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_appointments_form_is_found_in_the_listing() {
        let apps = json!({"apps": [{"id": "x", "forms": [{"packFormId": "customers", "formId": "c1"}, {"packFormId": "appointments", "formId": "cd1d"}]}]});
        assert_eq!(find_form(&apps).as_deref(), Some("cd1d"));
        assert_eq!(find_form(&json!({"apps": []})), None);
    }

    #[test]
    fn statuses_and_sources_go_both_ways() {
        for s in [Status::Requested, Status::Confirmed, Status::Done, Status::Cancelled] {
            assert_eq!(status_here(status_there(s)), s);
        }
        assert_eq!(status_there(Status::Declined), "cancelled");
        assert_eq!(status_here("no-show"), Status::Cancelled);
        assert_eq!((source_there("text"), source_here("sms")), ("sms", "text"));
    }

    #[test]
    fn a_record_becomes_a_change_and_an_appointment_its_answers() {
        let c = change_from(&json!({"caller_name": "Lanes", "service": "Lawnmowing", "date": "2026-10-01", "time": "10:00", "status": "requested", "phone": "0491570006"}));
        assert_eq!((c.start.as_deref(), c.status, c.name.as_deref()), (Some("2026-10-01T10:00"), Some(Status::Requested), Some("Lanes")));
        assert_eq!(change_from(&json!({"date": "2026-10-01", "time": "10:00:00"})).start.as_deref(), Some("2026-10-01T10:00"));
        let dir = std::env::temp_dir().join(format!("oaiy-sync-{}", uuid::Uuid::new_v4().simple()));
        let cal = Calendar::open(&dir, None);
        let a = cal.create(NewAppointment { service: "Cut".into(), start: "2026-10-02T14:30".into(), name: "Sam".into(), source: "text".into(), status: Some(Status::Confirmed), ..Default::default() }).unwrap();
        let v = answers(&a);
        assert_eq!((v["date"].as_str(), v["time"].as_str(), v["status"].as_str(), v["source"].as_str(), v["caller_name"].as_str()), (Some("2026-10-02"), Some("14:30"), Some("confirmed"), Some("sms"), Some("Sam")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn times_from_both_sides_compare() {
        assert!(when("2026-09-28 09:21:46") < when("2026-09-28T09:21:47Z"));
        assert!(when("nonsense").is_none());
    }
}
