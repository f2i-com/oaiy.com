//! The calendar: the business's opening hours and services, and its
//! appointments. It lives on this machine (`<data>/calendar/calendar.json`)
//! and needs nothing else; when the desktop is linked to FormLogic its
//! records sync with FormLogic's.
//!
//! The phone uses it twice. Aokie's `lookup_business_data` runs the
//! `business-lookup` flow, which (unless a flow of that name is stored) is
//! answered here with a digest: the hours, the services, the free times and
//! the caller's own appointments. And an appointment the caller agreed to on a
//! call (`aokie.appointment.requested`) is recorded here as a request, for
//! staff to confirm.
//!
//! Times are local wall-clock times (`YYYY-MM-DDTHH:MM`), as the business
//! keeps them.

pub mod routes;
pub mod sync;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::{Datelike, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Weekday};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One kind of appointment the business offers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Service {
    #[serde(default)]
    pub id: String,
    pub name: String,
    /// How long it takes.
    pub minutes: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// As the business says it ("from $60"); free text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub price: String,
}

/// Open from `open` to `close` (`HH:MM`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Span {
    pub open: String,
    pub close: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// The business's name, as the receptionist says it.
    #[serde(default)]
    pub business: String,
    /// Seven days, Monday first, each with its opening spans (none: closed).
    pub hours: Vec<Vec<Span>>,
    pub services: Vec<Service>,
    /// The step between the times offered.
    pub slot_minutes: u32,
    /// How soon from now a time may be offered.
    pub notice_minutes: u32,
    /// How far ahead times are offered.
    pub horizon_days: u32,
    /// Text the person when an appointment they asked for is confirmed.
    pub text_confirmations: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let day = |open: &str, close: &str| vec![Span { open: open.into(), close: close.into() }];
        Self {
            business: String::new(),
            hours: vec![day("09:00", "17:00"), day("09:00", "17:00"), day("09:00", "17:00"), day("09:00", "17:00"), day("09:00", "17:00"), vec![], vec![]],
            services: vec![Service { id: "appointment".into(), name: "Appointment".into(), minutes: 30, description: String::new(), price: String::new() }],
            slot_minutes: 30,
            notice_minutes: 60,
            horizon_days: 30,
            text_confirmations: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Asked for (on a call, by text, by the agent): staff have not confirmed it.
    Requested,
    Confirmed,
    Declined,
    Cancelled,
    Done,
}

impl Status {
    /// Whether it holds its time (no one else is offered it).
    pub fn holds_time(self) -> bool {
        matches!(self, Status::Requested | Status::Confirmed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Appointment {
    pub id: String,
    pub service: String,
    /// Local time, `YYYY-MM-DDTHH:MM`.
    pub start: String,
    pub minutes: u32,
    pub status: Status,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub phone: String,
    #[serde(default)]
    pub notes: String,
    /// Where it came from: `call`, `text`, `agent`, `manual`, `formlogic`.
    pub source: String,
    /// Aokie's appointment request, when it came from a call (one record per request).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// FormLogic's copy of it, once synced: `{id, revision, syncedAt}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formlogic: Option<Value>,
}

impl Appointment {
    pub fn starts(&self) -> Option<NaiveDateTime> {
        parse_start(&self.start)
    }

    pub fn ends(&self) -> Option<NaiveDateTime> {
        self.starts().map(|s| s + Duration::minutes(self.minutes as i64))
    }
}

/// What a new appointment is given; the rest is filled in.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewAppointment {
    #[serde(default)]
    pub service: String,
    /// `YYYY-MM-DDTHH:MM`, or `date` and `time` apart.
    #[serde(default)]
    pub start: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub time: String,
    #[serde(default)]
    pub minutes: Option<u32>,
    #[serde(default)]
    pub status: Option<Status>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub phone: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub source: String,
}

/// What may change about an appointment.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Change {
    pub status: Option<Status>,
    pub start: Option<String>,
    pub minutes: Option<u32>,
    pub service: Option<String>,
    pub name: Option<String>,
    pub phone: Option<String>,
    pub notes: Option<String>,
    pub formlogic: Option<Value>,
}

#[derive(Default, Serialize, Deserialize)]
struct Book {
    #[serde(default)]
    settings: Settings,
    #[serde(default)]
    appointments: Vec<Appointment>,
    /// Appointments deleted here that FormLogic may still have, kept until it
    /// has been told (see `sync`), so a deletion made offline is not undone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    deleted: Vec<Tombstone>,
    /// Where the sync with FormLogic has got to.
    #[serde(default)]
    sync: sync::State,
}

/// An appointment deleted on this machine, until FormLogic has deleted its copy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Tombstone {
    /// The local id it had.
    pub id: String,
    /// FormLogic's record, when the two were paired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formlogic_id: Option<String>,
    /// The key FormLogic's copy carries as `request_id` (a call's request id,
    /// or `oaiy:<id>` for one made here), to find a copy made before the two
    /// were paired, or after (FormLogic's own flow records a call's request).
    pub request_key: String,
    pub deleted_at: String,
    /// Looked for once and not found: kept for a copy that turns up late, but
    /// no longer a change waiting to sync.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub checked: bool,
}

impl Book {
    /// Add an appointment (the checks and defaults `Calendar::create` applies).
    fn create(&mut self, new: NewAppointment) -> Result<Appointment, String> {
        let start = if !new.start.trim().is_empty() {
            parse_start(&new.start).ok_or("start is YYYY-MM-DDTHH:MM")?
        } else {
            let date = NaiveDate::parse_from_str(new.date.trim(), "%Y-%m-%d").map_err(|_| "date is YYYY-MM-DD".to_string())?;
            let time = parse_time(&new.time).ok_or("time is HH:MM")?;
            date.and_time(time)
        };
        let service = Calendar::service_named(&self.settings, &new.service);
        let minutes = new.minutes.filter(|m| *m > 0).or(service.map(|s| s.minutes)).unwrap_or(self.settings.slot_minutes);
        let now = now_rfc3339();
        let mut a = Appointment {
            id: format!("appt_{}", uuid::Uuid::new_v4().simple()),
            service: service.map(|s| s.name.clone()).unwrap_or_else(|| new.service.trim().to_string()),
            start: format_start(start),
            minutes,
            status: new.status.unwrap_or(Status::Confirmed),
            name: new.name.trim().to_string(),
            phone: new.phone.trim().to_string(),
            notes: new.notes.trim().to_string(),
            source: if new.source.trim().is_empty() { "manual".into() } else { new.source.trim().to_string() },
            request_id: None,
            call_id: None,
            created_at: now.clone(),
            updated_at: now,
            formlogic: None,
        };
        // How it began, so a sync can tell what changed here from what changed in FormLogic.
        a.formlogic = sync::first_version(&a);
        self.appointments.push(a.clone());
        Ok(a)
    }

    /// Change an appointment (what `Calendar::update` does).
    fn change(&mut self, id: &str, change: Change) -> Result<Appointment, String> {
        let start = match &change.start {
            Some(s) => Some(parse_start(s).ok_or("start is YYYY-MM-DDTHH:MM")?),
            None => None,
        };
        let settings = &self.settings;
        let a = self.appointments.iter_mut().find(|a| a.id == id).ok_or_else(|| format!("no appointment {id}"))?;
        if let Some(s) = change.status {
            a.status = s;
        }
        if let Some(t) = start {
            a.start = format_start(t);
        }
        if let Some(svc) = change.service {
            match Calendar::service_named(settings, &svc) {
                Some(found) => {
                    a.service = found.name.clone();
                    if change.minutes.is_none() {
                        a.minutes = found.minutes;
                    }
                }
                None => a.service = svc.trim().to_string(),
            }
        }
        if let Some(m) = change.minutes.filter(|m| *m > 0) {
            a.minutes = m;
        }
        if let Some(n) = change.name {
            a.name = n.trim().to_string();
        }
        if let Some(p) = change.phone {
            a.phone = p.trim().to_string();
        }
        if let Some(n) = change.notes {
            a.notes = n.trim().to_string();
        }
        if let Some(f) = change.formlogic {
            a.formlogic = Some(f);
        }
        a.updated_at = later_than(&a.updated_at);
        Ok(a.clone())
    }
}

/// Free times on one day.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FreeDay {
    pub date: String,
    /// `HH:MM`.
    pub times: Vec<String>,
}

pub struct Calendar {
    path: PathBuf,
    /// Where stored flows live: a stored `business-lookup` flow answers instead of the calendar.
    flows_dir: Option<PathBuf>,
    book: Mutex<Book>,
}

static SHARED: OnceLock<Calendar> = OnceLock::new();
static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Open the calendar in `data_dir/calendar` for the whole desktop (once; later calls keep the first).
pub fn init(data_dir: &Path) -> &'static Calendar {
    let _ = DATA_DIR.set(data_dir.to_path_buf());
    SHARED.get_or_init(|| Calendar::open(&data_dir.join("calendar"), Some(data_dir.join("flows"))))
}

/// Whether the calendar is in use: it is the phone receptionist's diary, so it
/// is there while the Aokie plugin (the phone) is installed. Kept either way.
pub fn available() -> bool {
    DATA_DIR.get().is_some_and(|d| receptionist_installed(d))
}

fn receptionist_installed(data_dir: &Path) -> bool {
    data_dir.join("plugins").join("aokie").join("manifest.json").is_file()
}

/// The desktop's calendar, once `init` has run.
pub fn shared() -> Option<&'static Calendar> {
    SHARED.get()
}

pub fn parse_start(s: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M").ok().or_else(|| NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M:%S").ok())
}

fn parse_time(s: &str) -> Option<NaiveTime> {
    let s = s.trim().to_ascii_lowercase();
    if let Ok(t) = NaiveTime::parse_from_str(&s, "%H:%M") {
        return Some(t);
    }
    // "10am", "10:30 pm", "3 pm"
    let (clock, pm) = if let Some(c) = s.strip_suffix("pm") {
        (c.trim(), true)
    } else if let Some(c) = s.strip_suffix("am") {
        (c.trim(), false)
    } else {
        return None;
    };
    let (h, m) = match clock.split_once(':') {
        Some((h, m)) => (h.trim().parse::<u32>().ok()?, m.trim().parse::<u32>().ok()?),
        None => (clock.parse::<u32>().ok()?, 0),
    };
    if !(1..=12).contains(&h) {
        return None;
    }
    NaiveTime::from_hms_opt(h % 12 + if pm { 12 } else { 0 }, m, 0)
}

fn format_start(t: NaiveDateTime) -> String {
    t.format("%Y-%m-%dT%H:%M").to_string()
}

/// "9:30 am", as a receptionist says it.
pub fn say_time(t: NaiveTime) -> String {
    let (pm, h) = t.hour12();
    if t.minute() == 0 {
        format!("{h} {}", if pm { "pm" } else { "am" })
    } else {
        format!("{h}:{:02} {}", t.minute(), if pm { "pm" } else { "am" })
    }
}

/// "Tue 29 Sep".
pub fn say_date(d: NaiveDate) -> String {
    d.format("%a %-d %b").to_string()
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// A change's time: now, and always after `prev`, so a change made in the
/// same instant as the last sync still reads as a change.
fn later_than(prev: &str) -> String {
    let now = chrono::Utc::now();
    let after = chrono::DateTime::parse_from_rfc3339(prev).map(|t| t.with_timezone(&chrono::Utc) + Duration::milliseconds(1)).unwrap_or(now);
    now.max(after).to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn slug(name: &str) -> String {
    let s: String = name.trim().to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "service".into() } else { s }
}

impl Calendar {
    /// The calendar in `dir` (empty, with default hours, when it has none yet).
    pub fn open(dir: &Path, flows_dir: Option<PathBuf>) -> Self {
        let path = dir.join("calendar.json");
        let book = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Book>(&t).ok()).unwrap_or_default();
        Self { path, flows_dir, book: Mutex::new(book) }
    }

    fn save(&self, book: &Book) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(book).map_err(|e| e.to_string())?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("could not save {}: {e}", self.path.display()))
    }

    fn with<T>(&self, change: impl FnOnce(&mut Book) -> Result<T, String>) -> Result<T, String> {
        let mut book = self.book.lock().map_err(|_| "the calendar is unavailable".to_string())?;
        let out = change(&mut book)?;
        self.save(&book)?;
        Ok(out)
    }

    pub fn settings(&self) -> Settings {
        self.book.lock().map(|b| b.settings.clone()).unwrap_or_default()
    }

    /// Replace the settings, checked: seven days of `HH:MM` spans, services with a length.
    pub fn set_settings(&self, mut s: Settings) -> Result<Settings, String> {
        if s.hours.len() != 7 {
            return Err("hours needs seven days, Monday first".into());
        }
        for (i, day) in s.hours.iter().enumerate() {
            for span in day {
                let (Some(open), Some(close)) = (parse_time(&span.open), parse_time(&span.close)) else {
                    return Err(format!("day {}: times are HH:MM", i + 1));
                };
                if open >= close {
                    return Err(format!("day {}: {} is not before {}", i + 1, span.open, span.close));
                }
            }
        }
        if !(5..=240).contains(&s.slot_minutes) {
            return Err("the step between times is 5 to 240 minutes".into());
        }
        let mut ids = std::collections::HashSet::new();
        for svc in &mut s.services {
            svc.name = svc.name.trim().to_string();
            if svc.name.is_empty() || svc.minutes == 0 || svc.minutes > 24 * 60 {
                return Err("each service needs a name and a length in minutes".into());
            }
            if svc.id.trim().is_empty() {
                svc.id = slug(&svc.name);
            }
            let base = svc.id.clone();
            let mut n = 2;
            while !ids.insert(svc.id.clone()) {
                svc.id = format!("{base}-{n}");
                n += 1;
            }
        }
        s.horizon_days = s.horizon_days.clamp(1, 366);
        self.with(|book| {
            book.settings = s.clone();
            Ok(s)
        })
    }

    /// Appointments whose start falls in `[from, to)` (either end open), in time order.
    pub fn list(&self, from: Option<NaiveDate>, to: Option<NaiveDate>) -> Vec<Appointment> {
        let book = match self.book.lock() {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        };
        let mut out: Vec<Appointment> = book
            .appointments
            .iter()
            .filter(|a| {
                let Some(d) = a.starts().map(|t| t.date()) else { return true };
                from.map_or(true, |f| d >= f) && to.map_or(true, |t| d < t)
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| a.start.cmp(&b.start));
        out
    }

    pub fn get(&self, id: &str) -> Option<Appointment> {
        self.book.lock().ok()?.appointments.iter().find(|a| a.id == id).cloned()
    }

    fn service_named<'a>(settings: &'a Settings, name: &str) -> Option<&'a Service> {
        let n = name.trim().to_ascii_lowercase();
        settings.services.iter().find(|s| s.id == n || s.name.to_ascii_lowercase() == n).or_else(|| {
            // "lawnmowing" for "Lawn mowing": compare without spaces and dashes.
            let squash = |s: &str| s.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>();
            let n = squash(name);
            (!n.is_empty()).then(|| settings.services.iter().find(|s| squash(&s.name) == n)).flatten()
        })
    }

    /// A new appointment. A change here is sent to FormLogic soon after, when linked.
    pub fn create(&self, new: NewAppointment) -> Result<Appointment, String> {
        let a = self.with(|book| book.create(new))?;
        sync::nudge();
        Ok(a)
    }

    pub fn update(&self, id: &str, change: Change) -> Result<Appointment, String> {
        let a = self.with(|book| book.change(id, change))?;
        sync::nudge();
        Ok(a)
    }

    /// Delete an appointment. One FormLogic has (or may have) a copy of leaves
    /// a tombstone, so the next sync deletes that copy instead of bringing the
    /// appointment back.
    pub fn remove(&self, id: &str) -> Result<(), String> {
        self.with(|book| {
            let at = book.appointments.iter().position(|a| a.id == id).ok_or_else(|| format!("no appointment {id}"))?;
            let a = book.appointments.remove(at);
            if let Some(t) = sync::tombstone_for(&a) {
                book.deleted.push(t);
            }
            Ok(())
        })?;
        sync::nudge();
        Ok(())
    }

    /// An appointment the caller agreed to on a call (`aokie.appointment.requested`):
    /// recorded once per request, for staff to confirm. None when it is already
    /// here or the event lacks a date and time.
    pub fn record_request(&self, data: &Value) -> Option<Appointment> {
        let text = |k: &str| data.get(k).and_then(Value::as_str).unwrap_or("").trim().to_string();
        let request_id = text("requestId");
        if request_id.is_empty() {
            return None;
        }
        let date = NaiveDate::parse_from_str(&text("date"), "%Y-%m-%d").ok()?;
        let time = parse_time(&text("time"))?;
        let call_id = text("callId");
        let made = self
            .with(|book| {
                // Once per request: one already here, or one deleted here (the
                // same event delivered again must not bring it back).
                if book.appointments.iter().any(|a| a.request_id.as_deref() == Some(request_id.as_str())) || book.deleted.iter().any(|t| t.request_key == request_id) {
                    return Ok(None);
                }
                let mut a = book.create(NewAppointment {
                    service: text("service"),
                    start: format_start(date.and_time(time)),
                    status: Some(Status::Requested),
                    name: text("callerName"),
                    phone: text("from"),
                    notes: String::new(),
                    source: "call".into(),
                    ..Default::default()
                })?;
                // In the same save, so no sync sees it without its request id.
                a.request_id = Some(request_id.clone());
                a.call_id = (!call_id.is_empty()).then(|| call_id.clone());
                if let Some(stored) = book.appointments.iter_mut().find(|x| x.id == a.id) {
                    *stored = a.clone();
                }
                Ok(Some(a))
            })
            .ok()??;
        sync::nudge();
        Some(made)
    }

    /// Free times from `from` for `days` days, for something `minutes` long,
    /// none sooner than the notice after `now`.
    pub fn free(&self, from: NaiveDate, days: u32, minutes: u32, now: NaiveDateTime) -> Vec<FreeDay> {
        let book = match self.book.lock() {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        };
        let s = &book.settings;
        let earliest = now + Duration::minutes(s.notice_minutes as i64);
        let last_day = now.date() + Duration::days(s.horizon_days as i64);
        let taken: Vec<(NaiveDateTime, NaiveDateTime)> = book.appointments.iter().filter(|a| a.status.holds_time()).filter_map(|a| Some((a.starts()?, a.ends()?))).collect();
        let step = Duration::minutes(s.slot_minutes.max(5) as i64);
        let length = Duration::minutes(minutes.max(1) as i64);
        let mut out = Vec::new();
        for offset in 0..days {
            let day = from + Duration::days(offset as i64);
            if day > last_day {
                break;
            }
            let spans = s.hours.get(day.weekday().num_days_from_monday() as usize).map(Vec::as_slice).unwrap_or(&[]);
            let mut times = Vec::new();
            for span in spans {
                let (Some(open), Some(close)) = (parse_time(&span.open), parse_time(&span.close)) else { continue };
                let mut t = day.and_time(open);
                let end = day.and_time(close);
                while t + length <= end {
                    let clash = taken.iter().any(|(a, b)| t < *b && *a < t + length);
                    if t >= earliest && !clash {
                        times.push(t.format("%H:%M").to_string());
                    }
                    t += step;
                }
            }
            out.push(FreeDay { date: day.format("%Y-%m-%d").to_string(), times });
        }
        out
    }

    /// Whether a stored flow answers `id` instead of the calendar.
    pub fn flow_answers(&self, id: &str) -> bool {
        self.flows_dir.as_ref().is_some_and(|d| d.join(format!("{id}.json")).is_file())
    }

    /// What the receptionist is told when it looks something up: the hours,
    /// the services, the free times and the caller's own appointments. Plain
    /// sentences, since a model reads them and a person may hear them.
    pub fn lookup(&self, question: &str, from: &str, now: NaiveDateTime) -> String {
        let settings = self.settings();
        let mut out = Vec::new();
        out.push(format!(
            "{}Now: {}, {}.",
            if settings.business.trim().is_empty() { String::new() } else { format!("Business: {}. ", settings.business.trim()) },
            now.date().format("%A %-d %B %Y"),
            say_time(now.time())
        ));
        const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        let hours: Vec<String> = settings
            .hours
            .iter()
            .enumerate()
            .map(|(i, spans)| {
                let when = if spans.is_empty() {
                    "closed".to_string()
                } else {
                    spans
                        .iter()
                        .filter_map(|s| Some(format!("{} to {}", say_time(parse_time(&s.open)?), say_time(parse_time(&s.close)?))))
                        .collect::<Vec<_>>()
                        .join(" and ")
                };
                format!("{} {}", DAYS[i % 7], when)
            })
            .collect();
        out.push(format!("Opening hours: {}.", hours.join("; ")));
        if !settings.services.is_empty() {
            let services: Vec<String> = settings
                .services
                .iter()
                .map(|s| {
                    let mut t = format!("{} ({} min", s.name, s.minutes);
                    if !s.price.is_empty() {
                        t.push_str(&format!(", {}", s.price));
                    }
                    t.push(')');
                    if !s.description.is_empty() {
                        t.push_str(&format!(": {}", s.description));
                    }
                    t
                })
                .collect();
            out.push(format!("Services: {}.", services.join("; ")));
        }
        // The service the question names, if any, sets the length of the times offered.
        let q = question.to_ascii_lowercase();
        let named = settings.services.iter().find(|s| q.contains(&s.name.to_ascii_lowercase()));
        let minutes = named.map(|s| s.minutes).or(settings.services.first().map(|s| s.minutes)).unwrap_or(settings.slot_minutes);
        let free = self.free(now.date(), 14, minutes, now);
        let open_days: Vec<String> = free
            .iter()
            .filter(|d| !d.times.is_empty())
            .take(7)
            .filter_map(|d| {
                let date = NaiveDate::parse_from_str(&d.date, "%Y-%m-%d").ok()?;
                let times: Vec<String> = d.times.iter().take(8).filter_map(|t| parse_time(t).map(say_time)).collect();
                let more = if d.times.len() > 8 { format!(" and {} more", d.times.len() - 8) } else { String::new() };
                Some(format!("{}: {}{}", say_date(date), times.join(", "), more))
            })
            .collect();
        if open_days.is_empty() {
            out.push("Free times: none in the next two weeks.".into());
        } else {
            out.push(format!("Free times for {} ({} min): {}.", named.map_or("an appointment", |s| s.name.as_str()), minutes, open_days.join(". ")));
        }
        let digits = |s: &str| s.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
        let caller = digits(from);
        if caller.len() >= 6 {
            let tail = &caller[caller.len() - 9.min(caller.len())..];
            let theirs: Vec<String> = self
                .list(Some(now.date()), None)
                .into_iter()
                .filter(|a| a.status.holds_time() && digits(&a.phone).ends_with(tail))
                .filter_map(|a| {
                    let t = a.starts()?;
                    Some(format!("{} at {}, {}{}", say_date(t.date()), say_time(t.time()), a.service, if a.status == Status::Requested { " (requested, not yet confirmed)" } else { " (confirmed)" }))
                })
                .collect();
            out.push(if theirs.is_empty() { "This caller has no upcoming appointments.".into() } else { format!("This caller's appointments: {}.", theirs.join("; ")) });
        }
        out.push("A time agreed on a call is a request until staff confirm it.".into());
        out.join("\n")
    }
}

/// Today's local time, to the minute.
pub fn local_now() -> NaiveDateTime {
    let n = Local::now().naive_local();
    n.with_second(0).and_then(|t| t.with_nanosecond(0)).unwrap_or(n)
}

/// Monday of the week `d` falls in.
pub fn week_of(d: NaiveDate) -> NaiveDate {
    d - Duration::days(d.weekday().num_days_from_monday() as i64)
}

#[allow(dead_code)]
fn weekday_index(w: Weekday) -> usize {
    w.num_days_from_monday() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_calendar_is_there_with_the_phone_receptionist() {
        let dir = std::env::temp_dir().join(format!("oaiy-calendar-available-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!receptionist_installed(&dir));
        std::fs::create_dir_all(dir.join("plugins/aokie")).unwrap();
        assert!(!receptionist_installed(&dir), "a folder alone is not an installed plugin");
        std::fs::write(dir.join("plugins/aokie/manifest.json"), "{}").unwrap();
        assert!(receptionist_installed(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
    use serde_json::json;

    fn calendar() -> (Calendar, PathBuf) {
        let dir = std::env::temp_dir().join(format!("oaiy-calendar-{}", uuid::Uuid::new_v4().simple()));
        (Calendar::open(&dir, None), dir)
    }

    fn at(s: &str) -> NaiveDateTime {
        parse_start(s).unwrap()
    }

    #[test]
    fn times_are_read_as_people_write_them() {
        assert_eq!(parse_time("10:00"), NaiveTime::from_hms_opt(10, 0, 0));
        assert_eq!(parse_time("10am"), NaiveTime::from_hms_opt(10, 0, 0));
        assert_eq!(parse_time("3:30 pm"), NaiveTime::from_hms_opt(15, 30, 0));
        assert_eq!(parse_time("12 pm"), NaiveTime::from_hms_opt(12, 0, 0));
        assert_eq!(parse_time("12am"), NaiveTime::from_hms_opt(0, 0, 0));
        assert_eq!(parse_time("13 pm"), None);
        assert_eq!(say_time(NaiveTime::from_hms_opt(9, 30, 0).unwrap()), "9:30 am");
        assert_eq!(say_time(NaiveTime::from_hms_opt(14, 0, 0).unwrap()), "2 pm");
    }

    #[test]
    fn free_times_skip_closed_days_taken_times_and_short_notice() {
        let (cal, dir) = calendar();
        // Monday 28 Sept 2026, 8:30 am: the notice (60 min) rules out 9:00 and 9:30.
        let now = at("2026-09-28T08:30");
        cal.create(NewAppointment { service: "Appointment".into(), start: "2026-09-28T11:00".into(), ..Default::default() }).unwrap();
        cal.create(NewAppointment { start: "2026-09-28T12:00".into(), status: Some(Status::Cancelled), ..Default::default() }).unwrap();
        let free = cal.free(now.date(), 7, 30, now);
        assert_eq!(free[0].date, "2026-09-28");
        assert_eq!(free[0].times.first().map(String::as_str), Some("09:30"));
        assert!(!free[0].times.contains(&"11:00".to_string()), "11:00 is taken");
        assert!(free[0].times.contains(&"12:00".to_string()), "a cancelled one frees its time");
        assert_eq!(free[0].times.last().map(String::as_str), Some("16:30"));
        // Saturday and Sunday are closed by default.
        assert!(free[5].times.is_empty() && free[6].times.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_call_request_is_recorded_once_and_found_again() {
        let (cal, dir) = calendar();
        let mut s = cal.settings();
        s.services = vec![Service { id: String::new(), name: "Lawn mowing".into(), minutes: 60, description: String::new(), price: "from $60".into() }];
        cal.set_settings(s).unwrap();
        let event = json!({"requestId": "appt_1", "callId": "call_1", "from": "0491570006", "callerName": "Lance", "service": "Lawnmowing", "date": "2026-10-01", "time": "10:00"});
        let a = cal.record_request(&event).unwrap();
        assert_eq!((a.status, a.service.as_str(), a.minutes, a.start.as_str(), a.source.as_str()), (Status::Requested, "Lawn mowing", 60, "2026-10-01T10:00", "call"));
        assert_eq!(a.request_id.as_deref(), Some("appt_1"));
        assert!(cal.record_request(&event).is_none(), "the same request again is not a second appointment");
        // It survives a restart.
        let again = Calendar::open(&dir, None);
        assert_eq!(again.list(None, None).len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_lookup_says_hours_services_free_times_and_the_callers_own() {
        let (cal, dir) = calendar();
        let mut s = cal.settings();
        s.business = "Green Lawns".into();
        s.services = vec![Service { id: String::new(), name: "Lawn mowing".into(), minutes: 60, description: String::new(), price: "from $60".into() }];
        cal.set_settings(s).unwrap();
        cal.record_request(&json!({"requestId": "r1", "from": "+61491570006", "callerName": "Lance", "service": "Lawn mowing", "date": "2026-10-01", "time": "10:00"})).unwrap();
        let digest = cal.lookup("Any times for lawn mowing this week?", "0491570006", at("2026-09-28T08:00"));
        assert!(digest.contains("Business: Green Lawns."), "{digest}");
        assert!(digest.contains("Now: Monday 28 September 2026, 8 am."), "{digest}");
        assert!(digest.contains("Mon 9 am to 5 pm"), "{digest}");
        assert!(digest.contains("Sat closed"), "{digest}");
        assert!(digest.contains("Lawn mowing (60 min, from $60)"), "{digest}");
        assert!(digest.contains("Free times for Lawn mowing (60 min): Mon 28 Sep: 9 am, 9:30 am"), "{digest}");
        assert!(digest.contains("This caller's appointments: Thu 1 Oct at 10 am, Lawn mowing (requested, not yet confirmed)."), "{digest}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn settings_are_checked() {
        let (cal, dir) = calendar();
        let mut s = cal.settings();
        s.hours[0] = vec![Span { open: "17:00".into(), close: "09:00".into() }];
        assert!(cal.set_settings(s).unwrap_err().contains("not before"));
        let mut s = cal.settings();
        s.services = vec![Service { id: String::new(), name: "Cut".into(), minutes: 30, description: String::new(), price: String::new() }, Service { id: String::new(), name: "cut".into(), minutes: 45, description: String::new(), price: String::new() }];
        let saved = cal.set_settings(s).unwrap();
        assert_eq!(saved.services.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["cut", "cut-2"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn appointments_change_and_go() {
        let (cal, dir) = calendar();
        let a = cal.create(NewAppointment { date: "2026-10-02".into(), time: "2pm".into(), name: "Sam".into(), status: Some(Status::Requested), source: "text".into(), ..Default::default() }).unwrap();
        assert_eq!((a.start.as_str(), a.minutes), ("2026-10-02T14:00", 30));
        let b = cal.update(&a.id, Change { status: Some(Status::Confirmed), start: Some("2026-10-02T15:00".into()), ..Default::default() }).unwrap();
        assert_eq!((b.status, b.start.as_str()), (Status::Confirmed, "2026-10-02T15:00"));
        assert_eq!(cal.list(NaiveDate::from_ymd_opt(2026, 10, 2), NaiveDate::from_ymd_opt(2026, 10, 3)).len(), 1);
        assert!(cal.list(NaiveDate::from_ymd_opt(2026, 10, 3), None).is_empty());
        cal.remove(&a.id).unwrap();
        assert!(cal.remove(&a.id).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
