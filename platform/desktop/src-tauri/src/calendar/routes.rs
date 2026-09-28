//! The calendar over HTTP, for the OAIY window (the Calendar page) and the
//! agent's calendar tools:
//!
//!   GET    /api/calendar?from=YYYY-MM-DD&to=YYYY-MM-DD   → {settings, appointments, now}
//!   PUT    /api/calendar/settings                          → the settings, checked
//!   GET    /api/calendar/free?from=&days=&service=&minutes= → {days: [{date, times}]}
//!   POST   /api/calendar/appointments                      → the new appointment (201)
//!   PATCH  /api/calendar/appointments/:id                  → the appointment, changed
//!   DELETE /api/calendar/appointments/:id                  → 204
//!   POST   /api/calendar/lookup {question, from}            → {digest} (what the phone is told)
//!   GET    /api/calendar/sync                               → how the FormLogic sync stands: its
//!                                                            state (offline...), when it last went
//!                                                            through, the changes waiting to go
//!   POST   /api/calendar/sync                               → sync now (answers within ten seconds,
//!                                                            `syncing` if it is still going)
//!
//! The changes (settings, appointments, a sync) answer 409 `module_disabled`
//! while no plugin provides the calendar; reading it still works.

use axum::extract::{Path, Query};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post, put};
use axum::{Json, Router};
use chrono::NaiveDate;
use serde::Deserialize;
use serde_json::json;

use super::{local_now, shared, Calendar, Change, NewAppointment, Settings};

fn calendar() -> Result<&'static Calendar, Response> {
    shared().ok_or_else(|| fail(StatusCode::SERVICE_UNAVAILABLE, "calendar_unavailable", "the calendar is not open"))
}

/// The calendar's changes answer only while a plugin provides it (else
/// `module_disabled`, before the body is read, and nothing is touched).
async fn require_calendar(request: axum::extract::Request, next: axum::middleware::Next) -> Response {
    if !super::available() {
        return crate::modules::disabled_response(crate::modules::CALENDAR);
    }
    next.run(request).await
}

fn fail(status: StatusCode, code: &str, message: &str) -> Response {
    (status, Json(json!({"error": {"code": code, "message": message}}))).into_response()
}

fn date(s: &Option<String>) -> Result<Option<NaiveDate>, Response> {
    match s.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(s) => NaiveDate::parse_from_str(s, "%Y-%m-%d").map(Some).map_err(|_| fail(StatusCode::BAD_REQUEST, "bad_date", "dates are YYYY-MM-DD")),
    }
}

pub fn router() -> Router {
    let changes = Router::new()
        .route("/api/calendar/settings", put(set_settings))
        .route("/api/calendar/appointments", post(create))
        .route("/api/calendar/appointments/:id", patch(update).delete(remove))
        .route_layer(axum::middleware::from_fn(require_calendar));
    Router::new()
        .route("/api/calendar", get(overview))
        .route("/api/calendar/free", get(free))
        .route("/api/calendar/lookup", post(lookup))
        // Reading how the sync stands is always there; syncing needs the calendar (see sync_now).
        .route("/api/calendar/sync", get(sync_status).post(sync_now))
        .merge(changes)
}

async fn sync_status() -> Response {
    Json(super::sync::last()).into_response()
}

async fn sync_now() -> Response {
    if !super::available() {
        return crate::modules::disabled_response(crate::modules::CALENDAR);
    }
    match tokio::task::spawn_blocking(super::sync::now).await {
        Ok(report) => Json(report).into_response(),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, "sync_failed", &e.to_string()),
    }
}

#[derive(Deserialize)]
struct Range {
    from: Option<String>,
    to: Option<String>,
}

async fn overview(Query(r): Query<Range>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    let (from, to) = match (date(&r.from), date(&r.to)) {
        (Ok(f), Ok(t)) => (f, t),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    Json(json!({"available": super::available(), "settings": cal.settings(), "appointments": cal.list(from, to), "now": local_now().format("%Y-%m-%dT%H:%M").to_string()})).into_response()
}

async fn set_settings(Json(s): Json<Settings>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    match cal.set_settings(s) {
        Ok(s) => Json(s).into_response(),
        Err(e) => fail(StatusCode::BAD_REQUEST, "bad_settings", &e),
    }
}

#[derive(Deserialize)]
struct FreeQuery {
    from: Option<String>,
    days: Option<u32>,
    service: Option<String>,
    minutes: Option<u32>,
}

async fn free(Query(q): Query<FreeQuery>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    let now = local_now();
    let from = match date(&q.from) {
        Ok(d) => d.unwrap_or(now.date()),
        Err(e) => return e,
    };
    let settings = cal.settings();
    let service = q.service.as_deref().and_then(|name| Calendar::service_named(&settings, name));
    let minutes = q.minutes.filter(|m| *m > 0).or(service.map(|s| s.minutes)).unwrap_or(settings.slot_minutes);
    Json(json!({"minutes": minutes, "service": service.map(|s| s.name.clone()), "days": cal.free(from, q.days.unwrap_or(7).clamp(1, 62), minutes, now)})).into_response()
}

async fn create(Json(new): Json<NewAppointment>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    match cal.create(new) {
        Ok(a) => (StatusCode::CREATED, Json(a)).into_response(),
        Err(e) => fail(StatusCode::BAD_REQUEST, "bad_appointment", &e),
    }
}

async fn update(Path(id): Path<String>, Json(change): Json<Change>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    match cal.update(&id, change) {
        Ok(a) => Json(a).into_response(),
        Err(e) if e.starts_with("no appointment") => fail(StatusCode::NOT_FOUND, "no_appointment", &e),
        Err(e) => fail(StatusCode::BAD_REQUEST, "bad_change", &e),
    }
}

async fn remove(Path(id): Path<String>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    match cal.remove(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => fail(StatusCode::NOT_FOUND, "no_appointment", &e),
    }
}

#[derive(Deserialize)]
struct LookupBody {
    #[serde(default)]
    question: String,
    #[serde(default)]
    from: String,
}

async fn lookup(Json(b): Json<LookupBody>) -> Response {
    let cal = match calendar() {
        Ok(c) => c,
        Err(e) => return e,
    };
    Json(json!({"digest": cal.lookup(&b.question, &b.from, local_now())})).into_response()
}
