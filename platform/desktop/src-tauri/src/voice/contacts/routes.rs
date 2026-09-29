//! The contacts over HTTP: the dashboard's Contacts page, the Agent's
//! control tools, and the app (what a call or a text needs to know).
//!
//!   GET    /api/contacts?q=                              → {contacts, total}: name-sorted; q finds
//!                                                          names, numbers written any way, notes, facts
//!   GET    /api/contacts/:number                         → the contact; 404 `no_contact`
//!   PUT    /api/contacts/:number {name?, notes?, number?} → the contact, made if there is none; a name
//!                                                          given is the person's (`nameBy: owner`)
//!   DELETE /api/contacts/:number                         → 204
//!   POST   /api/contacts/:number/facts {text, by?}       → {contact, added, dropped}: 201 when added,
//!                                                          200 when it was there; `by` agent (default) or owner
//!   DELETE /api/contacts/:number/facts/:index?text=      → {contact, forgotten}; with `text`, 409
//!                                                          `fact_changed` when the fact there says otherwise
//!   GET    /api/contacts/export.csv                      → the CSV file (see [`super::csv`])
//!   POST   /api/contacts/import {csv, country?, replaceNames?, preview?} → what it did, or would do
//!
//! `:number` is a number written any way (`0491 570 006`, `+61491570006`,
//! the key `491570006`). A refusal is `{error: {code, message}}`.
//!
//! None of these need the phone. The contacts are the person's own record,
//! with no effect outside this computer: they are read and changed whether or
//! not a plugin provides the phone (setting names and notes, or importing,
//! before the phone is paired or while its plugin is stopped). What talks to a
//! caller stays the phone's: `PUT /api/voice/callers`, the receptionist's
//! own naming during a call, still answers `module_disabled` without it.
//! All of it is gated like `/api/voice/*` (`http.rs`: reading is a restricted
//! read, changing takes the privileged gate).

use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::{csv, By, Change, Error, Store};

pub fn router(store: Store) -> Router {
    Router::new()
        .route("/api/contacts", get(list))
        .route("/api/contacts/export.csv", get(export))
        .route("/api/contacts/import", post(import).layer(DefaultBodyLimit::max(csv::MAX_BYTES)))
        .route("/api/contacts/:number", get(one).put(set).delete(remove))
        .route("/api/contacts/:number/facts", post(add_fact))
        .route("/api/contacts/:number/facts/:index", delete(forget_fact))
        .with_state(store)
}

fn fail(e: Error) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(json!({"error": {"code": e.code, "message": e.message}}))).into_response()
}

fn answer<T: serde::Serialize>(result: Result<T, Error>) -> Response {
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => fail(e),
    }
}

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default)]
    q: String,
}

async fn list(State(s): State<Store>, Query(q): Query<ListQuery>) -> Response {
    match s.list(&q.q) {
        Ok((contacts, total)) => Json(json!({"contacts": contacts, "total": total})).into_response(),
        Err(e) => fail(e),
    }
}

async fn one(State(s): State<Store>, Path(number): Path<String>) -> Response {
    answer(s.get(&number))
}

async fn set(State(s): State<Store>, Path(number): Path<String>, Json(change): Json<Change>) -> Response {
    answer(s.set(&number, change))
}

async fn remove(State(s): State<Store>, Path(number): Path<String>) -> Response {
    match s.remove(&number) {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => fail(e),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewFact {
    text: String,
    #[serde(default)]
    by: Option<String>,
}

async fn add_fact(State(s): State<Store>, Path(number): Path<String>, Json(body): Json<NewFact>) -> Response {
    let by = match body.by.as_deref().map(str::trim).filter(|b| !b.is_empty()) {
        None => By::Agent,
        Some(b) => match By::parse(b) {
            Some(by) => by,
            None => return fail(Error::new(400, "bad_by", format!("by is agent or owner, not {b:?}"))),
        },
    };
    match s.add_fact(&number, &body.text, by) {
        Ok(r) => (if r.added { StatusCode::CREATED } else { StatusCode::OK }, Json(r)).into_response(),
        Err(e) => fail(e),
    }
}

#[derive(Deserialize)]
struct ForgetQuery {
    text: Option<String>,
}

async fn forget_fact(State(s): State<Store>, Path((number, index)): Path<(String, usize)>, Query(q): Query<ForgetQuery>) -> Response {
    match s.forget_fact(&number, index, q.text.as_deref()) {
        Ok((contact, fact)) => Json(json!({"contact": contact, "forgotten": fact})).into_response(),
        Err(e) => fail(e),
    }
}

async fn export(State(s): State<Store>) -> Response {
    match tokio::task::spawn_blocking(move || s.export_csv()).await {
        Ok(Ok((csv, _))) => {
            let file = format!("oaiy-contacts-{}.csv", chrono::Local::now().format("%Y-%m-%d"));
            let headers = [
                (header::CONTENT_TYPE, "text/csv; charset=utf-8".to_string()),
                (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{file}\"")),
                (header::CACHE_CONTROL, "no-store".to_string()),
            ];
            (headers, csv).into_response()
        }
        Ok(Err(e)) => fail(e),
        Err(e) => fail(Error::new(500, "export_failed", e.to_string())),
    }
}

async fn import(State(s): State<Store>, Json(request): Json<csv::ImportRequest>) -> Response {
    match tokio::task::spawn_blocking(move || s.import(&request)).await {
        Ok(result) => answer(result),
        Err(e) => fail(Error::new(500, "import_failed", e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Dir;
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use serde_json::Value;
    use tower::ServiceExt as _;

    async fn send(app: &Router, method: Method, path: &str, headers: &[(&str, &str)], body: Option<Value>) -> (StatusCode, Value, axum::http::HeaderMap, Vec<u8>) {
        let mut req = Request::builder().method(method).uri(path);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        if body.is_some() {
            req = req.header("content-type", "application/json");
        }
        let response = app.clone().oneshot(req.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap()).await.unwrap();
        let (status, head) = (response.status(), response.headers().clone());
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap().to_vec();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), head, bytes)
    }

    #[tokio::test]
    async fn a_contact_is_made_read_changed_and_forgotten_over_http_with_the_phone_off() {
        // The contacts are the person's own: nothing here needs the phone.
        let _off = crate::modules::test_gate::enable(&[]);
        let d = Dir::new("routes");
        let app = router(d.store());
        let (status, v, ..) = send(&app, Method::GET, "/api/contacts", &[], None).await;
        assert_eq!((status, v), (StatusCode::OK, json!({"contacts": [], "total": 0})));

        // Made by the person, with one name: theirs.
        let (status, v, ..) = send(&app, Method::PUT, "/api/contacts/0491%20570%20006", &[], Some(json!({"name": "Lance", "notes": "Prefers texts"}))).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!((v["key"].as_str(), v["name"].as_str(), v["nameBy"].as_str(), v["notes"].as_str()), (Some("491570006"), Some("Lance"), Some("owner"), Some("Prefers texts")));
        // Read by any way of writing the number.
        for path in ["/api/contacts/%2B61491570006", "/api/contacts/491570006", "/api/contacts/0491570006"] {
            let (status, v, ..) = send(&app, Method::GET, path, &[], None).await;
            assert_eq!((status, v["name"].as_str()), (StatusCode::OK, Some("Lance")), "{path}");
        }
        let (status, v, ..) = send(&app, Method::GET, "/api/contacts/0400000001", &[], None).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("no_contact")));
        let (status, v, ..) = send(&app, Method::GET, "/api/contacts/Private", &[], None).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("hidden")));
        let (status, _, ..) = send(&app, Method::PUT, "/api/contacts/0491570006", &[], Some(json!({"nickname": "L"}))).await;
        assert!(status.is_client_error(), "an unknown field is refused, not ignored");

        // What the receptionist remembers: 201 when new, 200 when it was there.
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/0491570006/facts", &[], Some(json!({"text": "Has a dog called Max"}))).await;
        assert_eq!((status, v["added"].as_bool(), v["contact"]["facts"][0]["by"].as_str()), (StatusCode::CREATED, Some(true), Some("agent")));
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/0491570006/facts", &[], Some(json!({"text": "has a dog called max"}))).await;
        assert_eq!((status, v["added"].as_bool()), (StatusCode::OK, Some(false)));
        let (status, ..) = send(&app, Method::POST, "/api/contacts/0491570006/facts", &[], Some(json!({"text": "Likes tea", "by": "owner"}))).await;
        assert_eq!(status, StatusCode::CREATED);
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/0491570006/facts", &[], Some(json!({"text": "x", "by": "caller"}))).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("bad_by")));
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/0491570006/facts", &[], Some(json!({"text": "z".repeat(301)}))).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("too_long")));

        // Forgotten by place, guarded by what it says.
        let (status, v, ..) = send(&app, Method::DELETE, "/api/contacts/0491570006/facts/0?text=Likes%20tea", &[], None).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("fact_changed")));
        let (status, v, ..) = send(&app, Method::DELETE, "/api/contacts/0491570006/facts/0?text=Has%20a%20dog%20called%20Max", &[], None).await;
        assert_eq!((status, v["forgotten"]["text"].as_str()), (StatusCode::OK, Some("Has a dog called Max")));
        assert_eq!(v["contact"]["facts"].as_array().unwrap().len(), 1);
        let (status, ..) = send(&app, Method::DELETE, "/api/contacts/0491570006/facts/7", &[], None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // The list, searched.
        send(&app, Method::PUT, "/api/contacts/0400000001", &[], Some(json!({"name": "Sam"}))).await;
        let (_, v, ..) = send(&app, Method::GET, "/api/contacts?q=texts", &[], None).await;
        assert_eq!((v["contacts"].as_array().unwrap().len(), v["contacts"][0]["name"].as_str(), v["total"].as_u64()), (1, Some("Lance"), Some(2)));

        // The export: a file, not a contact called "export.csv".
        let (status, _, head, bytes) = send(&app, Method::GET, "/api/contacts/export.csv", &[], None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(head[header::CONTENT_TYPE], "text/csv; charset=utf-8");
        let disposition = head[header::CONTENT_DISPOSITION].to_str().unwrap();
        assert!(disposition.starts_with("attachment; filename=\"oaiy-contacts-") && disposition.ends_with(".csv\""), "{disposition}");
        assert_eq!(&bytes[..3], [0xEF, 0xBB, 0xBF], "UTF-8 with a byte-order mark");
        assert!(String::from_utf8(bytes).unwrap().contains("Lance,491570006,Prefers texts,Likes tea\r\n"));

        // The import: a preview first, then the one write.
        let csv = "name,mobile\nKim,0411 222 333\nNo number,\n";
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/import", &[], Some(json!({"csv": csv, "preview": true}))).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!((v["preview"].as_bool(), v["added"].as_u64(), v["skipped"].as_u64(), v["country"].as_str()), (Some(true), Some(1), Some(1), Some("AU")));
        assert_eq!(v["skips"][0]["reason"], "no_number");
        let (_, v, ..) = send(&app, Method::GET, "/api/contacts", &[], None).await;
        assert_eq!(v["total"], 2, "a preview writes nothing");
        let (_, v, ..) = send(&app, Method::POST, "/api/contacts/import", &[], Some(json!({"csv": csv, "country": "AU"}))).await;
        assert_eq!((v["preview"].as_bool(), v["added"].as_u64()), (Some(false), Some(1)));
        let (_, v, ..) = send(&app, Method::GET, "/api/contacts/0411222333", &[], None).await;
        assert_eq!((v["name"].as_str(), v["number"].as_str(), v["nameBy"].as_str()), (Some("Kim"), Some("+61411222333"), Some("owner")));
        let (status, v, ..) = send(&app, Method::POST, "/api/contacts/import", &[], Some(json!({"csv": csv, "country": "Mars"}))).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("bad_country")));

        // Deleted.
        let (status, ..) = send(&app, Method::DELETE, "/api/contacts/%2B61491570006", &[], None).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (status, ..) = send(&app, Method::DELETE, "/api/contacts/0491570006", &[], None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn the_contacts_are_closed_to_a_stranger_and_open_to_the_token_and_the_window() {
        let d = Dir::new("gate");
        d.store().set("0491570006", Change { name: Some("Lance".into()), ..Change::default() }).unwrap();
        let body = || Some(json!({"name": "Mallory"}));
        // A headless server: the token or nothing.
        let headless = crate::http::guarded_for_tests(router(d.store()), Some("desk-token".into()), false);
        for (m, path, b) in [(Method::GET, "/api/contacts", None), (Method::GET, "/api/contacts/0491570006", None), (Method::GET, "/api/contacts/export.csv", None), (Method::PUT, "/api/contacts/0491570006", body())] {
            let (status, ..) = send(&headless, m.clone(), path, &[], b.clone()).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{m} {path} without the token");
            let (status, ..) = send(&headless, m.clone(), path, &[("Authorization", "Bearer wrong")], b.clone()).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{m} {path} with a wrong token");
        }
        let (status, v, ..) = send(&headless, Method::GET, "/api/contacts", &[("Authorization", "Bearer desk-token")], None).await;
        assert_eq!((status, v["total"].as_u64()), (StatusCode::OK, Some(1)));

        // The desktop's window: its own origin reads and changes; a web page, or a caller with no origin, reads nothing.
        let gui = crate::http::guarded_for_tests(router(d.store()), None, true);
        for origin in ["https://evil.example", "null"] {
            let (status, ..) = send(&gui, Method::GET, "/api/contacts", &[("Origin", origin)], None).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "a read from {origin}");
            let (status, ..) = send(&gui, Method::PUT, "/api/contacts/0491570006", &[("Origin", origin)], body()).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "a change from {origin}");
        }
        let (status, ..) = send(&gui, Method::GET, "/api/contacts/0491570006", &[], None).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "a restricted read needs an origin it knows");
        let (status, ..) = send(&gui, Method::DELETE, "/api/contacts/0491570006", &[], None).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "a change fails closed with no origin");
        let (status, v, ..) = send(&gui, Method::GET, "/api/contacts/0491570006", &[("Origin", "tauri://localhost")], None).await;
        assert_eq!((status, v["name"].as_str()), (StatusCode::OK, Some("Lance")));
        let (status, v, ..) = send(&gui, Method::PUT, "/api/contacts/0491570006", &[("Origin", "tauri://localhost")], Some(json!({"notes": "From the window"}))).await;
        assert_eq!((status, v["notes"].as_str()), (StatusCode::OK, Some("From the window")));
        assert_eq!(d.store().get("0491570006").unwrap().name, "Lance", "Mallory never got in");
    }
}
