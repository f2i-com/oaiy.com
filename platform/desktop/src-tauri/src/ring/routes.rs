//! The ring over HTTP, for the dashboard (the owner's settings) and the app.
//!
//!   GET /api/ring/settings  → {settings, features}: what the owner chose, and what the
//!                              receptionist may do because of it (`features.transfer`,
//!                              `features.messages`)
//!   PUT /api/ring/settings  → the same, after the change: any of the settings (see
//!                              [`super::settings::RingSettings`]), `quietHours` and `limits`
//!                              member by member; a name that is not a setting, or a value of
//!                              the wrong kind, is a 400 `bad_settings` and changes nothing
//!   GET /api/ring/preview   → what a caller who asks for the owner would get right now, in words, and what the phone
//!                              plugin has done with the calls while transfers were on (`preview.rs`); nothing is counted
//!   GET /api/ring/active    → {rings}: whom the receptionist is trying to reach the owner for now
//!   POST /api/ring/active/:id/respond {action} → {ok, note}: the owner answers in the dialog
//!
//! Gated like `/api/voice/*` (`http.rs`): reading is a restricted read, changing takes the
//! privileged gate. Nothing here needs the phone: the owner can set this up first.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::Ring;

pub fn router(ring: Arc<Ring>) -> Router {
    Router::new()
        .route("/api/ring/settings", get(get_settings).put(put_settings))
        .route("/api/ring/preview", get(preview))
        .route("/api/ring/active", get(active))
        .route("/api/ring/active/:id/respond", post(respond))
        .route("/api/ring/notices/:id/dismiss", post(dismiss))
        .with_state(ring)
}

/// `GET /api/ring/preview` → [`super::Preview`]: what would happen to a caller who asked for the owner now.
async fn preview(State(ring): State<Arc<Ring>>) -> Json<super::Preview> {
    Json(ring.preview())
}

/// `GET /api/ring/active` → `{rings, notices}`: the callers the receptionist is trying to reach the owner for now (see
/// [`super::session::ActiveRing`]), and the callers who asked for the owner when no device was set up to take a
/// transfer ([`super::session::Notice`]); the dashboard's dialog asks every second while it is visible.
async fn active(State(ring): State<Arc<Ring>>) -> Json<Value> {
    Json(json!({"rings": ring.active(), "notices": ring.notices()}))
}

/// `POST /api/ring/notices/:id/dismiss` → `{ok}`: the owner has read a notice. A notice that is gone is a 404.
async fn dismiss(State(ring): State<Arc<Ring>>, Path(id): Path<String>) -> Response {
    if ring.dismiss_notice(&id) {
        Json(json!({"ok": true})).into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(json!({"error": {"code": "no_notice", "message": "that notice is gone"}}))).into_response()
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Respond {
    action: String,
}

/// `POST /api/ring/active/:id/respond {action}` (`decline` or `message`) → `{ok, note}`: the phone is asked, on the call's
/// own stream, to withdraw the request (`transfer_cancel`), and its answer decides what the caller hears. The ring shows
/// as stopping until then. Taking the call is the Companion's: there is no accept. A ring that is over is a 404 `no_ring`.
async fn respond(State(ring): State<Arc<Ring>>, Path(id): Path<String>, Json(body): Json<Respond>) -> Response {
    let Some(action) = super::session::Action::parse(&body.action) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": {"code": "bad_action", "message": "the action is decline or message"}}))).into_response();
    };
    match ring.respond(&id, action).await {
        Ok(r) => Json(json!({"ok": r.ok, "note": r.note})).into_response(),
        Err(e) => (StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(json!({"error": {"code": e.code, "message": e.message}}))).into_response(),
    }
}

fn shown(ring: &Ring) -> Value {
    let settings = ring.settings.get();
    json!({"settings": settings, "features": ring.features(), "loadProblem": ring.settings.load_problem()})
}

async fn get_settings(State(ring): State<Arc<Ring>>) -> Json<Value> {
    Json(shown(&ring))
}

async fn put_settings(State(ring): State<Arc<Ring>>, Json(change): Json<Value>) -> Response {
    match ring.change_settings(&change) {
        Ok(()) => Json(shown(&ring)).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"code": "bad_settings", "message": e.to_string()}}))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::TempDir;

    async fn serve(ring: Arc<Ring>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(ring)).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_settings_say_when_their_file_could_not_be_used_and_where_it_is_kept() {
        let dir = TempDir::new("ring-routes-problem");
        std::fs::write(dir.0.join(crate::ring::settings::FILE_NAME), [0xC3, 0x28]).unwrap();
        let base = serve(Ring::open(&dir.0)).await;
        let read: Value = reqwest::get(format!("{base}/api/ring/settings")).await.unwrap().json().await.unwrap();
        assert_eq!(read["settings"]["enabled"], false);
        assert!(read["loadProblem"].as_str().is_some_and(|p| p.contains("ring.json.corrupt")), "{read}");
        let clean = TempDir::new("ring-routes-no-problem");
        let base = serve(Ring::open(&clean.0)).await;
        let read: Value = reqwest::get(format!("{base}/api/ring/settings")).await.unwrap().json().await.unwrap();
        assert!(read["loadProblem"].is_null(), "{read}");
    }

    #[tokio::test]
    async fn the_settings_are_read_changed_and_kept() {
        let dir = TempDir::new("ring-routes");
        let base = serve(Ring::open(&dir.0)).await;
        let client = reqwest::Client::new();
        let read: Value = client.get(format!("{base}/api/ring/settings")).send().await.unwrap().json().await.unwrap();
        assert_eq!(read["settings"]["enabled"], false);
        assert_eq!(read["features"], json!({"transfer": false, "messages": false}), "off until the owner turns it on");
        let put = client.put(format!("{base}/api/ring/settings")).json(&json!({"enabled": true, "ringSeconds": 50})).send().await.unwrap();
        assert_eq!(put.status(), 200);
        let put: Value = put.json().await.unwrap();
        assert_eq!((put["settings"]["enabled"].clone(), put["settings"]["ringSeconds"].clone(), put["settings"]["takeMessages"].clone()), (json!(true), json!(50), json!(true)));
        assert_eq!(put["features"], json!({"transfer": true, "messages": true}), "taking messages comes with transfers");
        // Kept: a fresh desktop on the same folder reads it back.
        let again = Ring::open(&dir.0);
        assert!(again.settings.get().enabled);
    }

    #[tokio::test]
    async fn a_bad_change_is_a_400_and_changes_nothing() {
        let dir = TempDir::new("ring-routes-bad");
        let ring = Ring::open(&dir.0);
        let base = serve(ring.clone()).await;
        let client = reqwest::Client::new();
        for body in [json!({"enabled": true, "surprise": 1}), json!({"phoneRing": "sometimes"}), json!({"ringSeconds": "long"}), json!([1, 2])] {
            let resp = client.put(format!("{base}/api/ring/settings")).json(&body).send().await.unwrap();
            assert_eq!(resp.status(), 400, "{body}");
            let e: Value = resp.json().await.unwrap();
            assert_eq!(e["error"]["code"], "bad_settings");
        }
        assert!(!ring.settings.get().enabled);
        assert!(!dir.0.join(super::super::settings::FILE_NAME).exists(), "nothing was written");
    }

    /// One request to `app` as the gate sees it, and what came back.
    async fn send(app: &Router, method: axum::http::Method, path: &str, headers: &[(&str, &str)], body: Option<Value>) -> (StatusCode, Value) {
        use tower::ServiceExt;
        let mut request = axum::http::Request::builder().method(method).uri(path);
        for (k, v) in headers {
            request = request.header(*k, *v);
        }
        if body.is_some() {
            request = request.header("content-type", "application/json");
        }
        let response = app.clone().oneshot(request.body(body.map_or_else(axum::body::Body::empty, |b| axum::body::Body::from(b.to_string()))).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    /// Who can turn transfers on, decline a ring or delete a message, as the desktop serves the routes (behind the gate `http.rs` puts in front
    /// of everything, the class of `/api/voice/*` and `/api/contacts`): the token, or OAIY's own window; not a web page, not a caller with no
    /// origin, and a headless server trusts the token alone. It is the strictest class the routes have; a program on this computer that
    /// presents the window's origin, as any local program can, is inside it, as it is for every route in that class.
    #[tokio::test]
    async fn the_ring_and_the_messages_are_closed_to_a_stranger_and_open_to_the_token_and_the_window() {
        use axum::http::Method;
        let dir = TempDir::new("ring-gate");
        let ring = Ring::open(&dir.0);
        let store = crate::messages::Store::default();
        let routes = || router(ring.clone()).merge(crate::messages::routes::router(store.clone()));
        let turn_on = || Some(json!({"enabled": true}));
        let every = [
            (Method::GET, "/api/ring/settings", None),
            (Method::PUT, "/api/ring/settings", turn_on()),
            (Method::GET, "/api/ring/preview", None),
            (Method::GET, "/api/ring/active", None),
            (Method::POST, "/api/ring/active/assist_1/respond", Some(json!({"action": "decline"}))),
            (Method::POST, "/api/ring/notices/notice_1/dismiss", None),
            (Method::GET, "/api/messages", None),
            (Method::PATCH, "/api/messages/msg_1", Some(json!({"state": "handled"}))),
            (Method::DELETE, "/api/messages/msg_1", None),
        ];

        // A headless server: the token or nothing, whatever origin is presented.
        let headless = crate::http::guarded_for_tests(routes(), Some("desk-token".into()), false);
        for (m, path, body) in &every {
            for headers in [vec![], vec![("Authorization", "Bearer wrong")], vec![("Origin", "tauri://localhost")], vec![("Origin", "tauri://localhost"), ("Authorization", "Bearer wrong")]] {
                let (status, ..) = send(&headless, m.clone(), path, &headers, body.clone()).await;
                assert_eq!(status, StatusCode::FORBIDDEN, "{m} {path} {headers:?}");
            }
            let (status, ..) = send(&headless, m.clone(), path, &[("Authorization", "Bearer desk-token")], body.clone()).await;
            assert_ne!(status, StatusCode::FORBIDDEN, "{m} {path} with the token");
        }
        assert!(ring.settings.get().enabled, "the token turned transfers on");
        ring.change_settings(&json!({"enabled": false})).unwrap();

        // The desktop's window: a web page, a page whose address only ends like ours, and a caller with no origin are all shut out, of
        // reads and changes alike, and nothing was turned on.
        let gui = crate::http::guarded_for_tests(routes(), None, true);
        for origin in ["https://evil.example", "null", "https://oaiy.com.evil.example", "https://evil.example/oaiy.com", "http://tauri.localhost.evil.example", "tauri://localhost.evil.example"] {
            for (m, path, body) in &every {
                let (status, ..) = send(&gui, m.clone(), path, &[("Origin", origin)], body.clone()).await;
                assert_eq!(status, StatusCode::FORBIDDEN, "{m} {path} from {origin}");
            }
        }
        for (m, path, body) in every.iter().filter(|(m, ..)| m != Method::GET) {
            let (status, ..) = send(&gui, m.clone(), path, &[], body.clone()).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{m} {path}: a change fails closed with no origin");
        }
        assert!(!ring.settings.get().enabled, "no stranger turned transfers on");
        // The window reads and changes: it turns transfers on, declines a ring that is not there, and finds no message to delete.
        let window = [("Origin", "tauri://localhost")];
        let (status, v) = send(&gui, Method::PUT, "/api/ring/settings", &window, turn_on()).await;
        assert_eq!((status, v["settings"]["enabled"].clone()), (StatusCode::OK, json!(true)));
        let (status, v) = send(&gui, Method::POST, "/api/ring/active/assist_1/respond", &window, Some(json!({"action": "decline"}))).await;
        assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("no_ring")));
        let (status, v) = send(&gui, Method::DELETE, "/api/messages/msg_1", &window, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{v}");
        let (status, v) = send(&gui, Method::GET, "/api/ring/preview", &window, None).await;
        assert_eq!((status, v["enabled"].clone()), (StatusCode::OK, json!(true)));
    }
}
