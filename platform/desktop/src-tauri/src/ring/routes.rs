//! The ring over HTTP, for the dashboard (the owner's settings) and the app.
//!
//!   GET /api/ring/settings  → {settings, features}: what the owner chose, and what the
//!                              receptionist may do because of it (`features.transfer`,
//!                              `features.messages`)
//!   PUT /api/ring/settings  → the same, after the change: any of the settings (see
//!                              [`super::settings::RingSettings`]), `quietHours` and `limits`
//!                              member by member; a name that is not a setting, or a value of
//!                              the wrong kind, is a 400 `bad_settings` and changes nothing
//!
//! Gated like `/api/voice/*` (`http.rs`): reading is a restricted read, changing takes the
//! privileged gate. Nothing here needs the phone: the owner can set this up first.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use super::Ring;

pub fn router(ring: Arc<Ring>) -> Router {
    Router::new().route("/api/ring/settings", get(get_settings).put(put_settings)).with_state(ring)
}

fn shown(ring: &Ring) -> Value {
    let settings = ring.settings.get();
    json!({"settings": settings, "features": ring.features()})
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
}
