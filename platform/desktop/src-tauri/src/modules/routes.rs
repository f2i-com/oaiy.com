//! The modules over HTTP, for the dashboard and the Agent app:
//!
//!   GET /api/modules         → the snapshot (`{revision, modules, contributions, warnings}`),
//!                              with its revision as the ETag: 304 on a matching If-None-Match
//!   GET /api/modules/events  → server-sent: the snapshot at once, then again on each change
//!                              (a keep-alive every 15 s)
//!
//! A restricted read, like `/api/plugins`, and never with a plugin's directory.

use std::convert::Infallible;
use std::time::Duration;

use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};

pub fn router() -> Router {
    Router::new().route("/api/modules", get(list)).route("/api/modules/events", get(events))
}

/// Does `If-None-Match` name `etag` (or `*`)?
fn matches(headers: &HeaderMap, etag: &str) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|t| t.trim().trim_start_matches("W/"))
        .any(|t| t == etag || t == "*")
}

async fn list(headers: HeaderMap) -> Response {
    let snapshot = super::snapshot();
    let etag = super::etag(&snapshot);
    let tag = HeaderValue::from_str(&etag).unwrap_or_else(|_| HeaderValue::from_static("\"0\""));
    let fresh = [(header::ETAG, tag), (header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))];
    if matches(&headers, &etag) {
        return (StatusCode::NOT_MODIFIED, fresh).into_response();
    }
    (fresh, Json(&*snapshot)).into_response()
}

async fn events() -> impl IntoResponse {
    let rx = super::subscribe();
    let stream = futures_util::stream::unfold((true, rx), |(first, mut rx)| async move {
        if !first && rx.changed().await.is_err() {
            return None;
        }
        let snapshot = rx.borrow_and_update().clone();
        let data = serde_json::to_string(&*snapshot).unwrap_or_else(|_| "{}".into());
        Some((Ok::<Event, Infallible>(Event::default().data(data)), (false, rx)))
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn serve() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router()).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_list_has_the_revision_as_its_etag_and_answers_304_when_unchanged() {
        let base = serve().await;
        let client = reqwest::Client::new();
        let first = client.get(format!("{base}/api/modules")).send().await.unwrap();
        assert_eq!(first.status(), 200);
        let etag = first.headers().get("etag").unwrap().to_str().unwrap().to_string();
        let body: serde_json::Value = first.json().await.unwrap();
        assert!(body["modules"].is_array() && body["warnings"].is_array() && body["contributions"].is_object());
        assert!(etag.starts_with(&format!("\"{}-", body["revision"])), "{etag}");
        let again = client.get(format!("{base}/api/modules")).header("if-none-match", &etag).send().await.unwrap();
        assert_eq!(again.status(), 304);
        let other = client.get(format!("{base}/api/modules")).header("if-none-match", "\"999999-0\"").send().await.unwrap();
        assert_eq!(other.status(), 200);
    }

    #[tokio::test]
    async fn the_events_start_with_the_snapshot() {
        use futures_util::StreamExt;
        let base = serve().await;
        let resp = reqwest::Client::new().get(format!("{base}/api/modules/events")).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.headers().get("content-type").unwrap().to_str().unwrap().starts_with("text/event-stream"));
        let mut stream = resp.bytes_stream();
        let mut text = String::new();
        while !text.contains("\n\n") {
            let chunk = tokio::time::timeout(Duration::from_secs(5), stream.next()).await.unwrap().unwrap().unwrap();
            text.push_str(&String::from_utf8_lossy(&chunk));
        }
        let data = text.lines().find_map(|l| l.strip_prefix("data:")).unwrap().trim();
        let first: serde_json::Value = serde_json::from_str(data).unwrap();
        assert!(first["revision"].is_u64() && first["modules"].is_array());
    }
}
