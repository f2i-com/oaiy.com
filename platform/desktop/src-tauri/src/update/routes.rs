//! The update routes on the local API.
//!
//! - `GET  /api/update/status`: what the window reads (see [`super::updater::Status`]). Read-only.
//!   On the headless server it is open like `/api/health` (it says which version is running and whether a
//!   newer one exists, which health already half says), so that can be asked without a token. On the desktop
//!   it is a restricted read, like the calls (`http::origin_guard`): its blockers say whether a phone call is live.
//! - `POST /api/update/check`: look for a newer release now. It is a plain GET of the release feed,
//!   at most once every 30 seconds however it is asked for (this route, the window's button and the tray
//!   share one limit), and privileged like the other routes that act (the desktop's own window, or the
//!   token), so a page the owner happens to have open cannot make OAIY phone GitHub.
//! - `POST /api/update/agent-flushed {nonce}`: the Agent page saying it saved its work when the desktop
//!   asked it to before an install. Only the nonce that was given to the page counts.
//!
//! Nothing here downloads or installs. The desktop does those through its own commands, which take
//! no address and no path and answer only the dashboard's own webview (`update::gui`); the headless
//! server has no way to at all.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::updater::{CheckRefusal, UpdaterHandle};

pub fn router(updater: UpdaterHandle) -> Router {
    Router::new()
        .route("/api/update/status", get(status))
        .route("/api/update/check", post(check))
        .route("/api/update/agent-flushed", post(agent_flushed))
        .with_state(updater)
}

async fn status(State(updater): State<UpdaterHandle>) -> axum::response::Response {
    // What is asked of the app (the engines, in the desktop) may block for a moment: not on the async threads.
    match tokio::task::spawn_blocking(move || updater.status()).await {
        Ok(status) => Json(status).into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"code": "status_failed", "message": "The update status could not be read."}}))).into_response(),
    }
}

async fn check(State(updater): State<UpdaterHandle>) -> axum::response::Response {
    match updater.check().await {
        Ok(()) => status(State(updater)).await,
        Err(refusal) => {
            let (code, http) = match &refusal {
                CheckRefusal::TooSoon { .. } => ("too_soon", StatusCode::TOO_MANY_REQUESTS),
                CheckRefusal::AlreadyChecking => ("already_checking", StatusCode::CONFLICT),
                CheckRefusal::Busy(_) => ("busy", StatusCode::CONFLICT),
            };
            let retry = match &refusal {
                CheckRefusal::TooSoon { retry_in } => Some(*retry_in),
                _ => None,
            };
            let now = { let updater = updater.clone(); tokio::task::spawn_blocking(move || updater.status()).await.ok() };
            let mut response = (http, Json(json!({"error": {"code": code, "message": refusal.to_string()}, "retryAfterSeconds": retry, "status": now}))).into_response();
            if let Some(seconds) = retry {
                if let Ok(value) = axum::http::HeaderValue::from_str(&seconds.to_string()) {
                    response.headers_mut().insert(axum::http::header::RETRY_AFTER, value);
                }
            }
            response
        }
    }
}

#[derive(Deserialize)]
struct Flushed {
    nonce: String,
}

async fn agent_flushed(State(updater): State<UpdaterHandle>, Json(body): Json<Flushed>) -> axum::response::Response {
    if updater.flush_ack(&body.nonce) {
        Json(json!({"ok": true})).into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(json!({"error": {"code": "not_waiting", "message": "Nothing is waiting for that."}}))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::{FeedSource, Updater};
    use axum::body::Body;
    use axum::http::{Method, Request};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;
    use tower::ServiceExt as _;

    const SIG: &str = "c2lnbmF0dXJl";

    fn feed(version: &str) -> String {
        json!({
            "version": version, "notes": format!("Notes for {version}."), "pub_date": "2026-10-01T02:03:04Z",
            "platforms": {
                "windows-x86_64": {"signature": SIG, "url": format!("https://github.com/f2i-com/oaiy.com/releases/download/v{version}/oaiy-desktop-{version}-windows-x64-setup.exe")},
                "linux-x86_64": {"signature": SIG, "url": format!("https://github.com/f2i-com/oaiy.com/releases/download/v{version}/oaiy-desktop-{version}-linux-x86_64.AppImage")}
            }
        }).to_string()
    }

    /// What a stub feed server does with a request for the feed.
    #[derive(Clone)]
    enum Serve {
        Body(String),
        Status(u16),
        /// A body of this many bytes, sent in chunks with no Content-Length.
        Streamed(usize),
        Redirect(&'static str),
        Loop,
    }

    /// A local server standing in for GitHub: returns its address and how many requests it has had.
    async fn stub(serve: Serve) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let base = Arc::new(std::sync::Mutex::new(String::new()));
        let base_for_handler = base.clone();
        let app = Router::new().route("/latest.json", get(move || {
            let (serve, counter, base) = (serve.clone(), counter.clone(), base_for_handler.clone());
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                match serve {
                    Serve::Body(body) => ([(axum::http::header::CONTENT_TYPE, "application/json")], body).into_response(),
                    Serve::Status(code) => StatusCode::from_u16(code).unwrap().into_response(),
                    Serve::Streamed(n) => {
                        let chunk = vec![b' '; 16 * 1024];
                        let chunks = n / chunk.len() + 1;
                        let stream = futures_util::stream::iter((0..chunks).map(move |_| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(chunk.clone()))));
                        Body::from_stream(stream).into_response()
                    }
                    Serve::Redirect(to) => (StatusCode::FOUND, [(axum::http::header::LOCATION, format!("{}{to}", base.lock().unwrap()))]).into_response(),
                    Serve::Loop => (StatusCode::FOUND, [(axum::http::header::LOCATION, format!("{}/latest.json", base.lock().unwrap()))]).into_response(),
                }
            }
        })).route("/real.json", get({
            let hits = hits.clone();
            move || {
                let hits = hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    ([(axum::http::header::CONTENT_TYPE, "application/json")], feed("0.2.0"))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        *base.lock().unwrap() = address.clone();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (address, hits)
    }

    /// An updater at 0.1.0 (Windows x86_64) reading `url`, and the router over it.
    fn app_at(url: &str) -> (Router, UpdaterHandle) {
        let updater = Updater::new("0.1.0", FeedSource { url: url.into(), insecure: true }, Instant::now());
        updater.set_platform_key(Some("windows-x86_64"));
        (router(updater.clone()), updater)
    }

    async fn call(router: &Router, method: Method, path: &str, body: Option<serde_json::Value>) -> (StatusCode, axum::http::HeaderMap, serde_json::Value) {
        let mut request = Request::builder().method(method).uri(path);
        let body = match body {
            Some(json) => {
                request = request.header("content-type", "application/json");
                Body::from(json.to_string())
            }
            None => Body::empty(),
        };
        let response = router.clone().oneshot(request.body(body).unwrap()).await.unwrap();
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        (parts.status, parts.headers, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test]
    async fn a_newer_release_in_the_feed_is_available_with_its_notes_and_date() {
        let (address, hits) = stub(Serve::Body(feed("0.2.0"))).await;
        let (app, _) = app_at(&format!("{address}/latest.json"));
        let (code, _, before) = call(&app, Method::GET, "/api/update/status", None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(before["state"], "idle");
        assert_eq!(hits.load(Ordering::SeqCst), 0, "asking for the status reads no feed");

        let (code, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(code, StatusCode::OK, "{status}");
        assert_eq!(status["state"], "available");
        assert_eq!(status["latestVersion"], "0.2.0");
        assert_eq!(status["currentVersion"], "0.1.0");
        assert_eq!(status["notes"], "Notes for 0.2.0.");
        assert_eq!(status["publishedAt"], "2026-10-01T02:03:04Z");
        assert!(status["lastCheckedAt"].is_string());
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The status then says the same.
        let (_, _, after) = call(&app, Method::GET, "/api/update/status", None).await;
        assert_eq!(after["state"], "available");
        assert_eq!(after["channel"], "stable");
    }

    #[tokio::test]
    async fn the_same_or_an_older_version_in_the_feed_is_up_to_date() {
        for served in ["0.1.0", "0.0.9", "0.1.0-beta.1"] {
            let (address, _) = stub(Serve::Body(feed(served))).await;
            let (app, _) = app_at(&format!("{address}/latest.json"));
            let (code, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
            assert_eq!(code, StatusCode::OK);
            assert_eq!(status["state"], "upToDate", "feed {served}: {status}");
            assert_eq!(status["latestVersion"], served);
            assert!(status["error"].is_null());
        }
    }

    #[tokio::test]
    async fn a_feed_that_is_down_is_a_failed_check_in_plain_words() {
        // Nothing listening: bind a port, then let it go.
        let port = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); l.local_addr().unwrap().port() };
        let (app, _) = app_at(&format!("http://127.0.0.1:{port}/latest.json"));
        let (code, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(status["state"], "failed");
        assert_eq!(status["failedDuring"], "check");
        assert!(status["error"].as_str().unwrap().starts_with("Could not reach the update server."));

        // A server that answers with an error.
        for (served, words) in [(500u16, "answered 500"), (404, "No update information is published")] {
            let (address, _) = stub(Serve::Status(served)).await;
            let (app, _) = app_at(&format!("{address}/latest.json"));
            let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
            assert_eq!(status["state"], "failed");
            assert!(status["error"].as_str().unwrap().contains(words), "{status}");
        }
    }

    #[tokio::test]
    async fn a_malformed_or_oversized_feed_is_refused_and_never_acted_on() {
        let oversized = feed("0.2.0").replace("\"Notes", &format!("\"x\":\"{}\",\"Notes", "y".repeat(300 * 1024)));
        for (name, served, words) in [
            ("html", Serve::Body("<html>maintenance</html>".into()), "not valid JSON"),
            ("missing platforms", Serve::Body(r#"{"version":"0.2.0"}"#.into()), "no \"platforms\""),
            ("wrong types", Serve::Body(r#"{"version":2,"platforms":{}}"#.into()), "wrong kind of value"),
            ("bad version", Serve::Body(feed("banana")), "not one"),
            ("oversized body", Serve::Body(oversized), "larger than expected"),
            ("streamed with no length", Serve::Streamed(400 * 1024), "larger than expected"),
        ] {
            let (address, _) = stub(served).await;
            let (app, _) = app_at(&format!("{address}/latest.json"));
            let (code, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
            assert_eq!(code, StatusCode::OK, "{name}");
            assert_eq!(status["state"], "failed", "{name}: {status}");
            assert!(status["error"].as_str().unwrap().contains(words), "{name}: {status}");
            assert!(status["latestVersion"].is_null(), "{name}: nothing from the feed is kept");
        }
    }

    #[tokio::test]
    async fn a_second_check_within_thirty_seconds_is_refused_and_reads_no_feed() {
        let (address, hits) = stub(Serve::Body(feed("0.2.0"))).await;
        let (app, updater) = app_at(&format!("{address}/latest.json"));
        let (code, _, _) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(code, StatusCode::OK);
        let (code, headers, body) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(code, StatusCode::TOO_MANY_REQUESTS, "{body}");
        assert_eq!(body["error"]["code"], "too_soon");
        let retry = body["retryAfterSeconds"].as_u64().unwrap();
        assert!((1..=30).contains(&retry), "{retry}");
        assert_eq!(headers.get("retry-after").unwrap().to_str().unwrap(), retry.to_string());
        assert_eq!(body["status"]["state"], "available", "the answer still says where things stand");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "the refused check read nothing");
        // The limit is the updater's, not the route's: the window's button and the tray share it.
        assert!(matches!(updater.check().await, Err(CheckRefusal::TooSoon { .. })));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_redirect_is_followed_and_a_loop_of_them_is_not() {
        let (address, _) = stub(Serve::Redirect("/real.json")).await;
        let (app, _) = app_at(&format!("{address}/latest.json"));
        let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(status["state"], "available", "{status}");

        let (address, hits) = stub(Serve::Loop).await;
        let (app, _) = app_at(&format!("{address}/latest.json"));
        let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(status["state"], "failed");
        assert!(status["error"].as_str().unwrap().contains("too many times"), "{status}");
        assert!(hits.load(Ordering::SeqCst) <= 7, "stopped after {} requests", hits.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_release_build_only_reads_https_and_so_never_the_stub() {
        let (address, hits) = stub(Serve::Body(feed("0.2.0"))).await;
        let updater = Updater::new("0.1.0", FeedSource { url: format!("{address}/latest.json"), insecure: false }, Instant::now());
        let app = router(updater);
        let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(status["state"], "failed");
        assert!(status["error"].as_str().unwrap().contains("not an https address"));
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_platform_the_feed_has_no_release_for_is_no_update_and_not_an_error() {
        let (address, _) = stub(Serve::Body(feed("0.2.0"))).await;
        let (app, updater) = app_at(&format!("{address}/latest.json"));
        updater.set_platform_key(None);
        let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(status["state"], "idle");
        assert!(status["error"].is_null());
        assert!(status["note"].as_str().unwrap().contains("no release of it for this platform"));
    }

    #[tokio::test]
    async fn the_headless_server_is_notify_only_it_says_a_release_exists_and_offers_nothing_to_install() {
        let (address, _) = stub(Serve::Body(feed("0.2.0"))).await;
        let (app, _) = app_at(&format!("{address}/latest.json"));
        let (_, _, status) = call(&app, Method::POST, "/api/update/check", None).await;
        assert_eq!(status["state"], "available");
        assert_eq!(status["canAutoUpdate"], false);
        assert!(status["manualReason"].as_str().unwrap().contains("headless server"));
        assert_eq!(status["blockers"], json!([]));
        assert_eq!(status["manualUrl"], "https://github.com/f2i-com/oaiy.com/releases/latest");
    }

    #[tokio::test]
    async fn the_routes_take_no_address_and_the_check_reads_only_the_feed_it_was_built_with() {
        // A request cannot change where the feed is read from: there is nowhere to put an address.
        let (address, hits) = stub(Serve::Body(feed("0.2.0"))).await;
        let (app, _) = app_at(&format!("{address}/latest.json"));
        let (code, _, _) = call(&app, Method::POST, "/api/update/check?url=http://evil.example/x.json", Some(json!({"url": "http://evil.example/x.json", "feed": "http://evil.example/x.json"}))).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        for (method, path) in [(Method::PUT, "/api/update/feed"), (Method::POST, "/api/update/download"), (Method::POST, "/api/update/install"), (Method::POST, "/api/update/feed")] {
            let (code, _, _) = call(&app, method.clone(), path, Some(json!({"url": "http://evil.example/x.json"}))).await;
            assert_eq!(code, StatusCode::NOT_FOUND, "{method} {path} does not exist: installing is a command of the dashboard's own window");
        }
    }

    #[tokio::test]
    async fn the_agents_flush_answer_completes_only_the_wait_it_was_given_the_nonce_of() {
        let (app, updater) = app_at("http://127.0.0.1:1/latest.json");
        let (code, _, _) = call(&app, Method::POST, "/api/update/agent-flushed", Some(json!({"nonce": "anything"}))).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        let (nonce, rx) = updater.arm_flush();
        let (code, _, _) = call(&app, Method::POST, "/api/update/agent-flushed", Some(json!({"nonce": "wrong"}))).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        assert!(rx.try_recv().is_err());
        let (code, _, body) = call(&app, Method::POST, "/api/update/agent-flushed", Some(json!({"nonce": nonce}))).await;
        assert_eq!((code, &body["ok"]), (StatusCode::OK, &json!(true)));
        assert!(rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn the_guard_lets_the_dashboard_check_and_turns_a_stranger_away_and_the_headless_status_needs_no_token() {
        use crate::http::guarded_for_tests;
        let (address, _) = stub(Serve::Body(feed("0.2.0"))).await;
        let (inner, _) = app_at(&format!("{address}/latest.json"));
        let post = |origin: Option<&str>, bearer: Option<&str>| {
            let mut r = Request::builder().method(Method::POST).uri("/api/update/check");
            if let Some(o) = origin { r = r.header("origin", o); }
            if let Some(b) = bearer { r = r.header("authorization", format!("Bearer {b}")); }
            r.body(Body::empty()).unwrap()
        };
        // The desktop (gui mode): its own window is let in; another page on this computer is not.
        let gui = guarded_for_tests(inner.clone(), None, true);
        assert_eq!(gui.clone().oneshot(post(Some("https://evil.example"), None)).await.unwrap().status(), StatusCode::FORBIDDEN);
        assert_eq!(gui.clone().oneshot(post(None, None)).await.unwrap().status(), StatusCode::FORBIDDEN);
        assert_eq!(gui.clone().oneshot(post(Some("http://tauri.localhost"), None)).await.unwrap().status(), StatusCode::OK);
        // The status says whether a call is live: on the desktop a page that is not OAIY's own may not read it.
        let read = |origin: Option<&str>, bearer: Option<&str>| {
            let mut r = Request::builder().method(Method::GET).uri("/api/update/status");
            if let Some(o) = origin { r = r.header("origin", o); }
            if let Some(b) = bearer { r = r.header("authorization", format!("Bearer {b}")); }
            r.body(Body::empty()).unwrap()
        };
        for stranger in [Some("https://evil.example"), Some("http://evil.localhost:8080"), None] {
            assert_eq!(gui.clone().oneshot(read(stranger, None)).await.unwrap().status(), StatusCode::FORBIDDEN, "{stranger:?}");
        }
        for own in ["http://tauri.localhost", "tauri://localhost", "http://oaiy.localhost", "https://oaiy.com"] {
            assert_eq!(gui.clone().oneshot(read(Some(own), None)).await.unwrap().status(), StatusCode::OK, "{own}");
        }
        let with_token = guarded_for_tests(inner.clone(), Some("s3cret".into()), true);
        assert_eq!(with_token.oneshot(read(None, Some("s3cret"))).await.unwrap().status(), StatusCode::OK);
        // The headless server (no gui mode): the check needs the token, the status does not.
        // (Another updater: checks are rate-limited, and the desktop one has just made one.)
        let (other, _) = app_at(&format!("{address}/latest.json"));
        let headless = guarded_for_tests(other, Some("s3cret".into()), false);
        assert_eq!(headless.clone().oneshot(post(Some("http://tauri.localhost"), None)).await.unwrap().status(), StatusCode::FORBIDDEN);
        assert_eq!(headless.clone().oneshot(post(None, Some("wrong"))).await.unwrap().status(), StatusCode::FORBIDDEN);
        let status = Request::builder().method(Method::GET).uri("/api/update/status").body(Body::empty()).unwrap();
        assert_eq!(headless.clone().oneshot(status).await.unwrap().status(), StatusCode::OK, "like /api/health, the status is public");
        assert_eq!(headless.clone().oneshot(post(None, Some("s3cret"))).await.unwrap().status(), StatusCode::OK);
    }
}
