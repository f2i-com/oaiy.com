//! The routes: one read-only status route for the dashboard, and the internal routes the Agent
//! page uses to hand over and take back its storage.
//!
//! - `GET /api/backup/status`: when the last backup was made, whether it worked, its size, and whether a
//!   restore waits. Read only.
//! - `POST /api/backup/agent/{id}/part?seq=n` and `POST /api/backup/agent/{id}/done`: the page posts
//!   its export into a session the desktop opened for one backup (id and token given to the page by
//!   the desktop itself).
//! - `GET /api/backup/agent-import`, `GET .../{id}/part/{i}`, `POST .../{id}/undo-part?seq=n`,
//!   `POST .../{id}/undo-done`, `POST .../{id}/done`: the page takes back the storage a restore left for it.
//!
//! **No route creates or restores a backup.** A backup is made, and a restore staged, applied and
//! undone, only by the commands of the dashboard's own window; these routes only carry bytes into
//! or out of a session or an import that such a command has already opened. All of them sit behind
//! the desktop's origin guard (see `http.rs`); the hand-over routes are also for the Agent's own page
//! only (its `Origin` must be the scheme the desktop serves it from, so a paired token or a page of
//! the linked provider gets nothing), and they demand a secret in `X-Backup-Token`: the session's own
//! for an export, and for an import the one the desktop put in the Agent's window (no route returns
//! it, so setting an `Origin` header is not enough).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use super::agent::{self, DonePayload, PartError, PART_SIZE};
use super::state;

#[derive(Clone)]
struct Ctx {
    data_dir: Arc<PathBuf>,
}

fn fail(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

fn part_error(e: PartError) -> Response {
    match e {
        PartError::Unknown => fail(StatusCode::NOT_FOUND, "no such backup session"),
        PartError::Denied => fail(StatusCode::FORBIDDEN, "wrong token"),
        PartError::Sequence => fail(StatusCode::CONFLICT, "a part came out of order"),
        PartError::TooLarge => fail(StatusCode::PAYLOAD_TOO_LARGE, "too large"),
        PartError::Closed => fail(StatusCode::CONFLICT, "the session is closed"),
        PartError::Io => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not store the part"),
    }
}

/// The origins the Agent's own page has, in its window: the scheme this desktop serves it from, or on a Mac its own
/// port while this desktop holds it (`embed::agent_http_origin`).
pub fn is_agent_origin(origin: &str) -> bool {
    matches!(origin, "oaiy://localhost" | "http://oaiy.localhost" | "https://oaiy.localhost") || crate::embed::agent_http_origin().is_some_and(|own| own == origin)
}

/// The hand-over routes are for the Agent's own page and nobody else: a paired token, a page of the
/// linked provider or the dashboard passes the desktop's guard for other routes, but the Agent's
/// storage is whole conversations, and none of them has a reason to ask for it.
fn agent_page_only(headers: &HeaderMap) -> Option<Response> {
    let origin = headers.get("origin").and_then(|v| v.to_str().ok()).unwrap_or("");
    if is_agent_origin(origin) {
        None
    } else {
        Some(fail(StatusCode::FORBIDDEN, "only the Agent's own page may use this"))
    }
}

fn token_of(headers: &HeaderMap) -> String {
    headers.get("x-backup-token").and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
}

#[derive(Deserialize)]
struct PartQuery {
    seq: u32,
}

async fn status(State(ctx): State<Ctx>) -> Response {
    let dir = ctx.data_dir.clone();
    match tokio::task::spawn_blocking(move || state::status(&dir)).await {
        Ok(s) => Json(s).into_response(),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not read the backup status"),
    }
}

async fn export_part(Path(id): Path<String>, Query(q): Query<PartQuery>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let token = token_of(&headers);
    match tokio::task::spawn_blocking(move || agent::receive_part(&id, &token, q.seq, &body)).await {
        Ok(Ok(())) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not store the part"),
    }
}

async fn export_done(Path(id): Path<String>, headers: HeaderMap, Json(done): Json<DonePayload>) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let token = token_of(&headers);
    match tokio::task::spawn_blocking(move || agent::finish(&id, &token, done)).await {
        Ok(Ok(())) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not finish"),
    }
}

async fn import_meta(State(ctx): State<Ctx>, headers: HeaderMap) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    if !agent::page_token_matches(&token_of(&headers)) {
        return fail(StatusCode::FORBIDDEN, "wrong token");
    }
    let dir = ctx.data_dir.clone();
    match tokio::task::spawn_blocking(move || agent::import_meta(&dir)).await {
        Ok(meta) => Json(meta).into_response(),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not read the import"),
    }
}

async fn import_part(State(ctx): State<Ctx>, Path((id, index)): Path<(String, u64)>, headers: HeaderMap) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let (dir, token) = (ctx.data_dir.clone(), token_of(&headers));
    match tokio::task::spawn_blocking(move || agent::import_part(&dir, &id, &token, index)).await {
        Ok(Ok(bytes)) => ([(axum::http::header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not read the part"),
    }
}

async fn undo_part(State(ctx): State<Ctx>, Path(id): Path<String>, Query(q): Query<PartQuery>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let (dir, token) = (ctx.data_dir.clone(), token_of(&headers));
    match tokio::task::spawn_blocking(move || agent::undo_part(&dir, &id, &token, q.seq, &body)).await {
        Ok(Ok(())) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not store the part"),
    }
}

async fn undo_done(State(ctx): State<Ctx>, Path(id): Path<String>, headers: HeaderMap, Json(done): Json<DonePayload>) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let (dir, token) = (ctx.data_dir.clone(), token_of(&headers));
    match tokio::task::spawn_blocking(move || agent::undo_done(&dir, &id, &token, &done)).await {
        Ok(Ok(())) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not finish"),
    }
}

async fn import_done(State(ctx): State<Ctx>, Path(id): Path<String>, headers: HeaderMap, Json(done): Json<agent::ImportReport>) -> Response {
    if let Some(refused) = agent_page_only(&headers) {
        return refused;
    }
    let (dir, token) = (ctx.data_dir.clone(), token_of(&headers));
    match tokio::task::spawn_blocking(move || agent::import_done(&dir, &id, &token, &done)).await {
        Ok(Ok(())) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(Err(e)) => part_error(e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "could not finish"),
    }
}

/// The routes, for the data folder `data_dir`.
pub fn router(data_dir: PathBuf) -> Router {
    let ctx = Ctx { data_dir: Arc::new(data_dir) };
    Router::new()
        .route("/api/backup/status", get(status))
        .route("/api/backup/agent/:id/part", post(export_part))
        .route("/api/backup/agent/:id/done", post(export_done))
        .route("/api/backup/agent-import", get(import_meta))
        .route("/api/backup/agent-import/:id/part/:index", get(import_part))
        .route("/api/backup/agent-import/:id/undo-part", post(undo_part))
        .route("/api/backup/agent-import/:id/undo-done", post(undo_done))
        .route("/api/backup/agent-import/:id/done", post(import_done))
        .layer(DefaultBodyLimit::max(PART_SIZE + 64 * 1024))
        .with_state(ctx)
}

/// Whether a path is one of this module's routes (for the origin guard in `http.rs`): the status, and the Agent page's hand-over of its
/// storage (`/api/backup/agent/...`, `/api/backup/agent-import...`). Not any other path under `/api/backup`: `/api/backup` itself and what
/// the vault design reserved there (`catalog`, `config`, `run`, `restore`, `rollback`, ...) are that design's, with its own rows in the access
/// table, and this guard claims none of them.
pub fn is_backup_path(path: &str) -> bool {
    path == "/api/backup/status" || path == "/api/backup/agent-import" || path.starts_with("/api/backup/agent-import/") || path.starts_with("/api/backup/agent/")
}

/// The one route that only reads and holds nothing an outside page could not learn from the dashboard.
pub fn is_status_path(path: &str) -> bool {
    path == "/api/backup/status"
}

/// Every route this module serves, method and path, for the test that no route can start a backup.
#[cfg(test)]
pub(crate) const ROUTES: &[(&str, &str)] = &[
    ("GET", "/api/backup/status"),
    ("POST", "/api/backup/agent/:id/part"),
    ("POST", "/api/backup/agent/:id/done"),
    ("GET", "/api/backup/agent-import"),
    ("GET", "/api/backup/agent-import/:id/part/:index"),
    ("POST", "/api/backup/agent-import/:id/undo-part"),
    ("POST", "/api/backup/agent-import/:id/undo-done"),
    ("POST", "/api/backup/agent-import/:id/done"),
];
