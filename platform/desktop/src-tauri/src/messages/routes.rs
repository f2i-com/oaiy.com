//! The messages over HTTP, for the dashboard's Messages page.
//!
//!   GET    /api/messages?state=&q=  → {messages, total, unread}: newest first; `state` is `new`,
//!                                     `seen` or `handled` (all when left out); `q` finds names,
//!                                     numbers written any way and words
//!   GET    /api/messages/:id        → the message; 404 `no_message`
//!   PATCH  /api/messages/:id {state, handledBy?} → the message, marked `new`, `seen` or `handled`
//!   DELETE /api/messages/:id        → 204
//!
//! There is no route that makes a message: only the receptionist's `take_message` tool does, on
//! the call it is answering (`/api/voice/calls/:id/message`), so a page cannot fill the owner's
//! list. All of it is gated like `/api/voice/*` (`http.rs`): reading is a restricted read,
//! changing takes the privileged gate. Messages are the owner's own record: they are read and
//! kept whether or not a plugin provides the phone.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::{Error, State as MessageState, Store};

pub fn router(store: Store) -> Router {
    Router::new().route("/api/messages", get(list)).route("/api/messages/:id", get(one).patch(mark).delete(remove)).with_state(store)
}

fn fail(e: Error) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(json!({"error": {"code": e.code, "message": e.message}}))).into_response()
}

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default)]
    state: String,
    #[serde(default)]
    q: String,
}

async fn list(State(s): State<Store>, Query(q): Query<ListQuery>) -> Response {
    let state = match q.state.trim() {
        "" | "all" => None,
        other => match MessageState::parse(other) {
            Some(state) => Some(state),
            None => return fail(Error { status: 400, code: "bad_state", message: format!("{other:?} is not a state: new, seen or handled") }),
        },
    };
    let messages = s.list(state, &q.q);
    Json(json!({"total": messages.len(), "unread": s.unread(), "messages": messages})).into_response()
}

async fn one(State(s): State<Store>, Path(id): Path<String>) -> Response {
    match s.get(&id) {
        Some(m) => Json(m).into_response(),
        None => fail(Error { status: 404, code: "no_message", message: format!("no message {id:?}") }),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Mark {
    state: String,
    #[serde(default)]
    handled_by: Option<String>,
}

async fn mark(State(s): State<Store>, Path(id): Path<String>, Json(body): Json<Mark>) -> Response {
    let Some(state) = MessageState::parse(body.state.trim()) else {
        return fail(Error { status: 400, code: "bad_state", message: format!("{:?} is not a state: new, seen or handled", body.state) });
    };
    match s.set_state(&id, state, body.handled_by.as_deref().unwrap_or("owner")) {
        Ok(m) => Json(m).into_response(),
        Err(e) => fail(e),
    }
}

async fn remove(State(s): State<Store>, Path(id): Path<String>) -> Response {
    match s.remove(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => fail(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::NewMessage;
    use serde_json::Value;

    async fn serve(store: Store) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(store)).await.unwrap() });
        format!("http://{addr}")
    }

    fn new(call: &str, from: &str, words: &str) -> NewMessage {
        NewMessage { call_id: call.into(), from: from.into(), name: "Alex".into(), callback: String::new(), message: words.into(), urgent: false, wants_callback: true }
    }

    #[tokio::test]
    async fn messages_are_listed_marked_and_deleted() {
        let store = Store::in_memory();
        let a = store.add(new("call_1", "+61491570006", "Please ring me about Friday.")).unwrap();
        let b = store.add(new("call_2", "+61491570156", "The gate is jammed.")).unwrap();
        let base = serve(store.clone()).await;
        let client = reqwest::Client::new();

        let all: Value = client.get(format!("{base}/api/messages")).send().await.unwrap().json().await.unwrap();
        assert_eq!((all["total"].clone(), all["unread"].clone()), (json!(2), json!(2)));
        let found: Value = client.get(format!("{base}/api/messages?q=0491%20570%20156")).send().await.unwrap().json().await.unwrap();
        assert_eq!(found["messages"][0]["id"], json!(b.id), "a number is found written any way");
        assert_eq!(found["messages"].as_array().unwrap().len(), 1);

        let seen: Value = client.patch(format!("{base}/api/messages/{}", a.id)).json(&json!({"state": "seen"})).send().await.unwrap().json().await.unwrap();
        assert_eq!(seen["state"], "seen");
        assert!(seen["seenAt"].is_string() && seen["handledAt"].is_null());
        let handled: Value = client.patch(format!("{base}/api/messages/{}", a.id)).json(&json!({"state": "handled"})).send().await.unwrap().json().await.unwrap();
        assert_eq!((handled["state"].clone(), handled["handledBy"].clone()), (json!("handled"), json!("owner")));
        let unread: Value = client.get(format!("{base}/api/messages?state=new")).send().await.unwrap().json().await.unwrap();
        assert_eq!(unread["messages"].as_array().unwrap().len(), 1);
        assert_eq!(unread["unread"], 1);

        assert_eq!(client.delete(format!("{base}/api/messages/{}", b.id)).send().await.unwrap().status(), 204);
        assert_eq!(client.delete(format!("{base}/api/messages/{}", b.id)).send().await.unwrap().status(), 404);
        assert_eq!(client.get(format!("{base}/api/messages/{}", b.id)).send().await.unwrap().status(), 404);
        let left: Value = client.get(format!("{base}/api/messages")).send().await.unwrap().json().await.unwrap();
        assert_eq!(left["total"], 1);
    }

    #[tokio::test]
    async fn a_bad_state_or_body_is_refused_and_a_page_cannot_make_a_message() {
        let store = Store::in_memory();
        let m = store.add(new("call_1", "+61491570006", "Hello.")).unwrap();
        let base = serve(store.clone()).await;
        let client = reqwest::Client::new();
        assert_eq!(client.get(format!("{base}/api/messages?state=done")).send().await.unwrap().status(), 400);
        assert_eq!(client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "done"})).send().await.unwrap().status(), 400);
        assert_eq!(client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "handled", "message": "rewritten"})).send().await.unwrap().status(), 422, "only the state changes: not its words");
        assert_eq!(client.patch(format!("{base}/api/messages/nope")).json(&json!({"state": "seen"})).send().await.unwrap().status(), 404);
        assert_eq!(store.get(&m.id).unwrap().message, "Hello.");
        // There is no route that makes one.
        assert_eq!(client.post(format!("{base}/api/messages")).json(&json!({"message": "spam", "from": "+61491570999"})).send().await.unwrap().status(), 405);
        assert_eq!(store.list(None, "").len(), 1);
    }
}
