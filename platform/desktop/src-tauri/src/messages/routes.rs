//! The messages over HTTP, for the dashboard's Messages page.
//!
//!   GET    /api/messages?state=&q=  → {messages, total, unread, notice}: newest first; `notice` is a
//!                                     plain word when new messages are being refused because too many
//!                                     wait for the owner (else null); `state` is `new`,
//!                                     `seen` or `handled` (all when left out); `q` finds names,
//!                                     numbers written any way and words
//!   GET    /api/messages/:id        → the message; 404 `no_message`
//!   PATCH  /api/messages/:id {state}   → the message, marked `new`, `seen` or `handled`; who marked it is
//!                                     the credential that asked (`handledBy`), and never a name in the body
//!   DELETE /api/messages/:id        → 204
//!
//! There is no route that makes a message: only the receptionist's `take_message` tool does, on
//! the call it is answering (`/api/voice/calls/:id/message`), so a page cannot fill the owner's
//! list. All of it is gated like `/api/voice/*` (`http.rs`): reading is a restricted read,
//! changing takes the privileged gate. Messages are the owner's own record: they are read and
//! kept whether or not a plugin provides the phone.

use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::{Error, State as MessageState, Store};
use crate::auth::presets::App;
use crate::auth::principal::{Principal, PrincipalKind};

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
    Json(json!({"total": messages.len(), "unread": s.unread(), "notice": s.notice(), "messages": messages})).into_response()
}

async fn one(State(s): State<Store>, Path(id): Path<String>) -> Response {
    match s.get(&id) {
        Some(m) => Json(m).into_response(),
        None => fail(Error { status: 404, code: "no_message", message: format!("no message {id:?}") }),
    }
}

/// The state to mark a message with, and nothing else: a name in the body ("handledBy") is refused, since who marked a message is not for the
/// caller of the route to say (a paired app that marked one "handled" by "owner (in person)" would have put its words in the owner's record).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mark {
    state: String,
}

/// Who marked a message, as the record says it: the credential that asked. The owner (the dashboard's own credential, a session on the dashboard
/// host, the console, and every request in `legacy` access mode, where whoever passes the guard is the owner as it sees them) is "owner", which is what
/// the record has always said; any other credential is its kind and the label it was made with ("pat: Alex's phone"). Where no guard is in front of
/// the route (a test of the route alone) there is no credential, and it is the owner's.
fn handled_by(principal: Option<&Principal>) -> String {
    let Some(p) = principal else { return "owner".to_string() };
    match (p.kind, p.app) {
        (PrincipalKind::Legacy | PrincipalKind::Console, _) | (PrincipalKind::Desk | PrincipalKind::Session, Some(App::Dash)) => "owner".to_string(),
        (kind, _) => format!("{}: {}", kind.name(), p.label),
    }
}

async fn mark(State(s): State<Store>, Path(id): Path<String>, principal: Option<Extension<Principal>>, Json(body): Json<Mark>) -> Response {
    let Some(state) = MessageState::parse(body.state.trim()) else {
        return fail(Error { status: 400, code: "bad_state", message: format!("{:?} is not a state: new, seen or handled", body.state) });
    };
    match s.set_state(&id, state, &handled_by(principal.as_ref().map(|Extension(p)| p))) {
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
        assert_eq!(client.post(format!("{base}/api/messages")).json(&json!({"message": "spam", "from": "+61491570157"})).send().await.unwrap().status(), 405);
        assert_eq!(store.list(None, "").len(), 1);
    }

    fn credential(kind: PrincipalKind, app: Option<App>, label: &str) -> Principal {
        Principal {
            id: "0123456789abcdef".into(),
            kind,
            label: label.into(),
            scopes: crate::auth::scopes::ScopeSet::empty(),
            origins: Vec::new(),
            app,
            elevated: false,
            chain: Vec::new(),
            persisted: false,
            expires_ms: None,
            preset: None,
            legacy_import: false,
        }
    }

    /// Who marked a message is the credential that asked, never a name the caller of the route gives: the record says "owner" for the owner's own
    /// credentials (as it always did) and the kind and label of any other, and a body that names someone ("handledBy") is refused and changes nothing.
    #[tokio::test]
    async fn who_marked_a_message_is_the_credential_that_asked_and_never_a_name_in_the_body() {
        let store = Store::in_memory();
        let m = store.add(new("call_1", "+61491570006", "Ring me.")).unwrap();
        let mut asked = Vec::new();
        for (who, principal) in [
            ("no credential (the route alone)", None),
            ("the legacy owner", Some(credential(PrincipalKind::Legacy, None, "legacy access"))),
            ("the dashboard's desk", Some(credential(PrincipalKind::Desk, Some(App::Dash), "webview"))),
            ("a session on the dashboard host", Some(credential(PrincipalKind::Session, Some(App::Dash), "Chrome"))),
            ("the console", Some(credential(PrincipalKind::Console, None, "console"))),
            ("a paired app", Some(credential(PrincipalKind::Pat, None, "Alex's phone"))),
            ("the Agent page's desk", Some(credential(PrincipalKind::Desk, Some(App::Agent), "webview"))),
            ("a label far too long", Some(credential(PrincipalKind::Pat, None, &"a very long label ".repeat(10)))),
        ] {
            let app = match principal {
                Some(p) => router(store.clone()).layer(Extension(p)),
                None => router(store.clone()),
            };
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let client = reqwest::Client::new();
            // Marked new again first, so that each is the one that marks it handled.
            client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "new"})).send().await.unwrap();
            let handled: Value = client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "handled"})).send().await.unwrap().json().await.unwrap();
            asked.push((who, handled["handledBy"].as_str().unwrap_or("").to_string()));
            // A name in the body is refused, and nothing changes.
            client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "new"})).send().await.unwrap();
            let named = client.patch(format!("{base}/api/messages/{}", m.id)).json(&json!({"state": "handled", "handledBy": "owner (in person)"})).send().await.unwrap();
            assert_eq!(named.status(), 422, "{who}: the caller does not say who marked it");
            assert_eq!(store.get(&m.id).unwrap().state, crate::messages::State::New, "{who}: nothing changed");
        }
        assert_eq!(
            asked,
            [
                ("no credential (the route alone)", "owner".to_string()),
                ("the legacy owner", "owner".to_string()),
                ("the dashboard's desk", "owner".to_string()),
                ("a session on the dashboard host", "owner".to_string()),
                ("the console", "owner".to_string()),
                ("a paired app", "pat: Alex's phone".to_string()),
                ("the Agent page's desk", "desk: webview".to_string()),
                ("a label far too long", "pat: a very long label a very long label".to_string()),
            ]
        );
    }
}
