//! Flows hand work to the agent: an "Ask the agent" node gives the agent in
//! OAIY's window a task and waits for its reply.
//!
//! - `POST /api/agent/tasks` `{task, from?, waitSeconds?}`: the task goes to the
//!   agent's page (`agent.task` on `GET /api/agent/events`), which answers it in
//!   a session of its own and posts the reply back; the request waits for it
//!   (up to `waitSeconds`, 600 by default) and answers `{id, status: "done",
//!   reply}`, `{id, status: "failed", error}`, or `{id, status: "pending"}` when
//!   the wait ran out (`GET /api/agent/tasks/{id}` then tells how it went).
//! - `POST /api/agent/tasks/{id}/reply` `{reply}` or `{error}`: the page's answer.
//!
//! No page answering (the lease [`ANSWER_TASKS`] unheld) is a clear 503, so a
//! flow is not left waiting on nobody.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};

/// The lease the agent's page holds while it answers flows' tasks.
pub const ANSWER_TASKS: &str = "answer-tasks";
/// How long a request waits for the reply, unless it says.
const DEFAULT_WAIT: u64 = 600;
const MAX_WAIT: u64 = 3_600;
/// Tasks kept (done ones are dropped oldest first past this).
const KEEP: usize = 200;

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Pending,
    Done(String),
    Failed(String),
}

struct Task {
    from: String,
    task: String,
    state: watch::Sender<State>,
    order: u64,
}

struct Hub {
    tasks: Mutex<HashMap<String, Task>>,
    events: broadcast::Sender<Value>,
    next: Mutex<u64>,
}

fn hub() -> &'static Arc<Hub> {
    static HUB: OnceLock<Arc<Hub>> = OnceLock::new();
    HUB.get_or_init(|| Arc::new(Hub { tasks: Mutex::new(HashMap::new()), events: broadcast::channel(256).0, next: Mutex::new(0) }))
}

impl Hub {
    /// A new task, told to the page. Returns its id and how it goes.
    fn add(&self, from: &str, task: &str) -> (String, watch::Receiver<State>) {
        let order = {
            let mut n = self.next.lock().unwrap_or_else(|e| e.into_inner());
            *n += 1;
            *n
        };
        let id = format!("task_{order}_{}", chrono::Utc::now().timestamp_millis());
        let (tx, rx) = watch::channel(State::Pending);
        let mut tasks = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
        if tasks.len() >= KEEP {
            let mut done: Vec<(u64, String)> = tasks.iter().filter(|(_, t)| *t.state.borrow() != State::Pending).map(|(k, t)| (t.order, k.clone())).collect();
            done.sort();
            for (_, k) in done.into_iter().take(tasks.len() + 1 - KEEP) {
                tasks.remove(&k);
            }
        }
        tasks.insert(id.clone(), Task { from: from.to_string(), task: task.to_string(), state: tx, order });
        drop(tasks);
        let _ = self.events.send(json!({"type": "agent.task", "id": id, "from": from, "task": task, "at": chrono::Utc::now().to_rfc3339()}));
        (id, rx)
    }

    /// The page's answer. False when there is no such task, or it was answered already.
    fn answer(&self, id: &str, state: State) -> bool {
        let tasks = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
        match tasks.get(id) {
            Some(t) if *t.state.borrow() == State::Pending => {
                // Kept even with no one waiting (the asker's wait ran out and it asks after).
                t.state.send_replace(state);
                true
            }
            _ => false,
        }
    }

    fn state(&self, id: &str) -> Option<State> {
        self.tasks.lock().unwrap_or_else(|e| e.into_inner()).get(id).map(|t| t.state.borrow().clone())
    }

    /// The tasks still waiting (a page that starts late picks them up).
    fn pending(&self) -> Vec<Value> {
        let tasks = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
        let mut open: Vec<(&u64, Value)> = tasks.iter().filter(|(_, t)| *t.state.borrow() == State::Pending).map(|(id, t)| (&t.order, json!({"type": "agent.task", "id": id, "from": t.from, "task": t.task}))).collect();
        open.sort_by_key(|(o, _)| **o);
        open.into_iter().map(|(_, v)| v).collect()
    }
}

fn outcome(id: &str, state: &State) -> Value {
    match state {
        State::Pending => json!({"id": id, "status": "pending"}),
        State::Done(reply) => json!({"id": id, "status": "done", "reply": reply}),
        State::Failed(error) => json!({"id": id, "status": "failed", "error": error}),
    }
}

fn error(status: StatusCode, code: &str, message: impl Into<String>) -> axum::response::Response {
    (status, Json(json!({"error": {"code": code, "message": message.into()}}))).into_response()
}

pub fn router() -> Router {
    Router::new()
        .route("/api/agent/tasks", post(ask))
        .route("/api/agent/tasks/:id", get(status))
        .route("/api/agent/tasks/:id/reply", post(reply))
        .route("/api/agent/events", get(events))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ask {
    task: String,
    from: Option<String>,
    wait_seconds: Option<u64>,
}

async fn ask(Json(body): Json<Ask>) -> axum::response::Response {
    let task = body.task.trim();
    if task.is_empty() {
        return error(StatusCode::BAD_REQUEST, "no_task", "the task is empty: say what the agent should do");
    }
    if task.chars().count() > 20_000 {
        return error(StatusCode::BAD_REQUEST, "too_long", "a task is at most 20,000 characters");
    }
    if crate::bridge::leases::holder(ANSWER_TASKS).is_none() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "agent_not_open", "the agent is not open in OAIY to take the task (open OAIY, whose agent answers flows' tasks)");
    }
    let from = body.from.as_deref().map(str::trim).filter(|f| !f.is_empty()).unwrap_or("a flow");
    let (id, mut rx) = hub().add(from, task);
    let wait = Duration::from_secs(body.wait_seconds.unwrap_or(DEFAULT_WAIT).clamp(1, MAX_WAIT));
    let settled = tokio::time::timeout(wait, rx.wait_for(|s| *s != State::Pending)).await;
    let state = match settled {
        Ok(Ok(s)) => s.clone(),
        _ => State::Pending,
    };
    Json(outcome(&id, &state)).into_response()
}

async fn status(Path(id): Path<String>) -> axum::response::Response {
    match hub().state(&id) {
        Some(s) => Json(outcome(&id, &s)).into_response(),
        None => error(StatusCode::NOT_FOUND, "no_task", format!("no task {id}")),
    }
}

#[derive(Deserialize)]
struct Reply {
    reply: Option<String>,
    error: Option<String>,
}

async fn reply(Path(id): Path<String>, Json(body): Json<Reply>) -> axum::response::Response {
    let state = match (body.reply, body.error) {
        (_, Some(e)) if !e.trim().is_empty() => State::Failed(e),
        (Some(r), _) => State::Done(r),
        _ => return error(StatusCode::BAD_REQUEST, "no_reply", "send reply or error"),
    };
    if hub().answer(&id, state) {
        Json(json!({"ok": true})).into_response()
    } else {
        error(StatusCode::CONFLICT, "not_pending", format!("task {id} is not waiting for a reply"))
    }
}

/// `agent.task` events, the tasks still waiting first.
async fn events() -> impl IntoResponse {
    let first = hub().pending();
    let rx = hub().events.subscribe();
    let stream = futures_util::stream::unfold((first, rx), |(mut first, mut rx)| async move {
        if !first.is_empty() {
            let v = first.remove(0);
            return Some((Ok::<Event, Infallible>(Event::default().data(v.to_string())), (first, rx)));
        }
        loop {
            match rx.recv().await {
                Ok(v) => return Some((Ok(Event::default().data(v.to_string())), (first, rx))),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_task_is_told_to_the_page_and_its_answer_settles_it() {
        let h = Hub { tasks: Mutex::new(HashMap::new()), events: broadcast::channel(8).0, next: Mutex::new(0) };
        let mut seen = h.events.subscribe();
        let (id, mut rx) = h.add("Welcome flow", "Write a welcome note for Sam");
        let told = seen.recv().await.unwrap();
        assert_eq!((told["id"].as_str(), told["task"].as_str(), told["from"].as_str()), (Some(id.as_str()), Some("Write a welcome note for Sam"), Some("Welcome flow")));
        assert_eq!(h.pending().len(), 1, "a page that starts late picks it up");
        assert!(h.answer(&id, State::Done("Dear Sam, welcome!".into())));
        assert!(!h.answer(&id, State::Failed("again".into())), "answered once");
        let s = rx.wait_for(|s| *s != State::Pending).await.unwrap().clone();
        assert_eq!(outcome(&id, &s), json!({"id": id, "status": "done", "reply": "Dear Sam, welcome!"}));
        assert!(h.pending().is_empty());
        assert!(!h.answer("task_nope", State::Done(String::new())));
    }

    #[test]
    fn an_answer_after_the_wait_ran_out_is_kept_for_the_asker_to_ask_after() {
        let h = Hub { tasks: Mutex::new(HashMap::new()), events: broadcast::channel(8).0, next: Mutex::new(0) };
        let (id, rx) = h.add("Nightly report", "Summarise the day");
        drop(rx);
        assert!(h.answer(&id, State::Done("A quiet day.".into())));
        assert_eq!(h.state(&id), Some(State::Done("A quiet day.".into())));
    }
}
