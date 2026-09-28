//! Calls answered by the agent.
//!
//! Aokie (the phone bridge) streams a live call's audio to this desktop in its
//! `desktop_realtime` mode, over a WebSocket at exactly
//! `ws://127.0.0.1:17872/api/ai/providers/{id}/v1/realtime/stream`, with the
//! gateway token this desktop gives it. The desktop finds the caller's words
//! ([`call`]), and the agent in the app answers them: the app follows
//! `GET /api/voice/events` (server-sent events) and says what to speak with
//! `POST /api/voice/calls/{id}/say`. The call's own tools (an appointment
//! request, a business lookup, finishing the call) go to Aokie from there too.
//! The audio stays on this computer: the destination Aokie is told, and holds
//! consent for, is [`DESTINATION`].

pub mod audio;
pub mod call;
pub mod engines;

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{broadcast, mpsc, oneshot};

use call::CallCommand;
use engines::Engines;

/// Where Aokie's realtime stream connects (it attaches its gateway token only here).
pub const GATEWAY_PORT: u16 = 17_872;
/// The processor Aokie is told a call's audio goes to: OAIY, on this computer.
pub const DESTINATION: &str = "https://oaiy.localhost";
/// The lease an app page holds while it answers calls (as `answer-texts` is for texts).
pub const ANSWER_CALLS: &str = "answer-calls";

/// The token Aokie presents on the gateway (`FORMLOGIC_AI_GATEWAY_TOKEN`): random per run, and good for nothing else.
pub fn gateway_token() -> &'static str {
    static TOKEN: OnceLock<String> = OnceLock::new();
    TOKEN.get_or_init(|| {
        // Whoever starts the desktop may name it (a test standing in for the phone).
        if let Some(given) = std::env::var("OAIY_GATEWAY_TOKEN").ok().filter(|t| t.len() >= 16) {
            return given;
        }
        let mut bytes = [0u8; 32];
        match getrandom::getrandom(&mut bytes) {
            Ok(()) => bytes.iter().map(|b| format!("{b:02x}")).collect(),
            Err(_) => String::new(),
        }
    })
}

/// Who a call is with: from the plugin's `call.incoming` / `call.caller_id` events.
type CallerOf = dyn Fn(&str) -> Option<(String, String)> + Send + Sync;

#[derive(Clone)]
pub struct VoiceHub {
    inner: Arc<Inner>,
}

struct Inner {
    engines: Engines,
    events: broadcast::Sender<Value>,
    calls: Mutex<HashMap<String, mpsc::UnboundedSender<CallCommand>>>,
    caller_of: Box<CallerOf>,
}

impl VoiceHub {
    pub fn new(engines: Engines, caller_of: impl Fn(&str) -> Option<(String, String)> + Send + Sync + 'static) -> Self {
        let (events, _) = broadcast::channel(512);
        Self { inner: Arc::new(Inner { engines, events, calls: Mutex::new(HashMap::new()), caller_of: Box::new(caller_of) }) }
    }

    /// Tell the app pages following the calls.
    pub fn emit(&self, mut event: Value) {
        event["at"] = json!(chrono::Utc::now().to_rfc3339());
        let _ = self.inner.events.send(event);
    }

    /// What the caller said: to the app, whose agent answers it. With no page
    /// answering calls, the caller is told so and the call is finished.
    fn caller_said(&self, call: &str, text: &str) {
        self.emit(json!({"type": "call.caller", "callId": call, "text": text}));
        if crate::bridge::leases::holder(ANSWER_CALLS).is_none() {
            if let Some(tx) = self.inner.calls.lock().unwrap().get(call) {
                let (reply, _) = oneshot::channel();
                let _ = tx.send(CallCommand::Finish { goodbye: "Sorry, no one can take your call right now. Please try again a little later. Goodbye!".into(), reply });
            }
        }
    }

    fn caller_of(&self, call: &str) -> Option<(String, String)> {
        (self.inner.caller_of)(call)
    }

    fn register(&self, call: &str, tx: mpsc::UnboundedSender<CallCommand>) {
        self.inner.calls.lock().unwrap().insert(call.to_string(), tx);
    }

    fn unregister(&self, call: &str) {
        self.inner.calls.lock().unwrap().remove(call);
    }

    fn command(&self, call: &str) -> Option<mpsc::UnboundedSender<CallCommand>> {
        self.inner.calls.lock().unwrap().get(call).cloned()
    }

    pub fn live_calls(&self) -> Vec<String> {
        self.inner.calls.lock().unwrap().keys().cloned().collect()
    }
}

// ---- The app's side (on the desktop's API, behind its guard) -----------------

/// `GET /api/voice/events` (server-sent events: `call.started`, `call.caller`,
/// `call.said`, `call.speech_started`, `call.interrupted`, `call.error`,
/// `call.ended`), `GET /api/voice/calls`, and per call `say`, `tool`, `finish`, `hush`.
pub fn app_router(hub: VoiceHub) -> Router {
    Router::new()
        .route("/api/voice/events", get(events))
        .route("/api/voice/calls", get(calls))
        .route("/api/voice/calls/:id/say", post(say))
        .route("/api/voice/calls/:id/tool", post(tool))
        .route("/api/voice/calls/:id/finish", post(finish))
        .route("/api/voice/calls/:id/hush", post(hush))
        .with_state(hub)
}

async fn events(State(hub): State<VoiceHub>) -> impl IntoResponse {
    let hello = json!({"type": "hello", "calls": hub.live_calls()});
    let rx = hub.inner.events.subscribe();
    let stream = futures_util::stream::unfold((Some(hello), rx), |(first, mut rx)| async move {
        if let Some(h) = first {
            return Some((Ok::<Event, Infallible>(Event::default().data(h.to_string())), (None, rx)));
        }
        loop {
            match rx.recv().await {
                Ok(v) => return Some((Ok(Event::default().data(v.to_string())), (None, rx))),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

async fn calls(State(hub): State<VoiceHub>) -> impl IntoResponse {
    Json(json!({"calls": hub.live_calls()}))
}

fn no_call(id: &str) -> axum::response::Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": {"code": "no_call", "message": format!("no live call {id}")}}))).into_response()
}

fn answer<T: serde::Serialize>(result: Result<T, String>) -> axum::response::Response {
    match result {
        Ok(v) => (StatusCode::OK, Json(json!({"ok": true, "result": v}))).into_response(),
        Err(e) => (StatusCode::CONFLICT, Json(json!({"error": {"code": "call_refused", "message": e}}))).into_response(),
    }
}

#[derive(Deserialize)]
struct SayBody {
    text: String,
}

async fn say(State(hub): State<VoiceHub>, Path(id): Path<String>, Json(body): Json<SayBody>) -> axum::response::Response {
    let Some(tx) = hub.command(&id) else { return no_call(&id) };
    let (reply, rx) = oneshot::channel();
    if tx.send(CallCommand::Say { text: body.text, reply }).is_err() {
        return no_call(&id);
    }
    answer(rx.await.unwrap_or_else(|_| Err("the call ended".into())).map(|item| json!({"itemId": item})))
}

#[derive(Deserialize)]
struct ToolBody {
    name: String,
    #[serde(default)]
    arguments: Value,
}

async fn tool(State(hub): State<VoiceHub>, Path(id): Path<String>, Json(body): Json<ToolBody>) -> axum::response::Response {
    let Some(tx) = hub.command(&id) else { return no_call(&id) };
    let (reply, rx) = oneshot::channel();
    let arguments = if body.arguments.is_object() { body.arguments } else { json!({}) };
    if tx.send(CallCommand::Tool { name: body.name, arguments, reply }).is_err() {
        return no_call(&id);
    }
    answer(match tokio::time::timeout(Duration::from_secs(20), rx).await {
        Ok(r) => r.unwrap_or_else(|_| Err("the call ended".into())),
        Err(_) => Err("the phone did not answer the tool in time".into()),
    })
}

#[derive(Deserialize)]
struct FinishBody {
    #[serde(default)]
    goodbye: String,
}

async fn finish(State(hub): State<VoiceHub>, Path(id): Path<String>, Json(body): Json<FinishBody>) -> axum::response::Response {
    let Some(tx) = hub.command(&id) else { return no_call(&id) };
    let (reply, rx) = oneshot::channel();
    if tx.send(CallCommand::Finish { goodbye: body.goodbye, reply }).is_err() {
        return no_call(&id);
    }
    answer(match tokio::time::timeout(Duration::from_secs(20), rx).await {
        Ok(r) => r.unwrap_or_else(|_| Err("the call ended".into())),
        Err(_) => Err("the phone did not answer in time".into()),
    })
}

async fn hush(State(hub): State<VoiceHub>, Path(id): Path<String>) -> axum::response::Response {
    let Some(tx) = hub.command(&id) else { return no_call(&id) };
    let _ = tx.send(CallCommand::Hush);
    answer(Ok(json!({})))
}

// ---- Aokie's side: the gateway on 17872 ------------------------------------------

fn bearer_ok(headers: &HeaderMap) -> bool {
    let token = gateway_token();
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    // Compared without a per-byte early exit.
    !token.is_empty() && presented.len() == token.len() && presented.bytes().zip(token.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

async fn realtime(State(hub): State<VoiceHub>, Path(_provider): Path<String>, headers: HeaderMap, ws: WebSocketUpgrade) -> axum::response::Response {
    if !bearer_ok(&headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {"code": "auth_required", "message": "the gateway token is required"}}))).into_response();
    }
    let engines = hub.inner.engines.clone();
    ws.max_message_size(256 * 1024).on_upgrade(move |socket| call::run(socket, hub, engines))
}

pub fn gateway_router(hub: VoiceHub) -> Router {
    Router::new()
        .route("/api/health", get(|| async { Json(json!({"status": "ok", "product": "oaiy-gateway"})) }))
        .route("/api/ai/providers/:id/v1/realtime/stream", get(realtime))
        .with_state(hub)
}

/// Serve the gateway on 127.0.0.1:17872 until the process ends (a port in use is logged, not fatal).
pub async fn serve_gateway(hub: VoiceHub) {
    let addr = SocketAddr::from(([127, 0, 0, 1], GATEWAY_PORT));
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => {
            log::info!("OAIY voice gateway listening on http://{addr}");
            if let Err(e) = axum::serve(listener, gateway_router(hub)).await {
                log::error!("voice gateway stopped: {e}");
            }
        }
        Err(e) => log::error!("voice gateway cannot listen on {addr}: {e} (calls cannot reach the agent)"),
    }
}

/// Who a call is with, read from the plugin host's recent events.
pub fn caller_from_events(events: &[Value], call: &str) -> Option<(String, String)> {
    events.iter().rev().find_map(|e| {
        let name = e.get("name").and_then(Value::as_str).unwrap_or("");
        if !matches!(name, "aokie.call.incoming" | "aokie.call.caller_id" | "aokie.call.outbound.dialing") {
            return None;
        }
        let data = e.get("data")?;
        let id = data.get("callId").and_then(Value::as_str).or_else(|| e.get("correlationId").and_then(Value::as_str))?;
        if id != call {
            return None;
        }
        let from = data.get("from").or_else(|| data.get("to")).and_then(Value::as_str)?.to_string();
        let who = data.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        Some((from, who))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_caller_is_found_by_call_id() {
        let events = vec![
            json!({"name": "aokie.call.incoming", "correlationId": "call_a", "data": {"callId": "call_a", "from": "+61400000001"}}),
            json!({"name": "aokie.call.incoming", "correlationId": "call_b", "data": {"callId": "call_b", "from": "+61400000002", "name": "Lance"}}),
        ];
        assert_eq!(caller_from_events(&events, "call_b"), Some(("+61400000002".into(), "Lance".into())));
        assert_eq!(caller_from_events(&events, "call_c"), None);
    }

    #[test]
    fn the_gateway_wants_its_own_token() {
        let mut h = HeaderMap::new();
        assert!(!bearer_ok(&h));
        h.insert(axum::http::header::AUTHORIZATION, format!("Bearer {}", gateway_token()).parse().unwrap());
        assert!(bearer_ok(&h));
        h.insert(axum::http::header::AUTHORIZATION, "Bearer nope".parse().unwrap());
        assert!(!bearer_ok(&h));
    }
}
