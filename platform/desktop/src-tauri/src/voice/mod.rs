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
pub mod callers;
pub mod engines;
pub mod voices;

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
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

    /// What the caller said: to the app, whose agent answers it, with `how`
    /// (when they said it, and whether over us). With no page answering calls,
    /// the caller is told so and the call is finished.
    fn caller_said(&self, call: &str, text: &str, how: Value) {
        let mut event = json!({"type": "call.caller", "callId": call, "text": text});
        if let (Some(event), Value::Object(how)) = (event.as_object_mut(), how) {
            event.extend(how);
        }
        self.emit(event);
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
/// `PUT /api/voice/callers` keeps the name a caller is greeted by. These are the
/// phone's: while no plugin provides the phone they answer `module_disabled`.
/// Speech to text and the voices are core (the agent's own tools use them).
pub fn app_router(hub: VoiceHub) -> Router {
    let phone = Router::new()
        .route("/api/voice/events", get(events))
        .route("/api/voice/calls", get(calls))
        .route("/api/voice/calls/:id/say", post(say))
        .route("/api/voice/calls/:id/tool", post(tool))
        .route("/api/voice/calls/:id/finish", post(finish))
        .route("/api/voice/calls/:id/hush", post(hush))
        // A caller's name, for their number (an empty name forgets it).
        .route("/api/voice/callers", put(caller_name))
        // `route_layer`: a path that is not one of these still answers 404.
        .route_layer(axum::middleware::from_fn(require_phone));
    Router::new()
        // Half a minute of 16 kHz speech is under 1 MB; the app sends pieces that size.
        .route("/api/voice/transcribe", post(transcribe).layer(axum::extract::DefaultBodyLimit::max(8 * 1024 * 1024)))
        // The voice calls are answered in: clips by name, the one chosen, a new
        // one (the clip's bytes as the body), and a line spoken in one to hear it.
        .route("/api/voice/voices", get(voices_list).post(voice_add).layer(axum::extract::DefaultBodyLimit::max(voices::MAX_CLIP_BYTES + 1024)))
        .route("/api/voice/voices/chosen", put(voice_choose))
        .route("/api/voice/voices/:name", delete(voice_remove))
        .route("/api/voice/voices/:name/try", post(voice_try))
        .merge(phone)
        .with_state(hub)
}

/// The phone's routes answer only while a plugin provides the phone.
async fn require_phone(request: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    if !crate::modules::is_enabled(crate::modules::PHONE) {
        return crate::modules::disabled_response(crate::modules::PHONE);
    }
    next.run(request).await
}

#[derive(Deserialize)]
struct CallerName {
    number: String,
    #[serde(default)]
    name: String,
}

async fn caller_name(Json(body): Json<CallerName>) -> axum::response::Response {
    match callers::remember(&body.number, &body.name) {
        Ok(name) => Json(json!({"number": body.number, "name": name})).into_response(),
        Err(e) => voice_error(StatusCode::BAD_REQUEST, "bad_caller", e),
    }
}

fn voice_error(status: StatusCode, code: &str, message: impl Into<String>) -> axum::response::Response {
    (status, Json(json!({"error": {"code": code, "message": message.into()}}))).into_response()
}

async fn voices_list() -> Json<Value> {
    Json(json!({"voices": voices::list(), "chosen": voices::chosen()}))
}

#[derive(Deserialize)]
struct Chosen {
    voice: String,
}

async fn voice_choose(Json(body): Json<Chosen>) -> axum::response::Response {
    match voices::choose(&body.voice) {
        Ok(name) => Json(json!({"chosen": name})).into_response(),
        Err(e) => voice_error(StatusCode::NOT_FOUND, "no_voice", e),
    }
}

#[derive(Deserialize)]
struct NewVoice {
    name: String,
    /// The clip's file name or extension (`clip.mp3`, `mp3`).
    file: String,
    /// What the clip says, word for word (else the speech server hears it).
    words: Option<String>,
    /// Choose it for calls.
    choose: Option<bool>,
}

async fn voice_add(Query(q): Query<NewVoice>, body: axum::body::Bytes) -> axum::response::Response {
    let extension = q.file.rsplit('.').next().unwrap_or("").to_string();
    match voices::add(&q.name, &extension, &body, q.words.as_deref()) {
        Ok(v) => {
            if q.choose.unwrap_or(false) {
                let _ = voices::choose(&v.name);
            }
            (StatusCode::CREATED, Json(json!({"voice": v, "chosen": voices::chosen()}))).into_response()
        }
        Err(e) => voice_error(StatusCode::BAD_REQUEST, "bad_voice", e),
    }
}

async fn voice_remove(Path(name): Path<String>) -> axum::response::Response {
    match voices::remove(&name) {
        Ok(()) => Json(json!({"removed": name, "chosen": voices::chosen()})).into_response(),
        Err(e) => voice_error(StatusCode::NOT_FOUND, "no_voice", e),
    }
}

#[derive(Deserialize, Default)]
struct TryLine {
    text: Option<String>,
}

/// A line spoken in a voice, as a WAV, to hear it before choosing it.
async fn voice_try(State(hub): State<VoiceHub>, Path(name): Path<String>, body: Option<Json<TryLine>>) -> axum::response::Response {
    if !voices::list().iter().any(|v| v.name.eq_ignore_ascii_case(&name)) {
        return voice_error(StatusCode::NOT_FOUND, "no_voice", format!("no voice called {name:?}"));
    }
    let text = body.and_then(|b| b.0.text).map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
    let text = text.unwrap_or_else(|| "Hi, thanks for calling! How can I help you today?".to_string());
    if text.chars().count() > 400 {
        return voice_error(StatusCode::BAD_REQUEST, "too_long", "a line to try is at most 400 characters");
    }
    let (tx, mut rx) = mpsc::channel::<Vec<i16>>(64);
    let engines = hub.inner.engines.clone();
    let speaking = tokio::spawn(async move { engines.speak(&text, Some(&name), &tx).await });
    let mut pcm: Vec<i16> = Vec::new();
    while let Some(piece) = rx.recv().await {
        pcm.extend_from_slice(&piece);
    }
    match speaking.await {
        Ok(Ok(())) => ([(axum::http::header::CONTENT_TYPE, "audio/wav")], audio::wav(&pcm, engines::WIRE_RATE)).into_response(),
        Ok(Err(e)) => voice_error(StatusCode::BAD_GATEWAY, "text_to_speech", e),
        Err(e) => voice_error(StatusCode::INTERNAL_SERVER_ERROR, "text_to_speech", e.to_string()),
    }
}

/// What was said in a recording (the agent's speech-to-text tool): a 16 kHz mono 16-bit WAV in, `{text}` out.
async fn transcribe(State(hub): State<VoiceHub>, body: axum::body::Bytes) -> axum::response::Response {
    match audio::wav_format(&body) {
        Some((16_000, 1, 16)) => {}
        Some((rate, channels, bits)) => {
            let message = format!("send 16 kHz mono 16-bit audio (this is {rate} Hz, {channels} channel(s), {bits}-bit)");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": {"code": "bad_audio", "message": message}}))).into_response();
        }
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": {"code": "bad_audio", "message": "the body is not a PCM WAV file"}}))).into_response(),
    }
    match hub.inner.engines.transcribe_wav(body.to_vec()).await {
        Ok(text) => (StatusCode::OK, Json(json!({"text": text}))).into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"error": {"code": "speech_to_text", "message": e}}))).into_response(),
    }
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
    // No phone, no calls: said before the token, so a plugin that is not the phone's hears why.
    if !crate::modules::is_enabled(crate::modules::PHONE) {
        return crate::modules::disabled_response(crate::modules::PHONE);
    }
    if !bearer_ok(&headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {"code": "auth_required", "message": "the gateway token is required"}}))).into_response();
    }
    let engines = hub.inner.engines.clone();
    ws.max_message_size(256 * 1024).on_upgrade(move |socket| call::run(socket, hub, engines))
}

/// The gateway's routes: a call's realtime stream, and (`chat`, behind the same
/// token) a provider's chat and models for Aokie's own speech lanes.
pub fn gateway_router(hub: VoiceHub, chat: Router) -> Router {
    Router::new()
        .route("/api/health", get(|| async { Json(json!({"status": "ok", "product": "oaiy-gateway"})) }))
        .route("/api/ai/providers/:id/v1/realtime/stream", get(realtime))
        .with_state(hub)
        .merge(chat.route_layer(axum::middleware::from_fn(require_gateway_token)))
}

/// The gateway token, for everything but the health check (the realtime stream checks it itself).
async fn require_gateway_token(request: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    if !bearer_ok(request.headers()) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {"code": "auth_required", "message": "the gateway token is required"}}))).into_response();
    }
    next.run(request).await
}

/// Serve the gateway on 127.0.0.1:17872 until the process ends (a port in use is logged, not fatal).
pub async fn serve_gateway(hub: VoiceHub, chat: Router) {
    let addr = SocketAddr::from(([127, 0, 0, 1], GATEWAY_PORT));
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => {
            log::info!("OAIY voice gateway listening on http://{addr}");
            if let Err(e) = axum::serve(listener, gateway_router(hub, chat)).await {
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

    /// A provider's chat on the gateway answers only with the gateway token.
    #[tokio::test]
    async fn the_gateways_chat_needs_its_token() {
        let chat = Router::new()
            .route("/api/ai/providers/:id/v1/models", get(|Path(id): Path<String>| async move { Json(json!({"provider": id})) }))
            .route_layer(axum::middleware::from_fn(require_gateway_token));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, chat).await.unwrap() });
        let url = format!("http://{addr}/api/ai/providers/studio/v1/models");
        let client = reqwest::Client::new();
        assert_eq!(client.get(&url).send().await.unwrap().status(), 401);
        assert_eq!(client.get(&url).bearer_auth("wrong-token-wrong-token").send().await.unwrap().status(), 401);
        let ok = client.get(&url).bearer_auth(gateway_token()).send().await.unwrap();
        assert_eq!(ok.status(), 200);
        assert_eq!(ok.json::<Value>().await.unwrap()["provider"], "studio");
    }

    /// The app's side of the calls, on a port of its own.
    async fn serve_app() -> String {
        let hub = VoiceHub::new(Engines::at("http://127.0.0.1:9", "http://127.0.0.1:9"), |_| None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app_router(hub)).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_call_routes_answer_module_disabled_with_the_phone_off() {
        let _off = crate::modules::test_gate::enable(&[]);
        let base = serve_app().await;
        let client = reqwest::Client::new();
        let refused = [
            client.get(format!("{base}/api/voice/events")),
            client.get(format!("{base}/api/voice/calls")),
            client.post(format!("{base}/api/voice/calls/call_1/say")).json(&json!({"text": "hi"})),
            client.post(format!("{base}/api/voice/calls/call_1/hush")),
            client.put(format!("{base}/api/voice/callers")).json(&json!({"number": "+61400000000", "name": "Lance"})),
        ];
        for request in refused {
            let resp = request.send().await.unwrap();
            let url = resp.url().to_string();
            assert_eq!(resp.status(), 409, "{url}");
            let body: Value = resp.json().await.unwrap();
            assert_eq!(body["error"]["code"], "module_disabled", "{url}");
        }
        // The voices and speech to text are the agent's too: not the phone's.
        assert_eq!(client.get(format!("{base}/api/voice/voices")).send().await.unwrap().status(), 200);
        assert_eq!(client.post(format!("{base}/api/voice/transcribe")).body("not a wav").send().await.unwrap().status(), 400);
        // An unknown path is still not found.
        assert_eq!(client.get(format!("{base}/api/voice/nothing")).send().await.unwrap().status(), 404);
    }

    #[tokio::test]
    async fn the_call_routes_answer_with_the_phone_on() {
        let _on = crate::modules::test_gate::enable(&[crate::modules::PHONE]);
        let base = serve_app().await;
        let client = reqwest::Client::new();
        let calls: Value = client.get(format!("{base}/api/voice/calls")).send().await.unwrap().json().await.unwrap();
        assert_eq!(calls["calls"], json!([]));
        let say = client.post(format!("{base}/api/voice/calls/call_1/say")).json(&json!({"text": "hi"})).send().await.unwrap();
        assert_eq!(say.status(), 404, "no such call, but the route answers");
    }

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
