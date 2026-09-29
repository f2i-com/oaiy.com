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
pub mod contacts;
pub mod engines;
pub mod voices;

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

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
    /// What this desktop itself knows of each call (who rang, what they said), for the calls
    /// that are live and those that ended lately: it is what a message or a transfer is judged
    /// by, never what the receptionist's model says.
    records: Mutex<HashMap<String, CallRecord>>,
    /// The owner's settings for transfers and messages, and where messages are kept.
    ring: RwLock<Arc<crate::ring::Ring>>,
    messages: RwLock<crate::messages::Store>,
    /// What tells the owner a message arrived, when not the desktop's own (a test's).
    notifier: RwLock<Option<Arc<dyn crate::messages::MessageNotifier>>>,
}

/// How long a call's record is kept after it ends: a caller who hangs up as they finish a message still has it kept.
const RECORD_KEEP: Duration = Duration::from_secs(600);
/// The most calls' records held.
const RECORDS_MAX: usize = 64;
/// The caller's last turns kept per call (the ring policy reads three).
const TURNS_KEPT: usize = 6;

/// What this desktop knows of one call.
struct CallRecord {
    /// The number the phone said the call came from, and the name it gave.
    from: String,
    name: String,
    /// What the caller said, last last: their own words as this desktop heard them.
    turns: VecDeque<String>,
    /// When the call ended (None while it is live).
    ended: Option<Instant>,
}

/// Every hub made in this process (there is one; a test may make more), so anything that has to know whether
/// a call is live (an update, before it restarts the app) can ask without being handed the hub.
static HUBS: Mutex<Vec<std::sync::Weak<Inner>>> = Mutex::new(Vec::new());

/// Phone calls live now, on every hub of this process.
pub fn live_call_count() -> usize {
    let mut hubs = HUBS.lock().unwrap_or_else(|e| e.into_inner());
    hubs.retain(|hub| hub.strong_count() > 0);
    hubs.iter().filter_map(std::sync::Weak::upgrade).map(|hub| hub.calls.lock().unwrap_or_else(|e| e.into_inner()).len()).sum()
}

impl VoiceHub {
    pub fn new(engines: Engines, caller_of: impl Fn(&str) -> Option<(String, String)> + Send + Sync + 'static) -> Self {
        let (events, _) = broadcast::channel(512);
        let ring = crate::ring::shared().unwrap_or_else(|| crate::ring::Ring::in_memory(Default::default()));
        let inner = Arc::new(Inner {
            engines,
            events,
            calls: Mutex::new(HashMap::new()),
            caller_of: Box::new(caller_of),
            records: Mutex::new(HashMap::new()),
            ring: RwLock::new(ring),
            messages: RwLock::new(crate::messages::shared()),
            notifier: RwLock::new(None),
        });
        HUBS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&inner));
        Self { inner }
    }

    /// The ring (the owner's settings for transfers and messages) this hub answers to.
    pub fn ring(&self) -> Arc<crate::ring::Ring> {
        self.inner.ring.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Use another ring (tests set one with transfers on; a desktop uses the one it opened).
    pub fn set_ring(&self, ring: Arc<crate::ring::Ring>) {
        *self.inner.ring.write().unwrap_or_else(|e| e.into_inner()) = ring;
    }

    pub fn messages(&self) -> crate::messages::Store {
        self.inner.messages.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Keep messages in `store` (tests).
    pub fn set_messages(&self, store: crate::messages::Store) {
        *self.inner.messages.write().unwrap_or_else(|e| e.into_inner()) = store;
    }

    /// Tell the owner of a message with `notifier` rather than the desktop's own (tests).
    pub fn set_message_notifier(&self, notifier: Option<Arc<dyn crate::messages::MessageNotifier>>) {
        *self.inner.notifier.write().unwrap_or_else(|e| e.into_inner()) = notifier;
    }

    /// A call began: who it is with, as the phone said.
    fn note_call(&self, call: &str, from: &str, name: &str) {
        let mut records = self.inner.records.lock().unwrap_or_else(|e| e.into_inner());
        prune(&mut records);
        let record = records.entry(call.to_string()).or_insert_with(|| CallRecord { from: String::new(), name: String::new(), turns: VecDeque::new(), ended: None });
        record.from = from.trim().to_string();
        record.name = name.trim().to_string();
        record.ended = None;
    }

    /// What the caller said, as this desktop heard it (not an acknowledgement: "mm-hmm" asks for nothing).
    fn note_turn(&self, call: &str, text: &str) {
        let mut records = self.inner.records.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = records.get_mut(call) {
            record.turns.push_back(text.to_string());
            while record.turns.len() > TURNS_KEPT {
                record.turns.pop_front();
            }
        }
    }

    /// The call ended (its record is kept a while: see [`RECORD_KEEP`]).
    fn note_ended(&self, call: &str) {
        let mut records = self.inner.records.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = records.get_mut(call) {
            record.ended.get_or_insert_with(Instant::now);
        }
        prune(&mut records);
    }

    /// Who a call is with, by this desktop's own record: for a call that is live or ended within ten minutes.
    pub fn call_facts(&self, call: &str) -> Option<(String, String)> {
        let mut records = self.inner.records.lock().unwrap_or_else(|e| e.into_inner());
        prune(&mut records);
        records.get(call).map(|r| (r.from.clone(), r.name.clone()))
    }

    /// What the caller said last on a call, oldest first (at most six).
    pub fn caller_turns(&self, call: &str) -> Vec<String> {
        let records = self.inner.records.lock().unwrap_or_else(|e| e.into_inner());
        records.get(call).map(|r| r.turns.iter().cloned().collect()).unwrap_or_default()
    }

    /// Keep a message the receptionist took on `call` for the owner, and tell the owner. The number is
    /// this desktop's own record of who rang (never what the model says), and it is refused when the
    /// call is not one this desktop knows (live, or ended within ten minutes), when the owner has not
    /// turned message taking on, or when a limit is reached: the receptionist is told, and says so.
    pub fn take_message(&self, call: &str, request: MessageRequest) -> Result<TakenMessage, crate::messages::Error> {
        use crate::messages::{clean, Error, NewMessage, MAX_NAME};
        let Some((from, phone_name)) = self.call_facts(call) else {
            return Err(Error::new(404, "no_call", format!("no call {call:?} to take a message on")));
        };
        if !self.ring().features().messages {
            return Err(Error::new(409, "messages_off", "taking messages is switched off: tell the caller you cannot take one"));
        }
        let urgent = match request.urgency.trim() {
            "" | "normal" => false,
            "urgent" => true,
            other => return Err(Error::new(400, "bad_urgency", format!("{other:?} is not an urgency: normal or urgent"))),
        };
        let given = clean(&request.caller_name, MAX_NAME);
        let name = if given.is_empty() { callers::name_of(&from).or_else(|| callers::looks_like_name(&phone_name).then(|| phone_name.trim().to_string())).unwrap_or_default() } else { given };
        let message = self.messages().add(NewMessage { call_id: call.to_string(), from, name, callback: request.callback_number, message: request.message, urgent, wants_callback: request.wants_callback })?;
        let told = self.inner.notifier.read().unwrap_or_else(|e| e.into_inner()).clone();
        let notified = match told {
            Some(notifier) => notifier.message_taken(&message),
            None => crate::messages::notify(&message),
        };
        self.emit(json!({"type": "message.new", "id": message.id, "callId": call, "urgency": message.urgency}));
        Ok(TakenMessage { message, notified })
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
        if how.get("backchannel").and_then(Value::as_bool) != Some(true) {
            self.note_turn(call, text);
        }
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
        self.note_ended(call);
    }

    fn command(&self, call: &str) -> Option<mpsc::UnboundedSender<CallCommand>> {
        self.inner.calls.lock().unwrap().get(call).cloned()
    }

    pub fn live_calls(&self) -> Vec<String> {
        self.inner.calls.lock().unwrap().keys().cloned().collect()
    }
}

/// What the receptionist gives to be recorded as a message.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MessageRequest {
    /// What the caller wants the owner to know.
    pub message: String,
    #[serde(default)]
    pub caller_name: String,
    /// Digits and an optional leading +: where to ring them, when not the number they rang from.
    #[serde(default)]
    pub callback_number: String,
    /// `normal` (or none) or `urgent`.
    #[serde(default)]
    pub urgency: String,
    #[serde(default)]
    pub wants_callback: bool,
}

/// A message kept, and whether the owner could be told of it.
pub struct TakenMessage {
    pub message: crate::messages::Message,
    pub notified: bool,
}

/// Records that have ended and are no longer to be kept (see [`RECORD_KEEP`]), and the oldest beyond [`RECORDS_MAX`].
fn prune(records: &mut HashMap<String, CallRecord>) {
    records.retain(|_, r| r.ended.is_none_or(|t| t.elapsed() < RECORD_KEEP));
    while records.len() > RECORDS_MAX {
        let oldest = records.iter().filter(|(_, r)| r.ended.is_some()).min_by_key(|(_, r)| r.ended).map(|(k, _)| k.clone());
        match oldest {
            Some(k) => {
                records.remove(&k);
            }
            None => break,
        }
    }
}

// ---- The app's side (on the desktop's API, behind its guard) -----------------

/// `GET /api/voice/events` (server-sent events: `call.started`, `call.caller`,
/// `call.said`, `call.speech_started`, `call.interrupted`, `call.resumed` (a
/// reply cut off by an acknowledgement goes on: `itemId`, `fromSentence`,
/// `sentences`), `call.error`, `call.ended`), `GET /api/voice/calls`, and per
/// call `say`, `tool`, `finish`, `hush`.
/// `PUT /api/voice/callers` keeps the name a caller is greeted by (the
/// receptionist's: never over a name the person set in Contacts, see
/// [`contacts`]). These are the phone's: while no plugin provides the phone
/// they answer `module_disabled`.
/// Speech to text and the voices are core (the agent's own tools use them), and
/// kept with the voices, `GET`/`PUT /api/voice/settings` is how long a call's
/// greeting waits after the call connects (`greetingDelayMs`, also in `GET /api/voice/voices`).
pub fn app_router(hub: VoiceHub) -> Router {
    let phone = Router::new()
        .route("/api/voice/events", get(events))
        .route("/api/voice/calls", get(calls))
        .route("/api/voice/calls/:id/say", post(say))
        .route("/api/voice/calls/:id/tool", post(tool))
        .route("/api/voice/calls/:id/finish", post(finish))
        .route("/api/voice/calls/:id/hush", post(hush))
        // A message the receptionist takes for the owner, on the call it is answering.
        .route("/api/voice/calls/:id/message", post(message))
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
        // How long a call's greeting waits after the call connects (`greetingDelayMs`, 0 to 5000).
        .route("/api/voice/settings", get(call_settings).put(call_settings_set))
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
    Json(json!({"voices": voices::list(), "chosen": voices::chosen(), "greetingDelayMs": voices::greeting_delay_ms()}))
}

async fn call_settings() -> Json<Value> {
    Json(json!({"greetingDelayMs": voices::greeting_delay_ms()}))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CallSettings {
    greeting_delay_ms: f64,
}

/// `PUT /api/voice/settings {greetingDelayMs}`: kept to 0 to 5000, and answered with what was kept.
async fn call_settings_set(Json(body): Json<CallSettings>) -> axum::response::Response {
    match voices::set_greeting_delay_ms(body.greeting_delay_ms) {
        Ok(ms) => Json(json!({"greetingDelayMs": ms})).into_response(),
        Err(e) => voice_error(StatusCode::BAD_REQUEST, "bad_settings", e),
    }
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
    /// A hold word ("Okay —"): said only while the caller is quiet, else skipped (`{"skipped": true}`).
    #[serde(default)]
    hold: bool,
}

async fn say(State(hub): State<VoiceHub>, Path(id): Path<String>, Json(body): Json<SayBody>) -> axum::response::Response {
    let Some(tx) = hub.command(&id) else { return no_call(&id) };
    let (reply, rx) = oneshot::channel();
    if tx.send(CallCommand::Say { text: body.text, hold: body.hold, reply }).is_err() {
        return no_call(&id);
    }
    answer(rx.await.unwrap_or_else(|_| Err("the call ended".into())).map(|item| if item.is_empty() { json!({"skipped": true}) } else { json!({"itemId": item}) }))
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

/// `POST /api/voice/calls/:id/message {message, callerName?, callbackNumber?, urgency?, wantsCallback?}`:
/// the receptionist's `take_message`. Answers `{ok, result: {recorded, id, notified}}`: `notified` is whether
/// the owner could be told now (a native notification was raised), and the receptionist says the owner
/// "will be told" only then, else that the message is kept.
async fn message(State(hub): State<VoiceHub>, Path(id): Path<String>, Json(body): Json<MessageRequest>) -> axum::response::Response {
    match hub.take_message(&id, body) {
        Ok(taken) => answer::<Value>(Ok(json!({"recorded": true, "id": taken.message.id, "notified": taken.notified}))),
        Err(e) => {
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, Json(json!({"error": {"code": e.code, "message": e.message}}))).into_response()
        }
    }
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

/// Whether this process serves the gateway: not when `OAIY_VOICE_GATEWAY=off`,
/// which a second OAIY on the same computer (a headless one tried beside the
/// desktop) sets, so it never takes the port the desktop's phone calls use.
pub fn gateway_wanted() -> bool {
    gateway_wanted_by(&std::env::var("OAIY_VOICE_GATEWAY").unwrap_or_default())
}

fn gateway_wanted_by(setting: &str) -> bool {
    !matches!(setting.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false")
}

#[cfg(test)]
#[test]
fn the_gateway_is_served_unless_turned_off() {
    assert!(gateway_wanted_by(""));
    assert!(gateway_wanted_by("on"));
    assert!(!gateway_wanted_by("off"));
    assert!(!gateway_wanted_by(" OFF "));
    assert!(!gateway_wanted_by("0"));
}

/// Serve the gateway on 127.0.0.1:17872 until the process ends (a port in use is logged, not fatal).
pub async fn serve_gateway(hub: VoiceHub, chat: Router) {
    if !gateway_wanted() {
        log::info!("OAIY voice gateway off (OAIY_VOICE_GATEWAY=off): calls cannot reach this OAIY");
        return;
    }
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

    #[test]
    fn a_call_on_any_hub_is_counted_for_an_update_to_see_and_stops_counting_when_it_ends() {
        // (Other tests hold calls on hubs of their own at the same time, so this asserts only what a call of ours guarantees.)
        let hub = VoiceHub::new(Engines::at("http://127.0.0.1:9", "http://127.0.0.1:9"), |_| None);
        let (tx, _rx) = mpsc::unbounded_channel();
        hub.register("update_test_call", tx);
        assert_eq!(hub.live_calls(), vec!["update_test_call".to_string()]);
        assert!(live_call_count() >= 1, "a live call is counted without being handed the hub");
        hub.unregister("update_test_call");
        assert!(hub.live_calls().is_empty());
        // A hub that is gone leaves the count (its calls went with it).
        drop(hub);
        let _ = live_call_count();
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
        let listed: Value = client.get(format!("{base}/api/voice/voices")).send().await.unwrap().json().await.unwrap();
        assert!(listed["greetingDelayMs"].as_u64().is_some_and(|ms| ms <= voices::MAX_GREETING_DELAY_MS), "{listed}");
        assert_eq!(client.get(format!("{base}/api/voice/settings")).send().await.unwrap().status(), 200);
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

    // ---- messages the receptionist takes -------------------------------------------------

    struct Told(Mutex<Vec<String>>);

    impl crate::messages::MessageNotifier for Told {
        fn message_taken(&self, m: &crate::messages::Message) -> bool {
            self.0.lock().unwrap().push(m.message.clone());
            true
        }
    }

    /// A hub answering with message taking on (or off), its own store and a stand-in for the owner's notification.
    async fn serve_messages(messages_on: bool) -> (String, VoiceHub, Arc<Told>) {
        let hub = VoiceHub::new(Engines::at("http://127.0.0.1:9", "http://127.0.0.1:9"), |_| None);
        hub.set_ring(crate::ring::Ring::in_memory(crate::ring::RingSettings { take_messages: messages_on, ..Default::default() }));
        hub.set_messages(crate::messages::Store::in_memory());
        let told = Arc::new(Told(Mutex::new(Vec::new())));
        hub.set_message_notifier(Some(told.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = app_router(hub.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), hub, told)
    }

    #[tokio::test]
    async fn a_message_is_kept_with_the_number_this_desktop_saw_and_the_owner_is_told() {
        let _on = crate::modules::test_gate::enable(&[crate::modules::PHONE]);
        let (base, hub, told) = serve_messages(true).await;
        let client = reqwest::Client::new();
        let url = format!("{base}/api/voice/calls/call_1/message");
        // A call this desktop has not seen: nothing to take a message on.
        let none = client.post(&url).json(&json!({"message": "Ring me."})).send().await.unwrap();
        assert_eq!(none.status(), 404);
        assert_eq!(none.json::<Value>().await.unwrap()["error"]["code"], "no_call");

        hub.note_call("call_1", "+61491570006", "Alex Smith");
        let body = json!({"message": "Please ring me about Friday.", "callerName": "Sam", "callbackNumber": "0491 570 156", "urgency": "urgent", "wantsCallback": true});
        let ok = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(ok.status(), 200);
        let ok: Value = ok.json().await.unwrap();
        assert_eq!((ok["ok"].clone(), ok["result"]["recorded"].clone(), ok["result"]["notified"].clone()), (json!(true), json!(true), json!(true)));
        let kept = hub.messages().get(ok["result"]["id"].as_str().unwrap()).expect("kept");
        assert_eq!((kept.from.as_str(), kept.name.as_str(), kept.callback.as_str(), kept.call_id.as_str()), ("+61491570006", "Sam", "0491570156", "call_1"));
        assert_eq!((kept.urgency, kept.wants_callback), (crate::messages::Urgency::Urgent, true));
        assert_eq!(told.0.lock().unwrap().as_slice(), ["Please ring me about Friday.".to_string()]);

        // The number is the call's own: a body that names another one is refused, not believed.
        let spoof = client.post(&url).json(&json!({"message": "Hello", "from": "+61491570999"})).send().await.unwrap();
        assert_eq!(spoof.status(), 422);
        assert_eq!(hub.messages().list(None, "").len(), 1);
        // No name given: the phone's, when it is a name.
        let unnamed = client.post(&url).json(&json!({"message": "Second one."})).send().await.unwrap().json::<Value>().await.unwrap();
        assert_eq!(hub.messages().get(unnamed["result"]["id"].as_str().unwrap()).unwrap().name, "Alex Smith");
        // Unknown urgency, an empty message and unknown members are refused.
        assert_eq!(client.post(&url).json(&json!({"message": "x", "urgency": "critical"})).send().await.unwrap().status(), 400);
        assert_eq!(client.post(&url).json(&json!({"message": "  "})).send().await.unwrap().status(), 400);
        // The third message is the last on a call.
        assert_eq!(client.post(&url).json(&json!({"message": "Third."})).send().await.unwrap().status(), 200);
        let fourth = client.post(&url).json(&json!({"message": "Fourth."})).send().await.unwrap();
        assert_eq!(fourth.status(), 429);
        assert_eq!(fourth.json::<Value>().await.unwrap()["error"]["code"], "call_limit");
    }

    #[tokio::test]
    async fn a_message_can_be_left_as_the_call_ends_and_for_ten_minutes_after() {
        let _on = crate::modules::test_gate::enable(&[crate::modules::PHONE]);
        let (base, hub, _) = serve_messages(true).await;
        let client = reqwest::Client::new();
        hub.note_call("call_1", "+61491570006", "");
        hub.unregister("call_1");
        assert_eq!(hub.call_facts("call_1"), Some(("+61491570006".into(), String::new())), "ended, but lately");
        assert_eq!(client.post(format!("{base}/api/voice/calls/call_1/message")).json(&json!({"message": "Hung up mid-sentence."})).send().await.unwrap().status(), 200);
        // Ended more than ten minutes ago: gone.
        if let Some(long_ago) = Instant::now().checked_sub(RECORD_KEEP + Duration::from_secs(1)) {
            hub.inner.records.lock().unwrap().get_mut("call_1").unwrap().ended = Some(long_ago);
            assert_eq!(hub.call_facts("call_1"), None);
            assert_eq!(client.post(format!("{base}/api/voice/calls/call_1/message")).json(&json!({"message": "Too late."})).send().await.unwrap().status(), 404);
        }
    }

    #[tokio::test]
    async fn with_messages_off_nothing_is_kept_and_the_receptionist_is_told_so() {
        let _on = crate::modules::test_gate::enable(&[crate::modules::PHONE]);
        let (base, hub, told) = serve_messages(false).await;
        hub.note_call("call_1", "+61491570006", "");
        let resp = reqwest::Client::new().post(format!("{base}/api/voice/calls/call_1/message")).json(&json!({"message": "Ring me."})).send().await.unwrap();
        assert_eq!(resp.status(), 409);
        assert_eq!(resp.json::<Value>().await.unwrap()["error"]["code"], "messages_off");
        assert!(hub.messages().list(None, "").is_empty() && told.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn without_anything_to_tell_the_owner_with_the_message_is_kept_and_not_notified() {
        let _on = crate::modules::test_gate::enable(&[crate::modules::PHONE]);
        let (base, hub, _) = serve_messages(true).await;
        hub.set_message_notifier(None);
        crate::messages::set_notifier(None);
        hub.note_call("call_1", "+61491570006", "");
        let ok: Value = reqwest::Client::new().post(format!("{base}/api/voice/calls/call_1/message")).json(&json!({"message": "Ring me."})).send().await.unwrap().json().await.unwrap();
        assert_eq!((ok["result"]["recorded"].clone(), ok["result"]["notified"].clone()), (json!(true), json!(false)), "kept, but nobody could be told: the receptionist says saved, not that the owner will be told");
    }

    #[tokio::test]
    async fn the_message_route_answers_module_disabled_with_the_phone_off() {
        let _off = crate::modules::test_gate::enable(&[]);
        let (base, hub, _) = serve_messages(true).await;
        hub.note_call("call_1", "+61491570006", "");
        let resp = reqwest::Client::new().post(format!("{base}/api/voice/calls/call_1/message")).json(&json!({"message": "Ring me."})).send().await.unwrap();
        assert_eq!(resp.status(), 409);
        assert_eq!(resp.json::<Value>().await.unwrap()["error"]["code"], "module_disabled");
    }

    #[test]
    fn what_the_caller_said_is_kept_as_this_desktop_heard_it() {
        let hub = VoiceHub::new(Engines::at("http://127.0.0.1:9", "http://127.0.0.1:9"), |_| None);
        // A call this desktop has not seen keeps nothing.
        hub.note_turn("call_1", "hello");
        assert!(hub.caller_turns("call_1").is_empty());
        hub.note_call("call_1", "+61491570006", "Alex");
        for n in 1..=8 {
            hub.caller_said("call_1", &format!("turn {n}"), json!({"backchannel": false}));
        }
        hub.caller_said("call_1", "mm-hmm", json!({"backchannel": true}));
        let turns = hub.caller_turns("call_1");
        assert_eq!(turns.len(), TURNS_KEPT);
        assert_eq!((turns.first().map(String::as_str), turns.last().map(String::as_str)), (Some("turn 3"), Some("turn 8")), "the last six, and an acknowledgement is not a turn");
        assert!(hub.caller_turns("call_9").is_empty());
    }

    #[test]
    fn the_records_of_ended_calls_are_bounded() {
        let hub = VoiceHub::new(Engines::at("http://127.0.0.1:9", "http://127.0.0.1:9"), |_| None);
        for n in 0..(RECORDS_MAX + 20) {
            hub.note_call(&format!("call_{n}"), "+61491570006", "");
            hub.note_ended(&format!("call_{n}"));
        }
        hub.note_call("live", "+61491570156", "");
        assert!(hub.inner.records.lock().unwrap().len() <= RECORDS_MAX + 1);
        assert!(hub.call_facts("live").is_some(), "a live call is never let go");
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
