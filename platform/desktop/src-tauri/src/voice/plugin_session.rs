//! Ready-only local speech for an explicitly enabled, parent-owned plugin UI
//! session. No calls, model/service start, URLs, credentials or raw microphone
//! access are granted to the opaque plugin frame.
use crate::{plugins::PluginHost, services::registry::RegistryHandle};
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::watch;

const CAPABILITY: &str = "oaiy.voice.session";
const AUDIO_BYTES: usize = 480_044; // 15 seconds, 16 kHz mono PCM16 + canonical WAV header.
const PCM_BYTES: usize = 960_000; // 20 seconds, 24 kHz mono PCM16.
const UNAVAILABLE: &str = "OAIY Voice is not already loaded and ready. Check Services and start it explicitly when other work is idle.";
type Failure = (StatusCode, &'static str, &'static str);
type Valid = Arc<dyn Fn() -> bool + Send + Sync>;
type Key = (String, String, String); // plugin, mounted session, request
fn error(f: Failure) -> Response {
    (f.0, Json(json!({"error":{"code":f.1,"message":f.2}}))).into_response()
}
fn invalid() -> Failure {
    (
        StatusCode::BAD_REQUEST,
        "voice_invalid_request",
        "Supply the current voice session, a fresh bounded request ID and a bounded voice request.",
    )
}
fn id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b':'))
}
fn session_id(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok() && s.len() == 36
}
pub fn is_route(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    matches!(parts.as_slice(), ["","api","plugins",id,"voice","status"|"open"|"transcribe"|"speak"|"cancel"|"close"] if !id.is_empty())
}

trait Gate: Send + Sync {
    fn lease(&self, plugin: &str) -> Result<Valid, Failure>;
}
struct HostGate(Arc<PluginHost>);
impl Gate for HostGate {
    fn lease(&self, plugin: &str) -> Result<Valid, Failure> {
        let lease = self
            .0
            .screen_capability(plugin, CAPABILITY)
            .map_err(|(c, m)| (StatusCode::FORBIDDEN, c, m))?;
        let host = self.0.clone();
        let plugin = plugin.to_string();
        Ok(Arc::new(move || {
            host.holds_screen_capability(&plugin, CAPABILITY, &lease)
        }))
    }
}
#[derive(Clone)]
struct Service {
    base: String,
    valid: Valid,
}
trait Services: Send + Sync {
    fn current(&self) -> Option<Service>;
    /// Phone calls live now: a call has the voice engine, so plugin voice waits for it to end. The desktop asks every hub of
    /// the process; a test asks its own, so that a call another test holds in the same process is not this test's.
    fn live_calls(&self) -> usize {
        super::live_call_count()
    }
}
struct RegisteredServices(RegistryHandle);
impl Services for RegisteredServices {
    fn current(&self) -> Option<Service> {
        let lease = self
            .0
            .lock()
            .ok()?
            .running_service_lease(super::engines::STT_SERVICE)?;
        let registry = self.0.clone();
        Some(Service {
            base: format!("http://127.0.0.1:{}", lease.port),
            valid: Arc::new(move || {
                registry
                    .lock()
                    .is_ok_and(|r| r.holds_running_service(super::engines::STT_SERVICE, &lease))
            }),
        })
    }
}
struct Session {
    valid: Valid,
    service: Option<Service>,
    at: Instant,
}
#[derive(Default)]
struct Memory {
    sessions: HashMap<(String, String), Session>,
    pending: HashMap<Key, watch::Sender<bool>>,
    recent: HashMap<Key, Instant>,
    closed: HashMap<(String, String), Instant>,
}
impl Memory {
    fn prune(&mut self) {
        self.sessions
            .retain(|_, session| session.at.elapsed() < Duration::from_secs(900));
        self.recent
            .retain(|_, at| at.elapsed() < Duration::from_secs(120));
        self.closed
            .retain(|_, at| at.elapsed() < Duration::from_secs(120));
    }
}
#[derive(Clone)]
struct Voice {
    gate: Arc<dyn Gate>,
    services: Arc<dyn Services>,
    memory: Arc<Mutex<Memory>>,
    isolated: bool,
}
pub fn router(registry: RegistryHandle, host: Arc<PluginHost>, isolated: bool) -> Router {
    routes(Voice {
        gate: Arc::new(HostGate(host)),
        services: Arc::new(RegisteredServices(registry)),
        memory: Default::default(),
        isolated,
    })
}
fn routes(state: Voice) -> Router {
    Router::new()
        .route("/api/plugins/:id/voice/status", get(status))
        .route("/api/plugins/:id/voice/open", post(open))
        .route("/api/plugins/:id/voice/transcribe", post(transcribe))
        .route("/api/plugins/:id/voice/speak", post(speak))
        .route("/api/plugins/:id/voice/cancel", post(cancel))
        .route("/api/plugins/:id/voice/close", post(close))
        .layer(DefaultBodyLimit::max(768 * 1024))
        .with_state(state)
}
fn initiation(st: &Voice, plugin: &str) -> Result<Valid, Failure> {
    if st.isolated {
        return Err((
            StatusCode::FORBIDDEN,
            "voice_unavailable",
            "Shared voice services are unavailable in an isolated launch.",
        ));
    }
    st.gate.lease(plugin)
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SessionInput {
    session_id: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StatusInput {
    session_id: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RequestInput {
    session_id: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AudioInput {
    session_id: String,
    request_id: String,
    audio: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeakInput {
    session_id: String,
    request_id: String,
    text: String,
}

// Retain the original process/service lease beyond native download, so trusted
// parent playback cannot follow a stopped or replaced service into a new lease.
fn session_service(st: &Voice, plugin: &str, id: &str) -> Option<Service> {
    if st.services.live_calls() > 0 {
        return None;
    }
    let (valid, service) = st.memory.lock().ok().and_then(|memory| {
        memory
            .sessions
            .get(&(plugin.to_string(), id.to_string()))
            .filter(|session| session.at.elapsed() < Duration::from_secs(900))
            .map(|session| (session.valid.clone(), session.service.clone()))
    })?;
    service.filter(|service| valid() && (service.valid)())
}
async fn status(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    query: Result<Query<StatusInput>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let Query(input) = match query {
        Ok(input) => input,
        Err(_) => return error(invalid()),
    };
    if input.session_id.as_ref().is_some_and(|id| !session_id(id)) {
        return error(invalid());
    }
    let valid = match initiation(&st, &plugin) {
        Ok(v) => v,
        Err(e) => return error(e),
    };
    if let Some(id) = input.session_id {
        let result = if let Some(service) = session_service(&st, &plugin, &id) {
            tokio::time::timeout(Duration::from_secs(4), ready_service(&st, service, true))
                .await
                .ok()
                .and_then(Result::ok)
                .map(|(_, metadata, _)| metadata)
        } else {
            None
        };
        let live = valid() && session_service(&st, &plugin, &id).is_some();
        let mut metadata = result
            .filter(|_| live)
            .unwrap_or_else(|| json!({"sttReady":false,"ttsReady":false,"reason":UNAVAILABLE}));
        // Internal parent-only projection; never included in PluginHost.voice.
        metadata["sessionLeaseValid"] = json!(live);
        return Json(metadata).into_response();
    }
    let result = tokio::time::timeout(Duration::from_secs(4), ready(&st, true)).await;
    if !valid() {
        return error((
            StatusCode::FORBIDDEN,
            "capability_unavailable",
            "The plugin stopped or lost its voice capability.",
        ));
    }
    match result {
        Ok(Ok((_, metadata, _))) => Json(metadata).into_response(),
        _ => Json(json!({"sttReady":false,"ttsReady":false,"reason":UNAVAILABLE})).into_response(),
    }
}
async fn open(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    body: Result<Json<SessionInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(i) => i,
        Err(_) => return error(invalid()),
    };
    if !session_id(&input.session_id) {
        return error(invalid());
    }
    let valid = match initiation(&st, &plugin) {
        Ok(v) => v,
        Err(e) => return error(e),
    };
    let service = st.services.current();
    let mut memory = st.memory.lock().unwrap_or_else(|e| e.into_inner());
    memory.prune();
    if memory.sessions.len() >= 32
        || memory.closed.len() >= 512
        || memory
            .sessions
            .contains_key(&(plugin.clone(), input.session_id.clone()))
        || memory
            .closed
            .contains_key(&(plugin.clone(), input.session_id.clone()))
    {
        return error((
            StatusCode::CONFLICT,
            "voice_busy",
            "Close the current voice session before opening another.",
        ));
    }
    memory.sessions.insert(
        (plugin, input.session_id.clone()),
        Session {
            valid,
            service,
            at: Instant::now(),
        },
    );
    Json(json!({"sessionId":input.session_id})).into_response()
}
struct Claim {
    state: Voice,
    key: Key,
    rx: watch::Receiver<bool>,
    valid: Valid,
    service: Service,
    service_valid: Arc<Mutex<Option<Valid>>>,
}
impl Claim {
    fn live(&self) -> bool {
        !*self.rx.borrow()
            && self.state.services.live_calls() == 0
            && (self.valid)()
            && self
                .service_valid
                .lock()
                .is_ok_and(|s| s.as_ref().map_or(true, |valid| valid()))
            && self.state.memory.lock().is_ok_and(|m| {
                m.sessions
                    .contains_key(&(self.key.0.clone(), self.key.1.clone()))
            })
    }
}
impl Drop for Claim {
    fn drop(&mut self) {
        if let Ok(mut memory) = self.state.memory.lock() {
            memory.pending.remove(&self.key);
        }
    }
}
fn claim(st: &Voice, plugin: String, session: String, request: String) -> Result<Claim, Failure> {
    if !session_id(&session) || !id(&request) {
        return Err(invalid());
    }
    let _ = initiation(st, &plugin)?;
    let mut m = st.memory.lock().unwrap_or_else(|e| e.into_inner());
    m.prune();
    let saved = m
        .sessions
        .get(&(plugin.clone(), session.clone()))
        .filter(|session| (session.valid)())
        .ok_or((
            StatusCode::FORBIDDEN,
            "voice_session_unavailable",
            "Enable a current voice session before recording or speaking.",
        ))?;
    let valid = saved.valid.clone();
    let service = saved
        .service
        .clone()
        .filter(|service| (service.valid)())
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "voice_unavailable",
            UNAVAILABLE,
        ))?;
    let key = (plugin, session, request);
    if m.recent.contains_key(&key) {
        return Err((
            StatusCode::CONFLICT,
            "voice_request_repeated",
            "Use a fresh request ID for every voice attempt.",
        ));
    }
    if m.pending.len() >= 2 || m.pending.keys().any(|k| k.0 == key.0) || m.recent.len() >= 512 {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            "voice_busy",
            "A voice request is already active. Cancel it or wait for it to finish.",
        ));
    }
    let (tx, rx) = watch::channel(false);
    m.recent.insert(key.clone(), Instant::now());
    m.pending.insert(key.clone(), tx);
    Ok(Claim {
        state: st.clone(),
        key,
        rx,
        valid,
        service_valid: Arc::new(Mutex::new(Some(service.valid.clone()))),
        service,
    })
}
fn client() -> Result<reqwest::Client, Failure> {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            )
        })
}
async fn bounded_json(response: reqwest::Response) -> Result<Value, Failure> {
    if !response.status().is_success() {
        return Err((StatusCode::BAD_GATEWAY, "voice_unavailable", UNAVAILABLE));
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| {
            (
                StatusCode::BAD_GATEWAY,
                "voice_invalid_response",
                "The voice service returned an invalid bounded response.",
            )
        })?;
        if bytes.len() + chunk.len() > 16 * 1024 {
            return Err((
                StatusCode::BAD_GATEWAY,
                "voice_invalid_response",
                "The voice service returned an invalid bounded response.",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "voice_invalid_response",
            "The voice service returned invalid response data.",
        )
    })
}
async fn ready(st: &Voice, need_voice: bool) -> Result<(Service, Value, Option<String>), Failure> {
    if st.services.live_calls() > 0 {
        return Err((
            StatusCode::CONFLICT,
            "voice_busy",
            "OAIY is handling a phone call. Wait for it to finish before using plugin voice.",
        ));
    }
    let service = st.services.current().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "voice_unavailable",
        UNAVAILABLE,
    ))?;
    ready_service(st, service, need_voice).await
}
async fn ready_service(
    st: &Voice,
    service: Service,
    need_voice: bool,
) -> Result<(Service, Value, Option<String>), Failure> {
    if st.services.live_calls() > 0 {
        return Err((
            StatusCode::CONFLICT,
            "voice_busy",
            "OAIY is handling a phone call. Wait for it to finish before using plugin voice.",
        ));
    }
    if !(service.valid)() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "voice_unavailable",
            UNAVAILABLE,
        ));
    }
    let http = client()?;
    let health = bounded_json(
        http.get(format!("{}/health", service.base))
            .send()
            .await
            .map_err(|_| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "voice_unavailable",
                    UNAVAILABLE,
                )
            })?,
    )
    .await?;
    let lane = |name: &str| {
        health.get("status").and_then(Value::as_str) == Some("ok")
            && health.get(name).and_then(Value::as_bool) == Some(true)
            && health.get("lanes").map_or(true, |lanes| {
                lanes.get(name).is_some_and(|lane| {
                    lane.get("loadState").and_then(Value::as_str) == Some("loaded")
                        && lane.get("lastError").map_or(true, Value::is_null)
                        && lane.get("fallbackReason").map_or(true, Value::is_null)
                })
            })
    };
    let stt = lane("stt");
    let mut tts = lane("tts");
    let mut voice = None;
    if tts && need_voice {
        if let Ok(response) = http
            .get(format!("{}/v1/audio/voices", service.base))
            .send()
            .await
        {
            if let Ok(voices) = bounded_json(response).await {
                voice = voices
                    .get("default")
                    .and_then(Value::as_str)
                    .filter(|name| {
                        !name.trim().is_empty()
                            && name.len() <= 120
                            && !name.chars().any(char::is_control)
                            && voices
                                .get("voices")
                                .and_then(Value::as_array)
                                .is_some_and(|names| {
                                    names.len() <= 100
                                        && names.iter().any(|n| n.as_str() == Some(*name))
                                })
                    })
                    .map(String::from);
            }
        }
        tts = voice.is_some();
    }
    if !(service.valid)() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "voice_unavailable",
            UNAVAILABLE,
        ));
    }
    let metadata = json!({"sttReady":stt,"ttsReady":tts,"reason":if stt && tts {Value::Null} else {json!(UNAVAILABLE)}});
    Ok((service, metadata, voice))
}
fn canonical_audio(encoded: &str) -> Result<Vec<u8>, Failure> {
    if encoded.len() > ((AUDIO_BYTES + 2) / 3) * 4 {
        return Err(invalid());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| invalid())?;
    if bytes.len() <= 44
        || bytes.len() > AUDIO_BYTES
        || bytes.len() % 2 != 0
        || super::audio::wav_format(&bytes) != Some((16_000, 1, 16))
        || &bytes[0..4] != b"RIFF"
        || &bytes[8..36] != &super::audio::wav(&[], 16_000)[8..36]
        || &bytes[36..40] != b"data"
        || u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize != bytes.len() - 8
        || u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize != bytes.len() - 44
    {
        return Err(invalid());
    }
    Ok(bytes)
}
async fn perform<T>(
    claim: &mut Claim,
    seconds: u64,
    future: impl std::future::Future<Output = Result<T, Failure>>,
) -> Result<T, Failure> {
    let mut timer = tokio::time::interval(Duration::from_millis(250));
    tokio::pin!(future);
    let deadline = tokio::time::sleep(Duration::from_secs(seconds));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            biased;
            _ = claim.rx.changed() => return Err((StatusCode::CONFLICT,"voice_cancelled","The voice request was cancelled.")),
            _ = timer.tick() => if !claim.live() { return Err((StatusCode::FORBIDDEN,"capability_unavailable","The voice session stopped or lost permission.")); },
            _ = &mut deadline => return Err((StatusCode::GATEWAY_TIMEOUT,"voice_timeout","The bounded voice request timed out.")),
            result = &mut future => { if !claim.live() { return Err((StatusCode::CONFLICT,"voice_cancelled","The voice request was cancelled.")); } return result; }
        }
    }
}
async fn transcribe(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    body: Result<Json<AudioInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(i) => i,
        Err(_) => return error(invalid()),
    };
    let audio = match canonical_audio(&input.audio) {
        Ok(a) => a,
        Err(e) => return error(e),
    };
    let mut owned = match claim(&st, plugin, input.session_id, input.request_id.clone()) {
        Ok(c) => c,
        Err(e) => return error(e),
    };
    let allowed = owned.valid.clone();
    let session = owned.key.1.clone();
    let service_valid = owned.service_valid.clone();
    let original_service = owned.service.clone();
    let future = async {
        let (service, metadata, _) = tokio::time::timeout(
            Duration::from_secs(4),
            ready_service(&st, original_service, false),
        )
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            )
        })??;
        if metadata["sttReady"] != true || !allowed() || st.services.live_calls() > 0 {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            ));
        }
        *service_valid.lock().unwrap_or_else(|e| e.into_inner()) = Some(service.valid.clone());
        if !(service.valid)() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            ));
        }
        let http = client()?;
        let response = http.post(format!("{}/v1/audio/transcriptions",service.base))
            .json(&json!({"audio":base64::engine::general_purpose::STANDARD.encode(audio),"response_format":"json"})).send().await
            .map_err(|_|(StatusCode::BAD_GATEWAY,"voice_failed","The local voice service could not transcribe this recording."))?;
        let result = bounded_json(response).await?;
        if !(service.valid)() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            ));
        }
        let text = result
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty() && text.len() <= 8000 && text.chars().count() <= 2000)
            .ok_or((
                StatusCode::BAD_GATEWAY,
                "voice_invalid_response",
                "The voice service returned no bounded transcript.",
            ))?;
        Ok(json!({"sessionId":session,"requestId":input.request_id,"text":text}))
    };
    // Do not retain a borrow of the claim in the future, which cancellation monitors.
    let result = perform(&mut owned, 20, future).await;
    match result {
        Ok(value) => Json(value).into_response(),
        Err(e) => error(e),
    }
}
async fn speak(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    body: Result<Json<SpeakInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(i) => i,
        Err(_) => return error(invalid()),
    };
    if input.text.trim().is_empty()
        || input.text.len() > 3200
        || input.text.chars().count() > 800
        || input
            .text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return error(invalid());
    }
    let mut owned = match claim(&st, plugin, input.session_id, input.request_id.clone()) {
        Ok(c) => c,
        Err(e) => return error(e),
    };
    let allowed = owned.valid.clone();
    let started = Instant::now();
    let service_valid = owned.service_valid.clone();
    let original_service = owned.service.clone();
    let future = async {
        let (service, metadata, voice) = tokio::time::timeout(
            Duration::from_secs(4),
            ready_service(&st, original_service, true),
        )
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            )
        })??;
        if metadata["ttsReady"] != true || !allowed() || st.services.live_calls() > 0 {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            ));
        }
        *service_valid.lock().unwrap_or_else(|e| e.into_inner()) = Some(service.valid.clone());
        if !(service.valid)() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "voice_unavailable",
                UNAVAILABLE,
            ));
        }
        let response = client()?
            .post(format!("{}/v1/audio/speech", service.base))
            .json(&json!({"input":input.text,"voice":voice,"response_format":"pcm"}))
            .send()
            .await
            .map_err(|_| {
                (
                    StatusCode::BAD_GATEWAY,
                    "voice_failed",
                    "The local voice service could not speak this text.",
                )
            })?;
        if !response.status().is_success()
            || response
                .headers()
                .get("x-sample-rate")
                .and_then(|v| v.to_str().ok())
                != Some("24000")
            || response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.split(';').next())
                != Some("audio/pcm")
        {
            return Err((
                StatusCode::BAD_GATEWAY,
                "voice_invalid_response",
                "The voice service returned invalid speech audio.",
            ));
        }
        Ok((response, service))
    };
    let (response, service) = match perform(&mut owned, 30, future).await {
        Ok(v) => v,
        Err(e) => return error(e),
    };
    let stream = response.bytes_stream().boxed();
    let body = futures_util::stream::unfold(
        (stream, owned, service, 0usize, false),
        move |(mut stream, mut claim, service, total, done)| async move {
            if done {
                return None;
            }
            let invalid = || std::io::Error::other("The bounded voice stream stopped.");
            let mut tick = tokio::time::interval(Duration::from_millis(250));
            let next = loop {
                tokio::select! {
                    biased;
                    _ = claim.rx.changed() => break Err(invalid()),
                    _ = tick.tick() => if !claim.live() || !(service.valid)() || started.elapsed()>Duration::from_secs(30) {break Err(invalid());},
                    chunk = stream.next() => break match chunk {Some(Ok(bytes))=>Ok(Some(bytes)),Some(Err(_))=>Err(invalid()),None=>Ok(None)}
                }
            };
            match next {
                Ok(Some(bytes))
                    if total + bytes.len() <= PCM_BYTES && total + bytes.len() <= 1024 * 1024 =>
                {
                    Some((
                        Ok::<Bytes, std::io::Error>(bytes.clone()),
                        (stream, claim, service, total + bytes.len(), false),
                    ))
                }
                Ok(None) if total > 0 && total % 2 == 0 => None,
                _ => Some((Err(invalid()), (stream, claim, service, total, true))),
            }
        },
    );
    (
        [
            (axum::http::header::CONTENT_TYPE, "audio/pcm"),
            (
                axum::http::HeaderName::from_static("x-sample-rate"),
                "24000",
            ),
        ],
        Body::from_stream(body),
    )
        .into_response()
}
async fn cancel(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    body: Result<Json<RequestInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(i) => i,
        Err(_) => return error(invalid()),
    };
    if !session_id(&input.session_id) || !id(&input.request_id) {
        return error(invalid());
    }
    let key = (plugin, input.session_id.clone(), input.request_id.clone());
    let mut memory = st.memory.lock().unwrap_or_else(|e| e.into_inner());
    memory.prune();
    if memory.recent.len() >= 512 && !memory.recent.contains_key(&key) {
        return error((
            StatusCode::TOO_MANY_REQUESTS,
            "voice_busy",
            "The bounded voice request window is full.",
        ));
    }
    let cancelled = memory
        .pending
        .get(&key)
        .map(|tx| {
            tx.send_replace(true);
            true
        })
        .unwrap_or(false);
    memory.recent.entry(key).or_insert_with(Instant::now);
    Json(json!({"sessionId":input.session_id,"requestId":input.request_id,"cancelled":cancelled}))
        .into_response()
}
async fn close(
    State(st): State<Voice>,
    Path(plugin): Path<String>,
    body: Result<Json<SessionInput>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(i) => i,
        Err(_) => return error(invalid()),
    };
    if !session_id(&input.session_id) {
        return error(invalid());
    }
    let mut memory = st.memory.lock().unwrap_or_else(|e| e.into_inner());
    memory.prune();
    for (key, tx) in &memory.pending {
        if key.0 == plugin && key.1 == input.session_id {
            tx.send_replace(true);
        }
    }
    let key = (plugin, input.session_id.clone());
    let closed = memory.sessions.remove(&key).is_some();
    if memory.closed.len() < 512 {
        memory.closed.entry(key).or_insert_with(Instant::now);
    }
    Json(json!({"sessionId":input.session_id,"closed":closed})).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tower::ServiceExt;
    const SESSION: &str = "12345678-1234-4321-8123-123456789abc";
    struct TestGate(Arc<AtomicBool>);
    impl Gate for TestGate {
        fn lease(&self, _: &str) -> Result<Valid, Failure> {
            if !self.0.load(Ordering::SeqCst) {
                return Err((
                    StatusCode::FORBIDDEN,
                    "capability_denied",
                    "Test voice capability denied.",
                ));
            }
            let valid = self.0.clone();
            Ok(Arc::new(move || valid.load(Ordering::SeqCst)))
        }
    }
    struct TestServices {
        service: Service,
        available: Arc<AtomicBool>,
        probes: Arc<AtomicUsize>,
    }
    impl Services for TestServices {
        fn current(&self) -> Option<Service> {
            self.probes.fetch_add(1, Ordering::SeqCst);
            self.available
                .load(Ordering::SeqCst)
                .then(|| self.service.clone())
        }
        // No call is live in these tests. The process-wide count would see the calls that other tests of the binary hold
        // while these run beside them, and every request here would then be refused as busy.
        fn live_calls(&self) -> usize {
            0
        }
    }
    struct Fixture {
        state: Voice,
        app: Router,
        health: Arc<Mutex<Value>>,
        voices: Arc<Mutex<Value>>,
        gate: Arc<AtomicBool>,
        available: Arc<AtomicBool>,
        valid: Arc<AtomicBool>,
        probes: Arc<AtomicUsize>,
        posted: Arc<Mutex<Vec<(String, Value)>>>,
        mode: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.server.abort();
        }
    }
    async fn fixture() -> Fixture {
        let health = Arc::new(Mutex::new(json!({"status":"ok","stt":true,"tts":true})));
        let voices = Arc::new(Mutex::new(
            json!({"voices":["Synthetic"],"default":"Synthetic"}),
        ));
        let posted: Arc<Mutex<Vec<(String, Value)>>> = Default::default();
        let mode = Arc::new(AtomicUsize::new(0));
        let h = health.clone();
        let v = voices.clone();
        let p = posted.clone();
        let m = mode.clone();
        let p2 = posted.clone();
        let m2 = mode.clone();
        let hm = mode.clone();
        let service = Router::new()
            .route(
                "/health",
                get(move || {
                    let h = h.clone();
                    let hm = hm.clone();
                    async move {
                        match hm.load(Ordering::SeqCst) {
                            6 => (
                                StatusCode::TEMPORARY_REDIRECT,
                                [("location", "/v1/audio/transcriptions")],
                            )
                                .into_response(),
                            7 => std::future::pending::<Response>().await,
                            _ => Json(h.lock().unwrap().clone()).into_response(),
                        }
                    }
                }),
            )
            .route(
                "/v1/audio/voices",
                get(move || {
                    let v = v.clone();
                    async move { Json(v.lock().unwrap().clone()) }
                }),
            )
            .route(
                "/v1/audio/transcriptions",
                post(move |Json(body): Json<Value>| {
                    let p = p.clone();
                    let m = m.clone();
                    async move {
                        p.lock().unwrap().push(("stt".into(), body));
                        match m.load(Ordering::SeqCst) {
                            1 => std::future::pending::<Response>().await,
                            2 => (
                                StatusCode::BAD_REQUEST,
                                Json(json!({"error":"secret upstream detail"})),
                            )
                                .into_response(),
                            3 => Json(json!({"text":"x".repeat(2001)})).into_response(),
                            _ => Json(json!({"text":"  Synthetic transcript  "})).into_response(),
                        }
                    }
                }),
            )
            .route(
                "/v1/audio/speech",
                post(move |Json(body): Json<Value>| {
                    let p = p2.clone();
                    let m = m2.clone();
                    async move {
                        p.lock().unwrap().push(("tts".into(), body));
                        let bytes = match m.load(Ordering::SeqCst) {
                            4 => vec![0; PCM_BYTES + 2],
                            5 => vec![0],
                            _ => vec![1, 0, 2, 0],
                        };
                        (
                            [("content-type", "audio/pcm"), ("x-sample-rate", "24000")],
                            bytes,
                        )
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, service).await.unwrap();
        });
        let gate = Arc::new(AtomicBool::new(true));
        let available = Arc::new(AtomicBool::new(true));
        let valid = Arc::new(AtomicBool::new(true));
        let probes = Arc::new(AtomicUsize::new(0));
        let current = valid.clone();
        let running = available.clone();
        let state = Voice {
            gate: Arc::new(TestGate(gate.clone())),
            services: Arc::new(TestServices {
                service: Service {
                    base,
                    valid: Arc::new(move || {
                        current.load(Ordering::SeqCst) && running.load(Ordering::SeqCst)
                    }),
                },
                available: available.clone(),
                probes: probes.clone(),
            }),
            memory: Default::default(),
            isolated: false,
        };
        Fixture {
            app: routes(state.clone()),
            state,
            health,
            voices,
            gate,
            available,
            valid,
            probes,
            posted,
            mode,
            server,
        }
    }
    async fn request(app: Router, method: &str, action: &str, body: Value) -> Response {
        app.oneshot(
            Request::builder()
                .method(method)
                .uri(format!("/api/plugins/probe/voice/{action}"))
                .header("content-type", "application/json")
                .body(if method == "GET" {
                    Body::empty()
                } else {
                    Body::from(body.to_string())
                })
                .unwrap(),
        )
        .await
        .unwrap()
    }
    async fn ask(app: Router, method: &str, action: &str, body: Value) -> (StatusCode, Value) {
        let response = request(app, method, action, body).await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    fn audio(request: &str) -> Value {
        json!({"sessionId":SESSION,"requestId":request,"audio":base64::engine::general_purpose::STANDARD.encode(super::super::audio::wav(&[0,3277],16000))})
    }
    fn speech(request: &str) -> Value {
        json!({"sessionId":SESSION,"requestId":request,"text":"Synthetic grounded text."})
    }
    async fn opened(f: &Fixture) {
        assert_eq!(
            ask(f.app.clone(), "POST", "open", json!({"sessionId":SESSION}))
                .await
                .0,
            StatusCode::OK
        );
    }
    async fn started(f: &Fixture) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while f.posted.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn ready_only_routes_send_canonical_audio_and_host_chosen_voice() {
        let f = fixture().await;
        opened(&f).await;
        assert_eq!(
            ask(f.app.clone(), "GET", "status", Value::Null).await.1,
            json!({"sttReady":true,"ttsReady":true,"reason":null})
        );
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("stt"))
                .await
                .1,
            json!({"sessionId":SESSION,"requestId":"stt","text":"Synthetic transcript"})
        );
        let response = request(f.app.clone(), "POST", "speak", speech("tts")).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            axum::body::to_bytes(response.into_body(), PCM_BYTES)
                .await
                .unwrap()
                .as_ref(),
            &[1, 0, 2, 0]
        );
        let posted = f.posted.lock().unwrap();
        assert_eq!(posted.len(), 2);
        assert_eq!(
            posted[0].1,
            json!({"audio":audio("unused")["audio"],"response_format":"json"})
        );
        assert_eq!(
            posted[1].1,
            json!({"input":"Synthetic grounded text.","voice":"Synthetic","response_format":"pcm"})
        );
        assert!(f.state.memory.lock().unwrap().pending.is_empty());
    }
    #[tokio::test]
    async fn scoped_status_retains_original_lease_after_download_and_never_adopts_replacement() {
        let f = fixture().await;
        opened(&f).await;
        assert_eq!(
            request(f.app.clone(), "POST", "speak", speech("downloaded"))
                .await
                .status(),
            StatusCode::OK
        );
        assert!(f.state.memory.lock().unwrap().pending.is_empty());
        let action = format!("status?sessionId={SESSION}");
        assert_eq!(
            ask(f.app.clone(), "GET", &action, Value::Null).await.1["sessionLeaseValid"],
            true
        );
        f.valid.store(false, Ordering::SeqCst);
        let mut replacement = f.state.clone();
        let mut service = replacement.services.current().unwrap();
        service.valid = Arc::new(|| true);
        let new_probes = Arc::new(AtomicUsize::new(0));
        replacement.services = Arc::new(TestServices {
            service,
            available: Arc::new(AtomicBool::new(true)),
            probes: new_probes.clone(),
        });
        let app = routes(replacement);
        assert_eq!(
            ask(app.clone(), "GET", "status", Value::Null).await.1["ttsReady"],
            true
        );
        assert_eq!(new_probes.load(Ordering::SeqCst), 1);
        let (_, metadata) = ask(app.clone(), "GET", &action, Value::Null).await;
        assert_eq!(metadata["sessionLeaseValid"], false);
        assert_eq!(metadata["ttsReady"], false);
        assert_eq!(
            new_probes.load(Ordering::SeqCst),
            1,
            "scoped status must not probe a replacement service"
        );
        assert_eq!(
            request(app, "POST", "speak", speech("replacement"))
                .await
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            f.posted.lock().unwrap().len(),
            1,
            "no second inference on a replacement lease"
        );
    }
    #[tokio::test]
    async fn scoped_status_distinguishes_own_busy_health_from_lost_or_closed_session() {
        let f = fixture().await;
        opened(&f).await;
        *f.health.lock().unwrap() = json!({"status":"ok","stt":true,"tts":true,"lanes":{"stt":{"loadState":"busy"},"tts":{"loadState":"busy"}}});
        let action = format!("status?sessionId={SESSION}");
        let (_, metadata) = ask(f.app.clone(), "GET", &action, Value::Null).await;
        assert_eq!(metadata["sttReady"], false);
        assert_eq!(metadata["ttsReady"], false);
        assert_eq!(
            metadata["sessionLeaseValid"], true,
            "own engine busy is not lease revocation"
        );
        assert_eq!(
            ask(f.app.clone(), "GET", "status", Value::Null)
                .await
                .1
                .as_object()
                .unwrap()
                .len(),
            3
        );
        ask(f.app.clone(), "POST", "close", json!({"sessionId":SESSION})).await;
        assert_eq!(
            ask(f.app.clone(), "GET", &action, Value::Null).await.1["sessionLeaseValid"],
            false
        );
        assert_eq!(
            ask(
                f.app.clone(),
                "GET",
                "status?sessionId=not-a-uuid",
                Value::Null
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ask(f.app.clone(), "GET", "status?extra=1", Value::Null)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
        assert!(f.posted.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn stopped_unloaded_busy_fallback_and_missing_voice_never_post() {
        let f = fixture().await;
        opened(&f).await;
        f.available.store(false, Ordering::SeqCst);
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("stopped"))
                .await
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        f.available.store(true, Ordering::SeqCst);
        for (i, state) in ["not_loaded", "busy", "failed"].iter().enumerate() {
            *f.health.lock().unwrap() = json!({"status":"ok","stt":true,"tts":true,"lanes":{"stt":{"loadState":state},"tts":{"loadState":state}}});
            assert_eq!(
                ask(
                    f.app.clone(),
                    "POST",
                    "transcribe",
                    audio(&format!("lane{i}"))
                )
                .await
                .0,
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(
                ask(f.app.clone(), "POST", "speak", speech(&format!("tts{i}")))
                    .await
                    .0,
                StatusCode::SERVICE_UNAVAILABLE
            );
        }
        *f.health.lock().unwrap() = json!({"status":"ok","stt":true,"tts":true,"lanes":{"stt":{"loadState":"loaded","lastError":"secret error"},"tts":{"loadState":"loaded","fallbackReason":"other engine"}}});
        let (_, metadata) = ask(f.app.clone(), "GET", "status", Value::Null).await;
        assert_eq!(metadata["sttReady"], false);
        assert_eq!(metadata["ttsReady"], false);
        assert!(!metadata.to_string().contains("secret"));
        *f.health.lock().unwrap() = json!({"status":"ok","stt":true,"tts":true});
        *f.voices.lock().unwrap() = json!({"voices":[],"default":"missing"});
        let (_, metadata) = ask(f.app.clone(), "GET", "status", Value::Null).await;
        assert_eq!(metadata["sttReady"], true);
        assert_eq!(metadata["ttsReady"], false);
        assert_eq!(
            ask(f.app.clone(), "POST", "speak", speech("missing-voice"))
                .await
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(f.posted.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn isolation_and_untrusted_gate_do_not_even_probe_a_service() {
        let f = fixture().await;
        f.gate.store(false, Ordering::SeqCst);
        assert_eq!(
            ask(f.app.clone(), "GET", "status", Value::Null).await.0,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            ask(f.app.clone(), "POST", "open", json!({"sessionId":SESSION}))
                .await
                .0,
            StatusCode::FORBIDDEN
        );
        let mut isolated = f.state.clone();
        isolated.isolated = true;
        f.gate.store(true, Ordering::SeqCst);
        assert_eq!(
            ask(routes(isolated), "GET", "status", Value::Null).await.0,
            StatusCode::FORBIDDEN
        );
        assert_eq!(f.probes.load(Ordering::SeqCst), 0);
        assert!(f.posted.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn bounded_closed_requests_replays_and_private_errors() {
        let f = fixture().await;
        opened(&f).await;
        let mut url = audio("bad");
        url["url"] = json!("http://external");
        let mut undecodable = audio("bad");
        undecodable["audio"] = json!("not-base64");
        let mut stale = audio("bad");
        stale["sessionId"] = json!("invalid-session");
        for body in [url, undecodable, stale] {
            assert_eq!(
                ask(f.app.clone(), "POST", "transcribe", body).await.0,
                StatusCode::BAD_REQUEST
            );
        }
        let mut too_long = speech("long");
        too_long["text"] = json!("x".repeat(801));
        assert_eq!(
            ask(f.app.clone(), "POST", "speak", too_long).await.0,
            StatusCode::BAD_REQUEST
        );
        let mut invalid = audio("format");
        invalid["audio"] =
            json!(base64::engine::general_purpose::STANDARD
                .encode(super::super::audio::wav(&[0], 24000)));
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", invalid).await.0,
            StatusCode::BAD_REQUEST
        );
        f.mode.store(2, Ordering::SeqCst);
        let (_, error) = ask(f.app.clone(), "POST", "transcribe", audio("secret")).await;
        assert!(!error.to_string().contains("secret upstream"));
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("secret"))
                .await
                .0,
            StatusCode::CONFLICT
        );
        f.mode.store(3, Ordering::SeqCst);
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("oversized"))
                .await
                .0,
            StatusCode::BAD_GATEWAY
        );
    }
    #[tokio::test]
    async fn cancellation_revocation_service_replacement_and_close_drop_inflight_results() {
        for reason in ["cancel", "revoke", "service", "close"] {
            let f = fixture().await;
            opened(&f).await;
            f.mode.store(1, Ordering::SeqCst);
            let app = f.app.clone();
            let running =
                tokio::spawn(async move { ask(app, "POST", "transcribe", audio("waiting")).await });
            started(&f).await;
            match reason {
                "cancel" => {
                    assert_eq!(
                        ask(
                            f.app.clone(),
                            "POST",
                            "cancel",
                            json!({"sessionId":SESSION,"requestId":"waiting"})
                        )
                        .await
                        .1["cancelled"],
                        true
                    );
                }
                "revoke" => {
                    f.gate.store(false, Ordering::SeqCst);
                }
                "service" => {
                    f.valid.store(false, Ordering::SeqCst);
                }
                _ => {
                    assert_eq!(
                        ask(f.app.clone(), "POST", "close", json!({"sessionId":SESSION}))
                            .await
                            .1["closed"],
                        true
                    );
                }
            }
            let (code, body) = tokio::time::timeout(Duration::from_secs(2), running)
                .await
                .unwrap()
                .unwrap();
            assert!(!code.is_success());
            assert!(body.get("text").is_none());
            assert!(f.state.memory.lock().unwrap().pending.is_empty());
        }
    }
    #[tokio::test]
    async fn cancellation_before_arrival_and_concurrency_are_bounded() {
        let f = fixture().await;
        opened(&f).await;
        ask(
            f.app.clone(),
            "POST",
            "cancel",
            json!({"sessionId":SESSION,"requestId":"early"}),
        )
        .await;
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("early"))
                .await
                .0,
            StatusCode::CONFLICT
        );
        f.mode.store(1, Ordering::SeqCst);
        let app = f.app.clone();
        let running =
            tokio::spawn(async move { ask(app, "POST", "transcribe", audio("waiting")).await });
        started(&f).await;
        assert_eq!(
            ask(f.app.clone(), "POST", "speak", speech("parallel"))
                .await
                .0,
            StatusCode::TOO_MANY_REQUESTS
        );
        ask(f.app.clone(), "POST", "close", json!({"sessionId":SESSION})).await;
        running.await.unwrap();
    }
    #[tokio::test]
    async fn close_before_open_cannot_revive_a_late_document_session() {
        let f = fixture().await;
        assert_eq!(
            ask(f.app.clone(), "POST", "close", json!({"sessionId":SESSION}))
                .await
                .1["closed"],
            false
        );
        assert_eq!(
            ask(f.app.clone(), "POST", "open", json!({"sessionId":SESSION}))
                .await
                .0,
            StatusCode::CONFLICT
        );
        assert!(f.state.memory.lock().unwrap().sessions.is_empty());
    }
    #[tokio::test]
    async fn redirects_and_stalled_health_are_bounded_without_posting() {
        let f = fixture().await;
        opened(&f).await;
        f.mode.store(6, Ordering::SeqCst);
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("redirect"))
                .await
                .0,
            StatusCode::BAD_GATEWAY
        );
        f.mode.store(7, Ordering::SeqCst);
        let start = Instant::now();
        assert_eq!(
            ask(f.app.clone(), "POST", "transcribe", audio("stalled"))
                .await
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(start.elapsed() < Duration::from_secs(6));
        assert!(f.posted.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn streamed_pcm_rejects_oversize_and_incomplete_samples() {
        for mode in [4, 5] {
            let f = fixture().await;
            opened(&f).await;
            f.mode.store(mode, Ordering::SeqCst);
            let response = request(f.app.clone(), "POST", "speak", speech("invalid-audio")).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert!(axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .is_err());
            assert!(f.state.memory.lock().unwrap().pending.is_empty());
        }
    }
    #[test]
    fn canonical_audio_has_a_real_duration_and_header_bound() {
        let encode = |samples, rate| {
            base64::engine::general_purpose::STANDARD
                .encode(super::super::audio::wav(&vec![0; samples], rate))
        };
        assert_eq!(
            canonical_audio(&encode(240_000, 16000)).unwrap().len(),
            AUDIO_BYTES
        );
        assert!(canonical_audio(&encode(240_001, 16000)).is_err());
        assert!(canonical_audio(&encode(1, 24000)).is_err());
        assert!(canonical_audio(&encode(0, 16000)).is_err());
        let mut malformed = super::super::audio::wav(&[0], 16000);
        malformed[40] = 0;
        assert!(
            canonical_audio(&base64::engine::general_purpose::STANDARD.encode(malformed)).is_err()
        );
    }
}
