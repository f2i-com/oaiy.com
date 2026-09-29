//! HTTP surface for the AI gateway — `/api/ai/*`.
//!
//! Mounted as its own axum `Router` with its own [`AiState`] (mirroring the
//! bridge), so `http.rs` only merges it. Provider config/credential routes are
//! management-plane; the chat/models gateway is credential-hidden — the browser
//! sends only its pairing token and OAIY attaches the provider key server-side.
//!
//! FormLogic's client sends an `X-FormLogic-Capability` header on `ai.*` calls;
//! OAIY has no such guard and simply IGNORES it — auth is the bearer/pairing
//! token checked in `http.rs`'s `origin_guard`.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::gateway::{self, GatewayError};
use super::providers::{AiProvider, AiProviderInput, AiProviderPublic, Capability, Protocol, ProviderStoreHandle};
use crate::services::registry::{RegistryHandle, ServiceStatus};

/// Shared state for the AI router (mirrors `BridgeState`): the provider store
/// plus an Arc clone of the services registry (the services half of the sources
/// union — the registry itself lives in `AppState`).
#[derive(Clone)]
pub struct AiState {
    pub providers: ProviderStoreHandle,
    pub registry: RegistryHandle,
    /// The ChatGPT connector: a managed `codex` CLI child that owns its own
    /// OAuth. Not in the provider store — it has no API key to hold.
    pub codex: super::codex::CodexHandle,
}

/// A provider's chat and model list, for the voice gateway on 17872 (behind its token):
/// where Aokie's own speech lanes ask for a reply when they pick one of OAIY's
/// providers (`provider:<id>`), as FormLogic's receptionist set them up.
pub fn provider_chat_router(state: AiState) -> Router {
    Router::new()
        .route("/api/ai/providers/:id/v1/models", get(models_for))
        .route("/api/ai/providers/:id/v1/chat/completions", post(chat_for))
        .with_state(state)
}

/// OAIY's own engine, as a provider: its gateway, answering with the model chosen in
/// Engines. Not in the store: it is there whenever the engines are, and follows them.
pub const ENGINE_PROVIDER_ID: &str = "oaiy-engine";

/// The engines' gateway and the model chosen in Engines, asked of the engines now
/// (their control page's state gives the gateway, the gateway's discovery the model).
async fn engine_now() -> Result<(String, String), String> {
    let (gateway, discovery) = engine_discovery().await?;
    Ok((gateway, chosen_model(&discovery)))
}

/// The engines' gateway and its discovery document, asked of the engines now. Only
/// asks: nothing is started (the gateway answers discovery from its configuration).
pub(super) async fn engine_discovery() -> Result<(String, Value), String> {
    let ui = engines_ui().ok_or("OAIY's engines are not running")?;
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(4)).build().map_err(|e| e.to_string())?;
    let state: Value = client.get(format!("{}/api/state", ui.trim_end_matches('/'))).send().await.map_err(|e| format!("the engines did not answer: {e}"))?.json().await.map_err(|e| e.to_string())?;
    let gateway = state.get("gateway_url").and_then(Value::as_str).filter(|g| !g.is_empty()).ok_or("the engines have no gateway yet")?.trim_end_matches('/').to_string();
    let discovery: Value = client.get(format!("{gateway}/v1/discovery")).send().await.map_err(|e| format!("the engines' gateway did not answer: {e}"))?.json().await.map_err(|e| e.to_string())?;
    Ok((gateway, discovery))
}

/// The model chosen in Engines, from the gateway's discovery: `defaults.llm`, else
/// the language model marked default ("" when there is none).
pub(super) fn chosen_model(discovery: &Value) -> String {
    discovery
        .pointer("/defaults/llm")
        .and_then(Value::as_str)
        .or_else(|| discovery.pointer("/models/llm").and_then(Value::as_array).and_then(|m| m.iter().find(|m| m.get("default").and_then(Value::as_bool) == Some(true))).and_then(|m| m.get("id")).and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

/// Where the engines' control pages are: the window runs the engines; the headless server has none.
#[cfg(feature = "gui")]
fn engines_ui() -> Option<String> {
    crate::engines::ui_url()
}
#[cfg(not(feature = "gui"))]
fn engines_ui() -> Option<String> {
    None
}

/// The engine as a provider record the gateway code can forward to.
fn engine_provider(gateway: String, model: String) -> AiProvider {
    AiProvider {
        id: ENGINE_PROVIDER_ID.into(),
        name: "OAIY engine".into(),
        category: Some("LLM".into()),
        protocol: Protocol::OpenAi,
        base_url: gateway,
        model: (!model.is_empty()).then_some(model),
        capabilities: vec![Capability::Chat],
        enabled: true,
        allow_local: true,
        api_key: None,
    }
}

pub fn router(state: AiState) -> Router {
    // OAIY's engine's models as flow services, and the calls to them.
    let engine = super::engine_services::router(state.clone());
    // The Agent's model (engine or ChatGPT), and what this computer can run.
    let agent_model = super::agent_model::router(state.clone());
    Router::new()
        // union of local services + configured providers, for the flow pickers
        .route("/api/ai/sources", get(list_ai_sources))
        // provider CRUD + credential + reachability probe (management-plane)
        .route("/api/ai/providers", get(list_ai_providers).post(upsert_ai_provider))
        .route("/api/ai/providers/:id", delete(delete_ai_provider))
        .route("/api/ai/providers/:id/key", post(set_ai_provider_key))
        .route("/api/ai/providers/:id/test", post(test_ai_provider))
        // gateway — default provider
        .route("/api/ai/v1/models", get(models_default))
        .route("/api/ai/v1/chat/completions", post(chat_default))
        // gateway — named provider (key injected server-side by id)
        .route("/api/ai/providers/:id/v1/models", get(models_for))
        .route("/api/ai/providers/:id/v1/chat/completions", post(chat_for))
        // ChatGPT connector (the managed codex agent): sign-in lifecycle. The
        // chat/models paths reuse the provider routes above via its provider id.
        .route("/api/ai/codex/status", get(codex_status))
        .route("/api/ai/codex/login", post(codex_login_start).delete(codex_login_cancel))
        .route("/api/ai/codex/logout", post(codex_logout))
        .with_state(state)
        .merge(engine)
        .merge(agent_model)
    // OUT OF SCOPE v1 (hook here later, same shape as the reference):
    //   POST /api/ai/v1/audio/transcriptions      -> Capability::Transcription
    //   POST /api/ai/v1/audio/chat/completions     -> buffered audio-chat
    //   POST /api/ai/v1/realtime/sessions          -> Capability::Realtime (WebRTC SDP)
    //   ...and the /api/ai/providers/:id/v1/... twins of each.
}

/// `{ "error": { "code", "message" } }` — identical taxonomy shape to the bridge.
pub(super) fn ai_error(status: StatusCode, code: &str, message: String) -> Response {
    (status, Json(json!({ "error": { "code": code, "message": message } }))).into_response()
}

fn gateway_err(e: GatewayError) -> Response {
    let status = match e {
        GatewayError::NoProvider(_) => StatusCode::NOT_FOUND,
        GatewayError::BadRequest(_) => StatusCode::BAD_REQUEST,
        GatewayError::Upstream(_) => StatusCode::BAD_GATEWAY,
    };
    ai_error(status, e.code(), e.message().to_string())
}

// ---------------------------------------------------------------------------
// Provider CRUD
// ---------------------------------------------------------------------------

async fn list_ai_providers(State(st): State<AiState>) -> Response {
    let store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
    (StatusCode::OK, Json(json!({ "providers": store.list() }))).into_response()
}

async fn upsert_ai_provider(State(st): State<AiState>, Json(input): Json<AiProviderInput>) -> Response {
    let mut store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
    match store.upsert(input) {
        Ok(id) => (StatusCode::OK, Json(json!({ "id": id }))).into_response(),
        Err(e) => ai_error(StatusCode::BAD_REQUEST, "invalid_request", e),
    }
}

async fn delete_ai_provider(State(st): State<AiState>, Path(id): Path<String>) -> Response {
    let mut store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
    match store.delete(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ai_error(StatusCode::NOT_FOUND, "invalid_request", e),
    }
}

#[derive(Deserialize)]
struct SetKeyBody {
    #[serde(default)]
    key: Option<String>,
}

async fn set_ai_provider_key(
    State(st): State<AiState>,
    Path(id): Path<String>,
    Json(body): Json<SetKeyBody>,
) -> Response {
    let mut store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
    match store.set_key(&id, body.key.as_deref()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ai_error(StatusCode::NOT_FOUND, "invalid_request", e),
    }
}

async fn test_ai_provider(State(st): State<AiState>, Path(id): Path<String>) -> Response {
    // Clone the FULL record out UNDER the lock and drop the guard before await:
    // a std MutexGuard held across `.await` makes the future !Send.
    let provider = { st.providers.lock().unwrap_or_else(|e| e.into_inner()).get_full(&id) };
    let Some(p) = provider else {
        return ai_error(StatusCode::NOT_FOUND, "invalid_request", format!("unknown provider {id:?}"));
    };
    match gateway::test(&p).await {
        Ok(()) => (StatusCode::OK, Json(json!({ "ok": true }))).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "ok": false, "error": { "code": e.code(), "message": e.message() } })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// ChatGPT connector (managed codex agent)
// ---------------------------------------------------------------------------

fn codex_err(e: super::codex::CodexError) -> Response {
    let status = match e {
        // Signed out is a precondition, not an auth failure of OUR API — a 401
        // would make a paired client drop its perfectly good pairing token.
        super::codex::CodexError::NotAuthenticated => StatusCode::PRECONDITION_REQUIRED,
        super::codex::CodexError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        super::codex::CodexError::Rpc(_) => StatusCode::BAD_GATEWAY,
    };
    ai_error(status, e.code(), e.message())
}

async fn codex_status(State(st): State<AiState>) -> Response {
    let codex = st.codex.clone();
    // Spawning and talking to the child blocks; keep it off the async worker.
    match tokio::task::spawn_blocking(move || codex.status()).await {
        Ok(s) => (StatusCode::OK, Json(s)).into_response(),
        Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexLoginBody {
    /// Device-code flow (show a code to type) vs. a browser redirect.
    #[serde(default)]
    device_code: bool,
}

async fn codex_login_start(State(st): State<AiState>, body: Option<Json<CodexLoginBody>>) -> Response {
    let device = body.map(|b| b.device_code).unwrap_or(false);
    let codex = st.codex.clone();
    match tokio::task::spawn_blocking(move || codex.start_login(device)).await {
        Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
        Ok(Err(e)) => codex_err(e),
        Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

async fn codex_login_cancel(State(st): State<AiState>) -> Response {
    let codex = st.codex.clone();
    match tokio::task::spawn_blocking(move || codex.cancel_login(None)).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => codex_err(e),
        Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

async fn codex_logout(State(st): State<AiState>) -> Response {
    let codex = st.codex.clone();
    match tokio::task::spawn_blocking(move || codex.logout()).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => codex_err(e),
        Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Gateway — chat + models
// ---------------------------------------------------------------------------

async fn chat_default(State(st): State<AiState>, Json(body): Json<Value>) -> Response {
    chat_impl(&st, None, body).await
}
async fn chat_for(State(st): State<AiState>, Path(id): Path<String>, Json(body): Json<Value>) -> Response {
    chat_impl(&st, Some(&id), body).await
}

async fn chat_impl(st: &AiState, provider_id: Option<&str>, mut body: Value) -> Response {
    // The ChatGPT connector is a managed agent, not a stored provider: it has no
    // key to inject, so it never goes through the egress/key path below.
    // The generic route, plus the four fixed live-call aliases. All of them are
    // the same managed agent — the alias only pins the turn's model and effort.
    let codex_alias = provider_id.and_then(super::codex::LiveCallAlias::from_id);
    if provider_id == Some(super::codex::CODEX_PROVIDER_ID) || codex_alias.is_some() {
        if let Some(o) = body.as_object_mut() {
            o.remove("provider");
        }
        let codex = st.codex.clone();
        // The generic route streams when asked, as an agent loop asks. A
        // live-call alias answers as it always has: one buffered completion.
        if codex_alias.is_none() && body.get("stream").and_then(Value::as_bool) == Some(true) {
            let model = body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(super::codex::CODEX_PROVIDER_ID)
                .to_string();
            return codex_stream(model, move |emit| codex.chat_streaming(&body, None, emit)).await;
        }
        return match tokio::task::spawn_blocking(move || codex.chat_as(&body, codex_alias)).await {
            Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
            Ok(Err(e)) => codex_err(e),
            Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
        };
    }
    // Resolve the FULL provider under the lock, drop the guard before await.
    let provider = if provider_id == Some(ENGINE_PROVIDER_ID) {
        match engine_now().await {
            // The engine answers with the model chosen in Engines: a model named here would load another.
            Ok((gateway, model)) => {
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("model");
                }
                Some(engine_provider(gateway, model))
            }
            Err(e) => return ai_error(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
        }
    } else {
        let store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
        match provider_id {
            Some(id) => store.get_full(id).filter(|p| p.supports(Capability::Chat)),
            None => store.default_for(Capability::Chat),
        }
    };
    let Some(p) = provider else {
        return ai_error(
            StatusCode::NOT_FOUND,
            "no_provider",
            "no AI provider is configured for chat — add one in OAIY Desktop → Providers".into(),
        );
    };
    if !p.enabled {
        return ai_error(StatusCode::BAD_REQUEST, "invalid_request", format!("provider {:?} is disabled", p.id));
    }
    if !p.has_key() && !p.allow_local {
        return ai_error(StatusCode::BAD_REQUEST, "invalid_request", format!("provider {:?} has no API key", p.id));
    }
    if let Some(obj) = body.as_object_mut() {
        obj.remove("provider"); // OAIY is flat — drop any routing hint
    }

    let wants_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if wants_stream {
        if !gateway::streamable(&p) {
            return ai_error(
                StatusCode::BAD_REQUEST,
                "streaming_unsupported",
                format!(
                    "provider {:?} ({:?}) cannot stream — retry without \"stream\": true",
                    p.id, p.protocol
                ),
            );
        }
        return match gateway::chat_stream(&p, body).await {
            Ok(upstream) => stream_passthrough(upstream),
            Err(e) => gateway_err(e),
        };
    }
    match gateway::chat(&p, body).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => gateway_err(e),
    }
}

async fn models_default(State(st): State<AiState>) -> Response {
    let provider = { st.providers.lock().unwrap_or_else(|e| e.into_inner()).default_for(Capability::Chat) };
    match provider {
        // No provider configured: an empty list (not a 404), so a discovery probe
        // reads "endpoint present, no models yet".
        None => (StatusCode::OK, Json(json!({ "object": "list", "data": [] }))).into_response(),
        Some(p) => match gateway::models(&p).await {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => gateway_err(e),
        },
    }
}

async fn models_for(State(st): State<AiState>, Path(id): Path<String>) -> Response {
    // An alias pins one model, so listing is the agent's own catalogue either
    // way — a caller asking what a fixed route can run deserves an answer, not
    // a 404 that reads like the route does not exist.
    if id == super::codex::CODEX_PROVIDER_ID || super::codex::LiveCallAlias::from_id(&id).is_some() {
        let codex = st.codex.clone();
        return match tokio::task::spawn_blocking(move || codex.models()).await {
            Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
            Ok(Err(e)) => codex_err(e),
            Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
        };
    }
    let provider = if id == ENGINE_PROVIDER_ID {
        match engine_now().await {
            Ok((gateway, model)) => Some(engine_provider(gateway, model)),
            Err(e) => return ai_error(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
        }
    } else {
        st.providers.lock().unwrap_or_else(|e| e.into_inner()).get_full(&id)
    };
    let Some(p) = provider else {
        return ai_error(StatusCode::NOT_FOUND, "no_provider", format!("unknown provider {id:?}"));
    };
    match gateway::models(&p).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => gateway_err(e),
    }
}

/// What a ChatGPT turn running on its blocking thread hands back.
enum CodexEvent {
    Delta(String),
    Done(Result<Value, super::codex::CodexError>),
}

/// The generic ChatGPT route's answer as OpenAI chat-completion chunks, for a
/// caller that asked for `stream: true` (the Agent app always does).
///
/// The turn runs on a blocking thread and its fragments come back through a
/// channel. The response is chosen when the FIRST thing arrives: a failure
/// before any text is an ordinary error with its own status — signed out stays
/// a 428 the caller can act on, not a 200 stream carrying an error — and
/// anything else opens the stream. The buffered completion closes it and is
/// the authority: a tool call goes out as a `tool_calls` chunk, and text the
/// stream held back (a reply that began like a tool call and was not one) as a
/// last content chunk.
async fn codex_stream<F>(model: String, turn: F) -> Response
where
    F: FnOnce(&mut dyn FnMut(&str)) -> Result<Value, super::codex::CodexError> + Send + 'static,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<CodexEvent>();
    tokio::task::spawn_blocking(move || {
        let deltas = tx.clone();
        let done = turn(&mut |d: &str| {
            let _ = deltas.send(CodexEvent::Delta(d.to_string()));
        });
        let _ = tx.send(CodexEvent::Done(done));
    });
    let first = match rx.recv().await {
        Some(CodexEvent::Done(Err(e))) => return codex_err(e),
        Some(event) => event,
        None => {
            return ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "the ChatGPT turn stopped without an answer".into())
        }
    };

    let head = ChunkHead {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        model,
    };
    let (out, out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        // A reader that went away makes these sends fail; the turn still ends on its own.
        let _ = out.send(head.data(json!({ "role": "assistant", "content": "" }), None));
        let mut streamed = String::new();
        let mut next = Some(first);
        loop {
            let event = match next.take() {
                Some(e) => e,
                None => match rx.recv().await {
                    Some(e) => e,
                    None => break,
                },
            };
            match event {
                CodexEvent::Delta(d) => {
                    streamed.push_str(&d);
                    let _ = out.send(head.data(json!({ "content": d }), None));
                }
                CodexEvent::Done(Ok(completion)) => {
                    for chunk in closing_chunks(&head, &completion, &streamed) {
                        let _ = out.send(chunk);
                    }
                    break;
                }
                CodexEvent::Done(Err(e)) => {
                    let error = json!({ "error": { "code": e.code(), "message": e.message() } });
                    let _ = out.send(format!("data: {error}\n\n"));
                    break;
                }
            }
        }
        let _ = out.send("data: [DONE]\n\n".to_string());
    });

    let body = axum::body::Body::from_stream(futures_util::stream::unfold(out_rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (Ok::<_, std::convert::Infallible>(chunk), rx))
    }));
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap_or_else(|_| ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "stream build failed".into()))
}

/// What every chunk of one streamed completion repeats.
struct ChunkHead {
    id: String,
    created: u64,
    model: String,
}

impl ChunkHead {
    /// One `chat.completion.chunk` event.
    fn data(&self, delta: Value, finish_reason: Option<&str>) -> String {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        });
        format!("data: {chunk}\n\n")
    }
}

/// The chunks that close a streamed turn, from its buffered completion and the
/// text already streamed (always a prefix of the completion's text: the held
/// fragments were released whole, or not at all).
fn closing_chunks(head: &ChunkHead, completion: &Value, streamed: &str) -> Vec<String> {
    let choice = completion.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let finish = choice.get("finish_reason").and_then(Value::as_str).unwrap_or("stop");
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut out = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array).filter(|c| !c.is_empty()) {
        // A streamed tool call carries its position in the reply's list.
        let calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut c = c.clone();
                c["index"] = json!(i);
                c
            })
            .collect();
        out.push(head.data(json!({ "tool_calls": calls }), None));
    } else {
        let content = message.get("content").and_then(Value::as_str).unwrap_or_default();
        let rest = content.strip_prefix(streamed).unwrap_or_default();
        if !rest.is_empty() {
            out.push(head.data(json!({ "content": rest }), None));
        }
    }
    out.push(head.data(json!({}), Some(finish)));
    out
}

/// Pipe an upstream streaming response straight through as `text/event-stream`.
fn stream_passthrough(upstream: reqwest::Response) -> Response {
    let ct = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();
    let body = axum::body::Body::from_stream(upstream.bytes_stream());
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, ct)
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap_or_else(|_| ai_error(StatusCode::BAD_GATEWAY, "upstream_error", "stream build failed".into()))
}

// ---------------------------------------------------------------------------
// Sources union (services + providers) — the shape FormLogic's picker consumes
// ---------------------------------------------------------------------------

/// Capability tags inferred from a service's category (OAIY's ServiceSnapshot has
/// no declared capabilities). A service with no AI capability is omitted.
fn capabilities_for_category(category: &str) -> Vec<&'static str> {
    let c = category.to_ascii_lowercase();
    if c.contains("llm") {
        vec!["chat"]
    } else if c.contains("speech") || c.contains("voice") {
        vec!["transcription", "speech"]
    } else if c.contains("image") {
        vec!["image"]
    } else {
        Vec::new()
    }
}

fn cap_str(cap: Capability) -> &'static str {
    match cap {
        Capability::Chat => "chat",
        Capability::Transcription => "transcription",
        Capability::Speech => "speech",
        Capability::Embeddings => "embeddings",
        Capability::Realtime => "realtime",
    }
}

fn protocol_str(p: Protocol) -> &'static str {
    match p {
        Protocol::OpenAi => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

fn configured_provider_source(p: &AiProviderPublic) -> Value {
    // Provider capabilities describe the upstream. Gateway capabilities name
    // the routes THIS desktop implements, so a realtime-capable provider does
    // not cause a plugin to offer a websocket path that does not exist here.
    let capabilities: Vec<&str> = if p.capabilities.is_empty() {
        vec!["chat"]
    } else {
        p.capabilities.iter().map(|c| cap_str(*c)).collect()
    };
    json!({
        "id": format!("provider:{}", p.id),
        "kind": "provider",
        "providerId": p.id,
        "name": p.name,
        "category": p.category,
        "status": "provider",
        "protocol": protocol_str(p.protocol),
        "capabilities": capabilities,
        "gatewayCapabilities": ["chat"],
        "hasKey": p.has_key,
        "enabled": p.enabled,
        "model": p.model,
        "useCases": ["flows"],
    })
}

async fn list_ai_sources(State(st): State<AiState>) -> Response {
    let mut sources: Vec<Value> = Vec::new();

    // ---- local SERVICES ----
    {
        let snap = match st.registry.lock() {
            Ok(reg) => reg.snapshot(),
            Err(_) => return ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "registry mutex poisoned".into()),
        };
        for s in &snap.services {
            let caps = capabilities_for_category(&s.category);
            if caps.is_empty() {
                continue; // non-AI services (browser automation, …) are not sources
            }
            let running = matches!(s.status, ServiceStatus::Running);
            sources.push(json!({
                "id": format!("service:{}", s.id),
                "kind": "service",
                "serviceId": s.id,
                "name": s.name,
                "category": s.category,
                "status": s.status,                 // serde → "running" / "stopped" / …
                "installed": s.installed,
                "port": s.port,
                "url": running.then(|| format!("http://127.0.0.1:{}", s.port)),
                "model": Value::Null,               // OAIY ServiceSnapshot has no per-service model
                "capabilities": caps,
                "useCases": ["background", "forms", "flows", "live-call"],
            }));
        }
    }

    // ---- OAIY's own engine: the model chosen in Engines ----
    if let Ok((gateway, model)) = engine_now().await {
        sources.push(json!({
            "id": format!("provider:{ENGINE_PROVIDER_ID}"),
            "kind": "provider",
            "providerId": ENGINE_PROVIDER_ID,
            "name": if model.is_empty() { "OAIY engine".to_string() } else { format!("OAIY engine ({model})") },
            "category": "LLM",
            "status": "running",
            "installed": true,
            "url": format!("{gateway}/v1"),
            "model": if model.is_empty() { Value::Null } else { json!(model) },
            "capabilities": ["chat"],
            "useCases": ["background", "forms", "flows", "live-call"],
        }));
    }

    // ---- configured PROVIDERS (public view — never the key) ----
    {
        let store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
        for p in store.list() {
            sources.push(configured_provider_source(&p));
        }
    }

    // ---- the ChatGPT connector, as a virtual provider ----
    // Advertised only when it's actually signed in: a source a flow cannot use
    // is worse than one that isn't offered.
    {
        let codex = st.codex.clone();
        if let Ok(status) = tokio::task::spawn_blocking(move || codex.status()).await {
            if status.available && status.connected {
                sources.push(json!({
                    "id": format!("provider:{}", super::codex::CODEX_PROVIDER_ID),
                    "kind": "provider",
                    "providerId": super::codex::CODEX_PROVIDER_ID,
                    "name": "ChatGPT (Codex)",
                    "category": "chatgpt",
                    "status": "provider",
                    "protocol": "codex",
                    "capabilities": ["chat"],
                    // No key to hold — the agent owns its own OAuth.
                    "hasKey": true,
                    "enabled": true,
                    "model": Value::Null,
                    "useCases": ["flows"],
                    "account": status.email,
                    "planType": status.plan_type,
                }));

                // The fixed live-call routes, so a phone agent can be pointed
                // at one from the same picker as everything else. Listed only
                // alongside a connected account, for the same reason as above.
                for alias in super::codex::LiveCallAlias::all() {
                    sources.push(json!({
                        "id": format!("provider:{}", alias.id()),
                        "kind": "provider",
                        "providerId": alias.id(),
                        "name": alias.display_name(),
                        "category": "chatgpt",
                        "status": "provider",
                        "protocol": "codex",
                        "capabilities": ["chat"],
                        "hasKey": true,
                        "enabled": true,
                        "model": alias.model(),
                        // Not offered for flows: these pin a low/no reasoning
                        // effort for latency, which is the wrong trade for
                        // background work. `liveCall` is what the phone agent
                        // filters on.
                        "useCases": ["liveCall"],
                        "account": status.email,
                        "planType": status.plan_type,
                    }));
                }
            }
        }
    }

    (StatusCode::OK, Json(json!({ "sources": sources }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_answers_with_the_model_chosen_in_engines() {
        assert_eq!(chosen_model(&json!({"defaults": {"llm": "Qwen3.8-Flash-Next"}})), "Qwen3.8-Flash-Next");
        let listed = json!({"models": {"llm": [{"id": "a", "default": false}, {"id": "b", "default": true}]}});
        assert_eq!(chosen_model(&listed), "b");
        assert_eq!(chosen_model(&json!({})), "");
        let p = engine_provider("http://127.0.0.1:8080".into(), "b".into());
        assert_eq!((p.id.as_str(), p.model.as_deref(), p.allow_local), (ENGINE_PROVIDER_ID, Some("b"), true));
    }

    // ---- ChatGPT's streamed answer, against a fake codex ----

    use crate::ai::codex::CodexError;

    /// A turn as the route runs it, with a fake codex answering `reply` in
    /// small fragments: the real completion code around a fake child.
    fn fake_turn(body: Value, reply: &'static str) -> impl FnOnce(&mut dyn FnMut(&str)) -> Result<Value, CodexError> + Send + 'static {
        move |emit| {
            crate::ai::codex::complete_with(&body, None, emit, |_prompt, _model, fragment| {
                let chars: Vec<char> = reply.chars().collect();
                for piece in chars.chunks(4) {
                    fragment(&piece.iter().collect::<String>());
                }
                Ok(reply.to_string())
            })
        }
    }

    fn with_tools() -> Value {
        json!({
            "stream": true,
            "messages": [{ "role": "user", "content": "Weather in Perth?" }],
            "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": { "type": "object" } } }],
        })
    }

    /// The events of a streamed answer, and whether it ended with `[DONE]`.
    async fn events_of(resp: Response) -> (Vec<Value>, bool) {
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[axum::http::header::CONTENT_TYPE], "text/event-stream");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let data: Vec<&str> = text.split("\n\n").filter_map(|e| e.strip_prefix("data: ")).collect();
        let done = data.last() == Some(&"[DONE]");
        (data.iter().filter(|d| **d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect(), done)
    }

    fn streamed_text(events: &[Value]) -> String {
        events.iter().filter_map(|e| e.pointer("/choices/0/delta/content").and_then(Value::as_str)).collect()
    }

    #[tokio::test]
    async fn a_streamed_answer_arrives_as_openai_chunks() {
        let reply = "Sunny, 24 degrees.";
        let (events, done) = events_of(codex_stream("gpt-5.5".into(), fake_turn(with_tools(), reply)).await).await;
        assert!(done, "the stream ends with [DONE]");
        assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
        for e in &events {
            assert_eq!(e["object"], "chat.completion.chunk");
            assert_eq!(e["model"], "gpt-5.5");
            assert_eq!(e["id"], events[0]["id"], "one id for the whole answer");
        }
        assert!(events.len() > 3, "it arrives in pieces, not at once: {events:?}");
        assert_eq!(streamed_text(&events), reply);
        let last = events.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["choices"][0]["delta"], json!({}));
    }

    #[tokio::test]
    async fn a_streamed_tool_call_arrives_as_a_tool_calls_chunk() {
        let reply = "```tool_call\n{\"tool\":\"get_weather\",\"input\":{\"city\":\"Perth\"}}\n```";
        let (events, done) = events_of(codex_stream("x".into(), fake_turn(with_tools(), reply)).await).await;
        assert!(done);
        assert_eq!(streamed_text(&events), "", "no text of the call is shown");
        let calls: Vec<&Value> = events.iter().filter_map(|e| e.pointer("/choices/0/delta/tool_calls")).collect();
        assert_eq!(calls.len(), 1, "{events:?}");
        let call = &calls[0][0];
        assert_eq!(call["index"], 0);
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), json!({ "city": "Perth" }));
        assert_eq!(events.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn a_streamed_reply_that_only_looked_like_a_call_still_arrives_whole() {
        let reply = "```tool_call\n{\"tool\":\"not_offered\",\"input\":{}}\n```";
        let (events, _) = events_of(codex_stream("x".into(), fake_turn(with_tools(), reply)).await).await;
        assert_eq!(streamed_text(&events), reply, "held back, then given as the answer");
        assert_eq!(events.last().unwrap()["choices"][0]["finish_reason"], "stop");
        assert!(events.iter().all(|e| e.pointer("/choices/0/delta/tool_calls").is_none()));
    }

    #[tokio::test]
    async fn a_failure_before_any_text_keeps_its_status() {
        // Signed out stays a 428 the caller can act on, not a 200 stream holding an error.
        let resp = codex_stream("x".into(), |_emit: &mut dyn FnMut(&str)| Err(CodexError::NotAuthenticated)).await;
        assert_eq!(resp.status(), StatusCode::PRECONDITION_REQUIRED);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], "codex_not_authenticated");

        // A failure after text has streamed can only be said in the stream.
        let resp = codex_stream("x".into(), |emit: &mut dyn FnMut(&str)| {
            emit("Half an ans");
            Err(CodexError::Rpc("turn/start timed out".into()))
        })
        .await;
        let (events, done) = events_of(resp).await;
        assert!(done);
        assert_eq!(streamed_text(&events), "Half an ans");
        assert_eq!(events.last().unwrap()["error"]["message"], "turn/start timed out");
    }

    #[test]
    fn a_realtime_upstream_does_not_advertise_a_realtime_gateway_that_is_not_implemented() {
        let source = configured_provider_source(&AiProviderPublic {
            id: "voice-provider".into(),
            name: "Voice provider".into(),
            category: None,
            protocol: Protocol::OpenAi,
            base_url: "https://provider.invalid/v1".into(),
            model: Some("voice-model".into()),
            capabilities: vec![Capability::Chat, Capability::Realtime],
            enabled: true,
            allow_local: false,
            has_key: true,
        });
        // Keep upstream metadata intact for clients that use it directly, but
        // let the Aokie picker explain why it cannot use Desktop for realtime.
        assert_eq!(source["capabilities"], json!(["chat", "realtime"]));
        assert_eq!(source["gatewayCapabilities"], json!(["chat"]));
        assert_eq!(source["useCases"], json!(["flows"]));
        assert_eq!(source["providerId"], "voice-provider");
        assert!(source.get("destinationOrigin").is_none());
        assert!(source.get("baseUrl").is_none());
        assert!(source.get("apiKey").is_none());
    }
}
