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
use super::providers::{AiProviderInput, Capability, Protocol, ProviderStoreHandle};
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

pub fn router(state: AiState) -> Router {
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
    // OUT OF SCOPE v1 (hook here later, same shape as the reference):
    //   POST /api/ai/v1/audio/transcriptions      -> Capability::Transcription
    //   POST /api/ai/v1/audio/chat/completions     -> buffered audio-chat
    //   POST /api/ai/v1/realtime/sessions          -> Capability::Realtime (WebRTC SDP)
    //   ...and the /api/ai/providers/:id/v1/... twins of each.
}

/// `{ "error": { "code", "message" } }` — identical taxonomy shape to the bridge.
fn ai_error(status: StatusCode, code: &str, message: String) -> Response {
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
    if provider_id == Some(super::codex::CODEX_PROVIDER_ID) {
        if let Some(o) = body.as_object_mut() {
            o.remove("provider");
        }
        let codex = st.codex.clone();
        return match tokio::task::spawn_blocking(move || codex.chat(&body)).await {
            Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
            Ok(Err(e)) => codex_err(e),
            Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
        };
    }
    // Resolve the FULL provider under the lock, drop the guard before await.
    let provider = {
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
    if id == super::codex::CODEX_PROVIDER_ID {
        let codex = st.codex.clone();
        return match tokio::task::spawn_blocking(move || codex.models()).await {
            Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
            Ok(Err(e)) => codex_err(e),
            Err(e) => ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
        };
    }
    let provider = { st.providers.lock().unwrap_or_else(|e| e.into_inner()).get_full(&id) };
    let Some(p) = provider else {
        return ai_error(StatusCode::NOT_FOUND, "no_provider", format!("unknown provider {id:?}"));
    };
    match gateway::models(&p).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => gateway_err(e),
    }
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

    // ---- configured PROVIDERS (public view — never the key) ----
    {
        let store = st.providers.lock().unwrap_or_else(|e| e.into_inner());
        for p in store.list() {
            // Empty caps == "all": advertise chat explicitly so FormLogic's
            // isDesktopFlowChatProvider (needs chat + useCases 'flows') accepts it.
            let capabilities: Vec<&str> = if p.capabilities.is_empty() {
                vec!["chat"]
            } else {
                p.capabilities.iter().map(|c| cap_str(*c)).collect()
            };
            sources.push(json!({
                "id": format!("provider:{}", p.id),
                "kind": "provider",
                "providerId": p.id,                 // → the client's refId
                "name": p.name,
                "category": p.category,
                "status": "provider",               // the literal 'provider' (client convention)
                "protocol": protocol_str(p.protocol),
                "capabilities": capabilities,
                "hasKey": p.has_key,
                "enabled": p.enabled,
                "model": p.model,
                "useCases": ["flows"],              // MUST include 'flows'
            }));
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
            }
        }
    }

    (StatusCode::OK, Json(json!({ "sources": sources }))).into_response()
}
