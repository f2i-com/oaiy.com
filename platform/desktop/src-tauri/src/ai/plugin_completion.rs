//! A text-only completion capability for opaque plugin screens. The desktop
//! supplies the plugin identity; neither source catalogue metadata nor a prompt
//! grants tools, URLs, credentials, provider management or model loading.

use super::{
    gateway,
    providers::{AiProvider, Capability, Protocol},
    routes::{ai_error, AiState, ENGINE_PROVIDER_ID},
};
use crate::plugins::{host::ScreenCapabilityLease, PluginHost};
use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::Response,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::watch;

const CAPABILITY: &str = "oaiy.ai.complete";
const DEADLINE: Duration = Duration::from_secs(18);
const MAX_PROMPT_CHARS: usize = 12_000;
const MAX_PROMPT_BYTES: usize = 48 * 1024;
const MAX_OUTPUT_CHARS: usize = 4_096;
const MAX_PENDING: usize = 4;
const MAX_RECENT: usize = 512;
const RECENT_TTL: Duration = Duration::from_secs(60);
const MAX_RESPONSE_BYTES: u64 = 256 * 1024;

type Failure = (StatusCode, &'static str, &'static str);
type Key = (String, String);
fn failure(e: Failure) -> Response {
    ai_error(e.0, e.1, e.2.into())
}
fn invalid() -> Failure {
    (StatusCode::BAD_REQUEST, "invalid_request", "Supply a fresh requestId, a configured provider sourceId, a bounded text prompt and maxOutputChars (1..4096).")
}

/// Exact canonical path predicate shared with origin/auth gates.
pub fn is_route(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    // Classification sees the raw URI, while Axum decodes :id. Encoded IDs
    // must get the same origin/token gate before reaching the handler.
    matches!(parts.as_slice(), ["", "api", "plugins", id, "ai", "sources" | "complete" | "cancel"] if !id.is_empty())
}

fn valid_provider_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
}
fn valid_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 96
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b':'))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Completion {
    request_id: String,
    source_id: String,
    prompt: String,
    max_output_chars: usize,
}
impl Completion {
    fn validate(&self) -> Result<&str, Failure> {
        let id = self
            .source_id
            .strip_prefix("provider:")
            .filter(|id| valid_provider_id(id))
            .ok_or_else(invalid)?;
        if !valid_request_id(&self.request_id)
            || self.prompt.trim().is_empty()
            || self.prompt.len() > MAX_PROMPT_BYTES
            || self.prompt.chars().count() > MAX_PROMPT_CHARS
            || self.max_output_chars == 0
            || self.max_output_chars > MAX_OUTPUT_CHARS
        {
            return Err(invalid());
        }
        // A managed agent can own tools/CLI work and cannot be cancelled by
        // dropping a blocking turn. It needs a separate explicit contract.
        if id == super::codex::CODEX_PROVIDER_ID
            || super::codex::LiveCallAlias::from_id(id).is_some()
        {
            return Err((StatusCode::BAD_REQUEST, "source_unsupported", "Managed-agent sources are unavailable through the bounded text completion capability."));
        }
        Ok(id)
    }
    fn body(&self) -> Value {
        json!({ "messages": [{"role":"user", "content":self.prompt}], "max_tokens":1024, "n":1, "stream":false })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Cancel {
    request_id: String,
}

trait Lease: Send + Sync {
    fn valid(&self) -> bool;
}
trait Gate: Send + Sync {
    fn lease(&self, plugin_id: &str) -> Result<Box<dyn Lease>, Failure>;
}
struct HostGate(Arc<PluginHost>);
struct HostLease {
    host: Arc<PluginHost>,
    plugin_id: String,
    lease: ScreenCapabilityLease,
}
impl Lease for HostLease {
    fn valid(&self) -> bool {
        self.host
            .holds_screen_capability(&self.plugin_id, CAPABILITY, &self.lease)
    }
}
impl Gate for HostGate {
    fn lease(&self, plugin_id: &str) -> Result<Box<dyn Lease>, Failure> {
        let lease = self
            .0
            .screen_capability(plugin_id, CAPABILITY)
            .map_err(|(code, message)| (StatusCode::FORBIDDEN, code, message))?;
        Ok(Box::new(HostLease {
            host: self.0.clone(),
            plugin_id: plugin_id.into(),
            lease,
        }))
    }
}

#[derive(Default)]
struct Requests {
    pending: HashMap<Key, watch::Sender<bool>>,
    // Cancellation can beat admission on separate HTTP connections. A bounded
    // tombstone also prevents a late cancel targeting a reused request ID.
    recent: HashMap<Key, Instant>,
}
impl Requests {
    fn prune(&mut self) {
        self.recent.retain(|_, at| at.elapsed() < RECENT_TTL);
    }
    fn claim(&mut self, key: Key) -> Result<(watch::Sender<bool>, watch::Receiver<bool>), Failure> {
        self.prune();
        if self.recent.contains_key(&key) {
            return Err((
                StatusCode::CONFLICT,
                "request_repeated",
                "This request ID was already used or cancelled. Use a fresh request ID.",
            ));
        }
        if self.pending.len() >= MAX_PENDING
            || self.pending.keys().any(|(plugin, _)| plugin == &key.0)
            || self.recent.len() >= MAX_RECENT
        {
            return Err((StatusCode::TOO_MANY_REQUESTS, "completion_busy", "An AI completion is already running, or the bounded request window is full. Cancel it or try again later."));
        }
        let (tx, rx) = watch::channel(false);
        self.recent.insert(key.clone(), Instant::now());
        self.pending.insert(key, tx.clone());
        Ok((tx, rx))
    }
    fn cancel(&mut self, key: Key) -> Result<bool, Failure> {
        self.prune();
        if let Some(tx) = self.pending.get(&key) {
            tx.send_replace(true);
            return Ok(true);
        }
        if !self.recent.contains_key(&key) {
            if self.recent.len() >= MAX_RECENT {
                return Err((
                    StatusCode::TOO_MANY_REQUESTS,
                    "completion_busy",
                    "The bounded request window is full. Try again later.",
                ));
            }
            self.recent.insert(key, Instant::now());
        }
        Ok(false)
    }
}
struct Claim {
    requests: Arc<Mutex<Requests>>,
    key: Key,
    channel: watch::Sender<bool>,
}
impl Drop for Claim {
    fn drop(&mut self) {
        let mut requests = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        if requests
            .pending
            .get(&self.key)
            .is_some_and(|tx| tx.same_channel(&self.channel))
        {
            requests.pending.remove(&self.key);
        }
    }
}

#[derive(Clone)]
struct PluginAi {
    ai: AiState,
    gate: Arc<dyn Gate>,
    requests: Arc<Mutex<Requests>>,
    isolated: bool,
}
pub fn router(ai: AiState, host: Arc<PluginHost>, isolated: bool) -> Router {
    routes(PluginAi {
        ai,
        gate: Arc::new(HostGate(host)),
        requests: Default::default(),
        isolated,
    })
}
fn routes(state: PluginAi) -> Router {
    Router::new()
        .route("/api/plugins/:id/ai/sources", get(sources))
        .route("/api/plugins/:id/ai/complete", post(complete))
        .route("/api/plugins/:id/ai/cancel", post(cancel))
        .layer(DefaultBodyLimit::max(320 * 1024))
        .with_state(state)
}

/// `allowLocal` is an explicit owner setting, but permits LAN targets in the
/// ordinary gateway. Isolation additionally requires a keyless literal
/// loopback endpoint, no query/userinfo/fragment, and bypasses system proxies.
fn isolated_provider(p: &AiProvider) -> bool {
    if p.protocol != Protocol::OpenAi
        || !p.allow_local
        || p.has_key()
        || p.api_key.is_some()
        || !p.enabled
        || !p.supports(Capability::Chat)
        || !p
            .model
            .as_deref()
            .is_some_and(|model| !model.trim().is_empty() && model.len() <= 256)
    {
        return false;
    }
    let Ok(url) = reqwest::Url::parse(&p.base_url) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && url
            .host_str()
            .and_then(|host| {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|ip| ip.is_loopback())
}

async fn sources(State(st): State<PluginAi>, Path(plugin): Path<String>) -> Response {
    let lease = match st.gate.lease(&plugin) {
        Ok(lease) => lease,
        Err(e) => return failure(e),
    };
    let mut sources: Vec<Value> = {
        let store = st.ai.providers.lock().unwrap_or_else(|e| e.into_inner());
        // Budget the serialized catalogue as well as individual fields; Unicode
        // names can use four UTF-8 bytes per character. Reserve room for engine.
        let mut catalogue_bytes = 0;
        store.list().into_iter().filter_map(|public| {
            if public.id == super::codex::CODEX_PROVIDER_ID
                || super::codex::LiveCallAlias::from_id(&public.id).is_some()
                || (!st.isolated && public.id == ENGINE_PROVIDER_ID) { return None; }
            let full = store.get_full(&public.id)?;
            if !full.model.as_deref().is_some_and(|model| !model.trim().is_empty() && model.len() <= 256)
                || !full.enabled || !full.supports(Capability::Chat) || (!full.has_key() && !full.allow_local)
                || (st.isolated && !isolated_provider(&full)) { return None; }
            let source = json!({"id":format!("provider:{}",public.id),"kind":"provider","providerId":public.id,
                "name":public.name.chars().take(160).collect::<String>(),"model":public.model.filter(|model| model.len() <= 256),
                "capabilities":["chat"],"gatewayCapabilities":["chat"],"enabled":true,
                "completionAvailable":true});
            let size = serde_json::to_vec(&source).ok()?.len() + 1;
            if catalogue_bytes + size > 60 * 1024 { return None; }
            catalogue_bytes += size;
            Some(source)
        }).take(64).collect()
    };
    // Engine discovery is deliberately absent in isolation. The normal route
    // asks the same existing engine service as the named-provider gateway.
    if !st.isolated {
        if let Ok(Ok(engine)) = tokio::time::timeout(
            Duration::from_secs(4),
            super::routes::plugin_engine_source(&st.ai),
        )
        .await
        {
            let mut source = json!({"id":format!("provider:{ENGINE_PROVIDER_ID}"),"kind":"provider","providerId":ENGINE_PROVIDER_ID,
                "name":"OAIY engine","model":engine.model,"capabilities":["chat"],"gatewayCapabilities":["chat"],"enabled":true,"completionAvailable":engine.completion_available});
            if let Some(reason) = engine.unavailable_reason { source["unavailableReason"] = json!(reason); }
            sources.push(source);
        }
    }
    if !lease.valid() {
        return failure((
            StatusCode::FORBIDDEN,
            "capability_unavailable",
            "The plugin stopped or lost its capability.",
        ));
    }
    use axum::response::IntoResponse;
    Json(json!({"sources":sources})).into_response()
}

async fn complete(
    State(st): State<PluginAi>,
    Path(plugin): Path<String>,
    body: Result<Json<Completion>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(body) => body,
        Err(_) => return failure(invalid()),
    };
    let provider_id = match input.validate() {
        Ok(id) => id.to_owned(),
        Err(e) => return failure(e),
    };
    let lease = match st.gate.lease(&plugin) {
        Ok(lease) => lease,
        Err(e) => return failure(e),
    };
    let key = (plugin, input.request_id.clone());
    let (channel, mut cancelled) = match st
        .requests
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .claim(key.clone())
    {
        Ok(claim) => claim,
        Err(e) => return failure(e),
    };
    let claim = Claim {
        requests: st.requests.clone(),
        key,
        channel,
    };
    let call = async {
        if !lease.valid() {
            return Err((
                StatusCode::FORBIDDEN,
                "capability_unavailable",
                "The plugin stopped or lost its capability.",
            ));
        }
        // Resolve within the deadline. Isolation never even resolves the shared
        // engine/managed agent or reads the ordinary launch's provider store.
        let provider = if st.isolated {
            st.ai.providers.lock().unwrap_or_else(|e| e.into_inner()).get_full(&provider_id)
                .filter(isolated_provider).ok_or((StatusCode::NOT_FOUND,"no_provider","No explicitly configured keyless loopback chat provider is available in this isolated launch."))?
        } else if provider_id == ENGINE_PROVIDER_ID {
            super::routes::resident_engine_provider(&st.ai)
                .await
                .map_err(|_| {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "engine_unavailable",
                        "The chosen engine model is not already ready and resident.",
                    )
                })?
        } else {
            super::routes::resolve_chat_provider(&st.ai, Some(&provider_id)).await
                .map_err(|(status, code, _)| (status,code,"The configured AI source is unavailable. Check its provider settings and running state."))?
        };
        if !provider
            .model
            .as_deref()
            .is_some_and(|model| !model.trim().is_empty() && model.len() <= 256)
        {
            return Err((
                StatusCode::NOT_FOUND,
                "no_provider",
                "Configure a bounded default model for this provider before using text completion.",
            ));
        }
        let result =
            gateway::chat_bounded(&provider, input.body(), MAX_RESPONSE_BYTES, st.isolated)
                .await
                .map_err(|_| {
                    (
                        StatusCode::BAD_GATEWAY,
                        "upstream_error",
                        "The AI provider could not complete this bounded text request.",
                    )
                })?;
        let text = completion_text(&result, input.max_output_chars)?;
        if !lease.valid() {
            return Err((
                StatusCode::FORBIDDEN,
                "capability_unavailable",
                "The plugin stopped or lost its capability.",
            ));
        }
        let mut out = json!({"requestId":input.request_id,"sourceId":input.source_id,"text":text});
        if let Some(model) = provider
            .model
            .as_deref()
            .filter(|model| !model.is_empty() && model.len() <= 256)
        {
            out["model"] = json!(model);
        }
        Ok::<Value, Failure>(out)
    };
    tokio::pin!(call);
    let mut capability_poll = tokio::time::interval(Duration::from_millis(250));
    let deadline = tokio::time::sleep(DEADLINE);
    tokio::pin!(deadline);
    let answer = loop {
        tokio::select! {
            biased;
            _ = cancelled.changed() => break Err((StatusCode::CONFLICT,"request_cancelled","The AI completion was cancelled.")),
            _ = &mut deadline => break Err((StatusCode::GATEWAY_TIMEOUT,"completion_timeout","The AI completion exceeded its 18-second deadline.")),
            _ = capability_poll.tick() => if !lease.valid() { break Err((StatusCode::FORBIDDEN,"capability_unavailable","The plugin stopped or lost its capability.")); },
            result = &mut call => break result,
        }
    };
    // Returning drops the request future and releases the slot. This closes our
    // HTTP request; a provider may continue internal compute after disconnect.
    use axum::response::IntoResponse;
    match answer {
        Ok(value) => {
            // Linearize completion with cancellation under the ticket lock.
            // A cancel accepted before this point can never yield success;
            // after this point cancel reports no active request.
            let mut requests = st.requests.lock().unwrap_or_else(|e| e.into_inner());
            if *claim.channel.borrow() {
                return failure((
                    StatusCode::CONFLICT,
                    "request_cancelled",
                    "The AI completion was cancelled.",
                ));
            }
            requests.pending.remove(&claim.key);
            Json(value).into_response()
        }
        Err(e) => failure(e),
    }
}

fn completion_text(result: &Value, limit: usize) -> Result<&str, Failure> {
    let invalid = (
        StatusCode::BAD_GATEWAY,
        "invalid_completion",
        "The AI provider did not return one complete assistant text response.",
    );
    let choices = result.get("choices").and_then(Value::as_array).filter(|choices| choices.len() == 1).ok_or(invalid)?;
    let choice = &choices[0];
    // A valid-looking JSON prefix can still be token-truncated. Accept only an
    // explicit complete response, never length/tool/refusal or unknown states.
    if choice.get("finish_reason").and_then(Value::as_str) != Some("stop") {
        return Err(invalid);
    }
    let message = choice.get("message").ok_or(invalid)?;
    if message.get("role").and_then(Value::as_str) != Some("assistant")
        || message.get("refusal").is_some_and(|refusal| !refusal.is_null()) {
        return Err(invalid);
    }
    if message.get("tool_calls").is_some_and(|calls| {
        !calls.is_null() && calls.as_array().map_or(true, |calls| !calls.is_empty())
    }) || message
        .get("function_call")
        .is_some_and(|call| !call.is_null())
    {
        return Err((
            StatusCode::BAD_GATEWAY,
            "invalid_completion",
            "Tool calls are unavailable through this text completion capability.",
        ));
    }
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or((
            StatusCode::BAD_GATEWAY,
            "invalid_completion",
            "The AI provider returned no text completion.",
        ))?;
    if text.chars().count() > limit {
        return Err((
            StatusCode::BAD_GATEWAY,
            "output_too_large",
            "The AI completion exceeded maxOutputChars; no truncated result was accepted.",
        ));
    }
    Ok(text)
}

async fn cancel(
    State(st): State<PluginAi>,
    Path(plugin): Path<String>,
    body: Result<Json<Cancel>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(input) = match body {
        Ok(body) => body,
        Err(_) => return failure(invalid()),
    };
    if !valid_request_id(&input.request_id) {
        return failure(invalid());
    }
    // Cancellation stays possible after stop/revocation. The HTTP origin/token
    // guard remains in force, and the path binds cancellation to this plugin.
    match st
        .requests
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .cancel((plugin, input.request_id.clone()))
    {
        Ok(cancelled) => {
            use axum::response::IntoResponse;
            Json(json!({"requestId":input.request_id,"cancelled":cancelled})).into_response()
        }
        Err(e) => failure(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use std::sync::atomic::{AtomicBool, Ordering};
    use tower::ServiceExt;

    struct TestGate(Arc<AtomicBool>);
    impl Gate for TestGate {
        fn lease(&self, _: &str) -> Result<Box<dyn Lease>, Failure> {
            if !self.0.load(Ordering::SeqCst) {
                return Err((
                    StatusCode::FORBIDDEN,
                    "capability_denied",
                    "Test capability is denied.",
                ));
            }
            Ok(Box::new(TestGate(self.0.clone())))
        }
    }
    impl Lease for TestGate {
        fn valid(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    struct Fixture {
        state: PluginAi,
        app: Router,
        gate: Arc<AtomicBool>,
        asked: Arc<Mutex<Vec<Value>>>,
        server: tokio::task::JoinHandle<()>,
        _dir: crate::secret_file::testing::TempDir,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.server.abort();
        }
    }
    async fn fixture() -> Fixture {
        let asked: Arc<Mutex<Vec<Value>>> = Default::default();
        let record = asked.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let provider = Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
            let asked = record.clone();
            async move {
                asked.lock().unwrap().push(body.clone());
                let prompt = body.pointer("/messages/0/content").and_then(Value::as_str).unwrap();
                if prompt == "wait" { std::future::pending::<()>().await; }
                if prompt == "failure" {
                    return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"secret-provider-error", "echo":body}})));
                }
                let message = if prompt == "tools" { json!({"role":"assistant", "content":"hello", "tool_calls":[{"name":"unsafe"}]}) }
                    else { json!({"role":"assistant", "content":if prompt == "large" { "x".repeat(MAX_RESPONSE_BYTES as usize) } else if prompt == "too-long" { "x".repeat(33) } else { "hello".into() }}) };
                (StatusCode::OK,Json(json!({"choices":[{"message":message,"finish_reason":if prompt == "truncated" {"length"} else {"stop"}}],"model":"untrusted-upstream-model"})))
            }
        })).route("/v1/messages", post(|Json(body): Json<Value>| async move {
            let prompt = body.pointer("/messages/0/content").and_then(Value::as_str).unwrap_or("");
            Json(json!({"role":"assistant", "content":if prompt == "anthropic-tool" {
                json!([{"type":"text","text":"hello"},{"type":"tool_use","id":"tool","name":"unsafe","input":{}}])
            } else {json!([{"type":"text","text":"hello"}])},
                "stop_reason":if prompt == "anthropic-truncated" {"max_tokens"} else {"end_turn"}}))
        }));
        let server = tokio::spawn(async move {
            axum::serve(listener, provider).await.unwrap();
        });
        let dir = crate::secret_file::testing::TempDir::new("plugin-ai");
        let registry = Arc::new(Mutex::new(crate::services::registry::Registry::empty(
            dir.0.join("data"),
            dir.0.join("models"),
        )));
        let providers = super::super::providers::new_handle();
        providers.lock().unwrap().upsert(serde_json::from_value(json!({
            "id":"test","name":"Synthetic test provider","baseUrl":base,"allowLocal":true,"model":"test-model"
        })).unwrap()).unwrap();
        let ai = AiState::new(providers, registry, super::super::codex::absent_for_tests())
            .with_engines_at(None);
        let gate = Arc::new(AtomicBool::new(true));
        let state = PluginAi {
            ai,
            gate: Arc::new(TestGate(gate.clone())),
            requests: Default::default(),
            isolated: true,
        };
        Fixture {
            app: routes(state.clone()),
            state,
            gate,
            asked,
            server,
            _dir: dir,
        }
    }
    fn input(id: &str, prompt: &str) -> Value {
        json!({"requestId":id,"sourceId":"provider:test","prompt":prompt,"maxOutputChars":32})
    }
    async fn ask(app: Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(if method == "GET" {
                Body::empty()
            } else {
                Body::from(body.to_string())
            })
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    async fn started(f: &Fixture) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while f.asked.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn closed_completion_reuses_real_gateway_with_only_host_chosen_fields() {
        let f = fixture().await;
        let (status, value) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            input("one", "hello"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            value,
            json!({"requestId":"one","sourceId":"provider:test","text":"hello","model":"test-model"})
        );
        assert_eq!(
            f.asked.lock().unwrap().as_slice(),
            &[
                json!({"messages":[{"role":"user","content":"hello"}],"max_tokens":1024,"n":1,"model":"test-model"})
            ]
        );
        assert!(f.state.requests.lock().unwrap().pending.is_empty());
        let (status, value) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            input("one", "hello"),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(value["error"]["code"], "request_repeated");
        assert_eq!(f.asked.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn invalid_bodies_and_capability_denial_never_reach_provider() {
        let f = fixture().await;
        let mut cases = vec![
            input("", "hello"),
            input("id/escape", "hello"),
            input("a", ""),
        ];
        for (field, value) in [
            ("url", json!("http://localhost")),
            ("model", json!("other")),
            ("tools", json!([])),
            ("maxOutputChars", json!(4097)),
            ("maxOutputChars", json!(1.5)),
            ("sourceId", json!("service:test")),
            ("prompt", json!("x".repeat(12001))),
        ] {
            let mut body = input("a", "hello");
            body[field] = value;
            cases.push(body);
        }
        for body in cases {
            assert_eq!(
                ask(
                    f.app.clone(),
                    "POST",
                    "/api/plugins/probe/ai/complete",
                    body
                )
                .await
                .0,
                StatusCode::BAD_REQUEST
            );
        }
        f.gate.store(false, Ordering::SeqCst);
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/complete",
                input("a", "hello")
            )
            .await
            .0,
            StatusCode::FORBIDDEN
        );
        assert!(f.asked.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn metadata_catalogue_is_bounded_and_is_not_a_provider_grant() {
        let f = fixture().await;
        let (_, value) = ask(
            f.app.clone(),
            "GET",
            "/api/plugins/probe/ai/sources",
            Value::Null,
        )
        .await;
        assert_eq!(value["sources"].as_array().unwrap().len(), 1);
        assert_eq!(value["sources"][0]["id"], "provider:test");
        for field in ["url", "baseUrl", "gatewayUrl", "apiKey", "hasKey"] {
            assert!(value["sources"][0].get(field).is_none());
        }
        let mut body = input("unknown", "hello");
        body["sourceId"] = json!("provider:unknown");
        let (status, value) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            body,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(value["error"]["code"], "no_provider");
        assert!(f.asked.lock().unwrap().is_empty());
        {
            let mut store = f.state.ai.providers.lock().unwrap();
            for index in 0..64 {
                store.upsert(serde_json::from_value(json!({"id":format!("p-{index:062}"),
                    "name":"😀".repeat(160),"baseUrl":"http://127.0.0.1:12345","model":"m".repeat(256),"allowLocal":true})).unwrap()).unwrap();
            }
        }
        let (_, catalogue) = ask(f.app.clone(), "GET", "/api/plugins/probe/ai/sources", Value::Null).await;
        assert!(serde_json::to_vec(&catalogue).unwrap().len() < 64 * 1024);
        assert!(catalogue["sources"].as_array().unwrap().len() < 65, "the byte budget must also bound multibyte names");
    }

    #[tokio::test]
    async fn cancellation_releases_owned_future_and_prevents_late_or_reused_admission() {
        let f = fixture().await;
        let request = tokio::spawn(ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            input("pending", "wait"),
        ));
        started(&f).await;
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/other/ai/cancel",
                json!({"requestId":"pending"})
            )
            .await
            .1["cancelled"],
            false
        );
        assert!(f
            .state
            .requests
            .lock()
            .unwrap()
            .pending
            .contains_key(&("probe".into(), "pending".into())));
        let (_, ack) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/cancel",
            json!({"requestId":"pending"}),
        )
        .await;
        assert_eq!(ack["cancelled"], true);
        let (status, value) = request.await.unwrap();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(value["error"]["code"], "request_cancelled");
        assert!(f.state.requests.lock().unwrap().pending.is_empty());
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/complete",
                input("fresh", "hello")
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/cancel",
                json!({"requestId":"early"})
            )
            .await
            .1["cancelled"],
            false
        );
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/complete",
                input("early", "hello")
            )
            .await
            .0,
            StatusCode::CONFLICT
        );
    }

    #[tokio::test]
    async fn revoked_live_lease_interrupts_call_and_cancel_remains_available() {
        let f = fixture().await;
        let request = tokio::spawn(ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            input("pending", "wait"),
        ));
        started(&f).await;
        f.gate.store(false, Ordering::SeqCst);
        let (status, value) = tokio::time::timeout(Duration::from_secs(2), request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(value["error"]["code"], "capability_unavailable");
        assert_eq!(
            ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/cancel",
                json!({"requestId":"pending"})
            )
            .await
            .0,
            StatusCode::OK
        );
        assert!(f.state.requests.lock().unwrap().pending.is_empty());
    }

    #[tokio::test]
    async fn provider_failures_tools_and_oversized_outputs_are_bounded_refusals() {
        let f = fixture().await;
        for (id, prompt, code) in [
            ("failure", "failure", "upstream_error"),
            ("tools", "tools", "invalid_completion"),
            ("truncated", "truncated", "invalid_completion"),
            ("too-long", "too-long", "output_too_large"),
            ("large", "large", "upstream_error"),
        ] {
            let (status, value) = ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/complete",
                input(id, prompt),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_GATEWAY);
            assert_eq!(value["error"]["code"], code);
            assert!(!value.to_string().contains("secret-provider-error"));
            assert!(value.to_string().len() < 1024);
        }
    }

    #[tokio::test]
    async fn anthropic_completion_cannot_normalize_tools_or_truncation_into_text() {
        let mut f = fixture().await;
        {
            let mut providers = f.state.ai.providers.lock().unwrap();
            let mut provider = providers.get_full("test").unwrap();
            provider.protocol = Protocol::Anthropic;
            providers.upsert(serde_json::from_value(serde_json::to_value(provider).unwrap()).unwrap()).unwrap();
        }
        f.state.isolated = false;
        f.app = routes(f.state.clone());
        for (id, prompt, status) in [("text", "hello", StatusCode::OK),
            ("tool", "anthropic-tool", StatusCode::BAD_GATEWAY),
            ("length", "anthropic-truncated", StatusCode::BAD_GATEWAY)] {
            let (actual, value) = ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", input(id, prompt)).await;
            assert_eq!(actual, status);
            if status == StatusCode::OK { assert_eq!(value["text"], "hello"); }
            else { assert_eq!(value["error"]["code"], "upstream_error"); }
        }
    }

    #[tokio::test]
    async fn whole_call_deadline_cancels_a_provider_that_never_responds() {
        let f = fixture().await;
        let start = Instant::now();
        let (status, value) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            input("timeout", "wait"),
        )
        .await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(value["error"]["code"], "completion_timeout");
        assert!(start.elapsed() >= DEADLINE && start.elapsed() < Duration::from_secs(22));
        assert!(f.state.requests.lock().unwrap().pending.is_empty());
    }

    #[test]
    fn isolated_provider_refuses_hostnames_network_targets_keys_and_other_protocols() {
        let mut provider: AiProvider = serde_json::from_value(json!({"id":"test","name":"test","baseUrl":"http://127.0.0.1:12345/v1","allowLocal":true,"model":"test"})).unwrap();
        assert!(isolated_provider(&provider));
        provider.base_url = "http://[::1]:12345/v1".into();
        assert!(isolated_provider(&provider));
        for base in [
            "http://localhost:12345",
            "http://10.0.0.1",
            "https://api.example.com",
            "http://user@127.0.0.1",
            "http://127.0.0.1?key=a",
            "http://127.0.0.1#fragment",
        ] {
            provider.base_url = base.into();
            assert!(!isolated_provider(&provider), "{base}");
        }
        provider.base_url = "http://127.0.0.1:12345".into();
        provider.api_key = Some("test-key".into());
        assert!(!isolated_provider(&provider));
        provider.api_key = None;
        provider.protocol = Protocol::Anthropic;
        assert!(!isolated_provider(&provider));
        provider.protocol = Protocol::OpenAi;
        provider.model = None;
        assert!(!isolated_provider(&provider));
    }

    #[tokio::test]
    async fn engine_catalogue_reports_readiness_without_loading_and_is_bounded_and_isolated() {
        use axum::response::IntoResponse;
        use std::sync::atomic::AtomicUsize;
        let mut f = fixture().await;
        let llm = Arc::new(Mutex::new(json!({"state":"stopped","resident":"","paused_for_media":false})));
        let discovery = Arc::new(Mutex::new(json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","files_present":true}]}})));
        let mode = Arc::new(AtomicUsize::new(0));
        let traffic: Arc<Mutex<Vec<String>>> = Default::default();
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let gateway = format!("{base}/gw");
        let state_llm = llm.clone();
        let state_mode = mode.clone();
        let state_entered = entered.clone();
        let state_release = release.clone();
        let discovery_mode = mode.clone();
        let document = discovery.clone();
        let requests = traffic.clone();
        let engine = Router::new()
            .route("/api/state", get(move || {
                let llm = state_llm.clone(); let mode = state_mode.clone(); let gateway = gateway.clone();
                let entered = state_entered.clone(); let release = state_release.clone();
                async move {
                    if mode.load(Ordering::SeqCst) == 5 { std::future::pending::<()>().await; }
                    if mode.load(Ordering::SeqCst) == 6 { entered.notify_one(); release.notified().await; }
                    Json(json!({"gateway_url":gateway,"llm":llm.lock().unwrap().clone()}))
                }
            }))
            .route("/gw/v1/discovery", get(move || {
                let mode = discovery_mode.clone(); let document = document.clone();
                async move {
                    match mode.load(Ordering::SeqCst) {
                        1 => Json(json!({"padding":"x".repeat(64*1024)})).into_response(),
                        2 => (StatusCode::FOUND, [("location", "/gw/redirected")], "redirect").into_response(),
                        3 => "invalid JSON".into_response(),
                        4 => (StatusCode::BAD_GATEWAY, "private engine failure at http://private.invalid").into_response(),
                        _ => Json(document.lock().unwrap().clone()).into_response(),
                    }
                }
            }))
            .fallback(|| async { StatusCode::METHOD_NOT_ALLOWED })
            .layer(axum::middleware::from_fn(move |request: axum::extract::Request, next: axum::middleware::Next| {
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(format!("{} {}", request.method(), request.uri().path()));
                    next.run(request).await
                }
            }));
        let server = tokio::spawn(async move { axum::serve(listener, engine).await.unwrap(); });
        f.state.ai = f.state.ai.clone().with_engines_at(Some(base));
        f.app = routes(f.state.clone());
        let read = |app| ask(app, "GET", "/api/plugins/probe/ai/sources", Value::Null);
        fn engine_source(value: &Value) -> Option<&Value> {
            value["sources"].as_array().unwrap().iter().find(|source| source["providerId"] == ENGINE_PROVIDER_ID)
        }
        let (status, value) = read(f.app.clone()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(engine_source(&value).is_none());
        assert!(traffic.lock().unwrap().is_empty(), "isolated catalogue must not contact even a configured engine URL");
        f.state.isolated = false;
        f.app = routes(f.state.clone());
        f.gate.store(false, Ordering::SeqCst);
        assert_eq!(read(f.app.clone()).await.0, StatusCode::FORBIDDEN);
        assert!(traffic.lock().unwrap().is_empty(), "untrusted catalogue must not contact engines");
        f.gate.store(true, Ordering::SeqCst);
        for (state, resident, paused, reason) in [
            ("stopped", "", false, "not loaded"),
            ("starting", "", false, "loading"),
            ("failed", "", false, "failed to start"),
            ("stopped", "selected", true, "paused for media"),
            ("ready", "other", false, "different local model"),
        ] {
            *llm.lock().unwrap() = json!({"state":state,"resident":resident,"paused_for_media":paused});
            let (status, value) = read(f.app.clone()).await;
            assert_eq!(status, StatusCode::OK);
            let source = engine_source(&value).unwrap();
            assert_eq!(source["model"], "selected");
            assert_eq!(source["completionAvailable"], false);
            assert!(source["unavailableReason"].as_str().unwrap().contains(reason));
        }
        *llm.lock().unwrap() = json!({"state":"ready","resident":"selected"});
        let (_, value) = read(f.app.clone()).await;
        let source = engine_source(&value).unwrap();
        assert_eq!(source["completionAvailable"], true);
        assert!(source.get("unavailableReason").is_none());
        discovery.lock().unwrap()["models"]["llm"][0]["files_present"] = json!(false);
        let (_, value) = read(f.app.clone()).await;
        let source = engine_source(&value).unwrap();
        assert_eq!(source["completionAvailable"], false);
        assert!(source["unavailableReason"].as_str().unwrap().contains("files are missing"));
        for invalid in [
            json!({"defaults":{"llm":"disabled"},"models":{"llm":[{"id":"selected","files_present":true}]}}),
            json!({"defaults":{"llm":"x".repeat(257)},"models":{"llm":[]}}),
            json!({"defaults":{"llm":"bad\nmodel"},"models":{"llm":[{"id":"bad\nmodel","files_present":true}]}}),
        ] {
            *discovery.lock().unwrap() = invalid;
            assert!(engine_source(&read(f.app.clone()).await.1).is_none());
        }
        *discovery.lock().unwrap() = json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","files_present":true}]}});
        for failed_transport in 1..=5 {
            mode.store(failed_transport, Ordering::SeqCst);
            let started = Instant::now();
            let (status, value) = read(f.app.clone()).await;
            assert_eq!(status, StatusCode::OK);
            assert!(engine_source(&value).is_none());
            assert!(started.elapsed() < Duration::from_secs(6), "catalogue discovery must have a whole-operation deadline");
            assert!(!value.to_string().contains("private.invalid"));
        }
        mode.store(6, Ordering::SeqCst);
        let pending = tokio::spawn(read(f.app.clone()));
        tokio::time::timeout(Duration::from_secs(2), entered.notified()).await.unwrap();
        f.gate.store(false, Ordering::SeqCst);
        release.notify_one();
        assert_eq!(pending.await.unwrap().0, StatusCode::FORBIDDEN, "a lost trust lease cannot publish stale model metadata");
        assert!(traffic.lock().unwrap().iter().all(|path| path == "GET /api/state" || path == "GET /gw/v1/discovery"), "catalogue reads cannot follow redirects, load models, or start inference");
        assert!(f.asked.lock().unwrap().is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn normal_engine_calls_require_the_chosen_model_to_be_already_ready_and_resident() {
        let mut f = fixture().await;
        let ready = Arc::new(AtomicBool::new(false));
        let chosen = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let oversized = Arc::new(AtomicBool::new(false));
        let discovery = Arc::new(Mutex::new(json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","files_present":true}]}})));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let gateway = format!("{base}/gw");
        let state_ready = ready.clone();
        let state_chosen = chosen.clone();
        let state_paused = paused.clone();
        let discovery_oversized = oversized.clone();
        let document = discovery.clone();
        let asked = f.asked.clone();
        let engine = Router::new()
            .route("/api/state",get(move || { let ready=state_ready.clone(); let chosen=state_chosen.clone(); let paused=state_paused.clone(); let gateway=gateway.clone(); async move {
                Json(json!({"gateway_url":gateway,"llm":{"state":if ready.load(Ordering::SeqCst) {"ready"} else {"stopped"},
                    "resident":if chosen.load(Ordering::SeqCst) {"selected"} else {"other"},"paused_for_media":paused.load(Ordering::SeqCst)}}))
            }}))
            .route("/gw/v1/discovery",get(move || { let oversized=discovery_oversized.clone(); let document=document.clone(); async move {
                Json(if oversized.load(Ordering::SeqCst) {json!({"defaults":{"llm":"selected"},"padding":"x".repeat(64*1024)})}
                    else {document.lock().unwrap().clone()})
            }}))
            .route("/gw/v1/chat/completions",post(move |Json(body):Json<Value>| { let asked=asked.clone(); async move {
                asked.lock().unwrap().push(body); Json(json!({"choices":[{"message":{"role":"assistant","content":"resident answer"},"finish_reason":"stop"}]}))
            }}));
        let server = tokio::spawn(async move {
            axum::serve(listener, engine).await.unwrap();
        });
        f.state.isolated = false;
        f.state.ai = f.state.ai.clone().with_engines_at(Some(base));
        f.app = routes(f.state.clone());
        for (id, is_ready, is_chosen) in [("stopped", false, true), ("other", true, false)] {
            ready.store(is_ready, Ordering::SeqCst);
            chosen.store(is_chosen, Ordering::SeqCst);
            let mut request = input(id, "hello");
            request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
            let (status, value) = ask(
                f.app.clone(),
                "POST",
                "/api/plugins/probe/ai/complete",
                request,
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(value["error"]["code"], "engine_unavailable");
            assert!(f.asked.lock().unwrap().is_empty());
        }
        ready.store(true, Ordering::SeqCst);
        chosen.store(true, Ordering::SeqCst);
        let mut request = input("resident", "hello");
        request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
        let (status, value) = ask(
            f.app.clone(),
            "POST",
            "/api/plugins/probe/ai/complete",
            request,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["text"], "resident answer");
        assert_eq!(value["model"], "selected");
        assert_eq!(f.asked.lock().unwrap()[0]["model"], "selected");
        // A direct completion cannot bypass the catalogue's refusal, even if
        // no catalogue read preceded it and the same model remains resident.
        for (index, files) in [json!(false), Value::Null, json!("true"), json!(1), json!({}), json!([])].into_iter().enumerate() {
            discovery.lock().unwrap()["models"]["llm"][0]["files_present"] = files;
            let mut request = input(&format!("invalid-files-{index}"), "hello");
            request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
            let (status, value) = ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", request).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(value["error"]["code"], "engine_unavailable");
            assert_eq!(f.asked.lock().unwrap().len(), 1, "invalid files flags cannot start inference");
        }
        *discovery.lock().unwrap() = json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","files_present":true}]}});
        paused.store(true, Ordering::SeqCst);
        let mut request = input("paused-modern", "hello");
        request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
        assert_eq!(ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", request).await.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(f.asked.lock().unwrap().len(), 1, "paused media work cannot start inference");
        paused.store(false, Ordering::SeqCst);
        for (index, legacy) in [
            json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","loaded":false}]}}),
            json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected"}]}}),
            json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"other","loaded":true}]}}),
        ].into_iter().enumerate() {
            *discovery.lock().unwrap() = legacy;
            let mut request = input(&format!("unconfirmed-legacy-{index}"), "hello");
            request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
            assert_eq!(ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", request).await.0, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(f.asked.lock().unwrap().len(), 1, "unconfirmed legacy metadata cannot start inference");
        }
        *discovery.lock().unwrap() = json!({"defaults":{"llm":"selected"},"models":{"llm":[{"id":"selected","loaded":true}]}});
        let (_, catalogue) = ask(f.app.clone(), "GET", "/api/plugins/probe/ai/sources", Value::Null).await;
        let source = catalogue["sources"].as_array().unwrap().iter().find(|source| source["providerId"] == ENGINE_PROVIDER_ID).unwrap();
        assert_eq!(source["completionAvailable"], true);
        assert_eq!(source["model"], "selected");
        assert_eq!(f.asked.lock().unwrap().len(), 1, "legacy catalogue reads cannot start inference");
        let mut request = input("resident-legacy", "hello");
        request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
        let (status, value) = ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", request).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["text"], "resident answer");
        assert_eq!(value["model"], "selected");
        assert_eq!(f.asked.lock().unwrap().len(), 2);
        assert_eq!(f.asked.lock().unwrap()[1]["model"], "selected");
        oversized.store(true, Ordering::SeqCst);
        let mut request = input("oversized-discovery", "hello");
        request["sourceId"] = json!(format!("provider:{ENGINE_PROVIDER_ID}"));
        let (status, value) = ask(f.app.clone(), "POST", "/api/plugins/probe/ai/complete", request).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(value["error"]["code"], "engine_unavailable");
        assert_eq!(f.asked.lock().unwrap().len(), 2, "oversized discovery must not start an inference request");
        server.abort();
    }

    #[test]
    fn global_and_plugin_concurrency_and_recent_memory_are_bounded() {
        let mut requests = Requests::default();
        let first = requests.claim(("a".into(), "1".into())).unwrap();
        assert_eq!(
            requests.claim(("a".into(), "2".into())).unwrap_err().1,
            "completion_busy"
        );
        for p in ["b", "c", "d"] {
            requests.claim((p.into(), "1".into())).unwrap();
        }
        assert_eq!(
            requests.claim(("e".into(), "1".into())).unwrap_err().1,
            "completion_busy"
        );
        requests.pending.clear();
        drop(first);
        for i in 4..MAX_RECENT {
            requests
                .cancel(("a".into(), format!("cancel-{i}")))
                .unwrap();
        }
        assert_eq!(requests.recent.len(), MAX_RECENT);
        assert_eq!(
            requests
                .cancel(("a".into(), "overflow".into()))
                .unwrap_err()
                .1,
            "completion_busy"
        );
    }

    #[test]
    fn encoded_ids_receive_same_auth_classification_and_text_outputs_cannot_grant_tools() {
        assert!(is_route("/api/plugins/%70robe/ai/complete"));
        assert!(is_route("/api/plugins/probe/ai/sources"));
        assert!(!is_route("/api/plugins/probe/ai/complete/extra"));
        let mut body = input("one", "hello");
        body["sourceId"] = json!(format!(
            "provider:{}",
            super::super::codex::CODEX_PROVIDER_ID
        ));
        let request: Completion = serde_json::from_value(body).unwrap();
        assert_eq!(request.validate().unwrap_err().1, "source_unsupported");
        assert!(completion_text(&json!({"choices":[{"message":{"content":null}}]}), 32).is_err());
        assert!(completion_text(
            &json!({"choices":[{"message":{"role":"assistant","content":"😀","tool_calls":[]},"finish_reason":"stop"}]}),
            1
        )
        .is_ok());
    }

    #[test]
    fn completion_requires_one_finished_nonrefusing_assistant_choice() {
        let valid = json!({"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]});
        assert_eq!(completion_text(&valid, 32).unwrap(), "hello");
        for reason in [json!("length"), json!("tool_calls"), json!("content_filter"), json!("unknown"), Value::Null] {
            let mut result = valid.clone(); result["choices"][0]["finish_reason"] = reason;
            assert_eq!(completion_text(&result, 32).unwrap_err().1, "invalid_completion");
        }
        for role in [json!("user"), Value::Null] {
            let mut result = valid.clone(); result["choices"][0]["message"]["role"] = role;
            assert!(completion_text(&result, 32).is_err());
        }
        let mut result = valid.clone(); result["choices"][0]["message"]["refusal"] = json!("refused");
        assert!(completion_text(&result, 32).is_err());
        result = valid.clone(); result["choices"].as_array_mut().unwrap().push(valid["choices"][0].clone());
        assert!(completion_text(&result, 32).is_err());
    }
}
