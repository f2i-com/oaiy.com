//! The AI proxy: forward an OpenAI-shaped chat/model request to a configured
//! provider, injecting the stored credential SERVER-SIDE (the browser never sees
//! the key). Adapted from the FormLogic reference gateway, trimmed to `openai`
//! and `anthropic`. Anthropic requests/responses are translated to/from the
//! OpenAI chat shape so every consumer sees one dialect.
//!
//! Credentials only ever travel in request HEADERS (never a URL or body). All
//! outbound URLs pass [`crate::ai::egress::validate`] first, and the client has
//! redirects disabled.

use super::egress;
use super::providers::{AiProvider, Protocol};
use futures_util::StreamExt;
use serde_json::{json, Value};

/// Anthropic API version pin (matches the FormLogic reference).
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Cap on a buffered (non-streaming) upstream response.
pub const MAX_RESPONSE_BYTES: u64 = 32 * 1024 * 1024;
/// Anthropic requires max_tokens; default when the OpenAI request omits it. (This
/// is a protocol constant, not a model id — safe to hardcode.)
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 1024;

/// Closed error taxonomy for the gateway. Mapped to `{error:{code,message}}` by
/// the route layer.
#[derive(Debug)]
pub enum GatewayError {
    /// No provider is configured / the named one can't serve the capability.
    NoProvider(String),
    /// The request or provider config is invalid (bad URL, missing key, …).
    BadRequest(String),
    /// The upstream provider failed or returned something unusable.
    Upstream(String),
}

impl GatewayError {
    pub fn code(&self) -> &'static str {
        match self {
            GatewayError::NoProvider(_) => "no_provider",
            GatewayError::BadRequest(_) => "invalid_request",
            GatewayError::Upstream(_) => "upstream_error",
        }
    }
    pub fn message(&self) -> &str {
        match self {
            GatewayError::NoProvider(m) | GatewayError::BadRequest(m) | GatewayError::Upstream(m) => m,
        }
    }
}

/// The path appended to a provider's base URL for a chat call.
fn chat_path(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::OpenAi => "/v1/chat/completions",
        Protocol::Anthropic => "/v1/messages",
    }
}

/// Only OpenAI-dialect providers can stream — bytes pass through unaltered.
/// Anthropic uses a different SSE event schema that would need per-chunk
/// translation, so it is refused (the handler falls back to a non-stream error).
pub fn streamable(provider: &AiProvider) -> bool {
    matches!(provider.protocol, Protocol::OpenAi)
}

/// Inject auth into a request builder, server-side, per protocol. A keyless
/// provider (e.g. a local model server) gets no auth header.
fn apply_auth(
    rb: reqwest::RequestBuilder,
    provider: &AiProvider,
    key: Option<&str>,
) -> reqwest::RequestBuilder {
    match (provider.protocol, key) {
        (Protocol::OpenAi, Some(k)) => rb.header(reqwest::header::AUTHORIZATION, format!("Bearer {k}")),
        (Protocol::Anthropic, Some(k)) => rb
            .header("x-api-key", k)
            .header("anthropic-version", ANTHROPIC_VERSION),
        (_, None) => rb, // local / keyless
    }
}

/// Attach a JSON body without reqwest's `json` feature (OAIY builds reqwest
/// without it — see lib.rs). Serializes with serde_json and sets Content-Type.
fn with_json_body(rb: reqwest::RequestBuilder, value: &Value) -> Result<reqwest::RequestBuilder, GatewayError> {
    let bytes = serde_json::to_vec(value).map_err(|e| GatewayError::BadRequest(format!("invalid request body: {e}")))?;
    Ok(rb.header(reqwest::header::CONTENT_TYPE, "application/json").body(bytes))
}

/// Read an upstream response body with a hard size cap.
async fn read_capped(resp: reqwest::Response) -> Result<Vec<u8>, GatewayError> {
    let mut stream = resp.bytes_stream();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| GatewayError::Upstream(format!("read failed: {e}")))?;
        if out.len() as u64 + chunk.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(GatewayError::Upstream("upstream response exceeded the size cap".into()));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Non-streaming chat. Returns an OpenAI `chat.completion` JSON regardless of the
/// provider's protocol.
pub async fn chat(provider: &AiProvider, mut body: Value) -> Result<Value, GatewayError> {
    ensure_model(provider, &mut body)?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("provider"); // non-OpenAI routing hint — never forward it
        obj.remove("stream"); // this path is buffered
    }

    let key = provider.api_key.clone();
    let target = egress::validate(&provider.base_url, chat_path(provider.protocol), provider.allow_local)
        .map_err(GatewayError::BadRequest)?;
    let out_body = match provider.protocol {
        Protocol::OpenAi => body,
        Protocol::Anthropic => openai_chat_to_anthropic(&body),
    };

    let client = egress::client(false).map_err(GatewayError::Upstream)?;
    let rb = apply_auth(with_json_body(client.post(target), &out_body)?, provider, key.as_deref());
    let resp = rb
        .send()
        .await
        .map_err(|e| GatewayError::Upstream(format!("request failed: {e}")))?;
    let status = resp.status();
    let bytes = read_capped(resp).await?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| GatewayError::Upstream(format!("non-JSON upstream ({status}): {e}")))?;
    if !status.is_success() {
        let detail = value
            .get("error")
            .map(|e| e.to_string())
            .unwrap_or_else(|| status.to_string());
        return Err(GatewayError::Upstream(format!("upstream {status}: {detail}")));
    }
    Ok(match provider.protocol {
        Protocol::OpenAi => value,
        Protocol::Anthropic => anthropic_response_to_openai(&value),
    })
}

/// Streaming chat — OpenAI dialect only. Returns the raw upstream `Response` so
/// the handler can pipe `bytes_stream()` straight through as `text/event-stream`.
/// A non-2xx upstream is buffered (capped) and surfaced as an error so an error
/// page never masquerades as an event stream.
pub async fn chat_stream(provider: &AiProvider, mut body: Value) -> Result<reqwest::Response, GatewayError> {
    ensure_model(provider, &mut body)?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("provider");
        obj.insert("stream".into(), Value::Bool(true));
    }
    let key = provider.api_key.clone();
    let target = egress::validate(&provider.base_url, "/v1/chat/completions", provider.allow_local)
        .map_err(GatewayError::BadRequest)?;
    let client = egress::client(true).map_err(GatewayError::Upstream)?;
    let rb = apply_auth(with_json_body(client.post(target), &body)?, provider, key.as_deref());
    let resp = rb
        .send()
        .await
        .map_err(|e| GatewayError::Upstream(format!("request failed: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let bytes = read_capped(resp).await.unwrap_or_default();
        let detail: String = String::from_utf8_lossy(&bytes).chars().take(500).collect();
        return Err(GatewayError::Upstream(format!("upstream {status}: {detail}")));
    }
    Ok(resp)
}

/// Models list. OpenAI proxies `GET base/v1/models`; Anthropic synthesizes a
/// one-entry list from the configured model (avoids an extra round trip).
pub async fn models(provider: &AiProvider) -> Result<Value, GatewayError> {
    if provider.protocol == Protocol::Anthropic {
        let data: Vec<Value> = provider
            .model
            .iter()
            .map(|m| json!({ "id": m, "object": "model" }))
            .collect();
        return Ok(json!({ "object": "list", "data": data }));
    }
    let key = provider.api_key.clone();
    let target = egress::validate(&provider.base_url, "/v1/models", provider.allow_local)
        .map_err(GatewayError::BadRequest)?;
    let client = egress::client(false).map_err(GatewayError::Upstream)?;
    let rb = apply_auth(client.get(target), provider, key.as_deref());
    let resp = rb
        .send()
        .await
        .map_err(|e| GatewayError::Upstream(format!("request failed: {e}")))?;
    let status = resp.status();
    let bytes = read_capped(resp).await?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| GatewayError::Upstream(format!("non-JSON upstream ({status}): {e}")))?;
    if !status.is_success() {
        return Err(GatewayError::Upstream(format!("upstream {status}")));
    }
    Ok(value)
}

/// Test = a real, cheap AUTHENTICATED round trip. `Ok` ⇒ reachable and, if a key
/// is set, authorized. OpenAI uses `GET /v1/models`; Anthropic can't (its models
/// list is synthesized), so it does a minimal `POST /v1/messages` — a bad key or
/// unreachable host is caught at Test time, not on the first real chat.
pub async fn test(provider: &AiProvider) -> Result<(), GatewayError> {
    match provider.protocol {
        Protocol::OpenAi => models(provider).await.map(|_| ()),
        Protocol::Anthropic => anthropic_probe(provider).await,
    }
}

/// A minimal authenticated Anthropic call for the reachability/auth probe.
async fn anthropic_probe(provider: &AiProvider) -> Result<(), GatewayError> {
    let key = provider.api_key.clone();
    let target = egress::validate(&provider.base_url, "/v1/messages", provider.allow_local)
        .map_err(GatewayError::BadRequest)?;
    let model = provider
        .model
        .clone()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| {
            GatewayError::BadRequest("set a default model for this Anthropic provider before testing".into())
        })?;
    let body = json!({
        "model": model,
        "max_tokens": 1,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let client = egress::client(false).map_err(GatewayError::Upstream)?;
    let rb = apply_auth(with_json_body(client.post(target), &body)?, provider, key.as_deref());
    let resp = rb
        .send()
        .await
        .map_err(|e| GatewayError::Upstream(format!("request failed: {e}")))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    let bytes = read_capped(resp).await.unwrap_or_default();
    let detail: String = String::from_utf8_lossy(&bytes).chars().take(300).collect();
    Err(GatewayError::Upstream(format!("upstream {status}: {detail}")))
}

/// Ensure the request carries a model: the caller's takes precedence, else the
/// provider's configured default. There is NO hardcoded fallback — model ids
/// change too often to bake one in — so if neither is set this is a clear,
/// actionable error rather than a silently-stale guess.
fn ensure_model(provider: &AiProvider, body: &mut Value) -> Result<(), GatewayError> {
    if body.get("model").and_then(Value::as_str).is_some_and(|m| !m.is_empty()) {
        return Ok(());
    }
    match provider.model.as_deref().filter(|m| !m.is_empty()) {
        Some(m) => {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("model".into(), Value::String(m.to_string()));
            }
            Ok(())
        }
        None => Err(GatewayError::BadRequest(
            "no model specified — set a default model for this provider or include a \"model\" in the request".into(),
        )),
    }
}

/// OpenAI message `content` may be a plain string OR an array of content parts
/// (`[{ "type":"text", "text":"…" }]` — the SDK-emitted form). Flatten to text so
/// the Anthropic proxy (which sends a string body) never silently drops the
/// message. A `null` content (an assistant tool-call turn) flattens to empty.
fn message_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// OpenAI chat-completions body → Anthropic `/v1/messages` body.
fn openai_chat_to_anthropic(body: &Value) -> Value {
    // The model is guaranteed present by ensure_model() before this runs; empty
    // only in a direct unit call (Anthropic would then reject it, as it should).
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    let max = body
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);
    let mut system = String::new();
    let mut messages = Vec::new();
    if let Some(arr) = body.get("messages").and_then(Value::as_array) {
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = message_text(m.get("content"));
            if role == "system" {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&content);
            } else {
                // Anthropic accepts only user/assistant roles.
                let role = if role == "assistant" { "assistant" } else { "user" };
                messages.push(json!({ "role": role, "content": content }));
            }
        }
    }
    let mut out = json!({ "model": model, "max_tokens": max, "messages": messages });
    if !system.is_empty() {
        out["system"] = Value::String(system);
    }
    if let Some(t) = body.get("temperature") {
        out["temperature"] = t.clone();
    }
    if let Some(tp) = body.get("top_p") {
        out["top_p"] = tp.clone();
    }
    match body.get("stop") {
        Some(Value::String(s)) => out["stop_sequences"] = json!([s]),
        Some(Value::Array(a)) => out["stop_sequences"] = Value::Array(a.clone()),
        _ => {}
    }
    out
}

/// Anthropic `/v1/messages` response → OpenAI `chat.completion`.
fn anthropic_response_to_openai(resp: &Value) -> Value {
    let text = resp
        .get("content")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<String>()
        })
        .unwrap_or_default();
    let finish = match resp.get("stop_reason").and_then(Value::as_str) {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop", // end_turn / stop_sequence / null
    };
    let input_tokens = resp.pointer("/usage/input_tokens").and_then(Value::as_u64).unwrap_or(0);
    let output_tokens = resp.pointer("/usage/output_tokens").and_then(Value::as_u64).unwrap_or(0);
    // `created` is a required field on the OpenAI chat.completion object (strict
    // SDK consumers reject its absence); the OpenAI passthrough path carries the
    // upstream's, so synthesize one here for parity. Saturating add so hostile
    // upstream token counts can't overflow.
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut out = json!({
        "object": "chat.completion",
        "created": created,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens.saturating_add(output_tokens),
        },
    });
    if let Some(id) = resp.get("id") {
        out["id"] = id.clone();
    }
    if let Some(m) = resp.get("model") {
        out["model"] = m.clone();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::providers::AiProvider;

    fn anthropic_provider() -> AiProvider {
        AiProvider {
            id: "claude".into(),
            name: "Claude".into(),
            category: None,
            protocol: Protocol::Anthropic,
            base_url: "https://api.anthropic.com".into(),
            model: Some("example-model".into()),
            capabilities: vec![],
            enabled: true,
            allow_local: false,
            api_key: Some("sk-ant".into()),
        }
    }

    #[test]
    fn anthropic_is_not_streamable_but_openai_is() {
        assert!(!streamable(&anthropic_provider()));
        let mut p = anthropic_provider();
        p.protocol = Protocol::OpenAi;
        assert!(streamable(&p));
    }

    #[test]
    fn openai_to_anthropic_pulls_system_out_and_defaults_max_tokens() {
        let body = json!({
            "model": "example-model",
            "messages": [
                { "role": "system", "content": "be terse" },
                { "role": "user", "content": "hi" }
            ]
        });
        let a = openai_chat_to_anthropic(&body);
        assert_eq!(a["system"], "be terse");
        assert_eq!(a["max_tokens"], 1024);
        assert_eq!(a["messages"].as_array().unwrap().len(), 1);
        assert_eq!(a["messages"][0]["role"], "user");
        // The model is passed through as-is (no hardcoded fallback).
        assert_eq!(a["model"], "example-model");
    }

    #[test]
    fn ensure_model_uses_request_then_provider_then_errors() {
        let mut p = anthropic_provider();
        // A model on the request wins.
        let mut body = json!({ "model": "req-model", "messages": [] });
        ensure_model(&p, &mut body).unwrap();
        assert_eq!(body["model"], "req-model");
        // Else the provider's configured default fills in.
        let mut body = json!({ "messages": [] });
        ensure_model(&p, &mut body).unwrap();
        assert_eq!(body["model"], "example-model");
        // Neither set → a clear error, NOT a stale hardcoded guess.
        p.model = None;
        let mut body = json!({ "messages": [] });
        let err = ensure_model(&p, &mut body).unwrap_err();
        assert!(matches!(err, GatewayError::BadRequest(_)));
        assert!(err.message().contains("no model"));
    }

    #[test]
    fn anthropic_response_maps_to_openai_chat_shape() {
        let resp = json!({
            "id": "msg_1",
            "model": "example-model",
            "content": [{ "type": "text", "text": "hello" }, { "type": "text", "text": " world" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 2 }
        });
        let o = anthropic_response_to_openai(&resp);
        assert_eq!(o["object"], "chat.completion");
        assert_eq!(o["choices"][0]["message"]["content"], "hello world");
        assert_eq!(o["choices"][0]["message"]["role"], "assistant");
        assert_eq!(o["choices"][0]["finish_reason"], "stop");
        assert_eq!(o["usage"]["prompt_tokens"], 5);
        assert_eq!(o["usage"]["completion_tokens"], 2);
        assert_eq!(o["usage"]["total_tokens"], 7);
        assert_eq!(o["id"], "msg_1");
    }

    #[test]
    fn max_tokens_stop_reason_maps_to_length() {
        let resp = json!({ "content": [{"text":"x"}], "stop_reason": "max_tokens" });
        assert_eq!(anthropic_response_to_openai(&resp)["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn openai_to_anthropic_flattens_array_form_content() {
        // OpenAI SDKs emit content as an array of parts; it must NOT be dropped.
        let body = json!({
            "messages": [
                { "role": "system", "content": [{ "type": "text", "text": "be terse" }] },
                { "role": "user", "content": [{ "type": "text", "text": "What is " }, { "type": "text", "text": "2+2?" }] }
            ]
        });
        let a = openai_chat_to_anthropic(&body);
        assert_eq!(a["system"], "be terse");
        assert_eq!(a["messages"][0]["content"], "What is 2+2?", "array parts must be flattened, not dropped");
    }

    #[test]
    fn anthropic_response_carries_created_and_saturates_tokens() {
        let resp = json!({
            "content": [{ "text": "x" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": u64::MAX, "output_tokens": 5 }
        });
        let o = anthropic_response_to_openai(&resp);
        assert!(o["created"].as_u64().is_some(), "created is a required chat.completion field");
        assert_eq!(o["usage"]["total_tokens"], u64::MAX, "total_tokens must saturate, not overflow");
    }

    #[tokio::test]
    async fn anthropic_models_are_synthesized_from_the_profile() {
        // No network: the Anthropic models() path short-circuits to the profile.
        let out = models(&anthropic_provider()).await.unwrap();
        assert_eq!(out["data"][0]["id"], "example-model");
    }
}
