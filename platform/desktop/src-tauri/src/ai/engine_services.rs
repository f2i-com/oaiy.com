//! OAIY's engine's models as services for flows.
//!
//! `GET /api/ai/engine/services`: every model the engine has configured whose
//! files are on this machine, per kind (the language model, pictures, video,
//! speech, music, sound effects, 3D models, background removal, upscaling),
//! each with how a flow node calls it: the endpoint, the method, the body
//! template, where the answer is and what it is, and for the kinds that run as
//! jobs how to poll one, fetch what it made and stop it. A kind with no such
//! model (or whose route is switched off) is not listed. Asking starts
//! nothing: the gateway answers discovery from its configuration, and when the
//! engines are not running the list is empty.
//!
//! `/api/ai/engine/gateway/...`: those calls, forwarded to the engines'
//! gateway. The flow editor's page cannot call the gateway itself (the gateway
//! answers only the origins in its own configuration, which the desktop leaves
//! alone), and a flow the desktop runs reaches the desktop anyway. Only the
//! media routes are forwarded: the language model is reached as the
//! `oaiy-engine` provider, which answers with the model chosen in Engines.
//!
//! OAIY Voice's transcription is listed with them while its service is
//! installed (`POST /api/voice/transcribe`: a 16 kHz mono WAV in, `{text}` out).

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{any, get},
    Json, Router,
};
use serde_json::{json, Value};

use super::routes::{ai_error, chosen_model, engine_discovery, AiState, ENGINE_PROVIDER_ID};

/// Where the forwarded calls go: this, then the gateway's own path.
pub const GATEWAY_PREFIX: &str = "/api/ai/engine/gateway";
/// Pictures, video and 3D models come in as data: URLs.
const FORWARD_BODY_LIMIT: usize = 96 << 20;
/// The gateway routes a flow may reach through the desktop.
const MEDIA_TARGETS: [&str; 10] = ["images", "edits", "videos", "speech", "voices", "music", "sound", "model3d", "background", "upscale"];

/// A job's sub-routes, below its endpoint (`{id}` is the job's id).
struct Job {
    status: &'static str,
    content: &'static str,
    /// When `content` needs a converter the engine may lack (FFmpeg for MP3), the plain file.
    fallback: Option<&'static str>,
    cancel: Option<&'static str>,
}

/// One kind of model, and how a node calls it.
struct Kind {
    /// Its list in discovery's `models`.
    kind: &'static str,
    /// The gateway route that serves it (discovery's endpoint `name`).
    target: &'static str,
    label: &'static str,
    /// What one model of it is called, for "the engine's default …".
    noun: &'static str,
    icon: &'static str,
    what: &'static str,
    /// The flow nodes it is offered in.
    node_types: &'static [&'static str],
    /// `{{name}}` is the node's value as JSON (`null` when it has none, and a
    /// field left `null` is not sent).
    body: &'static str,
    /// `json` (the answer at `response_path`) or `binary` (the answer is the file).
    response_type: &'static str,
    response_path: &'static str,
    /// What it makes: image | video | audio | model3d | text.
    output: &'static str,
    /// The file's format (a base64 answer or a job's content).
    format: &'static str,
    job: Option<Job>,
    /// The Service Call node's inputs for it (id, label, type).
    inputs: &'static [(&'static str, &'static str, &'static str)],
}

const KINDS: [Kind; 8] = [
    Kind {
        kind: "image",
        target: "images",
        label: "Image",
        noun: "picture model",
        icon: "🎨",
        what: "Pictures from a prompt; with reference pictures, edits",
        node_types: &["image_gen", "service_call"],
        body: r#"{"model": {{model}}, "prompt": {{prompt}}, "negative_prompt": {{negative_prompt}}, "size": {{size}}, "images": {{images}}, "seed": {{seed}}, "n": 1, "response_format": "b64_json"}"#,
        response_type: "json",
        response_path: "data.0.b64_json",
        output: "image",
        format: "png",
        job: None,
        inputs: &[("prompt", "Prompt", "string")],
    },
    Kind {
        kind: "video",
        target: "videos",
        label: "Video",
        noun: "video model",
        icon: "🎬",
        what: "Video from a prompt, and a start picture if given",
        node_types: &["video_gen", "service_call"],
        body: r#"{"model": {{model}}, "prompt": {{prompt}}, "seconds": {{seconds}}, "size": {{size}}, "input_reference": {{image}}, "seed": {{seed}}}"#,
        response_type: "json",
        response_path: "",
        output: "video",
        format: "mp4",
        job: Some(Job { status: "/{id}", content: "/{id}/content", fallback: None, cancel: None }),
        inputs: &[("prompt", "Prompt", "string"), ("image", "Start picture", "image")],
    },
    Kind {
        kind: "speech",
        target: "speech",
        label: "Speech",
        noun: "speech model",
        icon: "🎙️",
        what: "Speech from text, in a saved voice or one described",
        node_types: &["text_to_speech", "service_call"],
        body: r#"{"model": {{model}}, "input": {{text}}, "voice": {{voice}}, "instructions": {{instructions}}, "language": {{language}}, "response_format": "wav"}"#,
        response_type: "binary",
        response_path: "",
        output: "audio",
        format: "wav",
        job: None,
        inputs: &[("text", "Text", "string")],
    },
    Kind {
        kind: "music",
        target: "music",
        label: "Music",
        noun: "music model",
        icon: "🎵",
        what: "Songs with vocals and instruments from a description and lyrics",
        node_types: &["music_gen", "service_call"],
        body: r#"{"model": {{model}}, "prompt": {{prompt}}, "lyrics": {{lyrics}}, "instrumental": {{instrumental}}, "duration": {{duration}}, "seed": {{seed}}}"#,
        response_type: "json",
        response_path: "",
        output: "audio",
        format: "mp3",
        job: Some(Job { status: "/{id}", content: "/{id}/content?format=mp3", fallback: Some("/{id}/content"), cancel: Some("/{id}/cancel") }),
        inputs: &[("prompt", "Style", "string"), ("lyrics", "Lyrics", "string")],
    },
    Kind {
        kind: "sound",
        target: "sound",
        label: "Sound effect",
        noun: "sound-effect model",
        icon: "🔊",
        what: "Sound effects from a description, up to 30 seconds",
        node_types: &["sound_effect", "service_call"],
        body: r#"{"model": {{model}}, "prompt": {{prompt}}, "seconds": {{seconds}}, "seed": {{seed}}}"#,
        response_type: "json",
        response_path: "",
        output: "audio",
        format: "wav",
        job: Some(Job { status: "/{id}", content: "/{id}/content", fallback: None, cancel: Some("/{id}/cancel") }),
        inputs: &[("prompt", "Description", "string")],
    },
    Kind {
        kind: "model3d",
        target: "model3d",
        label: "3D model",
        noun: "3D model",
        icon: "🧊",
        what: "A 3D model (GLB) from a picture of an object",
        node_types: &["model_3d", "service_call"],
        body: r#"{"model": {{model}}, "image": {{image}}, "resolution": {{resolution}}, "faces": {{faces}}, "seed": {{seed}}}"#,
        response_type: "json",
        response_path: "",
        output: "model3d",
        format: "glb",
        job: Some(Job { status: "/{id}", content: "/{id}/content", fallback: None, cancel: Some("/{id}/cancel") }),
        inputs: &[("image", "Picture", "image")],
    },
    Kind {
        kind: "background",
        target: "background",
        label: "Background removal",
        noun: "background-removal model",
        icon: "✂️",
        what: "A picture's background removed (transparent PNG)",
        node_types: &["background_removal", "service_call"],
        body: r#"{"image": {{image}}, "response_format": "b64_json"}"#,
        response_type: "json",
        response_path: "data.0.b64_json",
        output: "image",
        format: "png",
        job: None,
        inputs: &[("image", "Picture", "image")],
    },
    Kind {
        kind: "upscale",
        target: "upscale",
        label: "Upscale",
        noun: "upscaler",
        icon: "🔍",
        what: "A picture made two or four times larger",
        node_types: &["image_upscale", "service_call"],
        body: r#"{"image": {{image}}, "scale": {{scale}}, "response_format": "b64_json"}"#,
        response_type: "json",
        response_path: "data.0.b64_json",
        output: "image",
        format: "png",
        job: None,
        inputs: &[("image", "Picture", "image")],
    },
];

pub fn router(state: AiState) -> Router {
    Router::new()
        .route("/api/ai/engine/services", get(list))
        // Pictures come as data: URLs, well past axum's 2 MB default.
        .route(&format!("{GATEWAY_PREFIX}/*path"), any(forward).layer(DefaultBodyLimit::max(FORWARD_BODY_LIMIT)))
        .with_state(state)
}

/// `GET /api/ai/engine/services`.
async fn list(State(st): State<AiState>) -> Response {
    let voice = voice_installed(&st).then(voice_entry).into_iter().collect::<Vec<_>>();
    match engine_discovery(&st).await {
        Ok((_, discovery)) => {
            let mut services = services_from_discovery(&discovery);
            services.extend(voice);
            Json(json!({ "running": true, "services": services })).into_response()
        }
        Err(reason) => Json(json!({ "running": false, "reason": reason, "services": voice })).into_response(),
    }
}

fn voice_installed(st: &AiState) -> bool {
    st.registry.lock().ok().is_some_and(|r| r.snapshot().services.iter().any(|s| s.id == crate::voice::engines::STT_SERVICE && s.installed))
}

/// OAIY Voice's speech-to-text (Parakeet), as the calls use it.
fn voice_entry() -> Value {
    json!({
        "id": "voice:transcribe",
        "kind": "transcription",
        "model": "parakeet",
        "default": true,
        "group": "engine",
        "name": "OAIY Voice · Transcription · Parakeet",
        "description": "Writes out what is said in a recording (OAIY Voice, started when needed)",
        "icon": "📝",
        "nodeTypes": ["speech_to_text"],
        "endpoint": "/api/voice/transcribe",
        "method": "POST",
        "requestFormat": "wav16k",
        "bodyTemplate": "",
        "responseType": "json",
        "responsePath": "text",
        "output": "text",
    })
}

fn id(m: &Value) -> &str {
    m.get("id").and_then(Value::as_str).unwrap_or("")
}

/// A model that can run: its parts configured and its files here. (A gateway
/// older than `files_present` says nothing, and its models are taken as here.)
fn runnable(m: &Value) -> bool {
    !id(m).is_empty() && m.get("files_present").and_then(Value::as_bool) != Some(false) && m.get("ready").and_then(Value::as_bool) != Some(false)
}

/// The gateway path serving `target` (an OpenAI-dialect route; images and
/// videos also have OAIY-dialect ones, which these contracts do not speak).
fn endpoint_path(discovery: &Value, target: &str) -> Option<String> {
    discovery.get("endpoints").and_then(Value::as_array)?.iter().find(|e| {
        e.get("name").and_then(Value::as_str) == Some(target) && e.get("spec").and_then(Value::as_str).unwrap_or("openai") == "openai"
    })
    .and_then(|e| e.get("path").and_then(Value::as_str))
    .map(|p| p.trim_end_matches('/').to_string())
    .filter(|p| p.starts_with('/'))
}

/// The engine's models as flow services (see the module docs). Defaults first within a kind.
pub fn services_from_discovery(discovery: &Value) -> Vec<Value> {
    let models = |kind: &str| -> Vec<Value> {
        let mut list: Vec<Value> = discovery.pointer(&format!("/models/{kind}")).and_then(Value::as_array).cloned().unwrap_or_default().into_iter().filter(runnable).collect();
        list.sort_by_key(|m| m.get("default").and_then(Value::as_bool) != Some(true));
        list
    };
    let mut out = Vec::new();
    // The language model: the one chosen in Engines (a flow naming another would load it in its place).
    let chosen = chosen_model(discovery);
    if endpoint_path(discovery, "chat").is_some() && !chosen.is_empty() {
        if let Some(m) = models("llm").into_iter().find(|m| id(m) == chosen) {
            out.push(llm_entry(&chosen, &m));
        }
    }
    for k in &KINDS {
        let Some(path) = endpoint_path(discovery, k.target) else { continue };
        for m in models(k.kind) {
            out.push(media_entry(k, &path, &m));
        }
    }
    out
}

fn llm_entry(model: &str, m: &Value) -> Value {
    let vision = m.get("vision").and_then(Value::as_bool) == Some(true);
    json!({
        "id": format!("engine:llm:{model}"),
        "kind": "llm",
        "model": model,
        "default": true,
        "group": "engine",
        "name": format!("OAIY engine · LLM · {model}"),
        "description": format!("The language model chosen in Engines{}", if vision { ", reads pictures" } else { "" }),
        "icon": "🤖",
        "nodeTypes": ["ai_llm", "service_call"],
        "endpoint": format!("/api/ai/providers/{ENGINE_PROVIDER_ID}/v1/chat/completions"),
        "method": "POST",
        "apiFormat": "openai",
        // For the Service Call node; the AI node sends its own OpenAI request.
        "bodyTemplate": r#"{"messages": [{"role": "system", "content": {{system}}}, {"role": "user", "content": {{input}}}], "stream": false}"#,
        "responseType": "json",
        "responsePath": "choices.0.message.content",
    })
}

fn media_entry(k: &Kind, path: &str, m: &Value) -> Value {
    let model = id(m);
    let default = m.get("default").and_then(Value::as_bool) == Some(true);
    let endpoint = format!("{GATEWAY_PREFIX}{path}");
    let mut e = json!({
        "id": format!("engine:{}:{model}", k.kind),
        "kind": k.kind,
        "model": model,
        "default": default,
        "group": "engine",
        "name": format!("OAIY engine · {} · {model}", k.label),
        "description": if default { format!("{} (the engine's default {})", k.what, k.noun) } else { k.what.to_string() },
        "icon": k.icon,
        "nodeTypes": k.node_types,
        "endpoint": endpoint,
        "method": "POST",
        "bodyTemplate": k.body,
        "responseType": k.response_type,
        "responsePath": k.response_path,
        "output": k.output,
        "outputFormat": k.format,
        "inputs": k.inputs.iter().map(|(id, name, ty)| json!({ "id": id, "name": name, "type": ty })).collect::<Vec<_>>(),
    });
    if let Some(j) = &k.job {
        let at = |sub: &str| format!("{endpoint}{sub}");
        e["job"] = json!({
            "idPath": "id",
            "statusUrl": at(j.status),
            "statusPath": "status",
            "progressPath": "progress",
            "done": ["completed"],
            "failed": ["failed", "cancelled"],
            "errorPath": "error.message",
            "contentUrl": at(j.content),
            "contentFallbackUrl": j.fallback.map(at),
            "cancelUrl": j.cancel.map(at),
        });
    }
    e
}

/// Whether a gateway path is one a flow may reach: below a media route's path.
fn forwardable(discovery: &Value, path: &str) -> bool {
    MEDIA_TARGETS.iter().filter_map(|t| endpoint_path(discovery, t)).any(|p| path == p || path.starts_with(&format!("{p}/")))
}

/// `/api/ai/engine/gateway/<gateway path>`: the call, sent on to the gateway and its answer streamed back.
async fn forward(State(st): State<AiState>, req: Request) -> Response {
    match engine_discovery(&st).await {
        Ok((gateway, discovery)) => forward_to(&gateway, &discovery, req).await,
        Err(e) => ai_error(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
    }
}

/// Send `req` (a `GATEWAY_PREFIX` route) on to `gateway`, if `discovery` has it as a media route.
async fn forward_to(gateway: &str, discovery: &Value, req: Request) -> Response {
    let method = req.method().as_str().to_ascii_uppercase();
    if !matches!(method.as_str(), "GET" | "POST" | "DELETE") {
        return ai_error(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", format!("{method} is not forwarded"));
    }
    let path = req.uri().path().strip_prefix(GATEWAY_PREFIX).unwrap_or("").to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    if !forwardable(discovery, &path) {
        return ai_error(StatusCode::NOT_FOUND, "not_forwarded", format!("{path} is not one of the engine's media routes"));
    }
    let content_type = req.headers().get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let range = req.headers().get(axum::http::header::RANGE).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let body = match axum::body::to_bytes(req.into_body(), FORWARD_BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return ai_error(StatusCode::PAYLOAD_TOO_LARGE, "too_large", e.to_string()),
    };
    // No overall timeout: a picture or a song is answered when it is made.
    let client = match reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(5)).build() {
        Ok(c) => c,
        Err(e) => return ai_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    };
    let Ok(m) = reqwest::Method::from_bytes(method.as_bytes()) else {
        return ai_error(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", method);
    };
    let mut call = client.request(m, format!("{gateway}{path}{query}"));
    if let Some(ct) = content_type {
        call = call.header("content-type", ct);
    }
    if let Some(r) = range {
        call = call.header("range", r);
    }
    if !body.is_empty() {
        call = call.body(body.to_vec());
    }
    let upstream = match call.send().await {
        Ok(r) => r,
        Err(e) => return ai_error(StatusCode::BAD_GATEWAY, "engine_unreachable", format!("the engine did not answer: {e}")),
    };
    let mut out = Response::builder().status(upstream.status().as_u16());
    for name in ["content-type", "content-length", "content-disposition", "content-range", "accept-ranges", "x-sample-rate"] {
        if let Some(v) = upstream.headers().get(name).and_then(|v| v.to_str().ok()) {
            out = out.header(name, v);
        }
    }
    out.body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|_| ai_error(StatusCode::BAD_GATEWAY, "upstream_error", "the answer could not be passed on".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovery() -> Value {
        json!({
            "endpoints": [
                {"name": "chat", "path": "/v1/chat/completions", "spec": "openai"},
                {"name": "images", "path": "/v1/images/generations", "spec": "openai"},
                {"name": "images", "path": "/oaiy/images", "spec": "oaiy"},
                {"name": "videos", "path": "/v1/videos", "spec": "openai"},
                {"name": "speech", "path": "/v1/audio/speech", "spec": "openai"},
                {"name": "music", "path": "/v1/audio/music", "spec": "openai"},
                {"name": "model3d", "path": "/v1/3d/models", "spec": "openai"},
                {"name": "background", "path": "/v1/images/background_removal", "spec": "openai"},
                {"name": "files", "path": "/files", "spec": "openai"},
            ],
            "models": {
                "llm": [{"id": "flash", "default": true, "files_present": true}, {"id": "big", "default": false}],
                "image": [{"id": "sdxl", "default": false}, {"id": "qwen", "default": true, "files_present": true}, {"id": "moved", "files_present": false, "missing_files": ["base"]}],
                "video": [],
                "speech": [{"id": "tts", "default": true}],
                "music": [{"id": "song", "default": true, "files_present": true}],
                // A sound model, but its route is switched off.
                "sound": [{"id": "moss", "default": true}],
                "model3d": [{"id": "pixal3d", "default": true, "ready": false}],
                "background": [{"id": "birefnet", "default": true, "ready": true, "files_present": true}],
                "upscale": [],
            },
            "defaults": {"llm": "flash", "image": "qwen"},
        })
    }

    fn ids(list: &[Value]) -> Vec<&str> {
        list.iter().map(|s| s["id"].as_str().unwrap()).collect()
    }

    #[test]
    fn only_kinds_with_a_model_that_can_run_are_listed() {
        let list = services_from_discovery(&discovery());
        assert_eq!(
            ids(&list),
            vec!["engine:llm:flash", "engine:image:qwen", "engine:image:sdxl", "engine:speech:tts", "engine:music:song", "engine:background:birefnet"],
            "no video or upscale model; the moved picture model's files are gone; sound's route is off; the 3D model is not ready; only the chosen LLM"
        );
        assert!(services_from_discovery(&json!({})).is_empty());
    }

    #[test]
    fn each_entry_carries_its_call() {
        let list = services_from_discovery(&discovery());
        let by = |id: &str| list.iter().find(|s| s["id"] == id).unwrap().clone();

        let llm = by("engine:llm:flash");
        assert_eq!(llm["endpoint"], "/api/ai/providers/oaiy-engine/v1/chat/completions");
        assert_eq!((llm["apiFormat"].as_str(), llm["nodeTypes"][0].as_str()), (Some("openai"), Some("ai_llm")));

        let image = by("engine:image:qwen");
        assert_eq!(image["name"], "OAIY engine · Image · qwen");
        assert_eq!(image["endpoint"], "/api/ai/engine/gateway/v1/images/generations", "the OpenAI route, not the OAIY-dialect one");
        assert_eq!((image["responsePath"].as_str(), image["output"].as_str()), (Some("data.0.b64_json"), Some("image")));
        assert_eq!(image["default"], true);
        assert_eq!(image["description"], "Pictures from a prompt; with reference pictures, edits (the engine's default picture model)");
        assert!(image.get("job").is_none());

        let music = by("engine:music:song");
        assert_eq!(music["name"], "OAIY engine · Music · song");
        assert_eq!(music["nodeTypes"], json!(["music_gen", "service_call"]));
        assert_eq!(music["job"]["statusUrl"], "/api/ai/engine/gateway/v1/audio/music/{id}");
        assert_eq!(music["job"]["contentUrl"], "/api/ai/engine/gateway/v1/audio/music/{id}/content?format=mp3");
        assert_eq!(music["job"]["contentFallbackUrl"], "/api/ai/engine/gateway/v1/audio/music/{id}/content");
        assert_eq!(music["job"]["cancelUrl"], "/api/ai/engine/gateway/v1/audio/music/{id}/cancel");
        assert!(music["bodyTemplate"].as_str().unwrap().contains("{{lyrics}}"));

        let speech = by("engine:speech:tts");
        assert_eq!((speech["responseType"].as_str(), speech["output"].as_str()), (Some("binary"), Some("audio")));

        let cutout = by("engine:background:birefnet");
        assert_eq!(cutout["inputs"], json!([{"id": "image", "name": "Picture", "type": "image"}]));
        assert_eq!(cutout["nodeTypes"][0], "background_removal");
    }

    #[test]
    fn only_media_routes_are_forwarded() {
        let d = discovery();
        assert!(forwardable(&d, "/v1/audio/music"));
        assert!(forwardable(&d, "/v1/audio/music/music_1/content"));
        assert!(forwardable(&d, "/v1/images/background_removal"));
        assert!(!forwardable(&d, "/v1/audio/musicx"));
        assert!(!forwardable(&d, "/v1/chat/completions"), "the language model goes through the provider");
        assert!(!forwardable(&d, "/files/a.png"));
        assert!(!forwardable(&d, "/oaiy/images"), "the OAIY-dialect route is not one of the contracts");
    }

    #[tokio::test]
    async fn media_forwarding_keeps_caller_credentials_at_the_desktop_boundary() {
        let app = Router::new().route("/v1/images/generations", any(|req: Request| async move {
            assert!(req.headers().get("authorization").is_none());
            assert!(req.headers().get("cookie").is_none());
            assert_eq!(req.headers().get("content-type").unwrap(), "application/json");
            StatusCode::NO_CONTENT
        }));
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let gateway = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let req = axum::http::Request::builder()
            .method("POST")
            .uri(format!("{GATEWAY_PREFIX}/v1/images/generations"))
            .header("authorization", "Bearer desktop-only-test-sentinel")
            .header("cookie", "desktop-only-test-cookie=sentinel")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = forward_to(&gateway, &discovery(), req).await;
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        server.abort();
        assert!(server.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn independently_keyed_desktop_and_studio_refuse_cross_service_credentials_without_generation() {
        use crate::auth::{clock::ManualClock, guard::{scoped_guard, GuardConfig},
            scopes::ScopeSet, store::{AuthStore, MintSpec}, token::Kind as TokenKind,
            AccessMode, Guard};
        use axum::{http::HeaderMap, middleware};
        use std::{net::SocketAddr, sync::{Arc, atomic::{AtomicUsize, Ordering}}};

        const STUDIO_KEY: &str = "isolated-studio-service-test-key";
        let hits = Arc::new(AtomicUsize::new(0));
        let count = hits.clone();
        let keyed = |headers: &HeaderMap| headers.get("authorization")
            .and_then(|h| h.to_str().ok()) == Some("Bearer isolated-studio-service-test-key");
        let studio_app = Router::new()
            .route("/v1/discovery", get(move |headers: HeaderMap| async move {
                Json(if keyed(&headers) { discovery() } else { json!({"models":{},"endpoints":[]}) })
            }))
            .route("/v1/images/generations", any(move |headers: HeaderMap| {
                let count = count.clone();
                async move {
                    if keyed(&headers) { count.fetch_add(1, Ordering::SeqCst); StatusCode::NO_CONTENT }
                    else { StatusCode::UNAUTHORIZED }
                }
            }));
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let gateway = format!("http://{}", listener.local_addr().unwrap());
        let studio_task = tokio::spawn(async move { axum::serve(listener, studio_app).await.unwrap() });
        let client = reqwest::Client::new();
        // This is the same unauthenticated upstream discovery call used by engine_discovery.
        let hidden: Value = client.get(format!("{gateway}/v1/discovery")).send().await.unwrap().json().await.unwrap();
        assert!(services_from_discovery(&hidden).is_empty());
        let clock = Arc::new(ManualClock::new(1_790_000_000_000));
        let store = Arc::new(AuthStore::memory(clock.clone()));
        let mint = |scopes: &[&str]| store.mint(MintSpec::new(TokenKind::Pat,
            "two-hop test", ScopeSet::of(scopes), 60_000)).unwrap().token;
        let read = mint(&["ai.read"]);
        let infer = mint(&["ai.use"]);
        let (config, _) = GuardConfig::from_env(&|_| None, false, false, 19379);
        let guard = Arc::new(Guard::new(AccessMode::Scoped, config, store, None, None, clock));
        let upstream = gateway.clone();
        let desktop_app = Router::new().route(&format!("{GATEWAY_PREFIX}/*path"),
            any(move |req: Request| {
                let (g, d) = (upstream.clone(), hidden.clone());
                async move { forward_to(&g, &d, req).await }
            })).layer(middleware::from_fn_with_state(guard, scoped_guard));
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let desktop = format!("http://{}{GATEWAY_PREFIX}/v1/images/generations", listener.local_addr().unwrap());
        let desktop_task = tokio::spawn(async move {
            axum::serve(listener, desktop_app.into_make_service_with_connect_info::<SocketAddr>()).await.unwrap()
        });
        // Host matches the scoped guard's fixed fixture origin; listeners use only dynamic ports.
        for (key, status) in [(None, 401), (Some(read.as_str()), 403), (Some(infer.as_str()), 404), (Some(STUDIO_KEY), 401)] {
            let mut call = client.post(&desktop).header("host", "localhost:19379").json(&json!({"prompt":"synthetic; no generation"}));
            if let Some(key) = key { call = call.bearer_auth(key); }
            assert_eq!(call.send().await.unwrap().status().as_u16(), status);
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0, "desktop never reached an unadvertised image service");
        for (key, status) in [(None, 401), (Some(infer.as_str()), 401), (Some(STUDIO_KEY), 204)] {
            let mut call = client.post(format!("{gateway}/v1/images/generations")).json(&json!({"prompt":"synthetic; no generation"}));
            if let Some(key) = key { call = call.bearer_auth(key); }
            assert_eq!(call.send().await.unwrap().status().as_u16(), status);
        }
        assert_eq!(hits.load(Ordering::SeqCst), 1, "only its own service key reached the response stub; no engine is launched");
        desktop_task.abort(); studio_task.abort();
        assert!(desktop_task.await.unwrap_err().is_cancelled());
        assert!(studio_task.await.unwrap_err().is_cancelled());
    }

    /// An explicitly started isolated Studio can verify the scoped guard, the
    /// actual forwarding function, native worker and configured style LoRA
    /// together. Credentials live in memory and are never written to evidence.
    #[tokio::test]
    #[ignore = "requires OAIY_KLEIN_LOCAL_TEST_GATEWAY and idle GPU 1; creates real native PNGs"]
    async fn isolated_native_klein_through_scoped_gateway() {
        use crate::auth::{clock::ManualClock, guard::{scoped_guard, GuardConfig},
            scopes::ScopeSet, store::{AuthStore, MintSpec}, token::Kind as TokenKind,
            AccessMode, Guard};
        use axum::{extract::ConnectInfo, http::{Method, Request as HttpRequest}, middleware};
        use base64::Engine as _;
        use sha2::{Digest, Sha256};
        use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::{Duration, Instant}};
        use tower::ServiceExt;

        let gateway = std::env::var("OAIY_KLEIN_LOCAL_TEST_GATEWAY").expect("isolated gateway URL");
        let url = reqwest::Url::parse(&gateway).unwrap();
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        assert!(url.port().is_some_and(|p| ![8080, 7860, 17972, 17872].contains(&p)));
        let evidence = PathBuf::from(std::env::var("OAIY_KLEIN_LOCAL_TEST_EVIDENCE").expect("owned evidence folder"));
        assert!(evidence.is_absolute());
        std::fs::create_dir_all(&evidence).unwrap();
        let discovery: Value = reqwest::get(format!("{gateway}/v1/discovery"))
            .await.unwrap().json().await.unwrap();
        const MODEL: &str = "codex-klein-integration-only-70a1e37";
        let models = discovery["models"]["image"].as_array().unwrap();
        assert!(models.iter().any(|m| m["id"] == MODEL), "refuse an unmarked/active gateway");
        let services = services_from_discovery(&discovery);
        assert!(services.iter().any(|s| s["id"] == format!("engine:image:{MODEL}")));

        let clock = Arc::new(ManualClock::new(1_790_000_000_000));
        let store = Arc::new(AuthStore::memory(clock.clone()));
        let mint = |scopes: &[&str]| store.mint(MintSpec::new(TokenKind::Pat,
            "isolated test", ScopeSet::of(scopes), 60_000)).unwrap().token;
        let read = mint(&["models.read", "ai.read"]);
        let infer = mint(&["ai.use"]);
        let (config, warnings) = GuardConfig::from_env(&|_| None, false, false, 19379);
        assert!(warnings.is_empty());
        let guard = Arc::new(Guard::new(AccessMode::Scoped, config, store, None, None, clock));
        let (g, d) = (gateway.clone(), discovery.clone());
        let app = Router::new().route(&format!("{GATEWAY_PREFIX}/*path"),
            any(move |req: Request| {
                let (g, d) = (g.clone(), d.clone());
                async move { forward_to(&g, &d, req).await }
            })).layer(middleware::from_fn_with_state(guard, scoped_guard));
        let mut records = Vec::new();
        let mut hashes = Vec::new();
        for (name, model, use_loras) in [("baseline", MODEL.to_string(), false),
            ("style", MODEL.to_string(), true), ("zero", format!("{MODEL}-zero"), true)] {
            let state = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
                .output().expect("GPU preflight");
            assert!(state.status.success());
            let gpu = String::from_utf8(state.stdout).unwrap();
            let row = gpu.lines().find(|r| r.split(',').next().unwrap().trim() == "1").unwrap();
            let values: Vec<usize> = row.split(',').map(|v| v.trim().parse().unwrap()).collect();
            assert!(values[1] >= 20_000 && values[2] <= 5, "GPU 1 is not idle; no model will be unloaded");
            let body = json!({"model": model,
                "prompt": "a simple clean vector illustration of a fox, flat colors, crisp outlines, white background",
                "size": "512x512", "seed": 747, "steps": 4, "cfg": 1,
                "images": [], "negative_prompt": "", "use_loras": use_loras,
                "response_format": "b64_json"});
            let request = |token: Option<&str>| {
                let mut b = HttpRequest::builder().method(Method::POST)
                    .uri(format!("{GATEWAY_PREFIX}/v1/images/generations"))
                    .header("host", "localhost:19379").header("content-type", "application/json");
                if let Some(token) = token { b = b.header("authorization", format!("Bearer {token}")); }
                let mut r = b.body(Body::from(body.to_string())).unwrap();
                r.extensions_mut().insert(ConnectInfo("127.0.0.1:50000".parse::<SocketAddr>().unwrap()));
                r
            };
            assert_eq!(app.clone().oneshot(request(None)).await.unwrap().status(), StatusCode::UNAUTHORIZED);
            assert_eq!(app.clone().oneshot(request(Some(&read))).await.unwrap().status(), StatusCode::FORBIDDEN);
            let started = Instant::now();
            let response = tokio::time::timeout(Duration::from_secs(600), app.clone().oneshot(request(Some(&infer))))
                .await.expect("native generation timed out").unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), FORWARD_BODY_LIMIT).await.unwrap();
            assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            let png = base64::engine::general_purpose::STANDARD
                .decode(value["data"][0]["b64_json"].as_str().expect("native PNG response")).unwrap();
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
            assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 512);
            assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 512);
            let hash = format!("{:x}", Sha256::digest(&png));
            let path = evidence.join(format!("{name}.png"));
            std::fs::write(&path, png).unwrap();
            records.push(json!({"run": name, "request": body, "seconds": started.elapsed().as_secs_f64(),
                "scope_statuses": {"anonymous":401,"read_only":403,"ai_use":200},
                "png":path,"sha256":hash,"gpu_preflight":gpu}));
            hashes.push(hash);
            std::fs::write(evidence.join("scoped-native-results.json"),
                serde_json::to_vec_pretty(&records).unwrap()).unwrap();
        }
        assert_ne!(hashes[0], hashes[1], "style LoRA had no effect");
        assert_eq!(hashes[0], hashes[2], "zero-strength LoRA changed the native baseline");
    }

    /// Against the running engines (`OAIY_LIVE_GATEWAY`, e.g. http://127.0.0.1:8080):
    /// lists their models as the desktop does and serves the two routes on
    /// `OAIY_LIVE_PORT` for `OAIY_LIVE_SECONDS`, so a flow editor pointed at them
    /// runs through this code. Starts nothing itself; a client's call runs a job.
    ///
    ///     OAIY_LIVE_GATEWAY=http://127.0.0.1:8080 OAIY_LIVE_PORT=17999 OAIY_LIVE_SECONDS=300 \
    ///       cargo test --lib ai::engine_services -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "needs running engines"]
    async fn live_engine_routes() {
        let Ok(gateway) = std::env::var("OAIY_LIVE_GATEWAY") else { return };
        let gateway = gateway.trim_end_matches('/').to_string();
        let discovery: Value = reqwest::get(format!("{gateway}/v1/discovery")).await.unwrap().json().await.unwrap();
        let services = services_from_discovery(&discovery);
        println!("{}", serde_json::to_string_pretty(&json!({ "services": services })).unwrap());
        assert!(!services.is_empty(), "the engines list no model that can run");
        let port: u16 = std::env::var("OAIY_LIVE_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(0);
        let seconds: u64 = std::env::var("OAIY_LIVE_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        if port == 0 || seconds == 0 {
            return;
        }
        let (g, d) = (gateway.clone(), discovery.clone());
        let app = Router::new()
            .route("/api/ai/engine/services", get(move || async move { Json(json!({ "running": true, "services": services })) }))
            .route(
                &format!("{GATEWAY_PREFIX}/*path"),
                any(move |req: Request| {
                    let (g, d) = (g.clone(), d.clone());
                    async move { forward_to(&g, &d, req).await }
                })
                .layer(DefaultBodyLimit::max(FORWARD_BODY_LIMIT)),
            )
            .layer(tower_http::cors::CorsLayer::permissive());
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
        println!("serving the engine routes on http://127.0.0.1:{port} for {seconds} s");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(seconds), axum::serve(listener, app)).await;
    }

    #[test]
    fn oaiy_voice_is_offered_for_transcription() {
        let v = voice_entry();
        assert_eq!(v["nodeTypes"], json!(["speech_to_text"]));
        assert_eq!((v["endpoint"].as_str(), v["requestFormat"].as_str(), v["responsePath"].as_str()), (Some("/api/voice/transcribe"), Some("wav16k"), Some("text")));
    }
}
