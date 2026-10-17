//! A worker's listener on the LAN: `/worker/v1/*` over mutual TLS ([`super::tls`]).
//!
//! Anyone on the LAN may ask `health` (what this node is) and ask to pair (the request waits for the person here to
//! approve it, with the code both screens show). Everything else answers a paired primary's certificate alone: this
//! computer's capabilities (its engines' discovery and resources, private paths taken out), its language model's
//! chat, and its engines' media routes and files. Nothing else of the computer is reachable here: no settings,
//! plugins, Agent, files beyond what its engines made, or pairing of anything but a primary.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::identity::{fingerprint, pairing_code, Identity};
use super::store::{now_ms, Peer, Role};
use super::{lock, tls, Network, PROTOCOL};

/// Pairing requests waiting at once, and how long one waits.
const MAX_WAITING: usize = 8;
const REQUEST_TTL: Duration = Duration::from_secs(5 * 60);
/// Connections served at once.
const MAX_CONNECTIONS: usize = 64;
const CHAT_LIMIT: usize = 16 << 20;
const MEDIA_LIMIT: usize = 96 << 20;

/// A primary asking this worker to pair, waiting for the person's approval.
#[derive(Clone, Debug)]
pub(crate) struct Incoming {
    pub id: String,
    pub node_id: String,
    pub name: String,
    pub fingerprint: String,
    pub address: String,
    pub code: String,
    pub status: Status,
    pub at: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Status {
    Pending,
    Approved,
    Denied,
}

/// Who is calling: the certificate its key signed the handshake with (none: it offered none), and from where.
#[derive(Clone, Debug)]
pub(crate) struct Caller {
    pub fingerprint: Option<String>,
    pub addr: SocketAddr,
}

/// Listen on `bind` (TLS, this computer's certificate), serving the worker's routes until the returned sender says stop.
pub(crate) async fn listen(net: Network, id: &Identity, bind: SocketAddr) -> Result<(SocketAddr, tokio::sync::watch::Sender<bool>), String> {
    let listener = tokio::net::TcpListener::bind(bind).await.map_err(|e| format!("the worker could not listen on {bind}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let acceptor = tokio_rustls::TlsAcceptor::from(tls::server_config(id)?);
    let app = routes(net);
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    let slots = Arc::new(tokio::sync::Semaphore::new(MAX_CONNECTIONS));
    tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                a = listener.accept() => a,
                _ = stopped.changed() => break,
            };
            let Ok((tcp, peer)) = accepted else { continue };
            // (the LAN alone: a connection from anywhere else is closed before TLS)
            if !tls::local_address(peer.ip()) {
                continue;
            }
            let Ok(slot) = slots.clone().try_acquire_owned() else { continue };
            let (acceptor, app, mut stopped) = (acceptor.clone(), app.clone(), stopped.clone());
            tokio::spawn(async move {
                let _slot = slot;
                let Ok(Ok(tls)) = tokio::time::timeout(Duration::from_secs(10), acceptor.accept(tcp)).await else { return };
                let fp = tls.get_ref().1.peer_certificates().and_then(|c| c.first()).map(|c| fingerprint(c.as_ref()));
                let svc = hyper_util::service::TowerToHyperService::new(app.layer(Extension(Caller { fingerprint: fp, addr: peer })));
                let builder = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new());
                let conn = builder.serve_connection_with_upgrades(hyper_util::rt::TokioIo::new(tls), svc);
                tokio::pin!(conn);
                tokio::select! {
                    _ = conn.as_mut() => {}
                    // (the worker switched off: its connections end with it)
                    _ = stopped.changed() => {}
                }
            });
        }
    });
    Ok((addr, stop))
}

fn routes(net: Network) -> Router {
    Router::new()
        .route("/worker/v1/health", get(health))
        .route("/worker/v1/pair", post(pair))
        .route("/worker/v1/pair/:id", get(pair_status))
        .route("/worker/v1/unpair", post(unpair))
        .route("/worker/v1/capabilities", get(capabilities))
        .route("/worker/v1/resources", get(resources))
        .route("/worker/v1/pins", post(pins))
        .route("/worker/v1/chat", post(chat).layer(DefaultBodyLimit::max(CHAT_LIMIT)))
        .route("/worker/v1/gateway/*path", any(gateway).layer(DefaultBodyLimit::max(MEDIA_LIMIT)))
        .with_state(net)
}

fn refuse(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "code": code, "message": message.into() } }))).into_response()
}

/// The paired primary calling, if it is one (and this computer is a worker, switched on).
fn primary_of(net: &Network, caller: &Caller) -> Option<Peer> {
    if net.settings().active_role() != Role::Worker {
        return None;
    }
    let fp = caller.fingerprint.as_deref()?;
    net.inner().peers().by_fingerprint(fp, Role::Primary).cloned()
}

/// `GET /worker/v1/health`: what this node is (no capabilities, no paths), and whether the caller is paired with it.
async fn health(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    let id = net.identity().map(|i| i.node_id).unwrap_or_default();
    Json(json!({
        "service": "oaiy-worker",
        "protocol": PROTOCOL,
        "nodeId": id,
        "name": net.display_name(),
        "appVersion": env!("CARGO_PKG_VERSION"),
        "paired": primary_of(&net, &caller).is_some(),
    }))
    .into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairRequest {
    node_id: String,
    name: String,
}

/// `POST /worker/v1/pair`: a primary asks to pair. It waits for the person here, who sees the same code as the
/// primary's screen shows.
async fn pair(State(net): State<Network>, Extension(caller): Extension<Caller>, Json(req): Json<PairRequest>) -> Response {
    if net.settings().active_role() != Role::Worker {
        return refuse(StatusCode::FORBIDDEN, "not_a_worker", "this computer is not taking pairing requests");
    }
    let Some(fp) = caller.fingerprint.clone() else {
        return refuse(StatusCode::UNAUTHORIZED, "no_certificate", "a primary presents its certificate to pair");
    };
    let node_id: String = req.node_id.chars().filter(|c| c.is_ascii_alphanumeric()).take(64).collect();
    let name: String = req.name.chars().filter(|c| !c.is_control()).take(80).collect();
    if node_id.is_empty() {
        return refuse(StatusCode::BAD_REQUEST, "invalid_request", "a node id is needed");
    }
    let Ok(id) = net.identity() else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "this worker has no identity");
    };
    let mut waiting = lock(&net.inner().incoming);
    waiting.retain(|r| r.at.elapsed() < REQUEST_TTL && r.fingerprint != fp);
    if waiting.iter().filter(|r| r.status == Status::Pending).count() >= MAX_WAITING {
        return refuse(StatusCode::TOO_MANY_REQUESTS, "too_many", "too many pairing requests are waiting");
    }
    let request = Incoming {
        id: uuid::Uuid::new_v4().simple().to_string(),
        node_id,
        name: if name.trim().is_empty() { "An OAIY computer".into() } else { name },
        code: pairing_code(&fp, &id.fingerprint),
        fingerprint: fp,
        address: caller.addr.ip().to_string(),
        status: Status::Pending,
        at: Instant::now(),
    };
    log::info!("network: {} ({}) asks to pair; code {}", request.name, request.address, request.code);
    let answer = json!({ "requestId": request.id, "nodeId": id.node_id, "name": net.display_name(), "protocol": PROTOCOL });
    waiting.push(request);
    (StatusCode::ACCEPTED, Json(answer)).into_response()
}

/// `GET /worker/v1/pair/:id`: the request's answer, to the certificate that asked alone.
async fn pair_status(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> Response {
    let waiting = lock(&net.inner().incoming);
    match waiting.iter().find(|r| r.id == id && Some(&r.fingerprint) == caller.fingerprint.as_ref() && r.at.elapsed() < REQUEST_TTL) {
        Some(r) => Json(json!({ "status": r.status })).into_response(),
        None => refuse(StatusCode::NOT_FOUND, "not_found", "no such pairing request (or it has expired)"),
    }
}

/// `POST /worker/v1/unpair`: the primary unpaired this worker: its certificate no longer lets it in here either.
async fn unpair(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    let Some(p) = primary_of(&net, &caller) else {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    };
    let _ = net.inner().peers().remove(&p.node_id);
    log::info!("network: {} unpaired this worker", p.name);
    StatusCode::NO_CONTENT.into_response()
}

/// The approval the person gives here (`/api/network/requests/:id/approve`): the primary is paired.
pub(crate) fn decide(net: &Network, id: &str, approve: bool) -> Result<Option<Peer>, String> {
    let mut waiting = lock(&net.inner().incoming);
    let Some(r) = waiting.iter_mut().find(|r| r.id == id && r.status == Status::Pending && r.at.elapsed() < REQUEST_TTL) else {
        return Err("no such pairing request (or it has expired)".into());
    };
    if !approve {
        r.status = Status::Denied;
        return Ok(None);
    }
    let peer = Peer {
        node_id: r.node_id.clone(),
        name: r.name.clone(),
        role: Role::Primary,
        fingerprint: r.fingerprint.clone(),
        address: r.address.clone(),
        paired_at_ms: now_ms(),
    };
    net.inner().peers().add(peer.clone())?;
    r.status = Status::Approved;
    Ok(Some(peer))
}

/// The pairing requests waiting here, for the Networking page.
pub(crate) fn waiting(net: &Network) -> Vec<Value> {
    let mut waiting = lock(&net.inner().incoming);
    waiting.retain(|r| r.at.elapsed() < REQUEST_TTL);
    waiting
        .iter()
        .filter(|r| r.status == Status::Pending)
        .map(|r| json!({
            "id": r.id,
            "nodeId": r.node_id,
            "name": r.name,
            "address": r.address,
            "code": r.code,
            "fingerprint": super::identity::short(&r.fingerprint),
            "secondsLeft": REQUEST_TTL.saturating_sub(r.at.elapsed()).as_secs(),
        }))
        .collect()
}

/// `GET /worker/v1/capabilities`: this worker's engines as a primary needs them: their discovery (the models, the
/// routes) and resources (the GPUs, what is loaded), with nothing of this computer's files or addresses.
async fn capabilities(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    let id = net.identity().map(|i| i.node_id).unwrap_or_default();
    let node = json!({
        "nodeId": id,
        "name": net.display_name(),
        "appVersion": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "protocol": PROTOCOL,
    });
    let (discovery, engines) = match crate::ai::engine_services::local_engines(&net.inner().ai).await {
        Ok((_, d)) => (sanitized_discovery(d), Value::Null),
        Err(e) => (Value::Null, json!(e)),
    };
    Json(json!({ "node": node, "discovery": discovery, "enginesError": engines, "resources": local_resources(&net).await })).into_response()
}

/// `GET /worker/v1/resources`: the GPUs and what holds them.
async fn resources(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    Json(local_resources(&net).await).into_response()
}

/// A model's name as the catalogue lists it, and nothing that could be read as a path.
pub(crate) fn model_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 200 && !s.contains("..") && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':' | '+'))
}

/// `POST /worker/v1/pins` `{model, kind, pin}`: a model of this computer's kept loaded for the paired primary, or let go
/// as any other again: the Studio's pins (`media.pinned`, a picture model in the picture worker on its GPU; `llm.pinned`,
/// the language model chosen here). By the model's name alone; the Studio refuses one it cannot keep beside the others.
async fn pins(State(net): State<Network>, Extension(caller): Extension<Caller>, Json(body): Json<Value>) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    let model = body.get("model").and_then(Value::as_str).unwrap_or_default();
    if !model_name(model) {
        return refuse(StatusCode::BAD_REQUEST, "bad_model", "model must be a model's name, as the catalogue lists it");
    }
    let kind = body.get("kind").and_then(Value::as_str).unwrap_or("image");
    if !matches!(kind, "image" | "llm") {
        return refuse(StatusCode::BAD_REQUEST, "not_pinnable", "picture models (image) and the language model (llm) are pinned");
    }
    let pin = body.get("pin").and_then(Value::as_bool).unwrap_or(true);
    let Some(ui) = crate::ai::engine_services::local_engines_ui(&net.inner().ai) else {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "engines_unavailable", "this computer's engines are not running");
    };
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(20)).build() {
        Ok(c) => c,
        Err(e) => return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    };
    let path = if pin { "/api/resources/pins" } else { "/api/resources/unpin" };
    let answer = client.post(format!("{}{path}", ui.trim_end_matches('/'))).json(&json!({ "model": model, "kind": kind })).send().await;
    match answer {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let v = r.json::<Value>().await.unwrap_or(Value::Null);
            if status.is_success() {
                Json(sanitized_resources(v)).into_response()
            } else {
                let why = v.get("error").and_then(|e| e.as_str().or_else(|| e.get("message").and_then(Value::as_str))).unwrap_or("the engines refused it").to_string();
                refuse(if status.is_client_error() { status } else { StatusCode::BAD_GATEWAY }, "pin_refused", why)
            }
        }
        Err(e) => refuse(StatusCode::SERVICE_UNAVAILABLE, "engines_unavailable", format!("this computer's engines did not answer: {e}")),
    }
}

/// The engines' resources (`GET /api/resources` of the Studio), what a primary shows: GPUs, models, the queue.
async fn local_resources(net: &Network) -> Value {
    let Some(ui) = crate::ai::engine_services::local_engines_ui(&net.inner().ai) else { return Value::Null };
    let client = reqwest::Client::builder().timeout(Duration::from_secs(4)).build();
    let Ok(client) = client else { return Value::Null };
    let Ok(r) = client.get(format!("{}/api/resources", ui.trim_end_matches('/'))).send().await else { return Value::Null };
    let Ok(v) = r.json::<Value>().await else { return Value::Null };
    sanitized_resources(v)
}

/// A discovery document without what is this computer's own: its loopback addresses and any file's path.
pub(crate) fn sanitized_discovery(mut d: Value) -> Value {
    if let Some(o) = d.as_object_mut() {
        o.remove("base_url");
        o.remove("openai_base_url");
    }
    if let Some(endpoints) = d.get_mut("endpoints").and_then(Value::as_array_mut) {
        for e in endpoints {
            if let Some(o) = e.as_object_mut() {
                o.remove("url");
            }
        }
    }
    if let Some(models) = d.get_mut("models").and_then(Value::as_object_mut) {
        for list in models.values_mut().filter_map(Value::as_array_mut) {
            for m in list {
                if let Some(o) = m.as_object_mut() {
                    o.retain(|k, _| !matches!(k.as_str(), "missing_files" | "path" | "paths" | "files" | "dir"));
                }
            }
        }
    }
    d
}

/// Resources without their event log (it names what moved and why, not what a primary needs).
pub(crate) fn sanitized_resources(mut r: Value) -> Value {
    if let Some(o) = r.as_object_mut() {
        o.remove("events");
    }
    r
}

/// `POST /worker/v1/chat`: an OpenAI chat request to this worker's language model (the one chosen in its Engines),
/// streamed back as it comes.
async fn chat(State(net): State<Network>, Extension(caller): Extension<Caller>, body: axum::body::Bytes) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    let (gateway, discovery) = match crate::ai::engine_services::local_engines(&net.inner().ai).await {
        Ok(g) => g,
        Err(e) => return refuse(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
    };
    let path = discovery
        .get("endpoints")
        .and_then(Value::as_array)
        .and_then(|e| e.iter().find(|e| e.get("name").and_then(Value::as_str) == Some("chat")))
        .and_then(|e| e.get("path").and_then(Value::as_str))
        .unwrap_or("/v1/chat/completions")
        .to_string();
    // (the engine answers with its chosen model, as the desktop's own engine provider does)
    let mut request: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, "invalid_request", format!("the body is not JSON: {e}")),
    };
    if let Some(o) = request.as_object_mut() {
        o.remove("model");
    }
    let client = match reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build() {
        Ok(c) => c,
        Err(e) => return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    };
    let upstream = match client.post(format!("{}{path}", gateway.trim_end_matches('/'))).json(&request).send().await {
        Ok(r) => r,
        Err(e) => return refuse(StatusCode::BAD_GATEWAY, "engine_unreachable", format!("the engine did not answer: {e}")),
    };
    passthrough(upstream)
}

/// `/worker/v1/gateway/<path>`: this worker's engines' media routes and files, as the desktop's own gateway route
/// forwards them (`crate::ai::engine_services`).
async fn gateway(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(path): Path<String>, req: Request) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    match crate::ai::engine_services::local_engines(&net.inner().ai).await {
        Ok((g, d)) => crate::ai::engine_services::forward_path(&g, &d, format!("/{path}"), req).await,
        Err(e) => refuse(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
    }
}

/// An upstream answer passed on as it streams (its status and the headers that describe the body).
pub(crate) fn passthrough(upstream: reqwest::Response) -> Response {
    let mut out = Response::builder().status(upstream.status().as_u16());
    for name in ["content-type", "content-length", "content-disposition", "content-range", "accept-ranges", "cache-control", "x-sample-rate"] {
        if let Some(v) = upstream.headers().get(name).and_then(|v| v.to_str().ok()) {
            out = out.header(name, v);
        }
    }
    out.body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|_| refuse(StatusCode::BAD_GATEWAY, "upstream_error", "the answer could not be passed on"))
}
