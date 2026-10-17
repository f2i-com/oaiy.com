//! The desktop's side of OAIY Network (`/api/network/*` on its own API):
//!
//! - this computer's settings and status, the workers found on the LAN, and pairing with one (the code both screens
//!   show; the worker's person approves) or approving a primary's request (on a worker);
//! - each paired worker's heartbeat: its capabilities asked every few seconds, so its models are listed (and shown
//!   offline when it does not answer);
//! - its models in the engine catalogue (`GET /api/ai/engine/services`, [`remote_services`]) under node-scoped ids,
//!   and the routes that carry their requests and files (`/api/network/nodes/:id/…`), so a flow, the Agent or an app
//!   never sees a worker's address or certificate.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::identity::{pairing_code, short, Identity};
use super::store::{now_ms, Peer, Role, Settings};
use super::tls::{self, Pin};
use super::{lock, worker, Network, PROTOCOL};

/// How often a paired worker is asked how it is, and how long it has to answer.
const HEARTBEAT: Duration = Duration::from_secs(7);
const HEARTBEAT_DEADLINE: Duration = Duration::from_secs(6);
/// How long a pairing in progress waits for the worker's approval.
const PAIRING_TTL: Duration = Duration::from_secs(5 * 60);
const CHAT_LIMIT: usize = 16 << 20;
const MEDIA_LIMIT: usize = 96 << 20;

/// A pairing this primary started, waiting for the worker's person.
#[derive(Clone, Debug)]
pub(crate) struct Outgoing {
    pub id: String,
    pub address: String,
    pub request_id: String,
    pub node_id: String,
    pub name: String,
    pub fingerprint: String,
    pub code: String,
    pub at: Instant,
}

/// What a primary last heard from a paired worker.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeState {
    pub online: bool,
    pub last_seen_ms: u64,
    pub last_error: Option<String>,
    /// Its last capabilities (`/worker/v1/capabilities`): node, discovery, resources.
    pub capabilities: Option<Value>,
}

pub fn router(net: Network) -> Router {
    Router::new()
        .route("/api/network/status", get(status))
        .route("/api/network/settings", put(put_settings))
        .route("/api/network/discover", post(discover))
        .route("/api/network/pairings/start", post(pair_start))
        .route("/api/network/pairings/confirm", post(pair_confirm))
        .route("/api/network/requests", get(requests))
        .route("/api/network/requests/:id/approve", post(approve))
        .route("/api/network/requests/:id/deny", post(deny))
        .route("/api/network/nodes", get(nodes))
        .route("/api/network/nodes/:id", delete(unpair))
        .route("/api/network/nodes/:id/resources", get(node_resources))
        .route("/api/network/nodes/:id/pins", put(pin))
        .route("/api/network/nodes/:id/pins/:pin", delete(unpin))
        .route("/api/network/nodes/:id/v1/chat/completions", post(node_chat).layer(DefaultBodyLimit::max(CHAT_LIMIT)))
        .route("/api/network/nodes/:id/gateway/*path", any(node_gateway).layer(DefaultBodyLimit::max(MEDIA_LIMIT)))
        .with_state(net)
}

fn error(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "code": code, "message": message.into() } }))).into_response()
}

// ---------------------------------------------------------------------------------------------------------------
// Settings and status
// ---------------------------------------------------------------------------------------------------------------

async fn status(State(net): State<Network>) -> Response {
    let mut s = net.status();
    s["waiting"] = json!(worker::waiting(&net).len());
    s["found"] = json!(lock(&net.inner().found).len());
    s["paired"] = json!(net.inner().peers().list().len());
    Json(s).into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SettingsChange {
    enabled: Option<bool>,
    role: Option<Role>,
    name: Option<String>,
    port: Option<u16>,
    bind: Option<String>,
}

/// `PUT /api/network/settings`: switch Network on or off, choose this computer's role, name and (a worker's) port.
async fn put_settings(State(net): State<Network>, Json(change): Json<SettingsChange>) -> Response {
    let mut s: Settings = net.settings();
    if let Some(v) = change.enabled {
        s.enabled = v;
    }
    if let Some(v) = change.role {
        s.role = v;
    }
    if let Some(v) = change.name {
        s.name = v.chars().filter(|c| !c.is_control()).take(80).collect::<String>().trim().to_string();
    }
    if let Some(v) = change.port {
        if v < 1024 {
            return error(StatusCode::BAD_REQUEST, "invalid_request", "a worker's port is 1024 or above");
        }
        s.port = v;
    }
    if let Some(v) = change.bind {
        s.bind = v.trim().to_string();
    }
    match net.set(s).await {
        Ok(()) => status(State(net)).await,
        Err(e) => error(StatusCode::BAD_REQUEST, "network_unavailable", e),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Finding and pairing
// ---------------------------------------------------------------------------------------------------------------

/// `POST /api/network/discover`: the workers advertised on the LAN (browsed continuously while this is a primary).
async fn discover(State(net): State<Network>) -> Response {
    if net.settings().active_role() != Role::Primary {
        return error(StatusCode::CONFLICT, "not_a_primary", "switch Network on as a primary to look for workers");
    }
    let mine = net.identity().map(|i| i.node_id).unwrap_or_default();
    let found: Vec<Value> = {
        let peers = net.inner().peers();
        let mut list: Vec<Value> = lock(&net.inner().found)
            .values()
            .filter(|f| f.node_id != mine)
            .map(|f| {
                let mut v = json!(f);
                v["paired"] = json!(peers.get(&f.node_id).is_some());
                v["compatible"] = json!(f.protocol == PROTOCOL);
                v
            })
            .collect();
        list.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        list
    };
    Json(json!({ "discovery": net.inner().mdns, "found": found })).into_response()
}

/// A worker's address as typed (`host`, `host:port`), resolved to one on the local network.
async fn resolve(address: &str, port: u16) -> Result<std::net::SocketAddr, String> {
    let address = address.trim().trim_start_matches("https://").trim_start_matches("http://").trim_end_matches('/');
    if address.is_empty() || address.contains('/') || address.contains('@') {
        return Err("type the worker's address as its Networking page shows it (an address and a port)".into());
    }
    let with_port = if address.parse::<std::net::SocketAddr>().is_ok() || (address.contains(':') && !address.contains('[') && address.matches(':').count() == 1) {
        address.to_string()
    } else if address.parse::<std::net::IpAddr>().is_ok() && address.contains(':') {
        format!("[{address}]:{port}")
    } else {
        format!("{address}:{port}")
    };
    let resolved: Vec<std::net::SocketAddr> = tokio::time::timeout(Duration::from_secs(5), tokio::net::lookup_host(with_port.clone()))
        .await
        .map_err(|_| format!("{with_port} could not be looked up in time"))?
        .map_err(|e| format!("{with_port} could not be looked up: {e}"))?
        .collect();
    resolved
        .into_iter()
        .find(|a| tls::local_address(a.ip()))
        .ok_or_else(|| format!("{with_port} is not on the local network: OAIY Network pairs with computers on the LAN alone"))
}

fn url(address: &str, path: &str) -> String {
    format!("https://{address}{path}")
}

static CLIENTS: Mutex<Option<HashMap<(String, String), reqwest::Client>>> = Mutex::new(None);

/// The HTTP client to a paired worker: this computer's certificate presented, the worker's pinned.
fn client_for(id: &Identity, worker_fingerprint: &str) -> Result<reqwest::Client, String> {
    let key = (id.fingerprint.clone(), worker_fingerprint.to_string());
    let mut clients = lock(&CLIENTS);
    let map = clients.get_or_insert_with(HashMap::new);
    if let Some(c) = map.get(&key) {
        return Ok(c.clone());
    }
    let c = tls::client(id, Pin::Fingerprint(worker_fingerprint.to_string()))?;
    map.insert(key, c.clone());
    Ok(c)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairStart {
    address: Option<String>,
    node_id: Option<String>,
}

/// `POST /api/network/pairings/start` `{address}` (or `{nodeId}` of a worker found on the LAN): ask the worker to pair.
/// Answers the code to compare with the worker's screen, where its person approves.
async fn pair_start(State(net): State<Network>, Json(req): Json<PairStart>) -> Response {
    if net.settings().active_role() != Role::Primary {
        return error(StatusCode::CONFLICT, "not_a_primary", "switch Network on as a primary to pair with a worker");
    }
    let id = match net.identity() {
        Ok(i) => i,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let typed = match (&req.address, &req.node_id) {
        (Some(a), _) if !a.trim().is_empty() => a.trim().to_string(),
        (_, Some(n)) => match lock(&net.inner().found).get(n).and_then(|f| f.addresses.first().cloned()) {
            Some(a) => a,
            None => return error(StatusCode::NOT_FOUND, "not_found", "that worker is no longer seen on the LAN: type its address"),
        },
        _ => return error(StatusCode::BAD_REQUEST, "invalid_request", "a worker's address (or a found worker) is needed"),
    };
    let address = match resolve(&typed, super::store::DEFAULT_PORT).await {
        Ok(a) => a.to_string(),
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_address", e),
    };
    let seen = Arc::new(Mutex::new(None));
    let client = match tls::client(&id, Pin::Record(seen.clone())) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let body = json!({ "nodeId": id.node_id, "name": net.display_name(), "protocol": PROTOCOL });
    let answer = match client.post(url(&address, "/worker/v1/pair")).json(&body).timeout(Duration::from_secs(10)).send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.unwrap_or(Value::Null),
        Ok(r) => {
            let status = r.status();
            let why = r.json::<Value>().await.ok().and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_owned)).unwrap_or_default();
            return error(StatusCode::BAD_GATEWAY, "pairing_refused", format!("the worker refused (HTTP {status}){}", if why.is_empty() { String::new() } else { format!(": {why}") }));
        }
        Err(e) => return error(StatusCode::BAD_GATEWAY, "unreachable", format!("{address} did not answer as an OAIY worker: {}", plain(&e))),
    };
    let Some(fp) = lock(&seen).clone() else {
        return error(StatusCode::BAD_GATEWAY, "unreachable", "the worker presented no certificate");
    };
    let (Some(request_id), Some(node_id)) = (answer.get("requestId").and_then(Value::as_str), answer.get("nodeId").and_then(Value::as_str)) else {
        return error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker's answer was not a pairing request");
    };
    if node_id == id.node_id {
        return error(StatusCode::BAD_REQUEST, "invalid_address", "that address is this computer");
    }
    let pending = Outgoing {
        id: uuid::Uuid::new_v4().simple().to_string(),
        address,
        request_id: request_id.to_string(),
        node_id: node_id.chars().filter(|c| c.is_ascii_alphanumeric()).take(64).collect(),
        name: answer.get("name").and_then(Value::as_str).unwrap_or("An OAIY worker").chars().filter(|c| !c.is_control()).take(80).collect(),
        code: pairing_code(&id.fingerprint, &fp),
        fingerprint: fp,
        at: Instant::now(),
    };
    let out = json!({
        "pairingId": pending.id,
        "code": pending.code,
        "worker": { "nodeId": pending.node_id, "name": pending.name, "address": pending.address, "fingerprint": short(&pending.fingerprint) },
        "status": "pending",
    });
    let mut outgoing = lock(&net.inner().outgoing);
    outgoing.retain(|o| o.at.elapsed() < PAIRING_TTL && o.node_id != pending.node_id);
    outgoing.push(pending);
    Json(out).into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairConfirm {
    pairing_id: String,
}

/// `POST /api/network/pairings/confirm` `{pairingId}`: has the worker's person approved? Approved, the worker is
/// paired here too (its certificate pinned) and its models are asked for at once.
async fn pair_confirm(State(net): State<Network>, Json(req): Json<PairConfirm>) -> Response {
    let Some(p) = lock(&net.inner().outgoing).iter().find(|o| o.id == req.pairing_id && o.at.elapsed() < PAIRING_TTL).cloned() else {
        return error(StatusCode::NOT_FOUND, "not_found", "no such pairing (or it has expired): start again");
    };
    let id = match net.identity() {
        Ok(i) => i,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let client = match client_for(&id, &p.fingerprint) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let answer = match client.get(url(&p.address, &format!("/worker/v1/pair/{}", p.request_id))).timeout(Duration::from_secs(8)).send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.unwrap_or(Value::Null),
        Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
            lock(&net.inner().outgoing).retain(|o| o.id != p.id);
            return Json(json!({ "status": "expired" })).into_response();
        }
        Ok(r) => return error(StatusCode::BAD_GATEWAY, "pairing_refused", format!("the worker answered HTTP {}", r.status())),
        Err(e) => return error(StatusCode::BAD_GATEWAY, "unreachable", format!("the worker did not answer: {}", plain(&e))),
    };
    match answer.get("status").and_then(Value::as_str) {
        Some("approved") => {
            let peer = Peer { node_id: p.node_id.clone(), name: p.name.clone(), role: Role::Worker, fingerprint: p.fingerprint.clone(), address: p.address.clone(), paired_at_ms: now_ms() };
            if let Err(e) = net.inner().peers().add(peer.clone()) {
                return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
            }
            lock(&net.inner().outgoing).retain(|o| o.id != p.id);
            check(&net, &peer).await;
            Json(json!({ "status": "approved", "node": node_json(&net, &peer) })).into_response()
        }
        Some("denied") => {
            lock(&net.inner().outgoing).retain(|o| o.id != p.id);
            Json(json!({ "status": "denied" })).into_response()
        }
        _ => Json(json!({ "status": "pending", "code": p.code })).into_response(),
    }
}

/// `GET /api/network/requests`: primaries asking this worker to pair, each with the code its screen shows.
async fn requests(State(net): State<Network>) -> Response {
    Json(json!({ "requests": worker::waiting(&net) })).into_response()
}

async fn approve(State(net): State<Network>, Path(id): Path<String>) -> Response {
    match worker::decide(&net, &id, true) {
        Ok(Some(peer)) => {
            log::info!("network: paired with {} (a primary)", peer.name);
            Json(json!({ "status": "approved", "node": node_json(&net, &peer) })).into_response()
        }
        Ok(None) => Json(json!({ "status": "denied" })).into_response(),
        Err(e) => error(StatusCode::NOT_FOUND, "not_found", e),
    }
}

async fn deny(State(net): State<Network>, Path(id): Path<String>) -> Response {
    match worker::decide(&net, &id, false) {
        Ok(_) => Json(json!({ "status": "denied" })).into_response(),
        Err(e) => error(StatusCode::NOT_FOUND, "not_found", e),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Paired nodes
// ---------------------------------------------------------------------------------------------------------------

/// A paired computer as the Networking page shows it: who, where, whether it answers, its GPUs and models.
fn node_json(net: &Network, p: &Peer) -> Value {
    let state = lock(&net.inner().nodes).get(&p.node_id).cloned().unwrap_or_default();
    let caps = state.capabilities.clone().unwrap_or(Value::Null);
    let resources = caps.get("resources").cloned().unwrap_or(Value::Null);
    let models: Vec<Value> = if caps.get("discovery").is_some_and(|d| !d.is_null()) {
        let pins = resources.get("pins").and_then(Value::as_array).cloned().unwrap_or_default();
        crate::ai::engine_services::services_from_discovery(&caps["discovery"])
            .into_iter()
            .map(|e| {
                let model = e["model"].as_str().unwrap_or("").to_string();
                let held = resources.get("models").and_then(Value::as_array).and_then(|l| l.iter().find(|m| m["model"].as_str() == Some(model.as_str())));
                // (a picture model's pin, loaded or not yet: `media.pinned`; the language model's, `llm.pinned`)
                let pin = pins.iter().find(|p| p["model"].as_str() == Some(model.as_str()));
                let pinned = pin.is_some() || held.is_some_and(|m| m["pinned"] == json!(true));
                json!({
                    "id": scoped_id(e["kind"].as_str().unwrap_or(""), &p.node_id, &model),
                    "kind": e["kind"],
                    "model": model,
                    "state": held.and_then(|m| m.get("state").cloned()).unwrap_or(json!("not loaded")),
                    "device": held.and_then(|m| m.get("device").cloned()).unwrap_or(Value::Null),
                    "pinned": pinned,
                    "pinState": pin.and_then(|p| p.get("state").cloned()).unwrap_or(if pinned { json!("ready") } else { Value::Null }),
                    "pinError": pin.and_then(|p| p.get("error").cloned()).unwrap_or(Value::Null),
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    json!({
        "nodeId": p.node_id,
        "name": p.name,
        "role": p.role,
        "address": p.address,
        "fingerprint": short(&p.fingerprint),
        "pairedAtMs": p.paired_at_ms,
        "online": state.online,
        "lastSeenMs": if state.last_seen_ms == 0 { Value::Null } else { json!(state.last_seen_ms) },
        "lastError": state.last_error,
        "node": caps.get("node").cloned().unwrap_or(Value::Null),
        "enginesError": caps.get("enginesError").cloned().unwrap_or(Value::Null),
        "devices": resources.get("devices").cloned().unwrap_or(json!([])),
        "queue": resources.get("queue").cloned().unwrap_or(Value::Null),
        "models": models,
    })
}

/// `GET /api/network/nodes`: the paired computers (a primary's workers, a worker's primary).
async fn nodes(State(net): State<Network>) -> Response {
    let peers: Vec<Peer> = net.inner().peers().list().to_vec();
    Json(json!({ "nodes": peers.iter().map(|p| node_json(&net, p)).collect::<Vec<_>>() })).into_response()
}

/// `DELETE /api/network/nodes/:id`: unpair. Its certificate no longer lets it in from now on; a worker is told, if it
/// answers, so it stops accepting this computer too.
async fn unpair(State(net): State<Network>, Path(node_id): Path<String>) -> Response {
    let Some(peer) = net.inner().peers().get(&node_id).cloned() else {
        return error(StatusCode::NOT_FOUND, "not_found", "no such paired computer");
    };
    if peer.role == Role::Worker {
        if let Ok(id) = net.identity() {
            if let Ok(c) = client_for(&id, &peer.fingerprint) {
                let _ = c.post(url(&peer.address, "/worker/v1/unpair")).timeout(Duration::from_secs(3)).send().await;
            }
        }
    }
    if let Err(e) = net.inner().peers().remove(&node_id) {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
    }
    lock(&net.inner().nodes).remove(&node_id);
    log::info!("network: unpaired {}", peer.name);
    StatusCode::NO_CONTENT.into_response()
}

/// A paired worker, by node id, and the client that reaches it.
fn worker_of(net: &Network, node_id: &str) -> Result<(Peer, reqwest::Client), Response> {
    if net.settings().active_role() != Role::Primary {
        return Err(error(StatusCode::CONFLICT, "network_off", "OAIY Network is not on as a primary here"));
    }
    let Some(peer) = net.inner().peers().get(node_id).filter(|p| p.role == Role::Worker).cloned() else {
        return Err(error(StatusCode::NOT_FOUND, "node_unknown", format!("no worker {node_id} is paired with this computer")));
    };
    let id = net.identity().map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e))?;
    let client = client_for(&id, &peer.fingerprint).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e))?;
    Ok((peer, client))
}

/// A request's failure in the words a person reads: its innermost cause (connection refused, the certificate that is
/// not the paired one), or that it took too long, without the URL reqwest puts first.
fn plain(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        return "it took too long to answer".into();
    }
    let mut cause: &dyn std::error::Error = e;
    while let Some(next) = cause.source() {
        cause = next;
    }
    let text = cause.to_string();
    if e.is_connect() && text.to_ascii_lowercase().contains("refused") {
        return "connection refused (is OAIY running there, with Network on as a worker?)".into();
    }
    text
}

/// A worker that did not answer: marked offline, and the caller told which, plainly.
fn offline(net: &Network, peer: &Peer, e: &reqwest::Error) -> Response {
    let e = plain(e);
    if let Some(s) = lock(&net.inner().nodes).get_mut(&peer.node_id) {
        s.online = false;
        s.last_error = Some(e.to_string());
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "node_offline", format!("{} (an OAIY worker) did not answer: {e}", peer.name))
}

async fn node_resources(State(net): State<Network>, Path(node_id): Path<String>) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    match client.get(url(&peer.address, "/worker/v1/resources")).timeout(HEARTBEAT_DEADLINE).send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

/// `PUT /api/network/nodes/:id/pins` `{model, kind}`: a model of the worker's kept loaded there (a picture model, or the
/// language model chosen there), by its name on the worker (or its id here, `engine:<kind>:node.<id>/<model>`): loaded
/// as the worker can, kept whatever comes, and loaded again as it restarts. The worker's engines refuse one they cannot
/// keep beside its other pins; nothing is downloaded or moved.
async fn pin(State(net): State<Network>, Path(node_id): Path<String>, Json(body): Json<Value>) -> Response {
    let asked = body.get("model").or_else(|| body.get("id")).and_then(Value::as_str).unwrap_or_default();
    let (kind, model) = match asked.strip_prefix("engine:").and_then(|rest| rest.split_once(':')) {
        Some((kind, rest)) => match rest.strip_prefix(&format!("node.{node_id}/")) {
            Some(model) => (kind.to_string(), model.to_string()),
            None => return error(StatusCode::BAD_REQUEST, "bad_model", format!("{asked} is not a model of this worker")),
        },
        None => (body.get("kind").and_then(Value::as_str).unwrap_or("image").to_string(), asked.to_string()),
    };
    pin_change(&net, &node_id, &model, &kind, true).await
}

/// `DELETE /api/network/nodes/:id/pins/:pin`: the pinned model (its name on the worker) let go as any other again.
async fn unpin(State(net): State<Network>, Path((node_id, model)): Path<(String, String)>) -> Response {
    // (its kind as the worker lists it: the language model's pin is its own)
    let kind = lock(&net.inner().nodes)
        .get(&node_id)
        .and_then(|s| s.capabilities.as_ref())
        .and_then(|c| c.pointer("/resources/models").and_then(Value::as_array).cloned())
        .and_then(|models| models.iter().find(|m| m.get("model").and_then(Value::as_str) == Some(model.as_str())).and_then(|m| m.get("kind").and_then(Value::as_str).map(str::to_owned)))
        .map(|k| if k == "language" { "llm".to_string() } else { "image".to_string() })
        .unwrap_or_else(|| "image".to_string());
    pin_change(&net, &node_id, &model, &kind, false).await
}

async fn pin_change(net: &Network, node_id: &str, model: &str, kind: &str, pin: bool) -> Response {
    if !worker::model_name(model) {
        return error(StatusCode::BAD_REQUEST, "bad_model", "model must be a model's name, as the catalogue lists it");
    }
    let (peer, client) = match worker_of(net, node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    let body = json!({ "model": model, "kind": kind, "pin": pin });
    match client.post(url(&peer.address, "/worker/v1/pins")).json(&body).timeout(Duration::from_secs(30)).send().await {
        Ok(r) => {
            let ok = r.status().is_success();
            let answer = worker::passthrough(r);
            if ok {
                // (its models' state seen at once, not at the next heartbeat)
                check(net, &peer).await;
            }
            answer
        }
        Err(e) => offline(net, &peer, &e),
    }
}

/// `POST /api/network/nodes/:id/v1/chat/completions`: an OpenAI chat request for the worker's language model, its
/// answer streamed back as the worker streams it (a cancelled request closes the worker's stream with it).
async fn node_chat(State(net): State<Network>, Path(node_id): Path<String>, body: axum::body::Bytes) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    match client.post(url(&peer.address, "/worker/v1/chat")).header("content-type", "application/json").body(body).send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

/// `/api/network/nodes/:id/gateway/<path>`: the worker's engines' media routes (a picture, a song, a video's job)
/// and the files they made, as this desktop's own gateway route serves its own.
async fn node_gateway(State(net): State<Network>, Path((node_id, path)): Path<(String, String)>, req: Request) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    let method = req.method().as_str().to_ascii_uppercase();
    let Ok(m) = reqwest::Method::from_bytes(method.as_bytes()) else {
        return error(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", method);
    };
    if !matches!(method.as_str(), "GET" | "POST" | "DELETE") {
        return error(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", format!("{method} is not forwarded"));
    }
    // (a path that stays below the worker's gateway: `../pins` would be another of its routes)
    if !crate::ai::engine_services::plain_path(&path) {
        return error(StatusCode::BAD_REQUEST, "bad_path", "not one of the worker's media routes");
    }
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let headers = req.headers().clone();
    let body = match axum::body::to_bytes(req.into_body(), MEDIA_LIMIT).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, "too_large", e.to_string()),
    };
    let mut call = client.request(m, url(&peer.address, &format!("/worker/v1/gateway/{path}{query}")));
    for name in ["content-type", "range"] {
        if let Some(v) = headers.get(name).and_then(|v| v.to_str().ok()) {
            call = call.header(name, v);
        }
    }
    if !body.is_empty() {
        call = call.body(body);
    }
    match call.send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Heartbeat and the catalogue
// ---------------------------------------------------------------------------------------------------------------

/// Ask each paired worker how it is, every [`HEARTBEAT`], until the returned sender says stop.
pub(crate) fn heartbeat(net: Network) -> tokio::sync::watch::Sender<bool> {
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        loop {
            let workers: Vec<Peer> = net.inner().peers().list().iter().filter(|p| p.role == Role::Worker).cloned().collect();
            futures_util::future::join_all(workers.iter().map(|p| check(&net, p))).await;
            {
                let paired: Vec<String> = workers.iter().map(|p| p.node_id.clone()).collect();
                lock(&net.inner().nodes).retain(|id, _| paired.contains(id));
            }
            tokio::select! {
                _ = tokio::time::sleep(HEARTBEAT) => {}
                _ = stopped.changed() => break,
            }
        }
    });
    stop
}

/// One worker's capabilities, asked now: at its address, else where the LAN last saw it (it moved).
pub(crate) async fn check(net: &Network, peer: &Peer) {
    let Ok(id) = net.identity() else { return };
    let Ok(client) = client_for(&id, &peer.fingerprint) else { return };
    let mut addresses = vec![peer.address.clone()];
    if let Some(f) = lock(&net.inner().found).get(&peer.node_id) {
        addresses.extend(f.addresses.iter().filter(|a| **a != peer.address).cloned());
    }
    let mut last_error = None;
    for address in addresses {
        match client.get(url(&address, "/worker/v1/capabilities")).timeout(HEARTBEAT_DEADLINE).send().await {
            Ok(r) if r.status().is_success() => {
                let caps = r.json::<Value>().await.unwrap_or(Value::Null);
                if address != peer.address {
                    let _ = net.inner().peers().moved(&peer.node_id, &address);
                }
                let mut nodes = lock(&net.inner().nodes);
                let s = nodes.entry(peer.node_id.clone()).or_default();
                *s = NodeState { online: true, last_seen_ms: now_ms(), last_error: None, capabilities: Some(caps) };
                return;
            }
            Ok(r) if r.status() == reqwest::StatusCode::FORBIDDEN => last_error = Some("the worker no longer accepts this computer (unpaired there)".to_string()),
            Ok(r) => last_error = Some(format!("the worker answered HTTP {}", r.status())),
            Err(e) => last_error = Some(plain(&e)),
        }
    }
    let mut nodes = lock(&net.inner().nodes);
    let s = nodes.entry(peer.node_id.clone()).or_default();
    s.online = false;
    s.last_error = last_error;
}

/// A remote model's id: never the same as a local one, and naming its node (`engine:image:node.<id>/<model>`).
pub fn scoped_id(kind: &str, node_id: &str, model: &str) -> String {
    format!("engine:{kind}:node.{node_id}/{model}")
}

/// The paired workers' models as engine services (what `GET /api/ai/engine/services` adds to this desktop's own):
/// node-scoped ids, never a default, their requests and files through this desktop's `/api/network/nodes/:id/…`
/// routes, and whether the node answers now (`ready`). A worker that stopped answering keeps its models listed,
/// not ready, so a flow that names one says so rather than running another.
pub fn remote_services() -> Vec<Value> {
    let Some(net) = super::shared() else { return Vec::new() };
    remote_services_of(net)
}

pub(crate) fn remote_services_of(net: &Network) -> Vec<Value> {
    if net.settings().active_role() != Role::Primary {
        return Vec::new();
    }
    let workers: Vec<Peer> = net.inner().peers().list().iter().filter(|p| p.role == Role::Worker).cloned().collect();
    let nodes = lock(&net.inner().nodes).clone();
    let mut out = Vec::new();
    for p in workers {
        let Some(state) = nodes.get(&p.node_id) else { continue };
        let Some(discovery) = state.capabilities.as_ref().and_then(|c| c.get("discovery")).filter(|d| !d.is_null()) else { continue };
        let resources = state.capabilities.as_ref().and_then(|c| c.get("resources")).cloned().unwrap_or(Value::Null);
        for e in crate::ai::engine_services::services_from_discovery(discovery) {
            let mut entry = remote_entry(e, &p, state.online);
            // (pinned there: kept loaded on the worker, `PUT /api/network/nodes/:id/pins`)
            let model = entry["model"].as_str().unwrap_or("").to_string();
            let pinned = resources.get("pins").and_then(Value::as_array).is_some_and(|l| l.iter().any(|x| x["model"].as_str() == Some(model.as_str())))
                || resources.get("models").and_then(Value::as_array).is_some_and(|l| l.iter().any(|x| x["model"].as_str() == Some(model.as_str()) && x["pinned"] == json!(true)));
            entry["pinned"] = json!(pinned);
            out.push(entry);
        }
    }
    out
}

fn remote_entry(mut e: Value, p: &Peer, ready: bool) -> Value {
    let kind = e["kind"].as_str().unwrap_or("").to_string();
    let model = e["model"].as_str().unwrap_or("").to_string();
    let gateway = format!("/api/network/nodes/{}/gateway", p.node_id);
    let chat = format!("/api/network/nodes/{}/v1/chat/completions", p.node_id);
    let rewrite = |s: &str| -> String {
        if let Some(rest) = s.strip_prefix(crate::ai::engine_services::GATEWAY_PREFIX) {
            format!("{gateway}{rest}")
        } else if s.starts_with("/api/ai/providers/") && s.ends_with("/v1/chat/completions") {
            chat.clone()
        } else {
            s.to_string()
        }
    };
    for field in ["endpoint", "filesUrl"] {
        if let Some(s) = e.get(field).and_then(Value::as_str).map(str::to_owned) {
            e[field] = json!(rewrite(&s));
        }
    }
    if let Some(job) = e.get_mut("job").and_then(Value::as_object_mut) {
        for (_, v) in job.iter_mut() {
            if let Some(s) = v.as_str().map(str::to_owned) {
                *v = json!(rewrite(&s));
            }
        }
    }
    let name = e["name"].as_str().unwrap_or("").to_string();
    e["id"] = json!(scoped_id(&kind, &p.node_id, &model));
    e["default"] = json!(false);
    e["name"] = json!(format!("{name} · {}", p.name));
    e["nodeId"] = json!(p.node_id);
    e["nodeName"] = json!(p.name);
    e["locality"] = json!("network");
    e["ready"] = json!(ready);
    e
}
