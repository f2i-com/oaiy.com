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

use super::identity::{commitment, is_nonce, nonce, pairing_code, short, Identity};
use super::store::{now_ms, Peer, Role, Settings, WorkerDefault};
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
    /// The worker's person approved: this computer's person is to say the codes matched too ([`pair_confirm`]).
    pub approved_there: bool,
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
        .route("/api/network/nodes/:id/flows", get(node_flows))
        .route("/api/network/nodes/:id/flows/:flow/runs", post(node_run_flow).layer(DefaultBodyLimit::max(MEDIA_LIMIT)))
        .route("/api/network/nodes/:id/runs/:run", get(node_run))
        .route("/api/network/nodes/:id/runs/:run/cancel", post(node_cancel_run))
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
    // (computers, not pairings: one paired both ways is one; and how many this one uses, and how many use it)
    {
        let peers = net.inner().peers();
        let count = |role: Option<Role>| peers.list().iter().filter(|p| role.is_none_or(|r| p.role == r)).map(|p| p.node_id.as_str()).collect::<std::collections::HashSet<_>>().len();
        s["paired"] = json!(count(None));
        s["using"] = json!(count(Some(Role::Worker)));
        s["usedBy"] = json!(count(Some(Role::Primary)));
    }
    // (a primary: its workers answering now, and the pairings it is in the middle of, so a page opened again shows
    // the code it was showing)
    let workers: Vec<String> = net.inner().peers().list().iter().filter(|p| p.role == Role::Worker).map(|p| p.node_id.clone()).collect();
    s["online"] = json!(lock(&net.inner().nodes).iter().filter(|(id, n)| n.online && workers.contains(id)).count());
    s["pairings"] = Value::Array(
        lock(&net.inner().outgoing)
            .iter()
            .filter(|o| o.at.elapsed() < PAIRING_TTL)
            .map(|o| json!({
                "pairingId": o.id,
                "code": o.code,
                "worker": { "nodeId": o.node_id, "name": o.name, "address": o.address, "fingerprint": short(&o.fingerprint) },
                "status": if o.approved_there { "approved_there" } else { "pending" },
            }))
            .collect(),
    );
    s["defaults"] = defaults_json(&net);
    s["services"] = services_json(&net);
    Json(s).into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SettingsChange {
    enabled: Option<bool>,
    /// One role, as before a computer could do both (`primary`, `worker`, `both`): `uses` and `lends` set from it.
    role: Option<Role>,
    /// It uses other computers.
    uses: Option<bool>,
    /// It lends itself to the computers that pair with it.
    lends: Option<bool>,
    /// The person lets the Agent set up OAIY Network here, and this computer pairs by itself with another that does.
    agent: Option<bool>,
    /// Lending itself, it lends its model files too.
    share_models: Option<bool>,
    /// Open the window to pair by itself with a computer that asks so, for this many minutes (1 to 10), or close it (0).
    open_pairing_minutes: Option<u64>,
    name: Option<String>,
    port: Option<u16>,
    bind: Option<String>,
    /// Engine defaults on a worker, by kind: a worker's model's id (`engine:<kind>:node.<nodeId>/<model>`), or null for
    /// this computer's own again.
    defaults: Option<std::collections::BTreeMap<String, Option<String>>>,
    /// Services used on a worker in place of this computer's own (`{"oaiy-voice": "<nodeId>"}`), or null for its own.
    services: Option<std::collections::BTreeMap<String, Option<String>>>,
}

/// `PUT /api/network/settings`: switch Network on or off; whether this computer uses other computers, lends itself, or
/// both; its name and (lending) port; and (using others) which of their models are this computer's engine defaults and
/// which of their services it uses.
async fn put_settings(State(net): State<Network>, Json(change): Json<SettingsChange>) -> Response {
    // (all of it checked before any of it is kept: a request refused changes nothing)
    let switches = change.enabled.is_some() || change.role.is_some() || change.uses.is_some() || change.lends.is_some() || change.name.is_some() || change.port.is_some() || change.bind.is_some();
    let mut s: Settings = net.settings();
    if let Some(v) = change.enabled {
        s.enabled = v;
    }
    if let Some(v) = change.role {
        s.role = v;
        s.uses = Some(matches!(v, Role::Primary | Role::Both));
        s.lends = Some(matches!(v, Role::Worker | Role::Both));
    }
    if let Some(v) = change.uses {
        s.uses = Some(v);
    }
    if let Some(v) = change.lends {
        s.lends = Some(v);
    }
    if let Some(v) = change.agent {
        s.agent = v;
    }
    if let Some(v) = change.share_models {
        s.share_models = Some(v);
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
    if let Some(m) = change.open_pairing_minutes {
        if m > 10 {
            return error(StatusCode::BAD_REQUEST, "invalid_request", "the window to pair by itself is open ten minutes at most");
        }
        if m > 0 && !s.lends() {
            return error(StatusCode::CONFLICT, "not_a_worker", "switch on Lend this computer to let a computer pair with it");
        }
    }
    let uses = s.uses();
    if let Some(defaults) = &change.defaults {
        if let Err(e) = set_defaults(&net, defaults, uses, false).await {
            return error(StatusCode::BAD_REQUEST, "invalid_request", e);
        }
    }
    if let Some(services) = &change.services {
        if let Err(e) = set_services(&net, services, uses, false) {
            return error(StatusCode::BAD_REQUEST, "invalid_request", e);
        }
    }
    // (the switches first, then what is chosen of the computers it uses)
    if switches {
        if let Err(e) = net.set(s).await {
            return error(StatusCode::BAD_REQUEST, "network_unavailable", e);
        }
    } else if change.agent.is_some() || change.share_models.is_some() {
        // (whether the Agent may set it up, or model files are lent, alone: kept, nothing running started again)
        if let Err(e) = net.keep(s) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
        }
    }
    if let Some(defaults) = &change.defaults {
        if let Err(e) = set_defaults(&net, defaults, uses, true).await {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
        }
    }
    if let Some(services) = &change.services {
        if let Err(e) = set_services(&net, services, uses, true) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
        }
    }
    if let Some(m) = change.open_pairing_minutes {
        net.open_pairing(m);
        log::info!("network: the window to pair by itself is {}", if m > 0 { format!("open for {m} min") } else { "closed".into() });
    }
    status(State(net)).await
}

// ---------------------------------------------------------------------------------------------------------------
// Finding and pairing
// ---------------------------------------------------------------------------------------------------------------

/// `POST /api/network/discover`: the workers advertised on the LAN (browsed continuously while this is a primary).
async fn discover(State(net): State<Network>) -> Response {
    if !net.settings().uses() {
        return error(StatusCode::CONFLICT, "not_a_primary", "switch on Use other computers to look for them");
    }
    let mine = net.identity().map(|i| i.node_id).unwrap_or_default();
    let found: Vec<Value> = {
        let peers = net.inner().peers();
        let mut list: Vec<Value> = lock(&net.inner().found)
            .values()
            .filter(|f| f.node_id != mine)
            .map(|f| {
                let mut v = json!(f);
                // (paired as one this computer uses: one that only uses this computer can still be used, once paired so)
                v["paired"] = json!(peers.get(&f.node_id, Role::Worker).is_some());
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

pub(super) fn url(address: &str, path: &str) -> String {
    format!("https://{address}{path}")
}

static CLIENTS: Mutex<Option<HashMap<(String, String), reqwest::Client>>> = Mutex::new(None);

/// The HTTP client to a paired worker: this computer's certificate presented, the worker's pinned.
pub(super) fn client_for(id: &Identity, worker_fingerprint: &str) -> Result<reqwest::Client, String> {
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
    /// Where both computers pair by themselves (see [`Settings::agent`]): let the other use this one too.
    #[serde(default)]
    both_ways: Option<bool>,
}

/// `POST /api/network/pairings/start` `{address}` (or `{nodeId}` of a worker found on the LAN): ask the worker to pair.
/// Answers the code to compare with the worker's screen, where its person approves.
async fn pair_start(State(net): State<Network>, Json(req): Json<PairStart>) -> Response {
    if !net.settings().uses() {
        return error(StatusCode::CONFLICT, "not_a_primary", "switch on Use other computers to pair with one");
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
    // (this computer pairs by itself where its person lets its Agent set up OAIY Network: the other approves at once
    // where it does too)
    let auto = net.settings().agent;
    let body = json!({ "nodeId": id.node_id, "name": net.display_name(), "protocol": PROTOCOL, "auto": auto });
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
    if !super::identity::node_id_ok(node_id) {
        return error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker's answer was not a pairing request (its node id is not one OAIY makes)");
    }
    // The code: the worker committed to its random value above; this computer's is told now, to that same certificate,
    // and the worker's told back is held to its commitment (identity::pairing_code).
    let Some(committed) = answer.get("commitment").and_then(Value::as_str).filter(|c| is_nonce(c)).map(str::to_ascii_lowercase) else {
        return error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker pairs another way: update OAIY on both computers to pair them");
    };
    let mine = match nonce() {
        Ok(n) => n,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let same_worker = match client_for(&id, &fp) {
        Ok(c) => c,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
    };
    let told = match same_worker.post(url(&address, &format!("/worker/v1/pair/{request_id}"))).json(&json!({ "nonce": mine })).timeout(Duration::from_secs(10)).send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.unwrap_or(Value::Null),
        Ok(r) => return error(StatusCode::BAD_GATEWAY, "pairing_refused", format!("the worker refused (HTTP {})", r.status())),
        Err(e) => return error(StatusCode::BAD_GATEWAY, "unreachable", format!("{address} did not answer as an OAIY worker: {}", plain(&e))),
    };
    let Some(theirs) = told.get("nonce").and_then(Value::as_str).filter(|n| is_nonce(n)).map(str::to_ascii_lowercase) else {
        return error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker's answer was not a pairing request");
    };
    if commitment(&id.fingerprint, &fp, &theirs) != committed {
        return error(StatusCode::BAD_GATEWAY, "pairing_refused", "the worker's answer does not match what it committed to: something on the network is in the way, so it is not paired");
    }
    let pending = Outgoing {
        id: uuid::Uuid::new_v4().simple().to_string(),
        address,
        request_id: request_id.to_string(),
        node_id: node_id.chars().filter(|c| c.is_ascii_alphanumeric()).take(64).collect(),
        name: answer.get("name").and_then(Value::as_str).unwrap_or("An OAIY worker").chars().filter(|c| !c.is_control()).take(80).collect(),
        code: pairing_code(&id.fingerprint, &fp, &mine, &theirs),
        fingerprint: fp,
        at: Instant::now(),
        approved_there: false,
    };
    let worker_json = json!({ "nodeId": pending.node_id, "name": pending.name, "address": pending.address, "fingerprint": short(&pending.fingerprint) });
    // (approved there at once, both computers pairing by themselves: paired here too, no person asked)
    if auto && told.get("approved") == Some(&json!(true)) {
        let code = pending.code.clone();
        // (both ways only where asked: a computer that paired by itself is not let use this one unasked)
        let answer = paired(&net, &pending, req.both_ways.unwrap_or(false), true).await;
        let (parts, body) = answer.into_parts();
        let Ok(bytes) = axum::body::to_bytes(body, 1 << 22).await else { return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "the pairing's answer") };
        let mut v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        if parts.status.is_success() {
            v["code"] = json!(code);
            v["worker"] = worker_json;
            v["auto"] = json!(true);
        }
        return (parts.status, Json(v)).into_response();
    }
    let out = json!({
        "pairingId": pending.id,
        "code": pending.code,
        "worker": worker_json,
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
    /// This computer's person says the worker showed the same code (true), or that it did not (false).
    #[serde(default)]
    accept: Option<bool>,
    /// And lets the worker use this computer too (this computer lending itself): paired both ways at once.
    #[serde(default)]
    both_ways: bool,
}

/// `POST /api/network/pairings/confirm` `{pairingId, accept?}`: has the worker's person approved? Approved there, it
/// is `approved_there` until this computer's person says the worker showed the same code (`accept: true`): only then
/// is the worker paired here (its certificate pinned) and its models asked for. Both screens are checked, so a
/// computer answering for the worker on the network cannot pair itself by approving its own request.
/// `accept: false` gives up, and tells the worker so: a request its person has not answered yet is withdrawn there,
/// and a worker that approved (perhaps since this computer last asked) forgets this computer.
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
    if req.accept == Some(false) {
        lock(&net.inner().outgoing).retain(|o| o.id != p.id);
        // (a worker paired already, asked again: its pairing stands, and a second yes there changes nothing)
        if !paired_already(&net, &p) {
            let _ = client.post(url(&p.address, "/worker/v1/unpair")).timeout(Duration::from_secs(3)).send().await;
        }
        return Json(json!({ "status": "declined" })).into_response();
    }
    if p.approved_there {
        return if req.accept == Some(true) { paired(&net, &p, req.both_ways, false).await } else { Json(json!({ "status": "approved_there", "code": p.code })).into_response() };
    }
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
        Some("approved") if req.accept == Some(true) => paired(&net, &p, req.both_ways, false).await,
        Some("approved") => {
            // (the person here has five more minutes to say the codes matched)
            if let Some(o) = lock(&net.inner().outgoing).iter_mut().find(|o| o.id == p.id) {
                o.approved_there = true;
                o.at = Instant::now();
            }
            Json(json!({ "status": "approved_there", "code": p.code })).into_response()
        }
        Some("denied") => {
            lock(&net.inner().outgoing).retain(|o| o.id != p.id);
            Json(json!({ "status": "denied" })).into_response()
        }
        _ => Json(json!({ "status": "pending", "code": p.code })).into_response(),
    }
}

/// Whether the worker of pairing `p` is paired here already, with the same certificate.
fn paired_already(net: &Network, p: &Outgoing) -> bool {
    net.inner().peers().get(&p.node_id, Role::Worker).is_some_and(|w| w.fingerprint == p.fingerprint)
}

/// The worker of pairing `p`, approved on both screens: paired here, its certificate pinned, its models asked for.
async fn paired(net: &Network, p: &Outgoing, both_ways: bool, by_itself: bool) -> Response {
    let peer = Peer {
        node_id: p.node_id.clone(),
        name: p.name.clone(),
        role: Role::Worker,
        fingerprint: p.fingerprint.clone(),
        address: p.address.clone(),
        paired_at_ms: now_ms(),
        by_itself,
    };
    // (the guard let go before anything is awaited)
    let added = net.inner().peers().add(peer.clone());
    if let Err(e) = added {
        lock(&net.inner().outgoing).retain(|o| o.id != p.id);
        // (approved there, not paired here: the worker is told, so it does not keep this computer one way)
        if let Ok(c) = net.identity().and_then(|id| client_for(&id, &p.fingerprint)) {
            let _ = c.post(url(&p.address, "/worker/v1/unpair")).timeout(Duration::from_secs(3)).send().await;
        }
        let (status, code) = if e == super::store::TAKEN { (StatusCode::CONFLICT, "node_id_taken") } else { (StatusCode::INTERNAL_SERVER_ERROR, "internal") };
        return error(status, code, e);
    }
    lock(&net.inner().outgoing).retain(|o| o.id != p.id);
    // (both ways: the worker may use this computer too, which lends itself, and is told where it listens)
    if both_ways && net.settings().lends() {
        lend_to(net, &peer).await;
    }
    check(net, &peer).await;
    Json(json!({ "status": "approved", "node": node_json(net, &p.node_id) })).into_response()
}

/// Let worker `peer` use this computer too (paired the other way as well): kept here as a computer that uses this one,
/// and the worker told where this computer listens (`/worker/v1/lends`), so it keeps this one as a computer it uses.
async fn lend_to(net: &Network, peer: &Peer) {
    // (kept here only once the worker can be told, and undone where it was not: no pairing one way half made)
    let (Some(port), Ok(id)) = (net.listening().map(|a| a.port()), net.identity()) else { return };
    let Ok(client) = client_for(&id, &peer.fingerprint) else { return };
    let host = peer.address.rsplit_once(':').map_or(peer.address.as_str(), |(h, _)| h).to_string();
    let user = Peer { role: Role::Primary, address: host, paired_at_ms: now_ms(), ..peer.clone() };
    if let Err(e) = net.inner().peers().add(user) {
        log::warn!("network: {} could not be let use this computer: {e}", peer.name);
        return;
    }
    let told = client.post(url(&peer.address, "/worker/v1/lends")).json(&json!({ "port": port, "name": net.display_name() })).timeout(Duration::from_secs(5)).send().await;
    let why = match told {
        Ok(r) if r.status().is_success() => return log::info!("network: {} may use this computer too", peer.name),
        Ok(r) => format!("HTTP {}", r.status()),
        Err(e) => plain(&e),
    };
    log::warn!("network: {} was not told it may use this computer ({why}): paired one way only", peer.name);
    let _ = net.inner().peers().remove_way(&peer.node_id, Role::Primary);
}

/// `GET /api/network/requests`: primaries asking this worker to pair, each with the code its screen shows.
async fn requests(State(net): State<Network>) -> Response {
    Json(json!({ "requests": worker::waiting(&net) })).into_response()
}

async fn approve(State(net): State<Network>, Path(id): Path<String>) -> Response {
    match worker::decide(&net, &id, true) {
        Ok(Some(peer)) => {
            log::info!("network: paired with {} (a primary)", peer.name);
            Json(json!({ "status": "approved", "node": node_json(&net, &peer.node_id) })).into_response()
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

/// A paired computer as the Networking page shows it: whether this computer uses it (`uses`: it is a worker here, with
/// its GPUs, models, flows and services) and whether it uses this computer (`usesThis`), and `role`, what it is to this
/// computer first (`worker` where this computer uses it, else `primary`); what it has (`machine`) and its queue (`jobs`).
fn node_json(net: &Network, node_id: &str) -> Value {
    let (worker, user) = {
        let peers = net.inner().peers();
        (peers.get(node_id, Role::Worker).cloned(), peers.get(node_id, Role::Primary).cloned())
    };
    let Some(p) = worker.clone().or(user.clone()) else { return Value::Null };
    let state = lock(&net.inner().nodes).get(&p.node_id).cloned().unwrap_or_default();
    let caps = state.capabilities.clone().unwrap_or(Value::Null);
    let resources = caps.get("resources").cloned().unwrap_or(Value::Null);
    let models: Vec<Value> = if caps.get("discovery").is_some_and(|d| !d.is_null()) {
        let pins = resources.get("pins").and_then(Value::as_array).cloned().unwrap_or_default();
        // (the language model's pin is its chosen one's, running or not)
        let llm_pinned = resources.pointer("/policy/llm_pinned") == Some(&json!(true));
        let chosen_llm = caps.pointer("/discovery/defaults/llm").and_then(Value::as_str).map(str::to_owned);
        crate::ai::engine_services::services_from_discovery(&caps["discovery"])
            .into_iter()
            .map(|e| {
                let model = e["model"].as_str().unwrap_or("").to_string();
                let held = resources.get("models").and_then(Value::as_array).and_then(|l| l.iter().find(|m| m["model"].as_str() == Some(model.as_str())));
                // (a picture model's pin, loaded or not yet: `media.pinned`; the language model's, `llm.pinned`)
                let pin = pins.iter().find(|p| p["model"].as_str() == Some(model.as_str()));
                let pinned = pin.is_some()
                    || held.is_some_and(|m| m["pinned"] == json!(true))
                    || (e["kind"] == "llm" && llm_pinned && chosen_llm.as_deref() == Some(model.as_str()));
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
        "uses": worker.is_some(),
        "usesThis": user.is_some(),
        "address": p.address,
        "fingerprint": short(&p.fingerprint),
        "pairedAtMs": p.paired_at_ms,
        "pairedByItself": worker.as_ref().is_some_and(|w| w.by_itself) || user.as_ref().is_some_and(|u| u.by_itself),
        "online": state.online,
        "lastSeenMs": if state.last_seen_ms == 0 { Value::Null } else { json!(state.last_seen_ms) },
        "lastError": state.last_error,
        "node": caps.get("node").cloned().unwrap_or(Value::Null),
        "enginesError": caps.get("enginesError").cloned().unwrap_or(Value::Null),
        "devices": resources.get("devices").cloned().unwrap_or(json!([])),
        // (the flows it shares, run there from here: `…/flows/:flow/runs`; the services it lends, OAIY Voice)
        "flows": caps.get("flows").cloned().filter(Value::is_array).unwrap_or(json!([])),
        "services": caps.get("services").cloned().filter(Value::is_array).unwrap_or(json!([])),
        // (what it has and how busy it is: its processor, memory and GPUs, and its queue of jobs and runs)
        "machine": caps.get("machine").cloned().unwrap_or(Value::Null),
        "jobs": caps.get("jobs").cloned().filter(Value::is_array).unwrap_or(json!([])),
        "queue": resources.get("queue").cloned().unwrap_or(Value::Null),
        "models": models,
    })
}

/// `GET /api/network/nodes`: the paired computers (a primary's workers, a worker's primary).
async fn nodes(State(net): State<Network>) -> Response {
    // (one a computer, paired one way or both)
    let mut ids: Vec<String> = Vec::new();
    for p in net.inner().peers().list() {
        if !ids.contains(&p.node_id) {
            ids.push(p.node_id.clone());
        }
    }
    Json(json!({ "nodes": ids.iter().map(|id| node_json(&net, id)).collect::<Vec<_>>() })).into_response()
}

/// `DELETE /api/network/nodes/:id`: unpair. Its certificate no longer lets it in from now on; a worker is told, if it
/// answers, so it stops accepting this computer too.
async fn unpair(State(net): State<Network>, Path(node_id): Path<String>) -> Response {
    let (as_worker, as_user) = {
        let peers = net.inner().peers();
        (peers.get(&node_id, Role::Worker).cloned(), peers.get(&node_id, Role::Primary).cloned())
    };
    let Some(peer) = as_worker.clone().or(as_user) else {
        return error(StatusCode::NOT_FOUND, "not_found", "no such paired computer");
    };
    // (both ways: a computer this one uses is told, and forgets this one both ways too; one that did not answer is
    // told when it does)
    if let Some(worker) = as_worker {
        if !tell_unpaired(&net, &worker).await {
            lock(&net.inner().untold).push(worker);
        }
    }
    // (its models no longer this computer's defaults, nor its services used here: its own again)
    if let Err(e) = net.forget(&node_id) {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e);
    }
    log::info!("network: unpaired {}", peer.name);
    StatusCode::NO_CONTENT.into_response()
}

/// Tell a worker unpaired here that it is: whether it heard (it answered, or no longer knows this computer).
async fn tell_unpaired(net: &Network, peer: &Peer) -> bool {
    let Some(c) = net.identity().ok().and_then(|id| client_for(&id, &peer.fingerprint).ok()) else { return false };
    matches!(c.post(url(&peer.address, "/worker/v1/unpair")).timeout(Duration::from_secs(3)).send().await, Ok(r) if r.status().is_success() || r.status() == reqwest::StatusCode::FORBIDDEN)
}

/// The workers unpaired here that did not hear of it, told again; one paired again since is not told.
pub(crate) async fn tell_untold(net: &Network) {
    let untold: Vec<Peer> = std::mem::take(&mut *lock(&net.inner().untold));
    for peer in untold {
        let again = net.inner().peers().get(&peer.node_id, Role::Worker).is_some_and(|p| p.fingerprint == peer.fingerprint);
        if !again && !tell_unpaired(net, &peer).await {
            lock(&net.inner().untold).push(peer);
        }
    }
}

/// A paired worker, by node id, and the client that reaches it.
fn worker_of(net: &Network, node_id: &str) -> Result<(Peer, reqwest::Client), Response> {
    if !net.settings().uses() {
        return Err(error(StatusCode::CONFLICT, "network_off", "OAIY Network is not on as a primary here"));
    }
    let Some(peer) = net.inner().peers().get(node_id, Role::Worker).cloned() else {
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
        return "connection refused (is OAIY running there, with Lend this computer on?)".into();
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
    // (its kind as the worker's catalogue lists it, loaded or not: the language model's pin is its own)
    let caps = lock(&net.inner().nodes).get(&node_id).and_then(|s| s.capabilities.clone()).unwrap_or(Value::Null);
    let listed = caps.get("discovery").filter(|d| !d.is_null()).map(crate::ai::engine_services::services_from_discovery).unwrap_or_default();
    let kind = listed
        .iter()
        .find(|e| e["model"].as_str() == Some(model.as_str()))
        .and_then(|e| e["kind"].as_str())
        .map(|k| if k == "llm" { "llm" } else { "image" })
        .unwrap_or("image");
    pin_change(&net, &node_id, &model, kind, false).await
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
    chat_on(&net, &node_id, body).await
}

pub(crate) async fn chat_on(net: &Network, node_id: &str, body: axum::body::Bytes) -> Response {
    let (peer, client) = match worker_of(net, node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    match client.post(url(&peer.address, "/worker/v1/chat")).header("content-type", "application/json").body(body).send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(net, &peer, &e),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Engine defaults on a worker
// ---------------------------------------------------------------------------------------------------------------

/// This computer's own default model of `kind` in a catalogue (its engine's entry marked default).
fn local_default(services: &[Value], kind: &str) -> Option<String> {
    services
        .iter()
        .find(|s| s["kind"] == kind && s.get("nodeId").is_none() && s["default"] == json!(true))
        .and_then(|s| s["model"].as_str())
        .map(str::to_owned)
}

/// The catalogue entry of `kind`'s default on a worker, where one is chosen and in effect: a primary's, its worker
/// paired and its models known here (a worker not heard from since this computer started leaves this computer's own
/// default in place until it is; one heard from and offline now is the default still, and says so when used). A
/// language model is the worker's chosen one, whichever that is now. `local` is this computer's own default of the kind
/// now, where known: one chosen since the worker's was (in Engines) is this computer's own again.
fn in_effect(net: &Network, kind: &str, remote: &[Value], local: Option<&str>) -> Option<Value> {
    if !net.settings().uses() {
        return None;
    }
    let mut chosen = net.inner().defaults().get(kind).cloned()?;
    if let Some(now) = local {
        // (this computer's own not known when the worker's was chosen, its engines not answering: it is now)
        if chosen.instead_of.is_none() {
            chosen.instead_of = Some(now.to_string());
            let _ = net.inner().defaults().set(kind, Some(chosen.clone()));
        }
        if Some(now) != chosen.instead_of.as_deref() {
            log::info!("network: {kind}'s default chosen here again ({now}), in place of {}", chosen.id);
            let _ = net.inner().defaults().set(kind, None);
            return None;
        }
    }
    remote
        .iter()
        .find(|e| e["kind"] == kind && if kind == "llm" { e["nodeId"].as_str() == Some(chosen.node_id()) } else { e["id"].as_str() == Some(chosen.id.as_str()) })
        .cloned()
}

/// The engine catalogue with the defaults on workers in effect: each such model marked its kind's default, and this
/// computer's own of that kind not (what `GET /api/ai/engine/services` answers).
pub fn apply_defaults(services: &mut [Value]) {
    if let Some(net) = super::shared() {
        apply_defaults_of(net, services);
    }
}

pub(crate) fn apply_defaults_of(net: &Network, services: &mut [Value]) {
    let kinds: Vec<String> = net.inner().defaults().all().keys().cloned().collect();
    for kind in kinds {
        let local = local_default(services, &kind);
        let remote: Vec<Value> = services.iter().filter(|s| s.get("nodeId").is_some()).cloned().collect();
        let Some(entry) = in_effect(net, &kind, &remote, local.as_deref()) else { continue };
        for s in services.iter_mut().filter(|s| s["kind"] == kind.as_str()) {
            s["default"] = json!(s["id"] == entry["id"]);
        }
    }
}

/// The worker whose language model is this computer's default now (`local`: this computer's own, where known): the
/// engine's chats go there ([`chat_on_worker`]).
pub fn default_chat_entry(local: Option<&str>) -> Option<Value> {
    default_chat_entry_of(super::shared()?, local)
}

pub(crate) fn default_chat_entry_of(net: &Network, local: Option<&str>) -> Option<Value> {
    in_effect(net, "llm", &remote_services_of(net), local)
}

/// A chat for this computer's engine, answered by the worker whose language model is the default (its chosen model,
/// whatever the caller named), streamed as the worker streams it.
pub async fn chat_on_worker(node_id: &str, mut body: Value) -> Response {
    let Some(net) = super::shared() else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "network_off", "OAIY Network is not on as a primary here");
    };
    if let Some(o) = body.as_object_mut() {
        o.remove("model");
        o.remove("provider");
    }
    chat_on(net, node_id, axum::body::Bytes::from(body.to_string())).await
}

// ---------------------------------------------------------------------------------------------------------------
// Services on a worker (OAIY Voice), used in place of this computer's own
// ---------------------------------------------------------------------------------------------------------------

/// A service on a worker, as this computer's speech reaches it: the client that reaches the worker (its certificate
/// pinned), the worker's root (`https://<address>`), the service's routes there and the worker's name.
#[derive(Clone)]
pub struct ServiceTarget {
    pub client: reqwest::Client,
    pub root: String,
    pub base: String,
    pub node_name: String,
    /// The worker's certificate: what the copies of voices kept there are kept for.
    pub fingerprint: String,
}

/// Where `service` runs for this computer, where a paired worker's is chosen in its place (a primary's, Network on):
/// calls' speech and the transcription route use it as they would this computer's own.
pub fn service_target(service: &str) -> Option<ServiceTarget> {
    service_target_of(super::shared()?, service)
}

pub(crate) fn service_target_of(net: &Network, service: &str) -> Option<ServiceTarget> {
    if !net.settings().uses() {
        return None;
    }
    let node = net.inner().services().get(service)?.to_string();
    let peer = net.inner().peers().get(&node, Role::Worker).cloned()?;
    let client = client_for(&net.identity().ok()?, &peer.fingerprint).ok()?;
    let root = url(&peer.address, "");
    Some(ServiceTarget { base: format!("{root}/worker/v1/services/{service}"), root, client, node_name: peer.name, fingerprint: peer.fingerprint })
}

/// The services used on workers, as the Services and Networking pages show them: by service, the worker.
fn services_json(net: &Network) -> Value {
    // (in effect only while this computer uses others: kept, and used again, for when it does)
    if !net.settings().uses() {
        return json!({});
    }
    let peers = net.inner().peers().list().to_vec();
    let chosen = net.inner().services().all().clone();
    Value::Object(
        chosen
            .into_iter()
            .filter(|(_, node)| peers.iter().any(|p| p.node_id == *node && p.role == Role::Worker))
            .map(|(service, node)| {
                let name = peers.iter().find(|p| p.node_id == node).map(|p| p.name.clone()).unwrap_or_default();
                (service, json!({ "nodeId": node, "nodeName": name }))
            })
            .collect(),
    )
}

/// Use (a node id) or let go (null) a worker's service in place of this computer's own. A service one of its paired
/// workers lends and has added.
/// Check (`keep` false) or keep the services chosen on workers; `uses`: whether this computer uses others then.
fn set_services(net: &Network, change: &std::collections::BTreeMap<String, Option<String>>, uses: bool, keep: bool) -> Result<(), String> {
    for (service, node) in change {
        let Some(node) = node.as_deref().map(str::trim).filter(|n| !n.is_empty()) else {
            if keep {
                net.inner().services().set(service, None)?;
            }
            continue;
        };
        if !uses {
            return Err("switch Network on as a primary to use a worker's services".into());
        }
        if !worker::SHARED_SERVICES.iter().any(|(id, _)| id == service) {
            return Err(format!("{service} is not a service a worker lends"));
        }
        let Some(peer) = net.inner().peers().get(node, Role::Worker).cloned() else {
            return Err(format!("{node} is not a worker paired with this computer"));
        };
        let listed = lock(&net.inner().nodes)
            .get(node)
            .and_then(|s| s.capabilities.as_ref())
            .and_then(|c| c.get("services").and_then(Value::as_array).cloned())
            .is_some_and(|l| l.iter().any(|x| x["id"] == service.as_str() && x["installed"] == json!(true)));
        if !listed {
            return Err(format!("{} has no {service} set up (add it on its Services page)", peer.name));
        }
        if keep {
            net.inner().services().set(service, Some(node))?;
            log::info!("network: {service} on {} in place of this computer's", peer.name);
        }
    }
    Ok(())
}

/// The defaults on workers as the Networking and Engines pages show them: by kind, its id, model and worker.
fn defaults_json(net: &Network) -> Value {
    if !net.settings().uses() {
        return json!({});
    }
    let peers = net.inner().peers().list().to_vec();
    let defaults = net.inner().defaults().all().clone();
    Value::Object(
        defaults
            .into_iter()
            .filter(|(_, d)| peers.iter().any(|p| p.node_id == d.node_id() && p.role == Role::Worker))
            .map(|(kind, d)| {
                let node = d.node_id().to_string();
                let name = peers.iter().find(|p| p.node_id == node).map(|p| p.name.clone()).unwrap_or_default();
                let model = d.id.rsplit_once('/').map_or("", |(_, m)| m).to_string();
                (kind, json!({ "id": d.id, "nodeId": node, "nodeName": name, "model": model, "insteadOf": d.instead_of }))
            })
            .collect(),
    )
}

/// Choose (an id) or let go (null) the defaults on workers, by kind. A model is one of a paired worker's, of that kind,
/// as this computer's catalogue lists it; this computer's own default then is kept beside it ([`in_effect`]).
/// Check (`keep` false) or keep the defaults chosen on workers; `uses`: whether this computer uses others then.
async fn set_defaults(net: &Network, change: &std::collections::BTreeMap<String, Option<String>>, uses: bool, keep: bool) -> Result<(), String> {
    let local = if keep { crate::ai::engine_services::local_catalogue(&net.inner().ai).await } else { Vec::new() };
    for (kind, id) in change {
        let Some(id) = id.as_deref().map(str::trim).filter(|i| !i.is_empty()) else {
            if keep {
                net.inner().defaults().set(kind, None)?;
            }
            continue;
        };
        if !uses {
            return Err("switch Network on as a primary to use a worker's model as a default".into());
        }
        let Some(rest) = id.strip_prefix(&format!("engine:{kind}:node.")) else {
            return Err(format!("{id} is not a worker's {kind} model (engine:{kind}:node.<nodeId>/<model>)"));
        };
        let node = rest.split_once('/').map_or("", |(n, _)| n);
        let Some(peer) = net.inner().peers().get(node, Role::Worker).cloned() else {
            return Err(format!("{node} is not a worker paired with this computer"));
        };
        if !remote_models_of(net, true).iter().any(|e| e["id"] == id && e["kind"] == kind.as_str()) {
            return Err(format!("{id} is not one of {}'s models now", peer.name));
        }
        if keep {
            net.inner().defaults().set(kind, Some(WorkerDefault { id: id.to_string(), instead_of: local_default(&local, kind) }))?;
            log::info!("network: {kind}'s default is {id} on {}", peer.name);
        }
    }
    Ok(())
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
    // (and the URL keeps it: one its parser would fold is refused, not sent somewhere else)
    let Some(target) = crate::ai::engine_services::exact_url(&url(&peer.address, ""), &format!("/worker/v1/gateway/{path}"), &query) else {
        return error(StatusCode::BAD_REQUEST, "bad_path", "not one of the worker's media routes");
    };
    let mut call = client.request(m, target);
    for name in ["content-type", "range"] {
        if let Some(v) = headers.get(name).and_then(|v| v.to_str().ok()) {
            call = call.header(name, v);
        }
    }
    // (the worker names its files at this desktop's route for it, as its caller reached it, not at its own loopback)
    if let Some(base) = base_for(&headers, &node_id) {
        call = call.header("x-oaiy-base", base);
    }
    if !body.is_empty() {
        call = call.body(body);
    }
    match call.send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

/// This desktop's route for a worker's gateway as its caller reached this desktop (its `Host`), where a worker names
/// the files it made: `http://<host>/api/network/nodes/<id>/gateway`.
fn base_for(headers: &axum::http::HeaderMap, node_id: &str) -> Option<String> {
    // (a caller in this process, the Agent's tools, sends no Host: this desktop's own address)
    let own = format!("127.0.0.1:{}", crate::api_port());
    let host = match headers.get("host").and_then(|v| v.to_str().ok()) {
        Some(h) => h,
        None => own.as_str(),
    };
    if host.is_empty() || host.len() > 255 || !host.chars().all(|c| c.is_ascii_alphanumeric() || ".-:[]".contains(c)) {
        return None;
    }
    Some(format!("http://{host}/api/network/nodes/{node_id}/gateway"))
}

// ---------------------------------------------------------------------------------------------------------------
// A worker's flows: the ones it shares, run there
// ---------------------------------------------------------------------------------------------------------------

/// A flow's id as the flows folder keeps it.
fn flow_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 128 && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

/// `GET /api/network/nodes/:id/flows`: the flows the worker shares (`Let apps run it` there), with their inputs.
async fn node_flows(State(net): State<Network>, Path(node_id): Path<String>) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    match client.get(url(&peer.address, "/worker/v1/flows")).timeout(Duration::from_secs(15)).send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunFlow {
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    /// Answer once it has finished (or `timeoutMs` has passed: then as it is, 202), not at once.
    #[serde(default)]
    wait: bool,
    #[serde(default)]
    key: Option<String>,
}

fn terminal(status: &str) -> bool {
    matches!(status, "succeeded" | "failed" | "timed_out" | "cancelled")
}

/// `POST /api/network/nodes/:id/flows/:flow/runs` `{input?, timeoutMs?, wait?, key?}`: one of the worker's shared
/// flows, run there by its own runner and engines. This computer's pictures in `input` are sent inline (the worker
/// cannot read this computer's addresses); the pictures the run makes are named at this computer's route for the
/// worker. Answered at once with the run (follow it at `…/runs/:run`), or with `wait` once it has finished.
async fn node_run_flow(State(net): State<Network>, Path((node_id, flow)): Path<(String, String)>, headers: axum::http::HeaderMap, Json(asked): Json<RunFlow>) -> Response {
    if !flow_id(&flow) {
        return error(StatusCode::BAD_REQUEST, "invalid_request", "a flow's id is letters, digits, - and _");
    }
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    let input = match asked.input {
        Some(i) => tokio::task::spawn_blocking(move || crate::bridge::pictures::inline_local(i)).await.ok(),
        None => None,
    };
    let base = base_for(&headers, &node_id);
    let with_base = |call: reqwest::RequestBuilder| match &base {
        Some(b) => call.header("x-oaiy-base", b),
        None => call,
    };
    let body = json!({ "input": input, "timeoutMs": asked.timeout_ms, "key": asked.key });
    let started = match with_base(client.post(url(&peer.address, &format!("/worker/v1/flows/{flow}/runs"))).json(&body)).timeout(Duration::from_secs(120)).send().await {
        Ok(r) => r,
        Err(e) => return offline(&net, &peer, &e),
    };
    if !asked.wait || !started.status().is_success() {
        return worker::passthrough(started);
    }
    let mut run: Value = match started.json().await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_GATEWAY, "upstream_error", format!("the worker's answer: {e}")),
    };
    let Some(run_id) = run["runId"].as_str().map(str::to_owned) else {
        return error(StatusCode::BAD_GATEWAY, "upstream_error", "the worker answered no run");
    };
    let deadline = Instant::now() + Duration::from_millis(asked.timeout_ms.unwrap_or(10 * 60 * 1000).min(60 * 60 * 1000)) + Duration::from_secs(10);
    while !terminal(run["status"].as_str().unwrap_or("")) {
        if Instant::now() >= deadline {
            return (StatusCode::ACCEPTED, Json(run)).into_response();
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
        match with_base(client.get(url(&peer.address, &format!("/worker/v1/runs/{run_id}")))).timeout(Duration::from_secs(15)).send().await {
            Ok(r) if r.status().is_success() => run = r.json().await.unwrap_or(run),
            Ok(r) => return worker::passthrough(r),
            Err(e) => return offline(&net, &peer, &e),
        }
    }
    Json(run).into_response()
}

/// `GET /api/network/nodes/:id/runs/:run`: a run of this computer's on the worker, as it is now.
async fn node_run(State(net): State<Network>, Path((node_id, run)): Path<(String, String)>, headers: axum::http::HeaderMap) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    if !flow_id(&run) {
        return error(StatusCode::BAD_REQUEST, "invalid_request", "not a run's id");
    }
    let mut call = client.get(url(&peer.address, &format!("/worker/v1/runs/{run}"))).timeout(Duration::from_secs(15));
    if let Some(b) = base_for(&headers, &node_id) {
        call = call.header("x-oaiy-base", b);
    }
    match call.send().await {
        Ok(r) => worker::passthrough(r),
        Err(e) => offline(&net, &peer, &e),
    }
}

/// `POST /api/network/nodes/:id/runs/:run/cancel`: a run of this computer's on the worker, stopped.
async fn node_cancel_run(State(net): State<Network>, Path((node_id, run)): Path<(String, String)>) -> Response {
    let (peer, client) = match worker_of(&net, &node_id) {
        Ok(w) => w,
        Err(r) => return r,
    };
    if !flow_id(&run) {
        return error(StatusCode::BAD_REQUEST, "invalid_request", "not a run's id");
    }
    match client.post(url(&peer.address, &format!("/worker/v1/runs/{run}/cancel"))).timeout(Duration::from_secs(15)).send().await {
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
            forget_lapsed(&net).await;
            tell_untold(&net).await;
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

/// Pairings that ran out here: a worker whose person approved, with no answer from this computer's person, is told to
/// forget this computer (it paired on its side only); one still asking is told to stop.
pub(crate) async fn forget_lapsed(net: &Network) {
    let lapsed: Vec<Outgoing> = {
        let mut outgoing = lock(&net.inner().outgoing);
        let (gone, kept): (Vec<Outgoing>, Vec<Outgoing>) = outgoing.drain(..).partition(|o| o.at.elapsed() >= PAIRING_TTL);
        *outgoing = kept;
        gone
    };
    let Ok(id) = net.identity() else { return };
    for p in lapsed {
        // (paired here, before or since: nothing to undo)
        if paired_already(net, &p) {
            continue;
        }
        if let Ok(client) = client_for(&id, &p.fingerprint) {
            let _ = client.post(url(&p.address, "/worker/v1/unpair")).timeout(Duration::from_secs(3)).send().await;
        }
    }
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
    // (this computer lending itself too: where it listens, so a worker that uses it as well keeps its address)
    let lends_on = net.listening().map(|a| a.port().to_string());
    for address in addresses {
        let mut ask = client.get(url(&address, "/worker/v1/capabilities")).timeout(HEARTBEAT_DEADLINE);
        if let Some(port) = &lends_on {
            ask = ask.header("x-oaiy-lends-on", port);
        }
        match ask.send().await {
            Ok(r) if r.status().is_success() => {
                let mut caps = r.json::<Value>().await.unwrap_or(Value::Null);
                if address != peer.address {
                    let _ = net.inner().peers().moved(&peer.node_id, &address);
                }
                let mut nodes = lock(&net.inner().nodes);
                let s = nodes.entry(peer.node_id.clone()).or_default();
                // (its engines not answering now, the computer itself answering: its models as last heard, not ready,
                // so a default on it says so when used rather than this computer's own running in its place)
                let engines_down = caps.get("discovery").is_none_or(Value::is_null);
                if engines_down {
                    if let Some(last) = s.capabilities.as_ref().and_then(|c| c.get("discovery")).filter(|d| !d.is_null()).cloned() {
                        caps["discovery"] = last;
                        caps["enginesDown"] = json!(true);
                    }
                }
                *s = NodeState { online: true, last_seen_ms: now_ms(), last_error: None, capabilities: Some(caps) };
                return;
            }
            Ok(r) if r.status() == reqwest::StatusCode::FORBIDDEN => last_error = Some("it no longer lends itself to this computer (unpaired there, or Lend this computer is off there)".to_string()),
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
    remote_models_of(net, net.settings().uses())
}

/// The paired workers' models as last heard ([`remote_services_of`]), while this computer uses others (`uses`).
fn remote_models_of(net: &Network, uses: bool) -> Vec<Value> {
    if !uses {
        return Vec::new();
    }
    let workers: Vec<Peer> = net.inner().peers().list().iter().filter(|p| p.role == Role::Worker).cloned().collect();
    let nodes = lock(&net.inner().nodes).clone();
    let mut out = Vec::new();
    for p in workers {
        let Some(state) = nodes.get(&p.node_id) else { continue };
        let Some(discovery) = state.capabilities.as_ref().and_then(|c| c.get("discovery")).filter(|d| !d.is_null()) else { continue };
        let resources = state.capabilities.as_ref().and_then(|c| c.get("resources")).cloned().unwrap_or(Value::Null);
        let engines_down = state.capabilities.as_ref().is_some_and(|c| c.get("enginesDown") == Some(&json!(true)));
        for e in crate::ai::engine_services::services_from_discovery(discovery) {
            let mut entry = remote_entry(e, &p, state.online && !engines_down);
            if !stays_below(&entry, &format!("/api/network/nodes/{}/gateway", p.node_id)) {
                continue;
            }
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

/// Whether each of a worker's entry's routes stays below that worker's gateway here: a worker naming `..` in its
/// discovery would send this desktop's clients (with their credentials) to another of this desktop's routes.
fn stays_below(e: &Value, gateway: &str) -> bool {
    let mut routes: Vec<&str> = ["endpoint", "filesUrl"].iter().filter_map(|f| e.get(*f).and_then(Value::as_str)).collect();
    if let Some(job) = e.get("job").and_then(Value::as_object) {
        routes.extend(job.values().filter_map(Value::as_str).filter(|v| v.starts_with('/')));
    }
    routes.into_iter().all(|r| match r.strip_prefix(gateway) {
        Some(rest) => crate::ai::engine_services::plain_path(rest.split('?').next().unwrap_or("")),
        None => !r.contains(".."),
    })
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
