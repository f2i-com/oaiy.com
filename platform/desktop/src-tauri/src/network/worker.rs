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

use super::identity::{commitment, fingerprint, is_nonce, nonce, pairing_code, Identity};
use super::store::{now_ms, Peer, Role};
use super::{lock, tls, Network, PROTOCOL};

/// Pairing requests waiting at once (from one address, and in all), and how long one waits.
const WAITING_PER_ADDRESS: usize = 2;
const MAX_WAITING: usize = 8;
const REQUEST_TTL: Duration = Duration::from_secs(5 * 60);
/// Connections served at once; of those, still in their TLS handshake from one address, and of computers not paired
/// here (asking `health` or to pair: one request each) in all and from one address. A computer on the LAN holding
/// connections open takes a share of them, and the paired primary the rest.
const MAX_CONNECTIONS: usize = 64;
const HANDSHAKES_PER_ADDRESS: usize = 16;
const MAX_STRANGERS: usize = 16;
const STRANGERS_PER_ADDRESS: usize = 4;
/// How long a connection has for its TLS handshake, and a request for its head (a paired primary's connection waits
/// for its next request for as long).
const HANDSHAKE: Duration = Duration::from_secs(10);
const HEAD: Duration = Duration::from_secs(15);
const IDLE: Duration = Duration::from_secs(120);
const CHAT_LIMIT: usize = 16 << 20;
const MEDIA_LIMIT: usize = 96 << 20;
/// A voice clip copied here, base64 in JSON (a clip is 20 MB at most).
const VOICE_LIMIT: usize = 32 << 20;

/// A primary asking this worker to pair, waiting for the person's approval.
#[derive(Clone, Debug)]
pub(crate) struct Incoming {
    pub id: String,
    pub node_id: String,
    pub name: String,
    pub fingerprint: String,
    pub address: String,
    /// This worker's random value for the code, committed to in its answer and told once the primary has told its own.
    pub nonce: String,
    /// The code both screens show, once both random values are known (the person is asked only then).
    pub code: Option<String>,
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

/// Connections counted by the address they come from; a count is given back as its [`Counted`] is dropped.
#[derive(Clone, Default)]
struct PerAddress(Arc<std::sync::Mutex<std::collections::HashMap<std::net::IpAddr, usize>>>);

struct Counted(PerAddress, std::net::IpAddr);

impl PerAddress {
    /// One more from `ip`, if it has fewer than `max`.
    fn take(&self, ip: std::net::IpAddr, max: usize) -> Option<Counted> {
        let mut map = lock(&self.0);
        let n = map.entry(ip).or_insert(0);
        if *n >= max {
            return None;
        }
        *n += 1;
        Some(Counted(self.clone(), ip))
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        let mut map = lock(&(self.0).0);
        if let Some(n) = map.get_mut(&self.1) {
            *n -= 1;
            if *n == 0 {
                map.remove(&self.1);
            }
        }
    }
}

/// Listen on `bind` (TLS, this computer's certificate), serving the worker's routes until the returned sender says
/// stop; the task returned ends once the listener is closed (its port free to bind again).
pub(crate) async fn listen(net: Network, id: &Identity, bind: SocketAddr) -> Result<(SocketAddr, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>), String> {
    let listener = tokio::net::TcpListener::bind(bind).await.map_err(|e| format!("the worker could not listen on {bind}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let acceptor = tokio_rustls::TlsAcceptor::from(tls::server_config(id)?);
    let app = routes(net.clone());
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    let slots = Arc::new(tokio::sync::Semaphore::new(MAX_CONNECTIONS));
    let strangers = Arc::new(tokio::sync::Semaphore::new(MAX_STRANGERS));
    let (handshaking, stranger_from) = (PerAddress::default(), PerAddress::default());
    let task = tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                a = listener.accept() => a,
                _ = stopped.changed() => break,
            };
            let (tcp, peer) = match accepted {
                Ok(a) => a,
                // (out of file descriptors, say: tried again in a moment, not at once in a loop)
                Err(_) => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };
            // (the LAN alone: a connection from anywhere else is closed before TLS)
            if !tls::local_address(peer.ip()) {
                continue;
            }
            let Ok(slot) = slots.clone().try_acquire_owned() else { continue };
            let Some(shaking) = handshaking.take(peer.ip(), HANDSHAKES_PER_ADDRESS) else { continue };
            let (acceptor, app, net, strangers, stranger_from, mut stopped) = (acceptor.clone(), app.clone(), net.clone(), strangers.clone(), stranger_from.clone(), stopped.clone());
            tokio::spawn(async move {
                let _slot = slot;
                let Ok(Ok(tls)) = tokio::time::timeout(HANDSHAKE, acceptor.accept(tcp)).await else { return };
                drop(shaking);
                let fp = tls.get_ref().1.peer_certificates().and_then(|c| c.first()).map(|c| fingerprint(c.as_ref()));
                let caller = Caller { fingerprint: fp, addr: peer };
                // (a computer not paired here: a request a connection, within the strangers' share)
                let paired = primary_of(&net, &caller).is_some();
                let _stranger = if paired {
                    None
                } else {
                    let Ok(permit) = strangers.try_acquire_owned() else { return };
                    let Some(counted) = stranger_from.take(peer.ip(), STRANGERS_PER_ADDRESS) else { return };
                    Some((permit, counted))
                };
                let svc = hyper_util::service::TowerToHyperService::new(app.layer(Extension(caller)));
                let mut builder = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new());
                if !paired {
                    builder = builder.http1_only();
                }
                builder.http1().timer(hyper_util::rt::TokioTimer::new()).header_read_timeout(if paired { IDLE } else { HEAD }).keep_alive(paired);
                builder.http2().timer(hyper_util::rt::TokioTimer::new()).keep_alive_interval(Some(IDLE)).keep_alive_timeout(HEAD);
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
    Ok((addr, stop, task))
}

fn routes(net: Network) -> Router {
    Router::new()
        .route("/worker/v1/health", get(health))
        .route("/worker/v1/pair", post(pair))
        .route("/worker/v1/pair/:id", get(pair_status).post(pair_reveal))
        .route("/worker/v1/unpair", post(unpair))
        .route("/worker/v1/capabilities", get(capabilities))
        .route("/worker/v1/resources", get(resources))
        .route("/worker/v1/pins", post(pins))
        .route("/worker/v1/flows", get(flows))
        .route("/worker/v1/flows/:id/runs", post(run_flow).layer(DefaultBodyLimit::max(MEDIA_LIMIT)))
        .route("/worker/v1/runs/:id", get(run_status))
        .route("/worker/v1/runs/:id/cancel", post(cancel_run))
        .route("/worker/v1/services/:service/*path", any(service).layer(DefaultBodyLimit::max(MEDIA_LIMIT)))
        .route("/worker/v1/voices/:name", axum::routing::put(voice_copy).layer(DefaultBodyLimit::max(VOICE_LIMIT)))
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

/// `POST /worker/v1/pair`: a primary asks to pair. Answered with this worker's commitment to a random value
/// ([`commitment`]); the primary then tells its own ([`pair_reveal`]), and the request waits for the person here, who
/// sees the same code as the primary's screen shows.
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
    let Ok(worker_nonce) = nonce() else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "no random value for the pairing");
    };
    let address = caller.addr.ip().to_string();
    let mut waiting = lock(&net.inner().incoming);
    waiting.retain(|r| r.at.elapsed() < REQUEST_TTL && r.fingerprint != fp);
    let pending = waiting.iter().filter(|r| r.status == Status::Pending);
    if pending.clone().count() >= MAX_WAITING || pending.filter(|r| r.address == address).count() >= WAITING_PER_ADDRESS {
        return refuse(StatusCode::TOO_MANY_REQUESTS, "too_many", "too many pairing requests are waiting");
    }
    let request = Incoming {
        id: uuid::Uuid::new_v4().simple().to_string(),
        node_id,
        name: if name.trim().is_empty() { "An OAIY computer".into() } else { name },
        nonce: worker_nonce,
        code: None,
        fingerprint: fp,
        address,
        status: Status::Pending,
        at: Instant::now(),
    };
    let answer = json!({
        "requestId": request.id,
        "nodeId": id.node_id,
        "name": net.display_name(),
        "protocol": PROTOCOL,
        "commitment": commitment(&request.fingerprint, &id.fingerprint, &request.nonce),
    });
    waiting.push(request);
    (StatusCode::ACCEPTED, Json(answer)).into_response()
}

#[derive(Deserialize)]
struct Reveal {
    nonce: String,
}

/// `POST /worker/v1/pair/:id` `{nonce}`: the primary that asked tells its random value, once; this worker's, which it
/// committed to, is told back, and the code both screens show is made of both. Only then is the person here asked.
async fn pair_reveal(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>, Json(req): Json<Reveal>) -> Response {
    if !is_nonce(&req.nonce) {
        return refuse(StatusCode::BAD_REQUEST, "invalid_request", "a random value of 64 hex digits is needed");
    }
    let Ok(me) = net.identity() else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "this worker has no identity");
    };
    let mut waiting = lock(&net.inner().incoming);
    let Some(r) = waiting.iter_mut().find(|r| r.id == id && Some(&r.fingerprint) == caller.fingerprint.as_ref() && r.at.elapsed() < REQUEST_TTL) else {
        return refuse(StatusCode::NOT_FOUND, "not_found", "no such pairing request (or it has expired)");
    };
    // (told once: a second value would let the caller try again for a code of its choosing)
    if r.code.is_some() {
        return refuse(StatusCode::CONFLICT, "already_told", "this pairing request has its code already");
    }
    let code = pairing_code(&r.fingerprint, &me.fingerprint, &req.nonce.to_ascii_lowercase(), &r.nonce);
    log::info!("network: {} ({}) asks to pair; code {code}", r.name, r.address);
    r.code = Some(code);
    Json(json!({ "nonce": r.nonce })).into_response()
}

/// `GET /worker/v1/pair/:id`: the request's answer, to the certificate that asked alone.
async fn pair_status(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> Response {
    let waiting = lock(&net.inner().incoming);
    match waiting.iter().find(|r| r.id == id && Some(&r.fingerprint) == caller.fingerprint.as_ref() && r.at.elapsed() < REQUEST_TTL) {
        Some(r) => Json(json!({ "status": r.status })).into_response(),
        None => refuse(StatusCode::NOT_FOUND, "not_found", "no such pairing request (or it has expired)"),
    }
}

/// `POST /worker/v1/unpair`: the primary unpaired this worker (its certificate no longer lets it in here either), or
/// gave up a pairing it asked for (its request is withdrawn, so the person here is no longer asked).
async fn unpair(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    let withdrawn = match caller.fingerprint.as_deref() {
        Some(fp) => {
            let mut waiting = lock(&net.inner().incoming);
            let before = waiting.len();
            waiting.retain(|r| r.fingerprint != fp);
            before != waiting.len()
        }
        None => false,
    };
    let Some(p) = primary_of(&net, &caller) else {
        return if withdrawn {
            StatusCode::NO_CONTENT.into_response()
        } else {
            refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller")
        };
    };
    let _ = net.inner().peers().remove(&p.node_id);
    log::info!("network: {} unpaired this worker", p.name);
    StatusCode::NO_CONTENT.into_response()
}

/// The approval the person gives here (`/api/network/requests/:id/approve`): the primary is paired.
pub(crate) fn decide(net: &Network, id: &str, approve: bool) -> Result<Option<Peer>, String> {
    let mut waiting = lock(&net.inner().incoming);
    let Some(r) = waiting.iter_mut().find(|r| r.id == id && r.status == Status::Pending && r.code.is_some() && r.at.elapsed() < REQUEST_TTL) else {
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
        // (a request whose primary has not told its random value has no code yet: nothing to ask the person)
        .filter(|r| r.status == Status::Pending && r.code.is_some())
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
    Json(json!({
        "node": node,
        "discovery": discovery,
        "enginesError": engines,
        "resources": local_resources(&net).await,
        "flows": shared_flows(&net)["flows"],
        "services": shared_services(&net),
    }))
    .into_response()
}

// ---------------------------------------------------------------------------------------------------------------
// Flows: the ones this worker shares ("Let apps run it" in its Flows editor), run here for its paired primary
// ---------------------------------------------------------------------------------------------------------------

/// The flows this worker shares with apps, its primary among them: `{flows: [{flowId, name, description, inputs}]}`.
pub(crate) fn shared_flows(net: &Network) -> Value {
    net.inner().bridge().map(|b| crate::bridge::flow_connector::list(&b.flows)).unwrap_or_else(|| json!({ "flows": [] }))
}

/// The caller a primary's runs are kept under here, so it reads and cancels its own alone.
fn run_caller(primary: &Peer) -> String {
    format!("oaiy-network:{}", primary.node_id)
}

/// `GET /worker/v1/flows`: the flows this worker shares.
async fn flows(State(net): State<Network>, Extension(caller): Extension<Caller>) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    Json(shared_flows(&net)).into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FlowRun {
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    /// The primary's own key for the run: asked twice, it runs once.
    #[serde(default)]
    key: Option<String>,
}

/// The longest a primary's run may take here.
const RUN_LIMIT_MS: u64 = 60 * 60 * 1000;

/// `POST /worker/v1/flows/:id/runs` `{input?, timeoutMs?, key?}`: one of the flows this worker shares, run here (its
/// runner, its engines) for its primary. Answered at once with the run (`queued`); [`run_status`] follows it.
async fn run_flow(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>, req: Request) -> Response {
    let Some(primary) = primary_of(&net, &caller) else {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    };
    let base = rebase_to(&req);
    let Some(bridge) = net.inner().bridge() else {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "flows_unavailable", "this computer runs no flows here");
    };
    let shared = bridge.flows.get(&id).and_then(|t| serde_json::from_str::<Value>(&t).ok()).is_some_and(|d| crate::bridge::flow_connector::shared(&d));
    if !shared {
        return refuse(StatusCode::NOT_FOUND, "flow_not_shared", format!("{} shares no flow {id}: on it, open the flow and switch on Let apps run it", net.display_name()));
    }
    let body = match axum::body::to_bytes(req.into_body(), MEDIA_LIMIT).await {
        Ok(b) => b,
        Err(e) => return refuse(StatusCode::PAYLOAD_TOO_LARGE, "too_large", e.to_string()),
    };
    let asked: FlowRun = match serde_json::from_slice(&body) {
        Ok(a) => a,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, "invalid_request", format!("send {{input?, timeoutMs?, key?}}: {e}")),
    };
    // (its pictures, given inline, stored with this computer's engines first, as the desktop's own runs do)
    let input = match asked.input {
        Some(i) => tokio::task::spawn_blocking(move || crate::bridge::pictures::referenced(i)).await.ok(),
        None => None,
    };
    let key = asked.key.filter(|k| !k.trim().is_empty() && k.len() <= 200).unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let request = crate::bridge::ledger::RunRequest {
        caller_product: run_caller(&primary),
        flow_id: Some(id.clone()),
        inline_graph: false,
        input,
        timeout_ms: Some(asked.timeout_ms.unwrap_or(10 * 60 * 1000).clamp(1000, RUN_LIMIT_MS)),
        mode: "async".into(),
        correlation_id: format!("network:{}", uuid::Uuid::new_v4().simple()),
        idempotency_key: format!("network:{}:{key}", primary.node_id),
        lineage: Default::default(),
        trigger_event: None,
    };
    let outcome = { lock(&bridge.ledger).reserve(&request) };
    match outcome {
        crate::bridge::ledger::ReserveOutcome::Reserved(rec) => {
            log::info!("network: {} runs {id} here ({})", primary.name, rec.run_id);
            run_answer(StatusCode::CREATED, &rec, base.as_deref())
        }
        crate::bridge::ledger::ReserveOutcome::Duplicate(rec) => run_answer(StatusCode::OK, &rec, base.as_deref()),
        crate::bridge::ledger::ReserveOutcome::Refused { reason } => refuse(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", format!("refused by a loop guard: {reason}")),
    }
}

/// `GET /worker/v1/runs/:id`: a run this primary asked for, as it is now (its output's pictures named at the
/// primary's route for this worker).
async fn run_status(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>, req: Request) -> Response {
    let Some(primary) = primary_of(&net, &caller) else {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    };
    let base = rebase_to(&req);
    match own_run(&net, &primary, &id) {
        Some(rec) => run_answer(StatusCode::OK, &rec, base.as_deref()),
        None => refuse(StatusCode::NOT_FOUND, "not_found", "no such run of this computer's"),
    }
}

/// `POST /worker/v1/runs/:id/cancel`: a run this primary asked for, stopped.
async fn cancel_run(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> Response {
    let Some(primary) = primary_of(&net, &caller) else {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    };
    let (Some(bridge), Some(_)) = (net.inner().bridge(), own_run(&net, &primary, &id)) else {
        return refuse(StatusCode::NOT_FOUND, "not_found", "no such run of this computer's");
    };
    let outcome = { lock(&bridge.ledger).request_cancel(&id) };
    match outcome {
        Ok(status) => Json(json!({ "runId": id, "status": status, "cancelRequested": true })).into_response(),
        Err(e) => refuse(StatusCode::CONFLICT, "not_cancellable", e),
    }
}

fn own_run(net: &Network, primary: &Peer, id: &str) -> Option<crate::bridge::ledger::RunRecord> {
    let bridge = net.inner().bridge()?;
    let rec = { lock(&bridge.ledger).get(id) }?;
    (rec.caller_product == run_caller(primary)).then_some(rec)
}

/// The primary's own address for this worker's gateway (`x-oaiy-base`), where a run's pictures are named.
fn rebase_to(req: &Request) -> Option<String> {
    req.headers().get("x-oaiy-base").and_then(|v| v.to_str().ok()).filter(|b| b.starts_with("http://") && b.len() < 512).map(str::to_owned)
}

/// A run's record, its pictures (this desktop's own engine route, `http://127.0.0.1:<port>/api/ai/engine/gateway/…`)
/// named at `base`, the primary's route for this worker's gateway.
fn run_answer(status: StatusCode, rec: &crate::bridge::ledger::RunRecord, base: Option<&str>) -> Response {
    let mut text = serde_json::to_string(rec).unwrap_or_else(|_| "{}".into());
    if let Some(base) = base {
        let port = crate::api_port();
        for host in ["127.0.0.1", "localhost"] {
            text = text.replace(&format!("http://{host}:{port}{}", crate::ai::engine_services::GATEWAY_PREFIX), base.trim_end_matches('/'));
        }
    }
    (status, [("content-type", "application/json")], text).into_response()
}

// ---------------------------------------------------------------------------------------------------------------
// Services: the ones a worker lends its primary (OAIY Voice), their speech routes alone
// ---------------------------------------------------------------------------------------------------------------

/// The services a worker lends its paired primary, each with the routes of its a primary reaches: OAIY Voice's speech
/// (its models and voices, speech-to-text and text-to-speech), so a primary connected to the phone can have this
/// computer hear and speak its calls. Nothing else of a service (its settings, files or other routes), and no other
/// service (one a person added could be anything).
pub(crate) const SHARED_SERVICES: [(&str, &[(&str, &str)]); 1] = [(
    crate::voice::engines::STT_SERVICE,
    &[("GET", "/v1/models"), ("GET", "/health"), ("GET", "/v1/audio/voices"), ("POST", "/v1/audio/transcriptions"), ("POST", "/v1/audio/speech")],
)];

/// The services this worker lends, as the primary shows them: `[{id, name, installed, status}]` (one not added here
/// is left out).
pub(crate) fn shared_services(net: &Network) -> Vec<Value> {
    let Ok(registry) = net.inner().ai.registry.lock() else { return Vec::new() };
    let snapshot = registry.snapshot();
    SHARED_SERVICES
        .iter()
        .filter_map(|(id, _)| snapshot.services.iter().find(|s| s.id == *id))
        .map(|s| json!({ "id": s.id, "name": s.name, "installed": s.installed, "status": s.status }))
        .collect()
}

/// `/worker/v1/services/:service/<path>`: one of a shared service's routes, on this computer's own (started where it
/// is not running: a call is coming), its answer streamed back as it comes (a line of speech, word by word).
async fn service(State(net): State<Network>, Extension(caller): Extension<Caller>, Path((service, path)): Path<(String, String)>, req: Request) -> Response {
    if primary_of(&net, &caller).is_none() {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    }
    let path = format!("/{path}");
    let method = req.method().as_str().to_ascii_uppercase();
    let Some((id, _)) = SHARED_SERVICES.iter().find(|(id, routes)| *id == service && routes.iter().any(|(m, p)| *m == method && *p == path)) else {
        return refuse(StatusCode::NOT_FOUND, "not_shared", format!("{method} {path} is not one of the routes {service} lends"));
    };
    let id = *id;
    let registry = net.inner().ai.registry.clone();
    let port = tokio::task::spawn_blocking(move || -> Result<u16, String> {
        let mut r = registry.lock().map_err(|_| "the services are unavailable".to_string())?;
        r.start(id).map_err(|e| format!("{id} did not start: {e}"))?;
        r.service_port(id).ok_or_else(|| format!("{id} has no port"))
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    let port = match port {
        Ok(p) => p,
        Err(e) => return refuse(StatusCode::CONFLICT, "service_unavailable", e),
    };
    let content_type = req.headers().get("content-type").and_then(|v| v.to_str().ok()).map(str::to_owned);
    let body = match axum::body::to_bytes(req.into_body(), MEDIA_LIMIT).await {
        Ok(b) => b,
        Err(e) => return refuse(StatusCode::PAYLOAD_TOO_LARGE, "too_large", e.to_string()),
    };
    let Ok(client) = reqwest::Client::builder().connect_timeout(Duration::from_secs(3)).build() else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "no HTTP client");
    };
    let Ok(m) = reqwest::Method::from_bytes(method.as_bytes()) else {
        return refuse(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", method);
    };
    let mut call = client.request(m, format!("http://127.0.0.1:{port}{path}"));
    if let Some(ct) = content_type {
        call = call.header("content-type", ct);
    }
    if !body.is_empty() {
        call = call.body(body);
    }
    match call.send().await {
        Ok(r) => passthrough(r),
        // (just started, loading its models: asked again until it answers, as this computer's own calls do)
        Err(e) if e.is_connect() => refuse(StatusCode::SERVICE_UNAVAILABLE, "service_starting", format!("{id} is starting on {}", net.display_name())),
        Err(e) => refuse(StatusCode::BAD_GATEWAY, "service_unreachable", format!("{id} did not answer: {e}")),
    }
}

#[derive(Deserialize)]
struct VoiceCopy {
    /// The clip, base64 (a WAV, MP3, FLAC or OGG of the voice).
    clip: String,
    #[serde(default)]
    extension: Option<String>,
    /// What it says, where its primary has that written down.
    #[serde(default)]
    words: Option<String>,
}

/// The name a primary's voice is kept under here: its own beside this computer's, never in place of one.
pub(crate) fn copied_voice_name(primary_node: &str, voice: &str) -> String {
    let short: String = primary_node.chars().take(6).collect();
    format!("from-{short}-{voice}")
}

/// `PUT /worker/v1/voices/:name` `{clip, extension?, words?}`: the voice its primary answers calls in, copied here
/// (under a name of its own, `from-<primary>-<name>`) so this computer's speech server speaks in it.
async fn voice_copy(State(net): State<Network>, Extension(caller): Extension<Caller>, Path(name): Path<String>, Json(copy): Json<VoiceCopy>) -> Response {
    let Some(primary) = primary_of(&net, &caller) else {
        return refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller");
    };
    let Ok(bytes) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, copy.clip.trim()) else {
        return refuse(StatusCode::BAD_REQUEST, "invalid_request", "the clip is not base64");
    };
    if bytes.len() > crate::voice::voices::MAX_CLIP_BYTES {
        return refuse(StatusCode::PAYLOAD_TOO_LARGE, "too_large", "a voice clip is 20 MB at most");
    }
    let stored = copied_voice_name(&primary.node_id, &name);
    let extension = copy.extension.unwrap_or_else(|| "wav".into());
    match tokio::task::spawn_blocking(move || crate::voice::voices::add(&stored, &extension, &bytes, copy.words.as_deref())).await {
        Ok(Ok(clip)) => Json(json!({ "voice": clip.name })).into_response(),
        Ok(Err(e)) => refuse(StatusCode::BAD_REQUEST, "invalid_request", e),
        Err(e) => refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
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
    // (the engine answers with its chosen model, as the desktop's own engine provider does: one asked for by name that
    // is not that one any more is refused, rather than answered by another model)
    let mut request: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, "invalid_request", format!("the body is not JSON: {e}")),
    };
    let asked = request.get("model").and_then(Value::as_str).map(str::trim).filter(|m| !m.is_empty()).map(str::to_owned);
    if let (Some(asked), Some(chosen)) = (asked, discovery.pointer("/defaults/llm").and_then(Value::as_str)) {
        if asked != chosen && !asked.ends_with(&format!("/{chosen}")) {
            return refuse(
                StatusCode::CONFLICT,
                "model_unavailable",
                format!("{} runs {chosen} now, not {asked}: choose its language model again", net.display_name()),
            );
        }
    }
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
    // (the primary's own address for this worker's gateway, where its answers name their files: see `rebase`)
    let base = req.headers().get("x-oaiy-base").and_then(|v| v.to_str().ok()).filter(|b| b.starts_with("http://") && b.len() < 512).map(str::to_owned);
    match crate::ai::engine_services::local_engines(&net.inner().ai).await {
        Ok((g, d)) => {
            let answer = crate::ai::engine_services::forward_path(&g, &d, format!("/{path}"), req).await;
            match base {
                Some(base) => rebase(answer, &g, &base).await,
                None => answer,
            }
        }
        Err(e) => refuse(StatusCode::SERVICE_UNAVAILABLE, "engine_unavailable", e),
    }
}

/// An engines' JSON answer with their own address (this computer's loopback, which on the primary is the primary's
/// own) put as `base`, the primary's route for this worker: a picture's `url` or `file` is fetched through the primary.
async fn rebase(answer: Response, gateway: &str, base: &str) -> Response {
    let json = answer.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("json"));
    if !json {
        return answer;
    }
    let (mut parts, body) = answer.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MEDIA_LIMIT).await else {
        return refuse(StatusCode::BAD_GATEWAY, "upstream_error", "the answer could not be passed on");
    };
    let text = String::from_utf8_lossy(&bytes).replace(gateway.trim_end_matches('/'), base.trim_end_matches('/'));
    parts.headers.remove("content-length");
    Response::from_parts(parts, Body::from(text))
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
